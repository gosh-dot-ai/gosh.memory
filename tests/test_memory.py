"""Tests for src/memory.py — MemoryServer."""

import asyncio
import json
import time

import numpy as np
import pytest

from src.memory import MemoryServer, _augment_commonality_facts
from src.storage import JSONNPZStorage

DIM = 3072


def _rand_embs(n, dim=DIM):
    return np.random.randn(n, dim).astype(np.float32)


def _rand_qemb(dim=DIM):
    return np.random.randn(dim).astype(np.float32)


def _fake_extract_result(n_facts=3, **tag_overrides):
    """Return (conv_id, session_num, session_date, facts, tlinks)."""
    facts = []
    for i in range(n_facts):
        f = {
            "id": f"f{i}",
            "fact": f"Test fact number {i}",
            "kind": "event",
            "entities": ["Alice"],
            "tags": ["test"],
            "session": 1,
            "scope": "swarm-shared",
            "agent_id": "default",
            "swarm_id": "default",
        }
        f.update(tag_overrides)
        facts.append(f)
    tlinks = [{"before": "f0", "after": "f1", "signal": "then"}]
    return ("test_conv", 1, "2024-06-01", facts, tlinks)


def test_legacy_conversation_write_log_entry_uses_store_not_document_ingest(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "legacy_conv_write_log")

    result = asyncio.run(
        ms._extract_write_log_entry(
            {
                "message_id": "raw:part-001",
                "session_id": "part-001",
                "content": "User: hello\nAssistant: hi",
                "content_family": "conversation",
                "timestamp_ms": 1712000000000,
                "agent_id": "agent-a",
                "swarm_id": "swarm-a",
                "scope": "swarm-shared",
                "owner_id": "agent:agent-a",
                "read": ["swarm:swarm-a"],
                "write": ["swarm:swarm-a"],
                "metadata": {
                    "logical_source_id": "q45_case",
                    "part_source_id": "q45_case_p001",
                    "part_idx": 1,
                },
            }
        )
    )

    assert result["facts_extracted"] >= 0
    assert not ms._raw_docs
    assert "q45_case" in ms._source_records
    assert ms._source_records["q45_case"]["family"] == "conversation"
    assert all(not source_id.startswith("part-001") for source_id in ms._source_records)
    doc_ids = [str(doc.get("doc_id") or "") for doc in ms._episode_corpus.get("documents", [])]
    assert doc_ids == ["conversation:q45_case"]


def test_memory_server_uses_legacy_sidecar_loaders(tmp_path, monkeypatch):
    storage = JSONNPZStorage(str(tmp_path), "legacy_loader")
    storage.save_facts({"granular": [], "cons": [], "cross": [], "tlinks": [], "raw_sessions": [], "raw_docs": {}, "n_sessions": 0})
    (tmp_path / "legacy_loader_corpus.json").write_text('{"documents": []}')
    (tmp_path / "legacy_loader_temporal.json").write_text('{"events": {}}')

    monkeypatch.setattr("src.memory.load_episode_corpus", lambda path, strict=False: {"documents": [{"doc_id": "sentinel", "episodes": []}]})
    monkeypatch.setattr("src.memory.load_temporal_index", lambda path: {"timelines": {}, "events": {"sentinel": {}}, "anchors": {}, "calendar_sorted_event_ids": []})

    ms = MemoryServer(str(tmp_path), "legacy_loader")
    assert ms._episode_corpus == {"documents": [{"doc_id": "sentinel", "episodes": []}]}
    assert ms._temporal_index["events"] == {"sentinel": {}}


@pytest.mark.asyncio
async def test_write_does_not_wait_for_file_lock(tmp_path, monkeypatch):
    ms = MemoryServer(str(tmp_path), "write_no_file_lock")

    def _fake_append_write_log(**kwargs):
        return {"message_id": kwargs["message_id"], "extraction_state": "pending", "inserted": True}

    monkeypatch.setattr(ms._storage, "append_write_log", _fake_append_write_log)

    await ms._file_lock.acquire()
    try:
        result = await asyncio.wait_for(
            ms.write(
                content="hello",
                content_family="chat",
                session_id="s1",
                message_id="m1",
                timestamp_ms=1712000000000,
            ),
            timeout=0.1,
        )
    finally:
        ms._file_lock.release()

    assert result["message_id"] == "m1"
    assert result["inserted"] is True

@pytest.mark.asyncio
async def test_write_remains_available_while_worker_extracts(tmp_path, monkeypatch):
    ms = MemoryServer(str(tmp_path), "write_during_extract")
    await ms.write(
        content="first pending write",
        content_family="chat",
        session_id="s1",
        message_id="m1",
        timestamp_ms=1712000000000,
    )

    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_extract(entry):
        started.set()
        await release.wait()
        return {"message_id": entry["message_id"]}

    monkeypatch.setattr(ms, "_extract_write_log_entry", _slow_extract)

    worker_task = asyncio.create_task(ms.process_write_log_once(batch_size=1))
    await asyncio.wait_for(started.wait(), timeout=1.0)

    second = await asyncio.wait_for(
        ms.write(
            content="second pending write",
            content_family="chat",
            session_id="s2",
            message_id="m2",
            timestamp_ms=1712000001000,
        ),
        timeout=0.5,
    )
    status = ms.write_status("m2")

    release.set()
    processed = await asyncio.wait_for(worker_task, timeout=1.0)

    assert processed == 1
    assert second["inserted"] is True
    assert status is not None
    assert status["extraction_state"] == "pending"



@pytest.mark.asyncio
async def test_write_log_worker_does_not_complete_on_stale_pending_extraction_raw_session(tmp_path, monkeypatch):
    ms = MemoryServer(str(tmp_path), "write_retry_pending_extraction")
    await ms.write(
        content="first pending write",
        content_family="chat",
        session_id="s1",
        message_id="m1",
        timestamp_ms=1712000000000,
    )
    async with ms._file_lock:
        ms._raw_sessions.append({
            "raw_session_id": "rs-stale",
            "message_id": "m1",
            "status": "pending_extraction",
        })

    async def _boom(_entry):
        raise RuntimeError("boom")

    monkeypatch.setattr(ms, "_extract_write_log_entry", _boom)

    processed = await ms.process_write_log_once(batch_size=1)
    status = ms.write_status("m1")

    assert processed == 0
    assert status is not None
    assert status["extraction_state"] == "pending"
    assert status["extraction_attempts"] == 1


@pytest.mark.asyncio
async def test_write_log_worker_skips_sync_store_entry_while_store_is_active(tmp_path, monkeypatch):
    ms = MemoryServer(str(tmp_path), "write_store_worker_race")

    release = asyncio.Event()
    first_extract_started = asyncio.Event()
    extract_calls: list[int] = []

    async def _blocking_extract_session(**kwargs):
        extract_calls.append(int(kwargs["session_num"]))
        if len(extract_calls) == 1:
            first_extract_started.set()
            await release.wait()
        return _fake_extract_result(1, session=int(kwargs["session_num"]))

    monkeypatch.setattr("src.memory.extract_session", _blocking_extract_session)

    store_task = asyncio.create_task(
        ms.store(
            content="User: My favorite database is PostgreSQL because I trust MVCC. Assistant: Noted.",
            session_num=1,
            session_date="2026-03-31",
            speakers="User and Assistant",
        )
    )

    await asyncio.wait_for(first_extract_started.wait(), timeout=1.0)
    assert len(ms._raw_sessions) == 1
    message_id = str(ms._raw_sessions[0]["message_id"])

    processed = await asyncio.wait_for(ms.process_write_log_once(batch_size=1), timeout=1.0)
    status = ms.write_status(message_id)

    assert processed == 0
    assert len([rs for rs in ms._raw_sessions if rs.get("message_id") == message_id]) == 1
    assert status is not None
    assert status["extraction_state"] == "pending"
    assert extract_calls == [1]

    release.set()
    result = await asyncio.wait_for(store_task, timeout=1.0)

    assert result["facts_extracted"] == 1
    assert len([rs for rs in ms._raw_sessions if rs.get("message_id") == message_id]) == 1
    assert ms.write_status(message_id)["extraction_state"] == "complete"


@pytest.mark.asyncio
async def test_write_and_raw_recall_meet_latency_targets_while_worker_busy(tmp_path, monkeypatch):
    ms = MemoryServer(str(tmp_path), "write_latency")
    await ms.write(
        content="first pending write",
        content_family="chat",
        session_id="s1",
        message_id="m1",
        timestamp_ms=1712000000000,
    )

    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_extract(entry):
        started.set()
        await release.wait()
        return {"message_id": entry["message_id"]}

    monkeypatch.setattr(ms, "_extract_write_log_entry", _slow_extract)

    worker_task = asyncio.create_task(ms.process_write_log_once(batch_size=1))
    await asyncio.wait_for(started.wait(), timeout=1.0)

    t0 = time.perf_counter()
    receipt = await ms.write(
        content="latency kiwi note",
        content_family="chat",
        session_id="s2",
        message_id="m2",
        timestamp_ms=1712000001000,
    )
    write_elapsed = time.perf_counter() - t0

    t1 = time.perf_counter()
    recall = await ms.recall(query="latency kiwi", caller_id="system")
    recall_elapsed = time.perf_counter() - t1

    release.set()
    await asyncio.wait_for(worker_task, timeout=1.0)

    assert receipt["inserted"] is True
    assert write_elapsed < 0.05
    assert recall_elapsed < 0.2
    assert recall["raw_recall_count"] >= 1
    assert "latency kiwi note" in recall["context"].lower()
# ── Patches ──

def _patch_extraction(monkeypatch, n_facts=3, **tag_overrides):
    """Patch extract_session and source aggregation."""

    async def mock_extract_session(**kwargs):
        sn = kwargs.get("session_num", 1)
        return _fake_extract_result(n_facts, session=sn, **tag_overrides)

    async def mock_consolidate_session(**kwargs):
        return ("test_conv", 1, "2024-06-01", [
            {"id": "c0", "fact": "Consolidated fact 0", "kind": "summary",
             "entities": ["Alice"], "tags": ["test"]},
        ])

    async def mock_cross_session_entity(**kwargs):
        return ("test_conv", "alice", [
            {"id": "x0", "fact": "Cross-session fact about Alice", "kind": "profile",
             "entities": ["Alice"], "tags": ["test"]},
        ])

    async def mock_extract_source_aggregation_facts(self, **kwargs):
        source_facts = kwargs.get("source_facts", [])
        source_id = kwargs.get("source_id", "source")
        if not source_facts:
            return []
        return [{
            "id": "xf0",
            "fact": f"Source aggregate fact for {source_id}",
            "kind": "fact",
            "entities": ["Alice"],
            "tags": ["substrate"],
            "source_ids": [f["id"] for f in source_facts],
            "metadata": {"source_aggregation": True},
        }]

    monkeypatch.setattr("src.memory.extract_session", mock_extract_session)
    monkeypatch.setattr("src.memory.consolidate_session", mock_consolidate_session)
    monkeypatch.setattr("src.memory.cross_session_entity", mock_cross_session_entity)
    monkeypatch.setattr(MemoryServer, "_extract_source_aggregation_facts", mock_extract_source_aggregation_facts)


def _patch_embeddings(monkeypatch):
    """Patch embed_texts and embed_query to return random arrays.

    embed_texts and embed_query are async, so mocks must return coroutines.
    """

    async def mock_embed_texts(texts, **kwargs):
        return _rand_embs(len(texts))

    async def mock_embed_query(text, **kwargs):
        return _rand_qemb()

    monkeypatch.setattr("src.memory.embed_texts", mock_embed_texts)
    monkeypatch.setattr("src.memory.embed_query", mock_embed_query)


def _patch_resolve_supersession(monkeypatch):
    """Patch resolve_supersession to be a no-op."""
    monkeypatch.setattr("src.memory.resolve_supersession", lambda facts, lookup: None)


def _patch_all(monkeypatch, n_facts=3, **tag_overrides):
    _patch_extraction(monkeypatch, n_facts, **tag_overrides)
    _patch_embeddings(monkeypatch)
    _patch_resolve_supersession(monkeypatch)


# ── Tests ──

def test_store_creates_cache_file(tmp_path, monkeypatch):
    """store() persists facts through the configured storage backend."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv1")

    asyncio.run(ms.store("Hello world", session_num=1, session_date="2024-06-01"))

    assert ms._storage.exists
    data = ms._storage.load_facts()
    assert len(data["granular"]) == 3
    assert data["n_sessions"] == 1
    assert data["n_sessions_with_facts"] == 1


def test_store_attaches_support_spans_when_facts_as_selectors_enabled(tmp_path, monkeypatch):
    async def mock_extract_session(**kwargs):
        return (
            "conv_selectors",
            1,
            "2024-06-01",
            [
                {
                    "id": "f_01",
                    "fact": "I drive a Prius hybrid every day.",
                    "kind": "preference",
                    "entities": ["Prius"],
                    "tags": ["vehicle"],
                    "session": 1,
                }
            ],
            [],
        )

    _patch_embeddings(monkeypatch)
    _patch_resolve_supersession(monkeypatch)
    monkeypatch.setattr("src.memory.extract_session", mock_extract_session)
    monkeypatch.setenv("GOSH_FACT_SELECTORS", "1")

    ms = MemoryServer(str(tmp_path), "conv_selectors")
    asyncio.run(
        ms.store(
            "User: I drive a Prius hybrid every day.\nAssistant: Nice car.",
            session_num=1,
            session_date="2024-06-01",
        )
    )

    fact = ms._all_granular[0]
    assert fact["fact_class"] == "extractive"
    assert fact["support_spans"]
    span = fact["support_spans"][0]
    assert span["source_field"] == "raw_text"
    assert span["episode_id"] == "conv_selectors_e0001"
    assert span["end"] > span["start"]


def test_calendar_seeking_resolution_renders_selector_source_for_temporal_queries(tmp_path, monkeypatch):
    monkeypatch.setenv("GOSH_FACT_SELECTORS", "1")
    ms = MemoryServer(str(tmp_path), "conv_temporal_seek")
    raw_text = "Evan: I drove my Prius hybrid car to work on March 3, 2024."
    episode_id = "conv_temporal_seek_e0001"
    ms._episode_corpus = {
        "documents": [{
            "doc_id": "conversation:conv_temporal_seek",
            "episodes": [{
                "episode_id": episode_id,
                "source_type": "conversation",
                "source_id": "conv_temporal_seek",
                "source_date": "2024-03-03",
                "topic_key": "session",
                "state_label": "session",
                "currentness": "unknown",
                "raw_text": raw_text,
                "provenance": {"raw_span": [0, len(raw_text)]},
            }],
        }],
    }
    fact = {
        "id": "f_temporal_seek",
        "fact": "Evan drove his Prius hybrid car to work.",
        "metadata": {"episode_id": episode_id},
        "support_spans": [{
            "episode_id": episode_id,
            "source_field": "raw_text",
            "start": 0,
            "end": len(raw_text),
            "role": "primary",
        }],
    }
    ms._fact_lookup = {fact["id"]: fact}

    async def mock_resolve_calendar_seeking(*, query, candidate_facts):
        return {
            "fact": fact,
            "event": {
                "event_id": "evt_seek_1",
                "time_start": "2024-03-03",
                "time_end": "2024-03-03",
                "time_granularity": "day",
            },
            "answer": "2024",
            "trace": {"query": query},
        }

    monkeypatch.setattr(ms, "_resolve_calendar_seeking", mock_resolve_calendar_seeking)
    recall_result = {
        "context": "RETRIEVED FACTS:\n[1] (S1) Evan drove his Prius hybrid car to work.",
        "_context_packet": {"tier1": [], "tier2": [], "tier3": [], "tier4": []},
    }

    result = asyncio.run(
        ms._attach_calendar_seeking_resolution(
            query="Which year did Evan drive his car to work?",
            recall_result=recall_result,
            candidate_facts=[fact],
        )
    )

    assert "TEMPORAL EVIDENCE:" in result["context"]
    assert f"Source (raw_text, Episode {episode_id}" in result["context"]
    assert "Prius hybrid car" in result["context"]
    assert result["temporal_resolution"]["mode"] == "calendar-seeking"
    assert any(
        "Source (raw_text, Episode" in str(item.get("text", ""))
        for item in result["_context_packet"]["tier1"]
        if isinstance(item, dict)
    )


def test_calendar_answer_resolution_falls_back_to_event_source_span_for_temporal_queries(tmp_path, monkeypatch):
    monkeypatch.setenv("GOSH_FACT_SELECTORS", "1")
    ms = MemoryServer(str(tmp_path), "conv_temporal_answer")
    raw_text = "Audrey adopted Pepper, Precious, and Panda three years ago."
    episode_id = "conv_temporal_answer_e0001"
    ms._episode_corpus = {
        "documents": [{
            "doc_id": "conversation:conv_temporal_answer",
            "episodes": [{
                "episode_id": episode_id,
                "source_type": "conversation",
                "source_id": "conv_temporal_answer",
                "source_date": "2023-01-21",
                "topic_key": "session",
                "state_label": "session",
                "currentness": "unknown",
                "raw_text": raw_text,
                "provenance": {"raw_span": [0, len(raw_text)]},
            }],
        }],
    }
    fact = {
        "id": "f_temporal_answer",
        "fact": "Audrey adopted Pepper, Precious, and Panda three years ago.",
        "metadata": {"episode_id": episode_id},
    }
    ms._fact_lookup = {fact["id"]: fact}
    recall_result = {
        "context": "RETRIEVED FACTS:\n[1] (S1) Audrey adopted Pepper, Precious, and Panda three years ago.",
        "_context_packet": {"tier1": [], "tier2": [], "tier3": [], "tier4": []},
    }
    resolution = {
        "events": [{
            "event_id": "evt_answer_1",
            "time_start": "2020-01-21",
            "support_fact_ids": [fact["id"]],
            "source_span": {
                "episode_id": episode_id,
                "source_field": "raw_text",
                "start_char": 0,
                "end_char": len(raw_text),
            },
        }],
        "facts": [fact],
    }

    result = ms._attach_calendar_answer_resolution(
        query="Which year did Audrey adopt first three dogs?",
        recall_result=recall_result,
        candidate_facts=[fact],
        resolution=resolution,
    )

    assert "TEMPORAL EVIDENCE:" in result["context"]
    assert f"Source (raw_text, Episode {episode_id}" in result["context"]
    assert "three years ago" in result["context"]
    assert result["temporal_resolution"]["mode"] == "calendar-answer"
    assert any(
        "Source (raw_text, Episode" in str(item.get("text", ""))
        for item in result["_context_packet"]["tier1"]
        if isinstance(item, dict)
    )

def test_ask_returns_deterministic_exact_step_answer_without_llm(tmp_path):
    ms = MemoryServer(str(tmp_path), "ordinal_direct_answer")
    episode_id = "ORDINAL_e01"
    ms._data_dict = {}
    ms._episode_corpus = {
        "documents": [{
            "doc_id": "document:ORDINAL",
            "episodes": [{
                "episode_id": episode_id,
                "source_type": "document",
                "source_id": "ORDINAL",
                "source_date": "2026-03-01",
                "topic_key": "step 8 sql",
                "state_label": "trace",
                "currentness": "historical",
                "raw_text": (
                    "[Step 8]\n"
                    "Action: execute_snowflake_sql: SELECT * FROM wholesale WHERE year BETWEEN 2020 AND 2023\n"
                    "Observation: ok"
                ),
                "provenance": {"raw_span": [0, 113]},
            }],
        }],
    }
    ms._raw_sessions = [{"session_num": 1, "session_date": "2026-03-01"}]
    ms._all_granular = [{
        "id": "f_step_8",
        "session": 1,
        "kind": "fact",
        "fact": "At step 8, the agent ran SQL over wholesale for years 2020 through 2023.",
        "metadata": {"episode_id": episode_id, "episode_source_id": "ORDINAL"},
    }]
    ms._all_cons = []
    ms._all_cross = []
    ms._fact_lookup = {fact["id"]: fact for fact in ms._all_granular}
    ms._source_records = {"ORDINAL": {"family": "document"}}
    ms._rebuild_temporal_index()

    result = asyncio.run(ms.ask("At step 8, what SQL did the agent run?"))

    assert result["answer"] == "SELECT * FROM wholesale WHERE year BETWEEN 2020 AND 2023"
    assert result["profile_used"] == "deterministic:temporal_v1"
    assert result["tool_called"] is False
    assert result["runtime_trace"]["temporal_resolution"]["query_class"] == "ordinal"
def test_store_overrides_llm_session_with_caller_session(tmp_path, monkeypatch):
    """store() must persist caller session_num, not bogus model-emitted session."""
    async def _mock_call_extract(model, system, user_msg, max_tokens=8192, sem=None):
        return {
            "facts": [{
                "id": "f_01",
                "fact": "Apollo 11 landed on the Moon.",
                "session": 1969,
                "entities": ["Apollo 11"],
                "tags": ["history"],
            }],
            "temporal_links": [],
        }

    monkeypatch.setattr("src.memory.call_extract", _mock_call_extract)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda facts, lookup: None)
    ms = MemoryServer(str(tmp_path), "conv_session_fix")

    asyncio.run(ms.store("Apollo 11 landed on the Moon.", session_num=1, session_date="2024-06-01"))

    assert len(ms._all_granular) == 1
    assert ms._all_granular[0]["session"] == 1
    data = ms._storage.load_facts()
    assert data["granular"][0]["session"] == 1


def test_build_index_creates_embs(tmp_path, monkeypatch):
    """build_index() persists embeddings through the configured storage backend."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv2")

    asyncio.run(ms.store("Hello world", session_num=1, session_date="2024-06-01"))
    result = asyncio.run(ms.build_index())

    loaded = ms._storage.load_embeddings()
    assert loaded is not None
    assert "gran" in loaded
    assert loaded["gran"].shape[0] == 3
    assert loaded["gran"].shape[1] == DIM
    assert result["granular"] == 3


def test_recall_returns_context(tmp_path, monkeypatch):
    """recall() returns context string and metadata."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv3")

    asyncio.run(ms.store("Hello world", session_num=1, session_date="2024-06-01"))
    result = asyncio.run(ms.recall("What happened?"))

    assert "context" in result
    assert "query_type" in result
    assert "retrieved" in result
    assert result["n_facts"] >= 3
    assert isinstance(result["context"], str)
    assert len(result["context"]) > 0


def test_cache_survives_restart(tmp_path, monkeypatch):
    """MemoryServer reloads persisted cache from disk on init."""
    _patch_all(monkeypatch)

    ms1 = MemoryServer(str(tmp_path), "conv4")
    asyncio.run(ms1.store("Hello", session_num=1, session_date="2024-06-01"))
    assert ms1.stats()["granular"] == 3

    # Create a new instance — should reload from cache
    ms2 = MemoryServer(str(tmp_path), "conv4")
    assert ms2.stats()["granular"] == 3
    assert ms2._n_sessions == 1


def test_ingest_document(tmp_path, monkeypatch):
    """ingest_document() keeps granular facts and substrate cross facts."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv5")

    # Two chunks (each >8000 chars) so cross-session threshold (2+ sessions) is met
    chunk1 = "First section. " * 600  # ~9000 chars
    chunk2 = "Second section. " * 600
    n = asyncio.run(ms.ingest_document(
        content=f"{chunk1}\n\n{chunk2}",
        source_id="doc1",
    ))

    assert n > 0
    s = ms.stats()
    assert s["granular"] > 0
    assert s["consolidated"] == 0
    assert s["cross_session"] > 0
    assert ms._all_cons == []
    for f in ms._all_cross:
        assert "scope" in f, f"cross fact missing scope: {f}"
        assert "agent_id" in f, f"cross fact missing agent_id: {f}"
        assert "swarm_id" in f, f"cross fact missing swarm_id: {f}"
        assert "created_at" in f, f"cross fact missing created_at: {f}"


def test_augment_commonality_facts_prefers_interest_pairs_over_event_pairs():
    facts = [
        {"id": "j_event", "fact": "Joanna took a road trip for research for her next movie.", "session": 9, "speaker": "Joanna", "entities": ["Joanna"]},
        {"id": "n_event", "fact": "Nate thinks the road trip sounds great.", "session": 9, "speaker": "Nate", "entities": ["Nate"]},
        {"id": "j_movie", "fact": "Joanna enjoys reading, watching movies, and exploring nature, in addition to writing.", "session": 1, "speaker": "Joanna", "entities": ["Joanna"]},
        {"id": "n_movie", "fact": "Nate's main hobbies are playing video games and watching movies.", "session": 1, "speaker": "Nate", "entities": ["Nate"]},
        {"id": "j_dessert", "fact": "Joanna tries to make dairy-free desserts just as delicious as non-dairy ones.", "session": 10, "speaker": "Joanna", "entities": ["Joanna"]},
        {"id": "n_dessert", "fact": "Nate started teaching people how to make dairy-free desserts.", "session": 10, "speaker": "Nate", "entities": ["Nate"]},
    ]

    extras = _augment_commonality_facts(
        "What kind of interests do Joanna and Nate share?",
        [],
        facts,
        limit=4,
    )

    extra_ids = [fact["id"] for fact in extras]
    assert "j_movie" in extra_ids
    assert "n_movie" in extra_ids
    assert "j_dessert" in extra_ids
    assert "n_dessert" in extra_ids
    assert "j_event" not in extra_ids
    assert "n_event" not in extra_ids


def test_conversation_structural_packet_can_use_cross_only_substrate_facts(tmp_path, monkeypatch):
    async def mock_embed_query(text, **kwargs):
        return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr("src.memory.embed_query", mock_embed_query)

    ms = MemoryServer(str(tmp_path), "conv_cross_only")
    ms._all_granular = []
    ms._all_cross = [
        {
            "id": "substrate_shared_root",
            "fact": "Both lost jobs and started their own businesses.",
            "kind": "fact",
            "entities": ["Jon", "Gina"],
            "source_id": "conv-30_cat1",
            "session": 19,
            "metadata": {
                "source_aggregation": True,
                "episode_id": "conv-30_cat1_e01",
                "episode_ids": ["conv-30_cat1_e01"],
            },
        }
    ]
    ms._data_dict = {
        "cross_embs": np.array([[1.0, 0.0]], dtype=np.float32),
    }

    packet = {
        "query_operator_plan": {
            "commonality": {"enabled": True},
            "list_set": {"enabled": False},
            "compare_diff": {"enabled": False},
        },
        "retrieved_episode_ids": ["conv-30_cat1_e01"],
        "selector_config": {"budget": 4000},
        "tuning_snapshot": {"packet": {"snippet_chars": 600}},
    }
    episode_lookup = {
        "conv-30_cat1_e01": {
            "episode_id": "conv-30_cat1_e01",
            "source_id": "conv-30_cat1",
            "source_type": "conversation",
            "raw_text": "Jon lost his job as a banker. Gina lost her Door Dash job. Both started their own businesses.",
        }
    }

    augmented_packet, retrieved = asyncio.run(
        ms._augment_conversation_structural_packet(
            query="What do Jon and Gina have in common?",
            packet=packet,
            episode_lookup=episode_lookup,
            fact_filter=lambda _fact: True,
        )
    )

    assert retrieved is not None
    assert [fact["id"] for fact in retrieved] == ["substrate_shared_root"]
    assert augmented_packet["retrieved_fact_ids"] == ["substrate_shared_root"]
    assert "Both lost jobs and started their own businesses." in augmented_packet["context"]
    assert augmented_packet["actual_injected_episode_ids"] == ["conv-30_cat1_e01"]


def test_document_structural_packet_can_use_cross_only_substrate_facts(tmp_path):
    ms = MemoryServer(str(tmp_path), "doc_cross_only")
    ms._all_granular = []
    ms._all_cross = [
        {
            "id": "substrate_permit_record",
            "fact": "permit T-17 status approved date 2026-02-12 section Operations Update.",
            "kind": "fact",
            "entities": ["permit_T_17"],
            "source_id": "DOC-022",
            "session": 10,
            "metadata": {
                "source_aggregation": True,
                "episode_id": "DOC-022_e10",
                "episode_ids": ["DOC-022_e10"],
            },
        }
    ]
    ms._data_dict = {}

    packet = {
        "query_operator_plan": {
            "bounded_chain": {"enabled": True},
        },
        "retrieved_episode_ids": ["DOC-022_e10"],
        "retrieved_fact_ids": [],
        "selector_config": {"budget": 4000},
        "tuning_snapshot": {"packet": {"snippet_chars": 600, "query_specificity_bonus": 0.0}},
    }
    episode_lookup = {
        "DOC-022_e10": {
            "episode_id": "DOC-022_e10",
            "source_id": "DOC-022",
            "source_type": "document",
            "raw_text": "Permit T-17 was approved on 2026-02-12.",
        }
    }

    augmented_packet, retrieved = asyncio.run(
        ms._augment_document_structural_packet(
            query="Which permit was approved?",
            packet=packet,
            episode_lookup=episode_lookup,
            fact_filter=lambda _fact: True,
        )
    )

    assert retrieved is not None
    assert any(fact["id"] == "substrate_permit_record" for fact in retrieved)
    assert "substrate_permit_record" in augmented_packet["retrieved_fact_ids"]
    assert "Permit T-17 was approved on 2026-02-12." in augmented_packet["context"]

@pytest.mark.asyncio
async def test_temporal_recall_prefers_semantic_temporal_fact_over_conflicting_granular(tmp_path, monkeypatch):
    ms = MemoryServer(str(tmp_path), "episode_temporal_preference")
    episode_id = "conv-42_e01"
    source_id = "conv-42"
    ms._episode_corpus = {
        "documents": [{
            "doc_id": f"conversation:{source_id}",
            "episodes": [{
                "episode_id": episode_id,
                "source_id": source_id,
                "source_type": "conversation",
                "raw_text": (
                    'Joanna: I first watched "Eternal Sunshine of the Spotless Mind" around 3 years ago.'
                ),
                "topic_key": "session_1",
                "state_label": "session",
                "currentness": "unknown",
            }],
        }],
    }
    ms._all_granular = [
        {
            "id": "g_wrong_2020",
            "fact": "Joanna first watched the movie around 2020.",
            "kind": "fact",
            "entities": ["Joanna"],
            "source_id": source_id,
            "event_date": "2020",
            "metadata": {"episode_id": episode_id, "episode_source_id": source_id},
        }
    ]
    ms._all_cons = []
    ms._all_cross = [
        {
            "id": "substrate_2019",
            "fact": "Joanna first watched Eternal Sunshine of the Spotless Mind in 2019.",
            "kind": "fact",
            "entities": ["Joanna", "Eternal Sunshine of the Spotless Mind"],
            "source_id": source_id,
            "metadata": {
                "semantic_class": "temporal_semantics",
                "source_aggregation": True,
                "episode_id": episode_id,
                "episode_ids": [episode_id],
                "episode_source_id": source_id,
                "resolved_year": 2019,
            },
        }
    ]
    ms._data_dict = {
        "atomic_embs": np.array([[1.0, 0.0]], dtype=float),
        "cons_embs": np.zeros((0, 2), dtype=float),
        "cross_embs": np.array([[1.0, 0.0]], dtype=float),
        "fact_lookup": {
            "g_wrong_2020": ms._all_granular[0],
            "substrate_2019": ms._all_cross[0],
        },
    }
    ms._fact_lookup = dict(ms._data_dict["fact_lookup"])

    async def _fake_embed_query(_text, model=None, provider=None):
        return np.array([1.0, 0.0], dtype=float)

    monkeypatch.setattr("src.memory.embed_query", _fake_embed_query)

    result = await ms.recall('When did Joanna first watch "Eternal Sunshine of the Spotless Mind"?')

    retrieved_ids = [fact["id"] for fact in result["retrieved"]]
    assert "substrate_2019" in retrieved_ids
    assert "g_wrong_2020" not in retrieved_ids
    assert "2020" not in result["context"]
def test_context_for_passes_swarm_id(tmp_path, monkeypatch):
    """Bug A regression: context_for must pass swarm_id to recall."""
    _patch_all(monkeypatch, scope="swarm-shared", swarm_id="sw1")
    ms = MemoryServer(str(tmp_path), "conv_cfor", scope="swarm-shared", swarm_id="sw1")
    asyncio.run(ms.store("Test", session_num=1, session_date="2024-06-01"))

    # context_for must accept and forward swarm_id
    result = asyncio.run(ms.context_for("query", agent_id="a1", swarm_id="sw1"))
    assert "context" in result


def test_concurrent_store_no_corruption(tmp_path, monkeypatch):
    """Multiple concurrent store() calls don't corrupt data."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv10")

    async def _concurrent():
        tasks = [
            ms.store(f"Message {i}", session_num=i, session_date="2024-06-01")
            for i in range(5)
        ]
        results = await asyncio.gather(*tasks)
        return results

    results = asyncio.run(_concurrent())

    # All 5 stores returned 3 facts each
    assert all(r["facts_extracted"] == 3 for r in results)
    # Total facts = 5 * 3 = 15
    assert ms.stats()["granular"] == 15
    assert ms._n_sessions == 5
    # Persisted snapshot remains readable through the storage backend
    data = ms._storage.load_facts()
    assert len(data["granular"]) == 15


# ── 3-tier guarantee tests ──


def test_tiers_dirty_set_on_store(tmp_path, monkeypatch):
    """store() must set _tiers_dirty = True."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv_dirty1")

    assert ms._tiers_dirty is False
    asyncio.run(ms.store("Hello world", session_num=1, session_date="2024-06-01"))
    assert ms._tiers_dirty is True


def test_tiers_dirty_set_on_ingest_document(tmp_path, monkeypatch):
    """ingest_document() must set _tiers_dirty = True."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv_dirty3")

    assert ms._tiers_dirty is False
    chunk1 = "First section. " * 600
    chunk2 = "Second section. " * 600
    asyncio.run(ms.ingest_document(
        content=f"{chunk1}\n\n{chunk2}",
        source_id="doc1",
    ))
    assert ms._tiers_dirty is True


def test_build_index_rebuilds_tiers(tmp_path, monkeypatch):
    """build_index() must rebuild substrate cross facts when dirty."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv_rebuild1")

    # Store 3 facts
    asyncio.run(ms.store("Hello world", session_num=1, session_date="2024-06-01"))
    assert ms._tiers_dirty is True

    # build_index should rebuild tiers and clear the flag
    result = asyncio.run(ms.build_index())
    assert ms._tiers_dirty is False
    assert result["granular"] == 3
    assert result["consolidated"] == 0
    assert result["cross_session"] >= 1


def test_build_index_rebuilds_when_cons_empty(tmp_path, monkeypatch):
    """build_index() must rebuild substrate cross facts when cross tier is missing."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv_rebuild2")

    asyncio.run(ms.store("Hello world", session_num=1, session_date="2024-06-01"))
    # Simulate loaded-from-cache with missing substrate cross.
    ms._tiers_dirty = False
    ms._tier2_built = True
    ms._tier3_built = False
    ms._all_cross = []

    result = asyncio.run(ms.build_index())
    assert ms._tiers_dirty is False
    assert result["consolidated"] == 0
    assert result["cross_session"] >= 1


def test_build_index_namespaces_derived_tier_ids(tmp_path, monkeypatch):
    """Derived tiers must never reuse granular extractor IDs."""
    _patch_embeddings(monkeypatch)
    _patch_resolve_supersession(monkeypatch)

    async def mock_extract_session(**kwargs):
        sn = kwargs.get("session_num", 1)
        return _fake_extract_result(3, session=sn)

    async def mock_extract_source_aggregation_facts(self, **kwargs):
        return [{
            "id": "s1_f_01",
            "fact": "Derived duplicate id",
            "kind": "fact",
            "entities": ["Alice"],
            "tags": ["substrate"],
            "source_ids": ["s1_f_01"],
            "metadata": {"source_aggregation": True},
        }]

    monkeypatch.setattr("src.memory.extract_session", mock_extract_session)
    monkeypatch.setattr(MemoryServer, "_extract_source_aggregation_facts", mock_extract_source_aggregation_facts)

    ms = MemoryServer(str(tmp_path), "conv_collision")
    asyncio.run(ms.store("Hello world", session_num=1, session_date="2024-06-01"))

    result = asyncio.run(ms.build_index())
    assert result["cross_session"] >= 1
    assert ms._all_granular[0]["id"] != ms._all_cross[0]["id"]
    assert ms._all_cross[0]["id"].startswith("substrate_")


def test_flush_background_delegates_to_rebuild_tiers(tmp_path, monkeypatch):
    """flush_background() delegates to _rebuild_tiers() and clears dirty flag."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv_flush1")

    asyncio.run(ms.store("Hello world", session_num=1, session_date="2024-06-01"))
    assert ms._tiers_dirty is True

    result = asyncio.run(ms.flush_background())
    assert ms._tiers_dirty is False
    assert result["rebuilt"] is True
    assert "total_consolidated" in result
    assert "total_cross_session" in result


def test_rebuild_tiers_replaces_not_appends(tmp_path, monkeypatch):
    """_rebuild_tiers() replaces substrate cross, not appends."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv_nodup")

    asyncio.run(ms.store("Hello world", session_num=1, session_date="2024-06-01"))

    # Rebuild twice — cons/cross counts should stay the same, not double
    asyncio.run(ms._rebuild_tiers())
    cons_after_first = len(ms._all_cons)
    cross_after_first = len(ms._all_cross)

    asyncio.run(ms._rebuild_tiers())
    assert len(ms._all_cons) == 0 == cons_after_first
    assert len(ms._all_cross) == cross_after_first


def test_rebuild_tiers_respects_scope_boundaries(tmp_path, monkeypatch):
    """_rebuild_tiers() preserves identity fields on substrate cross facts."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv_scope1")

    # Store facts from two different agents with different scopes
    asyncio.run(ms.store("Agent A data", session_num=1, session_date="2024-06-01",
                         agent_id="agent-A", scope="agent-private"))
    asyncio.run(ms.store("Agent B data", session_num=2, session_date="2024-06-01",
                         agent_id="agent-B", scope="agent-private"))

    asyncio.run(ms._rebuild_tiers())

    assert ms._all_cons == []
    for f in ms._all_cross:
        assert "agent_id" in f, f"cross fact missing agent_id: {f}"
        assert "swarm_id" in f, f"cross fact missing swarm_id: {f}"
        assert "scope" in f, f"cross fact missing scope: {f}"
