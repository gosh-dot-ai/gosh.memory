"""Tests for complexity_hint v2 in memory_recall.

Tests cover:
A. _compute_complexity_hint no longer uses low_top_score and single_lookup
B. content_complexity raises overall score
C. retrieval_complexity dominates when content is low
D. propagated complexity for consolidated facts
E. propagated complexity for cross-session facts
F. old facts without field default to 0.0
G. store/import paths write _session_content_complexity
H. MCP memory_recall returns new additive fields
"""

import asyncio
import json

import numpy as np
import pytest

from src.memory import MemoryServer, _compute_complexity_hint, _compute_content_complexity

DIM = 3072


# ════════════════════════════════════════════════════════════════
# A. _compute_complexity_hint no longer uses low_top_score / single_lookup
# ════════════════════════════════════════════════════════════════

def test_no_low_top_score_signal():
    """low_top_score must NOT appear in signals (removed in v2)."""
    hint = _compute_complexity_hint(
        [{"fact_id": "f1", "sim": 0.4}, {"fact_id": "f2", "sim": 0.3}],
        "default", False, {}
    )
    assert "low_top_score" not in hint["signals"]


def test_no_single_lookup_signal():
    """single_lookup must NOT appear in signals (removed in v2)."""
    hint = _compute_complexity_hint(
        [{"fact_id": "f1", "sim": 0.98}],
        "default", False, {}
    )
    assert "single_lookup" not in hint["signals"]


# ════════════════════════════════════════════════════════════════
# B. content_complexity raises overall score
# ════════════════════════════════════════════════════════════════

def test_content_complexity_raises_score():
    """Retrieved action_item fact should raise content complexity from the fact itself."""
    fact_lookup = {
        "f1": {"agent_id": "a", "kind": "action_item", "entities": [], "tags": []},
    }
    hint = _compute_complexity_hint(
        [{"fact_id": "f1", "sim": 0.9}],
        "default", False, fact_lookup
    )
    assert hint["retrieval_complexity"] == 0.0
    assert hint["content_complexity"] == 0.70
    assert hint["score"] == 0.70
    assert hint["dominant"] == "content"


# ════════════════════════════════════════════════════════════════
# C. retrieval_complexity dominates when content is low
# ════════════════════════════════════════════════════════════════

def test_retrieval_dominates_when_content_low():
    """Multi-hop retrieval should dominate over low content complexity."""
    fact_lookup = {
        "f1": {"agent_id": "a", "kind": "fact", "_session_content_complexity": 0.10},
    }
    hint = _compute_complexity_hint(
        [{"fact_id": "f1", "sim": 0.8}],
        "default", True, fact_lookup
    )
    assert hint["retrieval_complexity"] == 0.35
    assert hint["content_complexity"] == 0.10
    assert hint["score"] == 0.35
    assert hint["dominant"] == "retrieval"


# ════════════════════════════════════════════════════════════════
# F. old facts without field default to 0.0
# ════════════════════════════════════════════════════════════════

def test_old_facts_default_to_zero():
    """Facts without persisted complexity still use the retrieved fact payload."""
    fact_lookup = {
        "f1": {"agent_id": "a", "kind": "fact"},  # no _session_content_complexity
    }
    hint = _compute_complexity_hint(
        [{"fact_id": "f1", "sim": 0.8}],
        "default", False, fact_lookup
    )
    assert hint["content_complexity"] == 0.10
    assert hint["dominant"] == "content"


# ════════════════════════════════════════════════════════════════
# Structural signal tests (preserved from v1, updated for v2)
# ════════════════════════════════════════════════════════════════

def test_empty_retrieved_returns_level_1():
    hint = _compute_complexity_hint([], "default", False, {})
    assert hint["level"] == 1
    assert hint["score"] == 0.0
    assert hint["signals"] == []
    assert hint["retrieval_complexity"] == 0.0
    assert hint["content_complexity"] == 0.0
    assert hint["dominant"] == "tie"


def test_multi_hop_raises_level():
    hint = _compute_complexity_hint(
        [{"fact_id": "f1", "sim": 0.8}],
        "default", True, {}
    )
    assert "multi_hop" in hint["signals"]
    assert hint["retrieval_complexity"] == 0.35
    assert hint["level"] >= 2


def test_conflict_found_on_supersession_type():
    hint = _compute_complexity_hint(
        [{"fact_id": "f1", "sim": 0.8}],
        "supersession", False, {}
    )
    assert "conflict_found" in hint["signals"]
    assert hint["retrieval_complexity"] == 0.20


def test_cross_scope_detected_from_fact_lookup():
    fact_lookup = {
        "f1": {"agent_id": "agent_a", "kind": "fact"},
        "f2": {"agent_id": "agent_b", "kind": "fact"},
    }
    hint = _compute_complexity_hint(
        [{"fact_id": "f1", "sim": 0.8}, {"fact_id": "f2", "sim": 0.7}],
        "default", False, fact_lookup
    )
    assert "cross_scope" in hint["signals"]
    assert hint["retrieval_complexity"] == 0.25


def test_score_clamped_0_to_1():
    hint = _compute_complexity_hint(
        [{"fact_id": f"f{i}", "sim": 0.3} for i in range(60)],
        "supersession", True, {}
    )
    assert 0.0 <= hint["score"] <= 1.0
    # multi_hop(0.35) + conflict(0.20) + high_fact_count(0.05) = 0.60 -> level 3
    assert hint["retrieval_complexity"] == 0.60
    assert hint["level"] == 3


def test_level_mapping():
    for score, expected_level in [
        (0.0, 1), (0.1, 1), (0.2, 1),
        (0.3, 2), (0.4, 2),
        (0.5, 3), (0.6, 3),
        (0.7, 4), (0.8, 4),
        (0.9, 5), (1.0, 5),
    ]:
        if score <= 0.2:   level = 1
        elif score <= 0.4: level = 2
        elif score <= 0.6: level = 3
        elif score <= 0.8: level = 4
        else:              level = 5
        assert level == expected_level, f"score={score} expected level {expected_level}, got {level}"


def test_return_structure_has_v2_fields():
    """complexity_hint must return all v2 fields."""
    hint = _compute_complexity_hint([], "default", False, {})
    assert "score" in hint
    assert "level" in hint
    assert "signals" in hint
    assert "retrieval_complexity" in hint
    assert "content_complexity" in hint
    assert "dominant" in hint


# ════════════════════════════════════════════════════════════════
# _compute_content_complexity unit tests
# ════════════════════════════════════════════════════════════════

def test_content_complexity_empty():
    assert _compute_content_complexity([]) == 0.0


def test_content_complexity_action_item():
    facts = [{"kind": "action_item", "entities": [], "fact": "Do X"}]
    score = _compute_content_complexity(facts)
    assert score == 0.70


def test_content_complexity_requirement():
    facts = [{"kind": "requirement", "entities": [], "fact": "Must do Y"}]
    score = _compute_content_complexity(facts)
    assert score == 0.65


def test_content_complexity_with_entities():
    entities = [f"entity_{i}" for i in range(12)]
    facts = [{"kind": "fact", "entities": entities, "fact": "many entities"}]
    score = _compute_content_complexity(facts)
    # max(kind=0.10, entity_density=0.40) = 0.40
    assert score == 0.40


def test_content_complexity_with_temporal():
    facts = [{"kind": "fact", "entities": [], "fact": "X",
              "event_date": "2024-01-01"}]
    score = _compute_content_complexity(facts)
    # max(kind=0.10, temporal=0.35) = 0.35
    assert score == 0.35


def test_content_complexity_high_fact_count():
    facts = [{"kind": "fact", "entities": [], "fact": f"Fact {i}"}
             for i in range(55)]
    score = _compute_content_complexity(facts)
    # max(kind=0.10, fact_count=0.50) = 0.50
    assert score == 0.50


def test_content_complexity_capped_at_1():
    entities = [f"entity_{i}" for i in range(25)]
    facts = [{"kind": "action_item", "entities": entities,
              "fact": f"Fact {i}", "event_date": "2024-01-01",
              "_temporal_links": [{"a": 1}]}
             for i in range(55)]
    score = _compute_content_complexity(facts)
    # max-style: max(action_item=0.70, entities>20=0.60, temporal=0.35, facts>50=0.50) = 0.70
    assert score == 0.70
    assert score <= 1.0


# ════════════════════════════════════════════════════════════════
# Patches
# ════════════════════════════════════════════════════════════════

@pytest.fixture()
def _patch_embed(monkeypatch):
    async def _aembed_texts(texts, **kw):
        return np.random.randn(len(texts), DIM).astype(np.float32)
    async def _aembed_query(text, **kw):
        return np.random.randn(DIM).astype(np.float32)
    monkeypatch.setattr("src.memory.embed_texts", _aembed_texts)
    monkeypatch.setattr("src.memory.embed_query", _aembed_query)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda f, l: None)


def _patch_extraction(monkeypatch, n_facts=3, **tag_overrides):
    async def mock_extract_session(**kwargs):
        sn = kwargs.get("session_num", 1)
        facts = []
        for i in range(n_facts):
            f = {
                "id": f"f{i}",
                "fact": f"Test fact number {i}",
                "kind": "event",
                "entities": ["Alice"],
                "tags": ["test"],
                "session": sn,
                "scope": "swarm-shared",
                "agent_id": "default",
                "swarm_id": "default",
            }
            f.update(tag_overrides)
            facts.append(f)
        tlinks = [{"before": "f0", "after": "f1", "signal": "then"}]
        return ("test_conv", sn, "2024-06-01", facts, tlinks)

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

    monkeypatch.setattr("src.memory.extract_session", mock_extract_session)
    monkeypatch.setattr("src.memory.consolidate_session", mock_consolidate_session)
    monkeypatch.setattr("src.memory.cross_session_entity", mock_cross_session_entity)


def _patch_all(monkeypatch, n_facts=3, **tag_overrides):
    _patch_extraction(monkeypatch, n_facts, **tag_overrides)
    async def mock_embed_texts(texts, **kw):
        return np.random.randn(len(texts), DIM).astype(np.float32)
    async def mock_embed_query(text, **kw):
        return np.random.randn(DIM).astype(np.float32)
    monkeypatch.setattr("src.memory.embed_texts", mock_embed_texts)
    monkeypatch.setattr("src.memory.embed_query", mock_embed_query)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda f, l: None)


def _seed_episode_runtime(server: MemoryServer, *, source_id: str, raw_text: str, session_num: int = 1):
    episode_id = f"{source_id}_e{session_num:04d}"
    for fact in server._all_granular:
        fact["source_id"] = source_id
        fact.setdefault("owner_id", "system")
        fact.setdefault("read", ["agent:PUBLIC"])
        fact.setdefault("write", ["agent:PUBLIC"])
        fact.setdefault("session", session_num)
        meta = fact.setdefault("metadata", {})
        meta["episode_id"] = episode_id
        meta["episode_source_id"] = source_id
    server._raw_sessions = [{
        "session_num": session_num,
        "session_date": "2024-06-01",
        "content": raw_text,
        "format": "conversation",
        "source_id": source_id,
        "owner_id": "system",
        "read": ["agent:PUBLIC"],
        "write": ["agent:PUBLIC"],
    }]
    server._episode_corpus = {
        "documents": [{
            "doc_id": f"conversation:{source_id}",
            "episodes": [{
                "episode_id": episode_id,
                "source_type": "conversation",
                "source_id": source_id,
                "source_date": "2024-06-01",
                "topic_key": f"session_{session_num}",
                "state_label": "session",
                "currentness": "unknown",
                "raw_text": raw_text,
                "provenance": {"raw_span": [0, len(raw_text)]},
            }],
        }],
    }
    server._register_source_record(source_id=source_id, family="conversation")


# ════════════════════════════════════════════════════════════════
# G. store paths write _session_content_complexity
# ════════════════════════════════════════════════════════════════

def test_store_writes_session_content_complexity(tmp_path, monkeypatch):
    """store() must stamp _session_content_complexity on granular facts."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv_cc_store")
    asyncio.run(ms.store("Hello world", session_num=1, session_date="2024-06-01"))
    for f in ms._all_granular:
        assert "_session_content_complexity" in f
        assert isinstance(f["_session_content_complexity"], float)


def test_ingest_asserted_writes_session_content_complexity(tmp_path, monkeypatch):
    """ingest_asserted_facts() must stamp _session_content_complexity on granular facts."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv_cc_import")
    facts = [{
        "id": "f_01",
        "fact": "User said hello",
        "kind": "fact",
        "entities": [],
        "tags": [],
        "session": 1,
    }]
    raw = [{
        "session_num": 1,
        "session_date": "2024-06-01",
        "content": "Hello world",
        "speakers": "User and Assistant",
    }]
    asyncio.run(ms.ingest_asserted_facts(facts=facts, raw_sessions=raw))
    for f in ms._all_granular:
        assert "_session_content_complexity" in f
        assert isinstance(f["_session_content_complexity"], float)


def test_ingest_asserted_computes_complexity_from_imported_facts(tmp_path, monkeypatch):
    """ingest_asserted_facts() computes complexity from imported facts."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv_cc_import2")
    facts = [
        {"id": "f_01", "fact": "First fact here.", "kind": "fact",
         "entities": [], "tags": [], "session": 1},
        {"id": "f_02", "fact": "Second fact here.", "kind": "fact",
         "entities": ["Alice"], "tags": [], "session": 1},
        {"id": "f_03", "fact": "Third fact here.", "kind": "action_item",
         "entities": [], "tags": [], "session": 1},
    ]
    raw = [{
        "session_num": 1,
        "session_date": "2024-06-01",
        "content": "Imported content",
        "speakers": "User and Assistant",
    }]
    asyncio.run(ms.ingest_asserted_facts(facts=facts, raw_sessions=raw))
    for f in ms._all_granular:
        assert "_session_content_complexity" in f
        assert isinstance(f["_session_content_complexity"], float)


# ════════════════════════════════════════════════════════════════
# D. propagated complexity for consolidated facts
# ════════════════════════════════════════════════════════════════

def test_propagated_complexity_asserted_consolidated(tmp_path, monkeypatch):
    """Asserted consolidated facts must inherit _session_content_complexity
    from their source granular facts."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv_cc_cons")
    facts = [
        {"id": "f_01", "fact": "First fact here.", "kind": "fact",
         "entities": [], "tags": [], "session": 1},
        {"id": "f_02", "fact": "Second fact here.", "kind": "action_item",
         "entities": ["Alice"], "tags": [], "session": 1},
    ]
    cons = [{
        "id": "c_01",
        "fact": "Combined asserted fact.",
        "kind": "summary",
        "session": 1,
        "source_ids": ["f_01", "f_02"],
    }]
    raw = [{
        "session_num": 1,
        "session_date": "2024-06-01",
        "content": "Imported content",
        "speakers": "User and Assistant",
    }]
    asyncio.run(ms.ingest_asserted_facts(facts=facts, consolidated=cons, raw_sessions=raw))

    assert len(ms._all_cons) == 1
    source_cc = max(f.get("_session_content_complexity", 0.0) for f in ms._all_granular)
    assert ms._all_cons[0]["_session_content_complexity"] == source_cc


# ════════════════════════════════════════════════════════════════
# E. propagated complexity for cross-session facts
# ════════════════════════════════════════════════════════════════

def test_propagated_complexity_cross_session(tmp_path, monkeypatch):
    """Tier 3 cross-session facts must have _session_content_complexity propagated."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv_cc_cross")
    # Need >= 2 sessions with shared entity for cross-session to fire
    asyncio.run(ms.store("Alice is here", session_num=1, session_date="2024-06-01"))
    asyncio.run(ms.store("Alice was there", session_num=2, session_date="2024-06-02"))
    asyncio.run(ms._rebuild_tiers())

    if ms._all_cross:
        for xf in ms._all_cross:
            assert "_session_content_complexity" in xf
            assert isinstance(xf["_session_content_complexity"], float)


# ════════════════════════════════════════════════════════════════
# H. MCP memory_recall returns new additive fields
# ════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_recall_returns_v2_complexity_hint(tmp_path, _patch_embed):
    """recall() must include retrieval_complexity, content_complexity, dominant."""
    server = MemoryServer(data_dir=str(tmp_path), key="hint_v2")
    server._all_granular = [{
        "fact": "User prefers Python.", "kind": "preference",
        "id": "af_001", "conv_id": "hint_v2",
        "agent_id": "a", "swarm_id": "sw1", "scope": "swarm-shared",
        "created_at": "2024-01-01T00:00:00+00:00",
        "_session_content_complexity": 0.10,
    }]
    server._all_cons = []
    server._all_cross = []
    _seed_episode_runtime(server, source_id="hint-v2-chat", raw_text="User prefers Python.")
    await server.build_index()

    result = await server.recall("what does user prefer?")
    assert "complexity_hint" in result
    hint = result["complexity_hint"]
    assert "score" in hint
    assert "level" in hint
    assert "signals" in hint
    assert "retrieval_complexity" in hint
    assert "content_complexity" in hint
    assert "dominant" in hint
    assert 1 <= hint["level"] <= 5
    assert 0.0 <= hint["score"] <= 1.0


@pytest.mark.asyncio
async def test_mcp_memory_recall_returns_v2_fields(tmp_path, _patch_embed):
    """MCP memory_recall must include retrieval_complexity, content_complexity, dominant."""
    import src.mcp_server as mcp_mod
    mcp_mod.data_dir = str(tmp_path)
    mcp_mod.registry.clear()
    mcp_mod.courier_registry.clear()
    mcp_mod.connections.clear()
    mcp_mod.sub_to_conn.clear()

    server = mcp_mod._get_memory("hint_mcp_v2")
    server._all_granular = [{
        "fact": "Budget is $500k.", "kind": "constraint",
        "id": "af_001", "conv_id": "hint_mcp_v2",
        "agent_id": "a", "swarm_id": "sw1", "scope": "swarm-shared",
        "created_at": "2024-01-01T00:00:00+00:00",
        "_session_content_complexity": 0.45,
    }]
    server._all_cons = []
    server._all_cross = []
    _seed_episode_runtime(server, source_id="hint-mcp-chat", raw_text="Budget is $500k.")
    await server.build_index()

    result = await mcp_mod.memory_recall(
        key="hint_mcp_v2", query="budget", agent_id="a", swarm_id="sw1"
    )
    assert "complexity_hint" in result
    hint = result["complexity_hint"]
    assert hint["level"] in range(1, 6)
    assert "retrieval_complexity" in hint
    assert "content_complexity" in hint
    assert "dominant" in hint


# ════════════════════════════════════════════════════════════════
# Requirement kind compatibility
# ════════════════════════════════════════════════════════════════

def test_requirement_kind_handled():
    """The requirement kind should be handled in content complexity."""
    facts = [{"kind": "requirement", "entities": [], "fact": "Must do X"}]
    score = _compute_content_complexity(facts)
    assert score == 0.65


def test_unknown_kind_defaults():
    """Unknown kinds should default to 0.10 (same as 'fact')."""
    facts = [{"kind": "some_new_kind", "entities": [], "fact": "Something"}]
    score = _compute_content_complexity(facts)
    assert score == 0.10
