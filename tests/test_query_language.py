"""Unit 2: query_type + kind parameters in memory_recall."""

import numpy as np
import pytest

from src.episode_features import extract_query_features
from src.retrieval import detect_query_type

DIM = 3072


def _seed_episode_runtime(server, fact_texts, source_id="qt_source"):
    episode_id = f"{source_id}_e0001"
    facts = []
    raw_text = "\n".join(fact_texts)
    for idx, text in enumerate(fact_texts, start=1):
        facts.append({
            "fact": text,
            "kind": "fact",
            "id": f"s1_f_{idx:02d}",
            "conv_id": server.key,
            "source_id": source_id,
            "episode_id": episode_id,
            "session": 1,
            "agent_id": "a",
            "swarm_id": "sw1",
            "scope": "swarm-shared",
            "owner_id": "system",
            "read": ["agent:PUBLIC"],
            "write": ["agent:PUBLIC"],
            "created_at": "2024-01-01T00:00:00+00:00",
            "entities": [],
            "tags": [],
            "metadata": {
                "episode_id": episode_id,
                "episode_source_id": source_id,
            },
        })

    server._all_granular = facts
    server._all_cons = []
    server._all_cross = []
    server._raw_sessions = [{
        "session_num": 1,
        "session_date": "2024-01-01",
        "content": raw_text,
        "speakers": "User and Assistant",
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
                "source_date": "2024-01-01",
                "topic_key": "session_1",
                "state_label": "session",
                "currentness": "unknown",
                "raw_text": raw_text,
                "provenance": {"raw_span": [0, len(raw_text)]},
            }],
        }],
    }
    server._register_source_record(source_id=source_id, family="conversation")


def test_detect_query_type_counting():
    assert detect_query_type("how many times did I visit the gym?") == "counting"


def test_detect_query_type_temporal():
    assert detect_query_type("when did the meeting happen?") == "temporal"
    assert detect_query_type("When was the company founded?") == "temporal"
    assert detect_query_type("Who did I meet last week?") == "temporal"
    assert detect_query_type("When Jon has lost his job as a banker?") == "temporal"


def test_detect_query_type_current():
    # Queries that explicitly ask for the current state route to the
    # current-state pipeline.
    assert detect_query_type("what is the current status?") == "current"
    assert detect_query_type("where does Alice currently live?") == "current"


def test_detect_query_type_rule():
    assert detect_query_type("what is the reimbursement policy?") == "rule"


def test_detect_query_type_default():
    assert detect_query_type("what color is the car?") == "default"


@pytest.fixture()
def _patch_embed(monkeypatch):
    async def _aembed_texts(texts, **kw):
        return np.random.randn(len(texts), DIM).astype(np.float32)
    async def _aembed_query(text, **kw):
        return np.random.randn(DIM).astype(np.float32)
    monkeypatch.setattr("src.memory.embed_texts", _aembed_texts)
    monkeypatch.setattr("src.memory.embed_query", _aembed_query)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda f, l: None)
    # Mock consolidation/cross-session so _rebuild_tiers() doesn't call real API
    async def _mock_consolidate(**kwargs):
        return ("mock", 1, "2024-01-01", [])
    async def _mock_cross(**kwargs):
        return ("mock", "ent", [])
    monkeypatch.setattr("src.memory.consolidate_session", _mock_consolidate)
    monkeypatch.setattr("src.memory.cross_session_entity", _mock_cross)
    monkeypatch.setattr("src.memory.call_extract", _mock_consolidate)


@pytest.mark.asyncio
async def test_memory_recall_accepts_query_type_param(tmp_path, _patch_embed):
    """memory_recall must accept query_type without error."""
    from src.memory import MemoryServer
    server = MemoryServer(data_dir=str(tmp_path), key="qt_test")
    _seed_episode_runtime(server, ["User prefers hiking in the mountains."])
    await server.build_index()

    result = await server.recall("what does user prefer?", query_type="synthesize")
    assert "query_type" in result
    assert result["query_type"] == "synthesis"


@pytest.mark.asyncio
async def test_memory_recall_synthesis_counts_source_excerpt_as_second_evidence(tmp_path, _patch_embed):
    from src.memory import MemoryServer

    server = MemoryServer(data_dir=str(tmp_path), key="qt_synth_source_support")
    _seed_episode_runtime(server, ["User prefers hiking in the mountains."])
    await server.build_index()

    result = await server.recall("What does user prefer and why?", query_type="synthesize")

    assert result["query_type"] == "synthesis"
    assert len(result["retrieved"]) >= 2
    assert any(item.get("kind") == "source_excerpt" for item in result["retrieved"])


@pytest.mark.asyncio
async def test_memory_recall_synthesis_tops_up_support_from_same_episode(tmp_path, _patch_embed, monkeypatch):
    from src.memory import MemoryServer

    server = MemoryServer(data_dir=str(tmp_path), key="qt_synth_topup")
    _seed_episode_runtime(server, [
        "User prefers PostgreSQL.",
        "User trusts PostgreSQL because of MVCC.",
    ])
    await server.build_index()

    def _fake_packet(*args, **kwargs):
        return {
            "context": "RETRIEVED FACTS:\n[1] (S1) User prefers PostgreSQL. [Episode: qt_source_e0001]",
            "retrieved_fact_ids": ["s1_f_01"],
            "retrieved_episode_ids": ["qt_source_e0001"],
            "actual_injected_episode_ids": ["qt_source_e0001"],
            "fact_episode_ids": ["qt_source_e0001"],
            "selection_scores": [],
            "query_operator_plan": extract_query_features("What database do I prefer and why?")["operator_plan"],
            "output_constraints": {},
            "retrieval_families": ["conversation"],
            "search_family": "conversation",
            "selector_config": {"budget": 8000},
            "tuning_snapshot": {"packet": {"snippet_chars": 1200}},
            "family_first_pass_trace": {"available_families": ["conversation"], "retrieval_families": ["conversation"], "requested_search_family": "auto", "per_family": []},
            "late_fusion_trace": {},
        }

    monkeypatch.setattr("src.memory.build_episode_hybrid_context", _fake_packet)

    result = await server.recall("What database do I prefer and why?", query_type="synthesize")

    assert result["query_type"] == "synthesis"
    assert len(result["retrieved"]) >= 2
    assert any("MVCC" in fact.get("fact", "") for fact in result["retrieved"])


@pytest.mark.asyncio
async def test_memory_recall_kind_filter(tmp_path, _patch_embed):
    """kind filter must exclude facts of wrong kind."""
    from src.memory import MemoryServer
    server = MemoryServer(data_dir=str(tmp_path), key="kind_test")
    _seed_episode_runtime(server, [
        "Budget is $500k.",
        "User prefers Python.",
    ])
    server._all_granular[0]["kind"] = "constraint"
    server._all_granular[1]["kind"] = "preference"
    await server.build_index()

    result = await server.recall("budget", kind="constraint")
    # Only constraint facts should appear
    for item in result.get("retrieved", []):
        assert item.get("kind") == "constraint", \
            f"Non-constraint fact leaked through kind filter: {item}"


@pytest.mark.asyncio
async def test_memory_recall_kind_filter_empty_returns_normal_empty_result(tmp_path, _patch_embed):
    from src.memory import MemoryServer

    server = MemoryServer(data_dir=str(tmp_path), key="kind_empty")
    _seed_episode_runtime(server, ["Budget is $500k."])
    server._all_granular[0]["kind"] = "constraint"
    await server.build_index()

    result = await server.recall("budget", kind="preference")

    assert "error" not in result
    assert result["retrieved"] == []
    assert result["actual_injected_episode_ids"] == []
    assert result["runtime_trace"]["reason"] == "empty_visible_facts"


@pytest.mark.asyncio
async def test_memory_recall_kind_all_returns_everything(tmp_path, _patch_embed):
    """kind='all' (default) must not filter by kind."""
    from src.memory import MemoryServer
    server = MemoryServer(data_dir=str(tmp_path), key="kind_all")
    _seed_episode_runtime(server, [
        "Budget is $500k.",
        "User prefers Python.",
    ])
    server._all_granular[0]["kind"] = "constraint"
    server._all_granular[1]["kind"] = "preference"
    await server.build_index()

    result = await server.recall("anything", kind="all")
    # With kind="all", no kind filtering is applied — both facts are candidates
    # (context may be empty with random embeddings, but retrieved should work)
    assert result["n_facts"] == 2


@pytest.mark.asyncio
async def test_mcp_memory_recall_new_params_accepted(tmp_path, monkeypatch):
    """MCP memory_recall accepts query_type and kind without error."""
    import src.mcp_server as mcp_mod
    mcp_mod.data_dir = str(tmp_path)
    mcp_mod.registry.clear()
    mcp_mod.courier_registry.clear()
    mcp_mod.connections.clear()
    mcp_mod.sub_to_conn.clear()

    async def _aembed_texts(texts, **kw):
        return np.random.randn(len(texts), DIM).astype(np.float32)
    async def _aembed_query(text, **kw):
        return np.random.randn(DIM).astype(np.float32)
    monkeypatch.setattr("src.memory.embed_texts", _aembed_texts)
    monkeypatch.setattr("src.memory.embed_query", _aembed_query)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda f, l: None)
    # Seed an episode-backed fact set so recall uses the production runtime
    server = mcp_mod._get_memory("param_test")
    _seed_episode_runtime(server, ["Test fact."])

    result = await mcp_mod.memory_recall(
        key="param_test",
        query="test",
        agent_id="a",
        swarm_id="sw1",
        query_type="lookup",
        kind="fact",
    )
    assert "context" in result
    assert "query_type" in result
