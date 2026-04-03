"""Phase 2: Integration tests for Features v1.2 (raw sessions + query language)."""

import asyncio

import numpy as np
import pytest

from src.memory import MemoryServer

DIM = 3072


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch):
    async def mock_extract(**kwargs):
        sn = kwargs.get("session_num", 1)
        text = kwargs.get("session_text", "")
        facts = [
            {"id": f"f{sn}_{i}", "fact": f"{text[:50]} (fact {i})", "kind": "event",
             "entities": ["User"], "tags": [], "session": sn}
            for i in range(3)
        ]
        # Detect rule-like content
        if "policy" in text.lower() or "require" in text.lower() or "rule" in text.lower():
            facts[0]["kind"] = "rule"
        if "prefer" in text.lower() or "like" in text.lower():
            facts[0]["kind"] = "preference"
        return ("conv", sn, "2024-06-01", facts, [])

    async def mock_consolidate(**kwargs):
        return ("conv", 1, "2024-06-01", [])

    async def mock_cross(**kwargs):
        return ("conv", "e", [])

    monkeypatch.setattr("src.memory.extract_session", mock_extract)
    monkeypatch.setattr("src.memory.consolidate_session", mock_consolidate)
    monkeypatch.setattr("src.memory.cross_session_entity", mock_cross)
    async def _aembed_texts(texts, **kw):
        return np.random.randn(len(texts), DIM).astype(np.float32)
    async def _aembed_query(text, **kw):
        return np.random.randn(DIM).astype(np.float32)
    monkeypatch.setattr("src.memory.embed_texts", _aembed_texts)
    monkeypatch.setattr("src.memory.embed_query", _aembed_query)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda f, l: None)


@pytest.mark.asyncio
async def test_reextract_after_multiple_stores(tmp_path):
    """reextract() after 3 stores must regenerate facts from all 3 raw sessions."""
    server = MemoryServer(data_dir=str(tmp_path), key="multi_reextract")
    contents = [
        "User: I spent 70 hours playing Assassin's Creed Odyssey.",
        "User: I moved to Seattle last year.",
        "User: My budget for the project is $50,000.",
    ]
    for i, c in enumerate(contents):
        await server.store(c, i + 1, "2024-01-01")

    assert len(server._raw_sessions) == 3

    # Clear facts to simulate corruption
    server._all_granular = []
    result = await server.reextract()

    assert result["sessions"] == 3
    assert result["reextracted"] >= 3  # at least 1 fact per session
    assert len(server._raw_sessions) == 3  # raw unchanged


@pytest.mark.asyncio
async def test_query_type_procedural_finds_rules(tmp_path):
    """query_type=procedural must route to rule pipeline."""
    server = MemoryServer(data_dir=str(tmp_path), key="proc_test")
    server._all_granular = [
        {"fact": "All expenses over $500 require VP approval.",
         "kind": "rule", "id": "af_001", "conv_id": "proc_test",
         "agent_id": "a", "swarm_id": "sw1", "scope": "swarm-shared",
         "created_at": "2024-01-01T00:00:00+00:00"},
        {"fact": "User went to the gym yesterday.",
         "kind": "fact", "id": "af_002", "conv_id": "proc_test",
         "agent_id": "a", "swarm_id": "sw1", "scope": "swarm-shared",
         "created_at": "2024-01-01T00:00:00+00:00"},
    ]
    server._all_cons = []
    server._all_cross = []
    await server.build_index()

    result = await server.recall("expense policy", query_type="procedural")
    assert result["query_type"] == "rule"


@pytest.mark.asyncio
async def test_raw_sessions_plus_query_type_end_to_end(tmp_path):
    """Full flow: store -> raw saved -> reextract -> recall with query_type."""
    server = MemoryServer(data_dir=str(tmp_path), key="e2e_test")
    await server.store(
        "The reimbursement policy requires receipts for all expenses over $100.",
        1, "2024-01-01", agent_id="a", swarm_id="sw1", scope="swarm-shared"
    )
    # Raw session stored
    assert len(server._raw_sessions) == 1

    await server.build_index()
    result = await server.recall(
        "reimbursement requirements",
        query_type="procedural",
        kind="all",
    )
    assert result["query_type"] == "rule"
    assert result["n_facts"] >= 3


@pytest.mark.asyncio
async def test_recall_summarize_end_to_end_uses_session_coverage(tmp_path):
    """Summarize recall should preserve full session coverage and summarize routing."""
    server = MemoryServer(data_dir=str(tmp_path), key="summarize_e2e")
    for sn in range(1, 4):
        await server.store(
            f"Session {sn}: user discussed a different topic with enough detail to summarize.",
            sn,
            "2024-01-01",
        )

    await server.build_index()
    result = await server.recall("Write a summary of about 1000 words.")

    assert result["query_type"] == "summarize"
    assert result["recommended_prompt_type"] == "summarize_with_metadata"
    assert result["use_tool"] is True
    assert result["total_sessions"] == 3
    assert result["sessions_in_context"] == 3
    assert result["coverage_pct"] == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_recall_summarize_returns_normalized_retrieved_items(tmp_path):
    """Summarize recall must expose canonical retrieved item shape after post-processing."""
    server = MemoryServer(data_dir=str(tmp_path), key="summarize_shape")
    for sn in range(1, 3):
        await server.store(
            f"Session {sn}: enough information for summarize retrieval.",
            sn,
            "2024-01-01",
        )

    await server.build_index()
    result = await server.recall("Write a summary of about 800 words.")

    assert result["query_type"] == "summarize"
    assert result["retrieved"], "summarize recall should return retrieved items"
    assert all("fact_id" in item for item in result["retrieved"])
    assert all("conv_id" in item for item in result["retrieved"])
    assert all("sim" in item for item in result["retrieved"])


@pytest.mark.asyncio
async def test_reextract_preserves_identity_from_raw(tmp_path):
    """reextract() must use agent_id/swarm_id/scope from raw sessions, not instance defaults."""
    server = MemoryServer(data_dir=str(tmp_path), key="reex_identity",
                          agent_id="default_agent", swarm_id="default_swarm")
    await server.store("Some content.", 1, "2024-01-01",
                       agent_id="agent_x", swarm_id="sw1", scope="agent-private")

    result = await server.reextract()
    assert result["reextracted"] >= 1
    # Facts should carry identity from raw session, not instance
    for f in server._all_granular:
        assert f["agent_id"] == "agent_x"
        assert f["swarm_id"] == "sw1"


@pytest.mark.asyncio
async def test_kind_filter_with_scope_filter_compose(tmp_path):
    """kind filter must compose with scope filter — both must be satisfied."""
    server = MemoryServer(data_dir=str(tmp_path), key="compose_test")
    server._all_granular = [
        {"fact": "Private rule.", "kind": "rule", "id": "af_001",
         "conv_id": "compose_test", "agent_id": "agent_a", "swarm_id": "sw1",
         "scope": "agent-private", "created_at": "2024-01-01T00:00:00+00:00"},
        {"fact": "Shared rule.", "kind": "rule", "id": "af_002",
         "conv_id": "compose_test", "agent_id": "agent_b", "swarm_id": "sw1",
         "scope": "swarm-shared", "created_at": "2024-01-01T00:00:00+00:00"},
        {"fact": "Shared fact.", "kind": "fact", "id": "af_003",
         "conv_id": "compose_test", "agent_id": "agent_b", "swarm_id": "sw1",
         "scope": "swarm-shared", "created_at": "2024-01-01T00:00:00+00:00"},
    ]
    server._all_cons = []
    server._all_cross = []
    await server.build_index()

    # agent_b sees swarm-shared only, filtered to kind=rule
    result = await server.recall("rules", agent_id="agent_b", swarm_id="sw1", kind="rule")
    # Private rule (agent_a) must not appear in context
    assert "Private rule" not in result["context"]
