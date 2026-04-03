"""BUG-1: Race condition in store/ingest_document — per-call agent_id/swarm_id/scope."""

import asyncio

import numpy as np
import pytest

from src.memory import MemoryServer

DIM = 3072


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch):
    async def mock_extract(**kwargs):
        text = kwargs.get("session_text", "")
        sn = kwargs.get("session_num", 1)
        facts = [
            {"id": f"f{sn}_{i}", "fact": f"{text} (fact {i})", "kind": "event",
             "entities": [], "tags": [], "session": sn}
            for i in range(3)
        ]
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
async def test_concurrent_store_distinct_agent_ids(tmp_path):
    """Two concurrent store() calls with different agent_ids must tag facts correctly."""
    server = MemoryServer(data_dir=str(tmp_path), key="test_race")

    await asyncio.gather(
        server.store("Alice loves hiking.", 1, "2024-01-01",
                     agent_id="agent_a", swarm_id="sw1", scope="swarm-shared"),
        server.store("Bob prefers swimming.", 2, "2024-01-01",
                     agent_id="agent_b", swarm_id="sw1", scope="swarm-shared"),
    )

    agent_a_facts = [f for f in server._all_granular if f.get("agent_id") == "agent_a"]
    agent_b_facts = [f for f in server._all_granular if f.get("agent_id") == "agent_b"]

    assert len(agent_a_facts) > 0, "agent_a facts missing"
    assert len(agent_b_facts) > 0, "agent_b facts missing"

    for f in agent_a_facts:
        assert f["agent_id"] == "agent_a"
    for f in agent_b_facts:
        assert f["agent_id"] == "agent_b"


@pytest.mark.asyncio
async def test_concurrent_store_scope_isolation(tmp_path):
    """Scope set per-call must be respected even under concurrency."""
    server = MemoryServer(data_dir=str(tmp_path), key="test_scope")

    await asyncio.gather(
        server.store("Private note from Alice.", 1, "2024-01-01",
                     agent_id="agent_a", swarm_id="sw1", scope="agent-private"),
        server.store("Shared team decision.", 2, "2024-01-01",
                     agent_id="agent_b", swarm_id="sw1", scope="swarm-shared"),
    )

    for f in server._all_granular:
        if "Private" in f.get("fact", ""):
            assert f["scope"] == "agent-private"
        if "decision" in f.get("fact", "").lower():
            assert f["scope"] == "swarm-shared"


@pytest.mark.asyncio
async def test_store_default_falls_back_to_instance(tmp_path):
    """When agent_id/swarm_id/scope not passed, instance values are used."""
    server = MemoryServer(
        data_dir=str(tmp_path), key="test_default",
        agent_id="default_agent", scope="agent-private", swarm_id="sw_default",
    )
    await server.store("Some fact.", 1, "2024-01-01")
    assert server._all_granular[0]["agent_id"] == "default_agent"
    assert server._all_granular[0]["scope"] == "agent-private"


@pytest.mark.asyncio
async def test_no_save_restore_in_mcp_store(tmp_path, monkeypatch):
    """MCP memory_store does not mutate server instance attributes."""
    import src.mcp_server as mcp_mod
    mcp_mod.data_dir = str(tmp_path)
    mcp_mod.registry.clear()

    server = mcp_mod._get_memory("mcp_race_key")
    server.agent_id = "original_agent"

    await mcp_mod.memory_store(
        key="mcp_race_key",
        content="Test content for MCP store.",
        session_num=1,
        session_date="2024-01-01",
        agent_id="other_agent",
        swarm_id="sw1",
        scope="swarm-shared",
    )

    assert server.agent_id == "original_agent", \
        "MCP store mutated server.agent_id — race condition present"
