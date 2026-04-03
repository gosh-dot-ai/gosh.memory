"""Tests for Unit 2 — upsert_by_key in memory_store."""

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
            {"id": f"f{sn}_{i}", "fact": f"{text[:60]} (fact {i})", "kind": "fact",
             "entities": ["User"], "tags": [], "session": sn}
            for i in range(3)
        ]
        return ("conv", sn, "2024-01-01", facts, [])

    async def mock_consolidate(**kwargs):
        return ("conv", 1, "2024-01-01", [])

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
async def test_upsert_creates_new_when_not_found(tmp_path):
    server = MemoryServer(data_dir=str(tmp_path), key="ups_new")
    result = await server.store(
        "User has 5 cats.", 1, "2024-01-01",
        agent_id="a", swarm_id="sw", scope="agent-private",
        upsert_by_key="agent_status",
    )
    assert result["upserted"] is False
    assert result["session_key"] == "agent_status"
    assert len(server._raw_sessions) == 1
    assert server._raw_sessions[0]["session_key"] == "agent_status"


@pytest.mark.asyncio
async def test_upsert_replaces_existing_raw_session(tmp_path):
    server = MemoryServer(data_dir=str(tmp_path), key="ups_replace")
    await server.store(
        "Phase: planning. Iteration: 1.", 1, "2024-01-01",
        agent_id="a", swarm_id="sw", scope="agent-private",
        upsert_by_key="agent_status",
    )
    result = await server.store(
        "Phase: execution. Iteration: 4.", 2, "2024-01-02",
        agent_id="a", swarm_id="sw", scope="agent-private",
        upsert_by_key="agent_status",
    )
    assert result["upserted"] is True
    # Still one raw session
    assert len(server._raw_sessions) == 1
    assert "execution" in server._raw_sessions[0]["content"]
    assert "planning" not in server._raw_sessions[0]["content"]


@pytest.mark.asyncio
async def test_upsert_removes_old_facts(tmp_path):
    server = MemoryServer(data_dir=str(tmp_path), key="ups_facts")
    await server.store(
        "Agent is in planning phase.", 1, "2024-01-01",
        agent_id="a", swarm_id="sw", scope="agent-private",
        upsert_by_key="status",
    )
    facts_after_first = len(server._all_granular)
    assert facts_after_first > 0

    await server.store(
        "Agent is in execution phase.", 1, "2024-01-02",
        agent_id="a", swarm_id="sw", scope="agent-private",
        upsert_by_key="status",
    )
    # Old facts removed, new extracted
    for f in server._all_granular:
        assert "planning" not in f.get("fact", "").lower()


@pytest.mark.asyncio
async def test_upsert_scope_error_on_swarm_shared(tmp_path):
    server = MemoryServer(data_dir=str(tmp_path), key="ups_scope")
    result = await server.store(
        "content", 1, "2024-01-01",
        agent_id="a", swarm_id="sw", scope="swarm-shared",
        upsert_by_key="bad_key",
    )
    assert result["code"] == "UPSERT_SCOPE_ERROR"


@pytest.mark.asyncio
async def test_upsert_different_agents_no_collision(tmp_path):
    """Two agents with same session_key must not collide."""
    server = MemoryServer(data_dir=str(tmp_path), key="ups_isolation")
    await server.store(
        "Agent A state.", 1, "2024-01-01",
        agent_id="agent_a", swarm_id="sw", scope="agent-private",
        upsert_by_key="status",
    )
    await server.store(
        "Agent B state.", 1, "2024-01-01",
        agent_id="agent_b", swarm_id="sw", scope="agent-private",
        upsert_by_key="status",
    )
    assert len(server._raw_sessions) == 2


@pytest.mark.asyncio
async def test_upsert_persists_across_cache_reload(tmp_path):
    server = MemoryServer(data_dir=str(tmp_path), key="ups_cache")
    await server.store(
        "Phase: execution.", 1, "2024-01-01",
        agent_id="a", swarm_id="sw", scope="agent-private",
        upsert_by_key="status",
    )
    server2 = MemoryServer(data_dir=str(tmp_path), key="ups_cache")
    assert server2._raw_sessions[0]["session_key"] == "status"
    assert "execution" in server2._raw_sessions[0]["content"]


@pytest.mark.asyncio
async def test_normal_store_still_returns_int_compat(tmp_path):
    """Normal store (no upsert) must return dict with facts_extracted."""
    server = MemoryServer(data_dir=str(tmp_path), key="ups_compat")
    result = await server.store(
        "User likes hiking.", 1, "2024-01-01",
        agent_id="a", swarm_id="sw", scope="agent-private",
    )
    # After refactor: store returns dict
    assert result["facts_extracted"] >= 1
    assert "upserted" not in result
