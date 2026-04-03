"""Unit 4: memory_list tool."""

from datetime import datetime, timezone

import pytest

import src.mcp_server as mcp_mod


@pytest.fixture(autouse=True)
def _reset(tmp_path):
    mcp_mod.data_dir = str(tmp_path)
    mcp_mod.registry.clear()
    mcp_mod.courier_registry.clear()
    mcp_mod.connections.clear()
    mcp_mod.sub_to_conn.clear()


def _seed_server(tmp_path, key="list_test"):
    server = mcp_mod._get_memory(key)
    now = datetime.now(timezone.utc).isoformat()
    server._all_granular = [
        {"fact": "task A", "kind": "task", "created_at": now,
         "agent_id": "orch", "swarm_id": "sw1", "scope": "swarm-shared"},
        {"fact": "analysis B", "kind": "analysis", "created_at": now,
         "agent_id": "analyst", "swarm_id": "sw1", "scope": "swarm-shared"},
        {"fact": "private C", "kind": "fact", "created_at": now,
         "agent_id": "agent_x", "swarm_id": "sw1", "scope": "agent-private"},
    ]
    server._save_cache()
    return server


@pytest.mark.asyncio
async def test_memory_list_returns_all_visible(tmp_path):
    _seed_server(tmp_path)
    result = await mcp_mod.memory_list(
        key="list_test", agent_id="orch", swarm_id="sw1"
    )
    assert result["total"] == 2  # agent-private not visible to orch
    kinds = {f["kind"] for f in result["facts"]}
    assert "task" in kinds
    assert "analysis" in kinds


@pytest.mark.asyncio
async def test_memory_list_filter_by_kind(tmp_path):
    _seed_server(tmp_path)
    result = await mcp_mod.memory_list(
        key="list_test", agent_id="orch", swarm_id="sw1", kind="task"
    )
    assert result["total"] == 1
    assert result["facts"][0]["kind"] == "task"


@pytest.mark.asyncio
async def test_memory_list_agent_private_visible_to_owner(tmp_path):
    _seed_server(tmp_path)
    result = await mcp_mod.memory_list(
        key="list_test", agent_id="agent_x", swarm_id="sw1"
    )
    facts_by_kind = {f["kind"] for f in result["facts"]}
    assert "fact" in facts_by_kind


@pytest.mark.asyncio
async def test_memory_list_pagination(tmp_path):
    server = mcp_mod._get_memory("page_test")
    now = datetime.now(timezone.utc).isoformat()
    server._all_granular = [
        {"fact": f"fact {i}", "kind": "fact", "created_at": now,
         "agent_id": "a", "swarm_id": "sw1", "scope": "swarm-shared"}
        for i in range(10)
    ]
    server._save_cache()

    page1 = await mcp_mod.memory_list(key="page_test", agent_id="a", swarm_id="sw1",
                                       limit=3, offset=0)
    page2 = await mcp_mod.memory_list(key="page_test", agent_id="a", swarm_id="sw1",
                                       limit=3, offset=3)

    assert len(page1["facts"]) == 3
    assert len(page2["facts"]) == 3
    assert page1["total"] == 10
    assert {f["fact"] for f in page1["facts"]}.isdisjoint(
        {f["fact"] for f in page2["facts"]})


@pytest.mark.asyncio
async def test_memory_list_cross_swarm_isolation(tmp_path):
    """Agent from sw2 cannot see sw1 swarm-shared facts."""
    server = mcp_mod._get_memory("iso_test")
    now = datetime.now(timezone.utc).isoformat()
    server._all_granular = [
        {"fact": "sw1 secret", "kind": "fact", "created_at": now,
         "agent_id": "a", "swarm_id": "sw1", "scope": "swarm-shared"},
    ]
    server._save_cache()

    result = await mcp_mod.memory_list(key="iso_test", agent_id="b", swarm_id="sw2")
    assert result["total"] == 0
