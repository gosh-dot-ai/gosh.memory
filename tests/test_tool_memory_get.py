"""Unit 5: memory_get tool."""

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


def _seed(tmp_path, key="get_test"):
    server = mcp_mod._get_memory(key)
    now = datetime.now(timezone.utc).isoformat()
    server._all_granular = [
        {"id": "fact_001", "fact": "shared fact", "kind": "fact", "created_at": now,
         "agent_id": "a", "swarm_id": "sw1", "scope": "swarm-shared"},
        {"id": "fact_002", "fact": "private fact", "kind": "fact", "created_at": now,
         "agent_id": "agent_x", "swarm_id": "sw1", "scope": "agent-private"},
    ]
    server._save_cache()
    return server


@pytest.mark.asyncio
async def test_memory_get_returns_fact(tmp_path):
    _seed(tmp_path)
    result = await mcp_mod.memory_get(key="get_test", fact_id="fact_001",
                                       agent_id="b", swarm_id="sw1")
    assert "fact" in result
    assert result["fact"]["id"] == "fact_001"


@pytest.mark.asyncio
async def test_memory_get_not_found(tmp_path):
    _seed(tmp_path)
    result = await mcp_mod.memory_get(key="get_test", fact_id="nonexistent",
                                       agent_id="a", swarm_id="sw1")
    assert result.get("code") == "NOT_FOUND"


@pytest.mark.asyncio
async def test_memory_get_scope_forbidden(tmp_path):
    """Agent B cannot fetch agent_x's private fact by ID."""
    _seed(tmp_path)
    result = await mcp_mod.memory_get(key="get_test", fact_id="fact_002",
                                       agent_id="agent_b", swarm_id="sw1")
    assert result.get("code") == "ACL_FORBIDDEN"


@pytest.mark.asyncio
async def test_memory_get_owner_can_fetch_private(tmp_path):
    """Owner can fetch their own private fact."""
    _seed(tmp_path)
    result = await mcp_mod.memory_get(key="get_test", fact_id="fact_002",
                                       agent_id="agent_x", swarm_id="sw1")
    assert "fact" in result
    assert result["fact"]["fact"] == "private fact"


@pytest.mark.asyncio
async def test_memory_get_searches_all_tiers(tmp_path):
    """memory_get finds facts in cons and cross tiers, not only granular."""
    server = mcp_mod._get_memory("tiers_get")
    now = datetime.now(timezone.utc).isoformat()
    server._all_cons = [{"id": "cons_001", "fact": "consolidated", "kind": "fact",
                          "created_at": now, "agent_id": "a", "swarm_id": "sw1",
                          "scope": "swarm-shared"}]
    server._save_cache()

    result = await mcp_mod.memory_get(key="tiers_get", fact_id="cons_001",
                                       agent_id="a", swarm_id="sw1")
    assert "fact" in result, "memory_get did not find cons-tier fact"
