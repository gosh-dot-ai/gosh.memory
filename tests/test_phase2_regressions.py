"""Regression tests for Phase 2 review bugs.

Bug 1: OAuth/API-key identity not wired into MCP tools
Bug 2: memory_list/memory_get ignore caller_role (admin bypass broken)
Bug 3: courier_subscribe doesn't get real subscriber identity

NOTE: Tests in Bug 1 use agent_key as a TRANSITIONAL unverified identity
carrier. When MCP OAuth (R35) is implemented via _verified_auth_resolver,
these must be updated to use the verified auth path.
"""

import asyncio
import json

import pytest

from src import mcp_server
from src.memory import MemoryServer


def _patch_all(monkeypatch):
    async def mock_extract(**kwargs):
        sn = kwargs.get("session_num", 1)
        return ("conv", sn, "2024-06-01", [
            {"id": f"f{sn}_0", "fact": f"Fact {sn}", "kind": "event",
             "entities": [], "tags": [], "session": sn}], [])

    async def mock_consolidate(**kwargs):
        return ("conv", 1, "2024-06-01", [
            {"id": "c0", "fact": "Cons", "kind": "summary", "entities": [], "tags": []}])

    async def mock_cross(**kwargs):
        return ("conv", "e", [
            {"id": "x0", "fact": "Cross", "kind": "profile", "entities": [], "tags": []}])

    async def mock_embed(texts, **kw):
        import numpy as np
        return np.random.randn(len(texts), 3072).astype(np.float32)

    async def mock_embed_q(text, **kw):
        import numpy as np
        return np.random.randn(3072).astype(np.float32)

    monkeypatch.setattr("src.memory.extract_session", mock_extract)
    monkeypatch.setattr("src.memory.consolidate_session", mock_consolidate)
    monkeypatch.setattr("src.memory.cross_session_entity", mock_cross)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda f, l: None)
    monkeypatch.setattr("src.memory.embed_texts", mock_embed)
    monkeypatch.setattr("src.memory.embed_query", mock_embed_q)


# ── Bug 1: Token/agent_key identity threading ──

def test_memory_recall_with_identity(tmp_path, monkeypatch):
    """memory_recall with caller identity resolves via agent_key."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "tok_test")
    asyncio.run(ms.store("Hello", session_num=1, session_date="2024-06-01",
                         owner_id="user:alice", read=[], write=[]))
    asyncio.run(ms.build_index())

    monkeypatch.setitem(mcp_server.registry, "tok_test", ms)

    # Without identity → system caller can't see owner-only fact
    result = asyncio.run(mcp_server.memory_recall(key="tok_test", query="test"))
    assert result["retrieved_count"] == 0

    # With agent_key for alice → sees her fact
    result = asyncio.run(mcp_server.memory_recall(
        key="tok_test", query="test", agent_key="user:alice"))
    assert result["retrieved_count"] > 0


def test_memory_list_with_identity(tmp_path, monkeypatch):
    """memory_list with caller identity resolves via agent_key."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "list_tok")
    asyncio.run(ms.store("Hello", session_num=1, session_date="2024-06-01",
                         owner_id="user:alice", read=[], write=[]))

    monkeypatch.setitem(mcp_server.registry, "list_tok", ms)

    # No identity → 0 visible
    result = asyncio.run(mcp_server.memory_list(key="list_tok"))
    assert result["total"] == 0

    # agent_key for alice → sees her facts
    result = asyncio.run(mcp_server.memory_list(
        key="list_tok", agent_key="user:alice"))
    assert result["total"] > 0


def test_memory_get_with_identity(tmp_path, monkeypatch):
    """memory_get with caller identity resolves via agent_key."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "get_tok")
    asyncio.run(ms.store("Hello", session_num=1, session_date="2024-06-01",
                         owner_id="user:alice", read=[], write=[]))

    monkeypatch.setitem(mcp_server.registry, "get_tok", ms)
    fid = ms._all_granular[0]["id"]

    # No identity → denied
    result = asyncio.run(mcp_server.memory_get(key="get_tok", fact_id=fid))
    assert result.get("code") == "ACL_FORBIDDEN"

    # agent_key for alice → gets fact
    result = asyncio.run(mcp_server.memory_get(
        key="get_tok", fact_id=fid, agent_key="user:alice"))
    assert "fact" in result


# ── Bug 2: Admin bypass in memory_list/memory_get ──

def test_memory_list_admin_bypass(tmp_path, monkeypatch):
    """Admin token → sees all facts regardless of ACL."""
    _patch_all(monkeypatch)
    monkeypatch.setattr(mcp_server, "ADMIN_TOKEN", "admin-secret")

    ms = MemoryServer(str(tmp_path), "admin_list")
    asyncio.run(ms.store("Secret", session_num=1, session_date="2024-06-01",
                         owner_id="user:alice", read=[], write=[]))

    monkeypatch.setitem(mcp_server.registry, "admin_list", ms)

    # Non-admin → 0
    result = asyncio.run(mcp_server.memory_list(key="admin_list"))
    assert result["total"] == 0

    # Admin token → sees all
    result = asyncio.run(mcp_server.memory_list(
        key="admin_list", token="admin-secret"))
    assert result["total"] > 0


def test_memory_get_admin_bypass(tmp_path, monkeypatch):
    """Admin token → gets any fact."""
    _patch_all(monkeypatch)
    monkeypatch.setattr(mcp_server, "ADMIN_TOKEN", "admin-secret")

    ms = MemoryServer(str(tmp_path), "admin_get")
    asyncio.run(ms.store("Secret", session_num=1, session_date="2024-06-01",
                         owner_id="user:alice", read=[], write=[]))

    monkeypatch.setitem(mcp_server.registry, "admin_get", ms)
    fid = ms._all_granular[0]["id"]

    result = asyncio.run(mcp_server.memory_get(
        key="admin_get", fact_id=fid, token="admin-secret"))
    assert "fact" in result


# ── Bug 3: Courier subscriber identity ──

def test_courier_subscribe_with_membership(tmp_path, monkeypatch):
    """courier_subscribe with agent_id → subscriber sees swarm facts via Courier directly."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "cour_mem")
    ms._membership_registry.register("agent:bob", "alpha")

    # Store a swarm-only fact
    asyncio.run(ms.store("Swarm data", session_num=1, session_date="2024-06-01",
                         owner_id="agent:owner", read=["swarm:alpha"], write=[]))

    # Test Courier directly (avoid MCP event loop issues)
    from src.courier import Courier
    courier = Courier(ms)
    delivered = []

    async def _test():
        async def cb(fact):
            delivered.append(fact)

        await courier.subscribe(
            filter={}, callback=cb, deliver_existing=True,
            owner_id="agent:bob", memberships=["swarm:alpha"])
        return len(delivered)

    count = asyncio.run(_test())
    assert count > 0, "Swarm fact should be delivered to member via deliver_existing"

    # Also verify non-member doesn't get it
    delivered2 = []

    async def _test2():
        async def cb2(fact):
            delivered2.append(fact)

        await courier.subscribe(
            filter={}, callback=cb2, deliver_existing=True,
            owner_id="agent:stranger", memberships=[])
        return len(delivered2)

    count2 = asyncio.run(_test2())
    assert count2 == 0, "Non-member should NOT receive swarm-only fact"


def test_courier_admin_sees_owner_only(tmp_path, monkeypatch):
    """Admin subscriber receives owner-only facts via deliver_existing."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "cour_admin")
    asyncio.run(ms.store("Secret", session_num=1, session_date="2024-06-01",
                         owner_id="user:alice", read=[], write=[]))

    from src.courier import Courier
    courier = Courier(ms)

    # Non-admin → 0
    non_admin = []
    async def _test_nonadmin():
        async def cb(f): non_admin.append(f)
        await courier.subscribe(filter={}, callback=cb, deliver_existing=True,
                                owner_id="agent:stranger", caller_role="user")
    asyncio.run(_test_nonadmin())
    assert len(non_admin) == 0, "Non-admin should NOT see owner-only fact"

    # Admin → sees it
    admin_facts = []
    async def _test_admin():
        async def cb(f): admin_facts.append(f)
        await courier.subscribe(filter={}, callback=cb, deliver_existing=True,
                                owner_id="system", caller_role="admin")
    asyncio.run(_test_admin())
    assert len(admin_facts) > 0, "Admin should see owner-only fact via Courier"
