"""Unit 3: ACL enforcement in all retrieval paths."""

import asyncio
import json

import numpy as np
import pytest

from src.memory import MemoryServer

DIM = 3072


def _patch_all(monkeypatch):
    async def mock_extract(**kwargs):
        sn = kwargs.get("session_num", 1)
        return ("conv", sn, "2024-06-01", [
            {"id": f"f{sn}_0", "fact": f"Fact {sn}", "kind": "event",
             "entities": [], "tags": [], "session": sn}], [])

    async def mock_consolidate(**kwargs):
        return ("conv", 1, "2024-06-01", [
            {"id": "c0", "fact": "Cons", "kind": "summary",
             "entities": [], "tags": []}])

    async def mock_cross(**kwargs):
        return ("conv", "e", [
            {"id": "x0", "fact": "Cross", "kind": "profile",
             "entities": [], "tags": []}])

    async def mock_embed(texts, **kw):
        return np.random.randn(len(texts), DIM).astype(np.float32)

    async def mock_embed_q(text, **kw):
        return np.random.randn(DIM).astype(np.float32)

    monkeypatch.setattr("src.memory.extract_session", mock_extract)
    monkeypatch.setattr("src.memory.consolidate_session", mock_consolidate)
    monkeypatch.setattr("src.memory.cross_session_entity", mock_cross)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda f, l: None)
    monkeypatch.setattr("src.memory.embed_texts", mock_embed)
    monkeypatch.setattr("src.memory.embed_query", mock_embed_q)


def _make_fact(fid, owner_id="system", read=None, write=None, **extra):
    f = {
        "id": fid, "fact": f"Fact {fid}", "kind": "event",
        "entities": [], "tags": [], "session": 1,
        "scope": "swarm-shared", "agent_id": "default", "swarm_id": "default",
        "conv_id": "test", "owner_id": owner_id,
        "read": read if read is not None else ["agent:PUBLIC"],
        "write": write if write is not None else ["agent:PUBLIC"],
        "created_at": "2024-01-01T00:00:00+00:00",
    }
    f.update(extra)
    return f


# ── _acl_allows tests ──

def test_acl_allows_public(tmp_path):
    ms = MemoryServer(str(tmp_path), "acl1")
    fact = _make_fact("f1", owner_id="system", read=["agent:PUBLIC"])
    assert ms._acl_allows(fact, "agent:anyone") is True


def test_acl_allows_owner(tmp_path):
    ms = MemoryServer(str(tmp_path), "acl2")
    fact = _make_fact("f1", owner_id="agent:alice", read=[])
    # Owner always has access
    assert ms._acl_allows(fact, "agent:alice") is True
    # Non-owner denied
    assert ms._acl_allows(fact, "agent:bob") is False


def test_acl_allows_swarm_member(tmp_path):
    ms = MemoryServer(str(tmp_path), "acl3")
    fact = _make_fact("f1", owner_id="agent:alice", read=["swarm:alpha"])
    # Agent in swarm:alpha → allowed
    assert ms._acl_allows(fact, "agent:bob", caller_memberships=["swarm:alpha"]) is True
    # Agent NOT in swarm:alpha → denied
    assert ms._acl_allows(fact, "agent:bob", caller_memberships=[]) is False


def test_acl_allows_system_caller(tmp_path):
    ms = MemoryServer(str(tmp_path), "acl4")
    fact = _make_fact("f1", owner_id="system", read=["agent:PUBLIC"])
    assert ms._acl_allows(fact, "system") is True


# ── 3 sessions visibility test ──

def test_three_sessions_visibility(tmp_path, monkeypatch):
    """3 sessions: owner-only, swarm-shared, public. Each caller sees correct subset."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "vis1")

    # Session 1: owner-only (alice, no read)
    asyncio.run(ms.store("Private data", session_num=1, session_date="2024-06-01",
                         owner_id="agent:alice", read=[], write=[]))
    # Session 2: swarm-shared
    asyncio.run(ms.store("Swarm data", session_num=2, session_date="2024-06-01",
                         owner_id="agent:bob", read=["swarm:alpha"], write=["swarm:alpha"]))
    # Session 3: public
    asyncio.run(ms.store("Public data", session_num=3, session_date="2024-06-01",
                         owner_id="system", read=["agent:PUBLIC"], write=["agent:PUBLIC"]))

    all_facts = ms._all_granular

    # Alice sees her own + public (2 facts)
    alice_visible = [f for f in all_facts if ms._acl_allows(f, "agent:alice")]
    assert len(alice_visible) == 2  # own + public

    # Bob in swarm:alpha sees swarm + public (2 facts)
    bob_visible = [f for f in all_facts
                   if ms._acl_allows(f, "agent:bob", caller_memberships=["swarm:alpha"])]
    assert len(bob_visible) == 2  # swarm + public

    # Random agent (no memberships) sees only public (1 fact)
    rando_visible = [f for f in all_facts if ms._acl_allows(f, "agent:rando")]
    assert len(rando_visible) == 1  # only public


# ── memory_list ACL enforcement ──

def test_memory_list_acl(tmp_path, monkeypatch):
    """memory_list uses ACL filtering."""
    _patch_all(monkeypatch)

    import src.mcp_server as mcp_mod
    mcp_mod.data_dir = str(tmp_path)
    mcp_mod.registry.clear()
    mcp_mod.courier_registry.clear()

    ms = mcp_mod._get_memory("acl_list")

    # Directly populate facts with different ACLs
    ms._all_granular = [
        _make_fact("f1", owner_id="agent:alice", read=[], conv_id="acl_list"),
        _make_fact("f2", owner_id="system", read=["agent:PUBLIC"], conv_id="acl_list"),
    ]

    # memory_list with agent_id=alice should see both (owner + public)
    result = asyncio.run(mcp_mod.memory_list(
        key="acl_list", agent_id="alice"))
    assert result["total"] == 2

    # memory_list with agent_id=bob sees only public
    result = asyncio.run(mcp_mod.memory_list(
        key="acl_list", agent_id="bob"))
    assert result["total"] == 1


# ── memory_get ACL enforcement ──

def test_memory_get_acl(tmp_path, monkeypatch):
    _patch_all(monkeypatch)

    import src.mcp_server as mcp_mod
    mcp_mod.data_dir = str(tmp_path)
    mcp_mod.registry.clear()

    ms = mcp_mod._get_memory("acl_get")
    ms._all_granular = [
        _make_fact("f1", owner_id="agent:alice", read=[], conv_id="acl_get"),
    ]

    # alice can get it
    result = asyncio.run(mcp_mod.memory_get(
        key="acl_get", fact_id="f1", agent_id="alice"))
    assert "fact" in result

    # bob cannot
    result = asyncio.run(mcp_mod.memory_get(
        key="acl_get", fact_id="f1", agent_id="bob"))
    assert result["code"] == "ACL_FORBIDDEN"
