"""Unit 7: Admin access."""

import asyncio
import os

import pytest

from src.mcp_server import ConnectionContext, _resolve_identity
from src.memory import MemoryServer


def _make_fact(fid, owner_id="agent:alice", read=None, **extra):
    f = {
        "id": fid, "fact": f"Fact {fid}", "kind": "event",
        "entities": [], "tags": [], "session": 1,
        "scope": "agent-private", "agent_id": "alice", "swarm_id": "default",
        "conv_id": "test", "owner_id": owner_id,
        "read": read if read is not None else [],
        "write": [],
        "created_at": "2024-01-01T00:00:00+00:00",
    }
    f.update(extra)
    return f


def test_admin_sees_owner_only_facts(tmp_path):
    """Admin caller_role bypasses ACL checks."""
    ms = MemoryServer(str(tmp_path), "admin1")
    fact = _make_fact("f1", owner_id="agent:alice", read=[])

    # Non-admin cannot see
    assert ms._acl_allows(fact, "agent:bob", caller_role="user") is False
    # Admin can see
    assert ms._acl_allows(fact, "agent:bob", caller_role="admin") is True


def test_non_admin_blocked(tmp_path):
    ms = MemoryServer(str(tmp_path), "admin2")
    fact = _make_fact("f1", owner_id="agent:alice", read=[])
    assert ms._acl_allows(fact, "agent:rando", caller_role="user") is False


def test_resolve_identity_admin_token(monkeypatch):
    """GOSH_MEMORY_ADMIN_TOKEN env var → admin role."""
    monkeypatch.setenv("GOSH_MEMORY_ADMIN_TOKEN", "secret-admin-token")
    # Need to reimport to pick up env var
    import importlib

    import src.mcp_server as mod
    importlib.reload(mod)

    ctx = mod._resolve_identity(token="secret-admin-token")
    assert ctx.caller_role == "admin"
    assert ctx.owner_id == "system"  # admin is system-level


def test_resolve_identity_unknown_token_no_admin():
    """Unknown token must NOT grant admin role or resolve to a user identity.

    _TOKEN_IDENTITIES is empty in production — any unrecognised token
    resolves to owner_id='system' with caller_role='user'.
    Admin role requires GOSH_MEMORY_ADMIN_TOKEN env var match.
    """
    ctx = _resolve_identity(token="test-token-admin")
    assert ctx.owner_id == "system"
    assert ctx.caller_role == "user"
