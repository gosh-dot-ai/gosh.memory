"""Unit 1: Canonical identity format + ACL data model."""

import asyncio
import json

import pytest

from src.memory import MemoryServer, _derive_acl_from_scope, _normalize_identity

# ── Identity normalization ──

def test_normalize_valid():
    assert _normalize_identity("user:mitja") == "user:mitja"
    assert _normalize_identity("agent:default") == "agent:default"
    assert _normalize_identity("swarm:alpha") == "swarm:alpha"
    assert _normalize_identity("system") == "system"
    assert _normalize_identity("anonymous") == "anonymous"


def test_normalize_no_prefix_raises():
    with pytest.raises(ValueError):
        _normalize_identity("mitja")


def test_normalize_bad_prefix_raises():
    with pytest.raises(ValueError):
        _normalize_identity("admin:root")


def test_normalize_public_as_owner_raises():
    """agent:PUBLIC is valid in ACL lists but NOT as owner_id."""
    with pytest.raises(ValueError):
        _normalize_identity("agent:PUBLIC", allow_public=False)


def test_normalize_public_in_acl():
    """agent:PUBLIC is allowed in ACL context."""
    assert _normalize_identity("agent:PUBLIC", allow_public=True) == "agent:PUBLIC"


# ── ACL derivation from legacy scope ──

def test_derive_agent_private():
    acl = _derive_acl_from_scope("agent-private", "alice", "default")
    assert acl["owner_id"] == "agent:alice"
    assert acl["read"] == []
    assert acl["write"] == []


def test_derive_swarm_shared_default():
    """swarm_id='default' → public (benchmark compat)."""
    acl = _derive_acl_from_scope("swarm-shared", "default", "default")
    assert acl["owner_id"] == "system"
    assert acl["read"] == ["agent:PUBLIC"]
    assert acl["write"] == ["agent:PUBLIC"]


def test_derive_swarm_shared_named():
    acl = _derive_acl_from_scope("swarm-shared", "bob", "alpha")
    assert acl["owner_id"] == "agent:bob"
    assert acl["read"] == ["swarm:alpha"]
    assert acl["write"] == ["swarm:alpha"]


def test_derive_system_wide():
    acl = _derive_acl_from_scope("system-wide", "x", "y")
    assert acl["owner_id"] == "system"
    assert acl["read"] == ["agent:PUBLIC"]
    assert acl["write"] == ["agent:PUBLIC"]


def test_derive_no_scope():
    acl = _derive_acl_from_scope(None, "x", "y")
    assert acl["owner_id"] == "system"
    assert acl["read"] == ["agent:PUBLIC"]


# ── ACL on sessions and facts ──

def _patch_all(monkeypatch, **overrides):
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


def test_store_with_acl(tmp_path, monkeypatch):
    """Explicit ACL on store → present on session and facts."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv1")
    asyncio.run(ms.store("Hello", session_num=1, session_date="2024-06-01",
                         owner_id="user:mitja", read=["swarm:alpha"], write=[]))

    rs = ms._raw_sessions[0]
    assert rs["owner_id"] == "user:mitja"
    assert rs["read"] == ["swarm:alpha"]
    assert rs["write"] == []

    fact = ms._all_granular[0]
    assert fact["owner_id"] == "user:mitja"
    assert fact["read"] == ["swarm:alpha"]


def test_store_default_acl(tmp_path, monkeypatch):
    """No ACL → defaults to system/public."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv2")
    asyncio.run(ms.store("Hello", session_num=1, session_date="2024-06-01"))

    fact = ms._all_granular[0]
    assert fact["owner_id"] == "system"
    assert fact["read"] == ["agent:PUBLIC"]


def test_store_agent_private_derives_owner(tmp_path, monkeypatch):
    """agent-private store derives owner/read/write from scope."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv3", agent_id="alice")
    asyncio.run(
        ms.store(
            "Hello",
            session_num=1,
            session_date="2024-06-01",
            scope="agent-private",
        )
    )

    fact = ms._all_granular[0]
    assert fact["owner_id"] == "agent:alice"
    assert fact["read"] == []
    assert fact["write"] == []
    assert ms._acl_allows(fact, "agent:alice", [], "user")
    assert not ms._acl_allows(fact, "agent:bob", [], "user")


def test_store_named_swarm_derives_swarm_acl(tmp_path, monkeypatch):
    """swarm-shared + named swarm derives swarm-scoped ACL."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv3b", agent_id="alice", swarm_id="alpha")
    asyncio.run(ms.store("Hello", session_num=1, session_date="2024-06-01"))

    fact = ms._all_granular[0]
    assert fact["owner_id"] == "agent:alice"
    assert fact["read"] == ["swarm:alpha"]
    assert fact["write"] == ["swarm:alpha"]


def test_backward_compat_load(tmp_path, monkeypatch):
    """Old cache without ACL fields → migration applies correct defaults."""
    _patch_all(monkeypatch)
    # Create old-style cache
    cache = {
        "granular": [
            {"id": "f1", "fact": "Old fact", "kind": "event", "entities": [],
             "tags": [], "session": 1, "scope": "swarm-shared",
             "agent_id": "default", "swarm_id": "default", "conv_id": "conv4"},
        ],
        "cons": [], "cross": [], "tlinks": [],
        "raw_sessions": [], "secrets": [],
        "n_sessions": 1, "n_sessions_with_facts": 1,
    }
    (tmp_path / "conv4.json").write_text(json.dumps(cache))

    ms = MemoryServer(str(tmp_path), "conv4")
    fact = ms._all_granular[0]
    # swarm-shared + swarm_id=default → public
    assert fact.get("owner_id") == "system"
    assert fact.get("read") == ["agent:PUBLIC"]
    assert fact.get("write") == ["agent:PUBLIC"]


def test_backward_compat_private(tmp_path, monkeypatch):
    """Old agent-private fact → owner=agent:{agent_id}, no read."""
    _patch_all(monkeypatch)
    cache = {
        "granular": [
            {"id": "f1", "fact": "Private", "kind": "event", "entities": [],
             "tags": [], "session": 1, "scope": "agent-private",
             "agent_id": "bob", "swarm_id": "default", "conv_id": "conv5"},
        ],
        "cons": [], "cross": [], "tlinks": [],
        "raw_sessions": [], "secrets": [],
        "n_sessions": 1, "n_sessions_with_facts": 1,
    }
    (tmp_path / "conv5.json").write_text(json.dumps(cache))

    ms = MemoryServer(str(tmp_path), "conv5")
    fact = ms._all_granular[0]
    assert fact.get("owner_id") == "agent:bob"
    assert fact.get("read") == []
    assert fact.get("write") == []
