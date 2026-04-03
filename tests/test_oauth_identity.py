"""MCP identity resolution — post-placeholder contract.

Tests the _resolve_identity() surface after removal of _TOKEN_IDENTITIES.

NOTE: agent_key is an UNVERIFIED caller-supplied header. It is accepted
as identity only as a transitional measure until real MCP OAuth is wired
via _verified_auth_resolver (R35). Tests that use agent_key are marked
[TRANSITIONAL].
"""

import pytest

import src.mcp_server as mcp_mod
from src.mcp_server import ConnectionContext, _resolve_identity

# ── Fallback behavior (no verified auth) ──

def test_no_token_param_derived():
    """No token → param-derived identity."""
    ctx = _resolve_identity(agent_id="charlie")
    assert ctx.owner_id == "agent:charlie"


def test_default_still_works():
    """No token, no agent_key, default agent_id → system."""
    ctx = _resolve_identity()
    assert ctx.owner_id == "system"


def test_unknown_token_blocks_fallback():
    """Unknown token blocks agent_id fallback — any token is an auth attempt."""
    ctx = _resolve_identity(token="unknown-token", agent_id="eve")
    assert ctx.owner_id == "system"


# ── Transitional: agent_key as unverified identity carrier ──

def test_agent_key_sets_identity_transitional():
    """[TRANSITIONAL] agent_key header sets owner_id directly.

    WARNING: agent_key is caller-supplied and unverified. This behavior
    exists only until MCP OAuth is implemented via _verified_auth_resolver.
    """
    ctx = _resolve_identity(agent_key="agent:dave")
    assert ctx.owner_id == "agent:dave"


# ── Precedence: token > agent_key > agent_id ──

def test_token_wins_over_agent_key():
    """When a token is present (auth attempt), agent_key is ignored."""
    ctx = _resolve_identity(
        token="some-token",
        agent_key="agent:evil",
        agent_id="also-evil",
    )
    assert ctx.owner_id == "system"  # unknown token → system
    assert ctx.owner_id != "agent:evil"


def test_agent_key_wins_over_agent_id_transitional():
    """[TRANSITIONAL] agent_key takes precedence over agent_id."""
    ctx = _resolve_identity(agent_key="agent:real", agent_id="fake")
    assert ctx.owner_id == "agent:real"


# ── Canonical identity format ──

def test_identity_is_canonical():
    """Resolved owner_id must be in canonical prefix:name format."""
    ctx = _resolve_identity(agent_id="alice")
    assert ":" in ctx.owner_id
    assert ctx.owner_id.startswith("agent:")


def test_agent_key_passthrough_transitional():
    """[TRANSITIONAL] agent_key value passes through as owner_id."""
    ctx = _resolve_identity(agent_key="user:alice")
    assert ctx.owner_id == "user:alice"

    ctx2 = _resolve_identity(agent_key="agent:bot")
    assert ctx2.owner_id == "agent:bot"


# ── Verified auth adapter (R35) ──

def test_verified_auth_resolver_takes_precedence(monkeypatch):
    """When _verified_auth_resolver is set, it resolves token → owner_id."""
    monkeypatch.setattr(mcp_mod, "_verified_auth_resolver",
                        lambda t: "user:alice" if t == "valid-token" else None)

    ctx = mcp_mod._resolve_identity(token="valid-token", agent_key="agent:evil")
    assert ctx.owner_id == "user:alice"
    assert ctx.owner_id != "agent:evil"


def test_verified_auth_resolver_unknown_token_falls_to_system(monkeypatch):
    """Resolver returns None for unknown token → owner_id=system."""
    monkeypatch.setattr(mcp_mod, "_verified_auth_resolver", lambda t: None)

    ctx = mcp_mod._resolve_identity(token="bad-token", agent_key="agent:evil")
    assert ctx.owner_id == "system"
    assert ctx.owner_id != "agent:evil"
