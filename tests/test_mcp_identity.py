"""Unit 2: Auth identity context in MCP server."""

import pytest

from src.mcp_server import ConnectionContext, _resolve_identity


def test_connection_context_defaults():
    ctx = ConnectionContext()
    assert ctx.owner_id == "system"
    assert ctx.agent_id == "default"
    assert ctx.caller_role == "user"


def test_resolve_identity_agent_alice():
    """agent_id='alice' → owner_id='agent:alice'."""
    ctx = _resolve_identity(agent_id="alice")
    assert ctx.owner_id == "agent:alice"
    assert ctx.agent_id == "alice"


def test_resolve_identity_default():
    """agent_id='default' → owner_id='system'."""
    ctx = _resolve_identity(agent_id="default")
    assert ctx.owner_id == "system"
    assert ctx.agent_id == "default"


def test_resolve_identity_no_agent():
    """No agent_id → defaults."""
    ctx = _resolve_identity()
    assert ctx.owner_id == "system"
    assert ctx.agent_id == "default"


def test_resolve_identity_swarm():
    """swarm_id passed through."""
    ctx = _resolve_identity(agent_id="bob", swarm_id="alpha")
    assert ctx.owner_id == "agent:bob"
    assert ctx.swarm_id == "alpha"
