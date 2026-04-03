"""Unit 5: Membership registry."""

import pytest

from src.membership import MembershipRegistry


def test_register_and_lookup():
    reg = MembershipRegistry()
    reg.register("agent:alice", "swarm:alpha")
    assert "swarm:alpha" in reg.memberships_for("agent:alice")


def test_unregister():
    reg = MembershipRegistry()
    reg.register("agent:alice", "swarm:alpha")
    reg.unregister("agent:alice", "swarm:alpha")
    assert "swarm:alpha" not in reg.memberships_for("agent:alice")


def test_different_prefix_no_match():
    """agent:alice != user:alice — canonical IDs."""
    reg = MembershipRegistry()
    reg.register("agent:alice", "swarm:alpha")
    assert reg.memberships_for("user:alice") == []


def test_multiple_memberships():
    reg = MembershipRegistry()
    reg.register("agent:alice", "swarm:alpha")
    reg.register("agent:alice", "swarm:beta")
    ms = reg.memberships_for("agent:alice")
    assert "swarm:alpha" in ms
    assert "swarm:beta" in ms


def test_unregister_nonexistent():
    """Unregistering a membership that doesn't exist is a no-op."""
    reg = MembershipRegistry()
    reg.unregister("agent:alice", "swarm:alpha")  # no error
    assert reg.memberships_for("agent:alice") == []


def test_register_idempotent():
    """Registering same membership twice doesn't duplicate."""
    reg = MembershipRegistry()
    reg.register("agent:alice", "swarm:alpha")
    reg.register("agent:alice", "swarm:alpha")
    assert reg.memberships_for("agent:alice").count("swarm:alpha") == 1
