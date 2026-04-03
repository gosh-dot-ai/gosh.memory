"""Unit 10: ACL enforcement in Courier."""

import asyncio
import json

import pytest

from src.courier import Courier, SubscriptionEntry
from src.memory import MemoryServer


def _make_facts_cache(facts):
    """Build a cache dict from a list of facts."""
    return {
        "granular": facts, "cons": [], "cross": [], "tlinks": [],
        "raw_sessions": [], "secrets": [],
        "n_sessions": 1, "n_sessions_with_facts": 1,
    }


def _make_fact(fid, owner_id="system", read=None, created_at="2024-01-01T00:00:01+00:00"):
    return {
        "id": fid, "fact": f"Fact {fid}", "kind": "event",
        "entities": [], "tags": [], "session": 1,
        "scope": "swarm-shared", "agent_id": "default", "swarm_id": "default",
        "conv_id": "courier_acl", "owner_id": owner_id,
        "read": read if read is not None else ["agent:PUBLIC"],
        "write": [],
        "created_at": created_at,
    }


def test_courier_subscribe_with_acl(tmp_path):
    """Subscribe as agent in swarm:alpha → only receives swarm:alpha and public facts."""
    # Set up facts: private (alice only), swarm:alpha, public
    facts = [
        _make_fact("f_priv", owner_id="agent:alice", read=[],
                   created_at="2024-01-01T00:00:02+00:00"),
        _make_fact("f_swarm", owner_id="agent:bob", read=["swarm:alpha"],
                   created_at="2024-01-01T00:00:03+00:00"),
        _make_fact("f_pub", owner_id="system", read=["agent:PUBLIC"],
                   created_at="2024-01-01T00:00:04+00:00"),
    ]

    cache = _make_facts_cache(facts)
    (tmp_path / "courier_acl.json").write_text(json.dumps(cache))

    ms = MemoryServer(str(tmp_path), "courier_acl")
    courier = Courier(ms)

    delivered = []

    async def _run():
        async def _cb(fact):
            delivered.append(fact)

        # Subscribe as agent:charlie who is in swarm:alpha
        await courier.subscribe(
            filter={},
            callback=_cb,
            deliver_existing=True,
            owner_id="agent:charlie",
            memberships=["swarm:alpha"],
        )

    asyncio.run(_run())

    # Should receive swarm:alpha + public, but NOT alice's private
    delivered_ids = {f["id"] for f in delivered}
    assert "f_swarm" in delivered_ids, "Should see swarm:alpha fact"
    assert "f_pub" in delivered_ids, "Should see public fact"
    assert "f_priv" not in delivered_ids, "Should NOT see alice's private fact"


def test_courier_poll_acl(tmp_path):
    """_poll only delivers facts that pass ACL check for subscriber."""
    facts_initial = [
        _make_fact("f_old", owner_id="system", read=["agent:PUBLIC"],
                   created_at="2024-01-01T00:00:01+00:00"),
    ]
    cache = _make_facts_cache(facts_initial)
    (tmp_path / "courier_acl2.json").write_text(json.dumps(cache))

    ms = MemoryServer(str(tmp_path), "courier_acl2")
    courier = Courier(ms)

    delivered = []

    async def _run():
        async def _cb(fact):
            delivered.append(fact)

        # Subscribe as agent:dave with no memberships
        await courier.subscribe(
            filter={},
            callback=_cb,
            deliver_existing=False,
            owner_id="agent:dave",
            memberships=[],
        )

        # Add new facts to disk: one private (alice), one public
        new_facts = facts_initial + [
            _make_fact("f_priv2", owner_id="agent:alice", read=[],
                       created_at="2024-01-02T00:00:01+00:00"),
            _make_fact("f_pub2", owner_id="system", read=["agent:PUBLIC"],
                       created_at="2024-01-02T00:00:02+00:00"),
        ]
        cache2 = _make_facts_cache(new_facts)
        (tmp_path / "courier_acl2.json").write_text(json.dumps(cache2))

        count = await courier._poll()
        return count

    asyncio.run(_run())

    delivered_ids = {f["id"] for f in delivered}
    assert "f_pub2" in delivered_ids, "Should see new public fact"
    assert "f_priv2" not in delivered_ids, "Should NOT see alice's private fact"
