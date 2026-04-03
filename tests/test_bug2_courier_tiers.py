"""BUG-2: Courier only reads granular tier, misses cons + cross."""

from datetime import datetime, timezone

import pytest

from src.courier import Courier
from src.memory import MemoryServer


async def _make_server_with_tiers(tmp_path, key="tier_test"):
    server = MemoryServer(data_dir=str(tmp_path), key=key)
    now = datetime.now(timezone.utc).isoformat()
    server._all_granular = [{"fact": "gran fact", "created_at": now, "kind": "fact",
                              "agent_id": "a", "swarm_id": "sw1", "scope": "swarm-shared"}]
    server._all_cons     = [{"fact": "cons fact", "created_at": now, "kind": "fact",
                              "agent_id": "a", "swarm_id": "sw1", "scope": "swarm-shared"}]
    server._all_cross    = [{"fact": "cross fact", "created_at": now, "kind": "fact",
                              "agent_id": "a", "swarm_id": "sw1", "scope": "swarm-shared"}]
    server._save_cache()
    return server


@pytest.mark.asyncio
async def test_courier_delivers_cons_and_cross_facts(tmp_path):
    """Courier must deliver facts from all three tiers, not only granular."""
    server = await _make_server_with_tiers(tmp_path)
    courier = Courier(server)
    courier._last_seen_at = "1970-01-01T00:00:00+00:00"

    delivered = []
    async def cb(fact): delivered.append(fact["fact"])

    await courier.subscribe({}, cb, deliver_existing=True)

    assert "gran fact" in delivered
    assert "cons fact" in delivered, "Courier missed cons tier"
    assert "cross fact" in delivered, "Courier missed cross tier"


@pytest.mark.asyncio
async def test_courier_poll_includes_all_tiers(tmp_path):
    """_poll() after new cons/cross facts must deliver them."""
    server = MemoryServer(data_dir=str(tmp_path), key="poll_tiers")
    courier = Courier(server)
    courier._last_seen_at = "1970-01-01T00:00:00+00:00"

    delivered = []
    async def cb(fact): delivered.append(fact.get("fact", ""))

    await courier.subscribe({}, cb)

    now = datetime.now(timezone.utc).isoformat()
    server._all_cons = [{"fact": "new cons", "created_at": now,
                          "agent_id": "a", "swarm_id": "sw1", "scope": "swarm-shared"}]
    server._save_cache()

    await courier._poll()
    assert "new cons" in delivered, "Courier._poll did not deliver cons fact"
