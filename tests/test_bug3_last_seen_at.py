"""BUG-3: Courier _last_seen_at initialized from wall clock, not DB high-water mark."""

from datetime import datetime, timedelta, timezone

import pytest

from src.courier import Courier
from src.memory import MemoryServer


def _make_server_with_old_facts(tmp_path, key="last_seen"):
    server = MemoryServer(data_dir=str(tmp_path), key=key)
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    server._all_granular = [
        {"fact": "old fact A", "created_at": old_ts,
         "agent_id": "a", "swarm_id": "sw1", "scope": "swarm-shared"},
        {"fact": "old fact B", "created_at": old_ts,
         "agent_id": "a", "swarm_id": "sw1", "scope": "swarm-shared"},
    ]
    server._save_cache()
    return server, old_ts


def test_courier_last_seen_initialized_from_existing_facts(tmp_path):
    """Courier._last_seen_at must be >= max(created_at) of existing facts."""
    server, old_ts = _make_server_with_old_facts(tmp_path)
    courier = Courier(server)
    assert courier._last_seen_at >= old_ts, (
        f"_last_seen_at={courier._last_seen_at} < existing max={old_ts}"
    )


@pytest.mark.asyncio
async def test_courier_does_not_redeliver_existing_facts_on_subscribe(tmp_path):
    """After restart, existing facts are NOT re-delivered to deliver_existing=False subscribers."""
    server, _ = _make_server_with_old_facts(tmp_path)
    courier = Courier(server)

    delivered = []
    async def cb(fact): delivered.append(fact["fact"])

    await courier.subscribe({}, cb, deliver_existing=False)
    await courier._poll()

    assert delivered == [], f"Old facts were re-delivered after restart: {delivered}"


@pytest.mark.asyncio
async def test_courier_epoch_when_no_existing_facts(tmp_path):
    """When no facts exist, _last_seen_at is epoch (not wall clock)."""
    server = MemoryServer(data_dir=str(tmp_path), key="empty")
    courier = Courier(server)
    assert courier._last_seen_at == "1970-01-01T00:00:00+00:00"


@pytest.mark.asyncio
async def test_new_facts_after_restart_delivered(tmp_path):
    """Facts written AFTER Courier init are delivered."""
    server, _ = _make_server_with_old_facts(tmp_path)
    courier = Courier(server)

    delivered = []
    async def cb(fact): delivered.append(fact["fact"])
    await courier.subscribe({}, cb, deliver_existing=False)

    new_ts = datetime.now(timezone.utc).isoformat()
    server._all_granular.append({
        "fact": "brand new fact", "created_at": new_ts,
        "agent_id": "a", "swarm_id": "sw1", "scope": "swarm-shared"
    })
    server._save_cache()

    await courier._poll()
    assert "brand new fact" in delivered
