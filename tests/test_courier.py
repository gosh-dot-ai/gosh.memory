"""Tests for src/courier.py — Courier."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from src.courier import Courier
from src.memory import MemoryServer


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _iso_offset(seconds):
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def _write_facts(tmp_path, key, facts):
    """Write facts directly to cache JSON (bypass MemoryServer.store)."""
    cache = tmp_path / f"{key}.json"
    data = {
        "granular": facts,
        "cons": [],
        "cross": [],
        "tlinks": [],
        "n_sessions": 0,
        "n_sessions_with_facts": 0,
    }
    cache.write_text(json.dumps(data))


def _make_fact(id, created_at=None, **kwargs):
    f = {
        "id": id,
        "fact": f"Fact {id}",
        "kind": "event",
        "scope": "swarm-shared",
        "agent_id": "default",
        "swarm_id": "default",
        "conv_id": "test",
        "session_date": "2024-06-01",
        "created_at": created_at or _now_iso(),
    }
    f.update(kwargs)
    return f


def test_subscribe_returns_sub_id(tmp_path):
    ms = MemoryServer(str(tmp_path), "c1")
    courier = Courier(ms)
    cb = AsyncMock()

    sub_id = asyncio.run(courier.subscribe(filter={}, callback=cb))

    assert isinstance(sub_id, str)
    assert sub_id.startswith("sub_")
    assert len(sub_id) > 4


def test_unsubscribe_idempotent(tmp_path):
    ms = MemoryServer(str(tmp_path), "c2")
    courier = Courier(ms)

    # Should not raise for unknown sub_id
    asyncio.run(courier.unsubscribe("sub_nonexistent"))
    asyncio.run(courier.unsubscribe("sub_nonexistent"))


def test_poll_delivers_new_facts(tmp_path):
    ms = MemoryServer(str(tmp_path), "c3")
    courier = Courier(ms)
    cb = AsyncMock()

    asyncio.run(courier.subscribe(filter={}, callback=cb))

    # Write fact with created_at in the future (after Courier init)
    fact = _make_fact("f1", created_at=_iso_offset(1))
    _write_facts(tmp_path, "c3", [fact])

    delivered = asyncio.run(courier._poll())

    assert delivered == 1
    cb.assert_called_once()
    assert cb.call_args[0][0]["id"] == "f1"


def test_poll_skips_old_facts(tmp_path):
    ms = MemoryServer(str(tmp_path), "c4")

    # Write fact with old timestamp BEFORE creating courier
    old_fact = _make_fact("f_old", created_at="2020-01-01T00:00:00+00:00")
    _write_facts(tmp_path, "c4", [old_fact])

    courier = Courier(ms)
    cb = AsyncMock()
    asyncio.run(courier.subscribe(filter={}, callback=cb))

    delivered = asyncio.run(courier._poll())

    assert delivered == 0
    cb.assert_not_called()


def test_filter_exact_match(tmp_path):
    ms = MemoryServer(str(tmp_path), "c5")
    courier = Courier(ms)
    cb = AsyncMock()

    asyncio.run(courier.subscribe(filter={"kind": "decision"}, callback=cb))

    ts = _iso_offset(1)
    facts = [
        _make_fact("f1", created_at=ts, kind="event"),
        _make_fact("f2", created_at=ts, kind="decision"),
    ]
    _write_facts(tmp_path, "c5", facts)

    delivered = asyncio.run(courier._poll())

    assert delivered == 1
    cb.assert_called_once()
    assert cb.call_args[0][0]["id"] == "f2"


def test_filter_multi_field(tmp_path):
    ms = MemoryServer(str(tmp_path), "c6")
    courier = Courier(ms)
    cb = AsyncMock()

    asyncio.run(courier.subscribe(
        filter={"kind": "rule", "scope": "swarm-shared"},
        callback=cb,
    ))

    ts = _iso_offset(1)
    facts = [
        _make_fact("f1", created_at=ts, kind="rule", scope="agent-private"),
        _make_fact("f2", created_at=ts, kind="event", scope="swarm-shared"),
        _make_fact("f3", created_at=ts, kind="rule", scope="swarm-shared"),
    ]
    _write_facts(tmp_path, "c6", facts)

    delivered = asyncio.run(courier._poll())

    assert delivered == 1
    cb.assert_called_once()
    assert cb.call_args[0][0]["id"] == "f3"


def test_callback_exception_does_not_stop_delivery(tmp_path):
    ms = MemoryServer(str(tmp_path), "c7")
    courier = Courier(ms)

    bad_cb = AsyncMock(side_effect=RuntimeError("boom"))
    good_cb = AsyncMock()

    asyncio.run(courier.subscribe(filter={}, callback=bad_cb))
    asyncio.run(courier.subscribe(filter={}, callback=good_cb))

    fact = _make_fact("f1", created_at=_iso_offset(1))
    _write_facts(tmp_path, "c7", [fact])

    delivered = asyncio.run(courier._poll())

    # bad_cb raised but good_cb still got called
    bad_cb.assert_called_once()
    good_cb.assert_called_once()
    # delivered counts only successful deliveries
    assert delivered == 1


def test_one_poll_per_interval(tmp_path):
    ms = MemoryServer(str(tmp_path), "c8")
    courier = Courier(ms)

    cb1 = AsyncMock()
    cb2 = AsyncMock()
    cb3 = AsyncMock()
    asyncio.run(courier.subscribe(filter={}, callback=cb1))
    asyncio.run(courier.subscribe(filter={}, callback=cb2))
    asyncio.run(courier.subscribe(filter={}, callback=cb3))

    fact = _make_fact("f1", created_at=_iso_offset(1))
    _write_facts(tmp_path, "c8", [fact])

    with patch.object(courier, "_load_facts", wraps=courier._load_facts) as mock_load:
        asyncio.run(courier._poll())
        assert mock_load.call_count == 1


def test_deliver_existing_true(tmp_path):
    ms = MemoryServer(str(tmp_path), "c9")

    # Write fact BEFORE subscribe
    fact = _make_fact("f_existing", created_at="2020-01-01T00:00:00+00:00")
    _write_facts(tmp_path, "c9", [fact])

    courier = Courier(ms)
    cb = AsyncMock()

    asyncio.run(courier.subscribe(filter={}, callback=cb, deliver_existing=True))

    # Callback called immediately during subscribe
    cb.assert_called_once()
    assert cb.call_args[0][0]["id"] == "f_existing"


def test_deliver_existing_false(tmp_path):
    ms = MemoryServer(str(tmp_path), "c10")

    # Write fact BEFORE subscribe
    old_ts = "2020-01-01T00:00:00+00:00"
    fact = _make_fact("f_old", created_at=old_ts)
    _write_facts(tmp_path, "c10", [fact])

    courier = Courier(ms)
    cb = AsyncMock()

    asyncio.run(courier.subscribe(filter={}, callback=cb, deliver_existing=False))

    # Not called during subscribe
    cb.assert_not_called()

    # Not called on next poll either (fact is old)
    asyncio.run(courier._poll())
    cb.assert_not_called()


def test_stop_exits_run_loop(tmp_path):
    ms = MemoryServer(str(tmp_path), "c11")
    courier = Courier(ms)

    async def _run_and_stop():
        async def _stop_after_poll():
            await asyncio.sleep(0.05)
            await courier.stop()

        stop_task = asyncio.create_task(_stop_after_poll())
        await courier.run(poll_interval=0.01)
        await stop_task

    # run() must exit cleanly — if it hangs, asyncio.run will timeout
    asyncio.run(asyncio.wait_for(_run_and_stop(), timeout=2.0))
    assert not courier._running


def test_last_seen_at_advances(tmp_path):
    ms = MemoryServer(str(tmp_path), "c12")
    courier = Courier(ms)
    cb = AsyncMock()
    asyncio.run(courier.subscribe(filter={}, callback=cb))

    # First poll: fact at t1 (future)
    t1 = _iso_offset(1)
    _write_facts(tmp_path, "c12", [_make_fact("f1", created_at=t1)])
    asyncio.run(courier._poll())
    assert cb.call_count == 1

    # Second poll: add fact at t0 (before t1) — must NOT be delivered
    t0 = _iso_offset(-10)
    _write_facts(tmp_path, "c12", [
        _make_fact("f1", created_at=t1),
        _make_fact("f0", created_at=t0),
    ])
    asyncio.run(courier._poll())
    assert cb.call_count == 1  # still 1, not 2
