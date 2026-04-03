"""Target routing and courier/query parity tests."""

from unittest.mock import AsyncMock

import pytest

from src.courier import Courier
from src.memory import MemoryServer


def _patch_extract(monkeypatch):
    async def mock_extract_session(**kwargs):
        sn = kwargs.get("session_num", 1)
        facts = [
            {
                "id": "f0",
                "fact": "Targeted fact",
                "kind": "event",
                "entities": ["planner"],
                "tags": ["routing"],
                "session": sn,
            }
        ]
        return ("conv", sn, kwargs.get("session_date", "2024-06-01"), facts, [])

    monkeypatch.setattr("src.memory.extract_session", mock_extract_session)


@pytest.mark.asyncio
async def test_store_normalizes_string_target(tmp_path, monkeypatch):
    _patch_extract(monkeypatch)
    ms = MemoryServer(str(tmp_path), "target_store_string")

    result = await ms.store(
        "hello",
        session_num=1,
        session_date="2024-06-01",
        target="agent:planner",
    )

    assert result["facts_extracted"] == 1
    assert ms._all_granular[0]["target"] == ["agent:planner"]


@pytest.mark.asyncio
async def test_store_normalizes_target_list_dedup_preserves_order(tmp_path, monkeypatch):
    _patch_extract(monkeypatch)
    ms = MemoryServer(str(tmp_path), "target_store_list")

    result = await ms.store(
        "hello",
        session_num=1,
        session_date="2024-06-01",
        target=["swarm:alpha", "agent:planner", "swarm:alpha", "agent:planner"],
    )

    assert result["facts_extracted"] == 1
    assert ms._all_granular[0]["target"] == ["swarm:alpha", "agent:planner"]


@pytest.mark.asyncio
async def test_store_rejects_invalid_target(tmp_path, monkeypatch):
    _patch_extract(monkeypatch)
    ms = MemoryServer(str(tmp_path), "target_store_invalid")

    result = await ms.store(
        "hello",
        session_num=1,
        session_date="2024-06-01",
        target="planner",
    )

    assert result["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_ingest_asserted_facts_normalizes_target(tmp_path):
    ms = MemoryServer(str(tmp_path), "target_asserted")

    result = await ms.ingest_asserted_facts(
        facts=[
            {
                "id": "t1",
                "fact": "Planner task",
                "kind": "task",
                "session": 1,
                "entities": [],
                "tags": [],
                "target": "agent:planner",
            }
        ],
        raw_sessions=[
            {"session_num": 1, "session_date": "2024-06-01", "content": "session 1"},
        ],
        enrich_l0=False,
    )

    assert result["granular_added"] == 1
    assert ms._all_granular[0]["target"] == ["agent:planner"]


async def _seed_target_facts(ms: MemoryServer, *, read=None, owner_id=None):
    return await ms.ingest_asserted_facts(
        facts=[
            {
                "id": "f_planner",
                "fact": "Planner handoff",
                "kind": "task",
                "session": 1,
                "entities": ["planner"],
                "tags": ["handoff"],
                "target": ["agent:planner", "agent:coder"],
                "metadata": {"workflow_id": "wf_17", "message_type": "handoff", "priority": 5},
            },
            {
                "id": "f_other",
                "fact": "Other handoff",
                "kind": "task",
                "session": 1,
                "entities": ["coder"],
                "tags": ["handoff"],
                "target": ["agent:coder"],
                "metadata": {"workflow_id": "wf_18", "message_type": "handoff", "priority": 1},
            },
            {
                "id": "f_legacy",
                "fact": "Legacy task",
                "kind": "task",
                "session": 1,
                "entities": ["legacy"],
                "tags": ["legacy"],
                "metadata": {"workflow_id": "wf_17", "message_type": "audit", "priority": 4},
            },
        ],
        raw_sessions=[
            {"session_num": 1, "session_date": "2024-06-01", "content": "session 1"},
        ],
        owner_id=owner_id,
        read=read,
        write=read,
        enrich_l0=False,
    )


@pytest.mark.asyncio
async def test_query_target_contains_and_metadata_filter(tmp_path):
    ms = MemoryServer(str(tmp_path), "target_query")
    await ms.set_metadata_schema({
        "workflow_id": {"type": "string"},
        "message_type": {"type": "string"},
        "priority": {"type": "integer"},
    })
    await _seed_target_facts(ms)

    result = await ms.query(
        filter={"target": "agent:planner", "metadata.workflow_id": "wf_17"},
        limit=100,
    )

    assert result["total"] == 1
    assert result["facts"][0]["fact"] == "Planner handoff"


@pytest.mark.asyncio
async def test_courier_query_parity_for_target_and_metadata_filters(tmp_path):
    ms = MemoryServer(str(tmp_path), "target_parity")
    await ms.set_metadata_schema({
        "workflow_id": {"type": "string"},
        "message_type": {"type": "string"},
        "priority": {"type": "integer"},
    })
    await _seed_target_facts(ms)

    filter_ = {
        "target": "agent:planner",
        "metadata.workflow_id": "wf_17",
        "metadata.priority": {"gte": 3},
    }
    query_result = await ms.query(filter=filter_, limit=100)
    expected_ids = {f["id"] for f in query_result["facts"]}

    courier = Courier(ms)
    delivered = []

    async def cb(fact):
        delivered.append(fact["id"])

    await courier.subscribe(
        filter=filter_,
        callback=cb,
        deliver_existing=True,
        owner_id="agent:planner",
    )

    assert set(delivered) == expected_ids
    assert len(delivered) == 1


@pytest.mark.asyncio
async def test_courier_target_does_not_override_acl(tmp_path):
    ms = MemoryServer(str(tmp_path), "target_acl_deny")
    await ms.set_metadata_schema({"workflow_id": {"type": "string"}})
    await _seed_target_facts(ms, read=["swarm:alpha"], owner_id="agent:writer")

    courier = Courier(ms)
    cb = AsyncMock()

    await courier.subscribe(
        filter={"target": "agent:planner"},
        callback=cb,
        deliver_existing=True,
        owner_id="agent:planner",
        memberships=[],
    )

    cb.assert_not_called()


@pytest.mark.asyncio
async def test_courier_target_filter_only_wakes_matching_subscriber(tmp_path):
    ms = MemoryServer(str(tmp_path), "target_acl_match")
    await ms.set_metadata_schema({"workflow_id": {"type": "string"}})
    await _seed_target_facts(ms, read=["swarm:alpha"], owner_id="agent:writer")

    courier = Courier(ms)
    planner_cb = AsyncMock()
    coder_cb = AsyncMock()

    await courier.subscribe(
        filter={"target": "agent:planner"},
        callback=planner_cb,
        deliver_existing=True,
        owner_id="agent:planner",
        memberships=["swarm:alpha"],
    )
    await courier.subscribe(
        filter={"target": "agent:coder2"},
        callback=coder_cb,
        deliver_existing=True,
        owner_id="agent:coder2",
        memberships=["swarm:alpha"],
    )

    planner_cb.assert_called_once()
    coder_cb.assert_not_called()


@pytest.mark.asyncio
async def test_legacy_filters_still_work_without_target(tmp_path):
    ms = MemoryServer(str(tmp_path), "target_legacy")
    await _seed_target_facts(ms)

    query_result = await ms.query(filter={"kind": "task", "metadata.message_type": "audit"}, limit=100)
    assert query_result["total"] == 1
    assert "target" not in query_result["facts"][0]

    courier = Courier(ms)
    delivered = []

    async def cb(fact):
        delivered.append(fact["fact"])

    await courier.subscribe(
        filter={"kind": "task", "metadata.message_type": "audit"},
        callback=cb,
        deliver_existing=True,
    )

    assert delivered == ["Legacy task"]


@pytest.mark.asyncio
async def test_rebuild_tiers_partitions_by_target_and_preserves_derived_target(tmp_path, monkeypatch):
    async def mock_build_index():
        return {}

    async def mock_consolidate_session(**kwargs):
        session_facts = kwargs.get("session_facts", [])
        sn = kwargs.get("sn", 0)
        return ("conv", sn, "2024-06-01", [
            {
                "id": f"cons_{sn}",
                "fact": f"Cons {sn}",
                "kind": "summary",
                "entities": ["shared"],
                "tags": ["derived"],
                "source_ids": [f["id"] for f in session_facts],
            }
        ])

    async def mock_cross_session_entity(**kwargs):
        efacts = kwargs.get("efacts", [])
        return ("conv", "shared", [
            {
                "id": "cross_shared",
                "fact": "Cross shared",
                "kind": "profile",
                "entities": ["shared"],
                "tags": ["derived"],
                "source_ids": [f["id"] for f in efacts],
            }
        ])

    async def mock_extract_source_aggregation_facts(self, **kwargs):
        source_facts = kwargs.get("source_facts", [])
        return [{
            "id": "substrate_shared",
            "fact": "Cross shared",
            "kind": "fact",
            "entities": ["shared"],
            "tags": ["derived"],
            "source_ids": [f["id"] for f in source_facts],
            "metadata": {"source_aggregation": True},
        }]

    monkeypatch.setattr("src.memory.consolidate_session", mock_consolidate_session)
    monkeypatch.setattr("src.memory.cross_session_entity", mock_cross_session_entity)
    monkeypatch.setattr(MemoryServer, "_extract_source_aggregation_facts", mock_extract_source_aggregation_facts)

    ms = MemoryServer(str(tmp_path), "target_rebuild")
    monkeypatch.setattr(ms, "build_index", mock_build_index)

    await ms.ingest_asserted_facts(
        facts=[
            {
                "id": "p1",
                "fact": "Planner S1",
                "kind": "task",
                "session": 1,
                "source_id": "shared-source",
                "entities": ["shared"],
                "tags": ["t"],
                "target": ["agent:planner"],
            },
            {
                "id": "c1",
                "fact": "Coder S1",
                "kind": "task",
                "session": 1,
                "source_id": "shared-source",
                "entities": ["shared"],
                "tags": ["t"],
                "target": ["agent:coder"],
            },
            {
                "id": "p2",
                "fact": "Planner S2",
                "kind": "task",
                "session": 2,
                "source_id": "shared-source",
                "entities": ["shared"],
                "tags": ["t"],
                "target": ["agent:planner"],
            },
            {
                "id": "c2",
                "fact": "Coder S2",
                "kind": "task",
                "session": 2,
                "source_id": "shared-source",
                "entities": ["shared"],
                "tags": ["t"],
                "target": ["agent:coder"],
            },
        ],
        raw_sessions=[
            {"session_num": 1, "session_date": "2024-06-01", "content": "session 1"},
            {"session_num": 2, "session_date": "2024-06-02", "content": "session 2"},
        ],
        enrich_l0=False,
    )

    await ms._rebuild_tiers()

    cons_targets = sorted(tuple(f.get("target", [])) for f in ms._all_cons)
    cross_targets = sorted(tuple(f.get("target", [])) for f in ms._all_cross)

    assert cons_targets == []
    assert cross_targets == [
        ("agent:coder",),
        ("agent:planner",),
    ]
