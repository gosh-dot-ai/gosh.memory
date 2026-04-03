"""Tests for memory_query — structured query on facts."""

import asyncio
import copy
import tempfile

import pytest

from src.memory import MemoryServer

TEST_FACTS = [
    {"fact": "Budget is 500k", "kind": "constraint", "session": 1,
     "entities": ["budget"], "tags": ["finance"],
     "speaker": "Alice", "event_date": "2026-03-01",
     "metadata": {"department": "engineering", "currency": "USD", "amount": 500000}},
    {"fact": "Use SQLite for storage", "kind": "decision", "session": 1,
     "entities": ["sqlite", "database"], "tags": ["tech"],
     "speaker": "Bob", "event_date": "2026-03-01",
     "metadata": {"component": "storage", "approved": True}},
    {"fact": "Ship feature by Friday", "kind": "action_item", "session": 2,
     "entities": ["feature"], "tags": ["deadline"],
     "speaker": "Alice", "event_date": "2026-03-10",
     "metadata": {"assignee": "bob", "priority": 1, "due_date": "2026-03-14"}},
    {"fact": "Alice prefers dark mode", "kind": "preference", "session": 2,
     "entities": ["alice", "ui"], "tags": ["ux"],
     "speaker": "Alice",
     "metadata": {"feature": "theme", "value": "dark"}},
    {"fact": "Karnali benchmark uses 300 queries", "kind": "fact", "session": 3,
     "entities": ["karnali", "benchmark"], "tags": ["karnali", "eval"],
     "speaker": "Bob",
     "metadata": {"benchmark": "karnali", "query_count": 300}},
]


def _make_server():
    """Create a MemoryServer with test facts ingested."""
    tmp = tempfile.mkdtemp()
    s = MemoryServer(key="test_query", data_dir=tmp)
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(s.ingest_asserted_facts(
            facts=[copy.deepcopy(f) for f in TEST_FACTS],
            raw_sessions=[
                {"session_num": 1, "session_date": "2026-03-10", "content": "session 1"},
                {"session_num": 2, "session_date": "2026-03-15", "content": "session 2"},
                {"session_num": 3, "session_date": "2026-03-18", "content": "session 3"},
            ],
            enrich_l0=False,
        ))
    finally:
        loop.close()
    return s


@pytest.fixture
def server():
    return _make_server()


# ── Test 1: basic query returns all facts ──

@pytest.mark.asyncio
async def test_query_returns_all_facts(server):
    result = await server.query(limit=100)
    assert "total" in result
    assert "facts" in result
    assert "has_more" in result
    assert result["total"] == 5


# ── Test 2: filter by kind ──

@pytest.mark.asyncio
async def test_filter_by_kind(server):
    result = await server.query(filter={"kind": "constraint"}, limit=100)
    assert result["total"] == 1
    assert result["facts"][0]["fact"] == "Budget is 500k"


# ── Test 3: filter by speaker ──

@pytest.mark.asyncio
async def test_filter_by_speaker(server):
    result = await server.query(filter={"speaker": "Alice"}, limit=100)
    assert result["total"] == 3
    for f in result["facts"]:
        assert f["speaker"] == "Alice"


# ── Test 4: filter by entities (contains match) ──

@pytest.mark.asyncio
async def test_filter_by_entities_contains(server):
    result = await server.query(filter={"entities": "budget"}, limit=100)
    assert result["total"] == 1
    assert "budget" in result["facts"][0]["entities"]


# ── Test 5: filter by tags (contains match) ──

@pytest.mark.asyncio
async def test_filter_by_tags_contains(server):
    result = await server.query(filter={"tags": "tech"}, limit=100)
    assert result["total"] == 1
    assert "tech" in result["facts"][0]["tags"]


# ── Test 6: filter by metadata exact match ──

@pytest.mark.asyncio
async def test_filter_by_metadata_exact(server):
    result = await server.query(filter={"metadata.department": "engineering"}, limit=100)
    assert result["total"] == 1
    assert result["facts"][0]["metadata"]["department"] == "engineering"


# ── Test 7: filter by metadata boolean ──

@pytest.mark.asyncio
async def test_filter_by_metadata_boolean(server):
    result = await server.query(filter={"metadata.approved": True}, limit=100)
    assert result["total"] == 1
    assert result["facts"][0]["fact"] == "Use SQLite for storage"


# ── Test 8: sort by session ascending ──

@pytest.mark.asyncio
async def test_sort_by_session_asc(server):
    result = await server.query(sort_by="session", sort_order="asc", limit=100)
    sessions = [f["session"] for f in result["facts"]]
    assert sessions == sorted(sessions)


# ── Test 9: sort by session descending ──

@pytest.mark.asyncio
async def test_sort_by_session_desc(server):
    result = await server.query(sort_by="session", sort_order="desc", limit=100)
    sessions = [f["session"] for f in result["facts"]]
    assert sessions == sorted(sessions, reverse=True)


# ── Test 10: sort by kind ascending ──

@pytest.mark.asyncio
async def test_sort_by_kind_asc(server):
    result = await server.query(sort_by="kind", sort_order="asc", limit=100)
    kinds = [f["kind"] for f in result["facts"]]
    assert kinds == sorted(kinds)


# ── Test 11: pagination limit ──

@pytest.mark.asyncio
async def test_pagination_limit(server):
    result = await server.query(limit=2)
    assert len(result["facts"]) == 2
    assert result["has_more"] is True
    assert result["total"] == 5


# ── Test 12: pagination offset ──

@pytest.mark.asyncio
async def test_pagination_offset(server):
    page1 = await server.query(sort_by="session", sort_order="asc", limit=2, offset=0)
    page2 = await server.query(sort_by="session", sort_order="asc", limit=2, offset=2)
    facts1 = {f["fact"] for f in page1["facts"]}
    facts2 = {f["fact"] for f in page2["facts"]}
    assert facts1.isdisjoint(facts2), "pages should not overlap"


# ── Test 13: pagination last page ──

@pytest.mark.asyncio
async def test_pagination_last_page(server):
    result = await server.query(limit=2, offset=4)
    assert len(result["facts"]) == 1
    assert result["has_more"] is False


# ── Test 14: invalid sort_order ──

@pytest.mark.asyncio
async def test_invalid_sort_order(server):
    result = await server.query(sort_order="invalid")
    assert "error" in result
    assert result["code"] == "INVALID_SORT_ORDER"


# ── Test 15: invalid sort_by field ──

@pytest.mark.asyncio
async def test_invalid_sort_by(server):
    result = await server.query(sort_by="nonexistent_field")
    assert "error" in result
    assert result["code"] == "INVALID_SORT_FIELD"


# ── Test 16: negative limit ──

@pytest.mark.asyncio
async def test_negative_limit(server):
    result = await server.query(limit=-1)
    assert "error" in result
    assert result["code"] == "INVALID_PAGINATION"


# ── Test 17: combined filter + sort + pagination ──

@pytest.mark.asyncio
async def test_combined_filter_sort_pagination(server):
    result = await server.query(
        filter={"speaker": "Alice"},
        sort_by="session",
        sort_order="asc",
        limit=2,
        offset=0,
    )
    assert result["total"] == 3
    assert len(result["facts"]) == 2
    assert result["has_more"] is True
    sessions = [f["session"] for f in result["facts"]]
    assert sessions == sorted(sessions)


# ── Test 18: metadata schema set + get ──

@pytest.mark.asyncio
async def test_metadata_schema_set_get(server):
    schema = {
        "department": {"type": "string", "required": True},
        "priority": {"type": "integer"},
    }
    await server.set_metadata_schema(schema)
    got = server.get_metadata_schema()
    assert got == schema


# ── Test 19: metadata schema validation on store rejects bad type ──

@pytest.mark.asyncio
async def test_metadata_schema_validation_rejects_bad_type(server):
    await server.set_metadata_schema({
        "department": {"type": "string", "required": True},
    })
    err = server._validate_metadata({"department": 123})
    assert err is not None
    assert "expected string" in err


# ── Test 20: metadata schema required field missing ──

@pytest.mark.asyncio
async def test_metadata_schema_required_field_missing(server):
    await server.set_metadata_schema({
        "department": {"type": "string", "required": True},
    })
    err = server._validate_metadata({"other_field": "value"})
    assert err is not None
    assert "required" in err


# ── Test 21: sort by metadata field with schema ──

@pytest.mark.asyncio
async def test_sort_by_metadata_field_with_schema(server):
    await server.set_metadata_schema({
        "priority": {"type": "integer"},
    })
    result = await server.query(
        filter={"metadata.priority": 1},
        sort_by="kind",
        sort_order="asc",
        limit=100,
    )
    assert result["total"] == 1
    assert result["facts"][0]["metadata"]["priority"] == 1


# ── Test 22: sort by metadata.* without schema returns error ──

@pytest.mark.asyncio
async def test_sort_by_metadata_without_schema_errors(server):
    result = await server.query(sort_by="metadata.priority")
    assert "error" in result
    assert result["code"] == "INVALID_SORT_FIELD"


# ── Test 23: metadata range filter on numeric field ──

@pytest.mark.asyncio
async def test_metadata_range_filter(server):
    await server.set_metadata_schema({
        "amount": {"type": "number"},
        "query_count": {"type": "number"},
        "priority": {"type": "integer"},
    })
    # Range: amount >= 100000
    result = await server.query(
        filter={"metadata.amount": {"gte": 100000}},
        limit=100,
    )
    assert result["total"] == 1
    assert result["facts"][0]["metadata"]["amount"] == 500000

    # Range: priority < 5
    result2 = await server.query(
        filter={"metadata.priority": {"lt": 5}},
        limit=100,
    )
    assert result2["total"] == 1
    assert result2["facts"][0]["metadata"]["priority"] == 1


@pytest.mark.asyncio
async def test_acl_private_facts_excluded():
    """Private facts are not visible to unauthorized callers in query()."""
    with tempfile.TemporaryDirectory() as tmp:
        server = MemoryServer(key="acl_test", data_dir=tmp)

        # Public facts
        await server.ingest_asserted_facts(
            facts=[{"fact": "Public fact", "kind": "fact", "session": 1,
                    "entities": ["public"], "tags": []}],
            raw_sessions=[{"session_num": 1, "session_date": "2026-03-20",
                           "content": "public"}],
            enrich_l0=False,
        )

        # Private fact owned by user:alice, read=[] (no one else can read)
        await server.ingest_asserted_facts(
            facts=[{"fact": "Alice secret", "kind": "fact", "session": 1,
                    "entities": ["secret"], "tags": []}],
            raw_sessions=[{"session_num": 1, "session_date": "2026-03-20",
                           "content": "private"}],
            owner_id="user:alice", read=[], write=[],
            enrich_l0=False,
        )

        # Bob cannot see Alice's private fact
        result = await server.query(
            caller_id="user:bob", caller_role="agent")
        assert all(f.get("owner_id") != "user:alice" for f in result["facts"])
        assert not any("secret" in f.get("entities", []) for f in result["facts"])

        # Admin can see everything
        result_admin = await server.query(caller_role="admin")
        assert result_admin["total"] > result["total"]


@pytest.mark.asyncio
async def test_query_tier_field():
    """Query results include _tier field identifying fact source tier."""
    with tempfile.TemporaryDirectory() as tmp:
        server = MemoryServer(key="tier_test", data_dir=tmp)
        await server.ingest_asserted_facts(
            facts=[{"fact": "Granular fact", "kind": "fact", "session": 1,
                    "entities": ["test"], "tags": []}],
            cross_session=[{"fact": "Cross fact", "kind": "fact",
                           "sessions": [1], "entities": ["cross"], "tags": []}],
            raw_sessions=[{"session_num": 1, "session_date": "2026-03-20",
                           "content": "tier test"}],
            enrich_l0=False,
        )

        result = await server.query(caller_role="admin", limit=100)
        tiers = set(f.get("_tier") for f in result["facts"])
        assert "granular" in tiers
        assert "cross_session" in tiers
