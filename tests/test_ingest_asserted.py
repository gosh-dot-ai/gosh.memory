"""Tests for ingest_asserted_facts — authoritative import of pre-extracted facts."""

import asyncio
import copy
import json

import numpy as np
import pytest

from src.memory import MemoryServer

DIM = 3072


# ── Patches ──

def _patch_embed(monkeypatch):
    async def mock_embed(texts, **kw):
        return np.random.randn(len(texts), DIM).astype(np.float32)
    async def mock_eq(text, **kw):
        return np.random.randn(DIM).astype(np.float32)
    monkeypatch.setattr("src.memory.embed_texts", mock_embed)
    monkeypatch.setattr("src.memory.embed_query", mock_eq)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda f, l: None)


def _patch_extraction(monkeypatch):
    """Patch extraction for store() calls (used in offset tests)."""
    async def mock_extract(**kwargs):
        sn = kwargs.get("session_num", 1)
        return ("conv", sn, "2024-06-01", [
            {"id": f"f{sn}_{i}", "fact": f"Stored fact {sn}-{i}", "kind": "event",
             "entities": [], "tags": [], "session": sn}
            for i in range(3)
        ], [])

    async def mock_consolidate(**kwargs):
        return ("conv", 1, "2024-06-01", [])

    async def mock_cross(**kwargs):
        return ("conv", "e", [])

    monkeypatch.setattr("src.memory.extract_session", mock_extract)
    monkeypatch.setattr("src.memory.consolidate_session", mock_consolidate)
    monkeypatch.setattr("src.memory.cross_session_entity", mock_cross)


def _patch_all(monkeypatch):
    _patch_embed(monkeypatch)
    _patch_extraction(monkeypatch)


def _make_facts(n, session=1):
    """Create n simple facts for session."""
    return [
        {"id": f"f_{i:02d}", "fact": f"Fact {i} from session {session}",
         "kind": "event", "entities": ["Alice"], "tags": ["test"],
         "session": session}
        for i in range(1, n + 1)
    ]


def _make_raw_sessions(n):
    """Create n raw sessions with dense numbering 1..n."""
    return [
        {"session_num": i, "session_date": f"2024-0{i}-01",
         "content": f"Raw content for session {i}",
         "speakers": "User and Assistant"}
        for i in range(1, n + 1)
    ]


# ── Tests ──


@pytest.mark.asyncio
async def test_basic_import(tmp_path, monkeypatch):
    """Import 3 facts -> granular_added=3, recall finds them."""
    _patch_embed(monkeypatch)
    server = MemoryServer(data_dir=str(tmp_path), key="basic_import")
    facts = _make_facts(3, session=1)
    raw = _make_raw_sessions(1)

    result = await server.ingest_asserted_facts(
        facts=facts, raw_sessions=raw)

    assert result["granular_added"] == 3
    assert result["session_offset"] == 0
    assert len(server._all_granular) == 3
    # All facts should be findable
    for f in server._all_granular:
        assert f["conv_id"] == "basic_import"
        assert "created_at" in f


@pytest.mark.asyncio
async def test_import_stores_raw_sessions_verbatim(tmp_path, monkeypatch):
    """Authoritative import must persist provided raw_sessions unchanged."""
    _patch_embed(monkeypatch)
    server = MemoryServer(data_dir=str(tmp_path), key="raw_import")
    facts = _make_facts(1, session=1)
    raw = [{
        "session_num": 1,
        "session_date": "2024-01-01",
        "content": "Imported raw content",
        "speakers": "User and Assistant",
    }]

    result = await server.ingest_asserted_facts(facts=facts, raw_sessions=raw)

    assert result["raw_sessions_added"] == 1
    assert len(server._raw_sessions) == 1
    assert server._raw_sessions[0]["content"] == "Imported raw content"
    assert server._raw_sessions[0]["session_date"] == "2024-01-01"
    assert server._raw_sessions[0]["session_num"] == 1


@pytest.mark.asyncio
async def test_import_writes_session_content_complexity(tmp_path, monkeypatch):
    """ingest_asserted_facts() computes _session_content_complexity from imported facts."""
    _patch_embed(monkeypatch)
    server = MemoryServer(data_dir=str(tmp_path), key="import_cc")
    facts = [
        {"id": "f_01", "fact": "Do thing", "kind": "action_item",
         "entities": ["Alice"], "tags": ["todo"], "session": 1},
        {"id": "f_02", "fact": "Need review", "kind": "requirement",
         "entities": ["Bob"], "tags": ["review"], "session": 1},
    ]
    raw = _make_raw_sessions(1)

    result = await server.ingest_asserted_facts(facts=facts, raw_sessions=raw)

    assert result["granular_added"] == 2
    for f in server._all_granular:
        assert "_session_content_complexity" in f
        assert isinstance(f["_session_content_complexity"], float)
        assert f["_session_content_complexity"] > 0.0


@pytest.mark.asyncio
async def test_import_propagates_explicit_identity_to_granular(tmp_path, monkeypatch):
    """ingest_asserted_facts() should honor explicit artifact/version IDs on imported facts."""
    _patch_embed(monkeypatch)
    server = MemoryServer(data_dir=str(tmp_path), key="import_identity")
    facts = _make_facts(2, session=1)
    raw = _make_raw_sessions(1)

    result = await server.ingest_asserted_facts(
        facts=facts,
        raw_sessions=raw,
        artifact_id="art_imported123",
        version_id="ver_imported123",
    )

    assert result["granular_added"] == 2
    for f in server._all_granular:
        assert f["artifact_id"] == "art_imported123"
        assert f["version_id"] == "ver_imported123"
        assert f["status"] == "active"


@pytest.mark.asyncio
async def test_non_empty_memory_offset(tmp_path, monkeypatch):
    """store() 10 sessions first, then import 3 facts with sessions 1,2,3 +
    raw_sessions -> offset=10, imported facts have session=11,12,13,
    get_more_context(11) works."""
    _patch_all(monkeypatch)
    server = MemoryServer(data_dir=str(tmp_path), key="offset_test")

    # Store 10 sessions via store()
    for sn in range(1, 11):
        await server.store(f"Content for session {sn}",
                           session_num=sn, session_date=f"2024-01-{sn:02d}")

    assert server._n_sessions == 10

    # Now import 3 facts across 3 sessions
    facts = (
        _make_facts(1, session=1) +
        _make_facts(1, session=2) +
        _make_facts(1, session=3)
    )
    raw = _make_raw_sessions(3)

    result = await server.ingest_asserted_facts(
        facts=facts, raw_sessions=raw)

    assert result["session_offset"] == 10
    # Check that imported facts have offset sessions
    imported_sessions = set()
    for f in server._all_granular:
        if "Fact" in f.get("fact", ""):
            imported_sessions.add(f["session"])
    assert 11 in imported_sessions
    assert 12 in imported_sessions
    assert 13 in imported_sessions

    # get_more_context: raw session 11 should be accessible
    rs_11 = [rs for rs in server._raw_sessions if rs["session_num"] == 11]
    assert len(rs_11) == 1
    assert "Raw content for session 1" in rs_11[0]["content"]


@pytest.mark.asyncio
async def test_two_sequential_imports(tmp_path, monkeypatch):
    """Batch A (3 sessions), batch B (2 sessions) -> offset=0 then offset=3,
    no ID collisions, recall finds both."""
    _patch_embed(monkeypatch)
    server = MemoryServer(data_dir=str(tmp_path), key="seq_import")

    # Batch A: 3 sessions
    facts_a = (
        _make_facts(2, session=1) +
        _make_facts(2, session=2) +
        _make_facts(2, session=3)
    )
    raw_a = _make_raw_sessions(3)

    result_a = await server.ingest_asserted_facts(
        facts=copy.deepcopy(facts_a), raw_sessions=copy.deepcopy(raw_a))
    assert result_a["session_offset"] == 0
    assert result_a["granular_added"] == 6

    # Batch B: 2 sessions
    facts_b = (
        _make_facts(2, session=1) +
        _make_facts(2, session=2)
    )
    raw_b = _make_raw_sessions(2)

    result_b = await server.ingest_asserted_facts(
        facts=copy.deepcopy(facts_b), raw_sessions=copy.deepcopy(raw_b))
    assert result_b["session_offset"] == 3
    assert result_b["granular_added"] == 4

    # Total: 10 granular facts
    assert len(server._all_granular) == 10

    # No ID collisions
    ids = [f["id"] for f in server._all_granular]
    assert len(ids) == len(set(ids)), f"ID collision: {ids}"

    # import_uids are different
    assert result_a["import_uid"] != result_b["import_uid"]


@pytest.mark.asyncio
async def test_n_sessions_includes_raw(tmp_path, monkeypatch):
    """Import 5 raw_sessions but only 3 have facts -> _n_sessions=5."""
    _patch_embed(monkeypatch)
    server = MemoryServer(data_dir=str(tmp_path), key="nsess_raw")

    # Facts only for sessions 1, 2, 3
    facts = (
        _make_facts(1, session=1) +
        _make_facts(1, session=2) +
        _make_facts(1, session=3)
    )
    raw = _make_raw_sessions(5)

    result = await server.ingest_asserted_facts(
        facts=facts, raw_sessions=raw)

    assert server._n_sessions == 5
    assert server._n_sessions_with_facts == 3


@pytest.mark.asyncio
async def test_unresolved_source_ids(tmp_path, monkeypatch):
    """Cons fact with source_ids=["nonexistent"] -> VALIDATION_ERROR."""
    _patch_embed(monkeypatch)
    server = MemoryServer(data_dir=str(tmp_path), key="unresolved")

    facts = _make_facts(2, session=1)
    cons = [
        {"id": "c_01", "fact": "Consolidated fact", "session": 1,
         "source_ids": ["nonexistent"]}
    ]
    raw = _make_raw_sessions(1)

    result = await server.ingest_asserted_facts(
        facts=facts, consolidated=cons, raw_sessions=raw)

    assert result["code"] == "VALIDATION_ERROR"
    assert "Unresolved source_id" in result["error"]


@pytest.mark.asyncio
async def test_ambiguous_source_ids(tmp_path, monkeypatch):
    """Cross fact spans sessions [1,2], source_id="f_01" exists in both
    -> VALIDATION_ERROR."""
    _patch_embed(monkeypatch)
    server = MemoryServer(data_dir=str(tmp_path), key="ambiguous")

    # f_01 exists in session 1 AND session 2
    facts = [
        {"id": "f_01", "fact": "Fact in session 1", "session": 1,
         "kind": "event", "entities": ["Alice"], "tags": []},
        {"id": "f_01", "fact": "Fact in session 2", "session": 2,
         "kind": "event", "entities": ["Alice"], "tags": []},
    ]
    cross = [
        {"id": "x_01", "fact": "Cross fact", "sessions": [1, 2],
         "entities": ["Alice"], "source_ids": ["f_01"]}
    ]
    raw = _make_raw_sessions(2)

    result = await server.ingest_asserted_facts(
        facts=facts, cross_session=cross, raw_sessions=raw)

    assert result["code"] == "VALIDATION_ERROR"
    assert "Ambiguous source_id" in result["error"]


@pytest.mark.asyncio
async def test_session_validation_dense(tmp_path, monkeypatch):
    """[1,2,3] success, [1,3,5] error, [1,1,2] error, missing session_num error."""
    _patch_embed(monkeypatch)

    # Success: dense [1,2,3]
    server = MemoryServer(data_dir=str(tmp_path), key="dense_ok")
    facts = _make_facts(1, session=1)
    raw_ok = _make_raw_sessions(3)
    result = await server.ingest_asserted_facts(
        facts=facts, raw_sessions=raw_ok)
    assert "error" not in result
    assert result["granular_added"] == 1

    # Error: sparse [1,3,5]
    server2 = MemoryServer(data_dir=str(tmp_path), key="dense_sparse")
    raw_sparse = [
        {"session_num": 1, "content": "a"},
        {"session_num": 3, "content": "b"},
        {"session_num": 5, "content": "c"},
    ]
    result2 = await server2.ingest_asserted_facts(
        facts=_make_facts(1, session=1), raw_sessions=raw_sparse)
    assert result2["code"] == "VALIDATION_ERROR"
    assert "dense 1..N" in result2["error"]

    # Error: duplicate [1,1,2]
    server3 = MemoryServer(data_dir=str(tmp_path), key="dense_dup")
    raw_dup = [
        {"session_num": 1, "content": "a"},
        {"session_num": 1, "content": "b"},
        {"session_num": 2, "content": "c"},
    ]
    result3 = await server3.ingest_asserted_facts(
        facts=_make_facts(1, session=1), raw_sessions=raw_dup)
    assert result3["code"] == "VALIDATION_ERROR"
    assert "Duplicate" in result3["error"]

    # Error: missing session_num
    server4 = MemoryServer(data_dir=str(tmp_path), key="dense_missing")
    raw_missing = [{"content": "no session_num"}]
    result4 = await server4.ingest_asserted_facts(
        facts=_make_facts(1, session=1), raw_sessions=raw_missing)
    assert result4["code"] == "VALIDATION_ERROR"
    assert "missing session_num" in result4["error"]


@pytest.mark.asyncio
async def test_scoped_id_collision(tmp_path, monkeypatch):
    """3 sessions each with id="f_01" -> 3 distinct entries after import."""
    _patch_embed(monkeypatch)
    server = MemoryServer(data_dir=str(tmp_path), key="id_collision")

    facts = [
        {"id": "f_01", "fact": "Session 1 fact", "session": 1,
         "kind": "event", "entities": [], "tags": []},
        {"id": "f_01", "fact": "Session 2 fact", "session": 2,
         "kind": "event", "entities": [], "tags": []},
        {"id": "f_01", "fact": "Session 3 fact", "session": 3,
         "kind": "event", "entities": [], "tags": []},
    ]
    raw = _make_raw_sessions(3)

    result = await server.ingest_asserted_facts(
        facts=facts, raw_sessions=raw)

    assert result["granular_added"] == 3
    ids = [f["id"] for f in server._all_granular]
    assert len(ids) == len(set(ids)), f"IDs not unique: {ids}"
    # All should contain the import_uid prefix
    for fid in ids:
        assert result["import_uid"] in fid


@pytest.mark.asyncio
async def test_kind_preservation(tmp_path, monkeypatch):
    """kind="action_item" in cons -> stays "action_item"."""
    _patch_embed(monkeypatch)
    server = MemoryServer(data_dir=str(tmp_path), key="kind_pres")

    facts = _make_facts(2, session=1)
    cons = [
        {"id": "c_01", "fact": "Action item fact", "kind": "action_item",
         "session": 1, "source_ids": ["f_01"]}
    ]
    raw = _make_raw_sessions(1)

    result = await server.ingest_asserted_facts(
        facts=facts, consolidated=cons, raw_sessions=raw)

    assert result["consolidated_added"] == 1
    assert server._all_cons[0]["kind"] == "action_item"


@pytest.mark.asyncio
async def test_acl_propagation(tmp_path, monkeypatch):
    """Import with owner_id="user:alice" -> all tiers + raw tagged correctly."""
    _patch_embed(monkeypatch)
    server = MemoryServer(data_dir=str(tmp_path), key="acl_prop")

    facts = _make_facts(2, session=1)
    cons = [
        {"id": "c_01", "fact": "Consolidated", "session": 1,
         "source_ids": ["f_01"]}
    ]
    cross = [
        {"id": "x_01", "fact": "Cross-session", "entities": ["Alice"],
         "sessions": [1], "source_ids": ["f_02"]}
    ]
    raw = _make_raw_sessions(1)

    result = await server.ingest_asserted_facts(
        facts=facts, consolidated=cons, cross_session=cross,
        raw_sessions=raw,
        owner_id="user:alice",
        read=["swarm:alpha"], write=["swarm:alpha"])

    # Check granular
    for f in server._all_granular:
        assert f["owner_id"] == "user:alice"
        assert f["read"] == ["swarm:alpha"]
        assert f["write"] == ["swarm:alpha"]

    # Check consolidated
    for cf in server._all_cons:
        assert cf["owner_id"] == "user:alice"
        assert cf["read"] == ["swarm:alpha"]

    # Check cross-session
    for xf in server._all_cross:
        assert xf["owner_id"] == "user:alice"
        assert xf["read"] == ["swarm:alpha"]

    # Check raw sessions
    for rs in server._raw_sessions:
        assert rs["owner_id"] == "user:alice"
        assert rs["read"] == ["swarm:alpha"]


@pytest.mark.asyncio
async def test_facts_cons_cross_authoritative(tmp_path, monkeypatch):
    """Imported cons/cross stored as-is, not re-derived."""
    _patch_embed(monkeypatch)
    server = MemoryServer(data_dir=str(tmp_path), key="authoritative")

    facts = _make_facts(2, session=1)
    cons = [
        {"id": "c_01", "fact": "My custom consolidated text",
         "session": 1, "source_ids": ["f_01"],
         "entities": ["Bob"], "kind": "summary"}
    ]
    cross = [
        {"id": "x_01", "fact": "My custom cross-session text",
         "entities": ["Bob"], "sessions": [1],
         "source_ids": ["f_02"], "kind": "profile"}
    ]
    raw = _make_raw_sessions(1)

    result = await server.ingest_asserted_facts(
        facts=facts, consolidated=cons, cross_session=cross,
        raw_sessions=raw)

    assert result["consolidated_added"] == 1
    assert result["cross_session_added"] == 1

    # Verify the text is stored exactly as provided
    assert server._all_cons[0]["fact"] == "My custom consolidated text"
    assert server._all_cons[0]["entities"] == ["Bob"]
    assert server._all_cons[0]["kind"] == "summary"

    assert server._all_cross[0]["fact"] == "My custom cross-session text"
    assert server._all_cross[0]["entities"] == ["Bob"]
    assert server._all_cross[0]["kind"] == "profile"

    # Tier flags must prevent re-derivation
    assert server._tiers_dirty is False
    assert server._tier2_built is True
    assert server._tier3_built is True


@pytest.mark.asyncio
async def test_asserted_derived_tiers_survive_restart(tmp_path, monkeypatch):
    """Asserted consolidated/cross tiers must persist across restart."""
    _patch_embed(monkeypatch)
    server = MemoryServer(data_dir=str(tmp_path), key="asserted_restart")

    facts = _make_facts(2, session=1)
    cons = [
        {"id": "c_01", "fact": "Restart-safe consolidated text",
         "session": 1, "source_ids": ["f_01"], "entities": ["Bob"], "kind": "summary"}
    ]
    cross = [
        {"id": "x_01", "fact": "Restart-safe cross text",
         "entities": ["Bob"], "sessions": [1], "source_ids": ["f_02"], "kind": "profile"}
    ]
    raw = _make_raw_sessions(1)

    result = await server.ingest_asserted_facts(
        facts=facts, consolidated=cons, cross_session=cross, raw_sessions=raw
    )

    assert result["consolidated_added"] == 1
    assert result["cross_session_added"] == 1

    restarted = MemoryServer(data_dir=str(tmp_path), key="asserted_restart")
    assert len(restarted._all_granular) == 2
    assert len(restarted._all_cons) == 1
    assert len(restarted._all_cross) == 1
    assert restarted._all_cons[0]["fact"] == "Restart-safe consolidated text"
    assert restarted._all_cross[0]["fact"] == "Restart-safe cross text"
    assert restarted._all_cons[0]["metadata"]["asserted_derived_tier"] is True
    assert restarted._all_cross[0]["metadata"]["asserted_derived_tier"] is True


@pytest.mark.asyncio
async def test_asserted_derived_tiers_survive_store(tmp_path, monkeypatch):
    """Ordinary store() must not discard authoritative asserted derived tiers."""
    _patch_all(monkeypatch)
    server = MemoryServer(data_dir=str(tmp_path), key="asserted_store")

    facts = _make_facts(2, session=1)
    cons = [
        {"id": "c_01", "fact": "Persisted derived summary",
         "session": 1, "source_ids": ["f_01"], "entities": ["Entity A"], "kind": "summary"}
    ]
    cross = [
        {"id": "x_01", "fact": "Persisted derived profile",
         "entities": ["Entity A"], "sessions": [1], "source_ids": ["f_02"], "kind": "profile"}
    ]
    raw = _make_raw_sessions(1)

    await server.ingest_asserted_facts(
        facts=facts, consolidated=cons, cross_session=cross, raw_sessions=raw
    )
    assert (len(server._all_cons), len(server._all_cross)) == (1, 1)

    await server.store("User: follow-up session", session_num=2, session_date="2024-06-02")

    assert len(server._all_cons) == 1
    assert len(server._all_cross) >= 1
    assert any(f["fact"] == "Persisted derived summary" for f in server._all_cons)
    assert any(f["fact"] == "Persisted derived profile" for f in server._all_cross)


@pytest.mark.asyncio
async def test_asserted_derived_tiers_survive_flush_background(tmp_path, monkeypatch):
    """flush_background() must preserve authoritative asserted derived tiers."""
    _patch_all(monkeypatch)
    server = MemoryServer(data_dir=str(tmp_path), key="asserted_flush")

    facts = _make_facts(2, session=1)
    cons = [
        {"id": "c_01", "fact": "Persisted derived summary",
         "session": 1, "source_ids": ["f_01"], "entities": ["Entity A"], "kind": "summary"}
    ]
    cross = [
        {"id": "x_01", "fact": "Persisted derived profile",
         "entities": ["Entity A"], "sessions": [1], "source_ids": ["f_02"], "kind": "profile"}
    ]
    raw = _make_raw_sessions(1)

    await server.ingest_asserted_facts(
        facts=facts, consolidated=cons, cross_session=cross, raw_sessions=raw
    )
    assert (len(server._all_cons), len(server._all_cross)) == (1, 1)

    result = await server.flush_background()

    assert result["rebuilt"] is True
    assert len(server._all_cons) == 1
    assert len(server._all_cross) >= 1
    assert any(f["fact"] == "Persisted derived summary" for f in server._all_cons)
    assert any(f["fact"] == "Persisted derived profile" for f in server._all_cross)


@pytest.mark.asyncio
async def test_backward_compat_store(tmp_path, monkeypatch):
    """Existing store() still works after ingest_asserted_facts addition."""
    _patch_all(monkeypatch)
    server = MemoryServer(data_dir=str(tmp_path), key="compat_store")

    result = await server.store(
        "User: Hello, I'm testing backward compat.",
        session_num=1, session_date="2024-06-01")

    assert result["facts_extracted"] > 0
    assert len(server._all_granular) > 0

    # Persisted snapshot exists and is valid through the storage backend
    assert server._storage.exists
    data = server._storage.load_facts()
    assert len(data["granular"]) > 0
