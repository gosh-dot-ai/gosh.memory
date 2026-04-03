"""Phase 4 review fixes — regression tests.

Covers:
1. _git_dedup_index persistence (save/load round-trip)
2. _is_visible enforces TTL in recall, memory_list, memory_get, courier
3. Lifecycle ACL: edit/retract require write access, purge requires admin
4. ingest_document dedup per chunk (source_id + content_hash)
5. get_more_context TTL check on raw sessions
6. Backward-compat ACL derivation from scope in _acl_allows
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from src.memory import MemoryServer, _derive_acl_from_scope, _is_visible

# ── Helpers ──

def _make_server(tmp_path, key="review_test"):
    return MemoryServer(data_dir=str(tmp_path), key=key)


def _make_fact(overrides=None):
    base = {
        "id": "f_001",
        "fact": "test fact",
        "kind": "fact",
        "conv_id": "review_test",
        "session": 1,
        "entities": [],
        "tags": [],
        "agent_id": "agent_a",
        "swarm_id": "sw1",
        "scope": "swarm-shared",
        "owner_id": "agent:agent_a",
        "read": ["swarm:sw1"],
        "write": ["swarm:sw1"],
        "status": "active",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if overrides:
        base.update(overrides)
    return base


# ═══════════════════════════════════════════════════════════════
# 1. _git_dedup_index persistence
# ═══════════════════════════════════════════════════════════════

def test_git_dedup_index_round_trip(tmp_path):
    """_git_dedup_index must survive save → load cycle."""
    server = _make_server(tmp_path)
    server._git_dedup_index[("repo1", "file.py")] = {
        "blob_sha": "abc123",
        "artifact_id": "art_001",
    }
    server._save_cache()

    # Create new server to load from disk
    server2 = MemoryServer(data_dir=str(tmp_path), key="review_test")
    assert ("repo1", "file.py") in server2._git_dedup_index
    assert server2._git_dedup_index[("repo1", "file.py")]["blob_sha"] == "abc123"


# ═══════════════════════════════════════════════════════════════
# 2. _is_visible enforces TTL
# ═══════════════════════════════════════════════════════════════

def test_is_visible_ttl_expired():
    """Fact with expired TTL should not be visible."""
    past = (datetime.now(timezone.utc) - timedelta(seconds=100)).isoformat()
    fact = _make_fact({"retention_ttl": 60, "created_at": past})
    now = datetime.now(timezone.utc)
    assert not _is_visible(fact, now=now)


def test_is_visible_ttl_not_expired():
    """Fact with unexpired TTL should be visible."""
    recent = datetime.now(timezone.utc).isoformat()
    fact = _make_fact({"retention_ttl": 3600, "created_at": recent})
    now = datetime.now(timezone.utc)
    assert _is_visible(fact, now=now)


def test_is_visible_retracted():
    """Retracted fact should not be visible."""
    fact = _make_fact({"status": "retracted"})
    assert not _is_visible(fact)


def test_is_visible_no_ttl():
    """Fact without TTL should be visible (status=active)."""
    fact = _make_fact()
    assert _is_visible(fact)


# ═══════════════════════════════════════════════════════════════
# 3. Lifecycle ACL: edit, retract, purge
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_edit_requires_write_access(tmp_path):
    """edit() should deny if caller has no write access."""
    server = _make_server(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    server._all_granular = [
        _make_fact({
            "artifact_id": "art_001",
            "owner_id": "agent:agent_a",
            "write": [],  # no public write
        }),
    ]

    result = await server.edit(
        "art_001", "new content",
        caller_id="agent:agent_b", caller_role="user",
    )
    assert result.get("code") == "ACL_FORBIDDEN"


@pytest.mark.asyncio
async def test_edit_allowed_for_owner(tmp_path):
    """edit() should allow the owner."""
    server = _make_server(tmp_path)
    server._all_granular = [
        _make_fact({
            "artifact_id": "art_001",
            "owner_id": "agent:agent_a",
            "write": [],
        }),
    ]

    result = await server.edit(
        "art_001", "new content",
        caller_id="agent:agent_a", caller_role="user",
    )
    assert "artifact_id" in result
    assert result.get("code") is None


@pytest.mark.asyncio
async def test_edit_allowed_for_admin(tmp_path):
    """edit() should allow admin even without write ACL."""
    server = _make_server(tmp_path)
    server._all_granular = [
        _make_fact({
            "artifact_id": "art_001",
            "owner_id": "agent:agent_a",
            "write": [],
        }),
    ]

    result = await server.edit(
        "art_001", "new content",
        caller_id="agent:admin_user", caller_role="admin",
    )
    assert "artifact_id" in result


@pytest.mark.asyncio
async def test_retract_requires_write_access(tmp_path):
    """retract() should deny if caller has no write access."""
    server = _make_server(tmp_path)
    server._all_granular = [
        _make_fact({
            "artifact_id": "art_002",
            "owner_id": "agent:agent_a",
            "write": [],
        }),
    ]

    result = await server.retract(
        "art_002",
        caller_id="agent:agent_b", caller_role="user",
    )
    assert result.get("code") == "ACL_FORBIDDEN"


@pytest.mark.asyncio
async def test_retract_allowed_for_owner(tmp_path):
    """retract() should allow the owner."""
    server = _make_server(tmp_path)
    server._all_granular = [
        _make_fact({
            "artifact_id": "art_002",
            "owner_id": "agent:agent_a",
            "write": [],
        }),
    ]

    result = await server.retract(
        "art_002",
        caller_id="agent:agent_a", caller_role="user",
    )
    assert result.get("status") == "retracted"


@pytest.mark.asyncio
async def test_purge_requires_admin(tmp_path):
    """purge() should deny non-admin callers."""
    server = _make_server(tmp_path)
    server._all_granular = [
        _make_fact({"artifact_id": "art_003"}),
    ]

    result = await server.purge(
        "art_003",
        caller_id="agent:agent_a", caller_role="user",
    )
    assert result.get("code") == "ACL_FORBIDDEN"


@pytest.mark.asyncio
async def test_purge_allowed_for_admin(tmp_path):
    """purge() should work for admin."""
    server = _make_server(tmp_path)
    server._all_granular = [
        _make_fact({"artifact_id": "art_003"}),
    ]

    result = await server.purge(
        "art_003",
        caller_id="agent:admin", caller_role="admin",
    )
    assert "purged_facts" in result


# ═══════════════════════════════════════════════════════════════
# 4. ingest_document dedup per chunk
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_ingest_document_dedup_skips_same_content(tmp_path):
    """ingest_document should skip chunks with same (source_id, content_hash)."""
    server = _make_server(tmp_path)

    content = "This is test content for dedup checking."

    with patch("src.memory.extract_session", new_callable=AsyncMock) as mock_extract, \
         patch("src.memory.consolidate_session", new_callable=AsyncMock) as mock_cons, \
         patch("src.memory.cross_session_entity", new_callable=AsyncMock) as mock_cross:
        mock_extract.return_value = (
            "review_test", 1, "2024-01-01",
            [{"id": "f_001", "fact": "extracted fact", "session": 1, "entities": [], "tags": []}],
            [],
        )
        mock_cons.return_value = ("review_test", 1, "2024-01-01", [])
        mock_cross.return_value = ("review_test", "ent", [])

        # First ingest
        count1 = await server.ingest_document(
            content, source_id="doc_001",
            agent_id="a", swarm_id="sw1", scope="swarm-shared",
        )
        assert count1 > 0

        # Second ingest with same content — should skip
        count2 = await server.ingest_document(
            content, source_id="doc_001",
            agent_id="a", swarm_id="sw1", scope="swarm-shared",
        )
        assert count2 == 0  # all chunks skipped


@pytest.mark.asyncio
async def test_ingest_document_dedup_allows_different_content(tmp_path):
    """ingest_document should process chunks with different content."""
    server = _make_server(tmp_path)

    with patch("src.memory.extract_session", new_callable=AsyncMock) as mock_extract, \
         patch("src.memory.consolidate_session", new_callable=AsyncMock) as mock_cons, \
         patch("src.memory.cross_session_entity", new_callable=AsyncMock) as mock_cross:
        mock_extract.return_value = (
            "review_test", 1, "2024-01-01",
            [{"id": "f_001", "fact": "extracted fact", "session": 1, "entities": [], "tags": []}],
            [],
        )
        mock_cons.return_value = ("review_test", 1, "2024-01-01", [])
        mock_cross.return_value = ("review_test", "ent", [])

        count1 = await server.ingest_document(
            "content version 1", source_id="doc_002",
            agent_id="a", swarm_id="sw1", scope="swarm-shared",
        )
        assert count1 > 0

        count2 = await server.ingest_document(
            "content version 2 is different", source_id="doc_002",
            agent_id="a", swarm_id="sw1", scope="swarm-shared",
        )
        assert count2 > 0  # different content → processed


@pytest.mark.asyncio
async def test_ingest_document_skip_dedup_flag(tmp_path):
    """skip_dedup=True should bypass dedup check."""
    server = _make_server(tmp_path)
    content = "Repeated content for skip_dedup test."

    with patch("src.memory.extract_session", new_callable=AsyncMock) as mock_extract, \
         patch("src.memory.consolidate_session", new_callable=AsyncMock) as mock_cons, \
         patch("src.memory.cross_session_entity", new_callable=AsyncMock) as mock_cross:
        mock_extract.return_value = (
            "review_test", 1, "2024-01-01",
            [{"id": "f_001", "fact": "extracted", "session": 1, "entities": [], "tags": []}],
            [],
        )
        mock_cons.return_value = ("review_test", 1, "2024-01-01", [])
        mock_cross.return_value = ("review_test", "ent", [])

        await server.ingest_document(
            content, source_id="doc_003",
            agent_id="a", swarm_id="sw1", scope="swarm-shared",
        )

        count2 = await server.ingest_document(
            content, source_id="doc_003",
            agent_id="a", swarm_id="sw1", scope="swarm-shared",
            skip_dedup=True,
        )
        assert count2 > 0  # skip_dedup bypasses check


# ═══════════════════════════════════════════════════════════════
# 5. get_more_context TTL check
# ═══════════════════════════════════════════════════════════════

def test_get_more_context_ttl_expired():
    """get_more_context should not return expired session content."""
    from src.inference import get_more_context

    past = (datetime.now(timezone.utc) - timedelta(seconds=100)).isoformat()
    raw_sessions = [
        {"content": "secret info", "status": "active",
         "retention_ttl": 60, "stored_at": past},
    ]
    result = get_more_context(1, raw_sessions)
    assert "not found" in result["result"]


def test_get_more_context_ttl_not_expired():
    """get_more_context should return active, non-expired session content."""
    from src.inference import get_more_context

    recent = datetime.now(timezone.utc).isoformat()
    raw_sessions = [
        {"content": "valid info", "status": "active",
         "retention_ttl": 3600, "stored_at": recent},
    ]
    result = get_more_context(1, raw_sessions)
    assert "valid info" in result["result"]


def test_get_more_context_retracted_status():
    """get_more_context should not return retracted session content."""
    from src.inference import get_more_context

    raw_sessions = [
        {"content": "retracted info", "status": "retracted"},
    ]
    result = get_more_context(1, raw_sessions)
    assert "not found" in result["result"]


# ═══════════════════════════════════════════════════════════════
# 6. Backward-compat ACL derivation from scope
# ═══════════════════════════════════════════════════════════════

def test_acl_allows_backward_compat_agent_private(tmp_path):
    """Facts without owner_id should derive ACL from scope (agent-private)."""
    server = _make_server(tmp_path)
    fact = {
        "fact": "private data",
        "agent_id": "agent_x",
        "swarm_id": "sw1",
        "scope": "agent-private",
        # No owner_id, read, write fields
    }

    # Owner should have access
    assert server._acl_allows(fact, "agent:agent_x")
    # Other agent should not
    assert not server._acl_allows(fact, "agent:agent_b")


def test_acl_allows_backward_compat_swarm_shared(tmp_path):
    """Facts without owner_id should derive ACL from scope (swarm-shared)."""
    server = _make_server(tmp_path)
    fact = {
        "fact": "shared data",
        "agent_id": "agent_a",
        "swarm_id": "sw1",
        "scope": "swarm-shared",
        # No owner_id, read, write fields
    }

    # Swarm member should have access
    assert server._acl_allows(fact, "agent:agent_b",
                               caller_memberships=["swarm:sw1"])
    # Non-member should not
    assert not server._acl_allows(fact, "agent:agent_c",
                                   caller_memberships=["swarm:other"])


def test_acl_allows_backward_compat_system_wide(tmp_path):
    """Facts without owner_id should derive ACL from scope (system-wide)."""
    server = _make_server(tmp_path)
    fact = {
        "fact": "public data",
        "agent_id": "default",
        "swarm_id": "default",
        "scope": "system-wide",
        # No owner_id, read, write fields
    }

    # Anyone should have access (agent:PUBLIC in read)
    assert server._acl_allows(fact, "agent:random_agent")


# ═══════════════════════════════════════════════════════════════
# Courier _is_visible integration
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_courier_skips_retracted_facts(tmp_path):
    """Courier _poll should not deliver retracted facts."""
    from src.courier import Courier

    server = _make_server(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    server._all_granular = [
        _make_fact({
            "status": "retracted",
            "created_at": now,
            "owner_id": "system",
            "read": ["agent:PUBLIC"],
        }),
    ]
    server._save_cache()

    courier = Courier(server)
    courier._last_seen_at = "1970-01-01T00:00:00+00:00"

    delivered = []
    async def _cb(fact):
        delivered.append(fact)

    await courier.subscribe(filter={}, callback=_cb)
    count = await courier._poll()
    assert count == 0
    assert len(delivered) == 0


@pytest.mark.asyncio
async def test_courier_skips_expired_ttl_facts(tmp_path):
    """Courier _poll should not deliver TTL-expired facts."""
    from src.courier import Courier

    server = _make_server(tmp_path)
    past = (datetime.now(timezone.utc) - timedelta(seconds=100)).isoformat()
    server._all_granular = [
        _make_fact({
            "retention_ttl": 60,
            "created_at": past,
            "owner_id": "system",
            "read": ["agent:PUBLIC"],
        }),
    ]
    server._save_cache()

    courier = Courier(server)
    courier._last_seen_at = "1970-01-01T00:00:00+00:00"

    delivered = []
    async def _cb(fact):
        delivered.append(fact)

    await courier.subscribe(filter={}, callback=_cb)
    count = await courier._poll()
    assert count == 0
    assert len(delivered) == 0
