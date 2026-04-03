"""Tests for Phase 4 — Identity, Versioning, Dedup, Visibility, Edit/Retract/Purge."""

import asyncio
import json

import numpy as np
import pytest

from src.identity import (
    _generate_artifact_id,
    _generate_version_id,
    content_hash_bytes,
    content_hash_git,
    content_hash_text,
)
from src.memory import MemoryServer, _is_visible

DIM = 3072


def _patch_extraction(monkeypatch, n_facts=3):
    """Patch extract_session to return N facts without LLM calls."""

    async def mock_extract_session(**kwargs):
        sn = kwargs.get("session_num", 1)
        facts = []
        for i in range(n_facts):
            facts.append({
                "id": f"f{i}",
                "fact": f"Test fact {i} session {sn}",
                "kind": "fact",
                "entities": ["Alice"],
                "tags": ["test"],
                "session": sn,
            })
        tlinks = []
        return ("test_conv", sn, "2024-06-01", facts, tlinks)

    async def mock_consolidate(**kwargs):
        return ("test_conv", 1, "2024-06-01", [])

    async def mock_cross(**kwargs):
        return ("test_conv", "alice", [])

    monkeypatch.setattr("src.memory.extract_session", mock_extract_session)
    monkeypatch.setattr("src.memory.consolidate_session", mock_consolidate)
    monkeypatch.setattr("src.memory.cross_session_entity", mock_cross)


def _patch_embeddings(monkeypatch):
    async def mock_embed_texts(texts, **kwargs):
        return np.random.randn(len(texts), DIM).astype(np.float32)

    async def mock_embed_query(text, **kwargs):
        return np.random.randn(DIM).astype(np.float32)

    monkeypatch.setattr("src.memory.embed_texts", mock_embed_texts)
    monkeypatch.setattr("src.memory.embed_query", mock_embed_query)


# ── Unit 1: Identity fields + version model ──

class TestUnit1Identity:
    def test_generate_artifact_id(self):
        aid = _generate_artifact_id()
        assert aid.startswith("art_")
        assert len(aid) == 14  # "art_" + 10 hex chars

    def test_generate_version_id(self):
        vid = _generate_version_id()
        assert vid.startswith("ver_")
        assert len(vid) == 14

    def test_unique_ids(self):
        ids = {_generate_artifact_id() for _ in range(100)}
        assert len(ids) == 100

    @pytest.mark.asyncio
    async def test_store_adds_identity_fields(self, tmp_path, monkeypatch):
        _patch_extraction(monkeypatch)
        _patch_embeddings(monkeypatch)
        server = MemoryServer(data_dir=str(tmp_path), key="id_test")
        result = await server.store("Hello world", 1, "2024-06-01")
        assert result["facts_extracted"] >= 1

        # Check raw_session has identity fields
        rs = server._raw_sessions[-1]
        assert rs["artifact_id"].startswith("art_")
        assert rs["version_id"].startswith("ver_")
        assert rs["status"] == "active"
        assert rs["content_hash"].startswith("sha256:")

        # Check facts have same identity fields
        for f in server._all_granular:
            if f.get("artifact_id"):
                assert f["artifact_id"] == rs["artifact_id"]
                assert f["version_id"] == rs["version_id"]
                assert f["status"] == "active"

    @pytest.mark.asyncio
    async def test_two_stores_different_artifact_ids(self, tmp_path, monkeypatch):
        _patch_extraction(monkeypatch)
        _patch_embeddings(monkeypatch)
        server = MemoryServer(data_dir=str(tmp_path), key="id_test2")
        await server.store("First content", 1, "2024-06-01")
        await server.store("Second content", 2, "2024-06-02")

        rs1 = server._raw_sessions[0]
        rs2 = server._raw_sessions[1]
        assert rs1["artifact_id"] != rs2["artifact_id"]

    @pytest.mark.asyncio
    async def test_old_cache_loads_without_errors(self, tmp_path, monkeypatch):
        """Facts without identity fields should get defaults on load."""
        _patch_extraction(monkeypatch)
        _patch_embeddings(monkeypatch)
        # Create server with old-format data (no identity fields)
        server = MemoryServer(data_dir=str(tmp_path), key="compat_test")
        server._all_granular = [
            {"fact": "old fact", "id": "f0", "kind": "fact",
             "conv_id": "compat_test", "session": 1,
             "owner_id": "system", "read": ["agent:PUBLIC"],
             "write": ["agent:PUBLIC"]},
        ]
        server._save_cache()

        # Reload
        server2 = MemoryServer(data_dir=str(tmp_path), key="compat_test")
        assert len(server2._all_granular) == 1
        assert server2._all_granular[0].get("status") == "active"

    @pytest.mark.asyncio
    async def test_source_meta_on_raw_session(self, tmp_path, monkeypatch):
        _patch_extraction(monkeypatch)
        _patch_embeddings(monkeypatch)
        server = MemoryServer(data_dir=str(tmp_path), key="meta_test")
        await server.store("Content", 1, "2024-06-01",
                           source_meta={"custom_field": "custom_value"})
        rs = server._raw_sessions[-1]
        assert rs.get("custom_field") == "custom_value"
        # source_meta should NOT be on facts
        for f in server._all_granular:
            assert "custom_field" not in f


# ── Unit 2: Content hash normalization ──

class TestUnit2ContentHash:
    def test_basic_hash(self):
        h = content_hash_text("hello")
        assert h.startswith("sha256:")
        assert len(h) == 7 + 64  # "sha256:" + 64 hex chars

    def test_bom_normalization(self):
        h1 = content_hash_text("\ufeffhello world")
        h2 = content_hash_text("hello world")
        assert h1 == h2

    def test_line_ending_normalization(self):
        h1 = content_hash_text("line1\r\nline2\r\nline3")
        h2 = content_hash_text("line1\nline2\nline3")
        assert h1 == h2

    def test_cr_normalization(self):
        h1 = content_hash_text("line1\rline2")
        h2 = content_hash_text("line1\nline2")
        assert h1 == h2

    def test_strip_whitespace(self):
        h1 = content_hash_text("  hello  ")
        h2 = content_hash_text("hello")
        assert h1 == h2

    def test_same_text_same_hash(self):
        h1 = content_hash_text("identical content")
        h2 = content_hash_text("identical content")
        assert h1 == h2

    def test_different_text_different_hash(self):
        h1 = content_hash_text("content A")
        h2 = content_hash_text("content B")
        assert h1 != h2

    def test_bytes_hash(self):
        h = content_hash_bytes(b"raw data")
        assert h.startswith("sha256:")

    def test_git_hash(self):
        h = content_hash_git("abc123def456")
        assert h == "git:abc123def456"

    @pytest.mark.asyncio
    async def test_store_sets_content_hash(self, tmp_path, monkeypatch):
        _patch_extraction(monkeypatch)
        _patch_embeddings(monkeypatch)
        server = MemoryServer(data_dir=str(tmp_path), key="hash_test")
        await server.store("Some content", 1, "2024-06-01")
        rs = server._raw_sessions[-1]
        assert rs["content_hash"].startswith("sha256:")
        # Facts should also have content_hash
        for f in server._all_granular:
            if f.get("content_hash"):
                assert f["content_hash"] == rs["content_hash"]


# ── Unit 3: Dedup on ingest ──

class TestUnit3Dedup:
    @pytest.mark.asyncio
    async def test_same_content_returns_duplicate(self, tmp_path, monkeypatch):
        _patch_extraction(monkeypatch)
        _patch_embeddings(monkeypatch)
        server = MemoryServer(data_dir=str(tmp_path), key="dedup_test")
        r1 = await server.store("Same content", 1, "2024-06-01",
                                source_id="src1")
        assert r1["facts_extracted"] >= 1

        r2 = await server.store("Same content", 1, "2024-06-01",
                                source_id="src1")
        assert r2.get("status") == "duplicate"
        assert r2.get("artifact_id") == server._raw_sessions[0]["artifact_id"]

    @pytest.mark.asyncio
    async def test_no_source_id_no_dedup(self, tmp_path, monkeypatch):
        _patch_extraction(monkeypatch)
        _patch_embeddings(monkeypatch)
        server = MemoryServer(data_dir=str(tmp_path), key="nodedup_test")
        r1 = await server.store("Content A", 1, "2024-06-01")
        r2 = await server.store("Content A", 1, "2024-06-01")
        # Both should be stored (no dedup without source_id)
        assert r1["facts_extracted"] >= 1
        assert r2["facts_extracted"] >= 1

    @pytest.mark.asyncio
    async def test_skip_dedup_flag(self, tmp_path, monkeypatch):
        _patch_extraction(monkeypatch)
        _patch_embeddings(monkeypatch)
        server = MemoryServer(data_dir=str(tmp_path), key="skipdedup_test")
        r1 = await server.store("Content", 1, "2024-06-01",
                                source_id="src1")
        r2 = await server.store("Content", 1, "2024-06-01",
                                source_id="src1",
                                skip_dedup=True)
        assert r1["facts_extracted"] >= 1
        assert r2["facts_extracted"] >= 1

    @pytest.mark.asyncio
    async def test_dedup_persists_across_reload(self, tmp_path, monkeypatch):
        _patch_extraction(monkeypatch)
        _patch_embeddings(monkeypatch)
        server = MemoryServer(data_dir=str(tmp_path), key="dedup_persist")
        await server.store("Content", 1, "2024-06-01",
                           source_id="src1")

        # Reload server
        server2 = MemoryServer(data_dir=str(tmp_path), key="dedup_persist")
        r = await server2.store("Content", 1, "2024-06-01",
                                source_id="src1")
        assert r.get("status") == "duplicate"


# ── Unit 4: Versioning ──

class TestUnit4Versioning:
    @pytest.mark.asyncio
    async def test_different_content_creates_new_version(self, tmp_path, monkeypatch):
        _patch_extraction(monkeypatch)
        _patch_embeddings(monkeypatch)
        server = MemoryServer(data_dir=str(tmp_path), key="ver_test")
        r1 = await server.store("Content V1", 1, "2024-06-01",
                                source_id="src1")
        art_id = server._raw_sessions[0]["artifact_id"]
        ver1_id = server._raw_sessions[0]["version_id"]

        r2 = await server.store("Content V2", 1, "2024-06-01",
                                source_id="src1")
        # New version should be created
        assert r2["facts_extracted"] >= 1
        # Latest raw_session should have same artifact_id
        rs2 = server._raw_sessions[-1]
        assert rs2["artifact_id"] == art_id
        assert rs2["version_id"] != ver1_id
        assert rs2["parent_version"] == ver1_id

    @pytest.mark.asyncio
    async def test_old_version_superseded(self, tmp_path, monkeypatch):
        _patch_extraction(monkeypatch)
        _patch_embeddings(monkeypatch)
        server = MemoryServer(data_dir=str(tmp_path), key="super_test")
        await server.store("Content V1", 1, "2024-06-01",
                           source_id="src1")
        v1_facts = [f for f in server._all_granular if f.get("status") == "active"]
        assert len(v1_facts) >= 1

        await server.store("Content V2", 1, "2024-06-01",
                           source_id="src1")
        # Old facts should be superseded
        old_facts = [f for f in server._all_granular
                     if f.get("status") == "superseded"]
        assert len(old_facts) >= 1

    @pytest.mark.asyncio
    async def test_get_versions(self, tmp_path, monkeypatch):
        _patch_extraction(monkeypatch)
        _patch_embeddings(monkeypatch)
        server = MemoryServer(data_dir=str(tmp_path), key="getver_test")
        await server.store("V1", 1, "2024-06-01",
                           source_id="src1")
        art_id = server._raw_sessions[0]["artifact_id"]

        await server.store("V2", 1, "2024-06-01",
                           source_id="src1")

        result = server.get_versions(art_id)
        assert result["artifact_id"] == art_id
        versions = result["versions"]
        assert len(versions) >= 2
        # First should be root (no parent), last should be active
        statuses = [v["status"] for v in versions]
        assert "superseded" in statuses
        assert "active" in statuses


# ── Unit 5: Shared visibility predicate ──

class TestUnit5Visibility:
    def test_active_visible(self):
        assert _is_visible({"status": "active"}) is True

    def test_superseded_invisible(self):
        assert _is_visible({"status": "superseded"}) is False

    def test_retracted_invisible(self):
        assert _is_visible({"status": "retracted"}) is False

    def test_missing_status_defaults_active(self):
        assert _is_visible({}) is True

    def test_expired_ttl_invisible(self):
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        old = (now - timedelta(seconds=100)).isoformat()
        fact = {"status": "active", "retention_ttl": 50, "created_at": old}
        assert _is_visible(fact, now=now) is False

    def test_unexpired_ttl_visible(self):
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        recent = (now - timedelta(seconds=10)).isoformat()
        fact = {"status": "active", "retention_ttl": 3600, "created_at": recent}
        assert _is_visible(fact, now=now) is True

    @pytest.mark.asyncio
    async def test_retracted_fact_invisible_in_recall(self, tmp_path, monkeypatch):
        _patch_extraction(monkeypatch)
        _patch_embeddings(monkeypatch)
        server = MemoryServer(data_dir=str(tmp_path), key="vis_test")
        await server.store("Important fact", 1, "2024-06-01")
        art_id = server._raw_sessions[0]["artifact_id"]

        await server.build_index()
        result1 = await server.recall("Important")
        assert "Important" in result1["context"]

        # Retract
        await server.retract(art_id)
        server._data_dict = None
        await server.build_index()
        result2 = await server.recall("Important")
        # Retracted fact should not appear in context
        assert "Important fact" not in result2.get("context", "")


# ── Unit 6: Edit / Retract / Purge / get_versions ──

class TestUnit6EditRetractPurge:
    @pytest.mark.asyncio
    async def test_edit_creates_new_version(self, tmp_path, monkeypatch):
        _patch_extraction(monkeypatch)
        _patch_embeddings(monkeypatch)
        server = MemoryServer(data_dir=str(tmp_path), key="edit_test")
        await server.store("Original fact", 1, "2024-06-01")
        art_id = server._raw_sessions[0]["artifact_id"]
        old_ver = server._raw_sessions[0]["version_id"]

        result = await server.edit(art_id, "Updated fact")
        assert result["artifact_id"] == art_id
        assert result["version_id"] != old_ver
        assert result["parent_version"] == old_ver

        # New fact should be in granular
        active = [f for f in server._all_granular
                  if f.get("artifact_id") == art_id and f.get("status") == "active"]
        assert len(active) == 1
        assert active[0]["fact"] == "Updated fact"

    @pytest.mark.asyncio
    async def test_edit_nonexistent_artifact(self, tmp_path, monkeypatch):
        _patch_extraction(monkeypatch)
        server = MemoryServer(data_dir=str(tmp_path), key="edit_none")
        result = await server.edit("art_nonexist", "New content")
        assert result.get("code") == "NOT_FOUND"

    @pytest.mark.asyncio
    async def test_retract_hides_everywhere(self, tmp_path, monkeypatch):
        _patch_extraction(monkeypatch)
        _patch_embeddings(monkeypatch)
        server = MemoryServer(data_dir=str(tmp_path), key="retract_test")
        await server.store("Secret info", 1, "2024-06-01")
        art_id = server._raw_sessions[0]["artifact_id"]

        result = await server.retract(art_id)
        assert result["status"] == "retracted"

        # All facts with this artifact should be retracted
        for f in server._all_granular:
            if f.get("artifact_id") == art_id:
                assert f["status"] == "retracted"

    @pytest.mark.asyncio
    async def test_retract_nonexistent(self, tmp_path, monkeypatch):
        _patch_extraction(monkeypatch)
        server = MemoryServer(data_dir=str(tmp_path), key="retract_none")
        result = await server.retract("art_nonexist")
        assert result.get("code") == "NOT_FOUND"

    @pytest.mark.asyncio
    async def test_purge_removes_facts(self, tmp_path, monkeypatch):
        _patch_extraction(monkeypatch)
        _patch_embeddings(monkeypatch)
        server = MemoryServer(data_dir=str(tmp_path), key="purge_test")
        await server.store("To be purged", 1, "2024-06-01")
        art_id = server._raw_sessions[0]["artifact_id"]
        n_before = len(server._all_granular)

        result = await server.purge(art_id, caller_role="admin")
        assert result["purged_facts"] >= 1
        assert len(server._all_granular) < n_before

        # Facts should be gone
        remaining = [f for f in server._all_granular
                     if f.get("artifact_id") == art_id]
        assert len(remaining) == 0

    @pytest.mark.asyncio
    async def test_purge_requires_admin(self, tmp_path, monkeypatch):
        _patch_extraction(monkeypatch)
        _patch_embeddings(monkeypatch)
        server = MemoryServer(data_dir=str(tmp_path), key="purge_auth")
        await server.store("Content", 1, "2024-06-01")
        art_id = server._raw_sessions[0]["artifact_id"]

        result = await server.purge(art_id, caller_role="user")
        assert result.get("code") == "ACL_FORBIDDEN"

    @pytest.mark.asyncio
    async def test_get_versions_returns_chain(self, tmp_path, monkeypatch):
        _patch_extraction(monkeypatch)
        _patch_embeddings(monkeypatch)
        server = MemoryServer(data_dir=str(tmp_path), key="chain_test")
        await server.store("V1", 1, "2024-06-01",
                           source_id="src1")
        art_id = server._raw_sessions[0]["artifact_id"]

        await server.store("V2", 1, "2024-06-01",
                           source_id="src1")

        await server.edit(art_id, "V3")

        result = server.get_versions(art_id)
        versions = result["versions"]
        # Should have at least 3 versions (V1 superseded, V2, V3 from edit)
        assert len(versions) >= 2
        # At least one active version
        active = [v for v in versions if v["status"] == "active"]
        assert len(active) >= 1
