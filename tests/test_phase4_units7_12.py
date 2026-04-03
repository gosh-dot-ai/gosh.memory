"""Tests for Phase 4 — Units 7-12."""

import asyncio
import json
import os
import time

import numpy as np
import pytest

from src.identity import _generate_artifact_id, _generate_version_id, content_hash_text
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
        return ("test_conv", sn, "2024-06-01", facts, [])

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


# ═══════════════════════════════════════════════════════════════
# Unit 7: Redaction
# ═══════════════════════════════════════════════════════════════

class TestUnit7Redaction:

    @pytest.mark.asyncio
    async def test_redact_replaces_fact_text(self, tmp_path, monkeypatch):
        _patch_extraction(monkeypatch)
        _patch_embeddings(monkeypatch)
        server = MemoryServer(data_dir=str(tmp_path), key="redact_test")
        result = await server.store("Secret data", 1, "2024-06-01")
        art_id = server._all_granular[0].get("artifact_id")

        r = await server.redact(art_id, ["fact", "entities", "content"],
                                caller_role="admin")
        assert r["redacted"] is True
        assert r["artifact_id"] == art_id

        # Fact text replaced
        for f in server._all_granular:
            if f.get("artifact_id") == art_id:
                assert f["fact"] == "[REDACTED]"
                assert f["entities"] == ["[REDACTED]"]
                assert f["status"] == "redacted"

    @pytest.mark.asyncio
    async def test_redacted_fact_invisible(self, tmp_path, monkeypatch):
        _patch_extraction(monkeypatch)
        _patch_embeddings(monkeypatch)
        server = MemoryServer(data_dir=str(tmp_path), key="redact_vis")
        await server.store("Data", 1, "2024-06-01")
        art_id = server._all_granular[0].get("artifact_id")

        await server.redact(art_id, ["fact"], caller_role="admin")

        for f in server._all_granular:
            if f.get("artifact_id") == art_id:
                assert not _is_visible(f)

    @pytest.mark.asyncio
    async def test_redact_raw_session_content(self, tmp_path, monkeypatch):
        _patch_extraction(monkeypatch)
        _patch_embeddings(monkeypatch)
        server = MemoryServer(data_dir=str(tmp_path), key="redact_raw")
        await server.store("Secret raw data", 1, "2024-06-01")
        art_id = server._all_granular[0].get("artifact_id")

        await server.redact(art_id, ["content"], caller_role="admin")

        for rs in server._raw_sessions:
            if rs.get("artifact_id") == art_id:
                assert rs["content"] == "[REDACTED]"

    @pytest.mark.asyncio
    async def test_redact_persists_to_disk(self, tmp_path, monkeypatch):
        _patch_extraction(monkeypatch)
        _patch_embeddings(monkeypatch)
        server = MemoryServer(data_dir=str(tmp_path), key="redact_disk")
        await server.store("Persist test", 1, "2024-06-01")
        art_id = server._all_granular[0].get("artifact_id")

        await server.redact(art_id, ["fact"], caller_role="admin")

        # Reload from disk
        server2 = MemoryServer(data_dir=str(tmp_path), key="redact_disk")
        for f in server2._all_granular:
            if f.get("artifact_id") == art_id:
                assert f["fact"] == "[REDACTED]"
                assert f["status"] == "redacted"

    @pytest.mark.asyncio
    async def test_redact_acl_check(self, tmp_path, monkeypatch):
        _patch_extraction(monkeypatch)
        _patch_embeddings(monkeypatch)
        server = MemoryServer(data_dir=str(tmp_path), key="redact_acl")
        await server.store("Data", 1, "2024-06-01",
                          owner_id="agent:alice", write=["agent:alice"])
        art_id = server._all_granular[0].get("artifact_id")

        # Non-owner, non-admin should be denied
        r = await server.redact(art_id, ["fact"],
                                caller_id="agent:bob", caller_role="user")
        assert r.get("code") == "ACL_FORBIDDEN"

    @pytest.mark.asyncio
    async def test_redact_not_found(self, tmp_path, monkeypatch):
        _patch_extraction(monkeypatch)
        server = MemoryServer(data_dir=str(tmp_path), key="redact_nf")
        r = await server.redact("art_nonexist", ["fact"], caller_role="admin")
        assert r.get("code") == "NOT_FOUND"


# ═══════════════════════════════════════════════════════════════
# Unit 8: Retention TTL
# ═══════════════════════════════════════════════════════════════

class TestUnit8RetentionTTL:

    @pytest.mark.asyncio
    async def test_store_with_ttl_propagates(self, tmp_path, monkeypatch):
        _patch_extraction(monkeypatch)
        _patch_embeddings(monkeypatch)
        server = MemoryServer(data_dir=str(tmp_path), key="ttl_test")
        await server.store("Data", 1, "2024-06-01", retention_ttl=3600)

        for f in server._all_granular:
            assert f.get("retention_ttl") == 3600
        for rs in server._raw_sessions:
            assert rs.get("retention_ttl") == 3600

    @pytest.mark.asyncio
    async def test_ttl_expired_invisible(self, tmp_path, monkeypatch):
        _patch_extraction(monkeypatch)
        _patch_embeddings(monkeypatch)
        server = MemoryServer(data_dir=str(tmp_path), key="ttl_exp")
        await server.store("Data", 1, "2024-06-01", retention_ttl=1)

        # Wait for expiry
        time.sleep(2)

        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        for f in server._all_granular:
            assert not _is_visible(f, now=now)

    @pytest.mark.asyncio
    async def test_ttl_none_never_expires(self, tmp_path, monkeypatch):
        _patch_extraction(monkeypatch)
        _patch_embeddings(monkeypatch)
        server = MemoryServer(data_dir=str(tmp_path), key="ttl_none")
        await server.store("Data", 1, "2024-06-01")

        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        for f in server._all_granular:
            assert _is_visible(f, now=now)


# ═══════════════════════════════════════════════════════════════
# Unit 9: Audit logging
# ═══════════════════════════════════════════════════════════════

class TestUnit9AuditLog:

    @pytest.mark.asyncio
    async def test_store_and_recall_create_audit_entries(self, tmp_path, monkeypatch):
        _patch_extraction(monkeypatch)
        _patch_embeddings(monkeypatch)
        server = MemoryServer(data_dir=str(tmp_path), key="audit_test")
        await server.store("Data", 1, "2024-06-01")
        await server.build_index()
        await server.recall("test query")

        log_file = tmp_path / "audit" / "audit.jsonl"
        assert log_file.exists()
        lines = log_file.read_text().strip().split("\n")
        events = [json.loads(line) for line in lines]
        event_names = [e["event"] for e in events]
        assert "store" in event_names
        assert "recall" in event_names

    @pytest.mark.asyncio
    async def test_retract_creates_audit_entry(self, tmp_path, monkeypatch):
        _patch_extraction(monkeypatch)
        _patch_embeddings(monkeypatch)
        server = MemoryServer(data_dir=str(tmp_path), key="audit_retract")
        await server.store("Data", 1, "2024-06-01")
        art_id = server._all_granular[0].get("artifact_id")
        await server.retract(art_id, caller_role="admin")

        log_file = tmp_path / "audit" / "audit.jsonl"
        lines = log_file.read_text().strip().split("\n")
        events = [json.loads(line) for line in lines]
        event_names = [e["event"] for e in events]
        assert "retract" in event_names

    @pytest.mark.asyncio
    async def test_audit_entry_has_timestamp(self, tmp_path, monkeypatch):
        _patch_extraction(monkeypatch)
        _patch_embeddings(monkeypatch)
        server = MemoryServer(data_dir=str(tmp_path), key="audit_ts")
        await server.store("Data", 1, "2024-06-01")

        log_file = tmp_path / "audit" / "audit.jsonl"
        lines = log_file.read_text().strip().split("\n")
        entry = json.loads(lines[0])
        assert "timestamp" in entry
        assert "caller_id" in entry


# ═══════════════════════════════════════════════════════════════
# Unit 10: Encryption at rest
# ═══════════════════════════════════════════════════════════════

class TestUnit10Encryption:

    def test_encrypt_decrypt_roundtrip(self, tmp_path):
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            pytest.skip("cryptography package not installed")

        from src.storage import JSONNPZStorage
        key = os.urandom(32)
        s = JSONNPZStorage(str(tmp_path), "enc_test", encryption_key=key)
        data = {"granular": [{"fact": "secret"}], "cons": [], "cross": []}
        s.save_facts(data)

        # File starts with MAGIC bytes
        raw = (tmp_path / "enc_test.json").read_bytes()
        assert raw[:4] == b"GME1"

        # Load decrypts correctly
        loaded = s.load_facts()
        assert loaded["granular"][0]["fact"] == "secret"

    def test_no_key_plaintext(self, tmp_path):
        from src.storage import JSONNPZStorage
        s = JSONNPZStorage(str(tmp_path), "plain_test")
        data = {"granular": [{"fact": "visible"}]}
        s.save_facts(data)

        raw = (tmp_path / "plain_test.json").read_bytes()
        assert not raw.startswith(b"GME1")
        assert b"visible" in raw

    def test_encrypted_without_key_errors(self, tmp_path):
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            pytest.skip("cryptography package not installed")

        from src.storage import JSONNPZStorage
        key = os.urandom(32)
        s_enc = JSONNPZStorage(str(tmp_path), "err_test", encryption_key=key)
        s_enc.save_facts({"granular": [{"fact": "secret"}]})

        # Try to load without key — should raise
        s_plain = JSONNPZStorage(str(tmp_path), "err_test")
        with pytest.raises(Exception):
            s_plain.load_facts()

    def test_encrypted_embeddings(self, tmp_path):
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            pytest.skip("cryptography package not installed")

        from src.storage import JSONNPZStorage
        key = os.urandom(32)
        s = JSONNPZStorage(str(tmp_path), "emb_enc", encryption_key=key)
        g = np.random.randn(5, DIM).astype(np.float32)
        c = np.random.randn(3, DIM).astype(np.float32)
        x = np.random.randn(2, DIM).astype(np.float32)
        s.save_embeddings(g, c, x)

        loaded = s.load_embeddings()
        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded["gran"], g)


# ═══════════════════════════════════════════════════════════════
# Unit 11: Git ingest with identity + dedup
# ═══════════════════════════════════════════════════════════════

class TestUnit11GitIngestIdentity:

    def test_import_git_returns_extended_fields(self, tmp_path, monkeypatch):
        """import_git returns sessions with artifact_path, blob_sha, etc."""
        from src.git_importer import import_git

        # Create a minimal repo
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        (repo_dir / "README.md").write_text("# Hello\nWorld")
        import subprocess
        subprocess.run(["git", "init", str(repo_dir)], capture_output=True)
        subprocess.run(["git", "-C", str(repo_dir), "add", "."], capture_output=True)
        subprocess.run(["git", "-C", str(repo_dir), "commit", "-m", "init"],
                       capture_output=True, env={**os.environ,
                       "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t",
                       "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t"})

        sessions = import_git(str(repo_dir))
        assert len(sessions) >= 1
        s = sessions[0]
        assert "artifact_path" in s
        assert "blob_sha" in s
        assert s["storage_mode"] == "inline"
        assert s["source_id"] == str(repo_dir)

    @pytest.mark.asyncio
    async def test_git_dedup_skip_unchanged(self, tmp_path, monkeypatch):
        """Re-import unchanged git repo skips all sessions."""
        _patch_extraction(monkeypatch)
        _patch_embeddings(monkeypatch)
        from src.git_importer import import_git

        repo_dir = tmp_path / "repo2"
        repo_dir.mkdir()
        (repo_dir / "test.md").write_text("# Test content")
        import subprocess
        subprocess.run(["git", "init", str(repo_dir)], capture_output=True)
        subprocess.run(["git", "-C", str(repo_dir), "add", "."], capture_output=True)
        subprocess.run(["git", "-C", str(repo_dir), "commit", "-m", "init"],
                       capture_output=True, env={**os.environ,
                       "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t",
                       "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t"})

        server = MemoryServer(data_dir=str(tmp_path / "data"), key="git_dedup")

        sessions = import_git(str(repo_dir))
        # First import
        for s in sessions:
            dedup_key = (s["source_id"], s["artifact_path"])
            server._git_dedup_index[dedup_key] = {
                "blob_sha": s["blob_sha"],
                "artifact_id": _generate_artifact_id(),
            }

        # Second import — all should match
        sessions2 = import_git(str(repo_dir))
        skipped = 0
        for s in sessions2:
            dedup_key = (s["source_id"], s["artifact_path"])
            existing = server._git_dedup_index.get(dedup_key)
            if existing and existing["blob_sha"] == s["blob_sha"]:
                skipped += 1
        assert skipped == len(sessions2)


# ═══════════════════════════════════════════════════════════════
# Unit 12: Reference mode degradation
# ═══════════════════════════════════════════════════════════════

class TestUnit12ReferenceDegradation:

    def test_get_more_context_reference_empty(self):
        from src.inference import get_more_context
        raw_sessions = [
            {"content": "", "storage_mode": "reference", "status": "active"},
        ]
        result = get_more_context(1, raw_sessions=raw_sessions)
        assert "unavailable" in result["result"].lower() or "reference" in result["result"].lower()

    def test_get_more_context_inline_normal(self):
        from src.inference import get_more_context
        raw_sessions = [
            {"content": "Hello world", "storage_mode": "inline", "status": "active"},
        ]
        result = get_more_context(1, raw_sessions=raw_sessions)
        assert "Hello world" in result["result"]
