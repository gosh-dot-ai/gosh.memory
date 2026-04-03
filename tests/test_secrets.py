"""Tests for Unit 1 — Secrets (memory_store_secret, memory_get_secret)."""

import json

import numpy as np
import pytest

from src.memory import MemoryServer

DIM = 3072


@pytest.fixture(autouse=True)
def _patch_embed(monkeypatch):
    async def _aembed_texts(texts, **kw):
        return np.random.randn(len(texts), DIM).astype(np.float32)
    async def _aembed_query(text, **kw):
        return np.random.randn(DIM).astype(np.float32)
    monkeypatch.setattr("src.memory.embed_texts", _aembed_texts)
    monkeypatch.setattr("src.memory.embed_query", _aembed_query)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda f, l: None)


def test_store_and_get_secret_system_wide(tmp_path):
    server = MemoryServer(data_dir=str(tmp_path), key="sec_test")
    server.store_secret("API_KEY", "sk-123", "admin", "sys", "system-wide")
    result = server.get_secret("API_KEY", "any_agent", "any_swarm")
    assert result["value"] == "sk-123"


def test_store_and_get_secret_swarm_shared(tmp_path):
    server = MemoryServer(data_dir=str(tmp_path), key="sec_swarm")
    server.store_secret("DB_PASS", "hunter2", "admin", "sw1", "swarm-shared")
    # Same swarm — ok
    assert server.get_secret("DB_PASS", "agent_x", "sw1")["value"] == "hunter2"
    # Different swarm — forbidden
    result = server.get_secret("DB_PASS", "agent_x", "sw2")
    assert result["code"] == "SECRET_FORBIDDEN"


def test_store_and_get_secret_agent_private(tmp_path):
    server = MemoryServer(data_dir=str(tmp_path), key="sec_priv")
    server.store_secret("MY_KEY", "abc", "agent_a", "sw1", "agent-private")
    # Same agent — ok
    assert server.get_secret("MY_KEY", "agent_a", "sw1")["value"] == "abc"
    # Different agent — forbidden
    assert server.get_secret("MY_KEY", "agent_b", "sw1")["code"] == "SECRET_FORBIDDEN"


def test_secret_upsert_by_name(tmp_path):
    server = MemoryServer(data_dir=str(tmp_path), key="sec_upsert")
    server.store_secret("KEY", "v1", "a", "sw", "system-wide")
    server.store_secret("KEY", "v2", "a", "sw", "system-wide")
    assert len(server._secrets) == 1
    assert server.get_secret("KEY", "a", "sw")["value"] == "v2"


def test_secret_not_found(tmp_path):
    server = MemoryServer(data_dir=str(tmp_path), key="sec_miss")
    result = server.get_secret("MISSING", "a", "sw")
    assert result["code"] == "SECRET_NOT_FOUND"


def test_secrets_persist_cache_roundtrip(tmp_path):
    server = MemoryServer(data_dir=str(tmp_path), key="sec_cache")
    server.store_secret("RELOAD_KEY", "xyz", "a", "sw", "system-wide")
    server2 = MemoryServer(data_dir=str(tmp_path), key="sec_cache")
    assert server2.get_secret("RELOAD_KEY", "a", "sw")["value"] == "xyz"


@pytest.mark.asyncio
async def test_secrets_not_in_recall(tmp_path):
    """Secrets must never appear in memory_recall results."""
    server = MemoryServer(data_dir=str(tmp_path), key="sec_recall")
    server.store_secret("SECRET_KEY", "leak-me", "a", "sw", "system-wide")
    server._all_granular = [{
        "fact": "User lives in Seattle.", "kind": "fact",
        "id": "af_001", "conv_id": "sec_recall",
        "agent_id": "a", "swarm_id": "sw", "scope": "swarm-shared",
        "created_at": "2024-01-01T00:00:00+00:00",
    }]
    server._all_cons = []
    server._all_cross = []
    await server.build_index()
    result = await server.recall("SECRET_KEY API key credentials")
    assert "leak-me" not in result["context"]


def test_stats_includes_secrets_count(tmp_path):
    server = MemoryServer(data_dir=str(tmp_path), key="sec_stats")
    server.store_secret("K1", "v1", "a", "sw", "system-wide")
    server.store_secret("K2", "v2", "a", "sw", "system-wide")
    stats = server.stats()
    assert stats["secrets"] == 2


def test_old_cache_without_secrets_loads_cleanly(tmp_path):
    cache = {"granular": [], "cons": [], "cross": [], "tlinks": [],
             "raw_sessions": [], "n_sessions": 0, "n_sessions_with_facts": 0}
    (tmp_path / "compat.json").write_text(json.dumps(cache))
    server = MemoryServer(data_dir=str(tmp_path), key="compat")
    assert server._secrets == []


def test_store_secret_invalid_scope(tmp_path):
    server = MemoryServer(data_dir=str(tmp_path), key="sec_badscope")
    result = server.store_secret("KEY", "val", "a", "sw", "invalid-scope")
    assert result["code"] == "INVALID_SCOPE"
