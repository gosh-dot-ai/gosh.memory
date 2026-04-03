"""Tests for src/storage.py — Storage backend abstraction."""

import json
from pathlib import Path

import numpy as np
import pytest

import src.storage as storage_mod
from src.storage import JSONNPZStorage, SQLiteStorageBackend, StorageBackend, make_storage, migrate_jsonnpz_to_sqlite


def test_json_npz_storage_implements_protocol(tmp_path):
    s = JSONNPZStorage(str(tmp_path), "test")
    assert isinstance(s, StorageBackend)


def test_storage_not_exists_initially(tmp_path):
    s = JSONNPZStorage(str(tmp_path), "new_key")
    assert not s.exists


def test_save_and_load_facts(tmp_path):
    s = JSONNPZStorage(str(tmp_path), "k")
    data = {"granular": [{"fact": "x"}], "n_sessions": 1, "secrets": []}
    s.save_facts(data)
    loaded = s.load_facts()
    assert loaded["granular"][0]["fact"] == "x"
    assert loaded["n_sessions"] == 1


def test_missing_keys_return_defaults(tmp_path):
    s = JSONNPZStorage(str(tmp_path), "k")
    (tmp_path / "k.json").write_text("{}")
    loaded = s.load_facts()
    assert loaded.get("granular", []) == []


def test_exists_after_save(tmp_path):
    s = JSONNPZStorage(str(tmp_path), "k")
    s.save_facts({"granular": []})
    assert s.exists


def test_save_and_load_embeddings(tmp_path):
    s = JSONNPZStorage(str(tmp_path), "k")
    gran = np.random.rand(5, 32).astype(np.float32)
    cons = np.random.rand(2, 32).astype(np.float32)
    cross = np.zeros((0, 32), dtype=np.float32)
    s.save_embeddings(gran, cons, cross)
    loaded = s.load_embeddings()
    assert loaded is not None
    np.testing.assert_allclose(loaded["gran"], gran)
    np.testing.assert_allclose(loaded["cons"], cons)


def test_load_embeddings_none_when_missing(tmp_path):
    s = JSONNPZStorage(str(tmp_path), "k")
    assert s.load_embeddings() is None


def test_make_storage_returns_sqlite_for_new_keys(tmp_path):
    s = make_storage(str(tmp_path), "k")
    assert isinstance(s, SQLiteStorageBackend)


def test_memory_server_uses_storage_backend(tmp_path):
    """MemoryServer must load/save via storage backend."""
    from src.memory import MemoryServer
    server = MemoryServer(data_dir=str(tmp_path), key="storage_test")
    assert hasattr(server, "_storage")
    assert isinstance(server._storage, StorageBackend)


def test_custom_storage_injected(tmp_path):
    """MemoryServer accepts custom storage via constructor injection."""
    from src.memory import MemoryServer
    storage = JSONNPZStorage(str(tmp_path), "custom_key")
    server = MemoryServer(data_dir=str(tmp_path), key="any_key", storage=storage)
    assert server._storage is storage


def test_storage_roundtrip_via_memory_server(tmp_path):
    """Data written by MemoryServer is readable by a new instance."""
    from src.memory import MemoryServer
    s1 = MemoryServer(data_dir=str(tmp_path), key="rt")
    s1._all_granular = [{"fact": "test fact", "kind": "fact"}]
    s1._n_sessions = 1
    s1._save_cache()

    s2 = MemoryServer(data_dir=str(tmp_path), key="rt")
    assert len(s2._all_granular) == 1
    assert s2._all_granular[0]["fact"] == "test fact"


def test_sqlite_storage_roundtrip_facts_and_embeddings(tmp_path):
    storage = SQLiteStorageBackend(str(tmp_path), "sqlite_rt")
    facts = {
        "granular": [{"id": "g1", "fact": "hello", "kind": "event", "session": 1}],
        "cons": [{"id": "c1", "fact": "summary", "kind": "summary"}],
        "cross": [],
        "tlinks": [{"kind": "before"}],
        "raw_sessions": [{
            "message_id": "m1",
            "raw_session_id": "rs1",
            "session_num": 1,
            "session_date": "2024-06-01",
            "content": "hello",
            "speakers": "User and Assistant",
            "agent_id": "default",
            "swarm_id": "default",
            "scope": "swarm-shared",
            "owner_id": "system",
            "read": ["agent:PUBLIC"],
            "write": ["agent:PUBLIC"],
            "stored_at": "2024-06-01T00:00:00+00:00",
            "format": "conversation",
            "source_id": "chat-1",
            "status": "active",
        }],
        "raw_docs": {"doc-1": "# Doc"},
        "episode_corpus": {"documents": [{"doc_id": "document:doc-1", "episodes": [{"episode_id": "e1", "raw_text": "# Doc"}]}]},
        "source_records": {"doc-1": {"family": "document", "metadata": {}, "target": [], "source_meta": {}, "read": ["agent:PUBLIC"], "write": ["agent:PUBLIC"]}},
        "n_sessions": 1,
    }
    storage.save_facts(facts)
    loaded = storage.load_facts()
    assert loaded["granular"][0]["fact"] == "hello"
    assert loaded["tlinks"][0]["kind"] == "before"
    assert loaded["raw_sessions"][0]["message_id"] == "m1"
    assert loaded["episode_corpus"]["documents"][0]["episodes"][0]["episode_id"] == "e1"

    gran = np.random.rand(1, 8).astype(np.float32)
    cons = np.random.rand(1, 8).astype(np.float32)
    cross = np.zeros((0, 8), dtype=np.float32)
    storage.save_embeddings(gran, cons, cross)
    embs = storage.load_embeddings()
    np.testing.assert_allclose(embs["gran"], gran)
    np.testing.assert_allclose(embs["cons"], cons)



def test_sqlite_save_facts_preserves_rowids_for_unchanged_snapshot(tmp_path):
    storage = SQLiteStorageBackend(str(tmp_path), "sqlite_stable_rows")
    facts = {
        "granular": [{"id": "g1", "fact": "hello", "kind": "event", "session": 1}],
        "cons": [],
        "cross": [],
        "tlinks": [{"kind": "before"}],
        "raw_sessions": [{
            "message_id": "m1",
            "raw_session_id": "rs1",
            "session_num": 1,
            "session_date": "2024-06-01",
            "content": "hello",
            "speakers": "User and Assistant",
            "agent_id": "default",
            "swarm_id": "default",
            "scope": "swarm-shared",
            "owner_id": "system",
            "read": ["agent:PUBLIC"],
            "write": ["agent:PUBLIC"],
            "stored_at": "2024-06-01T00:00:00+00:00",
            "format": "conversation",
            "source_id": "chat-1",
            "status": "active",
        }],
        "raw_docs": {"doc-1": "# Doc"},
        "episode_corpus": {"documents": [{"doc_id": "document:doc-1", "episodes": [{"episode_id": "e1", "raw_text": "# Doc"}]}]},
        "source_records": {"doc-1": {"family": "document", "metadata": {}, "target": [], "source_meta": {}, "read": ["agent:PUBLIC"], "write": ["agent:PUBLIC"]}},
        "n_sessions": 1,
    }
    storage.save_facts(facts)
    with storage._connect() as conn:
        first = {
            "raw_sessions": conn.execute("SELECT rowid FROM raw_sessions WHERE raw_session_id = ?", ("rs1",)).fetchone()[0],
            "facts": conn.execute("SELECT rowid FROM facts WHERE tier = ? AND sort_order = 0", ("granular",)).fetchone()[0],
            "episode_corpus": conn.execute("SELECT rowid FROM episode_corpus WHERE doc_id = ? AND episode_id = ?", ("document:doc-1", "e1")).fetchone()[0],
            "source_records": conn.execute("SELECT rowid FROM source_records WHERE source_id = ?", ("doc-1",)).fetchone()[0],
            "state_json": conn.execute("SELECT rowid FROM state_json WHERE name = ?", ("n_sessions",)).fetchone()[0],
        }
    storage.save_facts(facts)
    with storage._connect() as conn:
        second = {
            "raw_sessions": conn.execute("SELECT rowid FROM raw_sessions WHERE raw_session_id = ?", ("rs1",)).fetchone()[0],
            "facts": conn.execute("SELECT rowid FROM facts WHERE tier = ? AND sort_order = 0", ("granular",)).fetchone()[0],
            "episode_corpus": conn.execute("SELECT rowid FROM episode_corpus WHERE doc_id = ? AND episode_id = ?", ("document:doc-1", "e1")).fetchone()[0],
            "source_records": conn.execute("SELECT rowid FROM source_records WHERE source_id = ?", ("doc-1",)).fetchone()[0],
            "state_json": conn.execute("SELECT rowid FROM state_json WHERE name = ?", ("n_sessions",)).fetchone()[0],
        }
    assert second == first

def test_sqlite_storage_serializes_list_event_date_hot_column(tmp_path):
    storage = SQLiteStorageBackend(str(tmp_path), "sqlite_event_date_list")
    facts = {
        "granular": [{"id": "g1", "fact": "dated", "kind": "event", "event_date": ["2024-01-01", "2024-01-02"]}],
        "cons": [],
        "cross": [],
        "tlinks": [],
        "raw_sessions": [],
        "raw_docs": {},
        "episode_corpus": {"documents": []},
    }

    storage.save_facts(facts)

    loaded = storage.load_facts()
    assert loaded["granular"][0]["event_date"] == ["2024-01-01", "2024-01-02"]

    with storage._connect() as conn:
        row = conn.execute("SELECT event_date FROM facts WHERE tier = ? AND fact_id = ?", ("granular", "g1")).fetchone()
    assert row[0] == json.dumps(["2024-01-01", "2024-01-02"], ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def test_migrate_jsonnpz_to_sqlite_preserves_snapshot_and_backs_up_legacy(tmp_path):
    legacy = JSONNPZStorage(str(tmp_path), "migrate")
    facts = {
        "granular": [{"id": "g1", "fact": "hello", "kind": "event"}],
        "cons": [],
        "cross": [],
        "tlinks": [],
        "raw_sessions": [],
        "raw_docs": {},
        "episode_corpus": {"documents": [{"doc_id": "d1", "episodes": [{"episode_id": "e1"}]}]},
        "n_sessions": 1,
    }
    legacy.save_facts(facts)
    (tmp_path / "migrate_corpus.json").write_text(json.dumps(facts["episode_corpus"]))
    gran = np.random.rand(1, 4).astype(np.float32)
    legacy.save_embeddings(gran, np.zeros((0, 4), dtype=np.float32), np.zeros((0, 4), dtype=np.float32))

    result = migrate_jsonnpz_to_sqlite(str(tmp_path), "migrate")
    assert result["migrated"] is True
    assert (tmp_path / "migrate.sqlite3").exists()
    assert (tmp_path / "migrate.json.bak").exists()
    assert (tmp_path / "migrate_embs.npz.bak").exists()
    assert (tmp_path / "migrate_corpus.json.bak").exists()

    storage = SQLiteStorageBackend(str(tmp_path), "migrate")
    loaded = storage.load_facts()
    assert loaded["granular"][0]["fact"] == "hello"
    assert loaded["episode_corpus"]["documents"][0]["episodes"][0]["episode_id"] == "e1"
    embs = storage.load_embeddings()
    np.testing.assert_allclose(embs["gran"], gran)


def test_migrate_jsonnpz_to_sqlite_accepts_legacy_snapshot_shape(tmp_path):
    legacy = JSONNPZStorage(str(tmp_path), "legacy_shape")
    facts = {
        "granular": [{"id": "g1", "fact": "hello", "kind": "event"}],
        "cons": [],
        "cross": [],
        "tlinks": [],
        "raw_sessions": [{
            "raw_session_id": "rs1",
            "session_num": 1,
            "session_date": "2026-01-01",
            "content": "hello",
            "speakers": "User and Assistant",
            "agent_id": "agent-a",
            "swarm_id": "swarm-a",
            "scope": "swarm-shared",
            "owner_id": "agent:agent-a",
            "read": ["swarm:swarm-a"],
            "write": ["swarm:swarm-a"],
            "stored_at": "2026-01-01T00:00:00+00:00",
            "format": "conversation",
            "source_id": "legacy_shape",
            "artifact_id": "art-1",
            "version_id": "ver-1",
            "content_hash": "sha256:abc",
            "status": "active",
            "episode_id": "e1",
            "ingest_transport": "text",
        }],
        "raw_docs": {},
        "episode_corpus": {"documents": [{"doc_id": "legacy_shape", "episodes": [{"episode_id": "e1", "raw_text": "hello"}]}]},
        "source_records": {
            "legacy_shape": {
                "source_id": "legacy_shape",
                "family": "document",
                "scope_id": "legacy_shape",
                "owner_id": "agent:agent-a",
                "read": ["swarm:swarm-a"],
                "write": ["swarm:swarm-a"],
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "artifact_id": "art-1",
                "version_id": "ver-1",
                "content_hash": "sha256:abc",
                "metadata": {"bench": "ama"},
                "source_meta": {"ingest_transport": "text"},
            }
        },
        "n_sessions": 1,
    }
    legacy.save_facts(facts)
    (tmp_path / "legacy_shape_corpus.json").write_text(json.dumps(facts["episode_corpus"]))

    result = migrate_jsonnpz_to_sqlite(str(tmp_path), "legacy_shape")

    assert result["migrated"] is True
    storage = SQLiteStorageBackend(str(tmp_path), "legacy_shape")
    loaded = storage.load_facts()
    assert loaded["raw_sessions"][0]["message_id"] == "raw:rs1"
    assert loaded["source_records"]["legacy_shape"]["family"] == "document"
    assert loaded["source_records"]["legacy_shape"]["target"] == []


def test_sqlite_backend_reuses_same_thread_connection(tmp_path, monkeypatch):
    calls = []
    orig_connect = storage_mod.sqlite3.connect

    def _wrapped_connect(*args, **kwargs):
        calls.append(1)
        return orig_connect(*args, **kwargs)

    monkeypatch.setattr(storage_mod.sqlite3, "connect", _wrapped_connect)
    storage = SQLiteStorageBackend(str(tmp_path), "write_log_reuse")
    assert len(calls) == 1

    storage.append_write_log(
        message_id="m1",
        session_id="s1",
        agent_id="agent-a",
        swarm_id="swarm-a",
        visibility="shared",
        owner_id="agent:agent-a",
        scope="swarm-shared",
        read=["swarm:swarm-a"],
        write=["swarm:swarm-a"],
        content_family="chat",
        content_text="hello",
        metadata={"role": "user"},
        timestamp_ms=1712000000000,
    )
    storage.get_write_status("m1")
    storage.list_write_log_entries(states=["pending"], order="asc")
    storage.mark_write_state("m1", "complete")

    assert len(calls) == 1
    storage.close()


def test_sqlite_write_log_idempotent_append_and_status(tmp_path):
    storage = SQLiteStorageBackend(str(tmp_path), "write_log")
    first = storage.append_write_log(
        message_id="m1",
        session_id="s1",
        agent_id="agent-a",
        swarm_id="swarm-a",
        visibility="shared",
        owner_id="agent:agent-a",
        scope="swarm-shared",
        read=["swarm:swarm-a"],
        write=["swarm:swarm-a"],
        content_family="chat",
        content_text="hello",
        metadata={"role": "user"},
        timestamp_ms=1712000000000,
    )
    second = storage.append_write_log(
        message_id="m1",
        session_id="s1",
        agent_id="agent-a",
        swarm_id="swarm-a",
        visibility="shared",
        owner_id="agent:agent-a",
        scope="swarm-shared",
        read=["swarm:swarm-a"],
        write=["swarm:swarm-a"],
        content_family="chat",
        content_text="hello",
        metadata={"role": "user"},
        timestamp_ms=1712000000000,
    )
    assert first["inserted"] is True
    assert second["inserted"] is False
    status = storage.get_write_status("m1")
    assert status["extraction_state"] == "pending"
