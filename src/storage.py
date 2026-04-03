"""gosh.memory — Storage backend abstraction.

Separates persistence from MemoryServer logic.
Legacy backend: JSONNPZStorage (flat files).
Default v2 backend: SQLiteStorageBackend (single durable store per key).
"""

from __future__ import annotations

import atexit
import hashlib
import io
import json
import os
import sqlite3
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .episodes import load_episode_corpus


@runtime_checkable
class StorageBackend(Protocol):
    """Persistence interface for MemoryServer."""

    def load_facts(self) -> dict:
        ...

    def save_facts(self, data: dict) -> None:
        ...

    def load_embeddings(self) -> dict | None:
        ...

    def save_embeddings(self, gran: np.ndarray, cons: np.ndarray, cross: np.ndarray) -> None:
        ...

    @property
    def exists(self) -> bool:
        ...


MAGIC = b"GME1"
SQLITE_SCHEMA_VERSION = 1
INTERNAL_EPISODE_DOC_ORDER = "_episode_doc_order"
INTERNAL_EPISODE_CORPUS_LAYOUT = "_episode_corpus_layout"
STATE_JSON_KEYS = {
    "secrets",
    "n_sessions",
    "n_sessions_with_facts",
    "_emb_fingerprints",
    "_dedup_index",
    "_git_dedup_index",
    "metadata_schema",
    "instance_config",
    "scope_record",
    "profiles",
    "profile_configs",
    "memory_config",
    INTERNAL_EPISODE_DOC_ORDER,
    INTERNAL_EPISODE_CORPUS_LAYOUT,
}


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _json_loads(raw: str | None, default: Any) -> Any:
    if raw in (None, ""):
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


def _sqlite_text_scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return _json_dumps(value)
    return value


def _timestamp_ms(value: Any, fallback: int) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.strip():
        text = value.strip()
        if text.isdigit():
            return int(text)
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1000)
        except Exception:
            pass
    return fallback


class JSONNPZStorage:
    """Legacy storage backend: facts as JSON, embeddings as NPZ."""

    def __init__(self, data_dir: str, key: str, encryption_key: bytes | None = None):
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._json_path = self._dir / f"{key}.json"
        self._embs_path = self._dir / f"{key}_embs.npz"
        self._encryption_key = encryption_key

    def _encrypt(self, data: bytes) -> bytes:
        if self._encryption_key is None:
            return data
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            return data
        nonce = os.urandom(12)
        aesgcm = AESGCM(self._encryption_key)
        ct = aesgcm.encrypt(nonce, data, None)
        return MAGIC + nonce + ct

    def _decrypt(self, data: bytes) -> bytes:
        if not data.startswith(MAGIC):
            return data
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError as exc:
            raise RuntimeError("Encrypted file but cryptography package not installed") from exc
        if self._encryption_key is None:
            raise RuntimeError("Encrypted file but no encryption key provided")
        nonce = data[4:16]
        ct = data[16:]
        aesgcm = AESGCM(self._encryption_key)
        return aesgcm.decrypt(nonce, ct, None)

    @property
    def exists(self) -> bool:
        return self._json_path.exists()

    @property
    def json_path(self) -> Path:
        return self._json_path

    @property
    def embs_path(self) -> Path:
        return self._embs_path

    def load_facts(self) -> dict:
        if not self._json_path.exists():
            return {}
        raw = self._json_path.read_bytes()
        decrypted = self._decrypt(raw)
        return json.loads(decrypted.decode("utf-8"))

    def save_facts(self, data: dict) -> None:
        plaintext = json.dumps(data).encode("utf-8")
        output = self._encrypt(plaintext)
        fd, tmp_path = tempfile.mkstemp(dir=str(self._dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(output)
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, str(self._json_path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def load_embeddings(self) -> dict | None:
        if not self._embs_path.exists():
            return None
        raw = self._embs_path.read_bytes()
        decrypted = self._decrypt(raw)
        if raw.startswith(MAGIC):
            buf = io.BytesIO(decrypted)
            loaded = np.load(buf)
        else:
            loaded = np.load(self._embs_path)
        return {k: loaded[k] for k in loaded.files}

    def save_embeddings(self, gran: np.ndarray, cons: np.ndarray, cross: np.ndarray) -> None:
        buf = io.BytesIO()
        np.savez_compressed(buf, gran=gran, cons=cons, cross=cross)
        plaintext = buf.getvalue()
        output = self._encrypt(plaintext)
        if output is plaintext and self._encryption_key is None:
            np.savez_compressed(self._embs_path, gran=gran, cons=cons, cross=cross)
        else:
            self._embs_path.write_bytes(output)


class SQLiteStorageBackend:
    """SQLite-backed single durable store for one memory key."""

    def __init__(
        self,
        data_dir: str,
        key: str,
        encryption_key: bytes | None = None,
        db_path: str | Path | None = None,
    ):
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._key = key
        self._path = Path(db_path) if db_path is not None else (self._dir / f"{key}.sqlite3")
        self._encryption_key = encryption_key
        self._sqlite_mod, self._sqlcipher = self._resolve_sqlite_module(encryption_key)
        self._connections: dict[int, Any] = {}
        self._connections_lock = threading.Lock()
        atexit.register(self.close)
        self._ensure_schema()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def exists(self) -> bool:
        return self._path.exists()

    def _resolve_sqlite_module(self, encryption_key: bytes | None):
        if encryption_key is None:
            return sqlite3, False
        try:
            from pysqlcipher3 import dbapi2 as sqlcipher_sqlite  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Encrypted SQLite mode requires SQLCipher (pysqlcipher3). "
                "Migration must abort when SQLCipher is unavailable."
            ) from exc
        return sqlcipher_sqlite, True

    def _open_connection(self):
        conn = self._sqlite_mod.connect(str(self._path), timeout=30, check_same_thread=False)
        if self._sqlcipher:
            conn.execute(f"PRAGMA key = \"x'{self._encryption_key.hex()}'\"")
        conn.row_factory = self._sqlite_mod.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _connection_for_thread(self):
        tid = threading.get_ident()
        with self._connections_lock:
            conn = self._connections.get(tid)
            if conn is not None:
                return conn
            conn = self._open_connection()
            self._connections[tid] = conn
            return conn

    @contextmanager
    def _connect(self):
        conn = self._connection_for_thread()
        try:
            yield conn
        except Exception:
            try:
                if getattr(conn, "in_transaction", False):
                    conn.rollback()
            except Exception:
                pass
            raise

    def close(self) -> None:
        with self._connections_lock:
            connections = list(self._connections.values())
            self._connections.clear()
        for conn in connections:
            try:
                conn.close()
            except Exception:
                pass

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    name TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS write_log (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    swarm_id TEXT NOT NULL,
                    visibility TEXT NOT NULL DEFAULT 'shared',
                    owner_id TEXT,
                    scope TEXT,
                    read_json TEXT,
                    write_json TEXT,
                    content_family TEXT NOT NULL,
                    content_text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    timestamp_ms INTEGER NOT NULL,
                    extraction_state TEXT NOT NULL DEFAULT 'pending',
                    extraction_attempts INTEGER NOT NULL DEFAULT 0,
                    last_extraction_attempt_ms INTEGER,
                    sort_order INTEGER NOT NULL UNIQUE
                );
                CREATE INDEX IF NOT EXISTS idx_wl_state ON write_log(extraction_state, timestamp_ms);
                CREATE INDEX IF NOT EXISTS idx_wl_swarm ON write_log(swarm_id, timestamp_ms);
                CREATE INDEX IF NOT EXISTS idx_wl_session ON write_log(session_id, timestamp_ms);

                CREATE TABLE IF NOT EXISTS raw_sessions (
                    raw_session_id TEXT PRIMARY KEY,
                    session_num INTEGER,
                    message_id TEXT NOT NULL UNIQUE REFERENCES write_log(message_id),
                    source_id TEXT,
                    format TEXT,
                    session_date TEXT,
                    speakers TEXT,
                    stored_at TEXT,
                    artifact_id TEXT,
                    version_id TEXT,
                    content_hash TEXT,
                    owner_id TEXT,
                    scope TEXT,
                    agent_id TEXT,
                    swarm_id TEXT,
                    read_json TEXT,
                    write_json TEXT,
                    target_json TEXT,
                    metadata_json TEXT,
                    source_meta_json TEXT,
                    status TEXT,
                    sort_order INTEGER NOT NULL UNIQUE
                );

                CREATE TABLE IF NOT EXISTS raw_docs (
                    source_id TEXT PRIMARY KEY,
                    message_id TEXT NOT NULL UNIQUE REFERENCES write_log(message_id),
                    metadata_json TEXT,
                    sort_order INTEGER NOT NULL UNIQUE
                );

                CREATE TABLE IF NOT EXISTS facts (
                    tier TEXT NOT NULL,
                    sort_order INTEGER NOT NULL,
                    fact_id TEXT,
                    kind TEXT,
                    session_num INTEGER,
                    source_id TEXT,
                    agent_id TEXT,
                    swarm_id TEXT,
                    scope TEXT,
                    owner_id TEXT,
                    status TEXT,
                    created_at TEXT,
                    event_date TEXT,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (tier, sort_order)
                );
                CREATE INDEX IF NOT EXISTS idx_facts_id ON facts(tier, fact_id);

                CREATE TABLE IF NOT EXISTS embeddings (
                    tier TEXT NOT NULL,
                    sort_order INTEGER NOT NULL,
                    fact_id TEXT,
                    dim INTEGER NOT NULL,
                    dtype TEXT NOT NULL,
                    vector_blob BLOB NOT NULL,
                    PRIMARY KEY (tier, sort_order)
                );
                CREATE INDEX IF NOT EXISTS idx_embeddings_fact_id ON embeddings(tier, fact_id);

                CREATE TABLE IF NOT EXISTS episode_corpus (
                    doc_id TEXT NOT NULL,
                    episode_id TEXT NOT NULL,
                    sort_order INTEGER NOT NULL,
                    episode_json TEXT NOT NULL,
                    PRIMARY KEY (doc_id, episode_id),
                    UNIQUE (doc_id, sort_order)
                );

                CREATE TABLE IF NOT EXISTS temporal_links (
                    sort_order INTEGER PRIMARY KEY,
                    link_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS source_records (
                    source_id TEXT PRIMARY KEY,
                    family TEXT NOT NULL,
                    owner_id TEXT,
                    read_json TEXT,
                    write_json TEXT,
                    artifact_id TEXT,
                    version_id TEXT,
                    content_hash TEXT,
                    metadata_json TEXT,
                    target_json TEXT,
                    source_meta_json TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS state_json (
                    name TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta(name, value_json) VALUES(?, ?)",
                ("schema_version", _json_dumps(SQLITE_SCHEMA_VERSION)),
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta(name, value_json) VALUES(?, ?)",
                ("storage_backend", _json_dumps("sqlite")),
            )
            conn.commit()

    def _legacy_message_id_for_session(self, raw_session: dict, fallback_idx: int | None = None) -> str:
        message_id = str(raw_session.get("message_id") or "").strip()
        if message_id:
            return message_id
        raw_session_id = str(raw_session.get("raw_session_id") or "").strip()
        if raw_session_id:
            return f"raw:{raw_session_id}"
        source_id = str(raw_session.get("source_id") or "").strip()
        session_num = raw_session.get("session_num")
        if source_id and session_num is not None:
            return f"legacy:{source_id}:{session_num}"
        if session_num is not None:
            return f"legacy:{session_num}"
        content = str(raw_session.get("content") or "")
        session_date = str(raw_session.get("session_date") or "")
        if content or session_date:
            seed = f"{source_id}|{session_num}|{session_date}|{content}"
            return f"legacy:auto:{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:16]}"
        if fallback_idx is not None:
            return f"legacy:row:{fallback_idx}"
        raise ValueError("raw_session missing stable identity for message_id derivation")

    def _legacy_message_id_for_doc(self, source_id: str) -> str:
        return f"rawdoc:{source_id}"

    def _raw_session_identity(self, raw_session: dict, fallback_idx: int | None = None) -> str:
        raw_session_id = str(raw_session.get("raw_session_id") or "").strip()
        if raw_session_id:
            return raw_session_id
        message_id = str(raw_session.get("message_id") or "").strip()
        if message_id:
            return message_id
        source_id = str(raw_session.get("source_id") or "").strip()
        session_num = raw_session.get("session_num")
        if source_id and session_num is not None:
            return f"legacy:{source_id}:{session_num}"
        if session_num is not None:
            return f"legacy:{session_num}"
        content = str(raw_session.get("content") or "")
        session_date = str(raw_session.get("session_date") or "")
        if content or session_date:
            seed = f"{source_id}|{session_num}|{session_date}|{content}"
            return f"legacy:auto:{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:16]}"
        if fallback_idx is not None:
            return f"legacy:row:{fallback_idx}"
        raise ValueError("raw_session missing stable identity")

    def _episode_corpus_layout(self, corpus: dict) -> dict[str, list[str]]:
        layout: dict[str, list[str]] = {}
        if not isinstance(corpus, dict):
            return {"documents": []}
        for group_name, docs in corpus.items():
            if not isinstance(docs, list):
                continue
            doc_ids = [
                str(doc.get("doc_id") or "")
                for doc in docs
                if isinstance(doc, dict) and str(doc.get("doc_id") or "")
            ]
            if doc_ids:
                layout[str(group_name)] = doc_ids
        layout.setdefault("documents", [])
        return layout

    def _sync_rows(
        self,
        conn,
        *,
        select_sql: str,
        desired_rows: list[tuple],
        key_len: int,
        delete_sql: str,
        insert_sql: str,
        select_params: tuple[Any, ...] = (),
    ) -> None:
        existing_rows = [tuple(row) for row in conn.execute(select_sql, select_params).fetchall()]
        existing = {row[:key_len]: row for row in existing_rows}
        desired = {row[:key_len]: row for row in desired_rows}

        delete_keys = [key for key in existing if key not in desired or existing[key] != desired[key]]
        if delete_keys:
            conn.executemany(
                delete_sql,
                [key if isinstance(key, tuple) else (key,) for key in delete_keys],
            )

        changed_rows = [row for row in desired_rows if existing.get(row[:key_len]) != row]
        if changed_rows:
            conn.executemany(insert_sql, changed_rows)

    def _upsert_write_log(
        self,
        conn,
        *,
        message_id: str,
        session_id: str,
        agent_id: str,
        swarm_id: str,
        visibility: str,
        owner_id: str | None,
        scope: str | None,
        read: list[str] | None,
        write: list[str] | None,
        content_family: str,
        content_text: str,
        metadata: dict | None,
        timestamp_ms: int,
        extraction_state: str,
        sort_order: int,
    ) -> None:
        conn.execute(
            """
            INSERT INTO write_log(
                message_id, session_id, agent_id, swarm_id, visibility,
                owner_id, scope, read_json, write_json, content_family,
                content_text, metadata_json, timestamp_ms, extraction_state,
                extraction_attempts, last_extraction_attempt_ms, sort_order
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?)
            ON CONFLICT(message_id) DO UPDATE SET
                session_id=excluded.session_id,
                agent_id=excluded.agent_id,
                swarm_id=excluded.swarm_id,
                visibility=excluded.visibility,
                owner_id=excluded.owner_id,
                scope=excluded.scope,
                read_json=excluded.read_json,
                write_json=excluded.write_json,
                content_family=excluded.content_family,
                content_text=excluded.content_text,
                metadata_json=excluded.metadata_json,
                timestamp_ms=excluded.timestamp_ms,
                extraction_state=excluded.extraction_state,
                sort_order=excluded.sort_order
            """,
            (
                message_id,
                session_id,
                agent_id,
                swarm_id,
                visibility,
                owner_id,
                scope,
                _json_dumps(read or []),
                _json_dumps(write or []),
                content_family,
                content_text,
                _json_dumps(metadata or {}),
                timestamp_ms,
                extraction_state,
                sort_order,
            ),
        )

    def _fact_row_payload(self, fact: dict) -> tuple:
        payload = dict(fact)
        payload.pop("_temporal_links", None)
        return (
            str(fact.get("id") or ""),
            fact.get("kind"),
            fact.get("session") if fact.get("session") is not None else fact.get("session_num"),
            fact.get("source_id"),
            fact.get("agent_id"),
            fact.get("swarm_id"),
            fact.get("scope"),
            fact.get("owner_id"),
            fact.get("status"),
            fact.get("created_at"),
            _sqlite_text_scalar(fact.get("event_date")),
            _json_dumps(payload),
        )

    def save_facts(self, data: dict) -> None:
        granular = data.get("granular", []) or []
        cons = data.get("cons", []) or []
        cross = data.get("cross", []) or []
        raw_sessions = data.get("raw_sessions", []) or []
        raw_docs = data.get("raw_docs", {}) or {}
        episode_corpus = data.get("episode_corpus", {"documents": []}) or {"documents": []}
        temporal_links = data.get("tlinks", []) or []
        source_records = data.get("source_records", {}) or {}

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing_write_log = {
                str(row["message_id"]): {
                    "session_id": row["session_id"],
                    "agent_id": row["agent_id"],
                    "swarm_id": row["swarm_id"],
                    "visibility": row["visibility"],
                    "owner_id": row["owner_id"],
                    "scope": row["scope"],
                    "read_json": row["read_json"],
                    "write_json": row["write_json"],
                    "content_family": row["content_family"],
                    "content_text": row["content_text"],
                    "metadata_json": row["metadata_json"],
                    "timestamp_ms": int(row["timestamp_ms"] or 0),
                    "extraction_state": row["extraction_state"],
                    "sort_order": int(row["sort_order"] or 0),
                }
                for row in conn.execute(
                    """
                    SELECT message_id, session_id, agent_id, swarm_id, visibility, owner_id, scope,
                           read_json, write_json, content_family, content_text, metadata_json,
                           timestamp_ms, extraction_state, sort_order
                    FROM write_log
                    """
                ).fetchall()
            }
            next_write_log_sort = max((row["sort_order"] for row in existing_write_log.values()), default=-1) + 1
            desired_write_log_ids: set[str] = set()

            for idx, raw_session in enumerate(raw_sessions):
                message_id = self._legacy_message_id_for_session(raw_session, fallback_idx=idx)
                desired_write_log_ids.add(message_id)
                session_num = raw_session.get("session_num")
                session_id = str(
                    raw_session.get("session_key")
                    or raw_session.get("raw_session_id")
                    or raw_session.get("message_id")
                    or f"session:{session_num}"
                )
                extraction_state = "pending" if raw_session.get("status") == "pending_extraction" else "complete"
                visibility = "private" if raw_session.get("scope") == "agent-private" else "shared"
                ts = _timestamp_ms(raw_session.get("stored_at") or raw_session.get("session_date"), idx)
                existing = existing_write_log.get(message_id)
                sort_order = existing["sort_order"] if existing is not None else next_write_log_sort
                if existing is None:
                    next_write_log_sort += 1
                desired_tuple = (
                    session_id,
                    str(raw_session.get("agent_id") or "default"),
                    str(raw_session.get("swarm_id") or "default"),
                    visibility,
                    raw_session.get("owner_id"),
                    raw_session.get("scope"),
                    _json_dumps(raw_session.get("read") or []),
                    _json_dumps(raw_session.get("write") or []),
                    str(raw_session.get("format") or "conversation"),
                    str(raw_session.get("content") or ""),
                    _json_dumps(raw_session.get("metadata") or {}),
                    ts,
                    extraction_state,
                    sort_order,
                )
                existing_tuple = None
                if existing is not None:
                    existing_tuple = (
                        existing["session_id"],
                        existing["agent_id"],
                        existing["swarm_id"],
                        existing["visibility"],
                        existing["owner_id"],
                        existing["scope"],
                        existing["read_json"],
                        existing["write_json"],
                        existing["content_family"],
                        existing["content_text"],
                        existing["metadata_json"],
                        existing["timestamp_ms"],
                        existing["extraction_state"],
                        existing["sort_order"],
                    )
                if existing_tuple != desired_tuple:
                    self._upsert_write_log(
                        conn,
                        message_id=message_id,
                        session_id=session_id,
                        agent_id=str(raw_session.get("agent_id") or "default"),
                        swarm_id=str(raw_session.get("swarm_id") or "default"),
                        visibility=visibility,
                        owner_id=raw_session.get("owner_id"),
                        scope=raw_session.get("scope"),
                        read=list(raw_session.get("read") or []),
                        write=list(raw_session.get("write") or []),
                        content_family=str(raw_session.get("format") or "conversation"),
                        content_text=str(raw_session.get("content") or ""),
                        metadata=raw_session.get("metadata") or {},
                        timestamp_ms=ts,
                        extraction_state=extraction_state,
                        sort_order=sort_order,
                    )

            source_records_by_id = source_records if isinstance(source_records, dict) else {}
            for idx, (source_id, raw_text) in enumerate(raw_docs.items()):
                record = source_records_by_id.get(source_id) or {}
                message_id = self._legacy_message_id_for_doc(str(source_id))
                desired_write_log_ids.add(message_id)
                existing = existing_write_log.get(message_id)
                sort_order = existing["sort_order"] if existing is not None else next_write_log_sort
                if existing is None:
                    next_write_log_sort += 1
                desired_tuple = (
                    f"doc:{source_id}",
                    str(record.get("owner_id") or "default"),
                    "default",
                    "shared",
                    record.get("owner_id"),
                    None,
                    _json_dumps(record.get("read") or []),
                    _json_dumps(record.get("write") or []),
                    str(record.get("family") or "document"),
                    str(raw_text or ""),
                    _json_dumps(record.get("metadata") or {}),
                    1_000_000 + idx,
                    "complete",
                    sort_order,
                )
                existing_tuple = None
                if existing is not None:
                    existing_tuple = (
                        existing["session_id"],
                        existing["agent_id"],
                        existing["swarm_id"],
                        existing["visibility"],
                        existing["owner_id"],
                        existing["scope"],
                        existing["read_json"],
                        existing["write_json"],
                        existing["content_family"],
                        existing["content_text"],
                        existing["metadata_json"],
                        existing["timestamp_ms"],
                        existing["extraction_state"],
                        existing["sort_order"],
                    )
                if existing_tuple != desired_tuple:
                    self._upsert_write_log(
                        conn,
                        message_id=message_id,
                        session_id=f"doc:{source_id}",
                        agent_id=str(record.get("owner_id") or "default"),
                        swarm_id="default",
                        visibility="shared",
                        owner_id=record.get("owner_id"),
                        scope=None,
                        read=record.get("read") or [],
                        write=record.get("write") or [],
                        content_family=str(record.get("family") or "document"),
                        content_text=str(raw_text or ""),
                        metadata=record.get("metadata") or {},
                        timestamp_ms=1_000_000 + idx,
                        extraction_state="complete",
                        sort_order=sort_order,
                    )

            raw_session_rows = []
            for idx, raw_session in enumerate(raw_sessions):
                message_id = self._legacy_message_id_for_session(raw_session, fallback_idx=idx)
                source_id = str(raw_session.get("source_id") or self._key)
                source_meta = {
                    key: value
                    for key, value in raw_session.items()
                    if key not in {
                        "message_id", "raw_session_id", "session_num", "session_date", "content", "speakers",
                        "agent_id", "swarm_id", "scope", "owner_id", "read", "write",
                        "stored_at", "format", "source_id", "artifact_id",
                        "version_id", "content_hash", "status", "metadata",
                        "target",
                    }
                }
                raw_session_rows.append(
                    (
                        self._raw_session_identity(raw_session, fallback_idx=idx),
                        raw_session.get("session_num"),
                        message_id,
                        source_id,
                        raw_session.get("format"),
                        raw_session.get("session_date"),
                        raw_session.get("speakers"),
                        raw_session.get("stored_at"),
                        raw_session.get("artifact_id"),
                        raw_session.get("version_id"),
                        raw_session.get("content_hash"),
                        raw_session.get("owner_id"),
                        raw_session.get("scope"),
                        raw_session.get("agent_id"),
                        raw_session.get("swarm_id"),
                        _json_dumps(raw_session.get("read") or []),
                        _json_dumps(raw_session.get("write") or []),
                        _json_dumps(raw_session.get("target") or []),
                        _json_dumps(raw_session.get("metadata") or {}),
                        _json_dumps(source_meta),
                        raw_session.get("status"),
                        idx,
                    )
                )
            self._sync_rows(
                conn,
                select_sql=(
                    "SELECT raw_session_id, session_num, message_id, source_id, format, session_date, "
                    "speakers, stored_at, artifact_id, version_id, content_hash, owner_id, scope, "
                    "agent_id, swarm_id, read_json, write_json, target_json, metadata_json, "
                    "source_meta_json, status, sort_order FROM raw_sessions ORDER BY sort_order"
                ),
                desired_rows=raw_session_rows,
                key_len=1,
                delete_sql="DELETE FROM raw_sessions WHERE raw_session_id = ?",
                insert_sql=(
                    "INSERT INTO raw_sessions("
                    "raw_session_id, session_num, message_id, source_id, format, session_date, "
                    "speakers, stored_at, artifact_id, version_id, content_hash, owner_id, scope, "
                    "agent_id, swarm_id, read_json, write_json, target_json, metadata_json, "
                    "source_meta_json, status, sort_order"
                    ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                ),
            )

            raw_doc_rows = [
                (
                    str(source_id),
                    self._legacy_message_id_for_doc(str(source_id)),
                    _json_dumps((source_records_by_id.get(source_id) or {}).get("metadata") or {}),
                    idx,
                )
                for idx, (source_id, _raw_text) in enumerate(raw_docs.items())
            ]
            self._sync_rows(
                conn,
                select_sql="SELECT source_id, message_id, metadata_json, sort_order FROM raw_docs ORDER BY sort_order",
                desired_rows=raw_doc_rows,
                key_len=1,
                delete_sql="DELETE FROM raw_docs WHERE source_id = ?",
                insert_sql="INSERT INTO raw_docs(source_id, message_id, metadata_json, sort_order) VALUES(?, ?, ?, ?)",
            )

            for tier_name, tier_facts in (("granular", granular), ("cons", cons), ("cross", cross)):
                fact_rows = [
                    (tier_name, idx, *self._fact_row_payload(fact))
                    for idx, fact in enumerate(tier_facts)
                ]
                self._sync_rows(
                    conn,
                    select_sql=(
                        "SELECT tier, sort_order, fact_id, kind, session_num, source_id, "
                        "agent_id, swarm_id, scope, owner_id, status, created_at, event_date, payload_json "
                        "FROM facts WHERE tier = ? ORDER BY sort_order"
                    ),
                    select_params=(tier_name,),
                    desired_rows=fact_rows,
                    key_len=2,
                    delete_sql="DELETE FROM facts WHERE tier = ? AND sort_order = ?",
                    insert_sql=(
                        "INSERT INTO facts("
                        "tier, sort_order, fact_id, kind, session_num, source_id, agent_id, swarm_id, "
                        "scope, owner_id, status, created_at, event_date, payload_json"
                        ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    ),
                )

            temporal_rows = [
                (idx, _json_dumps(link))
                for idx, link in enumerate(temporal_links)
            ]
            self._sync_rows(
                conn,
                select_sql="SELECT sort_order, link_json FROM temporal_links ORDER BY sort_order",
                desired_rows=temporal_rows,
                key_len=1,
                delete_sql="DELETE FROM temporal_links WHERE sort_order = ?",
                insert_sql="INSERT INTO temporal_links(sort_order, link_json) VALUES(?, ?)",
            )

            corpus_layout = self._episode_corpus_layout(episode_corpus)
            doc_order = [doc_id for doc_ids in corpus_layout.values() for doc_id in doc_ids]
            episode_rows = []
            for docs in (
                docs
                for docs in episode_corpus.values()
                if isinstance(episode_corpus, dict) and isinstance(docs, list)
            ):
                for doc in docs:
                    if not isinstance(doc, dict):
                        continue
                    doc_id = str(doc.get("doc_id") or "")
                    if not doc_id:
                        continue
                    for idx, episode in enumerate(doc.get("episodes", []) or []):
                        episode_rows.append((doc_id, str(episode.get("episode_id") or ""), idx, _json_dumps(episode)))
            self._sync_rows(
                conn,
                select_sql="SELECT doc_id, episode_id, sort_order, episode_json FROM episode_corpus ORDER BY doc_id, sort_order",
                desired_rows=episode_rows,
                key_len=2,
                delete_sql="DELETE FROM episode_corpus WHERE doc_id = ? AND episode_id = ?",
                insert_sql="INSERT INTO episode_corpus(doc_id, episode_id, sort_order, episode_json) VALUES(?, ?, ?, ?)",
            )

            source_rows = [
                (
                    str(source_id),
                    record.get("family") or "unknown",
                    record.get("owner_id"),
                    _json_dumps(record.get("read") or []),
                    _json_dumps(record.get("write") or []),
                    record.get("artifact_id"),
                    record.get("version_id"),
                    record.get("content_hash"),
                    _json_dumps(record.get("metadata") or {}),
                    _json_dumps(record.get("target") or []),
                    _json_dumps(record.get("source_meta") or {}),
                    record.get("created_at"),
                    record.get("updated_at"),
                )
                for source_id, record in source_records_by_id.items()
            ]
            self._sync_rows(
                conn,
                select_sql=(
                    "SELECT source_id, family, owner_id, read_json, write_json, artifact_id, version_id, "
                    "content_hash, metadata_json, target_json, source_meta_json, created_at, updated_at "
                    "FROM source_records ORDER BY source_id"
                ),
                desired_rows=source_rows,
                key_len=1,
                delete_sql="DELETE FROM source_records WHERE source_id = ?",
                insert_sql=(
                    "INSERT INTO source_records("
                    "source_id, family, owner_id, read_json, write_json, artifact_id, version_id, "
                    "content_hash, metadata_json, target_json, source_meta_json, created_at, updated_at"
                    ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                ),
            )

            state_values = {name: data.get(name) for name in STATE_JSON_KEYS if name in data}
            state_values[INTERNAL_EPISODE_DOC_ORDER] = doc_order
            state_values[INTERNAL_EPISODE_CORPUS_LAYOUT] = corpus_layout
            state_rows = [(name, _json_dumps(value)) for name, value in state_values.items()]
            self._sync_rows(
                conn,
                select_sql="SELECT name, value_json FROM state_json ORDER BY name",
                desired_rows=state_rows,
                key_len=1,
                delete_sql="DELETE FROM state_json WHERE name = ?",
                insert_sql="INSERT INTO state_json(name, value_json) VALUES(?, ?)",
            )
            if desired_write_log_ids:
                placeholders = ",".join("?" for _ in desired_write_log_ids)
                # Dynamic placeholder arity is required here; values are still bound parameters.
                conn.execute(
                    f"DELETE FROM write_log WHERE extraction_state = 'complete' AND message_id NOT IN ({placeholders})",  # noqa: S608  # nosec B608
                    tuple(desired_write_log_ids),
                )
            else:
                conn.execute("DELETE FROM write_log WHERE extraction_state = 'complete'")
            conn.commit()
    def _load_tier(self, conn, tier: str) -> list[dict]:
        rows = conn.execute(
            "SELECT payload_json FROM facts WHERE tier = ? ORDER BY sort_order",
            (tier,),
        ).fetchall()
        return [_json_loads(row["payload_json"], {}) for row in rows]

    def _load_raw_sessions(self, conn) -> list[dict]:
        rows = conn.execute(
            """
            SELECT rs.*, wl.content_text
            FROM raw_sessions rs
            JOIN write_log wl ON wl.message_id = rs.message_id
            ORDER BY rs.sort_order
            """
        ).fetchall()
        sessions: list[dict] = []
        for row in rows:
            message_id = str(row["message_id"] or "")
            raw_session_id = message_id.removeprefix("raw:") if message_id.startswith("raw:") else message_id
            session = {
                "message_id": message_id,
                "raw_session_id": row["raw_session_id"] or raw_session_id,
                "session_num": row["session_num"],
                "session_date": row["session_date"],
                "content": row["content_text"],
                "speakers": row["speakers"],
                "agent_id": row["agent_id"],
                "swarm_id": row["swarm_id"],
                "scope": row["scope"],
                "owner_id": row["owner_id"],
                "read": _json_loads(row["read_json"], []),
                "write": _json_loads(row["write_json"], []),
                "stored_at": row["stored_at"],
                "format": row["format"],
                "source_id": row["source_id"],
                "artifact_id": row["artifact_id"],
                "version_id": row["version_id"],
                "content_hash": row["content_hash"],
                "status": row["status"],
            }
            metadata = _json_loads(row["metadata_json"], {})
            if metadata:
                session["metadata"] = metadata
            target = _json_loads(row["target_json"], [])
            if target:
                session["target"] = target
            source_meta = _json_loads(row["source_meta_json"], {})
            if source_meta:
                session.update(source_meta)
            sessions.append(session)
        return sessions

    def _load_raw_docs(self, conn) -> dict[str, str]:
        rows = conn.execute(
            """
            SELECT rd.source_id, wl.content_text
            FROM raw_docs rd
            JOIN write_log wl ON wl.message_id = rd.message_id
            ORDER BY rd.sort_order
            """
        ).fetchall()
        return {str(row["source_id"]): str(row["content_text"] or "") for row in rows}

    def _load_episode_corpus(self, conn) -> dict:
        layout_row = conn.execute(
            "SELECT value_json FROM state_json WHERE name = ?",
            (INTERNAL_EPISODE_CORPUS_LAYOUT,),
        ).fetchone()
        corpus_layout = _json_loads(layout_row["value_json"] if layout_row else None, {})
        doc_order_row = conn.execute(
            "SELECT value_json FROM state_json WHERE name = ?",
            (INTERNAL_EPISODE_DOC_ORDER,),
        ).fetchone()
        doc_order = _json_loads(doc_order_row["value_json"] if doc_order_row else None, [])
        rows = conn.execute(
            "SELECT doc_id, episode_json FROM episode_corpus ORDER BY doc_id, sort_order"
        ).fetchall()
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(str(row["doc_id"]), []).append(_json_loads(row["episode_json"], {}))
        if not isinstance(corpus_layout, dict) or not corpus_layout:
            ordered_doc_ids = [doc_id for doc_id in doc_order if doc_id in grouped]
            ordered_doc_ids.extend(doc_id for doc_id in grouped if doc_id not in ordered_doc_ids)
            return {
                "documents": [
                    {"doc_id": doc_id, "episodes": grouped.get(doc_id, [])}
                    for doc_id in ordered_doc_ids
                ]
            }

        seen_doc_ids: list[str] = []
        corpus: dict[str, list[dict]] = {}
        for group_name, group_doc_ids in corpus_layout.items():
            if not isinstance(group_doc_ids, list):
                continue
            ordered_group_doc_ids = [doc_id for doc_id in group_doc_ids if doc_id in grouped]
            if group_name == "documents":
                ordered_group_doc_ids.extend(
                    doc_id
                    for doc_id in doc_order
                    if doc_id in grouped and doc_id not in ordered_group_doc_ids and doc_id not in seen_doc_ids
                )
            if not ordered_group_doc_ids:
                corpus[str(group_name)] = []
                continue
            seen_doc_ids.extend(doc_id for doc_id in ordered_group_doc_ids if doc_id not in seen_doc_ids)
            corpus[str(group_name)] = [
                {"doc_id": doc_id, "episodes": grouped.get(doc_id, [])}
                for doc_id in ordered_group_doc_ids
            ]

        remaining_doc_ids = [doc_id for doc_id in doc_order if doc_id in grouped and doc_id not in seen_doc_ids]
        remaining_doc_ids.extend(doc_id for doc_id in grouped if doc_id not in seen_doc_ids and doc_id not in remaining_doc_ids)
        documents = corpus.setdefault("documents", [])
        documents.extend(
            {"doc_id": doc_id, "episodes": grouped.get(doc_id, [])}
            for doc_id in remaining_doc_ids
        )
        return corpus

    def load_facts(self) -> dict:
        if not self.exists:
            return {}
        with self._connect() as conn:
            state_rows = conn.execute("SELECT name, value_json FROM state_json").fetchall()
            state = {str(row["name"]): _json_loads(row["value_json"], None) for row in state_rows}
            payload = {
                "granular": self._load_tier(conn, "granular"),
                "cons": self._load_tier(conn, "cons"),
                "cross": self._load_tier(conn, "cross"),
                "tlinks": [
                    _json_loads(row["link_json"], {})
                    for row in conn.execute("SELECT link_json FROM temporal_links ORDER BY sort_order").fetchall()
                ],
                "raw_sessions": self._load_raw_sessions(conn),
                "raw_docs": self._load_raw_docs(conn),
                "episode_corpus": self._load_episode_corpus(conn),
                "source_records": {},
            }
            for row in conn.execute("SELECT * FROM source_records ORDER BY source_id").fetchall():
                payload["source_records"][row["source_id"]] = {
                    "family": row["family"],
                    "owner_id": row["owner_id"],
                    "read": _json_loads(row["read_json"], []),
                    "write": _json_loads(row["write_json"], []),
                    "artifact_id": row["artifact_id"],
                    "version_id": row["version_id"],
                    "content_hash": row["content_hash"],
                    "metadata": _json_loads(row["metadata_json"], {}),
                    "target": _json_loads(row["target_json"], []),
                    "source_meta": _json_loads(row["source_meta_json"], {}),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            for key in STATE_JSON_KEYS:
                if key in {INTERNAL_EPISODE_DOC_ORDER, INTERNAL_EPISODE_CORPUS_LAYOUT}:
                    continue
                if key in state:
                    payload[key] = state[key]
            return payload

    def load_embeddings(self) -> dict | None:
        if not self.exists:
            return None
        with self._connect() as conn:
            count = int(conn.execute("SELECT COUNT(*) AS n FROM embeddings").fetchone()["n"])
            if count == 0:
                return None
            result: dict[str, np.ndarray] = {}
            for tier_key in ("gran", "cons", "cross"):
                rows = conn.execute(
                    "SELECT dim, dtype, vector_blob FROM embeddings WHERE tier = ? ORDER BY sort_order",
                    (tier_key,),
                ).fetchall()
                if not rows:
                    result[tier_key] = np.zeros((0, 3072), dtype=np.float32)
                    continue
                dim = int(rows[0]["dim"])
                dtype = np.dtype(rows[0]["dtype"])
                matrix = np.zeros((len(rows), dim), dtype=dtype)
                for idx, row in enumerate(rows):
                    matrix[idx] = np.frombuffer(row["vector_blob"], dtype=dtype, count=dim)
                result[tier_key] = matrix
            return result

    def save_embeddings(self, gran: np.ndarray, cons: np.ndarray, cross: np.ndarray) -> None:
        tier_map = {
            "gran": ("granular", gran),
            "cons": ("cons", cons),
            "cross": ("cross", cross),
        }
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM embeddings")
            for tier_key, (fact_tier, matrix) in tier_map.items():
                fact_rows = conn.execute(
                    "SELECT fact_id, sort_order FROM facts WHERE tier = ? ORDER BY sort_order",
                    (fact_tier,),
                ).fetchall()
                if len(fact_rows) != len(matrix):
                    raise ValueError(
                        f"Embedding count mismatch for {tier_key}: {len(matrix)} vectors, {len(fact_rows)} facts"
                    )
                batch: list[tuple[Any, ...]] = []
                for idx, row in enumerate(fact_rows):
                    vector = np.asarray(matrix[idx])
                    batch.append(
                        (
                            tier_key,
                            row["fact_id"],
                            row["sort_order"],
                            int(vector.shape[0]),
                            str(vector.dtype),
                            vector.tobytes(),
                        )
                    )
                    if len(batch) >= 1000:
                        conn.executemany(
                            "INSERT INTO embeddings(tier, fact_id, sort_order, dim, dtype, vector_blob) VALUES(?, ?, ?, ?, ?, ?)",
                            batch,
                        )
                        batch.clear()
                if batch:
                    conn.executemany(
                        "INSERT INTO embeddings(tier, fact_id, sort_order, dim, dtype, vector_blob) VALUES(?, ?, ?, ?, ?, ?)",
                        batch,
                    )
            conn.commit()

    def append_write_log(
        self,
        *,
        message_id: str,
        session_id: str,
        agent_id: str,
        swarm_id: str,
        visibility: str,
        owner_id: str | None,
        scope: str | None,
        read: list[str] | None,
        write: list[str] | None,
        content_family: str,
        content_text: str,
        metadata: dict | None,
        timestamp_ms: int,
    ) -> dict:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT message_id, extraction_state FROM write_log WHERE message_id = ?",
                (message_id,),
            ).fetchone()
            if existing is not None:
                conn.commit()
                return {
                    "message_id": message_id,
                    "extraction_state": existing["extraction_state"],
                    "inserted": False,
                }
            sort_order = int(
                conn.execute("SELECT COALESCE(MAX(sort_order), -1) + 1 AS next_sort FROM write_log").fetchone()["next_sort"]
            )
            conn.execute(
                """
                INSERT INTO write_log(
                    message_id, session_id, agent_id, swarm_id, visibility,
                    owner_id, scope, read_json, write_json, content_family,
                    content_text, metadata_json, timestamp_ms, extraction_state,
                    extraction_attempts, last_extraction_attempt_ms, sort_order
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, NULL, ?)
                """,
                (
                    message_id,
                    session_id,
                    agent_id,
                    swarm_id,
                    visibility,
                    owner_id,
                    scope,
                    _json_dumps(read or []),
                    _json_dumps(write or []),
                    content_family,
                    content_text,
                    _json_dumps(metadata or {}),
                    int(timestamp_ms),
                    sort_order,
                ),
            )
            conn.commit()
            return {"message_id": message_id, "extraction_state": "pending", "inserted": True}

    def get_write_status(self, message_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT message_id, extraction_state, extraction_attempts, last_extraction_attempt_ms, "
                "owner_id, scope, read_json, write_json, agent_id, swarm_id "
                "FROM write_log WHERE message_id = ?",
                (message_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "message_id": row["message_id"],
                "extraction_state": row["extraction_state"],
                "extraction_attempts": int(row["extraction_attempts"] or 0),
                "last_extraction_attempt_ms": row["last_extraction_attempt_ms"],
                "owner_id": row["owner_id"],
                "scope": row["scope"],
                "read": _json_loads(row["read_json"], []),
                "write": _json_loads(row["write_json"], []),
                "agent_id": row["agent_id"],
                "swarm_id": row["swarm_id"],
            }

    def list_write_log_entries(
        self,
        *,
        states: list[str] | None = None,
        swarm_id: str | None = None,
        order: str = "asc",
    ) -> list[dict]:
        states = states or ["pending", "in_progress", "failed"]
        clauses = [f"extraction_state IN ({','.join('?' for _ in states)})"]
        params: list[Any] = list(states)
        if swarm_id is not None:
            clauses.append("swarm_id = ?")
            params.append(swarm_id)
        order_sql = "ASC" if order.lower() == "asc" else "DESC"
        sql = (
            "SELECT message_id, session_id, agent_id, swarm_id, visibility, owner_id, scope, "
            "read_json, write_json, content_family, content_text, metadata_json, timestamp_ms, "
            "extraction_state, extraction_attempts, last_extraction_attempt_ms, sort_order "
            "FROM write_log WHERE " + " AND ".join(clauses) + f" ORDER BY sort_order {order_sql}, timestamp_ms {order_sql}"  # nosec B608
        )
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "message_id": row["message_id"],
                "session_id": row["session_id"],
                "agent_id": row["agent_id"],
                "swarm_id": row["swarm_id"],
                "visibility": row["visibility"],
                "owner_id": row["owner_id"],
                "scope": row["scope"],
                "read": _json_loads(row["read_json"], []),
                "write": _json_loads(row["write_json"], []),
                "content_family": row["content_family"],
                "content": row["content_text"],
                "metadata": _json_loads(row["metadata_json"], {}),
                "timestamp_ms": int(row["timestamp_ms"] or 0),
                "extraction_state": row["extraction_state"],
                "extraction_attempts": int(row["extraction_attempts"] or 0),
                "last_extraction_attempt_ms": row["last_extraction_attempt_ms"],
                "sort_order": int(row["sort_order"] or 0),
            }
            for row in rows
        ]

    def mark_write_state(self, message_id: str, state: str, *, attempts_delta: int = 0) -> None:
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE write_log
                SET extraction_state = ?,
                    extraction_attempts = extraction_attempts + ?,
                    last_extraction_attempt_ms = ?
                WHERE message_id = ?
                """,
                (state, attempts_delta, now_ms, message_id),
            )
            conn.commit()


def migrate_jsonnpz_to_sqlite(data_dir: str, key: str) -> dict:
    enc_key_hex = os.environ.get("GOSH_MEMORY_ENCRYPTION_KEY")
    enc_key = bytes.fromhex(enc_key_hex) if enc_key_hex else None
    data_dir_path = Path(data_dir)
    legacy = JSONNPZStorage(data_dir, key, encryption_key=enc_key)
    if not legacy.exists:
        raise FileNotFoundError(f"Legacy JSON storage not found for key={key!r}")

    snapshot = legacy.load_facts()
    corpus_path = data_dir_path / f"{key}_corpus.json"
    if corpus_path.exists():
        snapshot["episode_corpus"] = load_episode_corpus(corpus_path, strict=False)
    embeddings = legacy.load_embeddings()

    tmp_path = data_dir_path / f"{key}.sqlite3.tmp"
    final_path = data_dir_path / f"{key}.sqlite3"
    if final_path.exists():
        raise FileExistsError(
            f"SQLite storage already exists for key={key!r}; refusing to overwrite existing {final_path.name}"
        )
    if tmp_path.exists():
        tmp_path.unlink()
    sqlite_backend = SQLiteStorageBackend(data_dir, key, encryption_key=enc_key, db_path=tmp_path)
    try:
        sqlite_backend.save_facts(snapshot)
        if embeddings is not None:
            sqlite_backend.save_embeddings(
                embeddings.get("gran", np.zeros((0, 3072), dtype=np.float32)),
                embeddings.get("cons", np.zeros((0, 3072), dtype=np.float32)),
                embeddings.get("cross", np.zeros((0, 3072), dtype=np.float32)),
            )

        loaded_snapshot = sqlite_backend.load_facts()
        def _normalize_snapshot(payload: dict) -> dict:
            normalized = dict(payload)
            normalized.setdefault("granular", [])
            normalized.setdefault("cons", [])
            normalized.setdefault("cross", [])
            normalized.setdefault("tlinks", [])
            normalized.setdefault("raw_sessions", [])
            normalized.setdefault("raw_docs", {})
            normalized.setdefault("episode_corpus", {"documents": []})
            normalized.setdefault("source_records", {})

            normalized_raw_sessions: list[dict] = []
            for idx, raw_session in enumerate(normalized["raw_sessions"]):
                item = dict(raw_session)
                if not str(item.get("message_id") or "").strip():
                    item["message_id"] = sqlite_backend._legacy_message_id_for_session(
                        item,
                        fallback_idx=idx,
                    )
                normalized_raw_sessions.append(item)
            normalized["raw_sessions"] = normalized_raw_sessions

            normalized_source_records: dict[str, dict] = {}
            for source_id, record in (normalized["source_records"] or {}).items():
                item = dict(record)
                # Legacy snapshots may carry redundant top-level ids that are not part of
                # the SQLite row shape but do not affect runtime semantics.
                item.pop("source_id", None)
                item.pop("scope_id", None)
                item.setdefault("target", [])
                normalized_source_records[str(source_id)] = item
            normalized["source_records"] = normalized_source_records
            return normalized
        if _normalize_snapshot(loaded_snapshot) != _normalize_snapshot(snapshot):
            raise RuntimeError("SQLite migration verification failed: facts snapshot mismatch")
        loaded_embeddings = sqlite_backend.load_embeddings()
        if embeddings is None:
            if loaded_embeddings is not None:
                raise RuntimeError("SQLite migration verification failed: unexpected embeddings present")
        else:
            for tier in ("gran", "cons", "cross"):
                expected = embeddings.get(tier)
                got = loaded_embeddings.get(tier) if loaded_embeddings is not None else None
                if expected is None and got is None:
                    continue
                if expected is None or got is None:
                    raise RuntimeError(f"SQLite migration verification failed: embedding mismatch for {tier}")
                if len(expected) == 0 and len(got) == 0:
                    continue
                if expected.shape != got.shape:
                    raise RuntimeError(f"SQLite migration verification failed: embedding mismatch for {tier}")
                if not np.array_equal(expected, got):
                    raise RuntimeError(f"SQLite migration verification failed: embedding mismatch for {tier}")
    finally:
        sqlite_backend.close()

    os.replace(tmp_path, final_path)
    legacy_paths = [
        data_dir_path / f"{key}.json",
        data_dir_path / f"{key}_embs.npz",
        data_dir_path / f"{key}_corpus.json",
        data_dir_path / f"{key}_temporal.json",
    ]
    for path in legacy_paths:
        if path.exists():
            bak = path.with_name(path.name + ".bak")
            if bak.exists():
                bak.unlink()
            os.replace(path, bak)

    return {
        "key": key,
        "sqlite_path": str(final_path),
        "migrated": True,
        "embeddings": embeddings is not None,
    }


def make_storage(data_dir: str, key: str) -> StorageBackend:
    """Storage selection.

    New keys default to SQLite.
    Legacy keys keep JSONNPZStorage until migrated.
    Once {key}.sqlite3 exists, runtime uses SQLite only.
    """
    enc_key_hex = os.environ.get("GOSH_MEMORY_ENCRYPTION_KEY")
    enc_key = bytes.fromhex(enc_key_hex) if enc_key_hex else None
    data_dir_path = Path(data_dir)
    sqlite_path = data_dir_path / f"{key}.sqlite3"
    json_path = data_dir_path / f"{key}.json"
    if sqlite_path.exists() or not json_path.exists():
        return SQLiteStorageBackend(data_dir, key, encryption_key=enc_key)
    return JSONNPZStorage(data_dir, key, encryption_key=enc_key)
