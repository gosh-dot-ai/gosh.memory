from types import SimpleNamespace

import numpy as np
import pytest

from src.cli import cmd_migrate_sqlite
from src.storage import JSONNPZStorage, SQLiteStorageBackend


def test_cmd_migrate_sqlite_migrates_legacy_key(tmp_path, capsys):
    legacy = JSONNPZStorage(str(tmp_path), "legacy_key")
    legacy.save_facts({
        "granular": [{"id": "g1", "fact": "hello", "kind": "event"}],
        "cons": [],
        "cross": [],
        "tlinks": [],
        "raw_sessions": [],
        "raw_docs": {},
        "episode_corpus": {"documents": [{"doc_id": "d1", "episodes": [{"episode_id": "e1"}]}]},
        "n_sessions": 1,
    })
    gran = np.random.rand(1, 4).astype(np.float32)
    legacy.save_embeddings(gran, np.zeros((0, 4), dtype=np.float32), np.zeros((0, 4), dtype=np.float32))

    cmd_migrate_sqlite(SimpleNamespace(data_dir=str(tmp_path), key="legacy_key"))

    out = capsys.readouterr().out
    assert "Migrated key 'legacy_key' to SQLite" in out
    assert (tmp_path / "legacy_key.sqlite3").exists()
    assert (tmp_path / "legacy_key.json.bak").exists()
    storage = SQLiteStorageBackend(str(tmp_path), "legacy_key")
    assert storage.load_facts()["granular"][0]["fact"] == "hello"


def test_cmd_migrate_sqlite_noops_for_existing_sqlite(tmp_path, capsys):
    storage = SQLiteStorageBackend(str(tmp_path), "already_sqlite")
    storage.save_facts({"granular": []})

    cmd_migrate_sqlite(SimpleNamespace(data_dir=str(tmp_path), key="already_sqlite"))

    out = capsys.readouterr().out
    assert "already uses SQLite" in out


def test_cmd_migrate_sqlite_errors_when_legacy_missing(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cmd_migrate_sqlite(SimpleNamespace(data_dir=str(tmp_path), key="missing_key"))

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "Legacy JSON storage not found" in err
