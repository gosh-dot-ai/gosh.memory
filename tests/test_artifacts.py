"""Tests for run-scoped artifact helpers."""

import json

from src.artifacts import append_jsonl, create_run_dir, latest_run_dir, write_json_atomic


def test_create_run_dir_is_append_only(tmp_path):
    root = tmp_path / "artifacts"
    run1 = create_run_dir(root, "selection")
    run2 = create_run_dir(root, "selection")

    assert run1 != run2
    assert run1.exists()
    assert run2.exists()
    manifest = (root / "run_manifest.jsonl").read_text().strip().splitlines()
    assert len(manifest) == 2
    assert latest_run_dir(root) == run2


def test_atomic_and_jsonl_writes(tmp_path):
    out_json = tmp_path / "result.json"
    out_jsonl = tmp_path / "partial.jsonl"

    write_json_atomic(out_json, {"ok": True, "n": 1})
    append_jsonl(out_jsonl, {"qid": "Q-1", "score": 5})
    append_jsonl(out_jsonl, {"qid": "Q-2", "score": 3})

    assert json.loads(out_json.read_text()) == {"ok": True, "n": 1}
    lines = out_jsonl.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["qid"] == "Q-1"
    assert json.loads(lines[1])["qid"] == "Q-2"
