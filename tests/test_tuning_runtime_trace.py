from __future__ import annotations

import numpy as np
import pytest

import src.tuning as tuning
from src.episode_retrieval import resolve_selection_config
from src.memory import MemoryServer

DIM = 3072


def _patch_embeddings(monkeypatch):
    async def mock_embed_texts(texts, **kwargs):
        return np.random.randn(len(texts), DIM).astype(np.float32)

    async def mock_embed_query(text, **kwargs):
        return np.random.randn(DIM).astype(np.float32)

    monkeypatch.setattr("src.memory.embed_texts", mock_embed_texts)
    monkeypatch.setattr("src.memory.embed_query", mock_embed_query)


def _patch_extract(monkeypatch):
    async def mock_extract_session(**kwargs):
        text = kwargs.get("session_text", "")
        sn = kwargs.get("session_num", 1)
        facts = [{
            "id": "f0",
            "fact": text[:80] or "fallback fact",
            "kind": "fact",
            "entities": ["generic"],
            "tags": ["test"],
            "session": sn,
        }]
        return ("conv", sn, kwargs.get("session_date", "2024-06-01"), facts, [])

    async def mock_consolidate_session(**kwargs):
        return ("conv", kwargs.get("sn", 1), "2024-06-01", [])

    async def mock_cross_session_entity(**kwargs):
        return ("conv", kwargs.get("ename", "entity"), [])

    monkeypatch.setattr("src.memory.extract_session", mock_extract_session)
    monkeypatch.setattr("src.memory.consolidate_session", mock_consolidate_session)
    monkeypatch.setattr("src.memory.cross_session_entity", mock_cross_session_entity)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda facts, lookup: None)


def _patch_all(monkeypatch):
    _patch_embeddings(monkeypatch)
    _patch_extract(monkeypatch)


def test_selector_config_reads_from_tuning_json(tmp_path, monkeypatch):
    tuning_file = tmp_path / "tuning.json"
    tuning_file.write_text(
        """
        {
          "routing": {
            "ambiguous_family_fanout": 1
          },
          "packet": {
            "budget": 6000
          },
          "retrieval": {
            "selector": {
              "budget": 4321,
              "late_fusion_per_family": 5
            }
          }
        }
        """
    )
    monkeypatch.setattr(tuning, "TUNING_FILE", tuning_file)
    tuning._CACHE["mtime_ns"] = None

    selector = resolve_selection_config()
    assert selector["budget"] == 4321
    assert selector["late_fusion_per_family"] == 5
    assert selector["max_sources_per_family"] == 2
    runtime_tuning = tuning.get_runtime_tuning()
    assert runtime_tuning["routing"]["ambiguous_family_fanout"] == 1
    assert runtime_tuning["packet"]["budget"] == 6000


@pytest.mark.asyncio
async def test_recall_returns_structured_runtime_trace(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    server = MemoryServer(str(tmp_path), "trace_server")

    await server.store(
        "User: David has 3 apples.\nAssistant: noted",
        session_num=1,
        session_date="2024-06-01",
        source_id="chat-42",
    )
    await server.build_index()
    result = await server.recall("Where did David have 3 apples?")

    trace = result["runtime_trace"]
    assert trace["runtime"] == "episode"
    assert trace["scope"]["scope_id"] == "trace_server"
    assert "chat-42" in trace["scope"]["source_ids"]
    assert trace["family_first_pass"]["retrieval_families"] == ["conversation"]
    assert trace["family_first_pass"]["per_family"]
    assert trace["family_first_pass"]["per_family"][0]["retrieval_target"]
    assert trace["query"]["output_constraints"]["return_only"] is False
    assert "selection" in trace
    assert "late_fusion" in trace
    assert "cross_contamination" in trace
    assert trace["packet"]["retrieved_fact_count"] >= 1
    assert trace["packet"]["actual_injected_episode_count"] >= 1
    assert "tuning" in trace
