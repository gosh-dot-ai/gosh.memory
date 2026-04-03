"""Tests for benchmarks/eval_memory/ — eval runner and judge."""

import asyncio
import json
from datetime import datetime, timezone

import numpy as np
import pytest

from benchmarks.eval_memory.run import (
    _build_result,
    _doc_swarm,
    _inject_artifact,
    _map_scope,
    compute_scores,
    run_query,
)
from src.config import MemoryConfig
from src.memory import MemoryServer

DIM = 3072


def _patch_embeddings(monkeypatch):
    async def mock_embed_texts(texts, **kw):
        return np.random.randn(len(texts), DIM).astype(np.float32)

    async def mock_embed_query(text, **kw):
        return np.random.randn(DIM).astype(np.float32)

    monkeypatch.setattr("src.memory.embed_texts", mock_embed_texts)
    monkeypatch.setattr("src.memory.embed_query", mock_embed_query)


def _patch_resolve_supersession(monkeypatch):
    monkeypatch.setattr("src.memory.resolve_supersession", lambda f, l: None)


def _patch_llm(monkeypatch):
    async def mock_call_oai(model, prompt, **kw):
        return "Via Dorvu Pass Tuesday convoy"

    async def mock_call_judge(model, prompt, **kw):
        return "yes"

    monkeypatch.setattr("benchmarks.eval_memory.run.call_oai", mock_call_oai)
    monkeypatch.setattr("benchmarks.eval_memory.judge.call_judge", mock_call_judge)


def _patch_all(monkeypatch):
    _patch_embeddings(monkeypatch)
    _patch_resolve_supersession(monkeypatch)
    _patch_llm(monkeypatch)


def _make_art(**overrides):
    art = {
        "id": "ART-TEST",
        "canary": "KAR_TEST01",
        "kind": "decision",
        "scope": "project-shared",
        "project": "alpha",
        "author": "pm_alpha",
        "title": "Test artifact",
        "body": "Test body with canary KAR_TEST01",
        "summary": "Test summary",
        "tags": ["test"],
        "status": "active",
        "supersedes": None,
        "superseded_by": None,
        "date": "2024-03-20",
    }
    art.update(overrides)
    return art


# ── Tests ──

def test_inject_artifact_adds_to_granular(tmp_path):
    server = MemoryServer(str(tmp_path), "test1")
    art = _make_art()

    _inject_artifact(server, art)

    assert len(server._all_granular) == 1
    fact = server._all_granular[0]
    assert fact["canary"] == "KAR_TEST01"
    assert fact["kind"] == "decision"
    assert fact["scope"] == "swarm-shared"  # project-shared → swarm-shared
    assert fact["conv_id"] == "test1"
    assert "Test artifact" in fact["fact"]


def test_inject_artifact_scope_system_wide(tmp_path):
    srv_alpha = MemoryServer(str(tmp_path), "test_alpha", swarm_id="alpha")
    srv_beta = MemoryServer(str(tmp_path), "test_beta", swarm_id="beta")

    art = _make_art(scope="system-wide", project="alpha")
    _inject_artifact(srv_alpha, art)
    _inject_artifact(srv_beta, art)

    assert len(srv_alpha._all_granular) == 1
    assert len(srv_beta._all_granular) == 1
    assert srv_alpha._all_granular[0]["scope"] == "system-wide"
    assert srv_beta._all_granular[0]["scope"] == "system-wide"


def test_doc_swarm_from_frontmatter(tmp_path):
    content = "---\ntitle: Test Doc\nswarm_id: swarm_alpha\n---\nContent here."
    path = tmp_path / "DOC-TEST.md"
    path.write_text(content)

    assert _doc_swarm(content, path) == "swarm_alpha"


def test_doc_swarm_fallback(tmp_path):
    content = "# Test Document\n\nNo frontmatter here."
    path = tmp_path / "DOC-TEST.md"
    path.write_text(content)

    assert _doc_swarm(content, path) == "system"


def test_scope_isolation_query(tmp_path, monkeypatch):
    _patch_all(monkeypatch)

    server = MemoryServer(str(tmp_path), "test_iso", swarm_id="swarm1")
    art = _make_art(
        scope="agent-private",
        author="agent_A",
        project="swarm1",
        body="Secret plan PRIVATE_DATA_HERE",
    )
    _inject_artifact(server, art)
    server._save_cache()
    asyncio.run(server.build_index())

    # System server needs at least one fact for build_index
    sys_server = MemoryServer(str(tmp_path), "test_sys")
    _inject_artifact(sys_server, _make_art(id="ART-SYS", scope="system-wide",
                                           body="Public info"))
    sys_server._save_cache()
    asyncio.run(sys_server.build_index())

    servers = {"swarm1": server, "system": sys_server}

    query = {
        "id": "q_iso_1",
        "category": "scope_isolation",
        "query": "What is the secret plan?",
        "ground_truth": "ACCESS_DENIED",
        "requesting_agent_id": "agent_B",
        "requesting_swarm_id": "intruder_swarm",
        "private_keywords": ["PRIVATE_DATA_HERE"],
    }

    sem = asyncio.Semaphore(5)
    result = asyncio.run(run_query(query, servers, MemoryConfig(), sem))

    # Intruder swarm doesn't have the server with the fact → CORRECT (no leak)
    assert result["judge"] == "CORRECT"


def test_extraction_quality_canary_hit(tmp_path, monkeypatch):
    _patch_all(monkeypatch)

    server = MemoryServer(str(tmp_path), "test_canary", swarm_id="alpha")
    art = _make_art(canary="KAR_TEST01", body="Route via Dorvu Pass KAR_TEST01")
    _inject_artifact(server, art)
    server._save_cache()
    asyncio.run(server.build_index())

    servers = {"alpha": server, "system": MemoryServer(str(tmp_path), "test_sys2")}

    query = {
        "id": "q_ext_1",
        "category": "extraction_quality",
        "query": "What is the delivery route?",
        "ground_truth": "Via Dorvu Pass",
        "expected_canary": "KAR_TEST01",
        "source_artifacts": ["ART-TEST"],
    }
    art_lookup = {"ART-TEST": "alpha"}

    sem = asyncio.Semaphore(5)
    result = asyncio.run(run_query(query, servers, MemoryConfig(), sem, art_lookup))

    assert result["canary_hit"] is True
    assert result["judge"] == "CORRECT"


def test_extraction_quality_canary_miss(tmp_path, monkeypatch):
    _patch_all(monkeypatch)

    server = MemoryServer(str(tmp_path), "test_miss", swarm_id="alpha")
    art = _make_art(canary="KAR_OTHER", body="Some content without the target canary")
    _inject_artifact(server, art)
    server._save_cache()
    asyncio.run(server.build_index())

    servers = {"alpha": server, "system": MemoryServer(str(tmp_path), "test_sys3")}

    query = {
        "id": "q_ext_2",
        "category": "extraction_quality",
        "query": "What route?",
        "ground_truth": "Via Dorvu Pass",
        "expected_canary": "KAR_MISSING",
        "source_artifacts": ["ART-TEST"],
    }
    art_lookup = {"ART-TEST": "alpha"}

    sem = asyncio.Semaphore(5)
    result = asyncio.run(run_query(query, servers, MemoryConfig(), sem, art_lookup))

    assert result["canary_hit"] is False
    assert result["judge"] == "INCORRECT"


def test_compute_scores_scope_isolation():
    results = [
        {"category": "scope_isolation", "judge": "CORRECT", "canary_hit": True,
         "requires_canary": ""},
    ] * 8 + [
        {"category": "scope_isolation", "judge": "LEAK", "canary_hit": True,
         "requires_canary": ""},
    ] * 2

    scores, _ = compute_scores(results)
    assert scores["scope_isolation"]["leak_rate"] == 0.2
    assert scores["scope_isolation"]["score"] == 0.8


def test_compute_scores_overall():
    results = [
        # Category A: 4/5 correct = 0.8
        *[{"category": "cat_a", "judge": "CORRECT", "canary_hit": True,
           "requires_canary": ""}] * 4,
        {"category": "cat_a", "judge": "INCORRECT", "canary_hit": True,
         "requires_canary": ""},
        # Category B: 3/5 correct = 0.6
        *[{"category": "cat_b", "judge": "CORRECT", "canary_hit": True,
           "requires_canary": ""}] * 3,
        *[{"category": "cat_b", "judge": "INCORRECT", "canary_hit": True,
           "requires_canary": ""}] * 2,
    ]

    scores, overall = compute_scores(results)
    assert scores["cat_a"]["score"] == 0.8
    assert scores["cat_b"]["score"] == 0.6
    assert overall == 0.7


def test_build_result_shape():
    result = _build_result(
        query_id="q_001",
        category="single_hop_recall",
        question="What route?",
        ground_truth="Via Dorvu Pass",
        context="Some context",
        pred="Dorvu Pass route",
        judge="CORRECT",
        canary_hit=True,
        canary="KAR_001",
    )

    required_keys = {
        "query_id", "category", "question", "ground_truth",
        "context_len", "prediction", "judge", "canary_hit", "requires_canary",
    }
    assert required_keys == set(result.keys())
    assert result["query_id"] == "q_001"
    assert result["context_len"] == len("Some context")
    assert result["requires_canary"] == "KAR_001"
