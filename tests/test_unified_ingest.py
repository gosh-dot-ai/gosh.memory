"""Unified ingest + full-corpus episode path tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

import src.mcp_server as mcp_mod
from src.ingest import ingest_input
from src.mcp_server import memory_ingest
from src.memory import MemoryServer
from src.source_detect import detect_source_family

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
        if "3 apples" in text:
            fact_text = "David has 3 apples."
            entities = ["David"]
        elif "Goliath" in text:
            fact_text = "Biblical David fought Goliath."
            entities = ["David", "Goliath"]
        elif "Route 1 final approved length" in text:
            fact_text = "Route 1 final approved length is 14.3 km."
            entities = ["Route 1"]
        else:
            fact_text = text[:80] or "fallback fact"
            entities = ["generic"]
        facts = [{
            "id": "f0",
            "fact": fact_text,
            "kind": "fact",
            "entities": entities,
            "tags": ["test"],
            "session": sn,
        }]
        return ("conv", sn, kwargs.get("session_date", "2024-06-01"), facts, [])

    async def mock_consolidate_session(**kwargs):
        sn = kwargs.get("sn", 1)
        return ("conv", sn, "2024-06-01", [{
            "id": f"c{sn}",
            "fact": f"Consolidated fact {sn}",
            "kind": "summary",
            "entities": [],
            "tags": [],
        }])

    async def mock_cross_session_entity(**kwargs):
        return ("conv", kwargs.get("ename", "entity"), [])

    monkeypatch.setattr("src.memory.extract_session", mock_extract_session)
    monkeypatch.setattr("src.memory.consolidate_session", mock_consolidate_session)
    monkeypatch.setattr("src.memory.cross_session_entity", mock_cross_session_entity)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda facts, lookup: None)


def _patch_answer(monkeypatch):
    async def mock_call_oai(model, prompt, max_tokens=300, **kwargs):
        if "David has 3 apples." in prompt:
            return "3"
        if "Route 1 final approved length is 14.3 km." in prompt:
            return "14.3 km"
        return "unknown"

    monkeypatch.setattr("src.memory.call_oai", mock_call_oai)


def _patch_all(monkeypatch):
    _patch_embeddings(monkeypatch)
    _patch_extract(monkeypatch)
    _patch_answer(monkeypatch)


def test_detect_source_family_conversation():
    family, evidence = detect_source_family("User: hello\nAssistant: hi")
    assert family == "conversation"
    assert evidence["signals"]


def test_detect_source_family_document():
    family, evidence = detect_source_family("# Overview\n\nRoute 1 final approved length is 14.3 km.")
    assert family == "document"
    assert evidence["signals"]


def test_detect_source_family_transcript_bundle_is_document():
    family, evidence = detect_source_family(
        "Dialogue 1:\n\n"
        "System: hello there\n"
        "User: I need a horror movie suggestion\n\n"
        "Dialogue 2:\n\n"
        "System: welcome back\n"
        "User: I need a comedy movie suggestion\n"
    )
    assert family == "document"
    assert "transcript_bundle_headings=2" in evidence["signals"]


@pytest.mark.asyncio
async def test_ingest_input_text_routes_to_conversation_and_persists_episodes(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    server = MemoryServer(str(tmp_path), "ingest_text_server")

    result = await ingest_input(
        server,
        text="User: David has 3 apples.\nAssistant: noted",
        session_num=1,
        session_date="2024-06-01",
    )

    assert result["source_family"] == "conversation"
    assert server._all_granular
    assert server._episode_corpus["documents"]
    fact = server._all_granular[0]
    assert fact["metadata"]["episode_id"].startswith("ingest_text_server_e")
    assert fact["metadata"]["episode_source_id"] == "ingest_text_server"
    assert server._scope_record["scope_id"] == "ingest_text_server"
    assert "ingest_text_server" in server._scope_record["source_ids"]
    assert server._source_records["ingest_text_server"]["family"] == "conversation"


@pytest.mark.asyncio
async def test_ingest_input_does_not_override_detected_document_when_session_fields_present(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    server = MemoryServer(str(tmp_path), "conversation_hint_server")

    result = await ingest_input(
        server,
        text="[Step 0] Action: left\nObservation: moved left",
        session_num=7,
        session_date="2024-06-01",
        speakers="Game",
    )

    assert result["source_family"] == "document"
    assert "conversation_fields_present" in result["detection_evidence"]["signals"]
    assert "step_trace_text" in result["detection_evidence"]["signals"]


@pytest.mark.asyncio
async def test_store_forces_conversation_format(tmp_path, monkeypatch):
    seen = {}

    async def mock_extract_session(**kwargs):
        seen["fmt"] = kwargs.get("fmt")
        sn = kwargs.get("session_num", 1)
        facts = [{
            "id": "f0",
            "fact": "David has 3 apples.",
            "kind": "fact",
            "entities": ["David"],
            "tags": ["test"],
            "session": sn,
        }]
        return ("conv", sn, kwargs.get("session_date", "2024-06-01"), facts, [])

    async def mock_consolidate_session(**kwargs):
        sn = kwargs.get("sn", 1)
        return ("conv", sn, "2024-06-01", [])

    async def mock_cross_session_entity(**kwargs):
        return ("conv", kwargs.get("ename", "entity"), [])

    async def mock_embed_texts(texts, **kwargs):
        return np.random.randn(len(texts), DIM).astype(np.float32)

    async def mock_embed_query(text, **kwargs):
        return np.random.randn(DIM).astype(np.float32)

    monkeypatch.setattr("src.memory.extract_session", mock_extract_session)
    monkeypatch.setattr("src.memory.consolidate_session", mock_consolidate_session)
    monkeypatch.setattr("src.memory.cross_session_entity", mock_cross_session_entity)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda facts, lookup: None)
    monkeypatch.setattr("src.memory.embed_texts", mock_embed_texts)
    monkeypatch.setattr("src.memory.embed_query", mock_embed_query)

    server = MemoryServer(str(tmp_path), "force_conversation")
    await server.store(
        "This text might look like a document heading\\n\\n## Header",
        session_num=1,
        session_date="2024-06-01",
    )

    assert seen["fmt"] == "CONVERSATION"


@pytest.mark.asyncio
async def test_ingest_input_path_routes_to_document_episode_pipeline(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    server = MemoryServer(str(tmp_path), "ingest_path_server")
    doc = tmp_path / "route.md"
    doc.write_text(
        "# Overview\n\n"
        "Route 1 final approved length is 14.3 km.\n\n"
        "## Appendix\n\n"
        "Reference table follows."
    )

    result = await ingest_input(server, path=str(doc), source_id="DOC-ROUTE")

    assert result["source_family"] == "document"
    assert result["facts_extracted"] >= 1
    cached = server._storage.load_facts()
    assert cached["episode_corpus"]["documents"]
    docs = server._episode_corpus["documents"]
    assert any(doc["doc_id"] == "document:DOC-ROUTE" for doc in docs)
    assert all(f["metadata"]["episode_source_id"] == "DOC-ROUTE" for f in server._all_granular)
    assert "DOC-ROUTE" in server._scope_record["source_ids"]
    assert server._source_records["DOC-ROUTE"]["family"] == "document"


@pytest.mark.asyncio
async def test_store_preserves_explicit_conversation_source_id_in_episode_runtime(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    server = MemoryServer(str(tmp_path), "conv_scope_server")

    await server.store(
        "User: David has 3 apples.\nAssistant: noted",
        session_num=1,
        session_date="2024-06-01",
        source_id="chat-42",
    )

    fact = server._all_granular[0]
    assert fact["metadata"]["episode_source_id"] == "chat-42"
    assert fact["source_id"] == "chat-42"
    docs = server._episode_corpus["documents"]
    assert any(doc["doc_id"] == "conversation:chat-42" for doc in docs)
    assert server._source_records["chat-42"]["family"] == "conversation"


@pytest.mark.asyncio
async def test_scope_and_source_records_round_trip(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    server = MemoryServer(str(tmp_path), "scope_roundtrip")

    await server.store(
        "User: David has 3 apples.\nAssistant: noted",
        session_num=1,
        session_date="2024-06-01",
        source_id="conv-alpha",
    )
    await server.ingest_document(
        "# Overview\n\nRoute 1 final approved length is 14.3 km.",
        source_id="DOC-ALPHA",
    )

    reloaded = MemoryServer(str(tmp_path), "scope_roundtrip")
    assert reloaded._scope_record["scope_id"] == "scope_roundtrip"
    assert set(reloaded._scope_record["source_ids"]) == {"DOC-ALPHA", "conv-alpha"}
    assert reloaded._source_records["conv-alpha"]["family"] == "conversation"
    assert reloaded._source_records["DOC-ALPHA"]["family"] == "document"


@pytest.mark.asyncio
async def test_facts_only_store_uses_generic_fact_runtime_without_rehydrating_episodes(tmp_path, monkeypatch):
    _patch_embeddings(monkeypatch)
    server = MemoryServer(str(tmp_path), "facts_only_store")
    server._all_granular = [{
        "id": "f1",
        "fact": "David has 3 apples.",
        "kind": "fact",
        "entities": ["David"],
        "tags": ["test"],
        "session": 1,
        "conv_id": "facts_only_store",
        "agent_id": "default",
        "swarm_id": "default",
        "scope": "swarm-shared",
        "owner_id": "system",
        "read": ["agent:PUBLIC"],
        "write": ["agent:PUBLIC"],
        "created_at": "2024-01-01T00:00:00+00:00",
        "_session_content_complexity": 0.1,
    }]
    await server.build_index()

    result = await server.recall("How many apples does David have?")

    assert "code" not in result
    assert result["runtime_trace"]["runtime"] == "fact"
    assert "David has 3 apples." in result["context"]
    assert result["retrieved"]


@pytest.mark.asyncio
async def test_ask_uses_generic_fact_runtime_for_facts_only_store(tmp_path, monkeypatch):
    _patch_embeddings(monkeypatch)
    _patch_answer(monkeypatch)
    server = MemoryServer(str(tmp_path), "facts_only_ask")
    server._all_granular = [{
        "id": "f1",
        "fact": "David has 3 apples.",
        "kind": "fact",
        "entities": ["David"],
        "tags": ["test"],
        "session": 1,
        "conv_id": "facts_only_ask",
        "agent_id": "default",
        "swarm_id": "default",
        "scope": "swarm-shared",
        "owner_id": "system",
        "read": ["agent:PUBLIC"],
        "write": ["agent:PUBLIC"],
        "created_at": "2024-01-01T00:00:00+00:00",
        "_session_content_complexity": 0.1,
    }]
    await server.build_index()

    result = await server.ask("How many apples does David have?", inference_model="test-model")

    assert result["runtime_trace"]["runtime"] == "fact"
    assert result["answer"] == "3"


@pytest.mark.asyncio
async def test_memory_ingest_mcp_accepts_path(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    mcp_mod.data_dir = str(tmp_path)
    mcp_mod.registry.clear()
    mcp_mod.courier_registry.clear()
    mcp_mod.connections.clear()
    mcp_mod.sub_to_conn.clear()
    mcp_mod._active_connections.clear()

    doc = tmp_path / "doc.md"
    doc.write_text("# Report\n\nRoute 1 final approved length is 14.3 km.")

    result = await memory_ingest(
        key="mcp_ingest_doc",
        path=str(doc),
        source_id="DOC-MCP",
    )

    assert result["source_family"] == "document"
    server = mcp_mod.registry["mcp_ingest_doc"]
    assert server._episode_corpus["documents"]
    assert server._all_granular


@pytest.mark.asyncio
async def test_ask_searches_full_mixed_corpus_without_source_hint(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    server = MemoryServer(str(tmp_path), "mixed_corpus")

    await server.store(
        "User: David has 3 apples.\nAssistant: noted",
        session_num=1,
        session_date="2024-06-01",
    )
    await server.ingest_document(
        content="# Chronicle\n\nBiblical David fought Goliath.",
        source_id="DOC-BIBLE",
    )

    result = await server.ask(
        "How many apples does David have?",
        inference_model="test-model",
    )

    assert result["answer"] == "3"
    recall = await server.recall("How many apples does David have?")
    assert "David has 3 apples." in recall["context"]
    assert "retrieved_episode_ids" in recall
