"""Unit 1: Raw sessions storage + memory_reextract."""

import asyncio
import json

import numpy as np
import pytest

from src.identity import content_hash_text
from src.memory import MemoryServer

DIM = 3072


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch):
    async def mock_extract(**kwargs):
        sn = kwargs.get("session_num", 1)
        text = kwargs.get("session_text", "")
        facts = [
            {"id": f"f{sn}_{i}", "fact": f"{text[:40]} (fact {i})", "kind": "event",
             "entities": [], "tags": [], "session": sn}
            for i in range(3)
        ]
        return ("conv", sn, "2024-06-01", facts, [])

    async def mock_consolidate(**kwargs):
        return ("conv", 1, "2024-06-01", [])

    async def mock_cross(**kwargs):
        return ("conv", "e", [])

    monkeypatch.setattr("src.memory.extract_session", mock_extract)
    monkeypatch.setattr("src.memory.consolidate_session", mock_consolidate)
    monkeypatch.setattr("src.memory.cross_session_entity", mock_cross)
    async def _aembed_texts(texts, **kw):
        return np.random.randn(len(texts), DIM).astype(np.float32)
    async def _aembed_query(text, **kw):
        return np.random.randn(DIM).astype(np.float32)
    monkeypatch.setattr("src.memory.embed_texts", _aembed_texts)
    monkeypatch.setattr("src.memory.embed_query", _aembed_query)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda f, l: None)


@pytest.mark.asyncio
async def test_raw_session_stored_on_store(tmp_path):
    """store() must persist raw content before extraction."""
    server = MemoryServer(data_dir=str(tmp_path), key="raw_test")
    content = "User: I spent 70 hours playing Assassin's Creed Odyssey."
    await server.store(content, 1, "2024-01-01")
    assert len(server._raw_sessions) == 1
    assert server._raw_sessions[0]["content"] == content
    assert server._raw_sessions[0]["format"] == "conversation"
    assert server._raw_sessions[0]["session_num"] == 1
    assert server._raw_sessions[0]["status"] == "active"


@pytest.mark.asyncio
async def test_raw_session_stored_with_per_call_identity(tmp_path):
    """Raw session must record per-call agent_id/swarm_id/scope, not instance defaults."""
    server = MemoryServer(data_dir=str(tmp_path), key="raw_identity",
                          agent_id="default_agent", swarm_id="default_swarm")
    content = "User: I moved to Seattle last month."
    await server.store(content, 1, "2024-01-01",
                       agent_id="agent_x", swarm_id="sw1", scope="agent-private")
    assert server._raw_sessions[0]["agent_id"] == "agent_x"
    assert server._raw_sessions[0]["swarm_id"] == "sw1"
    assert server._raw_sessions[0]["scope"] == "agent-private"


@pytest.mark.asyncio
async def test_raw_sessions_survive_cache_roundtrip(tmp_path):
    """raw_sessions must survive save -> reload cycle."""
    server = MemoryServer(data_dir=str(tmp_path), key="raw_persist")
    content = "User: I have 38 pre-1920 American coins in my collection."
    await server.store(content, 1, "2024-01-01")

    server2 = MemoryServer(data_dir=str(tmp_path), key="raw_persist")
    assert len(server2._raw_sessions) == 1
    assert server2._raw_sessions[0]["content"] == content


@pytest.mark.asyncio
async def test_zero_fact_store_still_persists_raw_session_to_disk(tmp_path, monkeypatch):
    async def _zero_fact_extract(**kwargs):
        return ("conv", kwargs.get("session_num", 1), "2024-01-01", [], [])

    monkeypatch.setattr("src.memory.extract_session", _zero_fact_extract)

    server = MemoryServer(data_dir=str(tmp_path), key="raw_zero_fact")
    content = "User: giant multipart session that currently extracts no facts."

    result = await server.store(content, 1, "2024-01-01")

    assert result == {"facts_extracted": 0}
    assert len(server._raw_sessions) == 1
    assert server._raw_sessions[0]["content"] == content
    assert server._raw_sessions[0]["status"] == "active"

    server2 = MemoryServer(data_dir=str(tmp_path), key="raw_zero_fact")
    assert len(server2._raw_sessions) == 1
    assert server2._raw_sessions[0]["content"] == content
    assert server2._raw_sessions[0]["status"] == "active"


@pytest.mark.asyncio
async def test_reextract_zero_fact_session_stays_active(tmp_path, monkeypatch):
    async def _zero_fact_extract(**kwargs):
        return ("conv", kwargs.get("session_num", 1), "2024-01-01", [], [])

    server = MemoryServer(data_dir=str(tmp_path), key="reextract_zero_fact_active")
    await server.store("User: I adopted a cat.", 1, "2024-01-01")

    monkeypatch.setattr("src.memory.extract_session", _zero_fact_extract)

    result = await server.reextract()

    assert result["sessions"] == 1
    assert result["reextracted"] == 0
    assert server._raw_sessions[0]["status"] == "active"


@pytest.mark.asyncio
async def test_raw_session_stored_on_ingest_document(tmp_path):
    """ingest_document() must persist raw chunks."""
    server = MemoryServer(data_dir=str(tmp_path), key="raw_doc")
    content = "This is a short document about water infrastructure."
    await server.ingest_document(content, source_id="DOC-001")
    assert len(server._raw_sessions) >= 1
    assert server._raw_sessions[0]["format"] == "document"
    assert server._raw_sessions[0]["source_id"] == "DOC-001"


@pytest.mark.asyncio
async def test_reextract_replaces_facts_preserves_raw(tmp_path):
    """reextract() must clear facts and re-extract; raw sessions unchanged."""
    server = MemoryServer(data_dir=str(tmp_path), key="reextract_test")
    content = "User: I have 38 pre-1920 American coins in my collection."
    await server.store(content, 1, "2024-01-01")
    original_raw = server._raw_sessions[0]["content"]

    result = await server.reextract()
    assert result["sessions"] == 1
    assert "reextracted" in result
    # Raw sessions unchanged
    assert len(server._raw_sessions) == 1
    assert server._raw_sessions[0]["content"] == original_raw


@pytest.mark.asyncio
async def test_reextract_restores_pending_raw_session_to_active(tmp_path):
    server = MemoryServer(data_dir=str(tmp_path), key="reextract_status")
    await server.store("User: I adopted a cat.", 1, "2024-01-01")
    server._raw_sessions[0]["status"] = "pending_reextract"

    result = await server.reextract()

    assert result["sessions"] == 1
    assert server._raw_sessions[0]["status"] == "active"


@pytest.mark.asyncio
async def test_zero_fact_retry_does_not_poison_dedup_or_supersede_old_version(tmp_path, monkeypatch):
    server = MemoryServer(data_dir=str(tmp_path), key="dedup_zero_fact")
    original = "User: I researched adoption agencies."
    updated = "User: I researched adoption agencies and family law."

    first = await server.store(original, 1, "2024-01-01", source_id="SRC-1")
    assert first["facts_extracted"] == 3
    original_version = server._dedup_index[("SRC-1", 1)]["version_id"]
    original_hash = server._dedup_index[("SRC-1", 1)]["content_hash"]

    async def _zero_fact_extract(**kwargs):
        return ("conv", kwargs.get("session_num", 1), "2024-01-01", [], [])

    monkeypatch.setattr("src.memory.extract_session", _zero_fact_extract)

    second = await server.store(updated, 1, "2024-01-02", source_id="SRC-1")
    assert second == {"facts_extracted": 0}
    assert server._dedup_index[("SRC-1", 1)]["version_id"] == original_version
    assert server._dedup_index[("SRC-1", 1)]["content_hash"] == original_hash
    active_original = [
        rs for rs in server._raw_sessions
        if rs.get("version_id") == original_version and rs.get("status") == "active"
    ]
    assert active_original

    server2 = MemoryServer(data_dir=str(tmp_path), key="dedup_zero_fact")
    assert server2._dedup_index[("SRC-1", 1)]["version_id"] == original_version
    assert server2._dedup_index[("SRC-1", 1)]["content_hash"] == content_hash_text(original)

    async def _success_extract(**kwargs):
        sn = kwargs.get("session_num", 1)
        text = kwargs.get("session_text", "")
        facts = [
            {"id": f"f{sn}_{i}", "fact": f"{text[:40]} (fact {i})", "kind": "event",
             "entities": [], "tags": [], "session": sn}
            for i in range(3)
        ]
        return ("conv", sn, "2024-06-01", facts, [])

    monkeypatch.setattr("src.memory.extract_session", _success_extract)

    third = await server2.store(updated, 1, "2024-01-02", source_id="SRC-1")
    assert third["facts_extracted"] == 3
    assert server2._dedup_index[("SRC-1", 1)]["content_hash"] == content_hash_text(updated)


@pytest.mark.asyncio
async def test_reextract_empty_returns_error(tmp_path):
    """reextract() on empty server returns error dict, not exception."""
    server = MemoryServer(data_dir=str(tmp_path), key="reextract_empty")
    result = await server.reextract()
    assert "error" in result


@pytest.mark.asyncio
async def test_mcp_memory_reextract_no_sessions(tmp_path):
    """MCP memory_reextract returns NO_RAW_SESSIONS code when empty."""
    import src.mcp_server as mcp_mod
    mcp_mod.data_dir = str(tmp_path)
    mcp_mod.registry.clear()
    mcp_mod.courier_registry.clear()
    mcp_mod.connections.clear()
    mcp_mod.sub_to_conn.clear()
    result = await mcp_mod.memory_reextract(key="empty_key")
    assert result.get("code") == "NO_RAW_SESSIONS"


@pytest.mark.asyncio
async def test_old_cache_without_raw_sessions_loads_cleanly(tmp_path):
    """Cache without raw_sessions field must load without error (backward compat)."""
    cache = {
        "granular": [], "cons": [], "cross": [], "tlinks": [],
        "n_sessions": 0, "n_sessions_with_facts": 0
        # intentionally no raw_sessions
    }
    (tmp_path / "compat_test.json").write_text(json.dumps(cache))
    server = MemoryServer(data_dir=str(tmp_path), key="compat_test")
    assert server._raw_sessions == []
