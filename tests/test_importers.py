"""Tests for Unit 3 — Batch History Import (src/importers.py + MCP tool)."""

import json

import numpy as np
import pytest

from src.importers import (
    parse_conversation_json,
    parse_directory,
    parse_history,
    parse_text,
)

DIM = 3072


# ── conversation_json ──

def test_parse_conversation_json_messages_array():
    """Standard messages array format."""
    data = [{
        "created_at": "2024-03-01T10:00:00Z",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    }]
    sessions = parse_conversation_json(json.dumps(data))
    assert len(sessions) == 1
    assert sessions[0]["session_date"] == "2024-03-01"
    assert "User: Hello" in sessions[0]["content"]
    assert "Assistant: Hi there!" in sessions[0]["content"]


def test_parse_conversation_json_chat_messages_array():
    """chat_messages variant (also supported)."""
    data = [{
        "created_at": "2024-03-01T00:00:00Z",
        "chat_messages": [
            {"role": "human", "content": [{"type": "text", "text": "Question"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Answer"}]},
        ]
    }]
    sessions = parse_conversation_json(json.dumps(data))
    assert len(sessions) == 1
    assert "User: Question" in sessions[0]["content"]
    assert "Assistant: Answer" in sessions[0]["content"]


def test_parse_conversation_json_multiple_conversations():
    data = [
        {"created_at": "2024-01-01T00:00:00Z",
         "messages": [
             {"role": "user", "content": "Q1"},
             {"role": "assistant", "content": "A1"},
         ]},
        {"created_at": "2024-02-01T00:00:00Z",
         "messages": [
             {"role": "user", "content": "Q2"},
             {"role": "assistant", "content": "A2"},
         ]},
    ]
    sessions = parse_conversation_json(json.dumps(data))
    assert len(sessions) == 2
    assert sessions[0]["session_num"] == 1
    assert sessions[1]["session_num"] == 2


def test_parse_conversation_json_empty_messages_skipped():
    data = [
        {"created_at": "2024-01-01T00:00:00Z", "messages": []},
        {"created_at": "2024-02-01T00:00:00Z",
         "messages": [{"role": "user", "content": "Hello"}]},
    ]
    sessions = parse_conversation_json(json.dumps(data))
    assert len(sessions) == 1


def test_parse_conversation_json_unix_timestamp():
    """create_time as unix timestamp."""
    data = [{"create_time": 1704067200,  # 2024-01-01
             "messages": [{"role": "user", "content": "hi"}]}]
    sessions = parse_conversation_json(json.dumps(data))
    assert sessions[0]["session_date"] == "2024-01-01"


def test_parse_conversation_json_single_object():
    """Single conversation object (not array) is accepted."""
    data = {"created_at": "2024-05-01T00:00:00Z",
            "messages": [
                {"role": "user", "content": "test"},
                {"role": "assistant", "content": "ok"},
            ]}
    sessions = parse_conversation_json(json.dumps(data))
    assert len(sessions) == 1


def test_parse_conversation_json_content_blocks():
    """Content as list of type/text blocks."""
    data = [{"created_at": "2024-01-01T00:00:00Z",
             "messages": [
                 {"role": "user",
                  "content": [{"type": "text", "text": "block content"}]},
             ]}]
    sessions = parse_conversation_json(json.dumps(data))
    assert "block content" in sessions[0]["content"]


# ── Text ──

def test_parse_text_single_session():
    content = "User: I love hiking.\nAssistant: That's great!"
    sessions = parse_text(content)
    assert len(sessions) == 1
    assert sessions[0]["session_num"] == 1
    assert sessions[0]["content"] == content.strip()
    assert sessions[0]["speakers"] == "User and Assistant"


def test_parse_text_date_is_today():
    from datetime import datetime, timezone
    sessions = parse_text("some content")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert sessions[0]["session_date"] == today


# ── Directory ──

def test_parse_directory_multiple_files():
    content = (
        "---FILE: 2024-01-15-work.txt---\n"
        "Work notes here.\n"
        "---FILE: 2024-02-20-personal.txt---\n"
        "Personal notes here.\n"
    )
    sessions = parse_directory(content)
    assert len(sessions) == 2
    assert sessions[0]["session_date"] == "2024-01-15"
    assert sessions[1]["session_date"] == "2024-02-20"
    assert "Work notes" in sessions[0]["content"]
    assert "Personal notes" in sessions[1]["content"]


def test_parse_directory_no_date_in_filename():
    content = (
        "---FILE: notes.txt---\n"
        "Some notes.\n"
    )
    sessions = parse_directory(content)
    assert len(sessions) == 1
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert sessions[0]["session_date"] == today


def test_parse_directory_empty_files_skipped():
    content = (
        "---FILE: empty.txt---\n"
        "\n"
        "---FILE: real.txt---\n"
        "Real content.\n"
    )
    sessions = parse_directory(content)
    assert len(sessions) == 1
    assert "Real content" in sessions[0]["content"]


# ── Dispatch ──

def test_parse_history_unknown_format_raises():
    with pytest.raises(ValueError, match="Unknown source_format"):
        parse_history("slack", "content")


def test_parse_history_dispatches_correctly():
    sessions = parse_history("text", "Hello world")
    assert len(sessions) == 1


# ── Integration: import → memory ──

@pytest.fixture()
def _patch_llm_for_import(monkeypatch):
    async def mock_extract(**kwargs):
        sn = kwargs.get("session_num", 1)
        text = kwargs.get("session_text", "")
        facts = [
            {"id": f"f{sn}_{i}", "fact": f"{text[:50]} (fact {i})", "kind": "fact",
             "entities": ["User"], "tags": [], "session": sn}
            for i in range(2)
        ]
        return ("conv", sn, "2024-01-01", facts, [])

    monkeypatch.setattr("src.memory.extract_session", mock_extract)
    async def mock_consolidate(**kw):
        return ("conv", 1, "2024-01-01", [])
    async def mock_cross(**kw):
        return ("conv", "e", [])
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
async def test_mcp_import_history_conversation_json(tmp_path, _patch_llm_for_import):
    """memory_import_history must process conversation_json export end-to-end."""
    import src.mcp_server as mcp_mod
    mcp_mod.data_dir = str(tmp_path)
    mcp_mod.registry.clear()

    data = [{
        "created_at": "2024-03-01T00:00:00Z",
        "messages": [
            {"role": "user", "content": "I live in Seattle."},
            {"role": "assistant", "content": "Got it!"},
        ]
    }]
    result = await mcp_mod.memory_import_history(
        key="import_test",
        source_format="conversation_json",
        content=json.dumps(data),
        agent_id="a",
        swarm_id="sw",
        scope="agent-private",
    )
    assert result["sessions_processed"] == 1
    assert result["facts_extracted"] >= 0
    assert result["errors"] == []

    # Raw session must be stored
    server = mcp_mod._get_memory("import_test")
    assert len(server._raw_sessions) == 1
    assert "Seattle" in server._raw_sessions[0]["content"]


@pytest.mark.asyncio
async def test_mcp_import_history_unknown_format(tmp_path):
    import src.mcp_server as mcp_mod
    mcp_mod.data_dir = str(tmp_path)
    mcp_mod.registry.clear()
    result = await mcp_mod.memory_import_history(
        key="fmt_test", source_format="slack", content="data",
        agent_id="a", swarm_id="sw",
    )
    assert result["code"] == "UNKNOWN_FORMAT"


@pytest.mark.asyncio
async def test_mcp_import_history_parse_error(tmp_path):
    import src.mcp_server as mcp_mod
    mcp_mod.data_dir = str(tmp_path)
    mcp_mod.registry.clear()
    result = await mcp_mod.memory_import_history(
        key="err_test", source_format="conversation_json",
        content="this is not valid json",
        agent_id="a", swarm_id="sw",
    )
    assert result["code"] == "PARSE_ERROR"


@pytest.mark.asyncio
async def test_import_raw_sessions_are_reextractable(tmp_path, _patch_llm_for_import):
    """Imported sessions must be reextractable — same as store()."""
    import src.mcp_server as mcp_mod
    mcp_mod.data_dir = str(tmp_path)
    mcp_mod.registry.clear()

    data = [{"created_at": "2024-01-01T00:00:00Z",
             "messages": [
                 {"role": "user", "content": "My budget is $10,000."},
             ]}]
    await mcp_mod.memory_import_history(
        key="reextract_test", source_format="conversation_json",
        content=json.dumps(data), agent_id="a", swarm_id="sw",
    )
    server = mcp_mod._get_memory("reextract_test")
    assert len(server._raw_sessions) == 1
    result = await server.reextract()
    assert result["sessions"] == 1
