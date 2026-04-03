"""Tests for v1.4 — Prompt Registry + content_type + librarian_prompt."""

from pathlib import Path

import numpy as np
import pytest

from src.prompt_registry import BUILTIN_CONTENT_TYPES, BUILTIN_PROMPTS, PromptRegistry

DIM = 3072


# ── Mock fixture for tests that call store() with extraction ──

@pytest.fixture()
def _patch_llm(monkeypatch):
    async def mock_extract(**kwargs):
        sn = kwargs.get("session_num", 1)
        text = kwargs.get("session_text", "")
        facts = [
            {"id": f"f{sn}_{i}", "fact": f"{text[:60]} (fact {i})", "kind": "fact",
             "entities": ["User"], "tags": [], "session": sn}
            for i in range(3)
        ]
        return ("conv", sn, "2024-01-01", facts, [])

    async def mock_consolidate(**kwargs):
        return ("conv", 1, "2024-01-01", [])

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


# ── Unit 1: PromptRegistry ──

def test_builtin_prompts_all_present():
    """All 6 built-in content types must exist."""
    expected = {"default", "financial", "technical", "personal", "regulatory", "agent_trace"}
    assert expected == set(BUILTIN_CONTENT_TYPES)


def test_get_builtin_prompt(tmp_path):
    reg = PromptRegistry(str(tmp_path))
    prompt = reg.get("financial")
    assert len(prompt) > 50
    assert "{session_date}" in prompt


def test_get_unknown_falls_back_to_default(tmp_path):
    reg = PromptRegistry(str(tmp_path))
    prompt = reg.get("nonexistent_type")
    assert prompt == BUILTIN_PROMPTS["default"]


def test_set_and_get_custom_prompt(tmp_path):
    reg = PromptRegistry(str(tmp_path))
    reg.set("legal", "Extract legal obligations. Date: {session_date}.")
    result = reg.get("legal")
    assert "legal obligations" in result


def test_custom_overrides_builtin(tmp_path):
    reg = PromptRegistry(str(tmp_path))
    reg.set("financial", "My custom financial prompt. {session_date}.")
    result = reg.get("financial")
    assert "My custom financial prompt" in result


def test_custom_persists_to_file(tmp_path):
    reg = PromptRegistry(str(tmp_path))
    reg.set("medical", "Extract diagnoses. {session_date}.")
    path = tmp_path / "librarian_prompts" / "medical.md"
    assert path.exists()
    assert "Extract diagnoses" in path.read_text()


def test_custom_survives_reload(tmp_path):
    reg1 = PromptRegistry(str(tmp_path))
    reg1.set("custom_type", "Custom prompt. {session_date}.")
    reg2 = PromptRegistry(str(tmp_path))
    assert "Custom prompt" in reg2.get("custom_type")


def test_list_includes_builtins(tmp_path):
    reg = PromptRegistry(str(tmp_path))
    result = reg.list()
    types = {r["content_type"] for r in result}
    assert "default" in types
    assert "financial" in types
    assert "agent_trace" in types


def test_list_marks_custom_source(tmp_path):
    reg = PromptRegistry(str(tmp_path))
    reg.set("legal", "Legal prompt. {session_date}.")
    result = reg.list()
    legal = next(r for r in result if r["content_type"] == "legal")
    assert legal["source"] == "custom"


def test_list_marks_overridden_builtin_as_custom(tmp_path):
    reg = PromptRegistry(str(tmp_path))
    reg.set("financial", "Override. {session_date}.")
    result = reg.list()
    fin = next(r for r in result if r["content_type"] == "financial")
    assert fin["source"] == "custom"


def test_exists_builtin(tmp_path):
    reg = PromptRegistry(str(tmp_path))
    assert reg.exists("technical") is True
    assert reg.exists("nonexistent") is False


def test_exists_custom(tmp_path):
    reg = PromptRegistry(str(tmp_path))
    reg.set("mytype", "prompt. {session_date}.")
    assert reg.exists("mytype") is True


# ── Unit 2: content_type + librarian_prompt in store() ──

@pytest.mark.asyncio
async def test_memory_store_content_type_accepted(tmp_path, _patch_llm):
    """memory_store with content_type must not raise."""
    from src.memory import MemoryServer
    server = MemoryServer(data_dir=str(tmp_path), key="ct_test")
    result = await server.store(
        "User: I spent $50,000 on equipment.",
        1, "2024-01-01",
        content_type="financial",
    )
    assert "facts_extracted" in result


@pytest.mark.asyncio
async def test_store_runs_librarian(tmp_path, _patch_llm):
    """store() must run the normal extraction pipeline."""
    from src.memory import MemoryServer
    server = MemoryServer(data_dir=str(tmp_path), key="no_skip")
    result = await server.store("User: I have 3 cats.", 1, "2024-01-01")
    assert "facts_extracted" in result


@pytest.mark.asyncio
async def test_librarian_prompt_inline_accepted_agent_private(tmp_path, _patch_llm):
    """librarian_prompt with agent-private scope must not raise."""
    from src.memory import MemoryServer
    server = MemoryServer(data_dir=str(tmp_path), key="inline_test")
    result = await server.store(
        "User: My salary is $120k.",
        1, "2024-01-01",
        scope="agent-private",
        librarian_prompt="Extract only salary information. Date: {session_date}.",
    )
    assert "facts_extracted" in result


@pytest.mark.asyncio
async def test_librarian_prompt_scope_error_on_swarm_shared(tmp_path):
    """librarian_prompt must return LIBRARIAN_PROMPT_SCOPE_ERROR for swarm-shared."""
    from src.memory import MemoryServer
    server = MemoryServer(data_dir=str(tmp_path), key="scope_err")
    result = await server.store(
        "content", 1, "2024-01-01",
        scope="swarm-shared",
        librarian_prompt="Custom prompt. {session_date}.",
    )
    assert result["code"] == "LIBRARIAN_PROMPT_SCOPE_ERROR"


@pytest.mark.asyncio
async def test_librarian_prompt_scope_error_on_system_wide(tmp_path):
    """librarian_prompt must return LIBRARIAN_PROMPT_SCOPE_ERROR for system-wide."""
    from src.memory import MemoryServer
    server = MemoryServer(data_dir=str(tmp_path), key="scope_sys")
    result = await server.store(
        "content", 1, "2024-01-01",
        scope="system-wide",
        librarian_prompt="Custom prompt. {session_date}.",
    )
    assert result["code"] == "LIBRARIAN_PROMPT_SCOPE_ERROR"


# ── Unit 3: MCP tools ──

@pytest.mark.asyncio
async def test_mcp_memory_list_prompts(tmp_path):
    """MCP memory_list_prompts must return all 6 builtins."""
    import src.mcp_server as mcp_mod
    mcp_mod.data_dir = str(tmp_path)
    mcp_mod.registry.clear()
    result = await mcp_mod.memory_list_prompts(key="list_test")
    types = {p["content_type"] for p in result["prompts"]}
    assert {"default", "financial", "technical", "personal",
            "regulatory", "agent_trace"}.issubset(types)


@pytest.mark.asyncio
async def test_mcp_memory_get_prompt_builtin(tmp_path):
    import src.mcp_server as mcp_mod
    mcp_mod.data_dir = str(tmp_path)
    mcp_mod.registry.clear()
    result = await mcp_mod.memory_get_prompt(key="get_test", content_type="financial")
    assert result["source"] == "builtin"
    assert "{session_date}" in result["prompt"]


@pytest.mark.asyncio
async def test_mcp_memory_get_prompt_not_found(tmp_path):
    import src.mcp_server as mcp_mod
    mcp_mod.data_dir = str(tmp_path)
    mcp_mod.registry.clear()
    result = await mcp_mod.memory_get_prompt(key="miss_test", content_type="unknown")
    assert result["code"] == "PROMPT_NOT_FOUND"


@pytest.mark.asyncio
async def test_mcp_memory_set_and_get_prompt(tmp_path):
    import src.mcp_server as mcp_mod
    mcp_mod.data_dir = str(tmp_path)
    mcp_mod.registry.clear()
    await mcp_mod.memory_set_prompt(
        key="set_test",
        content_type="legal",
        prompt="Extract legal clauses. {session_date}.",
    )
    result = await mcp_mod.memory_get_prompt(key="set_test", content_type="legal")
    assert result["source"] == "custom"
    assert "legal clauses" in result["prompt"]


@pytest.mark.asyncio
async def test_mcp_memory_set_prompt_invalid_content_type(tmp_path):
    import src.mcp_server as mcp_mod
    mcp_mod.data_dir = str(tmp_path)
    mcp_mod.registry.clear()
    result = await mcp_mod.memory_set_prompt(
        key="inv_test", content_type="bad type!", prompt="x"
    )
    assert result["code"] == "INVALID_CONTENT_TYPE"


# ── Regression tests for review bugs ──

@pytest.mark.asyncio
async def test_bug1_upsert_matches_swarm_and_scope(tmp_path, _patch_llm):
    """BUG 1: upsert must match session_key + agent_id + swarm_id + scope.

    Two agents in different swarms with same session_key must NOT collide.
    """
    from src.memory import MemoryServer
    server = MemoryServer(data_dir=str(tmp_path), key="bug1")
    await server.store(
        "Swarm A state.", 1, "2024-01-01",
        agent_id="agent_x", swarm_id="swarm_a", scope="agent-private",
        upsert_by_key="status",
    )
    await server.store(
        "Swarm B state.", 1, "2024-01-01",
        agent_id="agent_x", swarm_id="swarm_b", scope="agent-private",
        upsert_by_key="status",
    )
    # Same agent, same key, different swarm → two separate raw sessions
    assert len(server._raw_sessions) == 2
    contents = {rs["content"] for rs in server._raw_sessions}
    assert "Swarm A state." in contents
    assert "Swarm B state." in contents


@pytest.mark.asyncio
async def test_bug2_ingest_document_raw_session_has_raw_session_id(tmp_path, _patch_llm):
    """BUG 2: ingest_document raw sessions must have raw_session_id."""
    from src.memory import MemoryServer
    server = MemoryServer(data_dir=str(tmp_path), key="bug2")
    await server.ingest_document(
        "Document content here.", source_id="doc1",
        agent_id="a", swarm_id="sw", scope="swarm-shared",
    )
    assert len(server._raw_sessions) >= 1
    for rs in server._raw_sessions:
        assert "raw_session_id" in rs, "raw session missing raw_session_id"
        assert len(rs["raw_session_id"]) > 0


@pytest.mark.asyncio
async def test_bug2_ingest_document_facts_have_raw_session_id(tmp_path, _patch_llm):
    """BUG 2: facts from ingest_document must be tagged with raw_session_id."""
    from src.memory import MemoryServer
    server = MemoryServer(data_dir=str(tmp_path), key="bug2b")
    await server.ingest_document(
        "Document content here.", source_id="doc1",
        agent_id="a", swarm_id="sw", scope="swarm-shared",
    )
    assert len(server._all_granular) > 0
    for f in server._all_granular:
        assert "raw_session_id" in f, "fact missing raw_session_id"
        assert f["raw_session_id"] == server._raw_sessions[0]["raw_session_id"]


@pytest.mark.asyncio
async def test_bug3_custom_prompt_unknown_placeholder_no_keyerror(tmp_path, _patch_llm):
    """BUG 3: custom prompt with unknown {topic} placeholder must not KeyError."""
    from src.memory import MemoryServer
    server = MemoryServer(data_dir=str(tmp_path), key="bug3")
    result = await server.store(
        "User: test content.", 1, "2024-01-01",
        scope="agent-private",
        librarian_prompt="Extract {topic} facts. Date: {session_date}. Extra: {unknown_var}.",
    )
    # Must not raise KeyError — unknown placeholders pass through
    assert "facts_extracted" in result


@pytest.mark.asyncio
async def test_gap_content_type_persisted_in_raw_session(tmp_path, _patch_llm):
    """Arch gap: content_type must be saved in raw session for reextract()."""
    from src.memory import MemoryServer
    server = MemoryServer(data_dir=str(tmp_path), key="gap_ct")
    await server.store(
        "User: spent $50k.", 1, "2024-01-01",
        content_type="financial",
    )
    assert len(server._raw_sessions) == 1
    assert server._raw_sessions[0]["content_type"] == "financial"


@pytest.mark.asyncio
async def test_gap_content_type_survives_cache_reload(tmp_path, _patch_llm):
    """Arch gap: content_type in raw session must survive cache roundtrip."""
    from src.memory import MemoryServer
    server = MemoryServer(data_dir=str(tmp_path), key="gap_reload")
    await server.store(
        "User: architecture decision.", 1, "2024-01-01",
        content_type="technical",
    )
    server2 = MemoryServer(data_dir=str(tmp_path), key="gap_reload")
    assert server2._raw_sessions[0]["content_type"] == "technical"
