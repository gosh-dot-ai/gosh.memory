import asyncio

import numpy as np
import pytest

import src.mcp_server as mcp_mod
from src.mcp_server import memory_get_config, memory_recall, memory_set_config, memory_set_profiles
from src.memory import MemoryServer

CONFIG = {
    "schema_version": 1,
    "embedding_model": "openai/text-embedding-3-large",
    "librarian_profile": "anthropic/claude-sonnet-4-6",
    "profiles": {
        1: "fast",
        2: "fast",
        3: "balanced",
        4: "strong",
        5: "strong",
    },
    "profile_configs": {
        "fast": {"model": "openai/gpt-4o-mini"},
        "balanced": {"model": "google/gemini-2.0-flash"},
        "strong": {"model": "anthropic/claude-sonnet-4-6"},
    },
    "retrieval": {
        "search_family": "auto",
        "default_token_budget": 4000,
    },
}


@pytest.fixture(autouse=True)
def reset_state(tmp_path):
    mcp_mod.data_dir = str(tmp_path)
    mcp_mod.registry.clear()
    mcp_mod.courier_registry.clear()
    yield


def test_memory_server_config_round_trip(tmp_path):
    server = MemoryServer(str(tmp_path), "cfg")
    asyncio.run(server.set_config(CONFIG))
    result = server.get_config()
    assert result["embedding_model"] == CONFIG["embedding_model"]
    assert result["librarian_profile"] == CONFIG["librarian_profile"]
    assert result["profiles"][3] == "balanced"
    assert result["profile_configs"]["strong"]["model"] == "anthropic/claude-sonnet-4-6"


def test_memory_server_config_persists_across_restart(tmp_path):
    server = MemoryServer(str(tmp_path), "cfg_restart")
    asyncio.run(server.set_config(CONFIG))

    restarted = MemoryServer(str(tmp_path), "cfg_restart")
    result = restarted.get_config()

    assert result["embedding_model"] == CONFIG["embedding_model"]
    assert result["librarian_profile"] == CONFIG["librarian_profile"]
    assert result["profiles"][5] == "strong"


def test_set_profiles_updates_only_inference_subset(tmp_path):
    server = MemoryServer(str(tmp_path), "cfg_subset")
    asyncio.run(server.set_config(CONFIG))

    asyncio.run(
        server.set_profiles(
            {1: "cheap", 2: "cheap", 3: "cheap", 4: "best", 5: "best"},
            {
                "cheap": {"model": "openai/gpt-4o-mini"},
                "best": {"model": "anthropic/claude-opus-4-6"},
            },
        )
    )

    result = server.get_config()
    assert result["embedding_model"] == CONFIG["embedding_model"]
    assert result["librarian_profile"] == CONFIG["librarian_profile"]
    assert result["profiles"][1] == "cheap"
    assert result["profile_configs"]["best"]["model"] == "anthropic/claude-opus-4-6"


def test_set_config_rejects_invalid_schema_version(tmp_path):
    server = MemoryServer(str(tmp_path), "cfg_invalid_schema")
    with pytest.raises(ValueError, match="schema_version must be 1"):
        asyncio.run(server.set_config({**CONFIG, "schema_version": 2}))


def test_set_config_rejects_unknown_top_level_key(tmp_path):
    server = MemoryServer(str(tmp_path), "cfg_invalid_key")
    with pytest.raises(ValueError, match="unknown memory config keys"):
        asyncio.run(server.set_config({**CONFIG, "unexpected": True}))


def test_set_config_rejects_bad_retrieval_budget(tmp_path):
    server = MemoryServer(str(tmp_path), "cfg_invalid_retrieval")
    bad = {
        **CONFIG,
        "retrieval": {
            "search_family": "auto",
            "default_token_budget": 0,
        },
    }
    with pytest.raises(ValueError, match="default_token_budget must be positive int"):
        asyncio.run(server.set_config(bad))


@pytest.mark.asyncio
async def test_memory_set_get_config_mcp_round_trip():
    response = await memory_set_config(key="cfg_mcp", config=CONFIG, agent_id="owner")
    assert response["status"] == "ok"
    result = await memory_get_config(key="cfg_mcp", agent_id="owner")
    assert result["schema_version"] == 1
    assert result["embedding_model"] == CONFIG["embedding_model"]


@pytest.mark.asyncio
async def test_memory_get_config_missing_instance_returns_not_found():
    result = await memory_get_config(key="missing_cfg", agent_id="owner")
    assert result["code"] == "NOT_FOUND"


@pytest.mark.asyncio
async def test_memory_recall_threads_token_budget():
    await memory_set_config(key="cfg_budget", config=CONFIG, agent_id="owner")
    server = mcp_mod.registry["cfg_budget"]
    captured = {}

    async def mock_recall(**kwargs):
        captured.update(kwargs)
        return {
            "context": "ctx",
            "retrieved": [],
            "query_type": "lookup",
            "complexity_hint": {"score": 0.1, "level": 1},
        }

    server.recall = mock_recall

    result = await memory_recall(
        key="cfg_budget",
        query="hello",
        agent_id="owner",
        token_budget=123,
    )

    assert captured["token_budget"] == 123
    assert result["token_estimate"] == 0


@pytest.mark.asyncio
async def test_memory_set_profiles_keeps_non_profile_config_fields():
    await memory_set_config(key="cfg_keep", config=CONFIG, agent_id="owner")
    await memory_set_profiles(
        key="cfg_keep",
        profiles={1: "cheap"},
        profile_configs={"cheap": {"model": "openai/gpt-4o-mini"}},
        agent_id="owner",
    )
    result = await memory_get_config(key="cfg_keep", agent_id="owner")
    assert result["embedding_model"] == CONFIG["embedding_model"]
    assert result["librarian_profile"] == CONFIG["librarian_profile"]
    assert result["profiles"] == {1: "cheap"}


@pytest.mark.asyncio
async def test_build_index_uses_runtime_embedding_dim_for_empty_tiers(monkeypatch, tmp_path):
    server = MemoryServer(str(tmp_path), "cfg_dim")
    await server.set_config({**CONFIG, "embedding_model": "custom/1536"})
    server._all_granular = [
        {"id": "g1", "fact": "alpha", "conv_id": "cfg_dim", "session": 1}
    ]
    server._all_cons = []
    server._all_cross = []

    async def fake_embed_texts(texts, **kwargs):
        return np.ones((len(texts), 1536), dtype=np.float32)

    monkeypatch.setattr("src.memory.embed_texts", fake_embed_texts)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda facts, lookup: None)

    await server.build_index()

    assert server._data_dict["atomic_embs"].shape[1] == 1536
    assert server._data_dict["cons_embs"].shape[1] == 1536
    assert server._data_dict["cross_embs"].shape[1] == 1536
