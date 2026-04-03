"""Tests for common.py → providers.py wiring + setup CLI transparency."""

import argparse
import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest


@pytest.mark.asyncio
async def test_call_extract_routes_via_get_client(monkeypatch):
    """call_extract must route to correct client via _get_client."""
    called = {}

    class FakeUsage:
        prompt_tokens = 10
        completion_tokens = 20
        total_tokens = 30

    class FakeChoice:
        class message:
            content = '{"facts": [], "temporal_links": []}'

    class FakeResp:
        usage = FakeUsage()
        choices = [FakeChoice()]

    class FakeCompletions:
        async def create(self, **kw):
            called["model"] = kw.get("model")
            return FakeResp()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr("src.common._oai_async", FakeClient())

    from src.common import call_extract
    result = await call_extract("gpt-4.1-mini", "sys", "user")
    assert called.get("model") == "gpt-4.1-mini"
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_call_extract_salvages_json_from_markdown_fence(monkeypatch):
    """call_extract must salvage valid JSON wrapped in markdown fences."""

    class FakeUsage:
        prompt_tokens = 10
        completion_tokens = 20
        total_tokens = 30

    class FakeChoice:
        class message:
            content = """```json
{"facts": [], "temporal_links": []}
```"""

    class FakeResp:
        usage = FakeUsage()
        choices = [FakeChoice()]

    class FakeCompletions:
        async def create(self, **kw):
            return FakeResp()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr("src.common._oai_async", FakeClient())

    from src.common import call_extract

    result = await call_extract("gpt-4.1-mini", "sys", "user")
    assert result == {"facts": [], "temporal_links": []}


@pytest.mark.asyncio
async def test_call_extract_salvages_json_from_surrounding_text(monkeypatch):
    """call_extract must salvage valid JSON when the model adds prose around it."""

    class FakeUsage:
        prompt_tokens = 10
        completion_tokens = 20
        total_tokens = 30

    class FakeChoice:
        class message:
            content = (
                "Here is the extracted JSON:\\n"
                '{"facts": [{"local_id": "f1"}], "temporal_links": []}\\n'
                "Done."
            )

    class FakeResp:
        usage = FakeUsage()
        choices = [FakeChoice()]

    class FakeCompletions:
        async def create(self, **kw):
            return FakeResp()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr("src.common._oai_async", FakeClient())

    from src.common import call_extract

    result = await call_extract("gpt-4.1-mini", "sys", "user")
    assert result == {"facts": [{"local_id": "f1"}], "temporal_links": []}


def test_embed_texts_sync_calls_openai(monkeypatch):
    """embed_texts_sync must call oai_sync().embeddings.create."""
    calls = {}

    class FakeUsage:
        total_tokens = 10
    class FakeEmbData:
        def __init__(self): self.embedding = [0.0] * 1024
    class FakeResp:
        usage = FakeUsage()
        data = [FakeEmbData(), FakeEmbData()]

    class FakeEmbeddings:
        def create(self, **kw):
            calls["called"] = True
            calls["model"] = kw.get("model")
            return FakeResp()

    class FakeClient:
        embeddings = FakeEmbeddings()

    monkeypatch.setattr("src.common._oai_sync", FakeClient())

    from src.common import embed_texts_sync
    result = embed_texts_sync(["hello", "world"])
    assert calls.get("called")
    assert result.shape == (2, 1024)


@pytest.mark.asyncio
async def test_embed_query_async(monkeypatch):
    """embed_query (async) must call oai_async().embeddings.create."""
    calls = {}

    class FakeUsage:
        total_tokens = 5
    class FakeEmbData:
        def __init__(self): self.embedding = [0.0] * 1024
    class FakeResp:
        usage = FakeUsage()
        data = [FakeEmbData()]

    class FakeEmbeddings:
        async def create(self, **kw):
            calls["called"] = True
            return FakeResp()

    class FakeClient:
        embeddings = FakeEmbeddings()

    monkeypatch.setattr("src.common._oai_async", FakeClient())

    from src.common import embed_query
    result = await embed_query("test query")
    assert calls.get("called")
    assert result.shape == (1024,)


@pytest.mark.asyncio
async def test_embed_texts_empty_no_api_call():
    """embed_texts([]) must return zeros without calling any API."""
    from src.common import embed_texts
    result = await embed_texts([])
    assert result.shape[0] == 0


def test_bge_is_default_local_model():
    """BAAI/bge-large-en-v1.5 must be the default local embedding model."""
    import inspect

    from src import providers
    sig = inspect.signature(providers._get_st_model)
    default = sig.parameters["model_name"].default
    assert default == "BAAI/bge-large-en-v1.5", \
        f"Expected BAAI/bge-large-en-v1.5, got {default}"


# ── Fix 3: key source transparency ──

def test_setup_show_prints_key_source(capsys, monkeypatch, tmp_path):
    """--show must print which storage backend holds the API key."""
    from src import setup_store

    monkeypatch.setattr(setup_store, "CONFIG_DIR", tmp_path / ".gosh-memory")
    monkeypatch.setattr(setup_store, "CONFIG_FILE", tmp_path / ".gosh-memory" / "config.json")
    # Clear env var so config file path is visible
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    (tmp_path / ".gosh-memory").mkdir()
    setup_store.save_config({
        "provider": "openai",
        "models": {"extraction": "gpt-4.1-mini"},
        "api_keys": {"openai": "sk-test-xxx"},
    })

    from src.cli import cmd_setup
    args = argparse.Namespace(show=True, provider=None, api_key=None, embed_provider=None)
    cmd_setup(args)

    out = capsys.readouterr().out
    assert "config.json" in out
    assert "***" in out


def test_setup_show_prints_env_source(capsys, monkeypatch, tmp_path):
    """--show must detect API key from env var and report it."""
    from src import setup_store

    monkeypatch.setattr(setup_store, "CONFIG_DIR", tmp_path / ".gosh-memory")
    monkeypatch.setattr(setup_store, "CONFIG_FILE", tmp_path / ".gosh-memory" / "config.json")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-env-key")

    (tmp_path / ".gosh-memory").mkdir()
    setup_store.save_config({
        "provider": "groq",
        "models": {"extraction": "qwen/qwen3-32b"},
    })

    from src.cli import cmd_setup
    args = argparse.Namespace(show=True, provider=None, api_key=None, embed_provider=None)
    cmd_setup(args)

    out = capsys.readouterr().out
    assert "env var $GROQ_API_KEY" in out
    assert "***" in out


def test_inception_client_uses_mercury_alias_without_nameerror(monkeypatch):
    """Inception provider must boot cleanly via MERCURY_API_KEY alias."""
    from src import common

    created = {}

    class FakeClient:
        def __init__(self, **kw):
            created.update(kw)

    monkeypatch.setenv("MERCURY_API_KEY", "mercury-test-key")
    monkeypatch.delenv("INCEPTION_API_KEY", raising=False)
    monkeypatch.setattr(common, "_inception_async", None)
    monkeypatch.setattr(common, "AsyncOpenAI", FakeClient)

    client = common.inception_async()
    assert isinstance(client, FakeClient)
    assert created["base_url"] == "https://api.inceptionlabs.ai/v1"
    assert created["api_key"] == "mercury-test-key"
    assert created["timeout"] == common._OPENAI_TIMEOUT
