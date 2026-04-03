"""Tests for embedding provider wiring — local vs OpenAI path isolation."""

import numpy as np
import pytest


@pytest.mark.asyncio
async def test_local_embed_does_not_call_openai(monkeypatch):
    """When embed_provider=local, OpenAI client must NOT be called."""
    from src import common

    monkeypatch.setattr("src.common.get_config", lambda: {
        "embed_provider": "local",
        "embed_model": "test-model",
    })

    openai_called = False

    # Mock oai_async to track calls
    class FakeEmbeddings:
        async def create(self, **kw):
            nonlocal openai_called
            openai_called = True
            raise RuntimeError("Should not be called")

    class FakeClient:
        embeddings = FakeEmbeddings()

    monkeypatch.setattr("src.common.oai_async", lambda: FakeClient())

    # Mock _embed_local to return fake embeddings
    monkeypatch.setattr("src.common._embed_local",
                        lambda texts, model: np.random.randn(len(texts), 384))

    result = await common.embed_batch(["test"], provider="local")
    assert not openai_called
    assert result.shape[0] == 1


@pytest.mark.asyncio
async def test_openai_embed_uses_openai_client(monkeypatch):
    """When embed_provider=openai, must use OpenAI async client."""
    from src import common

    monkeypatch.setattr("src.common.get_config", lambda: {
        "embed_provider": "openai",
        "embed_model": "text-embedding-3-large",
    })

    # Mock OpenAI to return fake embeddings
    class FakeUsage:
        total_tokens = 10

    class FakeEmbData:
        def __init__(self):
            self.embedding = [0.1] * 10

    class FakeEmbResp:
        data = [FakeEmbData()]
        usage = FakeUsage()

    class FakeEmbeddings:
        async def create(self, **kw):
            return FakeEmbResp()

    class FakeClient:
        embeddings = FakeEmbeddings()

    monkeypatch.setattr("src.common.oai_async", lambda: FakeClient())

    result = await common.embed_batch(["test"])
    assert result.shape == (1, 10)


@pytest.mark.asyncio
async def test_embed_batch_explicit_provider_overrides_config(monkeypatch):
    """Explicit provider= arg must override config file setting."""
    from src import common

    # Config says openai, but we pass provider="local" explicitly
    monkeypatch.setattr("src.common.get_config", lambda: {
        "embed_provider": "openai",
        "embed_model": "text-embedding-3-large",
    })

    openai_called = False

    class FakeEmbeddings:
        async def create(self, **kw):
            nonlocal openai_called
            openai_called = True
            raise RuntimeError("Should not be called")

    class FakeClient:
        embeddings = FakeEmbeddings()

    monkeypatch.setattr("src.common.oai_async", lambda: FakeClient())
    monkeypatch.setattr("src.common._embed_local",
                        lambda texts, model: np.random.randn(len(texts), 384))

    result = await common.embed_batch(["test"], provider="local")
    assert not openai_called
    assert result.shape == (1, 384)


def test_sync_local_embed_does_not_call_openai(monkeypatch):
    """Sync wrapper: local provider must not touch OpenAI."""
    from src import common

    monkeypatch.setattr("src.common.get_config", lambda: {
        "embed_provider": "local",
        "embed_model": "test-model",
    })

    openai_called = False

    class FakeEmbeddings:
        def create(self, **kw):
            nonlocal openai_called
            openai_called = True
            raise RuntimeError("Should not be called")

    class FakeClient:
        embeddings = FakeEmbeddings()

    monkeypatch.setattr("src.common.oai_sync", lambda: FakeClient())
    monkeypatch.setattr("src.common._embed_local",
                        lambda texts, model: np.random.randn(len(texts), 384))

    result = common.embed_batch_sync(["test"], provider="local")
    assert not openai_called
    assert result.shape == (1, 384)


def test_sync_openai_embed_uses_openai(monkeypatch):
    """Sync wrapper: openai provider must use oai_sync()."""
    from src import common

    monkeypatch.setattr("src.common.get_config", lambda: {
        "embed_provider": "openai",
        "embed_model": "text-embedding-3-large",
    })

    class FakeUsage:
        total_tokens = 5

    class FakeEmbData:
        def __init__(self):
            self.embedding = [0.2] * 8

    class FakeResp:
        data = [FakeEmbData()]
        usage = FakeUsage()

    class FakeEmbeddings:
        def create(self, **kw):
            return FakeResp()

    class FakeClient:
        embeddings = FakeEmbeddings()

    monkeypatch.setattr("src.common.oai_sync", lambda: FakeClient())

    result = common.embed_batch_sync(["test"], provider="openai")
    assert result.shape == (1, 8)


def test_resolve_embed_config_defaults():
    """_resolve_embed_config returns sane defaults when no config."""
    from src.common import _resolve_embed_config

    # When get_config fails, should still return defaults
    model, provider = _resolve_embed_config()
    assert provider in ("openai", "local")
    assert model is not None


@pytest.mark.asyncio
async def test_embed_query_passes_provider(monkeypatch):
    """embed_query must pass provider through to embed_batch."""
    from src import common

    monkeypatch.setattr("src.common.get_config", lambda: {
        "embed_provider": "local",
        "embed_model": "test-model",
    })
    monkeypatch.setattr("src.common._embed_local",
                        lambda texts, model: np.random.randn(len(texts), 384))

    result = await common.embed_query("test", provider="local")
    assert result.shape == (384,)


def test_embed_query_sync_passes_provider(monkeypatch):
    """embed_query_sync must pass provider through."""
    from src import common

    monkeypatch.setattr("src.common.get_config", lambda: {
        "embed_provider": "local",
        "embed_model": "test-model",
    })
    monkeypatch.setattr("src.common._embed_local",
                        lambda texts, model: np.random.randn(len(texts), 384))

    result = common.embed_query_sync("test", provider="local")
    assert result.shape == (384,)
