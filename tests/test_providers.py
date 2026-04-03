"""Tests for src/providers.py — unified LLM and embedding provider layer."""

import asyncio
import os
import types
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src import providers, setup_store

# ── provider_from_model ──

def test_provider_from_model_bare_names():
    """Bare model names (no prefix) -> openai."""
    assert providers.provider_from_model("gpt-4.1-mini") == "openai"
    assert providers.provider_from_model("text-embedding-3-large") == "openai"


def test_provider_from_model_groq_prefixes():
    """openai/ and qwen/ prefixes route to groq."""
    assert providers.provider_from_model("openai/gpt-oss-120b") == "groq"
    assert providers.provider_from_model("qwen/qwen3-32b") == "groq"


def test_provider_from_model_other_providers():
    """anthropic/ and google/ prefixes route correctly."""
    assert providers.provider_from_model("anthropic/claude-sonnet-4-6") == "anthropic"
    assert providers.provider_from_model("google/gemini-2.5-pro") == "google"
    assert providers.provider_from_model("inception/mercury-2") == "inception"


# ── ensure_api_key ──

def test_ensure_api_key_from_env(monkeypatch):
    """ensure_api_key finds key from env var."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")

    key = providers.ensure_api_key("gpt-4.1-mini")
    assert key == "sk-test-123"


def test_ensure_api_key_from_inception_alias_env(monkeypatch):
    """Mercury alias env var must satisfy inception provider."""
    monkeypatch.delenv("INCEPTION_API_KEY", raising=False)
    monkeypatch.setenv("MERCURY_API_KEY", "mercury-test-123")

    key = providers.ensure_api_key("inception/mercury-2")
    assert key == "mercury-test-123"
    assert os.environ.get("INCEPTION_API_KEY") == "mercury-test-123"


def test_ensure_api_key_missing_raises(monkeypatch, tmp_path):
    """ensure_api_key raises RuntimeError when no key found."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(setup_store, "CONFIG_FILE", tmp_path / "config.json")

    with pytest.raises(RuntimeError, match="No API key for groq"):
        providers.ensure_api_key("openai/gpt-oss-120b")


def test_ensure_api_key_from_config_file(monkeypatch, tmp_path):
    """ensure_api_key finds key from config file and injects into env."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(setup_store, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(setup_store, "CONFIG_FILE", tmp_path / "config.json")

    setup_store.store_api_key("groq", "gsk-from-config")

    key = providers.ensure_api_key("openai/gpt-oss-120b")
    assert key == "gsk-from-config"
    assert os.environ.get("GROQ_API_KEY") == "gsk-from-config"

    # Cleanup
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


# ── acomplete ──

def test_acomplete_calls_litellm(monkeypatch):
    """acomplete delegates to litellm.acompletion."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    # Mock litellm
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5

    mock_choice = MagicMock()
    mock_choice.message.content = "  Hello world  "

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    mock_lm = MagicMock()
    mock_lm.acompletion = AsyncMock(return_value=mock_response)
    mock_lm.drop_params = True

    # Inject mock
    providers._litellm = mock_lm

    try:
        result = asyncio.run(providers.acomplete(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Hi"}],
        ))
        assert result["content"] == "Hello world"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        mock_lm.acompletion.assert_called_once()
    finally:
        providers._litellm = None


# ── aextract ──

def test_aextract_returns_parsed_json(monkeypatch):
    """aextract returns parsed JSON from LLM response."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    mock_choice = MagicMock()
    mock_choice.message.content = '```json\n{"facts": [{"fact": "test"}]}\n```'

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None

    mock_lm = MagicMock()
    mock_lm.acompletion = AsyncMock(return_value=mock_response)
    mock_lm.drop_params = True

    providers._litellm = mock_lm

    try:
        result = asyncio.run(providers.aextract(
            model="gpt-4.1-mini",
            system="Extract facts.",
            user_msg="Alice met Bob.",
        ))
        assert "facts" in result
        assert result["facts"][0]["fact"] == "test"
    finally:
        providers._litellm = None


# ── embed ──

def test_embed_empty_texts():
    """embed([]) returns empty array with correct dimensions."""
    result = providers.embed([], provider="openai")
    assert result.shape == (0, 3072)

    result_local = providers.embed([], provider="local")
    assert result_local.shape == (0, 384)


def test_embed_local_sentence_transformers(monkeypatch):
    """embed with provider=local uses sentence-transformers."""
    mock_st = MagicMock()
    mock_st.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    providers._st_model_cache["all-MiniLM-L6-v2"] = mock_st

    try:
        result = providers.embed(
            ["hello", "world"],
            model="all-MiniLM-L6-v2",
            provider="local",
        )
        assert result.shape == (2, 3)
        mock_st.encode.assert_called_once()
    finally:
        providers._st_model_cache.clear()


def test_embed_openai_via_litellm(monkeypatch):
    """embed with provider=openai uses litellm.embedding."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    mock_data = [
        {"embedding": [0.1] * 3072},
        {"embedding": [0.2] * 3072},
    ]
    mock_response = MagicMock()
    mock_response.data = mock_data

    mock_lm = MagicMock()
    mock_lm.embedding.return_value = mock_response
    mock_lm.drop_params = True

    providers._litellm = mock_lm

    try:
        result = providers.embed(["hello", "world"], provider="openai")
        assert result.shape == (2, 3072)
        mock_lm.embedding.assert_called_once()
    finally:
        providers._litellm = None


def test_embed_one_returns_1d(monkeypatch):
    """embed_one returns a 1D array."""
    mock_st = MagicMock()
    mock_st.encode.return_value = np.array([[0.1, 0.2, 0.3]])

    providers._st_model_cache["all-MiniLM-L6-v2"] = mock_st

    try:
        result = providers.embed_one("hello", model="all-MiniLM-L6-v2", provider="local")
        assert result.ndim == 1
        assert len(result) == 3
    finally:
        providers._st_model_cache.clear()


def test_embed_replaces_empty_strings(monkeypatch):
    """Empty strings replaced with [empty] before embedding."""
    mock_st = MagicMock()
    mock_st.encode.return_value = np.array([[0.1, 0.2]])

    providers._st_model_cache["all-MiniLM-L6-v2"] = mock_st

    try:
        providers.embed(["", "  "], model="all-MiniLM-L6-v2", provider="local")
        call_args = mock_st.encode.call_args[0][0]
        assert call_args == ["[empty]", "[empty]"]
    finally:
        providers._st_model_cache.clear()
