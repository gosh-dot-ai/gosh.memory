"""Tests for src/setup_store.py — config and key storage."""

import json
import os
from pathlib import Path

import pytest

from src import setup_store


def test_save_and_load_config(tmp_path, monkeypatch):
    """Config round-trips through save -> load."""
    monkeypatch.setattr(setup_store, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(setup_store, "CONFIG_FILE", tmp_path / "config.json")

    setup_store.save_config({
        "provider": "groq",
        "models": {"extraction": "qwen/qwen3-32b"},
    })

    loaded = setup_store.load_config()
    assert loaded["provider"] == "groq"
    assert loaded["models"]["extraction"] == "qwen/qwen3-32b"


def test_load_config_missing_file(tmp_path, monkeypatch):
    """Missing config file returns empty dict."""
    monkeypatch.setattr(setup_store, "CONFIG_FILE", tmp_path / "nope.json")
    assert setup_store.load_config() == {}


def test_config_creates_directory(tmp_path, monkeypatch):
    """save_config creates parent directory if missing."""
    nested = tmp_path / "sub" / "dir"
    monkeypatch.setattr(setup_store, "CONFIG_DIR", nested)
    monkeypatch.setattr(setup_store, "CONFIG_FILE", nested / "config.json")

    setup_store.save_config({"test": True})
    assert (nested / "config.json").exists()


def test_store_and_get_api_key_via_config(tmp_path, monkeypatch):
    """API key round-trips through config file."""
    monkeypatch.setattr(setup_store, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(setup_store, "CONFIG_FILE", tmp_path / "config.json")

    # Clear env var so it doesn't shadow config file
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    setup_store.store_api_key("openai", "sk-test-key-123")
    assert setup_store.get_api_key("openai") == "sk-test-key-123"


def test_get_api_key_env_first(monkeypatch, tmp_path):
    """Env var takes priority over config file."""
    monkeypatch.setattr(setup_store, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(setup_store, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

    assert setup_store.get_api_key("openai") == "sk-from-env"


def test_get_api_key_supports_inception_alias_env(monkeypatch, tmp_path):
    monkeypatch.setattr(setup_store, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.delenv("INCEPTION_API_KEY", raising=False)
    monkeypatch.setenv("MERCURY_API_KEY", "mercury-env-key")

    assert setup_store.get_api_key("inception") == "mercury-env-key"


def test_get_api_key_returns_none_when_nothing(monkeypatch, tmp_path):
    """Returns None when no env var and no config file key."""
    monkeypatch.setattr(setup_store, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    assert setup_store.get_api_key("groq") is None


def test_delete_api_key(tmp_path, monkeypatch):
    """delete_api_key removes from config file."""
    monkeypatch.setattr(setup_store, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(setup_store, "CONFIG_FILE", tmp_path / "config.json")
    # Clear env so fallback doesn't kick in
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    setup_store.store_api_key("anthropic", "sk-ant-xxx")
    assert setup_store.get_api_key("anthropic") == "sk-ant-xxx"

    setup_store.delete_api_key("anthropic")
    assert setup_store.get_api_key("anthropic") is None


def test_is_configured(tmp_path, monkeypatch):
    """is_configured returns True only when provider is set."""
    monkeypatch.setattr(setup_store, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(setup_store, "CONFIG_FILE", tmp_path / "config.json")

    assert not setup_store.is_configured()

    setup_store.save_config({"provider": "groq"})
    assert setup_store.is_configured()


def test_store_api_key_adds_to_config(tmp_path, monkeypatch):
    """store_api_key writes key to config.json under api_keys dict."""
    monkeypatch.setattr(setup_store, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(setup_store, "CONFIG_FILE", tmp_path / "config.json")

    setup_store.save_config({
        "provider": "openai",
        "models": {"extraction": "gpt-4.1-mini"},
    })

    setup_store.store_api_key("openai", "sk-test-123")

    content = json.loads((tmp_path / "config.json").read_text())
    assert content["api_keys"]["openai"] == "sk-test-123"


def test_list_configured_providers(tmp_path, monkeypatch):
    """Lists providers with available keys."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    # Use empty config so file-based api_keys don't interfere
    monkeypatch.setattr(setup_store, "CONFIG_FILE", tmp_path / "config.json")

    providers = setup_store.list_configured_providers()
    assert "groq" in providers
    assert "openai" not in providers
