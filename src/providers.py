#!/usr/bin/env python3
"""Unified LLM and embedding provider layer.

Uses litellm for chat completions (automatic provider routing).
Uses sentence-transformers for local embeddings (no API key).
Resolves API keys from setup_store (env vars → config file).

Optional deps:
  pip install litellm                 # multi-provider LLM
  pip install sentence-transformers   # local embeddings
"""

import asyncio
import json
import os
from typing import Any, Optional

import numpy as np

from . import setup_store

# ── Lazy imports ──

_litellm = None
_st_model_cache = {}


def _get_litellm():
    global _litellm
    if _litellm is not None:
        return _litellm
    raise ImportError(
        "litellm has been disabled due to CVE-2026-33634 (supply chain attack, "
        "CVSS 9.4). Use direct provider SDKs via src/common.py instead. "
        "See: https://docs.litellm.ai/blog/security-update-march-2026"
    )


def _get_st_model(model_name: str = "BAAI/bge-large-en-v1.5"):
    if model_name not in _st_model_cache:
        from sentence_transformers import SentenceTransformer
        _st_model_cache[model_name] = SentenceTransformer(model_name)
    return _st_model_cache[model_name]


# ── Provider resolution ──

_PROVIDER_ENV = {
    "openai":    "OPENAI_API_KEY",
    "groq":      "GROQ_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google":    "GOOGLE_API_KEY",
    "inception": "INCEPTION_API_KEY",
}
_PROVIDER_ENV_ALIASES = {
    "inception": ("INCEPTION_API_KEY", "MERCURY_API_KEY"),
}

# Public alias for CLI/diagnostics
PROVIDER_ENV_VARS = _PROVIDER_ENV


def provider_from_model(model: str) -> str:
    """Extract provider from model name prefix.

    openai/gpt-oss-120b → groq (OpenAI-compat models on Groq)
    qwen/...            → groq
    anthropic/...       → anthropic
    google/...          → google
    gpt-4.1-mini        → openai (bare names)
    """
    if "/" in model:
        prefix = model.split("/")[0]
        if prefix in ("openai", "qwen"):
            return "groq"
        if prefix == "inception":
            return "inception"
        return prefix
    return "openai"


def _provider_env_names(provider: str) -> list[str]:
    aliases = list(_PROVIDER_ENV_ALIASES.get(provider, ()))
    primary = _PROVIDER_ENV.get(provider)
    if primary and primary not in aliases:
        aliases.insert(0, primary)
    return aliases


def _inject_key(model: str):
    """Inject API key into env if found in setup_store (non-fatal)."""
    provider = provider_from_model(model)
    key = setup_store.get_api_key(provider)
    if key:
        for env_name in _provider_env_names(provider):
            if env_name and not os.environ.get(env_name):
                os.environ[env_name] = key


def ensure_api_key(model: str) -> str:
    """Ensure the provider for a model has an API key. Returns the key."""
    provider = provider_from_model(model)
    key = setup_store.get_api_key(provider)
    if not key:
        env_name = _provider_env_names(provider)[0] if _provider_env_names(provider) else f"{provider.upper()}_API_KEY"
        raise RuntimeError(
            f"No API key for {provider}. "
            f"Run 'gosh-memory setup' or set ${env_name}"
        )
    # Inject into env so litellm/SDKs can find it
    for env_name in _provider_env_names(provider):
        if env_name and not os.environ.get(env_name):
            os.environ[env_name] = key
    return key


# ── LLM completion ──

async def acomplete(model: str, messages: list[dict[str, str]],
                    max_tokens: int = 300, temperature: float = 0.0,
                    json_mode: bool = False, timeout: int = 60) -> dict:
    """Unified async chat completion via litellm.

    Returns: {"content": str, "usage": {"prompt_tokens": int, "completion_tokens": int}}
    """
    _inject_key(model)
    lm = _get_litellm()

    kw = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "timeout": timeout,
    }
    if temperature is not None:
        kw["temperature"] = temperature
    if json_mode:
        kw["response_format"] = {"type": "json_object"}

    response = await lm.acompletion(**kw)

    usage = {}
    if hasattr(response, "usage") and response.usage:
        usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
        }

    return {
        "content": response.choices[0].message.content.strip(),
        "usage": usage,
    }


async def aextract(model: str, system: str, user_msg: str,
                   max_tokens: int = 8192) -> dict:
    """Extraction call — returns parsed JSON dict via litellm."""
    result = await acomplete(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    from .common import parse_json_response
    return parse_json_response(result["content"])


# ── Embeddings ──

def embed(texts: list[str], model: str = "text-embedding-3-large",
          provider: str = "openai") -> np.ndarray:
    """Batch embed texts.

    provider="openai" → OpenAI API (via litellm or direct)
    provider="local"  → sentence-transformers (no API key needed)
    """
    if not texts:
        dim = 384 if provider == "local" else 3072
        return np.zeros((0, dim))

    texts = [t if t.strip() else "[empty]" for t in texts]

    if provider == "local":
        st = _get_st_model(model)
        return np.array(st.encode(texts, convert_to_numpy=True))

    # OpenAI embedding via litellm
    _inject_key("openai")
    lm = _get_litellm()
    response = lm.embedding(model=model, input=texts)
    return np.array([e["embedding"] for e in response.data])


def embed_one(text: str, model: str = "text-embedding-3-large",
              provider: str = "openai") -> np.ndarray:
    """Embed a single text string."""
    return embed([text], model=model, provider=provider)[0]
