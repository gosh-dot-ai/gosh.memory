#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GOSH Memory — shared utilities.

Extracted from sprint-17c/s17c_full_scale.py + sprint-9/common.py.
"""

import asyncio
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
from openai import AsyncOpenAI, OpenAI

try:
    import simplemma
    HAS_SIMPLEMMA = True
except ImportError:
    simplemma = None
    HAS_SIMPLEMMA = False

# providers — optional (litellm + sentence-transformers may not be installed)
try:
    from .providers import aextract as _aextract
    from .providers import embed as _embed
    from .providers import embed_one as _embed_one
    HAS_PROVIDERS = True
except ImportError:
    HAS_PROVIDERS = False

try:
    from .setup_store import get_config
except ImportError:
    def get_config(): return {}

# ── Constants ──

STOP_WORDS = {"did", "we", "the", "a", "an", "is", "are", "was", "were",
              "what", "which", "how", "about", "does", "do", "for",
              "of", "in", "on", "to", "and", "or", "not", "no", "now",
              "that", "this", "it", "its", "there", "know", "make",
              "has", "have", "had", "be", "been", "being", "with",
              "you", "your", "they", "their", "she", "her", "he", "his",
              "who", "whom", "when", "where", "why", "can", "could",
              "would", "should", "will", "shall", "may", "might"}
_OPENAI_TIMEOUT = float(os.getenv("GOSH_OPENAI_TIMEOUT", os.getenv("OPENAI_TIMEOUT", "300")))

# ── Lazy API clients ──

_oai_sync = None
_oai_async = None
_groq_async = None
_inception_async = None
_anthropic_client = None
_google_client = None


def oai_sync():
    global _oai_sync
    if _oai_sync is None:
        _oai_sync = OpenAI()
    return _oai_sync


def oai_async():
    global _oai_async
    if _oai_async is None:
        _oai_async = AsyncOpenAI()
    return _oai_async


def groq_async():
    global _groq_async
    if _groq_async is None:
        _groq_async = AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY", ""),
        )
    return _groq_async


def inception_async():
    global _inception_async
    if _inception_async is None:
        _inception_async = AsyncOpenAI(
            base_url="https://api.inceptionlabs.ai/v1",
            api_key=os.environ.get("INCEPTION_API_KEY") or os.environ.get("MERCURY_API_KEY", ""),
            timeout=_OPENAI_TIMEOUT,
            max_retries=0,
        )
    return _inception_async


def anthropic_async():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    return _anthropic_client


def google_client():
    global _google_client
    if _google_client is None:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        _google_client = genai
    return _google_client


_GROQ_PREFIXES = ("openai/", "groq/", "qwen/", "meta-llama/", "moonshotai/", "canopylabs/")
_INCEPTION_PREFIXES = ("inception/",)


def _get_client(model):
    """Return the right async client based on model name prefix."""
    if any(model.startswith(p) for p in _GROQ_PREFIXES):
        return groq_async()
    if any(model.startswith(p) for p in _INCEPTION_PREFIXES):
        return inception_async()
    if model.startswith("anthropic/"):
        return anthropic_async()
    if model.startswith("google/"):
        return google_client()
    return oai_async()


def _api_model(model):
    """Return the API model name. Groq-hosted models keep their prefix."""
    if any(model.startswith(p) for p in _GROQ_PREFIXES):
        return model  # Groq uses full prefixed model IDs
    if any(model.startswith(p) for p in _INCEPTION_PREFIXES):
        return model.split("/", 1)[1]
    return model.split("/", 1)[1] if "/" in model else model


def _tok_key(model):
    """GPT-5.x / 5-mini / o-series use max_completion_tokens instead of max_tokens."""
    return "max_completion_tokens" if any(x in model for x in ["5.2", "5.3", "5.4", "5-mini", "o3", "o4"]) else "max_tokens"


def _supports_temperature(model):
    """Some models (o-series, gpt-5-mini) don't support temperature parameter."""
    return not any(x in model for x in ["5-mini", "o1", "o3", "o4"])


# ── Cost tracking ──

# Per-million-token pricing (USD) as of 2026-Q1
_PRICING = {
    # OpenAI
    "gpt-4.1-mini":               {"input": 0.40,  "output": 1.60},
    "gpt-4.1":                    {"input": 2.00,  "output": 8.00},
    "gpt-4.1-nano":               {"input": 0.10,  "output": 0.40},
    "gpt-4o":                     {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":                {"input": 0.15,  "output": 0.60},
    "text-embedding-3-large":     {"input": 0.13,  "output": 0.0},
    "text-embedding-3-small":     {"input": 0.02,  "output": 0.0},
    # Groq (OpenAI OSS)
    "openai/gpt-oss-120b":        {"input": 0.15,  "output": 0.60},
    "openai/gpt-oss-20b":         {"input": 0.075, "output": 0.30},
    # Anthropic
    "anthropic/claude-opus-4-6":   {"input": 15.0,  "output": 75.0},
    "anthropic/claude-sonnet-4-6": {"input": 3.0,   "output": 15.0},
    "anthropic/claude-haiku-4-5":  {"input": 0.80,  "output": 4.0},
    # Google
    "google/gemini-2.5-pro":      {"input": 1.25,  "output": 10.0},
    "google/gemini-2.0-flash":    {"input": 0.10,  "output": 0.40},
}

_cost_accum = {"input_tokens": 0, "output_tokens": 0, "embed_tokens": 0,
               "cost_usd": 0.0, "calls": 0}


def _track_usage(model, usage):
    """Accumulate token usage and compute cost."""
    if not usage:
        return
    inp = getattr(usage, "prompt_tokens", 0) or 0
    out = getattr(usage, "completion_tokens", 0) or 0
    total_tok = getattr(usage, "total_tokens", 0) or 0
    _cost_accum["input_tokens"] += inp
    _cost_accum["output_tokens"] += out
    _cost_accum["calls"] += 1
    p = _PRICING.get(model, {"input": 2.0, "output": 8.0})
    _cost_accum["cost_usd"] += inp / 1e6 * p["input"] + out / 1e6 * p["output"]


def _track_embed_usage(model, n_tokens):
    """Track embedding token usage."""
    _cost_accum["embed_tokens"] += n_tokens
    _cost_accum["calls"] += 1
    p = _PRICING.get(model, {"input": 0.13, "output": 0.0})
    _cost_accum["cost_usd"] += n_tokens / 1e6 * p["input"]


def get_cost_summary():
    """Return cost summary dict."""
    return dict(_cost_accum)


def reset_cost_tracking():
    """Reset cost accumulators."""
    _cost_accum.update({"input_tokens": 0, "output_tokens": 0, "embed_tokens": 0,
                        "cost_usd": 0.0, "calls": 0})


# ── LLM calls ──

import logging as _logging

_log = _logging.getLogger(__name__)


def _rate_limit_delay(exc, attempt: int) -> float:
    """Extract delay from a 429 response, falling back to exponential backoff."""
    # OpenAI/Groq SDKs attach the response on the exception
    retry_after = None
    resp = getattr(exc, "response", None)
    if resp is not None:
        retry_after = resp.headers.get("retry-after")
    if retry_after is not None:
        try:
            return min(float(retry_after), 120.0)
        except (ValueError, TypeError):
            pass
    # Exponential backoff: 2, 4, 8, 16, 32 — capped at 60s
    return min(2 ** (attempt + 1), 60)

async def call_oai(model, prompt, max_tokens=300, json_mode=False,
                   temperature=0, semaphore=None):
    """Async OpenAI chat completion with retry.

    Semaphore is held ONLY during the API call, released during backoff sleep.
    """
    api_model = _api_model(model)
    kw = {"model": api_model,
          "messages": [{"role": "user", "content": prompt}],
          _tok_key(model): max_tokens,
          "seed": 42}
    if _supports_temperature(model):
        kw["temperature"] = temperature
    if json_mode:
        kw["response_format"] = {"type": "json_object"}
    client = _get_client(model)
    # Scale timeout with max_tokens: large inference calls need more time
    _timeout = max(90, min(300, max_tokens // 5))

    for attempt in range(5):
        try:
            if semaphore:
                async with semaphore:
                    r = await asyncio.wait_for(
                        client.chat.completions.create(**kw),
                        timeout=_timeout,
                    )
            else:
                r = await asyncio.wait_for(
                    client.chat.completions.create(**kw),
                    timeout=_timeout,
                )
            _track_usage(model, r.usage)
            return r.choices[0].message.content.strip()
        except asyncio.TimeoutError:
            if attempt < 4:
                await asyncio.sleep(2)
                continue
            raise RuntimeError(f"OAI call timed out after {attempt+1} attempts")
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                delay = _rate_limit_delay(e, attempt)
                _log.info("Rate limited (attempt %d/%d), waiting %.1fs", attempt + 1, 5, delay)
                await asyncio.sleep(delay)
            else:
                raise
    raise RuntimeError("OAI call failed after 5 retries")


async def _call_model(model, messages, max_tokens=300, temperature=0.0,
                      json_mode=False):
    """Unified model call — routes to correct SDK by model prefix.

    messages: OpenAI-style [{"role": "user", "content": "..."}]
    Returns: response text string.
    """
    if model.startswith("anthropic/"):
        model_id = model.split("/", 1)[1]
        client = anthropic_async()
        # Separate system message if present
        system_msg = None
        chat_msgs = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                chat_msgs.append(m)
        kw = {"model": model_id, "max_tokens": max_tokens, "messages": chat_msgs}
        if system_msg:
            kw["system"] = system_msg
        resp = await client.messages.create(**kw)
        return resp.content[0].text

    if model.startswith("google/"):
        model_id = model.split("/", 1)[1]
        gc = google_client()
        prompt = "\n\n".join(
            f"[{m['role'].upper()}] {m['content']}" for m in messages
        )
        resp = await gc.GenerativeModel(model_id).generate_content_async(prompt)
        return resp.text

    # OpenAI / Groq (OpenAI-compatible)
    client = _get_client(model)
    api_model = _api_model(model)
    kw = {"model": api_model, _tok_key(model): max_tokens, "messages": messages,
          "seed": 42}
    if _supports_temperature(model):
        kw["temperature"] = temperature
    if json_mode:
        kw["response_format"] = {"type": "json_object"}
    for attempt in range(5):
        try:
            r = await asyncio.wait_for(
                client.chat.completions.create(**kw),
                timeout=120,
            )
            _track_usage(model, r.usage)
            return r.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                delay = _rate_limit_delay(e, attempt)
                _log.info("Rate limited in _call_model (attempt %d/%d), waiting %.1fs", attempt + 1, 5, delay)
                await asyncio.sleep(delay)
            else:
                raise
    raise RuntimeError("_call_model failed after 5 retries")


async def call_extract(model, system, user_msg, max_tokens=8192, semaphore=None):
    """Extraction call — JSON extraction from LLM. Direct OpenAI/Groq."""
    client = _get_client(model)
    api_model = _api_model(model)
    ekw = {"model": api_model,
           "messages": [{"role": "system", "content": system},
                        {"role": "user", "content": user_msg}],
           _tok_key(model): max_tokens,
           "response_format": {"type": "json_object"},
           "seed": 42}
    if _supports_temperature(model):
        ekw["temperature"] = 0

    async def _do():
        for attempt in range(5):
            try:
                r = await asyncio.wait_for(
                    client.chat.completions.create(**ekw), timeout=90)
                _track_usage(model, r.usage)
                text = r.choices[0].message.content.strip()
                # Strip <think>...</think> blocks (Qwen reasoning)
                text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
                if not text:
                    return {}
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    try:
                        return parse_json_response(text)
                    except Exception:
                        # Try to extract JSON from markdown code blocks
                        m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
                        if m:
                            try:
                                return json.loads(m.group(1))
                            except Exception:
                                pass
                        return {}
            except asyncio.TimeoutError:
                if attempt < 4:
                    await asyncio.sleep(2)
                    continue
                return {}
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    delay = _rate_limit_delay(e, attempt)
                    _log.info("Rate limited in extract (attempt %d/%d), waiting %.1fs", attempt + 1, 5, delay)
                    await asyncio.sleep(delay)
                else:
                    return {}
        return {}

    if semaphore:
        async with semaphore:
            return await _do()
    return await _do()


def parse_json_response(text):
    """Extract JSON from LLM response (handles ```json blocks)."""
    if "```json" in text:
        s = text.index("```json") + 7
        e = text.find("```", s)
        text = text[s:e].strip() if e != -1 else text[s:].strip()
    elif "```" in text:
        s = text.index("```") + 3
        e = text.find("```", s)
        text = text[s:e].strip() if e != -1 else text[s:].strip()
    f, l = text.find("{"), text.rfind("}")
    if f >= 0 and l > f:
        text = text[f:l + 1]
    return json.loads(text)


# ── Embeddings (async-first API) ──

def _resolve_embed_config(model=None, provider=None):
    """Resolve embedding model and provider from args, config, or defaults."""
    try:
        cfg = get_config()
    except Exception:
        cfg = {}
    _provider = provider or cfg.get("embed_provider", "openai")
    _model = model or cfg.get("embed_model", "text-embedding-3-large")
    return _model, _provider


def _embed_local(texts, model="BAAI/bge-large-en-v1.5"):
    """Local embedding via sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError(
            "sentence-transformers not installed. Install: pip install sentence-transformers\n"
            "Or switch to OpenAI embeddings: gosh-memory setup"
        )
    # Cache model instance
    if not hasattr(_embed_local, '_st') or _embed_local._model_name != model:
        _embed_local._st = SentenceTransformer(model)
        _embed_local._model_name = model
    return np.array(_embed_local._st.encode(texts, convert_to_numpy=True))


async def embed_batch(texts, model=None, provider=None):
    """Async batch embed. Primary API.

    provider="openai" (default) -> OpenAI API
    provider="local"            -> sentence-transformers (no API key)
    """
    texts = [t if t.strip() else "[empty]" for t in texts]
    _model, _provider = _resolve_embed_config(model, provider)

    if _provider == "local":
        # sentence-transformers is sync, run in executor
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _embed_local, texts, _model)
    else:
        # OpenAI async (with timeout to prevent hanging)
        import asyncio as _aio
        for _attempt in range(3):
            try:
                resp = await _aio.wait_for(
                    oai_async().embeddings.create(input=texts, model=_model),
                    timeout=120,
                )
                _track_embed_usage(_model, getattr(resp.usage, "total_tokens", 0) or 0)
                return np.array([e.embedding for e in resp.data])
            except _aio.TimeoutError:
                if _attempt < 2:
                    await _aio.sleep(2 ** _attempt)
                else:
                    raise


async def embed_texts(
    texts,
    model=None,
    provider=None,
    dim=3072,
    batch_size=100,
    label="items",
    concurrency=5,
):
    """Async embed all texts in batches."""
    _model, _provider = _resolve_embed_config(model, provider)
    if _provider == "local":
        dim = 384  # default local model dimension
    if not texts:
        return np.zeros((0, dim))
    import asyncio

    batches = [
        (i, texts[i:i + batch_size])
        for i in range(0, len(texts), batch_size)
    ]
    sem = asyncio.Semaphore(max(1, concurrency))
    completed = 0
    results: dict[int, np.ndarray] = {}

    async def _one(start: int, batch: list[str]) -> tuple[int, np.ndarray]:
        async with sem:
            emb = await embed_batch(batch, _model, _provider)
            return start, emb

    tasks = [asyncio.create_task(_one(start, batch)) for start, batch in batches]
    for task in asyncio.as_completed(tasks):
        start, emb = await task
        results[start] = emb
        completed += len(emb)
        if completed % 500 == 0 or completed == len(texts):
            print(f"  Embedded {completed}/{len(texts)} {label}", flush=True)

    ordered = [results[start] for start, _batch in batches]
    return np.vstack(ordered)


async def embed_query(text, model=None, provider=None):
    """Async embed single query."""
    return (await embed_batch([text], model, provider))[0]


# Aliases for callers that already used the _async suffix
embed_batch_async = embed_batch
embed_texts_async = embed_texts
embed_query_async = embed_query


# ── Sync wrappers (for tests, non-async scripts, backward compat) ──

def embed_batch_sync(texts, model=None, provider=None):
    """Sync wrapper for embed_batch. For tests and non-async code."""
    texts = [t if t.strip() else "[empty]" for t in texts]
    _model, _provider = _resolve_embed_config(model, provider)

    if _provider == "local":
        return _embed_local(texts, _model)
    else:
        resp = oai_sync().embeddings.create(input=texts, model=_model)
        _track_embed_usage(_model, getattr(resp.usage, "total_tokens", 0) or 0)
        return np.array([e.embedding for e in resp.data])


def embed_texts_sync(texts, model=None, provider=None, dim=3072, batch_size=100, label="items"):
    """Sync wrapper for embed_texts."""
    _model, _provider = _resolve_embed_config(model, provider)
    if _provider == "local":
        dim = 384
    if not texts:
        return np.zeros((0, dim))
    all_embs = []
    for i in range(0, len(texts), batch_size):
        emb = embed_batch_sync(texts[i:i + batch_size], _model, _provider)
        all_embs.append(emb)
        done = min(i + batch_size, len(texts))
        if done % 500 == 0 or done == len(texts):
            print(f"  Embedded {done}/{len(texts)} {label}", flush=True)
    return np.vstack(all_embs)


def embed_query_sync(text, model=None, provider=None):
    """Sync wrapper for embed_query."""
    _model, _provider = _resolve_embed_config(model, provider)

    if _provider == "local":
        return _embed_local([text], _model)[0]
    else:
        resp = oai_sync().embeddings.create(input=[text], model=_model)
        return np.array(resp.data[0].embedding)


# ── Text normalization & metrics ──

def normalize_answer(s):
    s = str(s).replace(',', "")
    s = re.sub(r'\b(a|an|the|and)\b', ' ', s.lower())
    s = ''.join(ch for ch in s if ch not in set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))
    return ' '.join(s.split())


_LEMMA_CANON = {
    "favourite": "favorite",
    "gf": "girlfriend",
    "bf": "boyfriend",
}


def _fallback_stem(w: str):
    for sfx, rep in [("ies", "y"), ("sses", "ss"), ("es", ""), ("ing", ""), ("ed", "")]:
        if w.endswith(sfx) and len(w) > len(sfx) + 2:
            return w[:-len(sfx)] + rep
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def stem(w):
    return normalize_term_token(w)


def normalize_term_token(token: str) -> str:
    token = (token or "").strip().lower()
    if not token:
        return ""
    if HAS_SIMPLEMMA and re.fullmatch(r"[a-z]+", token):
        lemma = simplemma.lemmatize(token, lang="en") or token
        return _LEMMA_CANON.get(lemma, lemma)
    return _fallback_stem(token)


def f1_single(pred, gold):
    pt = [stem(w) for w in normalize_answer(pred).split()]
    gt = [stem(w) for w in normalize_answer(gold).split()]
    if not pt or not gt:
        return 0.0
    common = Counter(pt) & Counter(gt)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    p, r = ns / len(pt), ns / len(gt)
    return (2 * p * r) / (p + r)


def f1_multi(pred, gold):
    preds = [p.strip() for p in pred.split(',')]
    golds = [g.strip() for g in gold.split(',')]
    return sum(max(f1_single(p, g) for p in preds) for g in golds) / len(golds) if golds else 0.0


def simple_term_frequency(query_text, searchable_text):
    """Simple TF score between query and document."""
    query_terms = set(re.findall(r'[a-z0-9_-]+', query_text.lower())) - STOP_WORDS
    if not query_terms:
        return 0.0
    doc_words = set(re.findall(r'[a-z0-9_-]+', searchable_text.lower()))
    doc_stems = {stem(w) for w in doc_words}
    hits = 0
    for t in query_terms:
        if t in doc_words or stem(t) in doc_stems:
            hits += 1
    return hits / len(query_terms)
