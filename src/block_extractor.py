#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Block-level fact extraction.

Copyright 2026 (c) Mitja Goroshevsky and GOSH Technology Ltd.
License: MIT.

Extracts facts from a single Block using family-specific prompts.
Does NOT wire into librarian.py — standalone module for Stage 2.
"""

import json
import logging
import re
from pathlib import Path

from .block_segmenter import Block

_log = logging.getLogger(__name__)
_PROMPT_DIR = Path(__file__).parent / "prompts" / "extraction"

_EMPTY_RESULT = {"facts": [], "temporal_links": []}

# Family → prompt file mapping
_FAMILY_PROMPT = {
    "PROSE": "prose_block.md",
    "LIST": "list_block.md",
    "TABLE": "table_block.md",
    "UNKNOWN": "fallback_block.md",
    "KV": "prose_block.md",
    "CODE": "fallback_block.md",
    "OBJECT": "fallback_block.md",
}


def _load_prompt(name: str) -> str:
    path = _PROMPT_DIR / name
    return path.read_text(encoding="utf-8")


def _format_prompt(template: str, block: Block, session_metadata: dict) -> str:
    return template.format(
        container_kind=session_metadata.get("container_kind", "conversation"),
        speaker=block.speaker or "unknown",
        lead_in=block.lead_in or "none",
        session_date=session_metadata.get("session_date", "unknown"),
        session_num=session_metadata.get("session_num", 0),
        section_path=block.section_path or "none",
    )


def _normalize_result(obj) -> dict | None:
    """Normalize parsed LLM output into {"facts": [...], "temporal_links": [...]}.

    Handles:
    - {"facts": [...], "temporal_links": [...]} — standard
    - bare list of fact dicts — legacy LLM output
    - dict with "id" instead of "local_id" — legacy format
    """
    if isinstance(obj, list):
        # Bare list: wrap as facts, normalize id→local_id
        facts = [f for f in obj if isinstance(f, dict)]
        for f in facts:
            if "id" in f and "local_id" not in f:
                f["local_id"] = f["id"]
        return {"facts": facts, "temporal_links": []}
    if isinstance(obj, dict):
        if "facts" in obj:
            # Normalize id→local_id on individual facts
            for f in obj.get("facts", []):
                if isinstance(f, dict) and "id" in f and "local_id" not in f:
                    f["local_id"] = f["id"]
            obj.setdefault("temporal_links", [])
            return obj
    return None


def _parse_result(raw: str) -> dict | None:
    """Parse LLM output as JSON. Returns None on failure."""
    # Strip think blocks
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL)
    raw = raw.strip()
    # Try direct parse
    try:
        obj = json.loads(raw)
        result = _normalize_result(obj)
        if result:
            return result
    except json.JSONDecodeError:
        pass
    # Try extracting JSON from markdown code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            result = _normalize_result(obj)
            if result:
                return result
        except json.JSONDecodeError:
            pass
    # Try finding outermost braces
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            result = _normalize_result(obj)
            if result:
                return result
        except json.JSONDecodeError:
            pass
    return None


async def extract_block(
    block: Block,
    session_metadata: dict,
    model: str | None = None,
    call_extract_fn=None,
    prompt_overrides: dict[str, str] | None = None,
) -> dict:
    """Extract facts from a single block.

    Args:
        block: Block to extract from.
        session_metadata: dict with session_date, session_num, etc.
        model: extraction model name. Passed through to call_extract_fn.
        call_extract_fn: async fn(model, system, user_msg, max_tokens) -> raw.
            If None, uses src.common.call_extract.
        prompt_overrides: optional dict mapping prompt_name (e.g. "prose_block")
            to full prompt text. When present, overrides the built-in prompt file.

    Returns:
        dict with "facts" and "temporal_links" keys.
    """
    if call_extract_fn is None:
        from .common import call_extract
        call_extract_fn = call_extract

    # Select prompt — MAL override takes precedence
    prompt_file = _FAMILY_PROMPT.get(block.family, "fallback_block.md")
    prompt_name = prompt_file.replace(".md", "")
    if prompt_overrides and prompt_name in prompt_overrides:
        template = prompt_overrides[prompt_name]
    else:
        template = _load_prompt(prompt_file)
    system_prompt = _format_prompt(template, block, session_metadata)
    user_msg = block.text

    def _try_normalize(raw):
        """Try to normalize raw LLM output into result dict."""
        if isinstance(raw, (dict, list)):
            result = _normalize_result(raw)
            if result:
                return result
        if isinstance(raw, str):
            return _parse_result(raw)
        return None

    # Attempt 1: normal extraction
    raw = await call_extract_fn(model, system_prompt, user_msg, max_tokens=4096)
    parsed = _try_normalize(raw)
    if parsed is not None:
        return parsed

    # Attempt 2: retry same prompt
    _log.info("Block extraction parse failure, retrying (block order=%d)", block.order)
    raw = await call_extract_fn(model, system_prompt, user_msg, max_tokens=4096)
    parsed = _try_normalize(raw)
    if parsed is not None:
        return parsed

    # Attempt 3: fallback prompt
    _log.info("Block extraction retry failed, trying fallback prompt (block order=%d)", block.order)
    fallback_template = _load_prompt("fallback_block.md")
    fallback_prompt = _format_prompt(fallback_template, block, session_metadata)
    raw = await call_extract_fn(model, fallback_prompt, user_msg, max_tokens=4096)
    parsed = _try_normalize(raw)
    if parsed is not None:
        return parsed

    # All attempts failed
    _log.warning("Block extraction failed after 3 attempts (block order=%d)", block.order)
    return dict(_EMPTY_RESULT)
