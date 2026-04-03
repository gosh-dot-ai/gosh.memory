"""Regression tests for prompt routing in extract_session()."""

import asyncio
from pathlib import Path

from src.librarian import extract_session


class _SpyExtract:
    def __init__(self):
        self.calls = []

    async def __call__(self, model, system, user_msg, max_tokens=8192):
        self.calls.append({
            "model": model,
            "system": system,
            "user_msg": user_msg,
            "max_tokens": max_tokens,
        })
        return {"facts": [], "temporal_links": []}


def _run_extract(session_text, *, fmt=None):
    spy = _SpyExtract()
    asyncio.run(extract_session(
        session_text=session_text,
        session_num=1,
        session_date="2024-06-01",
        conv_id="conv-1",
        speakers="User and Assistant",
        model="test-model",
        call_extract_fn=spy,
        fmt=fmt,
    ))
    assert len(spy.calls) >= 1
    return spy.calls


def test_conversation_route_uses_conversation_prompt():
    calls = _run_extract("user: hello\nassistant: hi")
    assert len(calls) == 2
    assert all(call["system"].startswith(
        "You are extracting structured atomic facts from a prose block.")
        for call in calls)
    assert all("Container: conversation" in call["system"] for call in calls)


def test_json_conv_route_uses_conversation_prompt_after_preprocessing():
    calls = _run_extract('[{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]')
    assert len(calls) == 2
    assert all(call["system"].startswith(
        "You are extracting structured atomic facts from a prose block.")
        for call in calls)
    assert "hello" in calls[0]["user_msg"].lower()


def test_document_route_uses_document_prompt():
    calls = _run_extract("# Overview\n\nThis document describes the pipeline.")
    assert len(calls) == 1
    assert calls[0]["system"].startswith(
        "You are extracting structured atomic facts from a prose block.")
    assert "Container: document" in calls[0]["system"]


def test_agent_trace_route_uses_agent_trace_prompt():
    call = _run_extract("[Step 1]\nAction: click\nObservation: done")[0]
    assert call["system"].startswith(
        "You are extracting memory-relevant facts from an agent execution trace.")


def test_fact_list_route_uses_fact_list_prompt():
    call = _run_extract(
        "1. The user prefers tea.\n2. The user lives in Berlin.\n"
        "3. The user owns a dog.\n4. The user bikes to work."
    )[0]
    assert call["system"].startswith(
        "You are extracting structured atomic facts from a fact list.")


def test_narrative_route_uses_narrative_prompt():
    call = _run_extract(
        "He stood on the quay and watched the ferry leave.\n\n"
        "\"We are too late,\" she said. After a minute, they turned back."
    )[0]
    assert call["system"].startswith(
        "You are extracting structured atomic facts from narrative prose.")


def test_unknown_route_uses_legacy_prompt():
    call = _run_extract("plain text", fmt="UNKNOWN")[0]
    assert call["system"].startswith(
        "You are extracting structured atomic facts from a conversation between friends.")


def test_conversation_prompt_drops_narrative_rules_and_uses_quality_budget():
    text = Path("src/prompts/extraction/conversation.md").read_text(encoding="utf-8")
    assert "RULE 7e — SEQUENCE EVENTS." not in text
    assert "RULE 7f — CHARACTER NAMES." not in text
    assert "RULE 10 — PRIORITIZE QUALITY OVER QUANTITY." in text
    assert "Extract ALL facts. Be thorough. Each detail = separate fact." not in text


def test_conversation_prompt_preserves_acquisition_events_with_time_anchors():
    text = Path("src/prompts/extraction/conversation.md").read_text(encoding="utf-8")
    assert "DELTA D — ACQUISITION EVENTS. CRITICAL." in text
    assert "bought/purchased/ordered/booked/got/acquired" in text
    assert "prefer the acquisition fact" in text


def test_narrative_prompt_contains_moved_rules():
    text = Path("src/prompts/extraction/narrative.md").read_text(encoding="utf-8")
    assert text.startswith("You are extracting structured atomic facts from narrative prose.")
    assert "RULE A — SEQUENCE EVENTS." in text
    assert "RULE B — CHARACTER NAMES." in text


def test_conversation_prompt_preserves_exact_named_targets_and_acquisition_events():
    text = Path("src/prompts/extraction/conversation.md").read_text(encoding="utf-8")
    assert "RULE 1b — EXACT NAMED TARGETS. CRITICAL." in text
    assert "Do NOT replace a named target with a generic activity or category." in text
    assert "If the text names what was acquired, keep that exact named item in the fact." in text


def test_librarian_prompts_preserve_distinct_targets_across_layers():
    from src.librarian import CONSOLIDATION_PROMPT, CROSS_SESSION_PROMPT

    assert "Preserve distinct named targets, places, events, and items." in CONSOLIDATION_PROMPT
    assert "Preserve distinct targets, places, events, and items as separate supported facts." in CROSS_SESSION_PROMPT
    assert "Do NOT replace them with generic activity labels." in CROSS_SESSION_PROMPT
