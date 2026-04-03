"""Tests for query-type-adaptive inference prompts.

Verifies all 8 prompt types in INF_PROMPTS and the get_inf_prompt() fallback.
"""

from src.inference import (
    INF_PROMPTS,
    get_inf_prompt,
    resolve_inference_prompt_key,
)


# ── get_inf_prompt basic contract ──

def test_get_inf_prompt_lookup():
    prompt = get_inf_prompt("lookup")
    assert "{context}" in prompt
    assert "{question}" in prompt


def test_get_inf_prompt_temporal():
    prompt = get_inf_prompt("temporal")
    assert "{context}" in prompt
    assert "{question}" in prompt
    assert "chronological" in prompt.lower()


def test_get_inf_prompt_aggregate():
    prompt = get_inf_prompt("aggregate")
    assert "COUNTING PROTOCOL" in prompt


def test_get_inf_prompt_current():
    prompt = get_inf_prompt("current")
    assert "MOST RECENT" in prompt or "UPDATED" in prompt


def test_get_inf_prompt_synthesize():
    prompt = get_inf_prompt("synthesize")
    assert "preference" in prompt.lower() or "pattern" in prompt.lower()


def test_get_inf_prompt_procedural():
    prompt = get_inf_prompt("procedural")
    assert "rules" in prompt.lower() or "policies" in prompt.lower()


def test_get_inf_prompt_prospective():
    prompt = get_inf_prompt("prospective")
    assert "planned" in prompt.lower() or "upcoming" in prompt.lower()


def test_get_inf_prompt_summarize():
    prompt = get_inf_prompt("summarize")
    assert "chronological" in prompt.lower()


def test_get_inf_prompt_unknown_returns_lookup():
    """Unknown query types fall back to the lookup prompt."""
    assert get_inf_prompt("nonexistent") == get_inf_prompt("lookup")


def test_get_inf_prompt_empty_returns_lookup():
    assert get_inf_prompt("") == get_inf_prompt("lookup")


# ── All prompts have required placeholders ──

def test_all_prompts_have_context_and_question():
    for qtype, prompt in INF_PROMPTS.items():
        assert "{context}" in prompt, f"{qtype} prompt missing {{context}}"
        assert "{question}" in prompt, f"{qtype} prompt missing {{question}}"


# ── Prompts are formattable ──

def test_lookup_prompt_formats():
    prompt = get_inf_prompt("lookup").format(context="some facts", question="what?")
    assert "some facts" in prompt
    assert "what?" in prompt


def test_summarize_prompt_formats():
    prompt = get_inf_prompt("summarize").format(context="facts here", question="summarize all")
    assert "facts here" in prompt
    assert "summarize all" in prompt


# ── Dict completeness ──

def test_inf_prompts_has_all_types():
    expected = {"lookup", "temporal", "aggregate", "current",
                "synthesize", "procedural", "prospective", "summarize", "icl",
                "hybrid", "tool", "summarize_with_metadata", "list_set",
                "slot_query", "compositional"}
    assert set(INF_PROMPTS.keys()) == expected


def test_resolve_inference_prompt_key_uses_slot_query_leaf():
    operator_plan = {
        "slot_query": {"enabled": True},
        "list_set": {"enabled": False},
        "ordinal": {"enabled": False},
        "commonality": {"enabled": False},
        "compare_diff": {"enabled": False},
    }
    assert resolve_inference_prompt_key("default", operator_plan) == "slot_query"


def test_resolve_inference_prompt_key_does_not_use_slot_query_leaf_for_bounded_chain_queries():
    operator_plan = {
        "slot_query": {"enabled": True},
        "bounded_chain": {"enabled": True},
        "ordinal": {"enabled": False},
        "commonality": {"enabled": False},
        "compare_diff": {"enabled": False},
        "list_set": {"enabled": False},
        "local_anchor": {"enabled": False},
        "temporal_grounding": {"enabled": False},
    }
    assert resolve_inference_prompt_key("hybrid", operator_plan) == "hybrid"


def test_resolve_inference_prompt_key_uses_list_set_leaf():
    operator_plan = {
        "list_set": {"enabled": True},
        "ordinal": {"enabled": False},
        "commonality": {"enabled": False},
        "compare_diff": {"enabled": False},
        "bounded_chain": {"enabled": False},
    }
    assert resolve_inference_prompt_key("hybrid", operator_plan) == "list_set"


def test_resolve_inference_prompt_key_maps_default_to_lookup_before_leafs():
    operator_plan = {
        "list_set": {"enabled": True},
        "ordinal": {"enabled": False},
        "commonality": {"enabled": False},
        "compare_diff": {"enabled": False},
        "bounded_chain": {"enabled": False},
    }
    assert resolve_inference_prompt_key("default", operator_plan) == "list_set"


def test_resolve_inference_prompt_key_respects_disabled_leaf_plugin():
    operator_plan = {
        "list_set": {"enabled": True},
        "ordinal": {"enabled": False},
        "commonality": {"enabled": False},
        "compare_diff": {"enabled": False},
        "bounded_chain": {"enabled": False},
    }
    assert (
        resolve_inference_prompt_key(
            "hybrid",
            operator_plan,
            plugin_state={"list_set": False},
        )
        == "hybrid"
    )


def test_get_inf_prompt_slot_query_is_short_leaf():
    prompt = get_inf_prompt("slot_query")
    assert "slot-filling or attribute question" in prompt
    assert "RAW SLOT CANDIDATES" in prompt


def test_get_inf_prompt_list_set_is_short_leaf():
    prompt = get_inf_prompt("list_set")
    assert "grounded list or set of items" in prompt
    assert "Return all distinct grounded items" in prompt
