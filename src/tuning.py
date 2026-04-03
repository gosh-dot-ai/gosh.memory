#!/usr/bin/env python3
"""Centralized runtime tuning loader.

All normal quality iteration should flow through `src/tuning.json`.
This module keeps the file data-only and merges it onto safe defaults.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

TUNING_FILE = Path(__file__).with_name("tuning.json")

DEFAULT_TUNING = {
    "routing": {
        "max_plausible_families": 2,
        "ambiguous_family_fanout": 2,
    },
    "episodes": {
        "document_grouping": {
            "prompt_mode": "strict_small",
            "size_cap_chars": 12000,
            "strip_duplicates": True,
            "attach_missing": False,
            "singleton_missing": True,
        },
    },
    "retrieval": {
        "selector": {
            "max_candidates": 80,
            "max_episodes_default": 3,
            "max_episodes_per_family": 3,
            "max_sources_per_family": 2,
            "late_fusion_per_family": 3,
            "rrf_k": 60,
            "word_overlap_bonus": 0.45,
            "number_overlap_bonus": 1.4,
            "chainage_overlap_bonus": 0.0,
            "identifier_overlap_bonus": 8.0,
            "entity_phrase_bonus": 2.0,
            "step_bonus": 3.5,
            "currentness_bonus": 0.8,
            "generic_penalty": 1.2,
            "mega_penalty": 8.0,
            "supporting_facts_per_episode": 3,
            "supporting_facts_total": 10,
            "budget": 8000,
            "snippet_mode": False,
        },
    },
    "operators": {
        "ordinal_candidate_budget": 12,
        "compare_alignment_budget": 12,
        "list_set_dedup_overlap": 0.9,
        "list_set_max_episodes": 6,
        "list_set_item_max_content_tokens": 4,
        "list_set_compact_item_bonus": 2.5,
        "list_set_enumeration_bonus": 6.0,
        "list_set_supporting_facts_total": 24,
        "list_set_supporting_facts_per_episode": 2,
        "conversation_structural_fact_sweep_top_k": 12,
        "conversation_structural_fact_sweep_bm25_pool": 24,
        "conversation_structural_fact_sweep_vector_pool": 24,
        "conversation_structural_fact_sweep_entity_pool": 12,
        "conversation_structural_fact_sweep_rrf_k": 60,
        "document_structural_seed_fact_count": 1,
        "document_structural_candidate_bundle_top_k": 18,
        "enable_snippet_for_ordinal": True,
        "enable_snippet_for_local_anchor": True,
        "local_anchor_window_chars": 1200,
        "structural_query_supporting_facts_total": 24,
        "structural_query_supporting_facts_per_episode": 4,
        "bounded_chain_max_hops": 2,
        "bounded_chain_root_support_sentence_budget": 1,
        "bounded_chain_support_sentence_budget": 2,
        "bounded_chain_same_source_bonus": 0.5,
        "bounded_chain_unresolved_overlap_weight": 2.0,
        "bounded_chain_novelty_weight": 0.75,
        "bounded_chain_location_bonus_weight": 1.5,
        "bounded_chain_bm25_carry_weight": 0.01,
        "bounded_chain_relation_novelty_weight": 1.0,
        "bounded_chain_relation_repeat_penalty": 1.0,
        "bounded_chain_lookahead_weight": 1.0,
        "bounded_chain_support_sentence_score_weight": 1.0,
        "bounded_chain_sentence_entity_match_weight": 6.0,
        "bounded_chain_sentence_word_match_weight": 1.0,
        "bounded_chain_sentence_number_match_weight": 1.5,
        "bounded_chain_sentence_frontier_overlap_weight": 2.0,
        "bounded_chain_entity_anchor_bonus_weight": 4.0,
        "bounded_chain_support_fact_seed_count": 1,
        "bounded_chain_support_fact_max_extra": 2,
    },
    "packet": {
        "budget": 8000,
        "max_facts": 10,
        "max_facts_per_episode": 3,
        "max_episodes": 3,
        "per_source_cap": 2,
        "per_family_cap": 3,
        "snippet_mode": False,
        "snippet_chars": 1200,
        "support_episode_pool_size": 20,
        "query_specificity_bonus": 4.0,
        "expand_conversation_source_for_structural_queries": True,
        "conversation_structural_support_episode_pool_size": 32,
        "inject_support_fact_episodes": True,
        "max_injected_support_fact_episodes": 8,
    },
    "telemetry": {
        "include_runtime_trace": True,
        "max_selection_scores": 12,
        "max_scope_source_ids": 32,
        "max_family_candidates": 8,
        "max_rejected_candidates": 8,
        "max_packet_fact_ids": 24,
    },
}

_CACHE = {"mtime_ns": None, "data": deepcopy(DEFAULT_TUNING)}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_tuning(path: str | Path | None = None) -> dict:
    tuning_path = Path(path) if path is not None else TUNING_FILE
    if not tuning_path.exists():
        return deepcopy(DEFAULT_TUNING)
    raw = json.loads(tuning_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("tuning.json must contain a top-level object")
    return _deep_merge(DEFAULT_TUNING, raw)


def get_runtime_tuning(path: str | Path | None = None) -> dict:
    tuning_path = Path(path) if path is not None else TUNING_FILE
    try:
        mtime_ns = tuning_path.stat().st_mtime_ns
    except FileNotFoundError:
        mtime_ns = None
    if _CACHE["mtime_ns"] != mtime_ns:
        _CACHE["mtime_ns"] = mtime_ns
        _CACHE["data"] = load_tuning(tuning_path)
    return deepcopy(_CACHE["data"])


def get_tuning_section(*keys: str) -> dict:
    current = get_runtime_tuning()
    for key in keys:
        current = current[key]
    return deepcopy(current)
