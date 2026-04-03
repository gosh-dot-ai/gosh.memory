#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Second-pass coverage recovery for multi-item queries.

This module is intentionally narrow:
- classify only multi-item query shapes
- measure first-pass diversity/coverage from already-selected evidence
- merge second-pass facts only when they improve diversity

It never changes scorer math or first-pass source ranking.
"""

from __future__ import annotations

from collections import Counter

from .common import normalize_term_token
from .episode_features import extract_query_features


def classify_coverage_query(query: str) -> str:
    """Return the multi-item query class relevant for coverage recovery."""
    qf = extract_query_features(query)
    operator_plan = qf.get("operator_plan") or {}
    if operator_plan.get("commonality", {}).get("enabled"):
        return "commonality"
    if operator_plan.get("compare_diff", {}).get("enabled"):
        return "compare"
    if operator_plan.get("list_set", {}).get("enabled"):
        return "list"
    return "none"


def _fact_episode_id(fact: dict) -> str:
    metadata = fact.get("metadata") or {}
    return str(
        metadata.get("episode_id")
        or fact.get("episode_id")
        or ""
    ).strip()


def _fact_source_id(fact: dict) -> str:
    metadata = fact.get("metadata") or {}
    return str(
        fact.get("source_id")
        or metadata.get("episode_source_id")
        or ""
    ).strip()


def _normalized_entities(fact: dict) -> set[str]:
    entities: set[str] = set()
    for raw in fact.get("entities") or []:
        text = raw if isinstance(raw, str) else str(raw)
        norm = normalize_term_token(text)
        if norm:
            entities.add(norm)
    return entities


def _support_span_keys(fact: dict) -> set[tuple[str, str, int, int]]:
    keys: set[tuple[str, str, int, int]] = set()
    for span in fact.get("support_spans") or []:
        if not isinstance(span, dict):
            continue
        start = span.get("start")
        end = span.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        keys.add(
            (
                str(span.get("episode_id") or _fact_episode_id(fact) or "").strip(),
                str(span.get("source_field") or "raw_text").strip(),
                int(start),
                int(end),
            )
        )
    return keys


def compute_coverage_stats(
    query_type: str,
    facts: list[dict],
    *,
    selected_source_count: int | None = None,
) -> dict:
    """Summarize first-pass diversity from already-selected evidence."""
    episode_ids: set[str] = set()
    entities: set[str] = set()
    support_spans: set[tuple[str, str, int, int]] = set()
    source_ids: set[str] = set()
    fact_ids: set[str] = set()

    for fact in facts:
        fact_id = str(fact.get("id") or "").strip()
        if fact_id:
            fact_ids.add(fact_id)
        episode_id = _fact_episode_id(fact)
        if episode_id:
            episode_ids.add(episode_id)
        source_id = _fact_source_id(fact)
        if source_id:
            source_ids.add(source_id)
        entities |= _normalized_entities(fact)
        support_spans |= _support_span_keys(fact)

    source_counter = Counter(
        _fact_source_id(fact)
        for fact in facts
        if _fact_source_id(fact)
    )
    total_sources = len(facts)
    dominant_source_share = 0.0
    if total_sources:
        dominant_source_share = max(source_counter.values()) / float(total_sources)

    return {
        "query_type": query_type,
        "fact_count": len(facts),
        "distinct_episodes": len(episode_ids),
        "distinct_entities": len(entities),
        "distinct_support_spans": len(support_spans),
        "source_diversity": len(source_counter),
        "selected_source_count": int(selected_source_count or 0),
        "dominant_source_share": dominant_source_share,
        "fact_ids": sorted(fact_ids),
        "episode_ids": sorted(episode_ids),
        "source_ids": sorted(source_ids),
        "entity_keys": sorted(entities),
        "support_span_keys": sorted(support_spans),
    }


def needs_coverage_recovery(query_type: str, first_pass_stats: dict) -> bool:
    """Return whether a multi-item query likely needs second-pass expansion."""
    if query_type == "none":
        return False

    fact_count = int(first_pass_stats.get("fact_count", 0))
    distinct_episodes = int(first_pass_stats.get("distinct_episodes", 0))
    distinct_entities = int(first_pass_stats.get("distinct_entities", 0))
    distinct_support_spans = int(first_pass_stats.get("distinct_support_spans", 0))
    source_diversity = int(first_pass_stats.get("source_diversity", 0))
    selected_source_count = int(first_pass_stats.get("selected_source_count", 0))

    if query_type == "commonality":
        if fact_count < 2 or distinct_episodes < 2:
            return True
        if selected_source_count > 1 and source_diversity < 2:
            return True
        return False

    if query_type == "compare":
        return fact_count < 2 or distinct_episodes < 2 or distinct_support_spans < 2

    if query_type == "aggregate":
        return fact_count < 3 or distinct_episodes < 3 or max(distinct_entities, distinct_support_spans) < 2

    if query_type == "list":
        if fact_count < 2:
            return True
        if max(distinct_entities, distinct_support_spans, distinct_episodes) < 2:
            return True
        if selected_source_count > 1 and source_diversity < 2:
            return True
        return False

    return False


def _coverage_diversity_gain(fact: dict, current: dict) -> tuple[int, int, int, int]:
    episode_gain = 0
    entity_gain = 0
    span_gain = 0
    source_gain = 0

    episode_id = _fact_episode_id(fact)
    if episode_id and episode_id not in set(current.get("episode_ids") or []):
        episode_gain = 1

    current_entity_keys = set(current.get("entity_keys") or [])
    new_entity_count = len(_normalized_entities(fact) - current_entity_keys)
    if new_entity_count > 0:
        entity_gain = new_entity_count

    current_span_keys = {tuple(item) for item in (current.get("support_span_keys") or [])}
    new_span_count = len(_support_span_keys(fact) - current_span_keys)
    if new_span_count > 0:
        span_gain = new_span_count

    source_id = _fact_source_id(fact)
    if source_id and source_id not in set(current.get("source_ids") or []):
        source_gain = 1

    return episode_gain, entity_gain, span_gain, source_gain


def merge_coverage_recovery_facts(
    query_type: str,
    existing_facts: list[dict],
    recovered_facts: list[dict],
    *,
    max_facts: int,
) -> list[dict]:
    """Merge second-pass facts only when they increase evidence diversity."""
    merged: list[dict] = []
    seen_ids: set[str] = set()

    for fact in existing_facts:
        fact_id = str(fact.get("id") or "").strip()
        if fact_id and fact_id in seen_ids:
            continue
        if fact_id:
            seen_ids.add(fact_id)
        merged.append(fact)
        if len(merged) >= max_facts:
            return merged

    if query_type == "none":
        return merged[:max_facts]

    current = compute_coverage_stats(query_type, merged)

    for fact in recovered_facts:
        fact_id = str(fact.get("id") or "").strip()
        if fact_id and fact_id in seen_ids:
            continue
        gain = _coverage_diversity_gain(fact, current)
        if gain <= (0, 0, 0, 0):
            continue
        if fact_id:
            seen_ids.add(fact_id)
        merged.append(fact)
        current = compute_coverage_stats(query_type, merged)
        if len(merged) >= max_facts:
            break

    return merged[:max_facts]
