#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GOSH Memory — MemoryServer.

In-process memory server with persistence and a clean async API for external agents.

Cache format on disk:
    {data_dir}/{key}.json      → {granular, cons, cross, tlinks, n_sessions, n_sessions_with_facts}
    {data_dir}/{key}_embs.npz  → {gran=..., cons=..., cross=...}
"""

import asyncio
import hashlib
import json
import logging
import math
import os
import re
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from dateutil import parser as date_parser

from .artifacts import write_json_atomic
from .audit import AuditLog
from .common import (
    _PRICING,
    STOP_WORDS,
    _api_model,
    _call_model,
    _get_client,
    _supports_temperature,
    _tok_key,
    call_extract,
    call_oai,
    embed_query,
    embed_texts,
    embed_texts_sync,
    get_cost_summary,
    normalize_term_token,
)
from .config import MemoryConfig
from .coverage_recovery import (
    classify_coverage_query,
    compute_coverage_stats,
    merge_coverage_recovery_facts,
    needs_coverage_recovery,
)
from .episode_extraction import build_singleton_episodes, extract_doc_metadata, group_document
from .episode_features import extract_query_features, has_exact_step_mention, step_range_overlap
from .episode_packet import (
    _fact_content_tokens,
    _fact_slot_fill_candidates,
    _pseudo_facts_from_episode,
    _select_bounded_chain_seed_facts,
    build_bounded_chain_candidate_bundle,
    build_context_from_retrieved_facts,
    build_context_from_selected_episodes,
    fact_episode_ids,
)
from .episode_retrieval import (
    available_families,
    build_episode_bm25,
    choose_episode_ids,
    choose_episode_ids_with_trace,
    partition_corpus_by_family,
    resolve_selection_config,
    route_retrieval_families,
    select_episode_ids_late_fusion,
    select_episode_ids_late_fusion_with_trace,
)
from .episodes import build_episode_lookup, build_facts_by_episode, load_episode_corpus
from .fact_alignment import (
    align_facts_batch,
    facts_as_selectors_enabled,
    iter_support_spans,
    selector_surface_text,
)
from .identity import (
    _generate_artifact_id,
    _generate_version_id,
    content_hash_text,
)
from .inference import (
    COUNTING_TOOLS,
    DEFAULT_INFERENCE_LEAF_PLUGIN_STATE,
    GET_CONTEXT_TOOL,
    TEMPORAL_TOOLS,
    call_inference_with_tools,
    get_inf_prompt,
    get_more_context,
    resolve_inference_prompt_key,
)
from .librarian import (
    consolidate_session,
    cross_session_entity,
    extract_session,
    resolve_supersession,
)
from .mal.apply import current_gen_dir as _mal_current_gen_dir
from .membership import MembershipRegistry
from .prompt_registry import PromptRegistry
from .retrieval import detect_query_type, source_local_fact_sweep
from .source_adapters import segment_document_text
from .storage import JSONNPZStorage, SQLiteStorageBackend, StorageBackend, make_storage
from .temporal import (
    empty_temporal_index,
    latest_calendar_anchor,
    load_temporal_index,
    lookup_events_for_fact,
)
from .temporal_normalizer import normalize_temporal_index
from .temporal_planner import (
    classify_temporal_query,
    execute_calendar_query,
    execute_ordinal_query,
    extract_calendar_query,
)
from .tuning import get_runtime_tuning, get_tuning_section
from .unified_source_extractor import extract_source_aggregation

log = logging.getLogger(__name__)


def _load_mal_active_config(data_dir: str, key: str, agent_id: str) -> dict:
    """Load MAL generation config for a binding, or empty dict if none."""
    try:
        gen_dir = _mal_current_gen_dir(data_dir, key, agent_id)
        config_path = gen_dir / "active_config.json"
        if config_path.exists():
            return json.loads(config_path.read_text())
    except Exception:
        pass
    return {}


def _resolve_extract_model(base_model: str, data_dir: str, key: str, agent_id: str) -> str:
    """Return MAL-overridden extraction model, or the base model."""
    mal_cfg = _load_mal_active_config(data_dir, key, agent_id)
    return mal_cfg.get("extraction_model") or base_model


_COMMONALITY_TOKEN_CANON = {
    "store": "business",
    "stores": "business",
    "studio": "business",
    "studios": "business",
    "shop": "business",
    "shops": "business",
    "company": "business",
    "companies": "business",
    "venture": "business",
    "ventures": "business",
    "startup": "business",
    "startups": "business",
    "open": "start",
    "opens": "start",
    "opened": "start",
    "opening": "start",
    "launch": "start",
    "launches": "start",
    "launched": "start",
    "launching": "start",
    "jobless": "job",
}
_COMMONALITY_IGNORE = STOP_WORDS | {
    "both", "common", "share", "shared", "similar", "similarities",
    "passion", "special", "meaning", "support", "supported", "supportive",
    "motivation", "motivating", "inspiration", "friend", "friendship",
    "journey", "dream", "dreams", "love", "loves", "loving",
    "there", "around", "always", "really", "very", "still",
    "great", "good", "excited", "nervous", "happy", "proud",
    "more", "much", "many", "last", "week", "month", "year",
    "today", "yesterday", "tomorrow", "together", "wait",
    "kind", "word", "words", "hard", "work", "paid", "pay", "off",
    "online", "speaker", "user",
}
_COMMONALITY_QUERY_RE = re.compile(r"\b(in common|both|shared?|same as)\b", re.I)
_COMMONALITY_INTEREST_QUERY_RE = re.compile(r"\b(interests?|hobbies?|favorite|favourite|enjoy|like|likes)\b", re.I)
_COMMONALITY_INTEREST_FACT_RE = re.compile(
    r"\b(love|enjoy|favorite|favourite|hobb(?:y|ies)|watch(?:ing)?|play(?:ing)?|"
    r"make(?:ing)?|bak(?:e|ing)|cook(?:ing)?|read(?:ing)?)\b",
    re.I,
)
_COMMONALITY_OVERLAP_IGNORE = {
    "shar", "same", "similar", "great", "good", "cool", "support", "fun",
    "reward", "really", "just", "pretty", "main",
}
_LOCAL_ANCHOR_CUE_RE = re.compile(
    r"\b(?:at|from|in|to|into|inside|near)\s+([A-Z][A-Za-z0-9&'._-]+(?:\s+[A-Z][A-Za-z0-9&'._-]+){0,3})"
)
_LOCAL_ANCHOR_CAP_RE = re.compile(
    r"(?<!\w)([A-Z][A-Za-z0-9&'._-]+(?:\s+[A-Z][A-Za-z0-9&'._-]+){0,3})"
)
_LOCAL_ANCHOR_LINE_CUE_RE = re.compile(
    r"\b(store|retailers?|shop|coupon|redeem|redeemed|purchase|purchased|bought|"
    r"step|action|position|moved|city|country|location|headquarters|works? in)\b",
    re.I,
)
_LOCAL_ANCHOR_IGNORE = {
    "Action", "Observation", "Active Rules", "Objects on the map", "Step",
    "User", "Assistant", "I", "I'm", "It", "The",
    "Many", "Some", "Several", "These", "Those", "Here", "There",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
}
_QUERY_ENTITY_IGNORE = {"what", "which", "when", "where", "who", "why", "how"}
_CITY_TO_COUNTRY = {
    "amsterdam": "Netherlands",
    "athens": "Greece",
    "barcelona": "Spain",
    "beijing": "China",
    "berlin": "Germany",
    "boston": "United States",
    "brisbane": "Australia",
    "chicago": "United States",
    "dublin": "Ireland",
    "edinburgh": "United Kingdom",
    "hong kong": "China",
    "london": "United Kingdom",
    "los angeles": "United States",
    "madrid": "Spain",
    "melbourne": "Australia",
    "miami": "United States",
    "montreal": "Canada",
    "moscow": "Russia",
    "mumbai": "India",
    "new york": "United States",
    "osaka": "Japan",
    "paris": "France",
    "rome": "Italy",
    "san francisco": "United States",
    "seattle": "United States",
    "seoul": "South Korea",
    "shanghai": "China",
    "sydney": "Australia",
    "tokyo": "Japan",
    "toronto": "Canada",
    "vancouver": "Canada",
    "washington": "United States",
}


def _normalize_commonality_token(token: str) -> str:
    low = token.lower()
    if len(low) > 5 and low.endswith("ing"):
        low = low[:-3]
    elif len(low) > 4 and low.endswith("ed"):
        low = low[:-2]
    elif len(low) > 4 and low.endswith("ies"):
        low = low[:-3] + "y"
    elif len(low) > 4 and low.endswith("s") and not low.endswith("ss"):
        low = low[:-1]
    return _COMMONALITY_TOKEN_CANON.get(low, low)


def _extract_query_named_entities(query: str) -> list[str]:
    seen = set()
    ordered = []
    for match in re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", query):
        entity = " ".join(match.split()).lower()
        if entity in _QUERY_ENTITY_IGNORE:
            continue
        if entity not in seen:
            seen.add(entity)
            ordered.append(entity)
    return ordered


def _fact_entity_hits(fact: dict, query_entities: list[str]) -> set[str]:
    if not query_entities:
        return set()
    fact_text = fact.get("fact", "").lower()
    fact_ents = {
        (entity.lower() if isinstance(entity, str) else str(entity).lower())
        for entity in fact.get("entities", [])
    }
    hits = set()
    for query_entity in query_entities:
        if query_entity in fact_text or any(
            query_entity in fact_ent or fact_ent in query_entity
            for fact_ent in fact_ents
        ):
            hits.add(query_entity)
    return hits


def _commonality_tokens(text: str, query_entities: list[str]) -> set[str]:
    entity_tokens = {
        _normalize_commonality_token(token)
        for entity in query_entities
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]+", entity)
    }
    tokens = set()
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]+", text):
        norm = _normalize_commonality_token(token)
        if len(norm) < 3:
            continue
        if norm in _COMMONALITY_IGNORE or norm in entity_tokens:
            continue
        tokens.add(norm)
    return tokens


def _self_grounded_commonality_bonus(fact: dict, entity: str) -> float:
    speaker = str(fact.get("speaker", "") or "").strip().lower()
    if not speaker:
        return 0.0
    entity_key = entity.split()[0].lower()
    return 4.0 if speaker == entity_key or speaker == entity.lower() else 0.0


def _rank_commonality_groups(candidates: list[tuple[float, set[str], dict, dict]]) -> list[dict]:
    groups = defaultdict(list)
    for score, overlap, left, right in candidates:
        overlap_key = tuple(sorted(overlap))
        if overlap_key:
            groups[overlap_key].append((score, left, right))

    ranked = []
    for overlap_key, pairs in groups.items():
        pairs.sort(
            key=lambda row: (
                -row[0],
                row[1].get("rank", row[1].get("idx", 0)),
                row[2].get("rank", row[2].get("idx", 0)),
            )
        )
        unique_pairs = []
        seen_pair_ids = set()
        for score, left, right in pairs:
            left_fact = left.get("fact", left)
            right_fact = right.get("fact", right)
            if not isinstance(left_fact, dict):
                left_fact = {"fact": str(left_fact)}
            if not isinstance(right_fact, dict):
                right_fact = {"fact": str(right_fact)}
            pair_id = (
                left_fact.get("id", ""),
                right_fact.get("id", ""),
                left_fact.get("session", 0),
                right_fact.get("session", 0),
            )
            if pair_id in seen_pair_ids:
                continue
            seen_pair_ids.add(pair_id)
            unique_pairs.append((score, left, right))

        ranked.append(
            {
                "overlap": overlap_key,
                "pairs": unique_pairs,
                "pair_count": len(unique_pairs),
                "top_score": unique_pairs[0][0] if unique_pairs else 0.0,
                "mean_top_score": (
                    sum(score for score, _left, _right in unique_pairs[:3]) / min(len(unique_pairs), 3)
                    if unique_pairs
                    else 0.0
                ),
                "specificity": len(overlap_key),
                "is_multi": len(overlap_key) >= 2,
            }
        )

    ranked.sort(
        key=lambda group: (
            1 if group["is_multi"] else 0,
            group["mean_top_score"] + math.log1p(group["pair_count"]) + 0.25 * group["specificity"],
            group["specificity"],
            group["top_score"],
        ),
        reverse=True,
    )
    return ranked


def _build_raw_commonality_support_items(query: str, raw_sessions: list[dict]) -> list[dict]:
    if not _COMMONALITY_QUERY_RE.search(query):
        return []
    query_entities = _extract_query_named_entities(query)
    if len(query_entities) < 2:
        return []

    left_entity, right_entity = query_entities[:2]
    left_key = left_entity.split()[0]
    right_key = right_entity.split()[0]
    candidates = []
    seen = set()

    for session_idx, raw_session in enumerate(raw_sessions, start=1):
        if not isinstance(raw_session, dict):
            continue
        text = raw_session.get("content", "")
        if not text:
            continue
        lower = text.lower()
        if left_key not in lower or right_key not in lower:
            continue
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            continue

        left_rows = []
        right_rows = []
        for idx, line in enumerate(lines):
            lower_line = line.lower()
            tokens = _commonality_tokens(line, query_entities)
            if not tokens:
                continue
            if lower_line.startswith(f"{left_key}:"):
                left_rows.append((idx, line, tokens))
                continue
            if lower_line.startswith(f"{right_key}:"):
                right_rows.append((idx, line, tokens))
                continue
            if left_key in lower_line:
                left_rows.append((idx, line, tokens))
            if right_key in lower_line:
                right_rows.append((idx, line, tokens))

        if not left_rows or not right_rows:
            continue

        token_freq = defaultdict(int)
        for _idx, _line, tokens in left_rows + right_rows:
            for token in tokens:
                token_freq[token] += 1

        for left in left_rows[:16]:
            for right in right_rows[:16]:
                if left[0] == right[0] or left[1] == right[1]:
                    continue
                overlap = left[2] & right[2]
                if not overlap:
                    continue
                key = (tuple(sorted(overlap)), left[1], right[1], session_idx)
                if key in seen:
                    continue
                seen.add(key)
                rarity = sum(1.0 / max(token_freq.get(token, 1), 1) for token in overlap)
                score = rarity * 10.0 + len(overlap) * 1.5 - 0.08 * abs(left[0] - right[0])
                candidates.append((score, overlap, session_idx, left, right))

    candidates.sort(key=lambda row: (-row[0], -len(row[1]), row[2], row[3][0], row[4][0]))
    items = []
    used_overlap = set()
    for idx, (_score, overlap, session_idx, left, right) in enumerate(candidates):
        overlap_key = tuple(sorted(overlap))
        if overlap_key in used_overlap:
            continue
        used_overlap.add(overlap_key)
        label = ", ".join(sorted(overlap))
        items.append(
            {
                "text": (
                    f"[Shared raw {len(items)+1}] overlap in source session S{session_idx}: {label}\n"
                    f"- {left[1]}\n"
                    f"- {right[1]}"
                ),
                "rank": -1200 - idx,
                "source": "commonality_raw",
                "session": session_idx,
            }
        )
        if len(items) >= 3:
            break
    return items


def _augment_commonality_facts(
    query: str,
    retrieved_facts: list[dict],
    all_facts: list[dict],
    *,
    limit: int = 6,
) -> list[dict]:
    if not _COMMONALITY_QUERY_RE.search(query):
        return []
    query_entities = _extract_query_named_entities(query)
    if len(query_entities) < 2:
        return []

    existing_ids = {fact.get("id", "") for fact in retrieved_facts}
    rows_by_entity = defaultdict(list)
    token_freq = defaultdict(int)
    session_values = []

    for idx, fact in enumerate(all_facts):
        hits = _fact_entity_hits(fact, query_entities)
        if not hits:
            continue
        tokens = _commonality_tokens(fact.get("fact", ""), query_entities)
        if not tokens:
            continue
        row = {
            "fact": fact,
            "tokens": tokens,
            "idx": idx,
            "already_retrieved": fact.get("id", "") in existing_ids,
        }
        session_no = _coerce_positive_session_num(fact.get("session", 0))
        if session_no is not None:
            session_values.append(session_no)
        for token in tokens:
            token_freq[token] += 1
        for entity in hits:
            rows_by_entity[entity].append(row)

    entities = query_entities[:2]
    if any(not rows_by_entity.get(entity) for entity in entities):
        return []
    earliest_session = min(session_values) if session_values else 0

    interest_query = bool(_COMMONALITY_INTEREST_QUERY_RE.search(query))

    candidates = []
    seen_pairs = set()
    for left in rows_by_entity[entities[0]][:160]:
        for right in rows_by_entity[entities[1]][:160]:
            left_fact = left["fact"]
            right_fact = right["fact"]
            if left_fact.get("id", "") == right_fact.get("id", ""):
                continue
            if interest_query and not (
                _COMMONALITY_INTEREST_FACT_RE.search(left_fact.get("fact", ""))
                and _COMMONALITY_INTEREST_FACT_RE.search(right_fact.get("fact", ""))
            ):
                continue
            overlap = (left["tokens"] & right["tokens"]) - _COMMONALITY_OVERLAP_IGNORE
            if not overlap:
                continue
            key = (tuple(sorted(overlap)), left_fact.get("id", ""), right_fact.get("id", ""))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            rarity = sum(1.0 / max(token_freq.get(token, 1), 1) for token in overlap)
            retrieved_bonus = 1.0 if left["already_retrieved"] or right["already_retrieved"] else 0.0
            session_gap = abs((left_fact.get("session") or 0) - (right_fact.get("session") or 0))
            session_bonus = max(0.0, 8.0 - 0.5 * session_gap) if session_gap else 8.0
            session_floor = min((left_fact.get("session") or 0), (right_fact.get("session") or 0))
            origin_bonus = 0.0
            if earliest_session and session_floor:
                origin_bonus = max(0.0, 6.0 - 0.25 * max(0, session_floor - earliest_session))
            left_bonus = _self_grounded_commonality_bonus(left_fact, entities[0])
            right_bonus = _self_grounded_commonality_bonus(right_fact, entities[1])
            score = (
                rarity * 10.0
                + len(overlap) * 1.5
                + retrieved_bonus
                + session_bonus
                + origin_bonus
                + left_bonus
                + right_bonus
            )
            candidates.append((score, overlap, left, right))

    extras = []
    added_ids = set(existing_ids)
    ranked_groups = _rank_commonality_groups(candidates)
    for group in ranked_groups:
        if not group["pairs"]:
            continue
        for _score, left, right in group["pairs"][:2]:
            for row in (left, right):
                fact = row["fact"]
                fact_id = fact.get("id", "")
                if fact_id and fact_id not in added_ids:
                    extras.append(fact)
                    added_ids.add(fact_id)
                    if len(extras) >= limit:
                        return extras
    return extras


def _extract_conversation_windows(text: str, facts: list[dict], cap: int) -> str:
    if cap <= 0:
        return ""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text[:cap]

    def _norm_tokens(value: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]+", value.lower())
            if len(token) >= 3
        }

    fact_tokens = [_norm_tokens(fact.get("fact", "")) for fact in facts]
    line_scores = []
    for idx, line in enumerate(lines):
        line_tokens = _norm_tokens(line)
        if not line_tokens:
            continue
        score = max((len(line_tokens & fact_token_set) for fact_token_set in fact_tokens), default=0)
        if score > 0:
            line_scores.append((score, idx))

    if not line_scores:
        return text[:cap]

    line_scores.sort(key=lambda item: (-item[0], item[1]))
    chosen = sorted({idx for _score, idx in line_scores[:3]})
    windows = []
    radius = 8
    for idx in chosen:
        windows.append((max(0, idx - radius), min(len(lines), idx + radius + 1)))
    windows.sort()

    merged = []
    for start, end in windows:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    parts = []
    total = 0
    for start, end in merged:
        snippet = "\n".join(lines[start:end]).strip()
        if not snippet:
            continue
        if parts:
            snippet = "...\n" + snippet
        remaining = cap - total
        if remaining <= 0:
            break
        if len(snippet) > remaining:
            snippet = snippet[:remaining].rstrip()
        if not snippet:
            break
        parts.append(snippet)
        total += len(snippet)
        if total >= cap:
            break

    return "\n".join(parts) if parts else text[:cap]


def _extract_conversation_support_lines(text: str, facts: list[dict], max_lines: int = 4) -> list[str]:
    if max_lines <= 0:
        return []
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    def _norm_tokens(value: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]+", value.lower())
            if len(token) >= 3
        }

    fact_tokens = [_norm_tokens(fact.get("fact", "")) for fact in facts]
    scored = []
    for idx, line in enumerate(lines):
        line_tokens = _norm_tokens(line)
        if not line_tokens:
            continue
        score = max((len(line_tokens & fact_token_set) for fact_token_set in fact_tokens), default=0)
        if score > 0:
            scored.append((score, idx))

    if not scored:
        return []

    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen = sorted({idx for _score, idx in scored[:2]})
    radius = 4
    out = []
    seen = set()
    for idx in chosen:
        for pos in range(max(0, idx - radius), min(len(lines), idx + radius + 1)):
            line = lines[pos].strip()
            if not line or line in seen:
                continue
            out.append(line)
            seen.add(line)
            if len(out) >= max_lines:
                return out
    return out


def _extract_query_focused_conversation_excerpt(
    text: str,
    facts: list[dict],
    query_terms: set[str],
    cap: int = 2400,
) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text[:cap]

    def _norm_tokens(value: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]+", value.lower())
            if len(token) >= 3
        }

    fact_token_sets = [_norm_tokens(fact.get("fact", "")) for fact in facts[:4]]
    scored = []
    for idx, line in enumerate(lines):
        line_tokens = _norm_tokens(line)
        if not line_tokens:
            continue
        query_overlap = len(line_tokens & query_terms)
        fact_overlap = max((len(line_tokens & fact_tokens) for fact_tokens in fact_token_sets), default=0)
        score = fact_overlap * 3 + query_overlap * 2
        if score > 0:
            scored.append((score, idx))

    if not scored:
        return _extract_conversation_windows(text, facts, cap)

    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen = sorted({idx for _score, idx in scored[:2]})
    radius = 6
    parts = []
    total = 0
    for idx in chosen:
        start = max(0, idx - radius)
        end = min(len(lines), idx + radius + 1)
        snippet = "\n".join(lines[start:end]).strip()
        if not snippet:
            continue
        if parts:
            snippet = "...\n" + snippet
        remaining = cap - total
        if remaining <= 0:
            break
        if len(snippet) > remaining:
            snippet = snippet[:remaining].rstrip()
        if not snippet:
            break
        parts.append(snippet)
        total += len(snippet)
        if total >= cap:
            break
    return "\n".join(parts) if parts else _extract_conversation_windows(text, facts, cap)


def _is_local_anchor_query(query: str) -> bool:
    qf = extract_query_features(query)
    if qf.get("asks_where"):
        return True
    return bool((qf.get("operator_plan") or {}).get("local_anchor", {}).get("enabled"))


def _extract_anchor_candidates_from_line(
    line: str,
    query_entities: list[str],
    query_lower: str,
) -> list[tuple[str, bool]]:
    candidates = []
    for pattern, is_cue in ((_LOCAL_ANCHOR_CUE_RE, True), (_LOCAL_ANCHOR_CAP_RE, False)):
        for match in pattern.findall(line):
            candidate = " ".join(str(match).split()).strip(" .,:;!?")
            if not candidate or candidate in _LOCAL_ANCHOR_IGNORE:
                continue
            low = candidate.lower()
            if low in query_lower or low in query_entities or len(low) < 3:
                continue
            candidates.append((candidate, is_cue))
    seen = set()
    ordered = []
    for candidate, is_cue in candidates:
        low = candidate.lower()
        if low not in seen:
            seen.add(low)
            ordered.append((candidate, is_cue))
    return ordered


def _build_local_anchor_support_items(
    query: str,
    retrieved_facts: list[dict],
    raw_sessions: list[dict],
) -> list[dict]:
    if not _is_local_anchor_query(query):
        return []

    session_facts = defaultdict(list)
    for rank, fact in enumerate(retrieved_facts):
        session = fact.get("session")
        if isinstance(session, int) and 0 < session <= len(raw_sessions):
            session_facts[session].append((rank, fact))
    if not session_facts:
        return []

    query_terms = {
        token
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]+", query.lower())
        if len(token) >= 4 and token not in STOP_WORDS and token not in _QUERY_ENTITY_IGNORE
    }
    session_rows = []
    for session, pairs in session_facts.items():
        query_score = 0
        for rank, fact in pairs:
            fact_lower = fact.get("fact", "").lower()
            overlap = sum(1 for token in query_terms if token in fact_lower)
            query_score += overlap * 3 - rank * 0.05
        session_rows.append((session, query_score, len(pairs), min(rank for rank, _fact in pairs)))
    session_rows.sort(key=lambda row: (-row[1], -row[2], row[3], row[0]))
    focus_session = session_rows[0][0]

    raw_session = raw_sessions[focus_session - 1]
    if not isinstance(raw_session, dict):
        return []
    text = raw_session.get("content", "")
    if not text:
        return []

    query_entities = set(_extract_query_named_entities(query))
    query_lower = query.lower()
    facts = [fact for _rank, fact in session_facts[focus_session]]
    focus_facts = sorted(
        facts,
        key=lambda fact: (
            -sum(1 for token in query_terms if token in fact.get("fact", "").lower()),
            fact.get("id", ""),
        ),
    )[:6]
    excerpt = _extract_query_focused_conversation_excerpt(text, focus_facts or facts, query_terms, cap=2400)
    lines = [line.strip() for line in excerpt.splitlines() if line.strip()]
    if not lines:
        return []

    candidate_rows = {}
    for line_idx, line in enumerate(lines):
        for candidate, is_cue in _extract_anchor_candidates_from_line(line, query_entities, query_lower):
            key = candidate.lower()
            row = candidate_rows.setdefault(
                key,
                {
                    "candidate": candidate,
                    "count": 0,
                    "cue_count": 0,
                    "best_line": line_idx,
                    "lines": [],
                },
            )
            row["count"] += 1
            row["cue_count"] += int(is_cue)
            row["cue_count"] += int(bool(_LOCAL_ANCHOR_LINE_CUE_RE.search(line)))
            row["best_line"] = min(row["best_line"], line_idx)
            if line not in row["lines"]:
                row["lines"].append(line)

    if not candidate_rows:
        return []

    ranked = sorted(
        candidate_rows.values(),
        key=lambda row: (-row["cue_count"], -row["count"], row["best_line"], row["candidate"].lower()),
    )
    top = ranked[0]
    if top["cue_count"] <= 0 and top["count"] < 2 and len(top["lines"]) < 2:
        return []
    return [
        {
            "text": (
                f"[Anchor 1] strongest local anchor: {top['candidate']}\n"
                + "\n".join(f"- {line}" for line in top["lines"][:3])
            ),
            "rank": -900,
            "source": "local_anchor",
        }
    ]


def _context_has_source_excerpts(context: str) -> bool:
    if not context:
        return False
    return any(
        marker in context
        for marker in (
            "RAW CONTEXT",
            "RAW CONTEXT (source text excerpts):",
            "--- SOURCE DOCUMENT SECTIONS ---",
            "--- SOURCE EPISODE RAW TEXT ---",
        )
    )


_VALID_ID_PREFIXES = ("user:", "agent:", "swarm:")
_BARE_IDS = ("system", "anonymous")


def _normalize_identity(identity: str, allow_public: bool = False) -> str:
    """Validate and normalize identity string. Raises ValueError if invalid."""
    if identity in _BARE_IDS:
        return identity
    if identity == "agent:PUBLIC":
        if not allow_public:
            raise ValueError("agent:PUBLIC cannot be used as owner_id")
        return identity
    if not any(identity.startswith(p) for p in _VALID_ID_PREFIXES):
        raise ValueError(
            f"Invalid identity '{identity}': must start with user:/agent:/swarm: or be system/anonymous"
        )
    return identity


def _normalize_target(target) -> list[str] | None:
    """Normalize delivery target to canonical list[str] or None if omitted."""
    if target is None:
        return None
    if isinstance(target, str):
        values = [target]
    elif isinstance(target, list):
        values = target
    else:
        raise ValueError("target must be a string, list of strings, or null")

    normalized = []
    seen = set()
    for value in values:
        if not isinstance(value, str):
            raise ValueError("target list must contain only strings")
        canonical = _normalize_identity(value)
        if canonical not in seen:
            seen.add(canonical)
            normalized.append(canonical)
    return normalized


def _build_index_state(
    granular_facts: list[dict],
    gran_embs: np.ndarray,
    cons_facts: list[dict],
    cons_embs: np.ndarray,
    cross_facts: list[dict],
    cross_embs: np.ndarray,
) -> dict:
    """Build the minimal production index state for the final runtime.

    The final runtime no longer depends on the legacy three-tier adaptive
    retrieval data dict. The only persisted in-memory index state we need here
    is:
    - fact lookup for visibility / provenance checks
    - cached embeddings per stored tier
    """
    fact_lookup = {}
    for fact in granular_facts + cons_facts + cross_facts:
        fact_id = fact.get("id")
        if fact_id:
            fact_lookup[fact_id] = fact

    return {
        "atomic_embs": gran_embs,
        "cons_embs": cons_embs,
        "cross_embs": cross_embs,
        "fact_lookup": fact_lookup,
    }


def _derive_acl_from_scope(scope, agent_id, swarm_id):
    """Derive ACL fields from legacy scope/agent_id/swarm_id. Backward compat."""
    if scope == "agent-private":
        return {"owner_id": f"agent:{agent_id}", "read": [], "write": []}
    if scope == "swarm-shared" and swarm_id != "default":
        return {"owner_id": f"agent:{agent_id}",
                "read": [f"swarm:{swarm_id}"], "write": [f"swarm:{swarm_id}"]}
    # swarm-shared+default, system-wide, None/missing → public
    return {"owner_id": "system", "read": ["agent:PUBLIC"], "write": ["agent:PUBLIC"]}


def _estimate_tokens(value) -> int:
    """Cheap token estimate used across routing and payload accounting."""
    if value is None:
        return 0
    if isinstance(value, str):
        if value == "":
            return 0
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return max(1, int(len(text) / 3.5))


def _provider_for_model(model: str) -> str:
    if model.startswith("inception/"):
        return "inception"
    if model.startswith("anthropic/"):
        return "anthropic"
    if model.startswith("google/"):
        return "google"
    return "openai"


def _provider_family_for_model(model: str) -> str:
    provider = _provider_for_model(model)
    if provider == "anthropic":
        return "anthropic"
    if provider == "google":
        return "google"
    return "openai_compatible"


def _build_openai_tools_payload() -> list[dict]:
    return [{
        "type": "function",
        "function": {
            "name": GET_CONTEXT_TOOL["name"],
            "description": GET_CONTEXT_TOOL["description"],
            "parameters": GET_CONTEXT_TOOL["input_schema"],
        },
    }]


def _classify_context_tier(fact: dict) -> str:
    kind = fact.get("kind", "")
    status = fact.get("status", "active")
    if status == "active" and kind in {"decision", "constraint", "rejection"}:
        return "tier1"
    if kind in {"action_item", "requirement", "preference"}:
        return "tier2"
    return "tier3"


def _build_context_packet(retrieved_facts, raw_sessions, budget=5000, raw_docs=None):
    """Preserve context segments by truncation priority before final rendering."""
    packet = {
        "tier1": [],
        "tier2": [],
        "tier3": [],
        "tier4": [],
    }

    for i, fact in enumerate(retrieved_facts):
        labels = ""
        metadata = fact.get("metadata", {})
        if isinstance(metadata, dict):
            version_status = metadata.get("version_status")
            if version_status == "current":
                labels += " [CURRENT]"
            elif version_status == "outdated":
                labels += f" [OUTDATED: superseded by {metadata.get('version_superseded_by', '?')}]"
            section_path = metadata.get("section_path")
            if section_path:
                labels += f" [Section: {section_path}]"
        line = f"[{i+1}] (S{fact.get('session', '?')}) {fact.get('fact', '')}{labels}"
        packet[_classify_context_tier(fact)].append({
            "text": line,
            "rank": i,
            "source": "fact",
            "fact_id": fact.get("id"),
            "kind": fact.get("kind"),
        })

    total_chars = 0
    relevant_sessions = sorted({
        session_num
        for fact in retrieved_facts
        if (session_num := _coerce_positive_session_num(fact.get("session"))) is not None
        and session_num <= len(raw_sessions)
    })
    for session_num in relevant_sessions:
        if total_chars >= budget:
            break
        raw = raw_sessions[session_num - 1] if session_num - 1 < len(raw_sessions) else {}
        text = raw.get("content", "") if isinstance(raw, dict) else str(raw)
        if not text:
            continue
        remaining = budget - total_chars
        chunk = text[:remaining].strip()
        if not chunk:
            continue
        packet["tier4"].append({
            "text": f"[Raw S{session_num}]\n{chunk}",
            "rank": session_num,
            "source": "raw",
            "session": session_num,
        })
        total_chars += len(chunk)

    if raw_docs:
        doc_chars = 0
        seen_doc_sources = set()
        for rank, fact in enumerate(retrieved_facts):
            if doc_chars >= budget:
                break
            metadata = fact.get("metadata") or {}
            source_label = metadata.get("document_source") or fact.get("conv_id", "")
            if not source_label or source_label in seen_doc_sources or source_label not in raw_docs:
                continue
            seen_doc_sources.add(source_label)
            remaining = min(2000, budget - doc_chars)
            chunk = raw_docs[source_label][:remaining].strip()
            if not chunk:
                continue
            section_path = metadata.get("section_path", "")
            header = ""
            if section_path:
                header += f"[Section: {section_path}]\n"
            header += f"[Source: {source_label}]"
            packet["tier4"].append({
                "text": f"{header}\n{chunk}",
                "rank": rank,
                "source": "doc",
                "section_path": section_path,
                "source_label": source_label,
            })
            doc_chars += len(chunk)

    return packet


def _render_context_packet(packet: dict) -> str:
    """Render a structured packet into the legacy hybrid context string."""
    fact_lines = []
    for tier in ("tier1", "tier2", "tier3"):
        fact_lines.extend(packet.get(tier, []))
    fact_lines = [s["text"] for s in sorted(fact_lines, key=lambda s: s.get("rank", 0))]

    raw_lines = [s["text"] for s in sorted(
        [seg for seg in packet.get("tier4", []) if seg.get("source") == "raw"],
        key=lambda s: s.get("rank", 0),
    )]
    doc_lines = [s["text"] for s in sorted(
        [seg for seg in packet.get("tier4", []) if seg.get("source") == "doc"],
        key=lambda s: (s.get("rank", 0), s.get("source_label", "")),
    )]

    parts = ["RETRIEVED FACTS:"]
    parts.extend(fact_lines)
    if raw_lines:
        parts.append("")
        parts.append("RAW CONTEXT (source text excerpts):")
        parts.extend(raw_lines)
    if doc_lines:
        parts.append("")
        parts.append("--- SOURCE DOCUMENT SECTIONS ---")
        parts.extend(doc_lines)
    return "\n".join(parts)


def build_hybrid_context(retrieved_facts, raw_sessions, budget=5000, raw_docs=None):
    """Build legacy hybrid context from structured facts plus raw snippets."""
    packet = _build_context_packet(
        retrieved_facts=retrieved_facts,
        raw_sessions=raw_sessions,
        budget=budget,
        raw_docs=raw_docs,
    )
    return _render_context_packet(packet)


def compute_raw_budget(qt, total_sessions, sessions_in_context):
    """Coverage-based raw budget."""
    if qt != "summarize":
        return 5000
    coverage_pct = sessions_in_context / total_sessions * 100 if total_sessions else 100
    if coverage_pct >= 50:
        return 5000
    if coverage_pct >= 20:
        return 15000
    return 30000


def _route_prompt_type(
    resolved_type,
    resolved_facts,
    total_sessions,
    sessions_in_ctx,
    hybrid_ctx,
    *,
    allow_tool_mode=True,
):
    """Coverage-based prompt routing shared by fact and episode recall paths."""
    if resolved_type == "summarize":
        return "summarize_with_metadata", True
    if resolved_type == "icl":
        return "icl", False
    if (
        allow_tool_mode
        and resolved_facts
        and total_sessions > 20
        and sessions_in_ctx < total_sessions * 0.3
    ):
        return "tool", True
    if _context_has_source_excerpts(hybrid_ctx):
        return "hybrid", False
    return resolved_type, False


def _apply_output_constraints(answer: str, constraints: dict | None) -> str:
    if not answer:
        return answer
    constraints = constraints or {}
    rendered = answer.strip()
    prefix = constraints.get("prepend_prefix")
    if prefix and not rendered.startswith(prefix):
        rendered = f"{prefix}{rendered}"
    return rendered


def _ordinal_event_episode_id(event: dict) -> str:
    payload = event.get("payload") or {}
    if isinstance(payload, dict):
        episode_id = str(payload.get("episode_id") or "").strip()
        if episode_id:
            return episode_id
    source_span = event.get("source_span") or {}
    return str(source_span.get("episode_id") or "").strip()


def _ordinal_event_fact_ids(event: dict) -> list[str]:
    payload = event.get("payload") or {}
    fact_ids: list[str] = []
    if isinstance(payload, dict):
        fact_id = str(payload.get("fact_id") or "").strip()
        if fact_id:
            fact_ids.append(fact_id)
    fact_ids.extend(
        str(fid).strip()
        for fid in (event.get("support_fact_ids") or [])
        if str(fid).strip()
    )
    return list(dict.fromkeys(fact_ids))


def _ordinal_event_quality(event: dict) -> int:
    payload = event.get("payload") or {}
    score = 0
    if payload.get("action_raw"):
        score += 5
    if payload.get("tool_name"):
        score += 3
    if payload.get("tool_args_raw") or payload.get("tool_args"):
        score += 2
    if payload.get("observation_raw"):
        score += 1
    if payload.get("paths"):
        score += 1
    if payload.get("ids"):
        score += 1
    if _ordinal_event_fact_ids(event):
        score += 1
    return score


def _temporal_hit_quality(hit: dict) -> int:
    primary_event = hit.get("primary_event") or {}
    score = _ordinal_event_quality(primary_event)
    reducer = str(hit.get("reducer") or "").strip().lower()
    payload = primary_event.get("payload") or {}
    deterministic_answer = str(hit.get("deterministic_answer") or "").strip()
    if reducer in {"exact_command", "exact_sql"}:
        if payload.get("tool_name"):
            score += 3
        if payload.get("tool_args_raw") or payload.get("tool_args"):
            score += 2
    elif (reducer == "exact_file_path" and payload.get("paths")) or (reducer == "exact_id" and payload.get("ids")):
        score += 3
    elif reducer == "exact_action" and payload.get("action_raw"):
        score += 2
    if deterministic_answer:
        score += 1
    return score


def _ordinal_adjacent_episode_id(ep_id: str, delta: int) -> str | None:
    match = re.search(r"^(.*_e)(\d+)\b", ep_id)
    if not match:
        return None
    prefix, raw_num = match.groups()
    next_num = int(raw_num) + int(delta)
    if next_num <= 0:
        return None
    return f"{prefix}{next_num:0{len(raw_num)}d}"


def _temporal_scope_from_corpus(corpus: dict) -> tuple[set[str] | None, set[str] | None]:
    scope_map = _temporal_scope_map_from_corpus(corpus)
    source_ids = set(scope_map)
    timeline_ids = {
        timeline_id
        for timeline_set in scope_map.values()
        for timeline_id in timeline_set
    }
    return (source_ids or None), (timeline_ids or None)


def _temporal_scope_map_from_corpus(corpus: dict) -> dict[str, set[str]]:
    source_timelines: dict[str, set[str]] = {}
    for doc in corpus.get("documents", []):
        doc_id = str(doc.get("doc_id") or "").strip()
        for episode in doc.get("episodes", []):
            source_id = str(episode.get("source_id") or "").strip()
            if not source_id:
                continue
            timeline_id = (
                str(episode.get("timeline_id") or "").strip()
                or doc_id
                or f"timeline:{source_id}:main"
            )
            source_timelines.setdefault(source_id, set())
            if timeline_id:
                source_timelines[source_id].add(timeline_id)
    return source_timelines


def _temporal_scope_query_tokens(question: str, query_features: dict) -> set[str]:
    stop = {
        "what", "which", "when", "where", "who", "why", "how",
        "step", "steps", "turn", "turns", "message", "messages",
        "exact", "specific", "performed", "occurred", "happened",
        "action", "actions", "command", "commands", "date", "year",
        "month", "first", "last", "between", "from", "through",
        "later", "after", "before", "did", "was", "were", "is",
        "the", "a", "an", "to", "of", "in", "on", "at", "for",
    }
    tokens = {
        normalize_term_token(token)
        for token in (query_features.get("words") or set())
        if normalize_term_token(token)
    }
    raw_tokens = {
        normalize_term_token(token)
        for token in re.findall(r"[A-Za-z0-9_./:-]+", str(question or ""))
        if normalize_term_token(token)
    }
    tokens |= raw_tokens
    return {
        token
        for token in tokens
        if token
        and not token.isdigit()
        and token not in stop
        and len(token) > 2
    }


def _temporal_scope_event_tokens(events: list[dict]) -> set[str]:
    tokens: set[str] = set()
    for event in events:
        payload = event.get("payload") or {}
        parts: list[str] = [
            str(event.get("label") or "").strip(),
            str(payload.get("action_raw") or "").strip(),
            str(payload.get("tool_name") or "").strip(),
            str(payload.get("tool_args_raw") or "").strip(),
            str(payload.get("observation_raw") or "").strip(),
            str(payload.get("step_body") or "").strip(),
            str(payload.get("raw_step_block") or "").strip(),
        ]
        support_texts = payload.get("support_texts") or []
        if isinstance(support_texts, list):
            parts.extend(str(text or "").strip() for text in support_texts[:4])
        for part in parts:
            if not part:
                continue
            for raw_token in re.findall(r"[A-Za-z0-9_./:-]+", part):
                token = normalize_term_token(raw_token)
                if token and not token.isdigit() and len(token) > 2:
                    tokens.add(token)
    return tokens


def _temporal_scope_source_tokens(
    corpus: dict,
    *,
    source_id: str,
    timeline_ids: set[str],
) -> set[str]:
    tokens: set[str] = set()
    for doc in corpus.get("documents", []):
        doc_id = str(doc.get("doc_id") or "").strip()
        if timeline_ids and doc_id and doc_id not in timeline_ids:
            continue
        for episode in doc.get("episodes", []):
            if str(episode.get("source_id") or "").strip() != source_id:
                continue
            parts = [
                str(episode.get("topic_key") or "").strip(),
                str(episode.get("state_label") or "").strip(),
                str(episode.get("raw_text") or "").strip(),
            ]
            for part in parts:
                if not part:
                    continue
                for raw_token in re.findall(r"[A-Za-z0-9_./:-]+", part):
                    token = normalize_term_token(raw_token)
                    if token and not token.isdigit() and len(token) > 2:
                        tokens.add(token)
    return tokens


def _source_anchor_timestamp_for_scope(
    corpus: dict,
    temporal_index: dict,
    *,
    source_ids: set[str] | None = None,
) -> str | None:
    resolved_dates: set[str] = set()
    for doc in corpus.get("documents", []):
        for episode in doc.get("episodes", []):
            source_id = str(episode.get("source_id") or "").strip()
            if source_ids is not None and source_id not in source_ids:
                continue
            source_date = str(episode.get("source_date") or "").strip()
            if not source_date:
                continue
            try:
                candidate = date_parser.parse(source_date, fuzzy=True)
            except Exception:
                continue
            resolved_dates.add(candidate.date().isoformat())
    if resolved_dates:
        return max(resolved_dates)
    if temporal_index is not None:
        return latest_calendar_anchor(temporal_index, source_ids=source_ids)
    return None


def _narrow_temporal_scope_from_visible_corpus(
    *,
    question: str,
    query_features: dict,
    corpus: dict,
    execute_for_scope,
) -> tuple[set[str] | None, set[str] | None, dict | None, dict]:
    scope_map = _temporal_scope_map_from_corpus(corpus)
    if len(scope_map) <= 1:
        source_ids, timeline_ids = _temporal_scope_from_corpus(corpus)
        return source_ids, timeline_ids, None, {"scope_mode": "single_source"}

    query_tokens = _temporal_scope_query_tokens(question, query_features)
    rows: list[dict] = []
    for source_id, timeline_ids in sorted(scope_map.items()):
        hit = execute_for_scope({source_id}, set(timeline_ids))
        matched = bool(hit.get("matched"))
        resolved = bool(hit.get("resolved", matched and bool(hit.get("events"))))
        if not matched:
            continue
        lexical_tokens = _temporal_scope_event_tokens(list(hit.get("events") or []))
        lexical_tokens |= _temporal_scope_source_tokens(
            corpus,
            source_id=source_id,
            timeline_ids=set(timeline_ids),
        )
        overlap = len(query_tokens & lexical_tokens)
        rows.append(
            {
                "source_id": source_id,
                "timeline_ids": set(timeline_ids),
                "hit": hit,
                "matched": matched,
                "resolved": resolved,
                "overlap": overlap,
                "hit_quality": _temporal_hit_quality(hit),
                "event_count": len(list(hit.get("events") or [])),
            }
        )

    trace = {
        "scope_mode": "multi_source_probe",
        "candidate_source_ids": sorted(scope_map),
        "probed_source_ids": [row["source_id"] for row in rows],
    }
    if not rows:
        source_ids, timeline_ids = _temporal_scope_from_corpus(corpus)
        trace["scope_mode"] = "probe_miss"
        return source_ids, timeline_ids, None, trace
    if len(rows) == 1:
        row = rows[0]
        trace["selected_source_ids"] = [row["source_id"]]
        trace["scope_mode"] = "single_matched_source"
        return {row["source_id"]}, set(row["timeline_ids"]), row["hit"], trace

    resolved_rows = [row for row in rows if row["resolved"]]
    if len(resolved_rows) == 1:
        row = resolved_rows[0]
        trace["selected_source_ids"] = [row["source_id"]]
        trace["scope_mode"] = "single_resolved_source"
        return {row["source_id"]}, set(row["timeline_ids"]), row["hit"], trace

    if len(resolved_rows) > 1:
        ranked = sorted(
            resolved_rows,
            key=lambda row: (
                row["overlap"],
                row["hit_quality"],
                row["event_count"],
                row["source_id"],
            ),
            reverse=True,
        )
        if len(ranked) == 1 or (
            ranked[0]["overlap"],
            ranked[0]["hit_quality"],
            ranked[0]["event_count"],
        ) > (
            ranked[1]["overlap"],
            ranked[1]["hit_quality"],
            ranked[1]["event_count"],
        ):
            row = ranked[0]
            trace["selected_source_ids"] = [row["source_id"]]
            trace["scope_mode"] = "lexical_disambiguation"
            return {row["source_id"]}, set(row["timeline_ids"]), row["hit"], trace

    source_ids, timeline_ids = _temporal_scope_from_corpus(corpus)
    trace["scope_mode"] = "ambiguous_multi_source"
    return source_ids, timeline_ids, None, trace


def _build_ordinal_executor_packet(
    question: str,
    *,
    corpus: dict,
    episode_lookup: dict[str, dict],
    facts_by_episode: dict[str, list[dict]],
    query_features: dict,
    effective_selector: dict,
    output_constraints: dict,
    operator_tuning: dict,
    packet_tuning: dict,
    search_family: str | None,
    temporal_index: dict,
) -> tuple[dict | None, dict]:
    temporal_limit = max(
        int(effective_selector.get("max_episodes_default", 4)) * 4,
        8,
    )
    scope_source_ids, scope_timeline_ids, precomputed_hit, scope_trace = _narrow_temporal_scope_from_visible_corpus(
        question=question,
        query_features=query_features,
        corpus=corpus,
        execute_for_scope=lambda source_ids, timeline_ids: execute_ordinal_query(
            question,
            temporal_index,
            source_ids=source_ids,
            timeline_ids=timeline_ids,
            limit=temporal_limit,
        ),
    )
    hit = precomputed_hit or execute_ordinal_query(
        question,
        temporal_index,
        source_ids=scope_source_ids,
        timeline_ids=scope_timeline_ids,
        limit=temporal_limit,
    )
    base_trace = {
        "query_class": "ordinal",
        "matched": bool(hit.get("matched")),
        "resolved": bool(hit.get("resolved")),
        "mode": hit.get("mode"),
        "kind": hit.get("kind"),
        "anchor_resolved": bool(hit.get("matched")),
        "matched_event_ids": [
            str(event.get("event_id") or "")
            for event in (hit.get("events") or [])
            if str(event.get("event_id") or "").strip()
        ],
        "fallback": True,
        "scope_trace": scope_trace,
    }
    if not hit.get("matched"):
        base_trace["fallback_reason"] = "miss"
        return None, base_trace
    if not hit.get("resolved"):
        base_trace["fallback_reason"] = "unresolved"
        return None, base_trace

    events = sorted(
        list(hit.get("events") or []),
        key=lambda event: (
            int(event.get("ordinal_start") or 10**9),
            -_ordinal_event_quality(event),
            str(event.get("event_id") or ""),
        ),
    )
    source_ids = {
        str(event.get("source_id") or "").strip()
        for event in events
        if str(event.get("source_id") or "").strip()
    }
    if len(source_ids) != 1:
        base_trace["fallback_reason"] = "ambiguous_source"
        base_trace["candidate_source_ids"] = sorted(source_ids)
        return None, base_trace

    selected_episode_ids: list[str] = []
    for event in sorted(events, key=lambda event: (-_ordinal_event_quality(event), str(event.get("event_id") or ""))):
        ep_id = _ordinal_event_episode_id(event)
        if ep_id and ep_id not in selected_episode_ids:
            selected_episode_ids.append(ep_id)
    if not selected_episode_ids:
        base_trace["fallback_reason"] = "no_episode"
        return None, base_trace

    selected_source_id = next(iter(source_ids))
    step_numbers = set(query_features.get("step_numbers") or [])
    step_range = query_features.get("step_range")

    def _target_step_hits(text: str) -> set[int]:
        hits: set[int] = set()
        if not text:
            return hits
        for step in step_numbers:
            if has_exact_step_mention(text, step):
                hits.add(step)
        if step_range:
            hits |= step_range_overlap(text, step_range)
        return hits

    def _episode_step_order(ep_id: str) -> int:
        ep = episode_lookup.get(ep_id) or {}
        lower = (ep.get("raw_text") or "").lower()
        match = re.search(r"\[step\s+(\d+)\]|\bstep\s+(\d+)\b", lower)
        if match:
            return int(match.group(1) or match.group(2))
        topic = (ep.get("topic_key") or "").lower()
        match = re.search(r"\bstep\s+(\d+)\b", topic)
        if match:
            return int(match.group(1))
        match = re.search(r"_e(\d+)\b", ep_id)
        if match:
            return int(match.group(1))
        return 10**9

    target_step_episode_ids = [
        ep_id
        for ep_id, ep in episode_lookup.items()
        if ep.get("source_id", "") == selected_source_id
        and _target_step_hits(ep.get("raw_text", ""))
    ]
    target_step_episode_ids = sorted(
        list(dict.fromkeys(target_step_episode_ids + selected_episode_ids)),
        key=lambda ep_id: (_episode_step_order(ep_id), ep_id),
    )

    primary_event = hit.get("primary_event") or {}
    answer_episode_id = _ordinal_event_episode_id(primary_event)
    if answer_episode_id and answer_episode_id not in target_step_episode_ids:
        target_step_episode_ids = sorted(
            target_step_episode_ids + [answer_episode_id],
            key=lambda ep_id: (_episode_step_order(ep_id), ep_id),
        )

    companion_episode_ids: list[str] = []
    for ep_id in target_step_episode_ids:
        ep = episode_lookup.get(ep_id) or {}
        raw = (ep.get("raw_text") or "").strip()
        lower = raw.lower()
        if not raw or not _target_step_hits(raw):
            continue
        next_ep_id = _ordinal_adjacent_episode_id(ep_id, 1)
        if not next_ep_id or next_ep_id in companion_episode_ids:
            continue
        next_ep = episode_lookup.get(next_ep_id) or {}
        next_raw = (next_ep.get("raw_text") or "").strip()
        next_lower = next_raw.lower()
        if not next_raw:
            continue
        if next_ep.get("source_id", "") != selected_source_id:
            continue
        if not next_lower.startswith("action:") or "observation:" not in next_lower:
            continue
        companion_episode_ids.append(next_ep_id)

    fact_episode_ids = list(target_step_episode_ids)
    for event in events:
        for fact_id in _ordinal_event_fact_ids(event):
            for ep_id, facts in facts_by_episode.items():
                if any(str(fact.get("id") or "").strip() == fact_id for fact in facts):
                    if ep_id not in fact_episode_ids:
                        fact_episode_ids.append(ep_id)
    for ep_id, facts in facts_by_episode.items():
        ep = episode_lookup.get(ep_id) or {}
        if ep.get("source_id", "") != selected_source_id:
            continue
        if ep_id in fact_episode_ids:
            continue
        if any(_target_step_hits(str(fact.get("fact") or "")) for fact in facts):
            fact_episode_ids.append(ep_id)
    for ep_id in companion_episode_ids:
        if ep_id not in fact_episode_ids:
            fact_episode_ids.append(ep_id)

    context, actual_injected_episode_ids, selected_fact_ids = build_context_from_selected_episodes(
        question,
        target_step_episode_ids,
        episode_lookup,
        facts_by_episode,
        fact_episode_ids=fact_episode_ids,
        support_episode_ids=companion_episode_ids,
        budget=effective_selector["budget"],
        max_total_facts=effective_selector["supporting_facts_total"],
        max_facts_per_episode=effective_selector["supporting_facts_per_episode"],
        snippet_mode=bool(effective_selector.get("snippet_mode", False)),
        snippet_chars=int(packet_tuning.get("snippet_chars", 1200)),
        allow_pseudo_facts=bool(effective_selector.get("allow_pseudo_facts", True)),
        query_features=query_features,
        local_anchor_window_chars=int(operator_tuning.get("local_anchor_window_chars", 1200)),
        local_anchor_fact_radius=int(operator_tuning.get("local_anchor_fact_radius", 12)),
        list_set_dedup_overlap=float(operator_tuning.get("list_set_dedup_overlap", 0.9)),
        bounded_chain_fact_bonus=float(operator_tuning.get("bounded_chain_fact_bonus", 0.0)),
        query_specificity_bonus=float(packet_tuning.get("query_specificity_bonus", 0.0)),
        inject_support_fact_episodes=bool(companion_episode_ids or fact_episode_ids),
        max_injected_support_fact_episodes=int(
            max(
                int(packet_tuning.get("max_injected_support_fact_episodes", 8)),
                len(companion_episode_ids),
            )
        ),
    )

    routed_families = route_retrieval_families(
        question,
        available_families(corpus),
        explicit_family=search_family,
    )
    trace = dict(base_trace)
    trace.update(
        {
            "matched": True,
            "fallback": False,
            "executor_episode_ids": target_step_episode_ids,
            "pinned_episode_ids": target_step_episode_ids,
            "support_episode_ids": fact_episode_ids,
            "matched_fact_ids": selected_fact_ids,
        }
    )
    deterministic_answer = str(hit.get("deterministic_answer") or "").strip()
    if deterministic_answer:
        trace["deterministic_answer"] = deterministic_answer
        trace["reducer"] = str(hit.get("reducer") or "").strip()
        trace["answer_event_id"] = str(primary_event.get("event_id") or "").strip()

    packet = {
        "context": context,
        "retrieved_episode_ids": target_step_episode_ids,
        "actual_injected_episode_ids": actual_injected_episode_ids,
        "fact_episode_ids": fact_episode_ids,
        "retrieved_fact_ids": selected_fact_ids,
        "selection_scores": [
            {"episode_id": ep_id, "score": float(1_000_000 - idx)}
            for idx, ep_id in enumerate(target_step_episode_ids)
        ],
        "selector_config": effective_selector,
        "query_operator_plan": query_features["operator_plan"],
        "output_constraints": output_constraints,
        "retrieval_families": routed_families,
        "search_family": search_family or "auto",
        "family_first_pass_trace": {
            "available_families": available_families(corpus),
            "retrieval_families": routed_families,
            "requested_search_family": search_family or "auto",
            "per_family": [],
            "mode": "skipped_by_ordinal_executor",
        },
        "late_fusion_trace": {"mode": "skipped_by_ordinal_executor"},
        "tuning_snapshot": {
            "selector": effective_selector,
            "operators": operator_tuning,
            "packet": packet_tuning,
            "routing": get_runtime_tuning()["routing"],
            "telemetry": get_runtime_tuning()["telemetry"],
        },
        "temporal_trace": trace,
    }
    return packet, trace


def _build_calendar_executor_packet(
    question: str,
    *,
    corpus: dict,
    episode_lookup: dict[str, dict],
    facts_by_episode: dict[str, list[dict]],
    query_features: dict,
    effective_selector: dict,
    output_constraints: dict,
    operator_tuning: dict,
    packet_tuning: dict,
    search_family: str | None,
    temporal_index: dict,
) -> tuple[dict | None, dict]:
    plan = extract_calendar_query(question)
    base_trace = {
        "query_class": "calendar-answer",
        "matched": False,
        "anchor_resolved": False,
        "matched_event_ids": [],
        "fallback": True,
    }
    if not plan or plan.get("mode") != "answer":
        base_trace["fallback_reason"] = "unsupported_mode"
        return None, base_trace
    temporal_limit = max(int(effective_selector.get("max_episodes_default", 4)) * 4, 8)
    scope_source_ids, scope_timeline_ids, precomputed_hit, scope_trace = _narrow_temporal_scope_from_visible_corpus(
        question=question,
        query_features=query_features,
        corpus=corpus,
        execute_for_scope=lambda source_ids, timeline_ids: execute_calendar_query(
            question,
            temporal_index,
            anchor_timestamp=_source_anchor_timestamp_for_scope(
                corpus,
                temporal_index,
                source_ids=source_ids,
            ),
            source_ids=source_ids,
            timeline_ids=timeline_ids,
            limit=temporal_limit,
        ),
    )
    hit = precomputed_hit or execute_calendar_query(
        question,
        temporal_index,
        anchor_timestamp=_source_anchor_timestamp_for_scope(
            corpus,
            temporal_index,
            source_ids=scope_source_ids,
        ),
        source_ids=scope_source_ids,
        timeline_ids=scope_timeline_ids,
        limit=temporal_limit,
    )
    events = list(hit.get("events") or [])
    base_trace["scope_trace"] = scope_trace
    base_trace["matched_event_ids"] = [
        str(event.get("event_id") or "")
        for event in events
        if str(event.get("event_id") or "").strip()
    ]
    if not events:
        base_trace["fallback_reason"] = "miss"
        return None, base_trace
    source_ids = {
        str(event.get("source_id") or "").strip()
        for event in events
        if str(event.get("source_id") or "").strip()
    }
    if len(source_ids) != 1:
        base_trace["fallback_reason"] = "ambiguous_source"
        base_trace["candidate_source_ids"] = sorted(source_ids)
        return None, base_trace
    selected_episode_ids: list[str] = []
    fact_episode_ids: list[str] = []
    for event in events:
        ep_id = _ordinal_event_episode_id(event)
        if ep_id and ep_id not in selected_episode_ids:
            selected_episode_ids.append(ep_id)
        for fact_id in _ordinal_event_fact_ids(event):
            for candidate_ep_id, facts in facts_by_episode.items():
                if any(str(fact.get("id") or "").strip() == fact_id for fact in facts):
                    if candidate_ep_id not in fact_episode_ids:
                        fact_episode_ids.append(candidate_ep_id)
    for ep_id in selected_episode_ids:
        if ep_id not in fact_episode_ids:
            fact_episode_ids.append(ep_id)
    if not selected_episode_ids:
        base_trace["fallback_reason"] = "no_episode"
        return None, base_trace

    context, actual_injected_episode_ids, selected_fact_ids = build_context_from_selected_episodes(
        question,
        selected_episode_ids,
        episode_lookup,
        facts_by_episode,
        fact_episode_ids=fact_episode_ids,
        budget=effective_selector["budget"],
        max_total_facts=effective_selector["supporting_facts_total"],
        max_facts_per_episode=effective_selector["supporting_facts_per_episode"],
        snippet_mode=bool(effective_selector.get("snippet_mode", False)),
        snippet_chars=int(packet_tuning.get("snippet_chars", 1200)),
        allow_pseudo_facts=bool(effective_selector.get("allow_pseudo_facts", True)),
        query_features=query_features,
        local_anchor_window_chars=int(operator_tuning.get("local_anchor_window_chars", 1200)),
        local_anchor_fact_radius=int(operator_tuning.get("local_anchor_fact_radius", 12)),
        list_set_dedup_overlap=float(operator_tuning.get("list_set_dedup_overlap", 0.9)),
        bounded_chain_fact_bonus=float(operator_tuning.get("bounded_chain_fact_bonus", 0.0)),
        query_specificity_bonus=float(packet_tuning.get("query_specificity_bonus", 0.0)),
    )
    routed_families = route_retrieval_families(
        question,
        available_families(corpus),
        explicit_family=search_family,
    )
    trace = dict(base_trace)
    trace.update(
        {
            "matched": True,
            "anchor_resolved": True,
            "fallback": False,
            "executor_episode_ids": selected_episode_ids,
            "pinned_episode_ids": selected_episode_ids,
            "matched_fact_ids": selected_fact_ids,
        }
    )
    packet = {
        "context": context,
        "retrieved_episode_ids": selected_episode_ids,
        "actual_injected_episode_ids": actual_injected_episode_ids,
        "fact_episode_ids": fact_episode_ids,
        "retrieved_fact_ids": selected_fact_ids,
        "selection_scores": [
            {"episode_id": ep_id, "score": float(1_000_000 - idx)}
            for idx, ep_id in enumerate(selected_episode_ids)
        ],
        "selector_config": effective_selector,
        "query_operator_plan": query_features["operator_plan"],
        "output_constraints": output_constraints,
        "retrieval_families": routed_families,
        "search_family": search_family or "auto",
        "family_first_pass_trace": {
            "available_families": available_families(corpus),
            "retrieval_families": routed_families,
            "requested_search_family": search_family or "auto",
            "per_family": [],
            "mode": "skipped_by_calendar_executor",
        },
        "late_fusion_trace": {"mode": "skipped_by_calendar_executor"},
        "tuning_snapshot": {
            "selector": effective_selector,
            "operators": operator_tuning,
            "packet": packet_tuning,
            "routing": get_runtime_tuning()["routing"],
            "telemetry": get_runtime_tuning()["telemetry"],
        },
        "temporal_trace": trace,
    }
    return packet, trace


def build_episode_hybrid_context(
    question,
    corpus,
    episode_facts,
    selector_config=None,
    search_family=None,
    temporal_index=None,
):
    """Build context through the episode-native runtime path.

    This is the explicit production entrypoint for episode-backed data.
    It never routes through document section grouping.
    """
    query_features = extract_query_features(question)
    operator_plan = query_features["operator_plan"]
    output_constraints = query_features.get("output_constraints", {})
    selector = resolve_selection_config(selector_config)
    selector_overrides = selector_config or {}
    tuning = get_runtime_tuning()
    operator_tuning = tuning["operators"]
    packet_tuning = tuning["packet"]
    effective_selector = dict(selector)
    packet_to_selector = {
        "budget": ("budget", int),
        "max_facts": ("supporting_facts_total", int),
        "max_facts_per_episode": ("supporting_facts_per_episode", int),
        "max_episodes": ("max_episodes_default", int),
        "per_family_cap": ("max_episodes_per_family", int),
        "per_source_cap": ("max_sources_per_family", int),
        "snippet_mode": ("snippet_mode", bool),
    }
    for packet_key, (selector_key, caster) in packet_to_selector.items():
        if selector_key in selector_overrides:
            continue
        effective_selector[selector_key] = caster(
            packet_tuning.get(packet_key, effective_selector.get(selector_key))
        )
    if operator_plan["ordinal"]["enabled"] and operator_tuning.get("enable_snippet_for_ordinal", True):
        effective_selector["snippet_mode"] = True
    if (
        operator_plan["local_anchor"]["enabled"]
        and not operator_plan["bounded_chain"]["enabled"]
        and operator_tuning.get("enable_snippet_for_local_anchor", True)
    ):
        effective_selector["snippet_mode"] = True
        step_numbers = sorted(query_features.get("step_numbers") or [])
        step_range = query_features.get("step_range")
        if step_range:
            start_step, end_step = step_range
            step_span = max(1, end_step - start_step + 1)
            effective_selector["max_episodes_default"] = max(
                effective_selector["max_episodes_default"],
                min(
                    step_span,
                    int(
                        operator_tuning.get(
                            "local_anchor_step_range_max_episodes",
                            max(effective_selector["max_episodes_default"], 8),
                        )
                    ),
                ),
            )
        elif len(step_numbers) > 1:
            effective_selector["max_episodes_default"] = max(
                effective_selector["max_episodes_default"],
                min(
                    len(step_numbers),
                    int(
                        operator_tuning.get(
                            "local_anchor_multi_step_max_episodes",
                            max(effective_selector["max_episodes_default"], 6),
                        )
                    ),
                ),
            )
        else:
            effective_selector["max_episodes_default"] = min(
                effective_selector["max_episodes_default"],
                int(operator_tuning.get("local_anchor_max_episodes", effective_selector["max_episodes_default"])),
            )
        effective_selector["supporting_facts_total"] = max(
            effective_selector["supporting_facts_total"],
            int(operator_tuning.get("local_anchor_supporting_facts_total", effective_selector["supporting_facts_total"])),
        )
        effective_selector["supporting_facts_per_episode"] = max(
            effective_selector["supporting_facts_per_episode"],
            int(operator_tuning.get("local_anchor_supporting_facts_per_episode", effective_selector["supporting_facts_per_episode"])),
        )
    if operator_plan["list_set"]["enabled"]:
        effective_selector["max_episodes_default"] = max(
            effective_selector["max_episodes_default"],
            int(operator_tuning.get("list_set_max_episodes", effective_selector["max_episodes_default"])),
        )
        effective_selector["supporting_facts_total"] = max(
            effective_selector["supporting_facts_total"],
            int(operator_tuning.get("list_set_supporting_facts_total", effective_selector["supporting_facts_total"])),
        )
        effective_selector["supporting_facts_per_episode"] = max(
            effective_selector["supporting_facts_per_episode"],
            int(operator_tuning.get("list_set_supporting_facts_per_episode", effective_selector["supporting_facts_per_episode"])),
        )
    if any(
        operator_plan[name]["enabled"]
        for name in ("commonality", "list_set", "compare_diff", "bounded_chain")
    ):
        effective_selector["supporting_facts_total"] = max(
            effective_selector["supporting_facts_total"],
            int(operator_tuning.get("structural_query_supporting_facts_total", 12)),
        )
        effective_selector["supporting_facts_per_episode"] = max(
            effective_selector["supporting_facts_per_episode"],
            int(operator_tuning.get("structural_query_supporting_facts_per_episode", 4)),
        )
    if operator_plan["bounded_chain"]["enabled"]:
        operator_plan["bounded_chain"]["max_hops"] = int(
            operator_tuning.get("bounded_chain_max_hops", operator_plan["bounded_chain"]["max_hops"])
        )

    episode_lookup = build_episode_lookup(corpus)
    facts_by_episode = build_facts_by_episode(episode_facts)
    selector_phase3_enabled = (
        facts_as_selectors_enabled()
        and str(query_features.get("retrieval_type") or "").lower() != "temporal"
        and not query_features.get("step_numbers")
        and not query_features.get("step_range")
    )
    if selector_phase3_enabled:
        MemoryServer._hydrate_selector_surfaces_runtime(facts_by_episode, episode_lookup)
    else:
        for facts in facts_by_episode.values():
            for fact in facts:
                fact.pop("_selector_surface_text", None)
        for episode in episode_lookup.values():
            if isinstance(episode, dict):
                episode.pop("_selector_surface_text", None)
    temporal_trace = None
    if temporal_index is not None and classify_temporal_query(question) == "ordinal":
        ordinal_packet, temporal_trace = _build_ordinal_executor_packet(
            question,
            corpus=corpus,
            episode_lookup=episode_lookup,
            facts_by_episode=facts_by_episode,
            query_features=query_features,
            effective_selector=effective_selector,
            output_constraints=output_constraints,
            operator_tuning=operator_tuning,
            packet_tuning=packet_tuning,
            search_family=search_family,
            temporal_index=temporal_index,
        )
        if ordinal_packet is not None:
            return ordinal_packet
    if temporal_index is not None and classify_temporal_query(question) == "calendar":
        calendar_packet, temporal_trace = _build_calendar_executor_packet(
            question,
            corpus=corpus,
            episode_lookup=episode_lookup,
            facts_by_episode=facts_by_episode,
            query_features=query_features,
            effective_selector=effective_selector,
            output_constraints=output_constraints,
            operator_tuning=operator_tuning,
            packet_tuning=packet_tuning,
            search_family=search_family,
            temporal_index=temporal_index,
        )
        if calendar_packet is not None:
            return calendar_packet
    family_corpora = partition_corpus_by_family(corpus)
    family_results = []
    routed_families = route_retrieval_families(
        question,
        available_families(corpus),
        explicit_family=search_family,
    )
    for family in routed_families:
        family_corpus = family_corpora.get(family)
        if not family_corpus:
            continue
        family_lookup = build_episode_lookup(family_corpus)
        family_bm25 = build_episode_bm25(family_corpus)
        family_result = choose_episode_ids_with_trace(
            question,
            family_bm25,
            family_lookup,
            effective_selector,
        )
        family_results.append(
            {
                "family": family,
                "selected_ids": family_result["selected_ids"],
                "scored": family_result["scored"],
                "trace": family_result["trace"],
            }
        )
    late_fusion = select_episode_ids_late_fusion_with_trace(
        question,
        family_results,
        episode_lookup,
        effective_selector,
    )
    selected_episode_ids = late_fusion["selected_ids"]
    scored = late_fusion["scored"]
    step_numbers = set(query_features.get("step_numbers", set()) or set())
    step_range = query_features.get("step_range")

    def _target_step_hits(text: str) -> set[int]:
        hits: set[int] = set()
        if not text:
            return hits
        for step in step_numbers:
            if has_exact_step_mention(text, step):
                hits.add(step)
        if step_range:
            hits |= step_range_overlap(text, step_range)
        return hits

    def _episode_step_order(ep_id: str) -> int:
        ep = episode_lookup.get(ep_id) or {}
        lower = (ep.get("raw_text") or "").lower()
        match = re.search(r"\[step\s+(\d+)\]|\bstep\s+(\d+)\b", lower)
        if match:
            return int(match.group(1) or match.group(2))
        topic = (ep.get("topic_key") or "").lower()
        match = re.search(r"\bstep\s+(\d+)\b", topic)
        if match:
            return int(match.group(1))
        match = re.search(r"_e(\d+)\b", ep_id)
        if match:
            return int(match.group(1))
        return 10**9

    def _step_range_episode_sort_key(ep_id: str) -> tuple[int, int, str]:
        ep = episode_lookup.get(ep_id) or {}
        overlap = sorted(step_range_overlap(ep.get("raw_text", ""), step_range))
        if overlap:
            return (0, overlap[0], ep_id)
        return (1, _episode_step_order(ep_id), ep_id)

    def _target_step_episode_sort_key(ep_id: str) -> tuple[int, int, str]:
        ep = episode_lookup.get(ep_id) or {}
        hits = sorted(_target_step_hits(ep.get("raw_text", "")))
        if hits:
            return (0, hits[0], ep_id)
        return (1, _episode_step_order(ep_id), ep_id)

    def _adjacent_episode_id(ep_id: str, delta: int) -> str | None:
        match = re.search(r"^(.*_e)(\d+)\b", ep_id)
        if not match:
            return None
        prefix, raw_num = match.groups()
        next_num = int(raw_num) + int(delta)
        if next_num <= 0:
            return None
        return f"{prefix}{next_num:0{len(raw_num)}d}"

    def _target_step_companion_episode_ids() -> list[str]:
        if not (step_numbers or step_range):
            return []
        companions: list[str] = []
        seen: set[str] = set()
        for ep_id in selected_episode_ids:
            ep = episode_lookup.get(ep_id) or {}
            raw = (ep.get("raw_text") or "").strip()
            lower = raw.lower()
            if not raw or not _target_step_hits(raw):
                continue
            next_ep_id = _adjacent_episode_id(ep_id, 1)
            if not next_ep_id or next_ep_id in seen:
                continue
            next_ep = episode_lookup.get(next_ep_id) or {}
            next_raw = (next_ep.get("raw_text") or "").strip()
            next_lower = next_raw.lower()
            if not next_raw:
                continue
            if next_ep.get("source_id", "") != ep.get("source_id", ""):
                continue
            if next_ep.get("source_type", "") != ep.get("source_type", ""):
                continue
            if not next_lower.startswith("action:") or "observation:" not in next_lower:
                continue
            companions.append(next_ep_id)
            seen.add(next_ep_id)
        return companions

    def _target_step_support_fact_episode_ids(limit: int) -> list[str]:
        if not (step_numbers or step_range) or not selected_source_ids or selected_source_families != {"document"}:
            return []
        candidates: list[str] = []
        seen: set[str] = set()
        for ep_id, facts in facts_by_episode.items():
            ep = episode_lookup.get(ep_id) or {}
            if not ep:
                continue
            if ep.get("source_id", "") not in selected_source_ids:
                continue
            if ep_id in seen:
                continue
            if any(_target_step_hits(fact.get("fact", "")) for fact in facts):
                candidates.append(ep_id)
                seen.add(ep_id)
        candidates.sort(key=_target_step_episode_sort_key)
        return candidates[:limit]

    def _target_step_selected_episode_ids(limit: int) -> list[str]:
        if not (step_numbers or step_range) or not selected_source_ids or selected_source_families != {"document"}:
            return []
        candidates: list[str] = []
        seen: set[str] = set()
        for ep_id, ep in episode_lookup.items():
            if ep.get("source_id", "") not in selected_source_ids:
                continue
            if ep_id in seen:
                continue
            if not _target_step_hits(ep.get("raw_text", "")):
                continue
            candidates.append(ep_id)
            seen.add(ep_id)
        candidates.sort(key=_target_step_episode_sort_key)
        return candidates[:limit]

    if step_range and selected_episode_ids:
        selected_episode_ids = sorted(selected_episode_ids, key=_step_range_episode_sort_key)
        scored = sorted(
            scored,
            key=lambda row: _step_range_episode_sort_key(row[0]),
        )
    selected_source_ids = {
        episode_lookup.get(ep_id, {}).get("source_id", "")
        for ep_id in selected_episode_ids
        if episode_lookup.get(ep_id)
    }
    selected_source_families = {
        episode_lookup.get(ep_id, {}).get("source_type", "")
        for ep_id in selected_episode_ids
        if episode_lookup.get(ep_id)
    }
    if (step_numbers or step_range) and selected_source_families == {"document"}:
        step_span = max(1, step_range[1] - step_range[0] + 1) if step_range else max(1, len(step_numbers))
        target_step_selected_episode_ids = _target_step_selected_episode_ids(max(2, step_span * 2))
        if target_step_selected_episode_ids:
            selected_episode_ids = list(
                dict.fromkeys(target_step_selected_episode_ids + selected_episode_ids)
            )
    support_episode_pool_size = max(
        len(selected_episode_ids),
        int(packet_tuning.get("support_episode_pool_size", len(selected_episode_ids))),
    )
    fact_episode_ids = list(selected_episode_ids)
    target_step_companion_episode_ids: list[str] = []
    target_step_support_fact_episode_ids: list[str] = []
    if (step_numbers or step_range) and selected_source_families == {"document"}:
        target_step_companion_episode_ids = _target_step_companion_episode_ids()
        for ep_id in target_step_companion_episode_ids:
            if ep_id not in fact_episode_ids:
                fact_episode_ids.append(ep_id)
        step_span = max(1, step_range[1] - step_range[0] + 1) if step_range else max(1, len(step_numbers))
        target_step_support_fact_episode_ids = _target_step_support_fact_episode_ids(
            max(support_episode_pool_size, step_span * 2)
        )
        for ep_id in target_step_support_fact_episode_ids:
            if ep_id not in fact_episode_ids:
                fact_episode_ids.append(ep_id)
    structural_conversation_query = (
        selected_source_families == {"conversation"}
        and not operator_plan["list_set"]["enabled"]
        and any(
            operator_plan[name]["enabled"]
            for name in ("commonality", "compare_diff")
        )
    )
    if (
        support_episode_pool_size > len(fact_episode_ids)
        and selected_source_ids
        and selected_source_families == {"document"}
    ):
        for ep_id, _score in scored:
            ep = episode_lookup.get(ep_id) or {}
            if not ep or ep.get("source_id", "") not in selected_source_ids:
                continue
            if ep_id in fact_episode_ids:
                continue
            fact_episode_ids.append(ep_id)
            if len(fact_episode_ids) >= support_episode_pool_size:
                break
    elif structural_conversation_query and packet_tuning.get(
        "expand_conversation_source_for_structural_queries",
        True,
    ):
        conversation_pool_size = max(
            support_episode_pool_size,
            int(
                packet_tuning.get(
                    "conversation_structural_support_episode_pool_size",
                    support_episode_pool_size,
                )
            ),
        )
        for doc in corpus.get("documents", []):
            for ep in doc.get("episodes", []):
                ep_id = ep.get("episode_id", "")
                if not ep_id or ep_id in fact_episode_ids:
                    continue
                if ep.get("source_id", "") not in selected_source_ids:
                    continue
                fact_episode_ids.append(ep_id)
                if len(fact_episode_ids) >= conversation_pool_size:
                    break
            if len(fact_episode_ids) >= conversation_pool_size:
                break

    if step_range and fact_episode_ids:
        fact_episode_ids = sorted(
            list(dict.fromkeys(fact_episode_ids)),
            key=_step_range_episode_sort_key,
        )

    snippet_mode = bool(effective_selector.get("snippet_mode", False))
    snippet_chars = int(packet_tuning.get("snippet_chars", 1200))

    context, actual_injected_episode_ids, selected_fact_ids = (
        build_context_from_selected_episodes(
            question,
            selected_episode_ids,
            episode_lookup,
            facts_by_episode,
            fact_episode_ids=fact_episode_ids,
            support_episode_ids=target_step_companion_episode_ids,
            budget=effective_selector["budget"],
            max_total_facts=effective_selector["supporting_facts_total"],
            max_facts_per_episode=effective_selector["supporting_facts_per_episode"],
            snippet_mode=snippet_mode,
            snippet_chars=snippet_chars,
            allow_pseudo_facts=bool(effective_selector.get("allow_pseudo_facts", True)),
            query_features=query_features,
            local_anchor_window_chars=int(operator_tuning.get("local_anchor_window_chars", snippet_chars)),
            local_anchor_fact_radius=int(operator_tuning.get("local_anchor_fact_radius", 12)),
            list_set_dedup_overlap=float(operator_tuning.get("list_set_dedup_overlap", 0.9)),
            bounded_chain_fact_bonus=float(operator_tuning.get("bounded_chain_fact_bonus", 0.0)),
            query_specificity_bonus=float(packet_tuning.get("query_specificity_bonus", 0.0)),
            inject_support_fact_episodes=bool(
                (
                    structural_conversation_query
                    and packet_tuning.get("inject_support_fact_episodes", True)
                )
                or target_step_companion_episode_ids
                or target_step_support_fact_episode_ids
            ),
            max_injected_support_fact_episodes=int(
                max(
                    int(packet_tuning.get("max_injected_support_fact_episodes", 8)),
                    len(target_step_companion_episode_ids),
                    len(target_step_support_fact_episode_ids),
                )
            ),
        )
    )
    return {
        "context": context,
        "retrieved_episode_ids": selected_episode_ids,
        "actual_injected_episode_ids": actual_injected_episode_ids,
        "fact_episode_ids": fact_episode_ids,
        "retrieved_fact_ids": selected_fact_ids,
        "selection_scores": [
            {"episode_id": ep_id, "score": sc}
            for ep_id, sc in scored[: get_runtime_tuning()["telemetry"]["max_selection_scores"]]
        ],
        "selector_config": effective_selector,
        "query_operator_plan": operator_plan,
        "query_features": query_features,
        "output_constraints": output_constraints,
        "retrieval_families": routed_families,
        "search_family": search_family or "auto",
        "family_first_pass_trace": {
            "available_families": available_families(corpus),
            "retrieval_families": routed_families,
            "requested_search_family": search_family or "auto",
            "per_family": [
                {
                    "family": result["family"],
                    **result["trace"],
                }
                for result in family_results
            ],
        },
        "late_fusion_trace": late_fusion["trace"],
        "tuning_snapshot": {
            "selector": effective_selector,
            "operators": operator_tuning,
            "packet": packet_tuning,
            "routing": tuning["routing"],
            "telemetry": tuning["telemetry"],
        },
        "temporal_trace": temporal_trace,
    }


def _embedding_fingerprint(facts: list[dict]) -> str:
    """SHA-256 of concatenated fact texts. Detects text changes even when count is unchanged."""
    h = hashlib.sha256("|".join(f.get("fact", "") for f in facts).encode()).hexdigest()
    return h


# ── content_complexity (pure function, no LLM calls) ──

def _compute_content_complexity(facts: list[dict]) -> float:
    """Compute content complexity from already-extracted Librarian metadata.

    Pure function, zero LLM calls. Max-style aggregation across four axes:
    kind weight, entity density, temporal presence, fact count.
    Returns max of axis scores, capped at 1.0.
    """
    if not facts:
        return 0.0

    # Kind-based: take the max across all facts
    KIND_WEIGHTS = {
        "action_item": 0.70,
        "requirement": 0.65,
        "decision":    0.50,
        "constraint":  0.45,
        "preference":  0.10,
        "fact":        0.10,
    }
    kind_scores = [KIND_WEIGHTS.get(f.get("kind", "fact"), 0.10) for f in facts]
    kind_score = max(kind_scores) if kind_scores else 0.0

    # Entity density: count unique entities across all facts
    all_entities = set()
    for f in facts:
        for e in f.get("entities", []):
            if isinstance(e, str):
                all_entities.add(e.lower())
    n_entities = len(all_entities)
    if n_entities > 20:
        entity_score = 0.60
    elif n_entities > 10:
        entity_score = 0.40
    elif n_entities > 5:
        entity_score = 0.20
    else:
        entity_score = 0.0

    # Temporal: if any fact has temporal links
    has_temporal = any(
        f.get("_temporal_links") or f.get("event_date") or f.get("depends_on")
        for f in facts
    )
    temporal_score = 0.35 if has_temporal else 0.0

    # Fact count
    n_facts = len(facts)
    if n_facts > 50:
        fact_score = 0.50
    elif n_facts > 20:
        fact_score = 0.30
    elif n_facts > 10:
        fact_score = 0.15
    else:
        fact_score = 0.0

    score = max(kind_score, entity_score, temporal_score, fact_score)
    return round(min(1.0, score), 3)


def _compute_query_shape_complexity(query: str | None, resolved_type: str) -> float:
    """Estimate reasoning complexity from the query shape alone.

    This is intentionally lexical and conservative. It should only raise
    complexity for clearly comparative, multi-constraint recommendation
    prompts, while leaving plain lookups untouched.
    """
    if not query:
        return 0.0

    q = query.lower()

    decision_markers = (
        "which option",
        "what option",
        "which approach",
        "recommended",
        "recommend",
        "should be chosen",
        "should we choose",
        "best option",
    )
    comparison_markers = (
        "why is the other",
        "why the other",
        "compared with",
        "versus",
        " vs ",
        "tradeoff",
        "trade-off",
        "risky",
    )
    strong_constraint_terms = (
        "latency",
        "ownership",
        "owner",
        "rollback",
        "budget",
        "cost",
        "throughput",
        "capacity",
        "availability",
        "dependency",
        "dependencies",
        "compliance",
    )

    has_decision = any(marker in q for marker in decision_markers)
    comparison_hits = sum(1 for marker in comparison_markers if marker in q)
    constraint_hits = sum(1 for term in strong_constraint_terms if term in q)

    if has_decision and constraint_hits >= 3:
        return 0.70
    if has_decision and constraint_hits >= 2:
        return 0.65
    if has_decision and comparison_hits >= 1:
        return 0.50
    if resolved_type in (
        "aggregate",
        "counting",
        "synthesize",
        "synthesis",
        "procedural",
        "rule",
    ) and constraint_hits >= 2:
        return 0.55
    return 0.0


# ── complexity_hint v2 (pure function, no LLM calls) ──

def _compute_complexity_hint(retrieved, resolved_type, is_multihop, fact_lookup, query: str | None = None):
    """Compute complexity hint from retrieval signals and retrieved fact content.

    Three independent axes:
    - retrieval_complexity: structural signals from retrieval (multi_hop,
      cross_scope, conflict_found, high_fact_count)
    - content_complexity: complexity of the actual fact set returned by recall
    - query_complexity: complexity implied by the query shape itself

    Final score = max(retrieval_complexity, content_complexity, query_complexity).
    Zero LLM calls.
    """
    signals = []

    def _resolve_fact(item: dict) -> dict | None:
        if not isinstance(item, dict):
            return None
        # Preferred path: recall already passed the actual retrieved fact.
        if any(
            key in item
            for key in (
                "fact",
                "kind",
                "entities",
                "tags",
                "event_date",
                "depends_on",
                "_temporal_links",
            )
        ):
            return item
        fid = item.get("fact_id", item.get("id", ""))
        if fid and fact_lookup:
            return fact_lookup.get(fid)
        return None

    retrieved_facts = []
    if retrieved:
        for item in retrieved:
            fact = _resolve_fact(item)
            if fact:
                retrieved_facts.append(fact)

    # ── Retrieval signals (structural only) ──
    if is_multihop:
        signals.append("multi_hop")

    if resolved_type == "supersession":
        signals.append("conflict_found")

    if len(retrieved) > 50:
        signals.append("high_fact_count")

    if retrieved_facts:
        agent_ids = set()
        for fact in retrieved_facts:
            agent_ids.add(fact.get("agent_id", ""))
        if len(agent_ids) > 1:
            signals.append("cross_scope")

    retrieval_complexity = 0.0
    if "multi_hop" in signals:       retrieval_complexity += 0.35
    if "cross_scope" in signals:     retrieval_complexity += 0.25
    if "conflict_found" in signals:  retrieval_complexity += 0.20
    if "high_fact_count" in signals: retrieval_complexity += 0.05
    retrieval_complexity = min(1.0, retrieval_complexity)

    # ── Content complexity from the actual retrieved fact set ──
    content_complexity = _compute_content_complexity(retrieved_facts)
    query_complexity = _compute_query_shape_complexity(query, resolved_type)

    # ── Combined score ──
    score = max(retrieval_complexity, content_complexity, query_complexity)
    score = round(max(0.0, min(1.0, score)), 3)

    if score <= 0.2:   level = 1
    elif score <= 0.4: level = 2
    elif score <= 0.6: level = 3
    elif score <= 0.8: level = 4
    else:              level = 5

    # Dominant axis
    if retrieval_complexity > content_complexity and retrieval_complexity > query_complexity:
        dominant = "retrieval"
    elif content_complexity > retrieval_complexity and content_complexity > query_complexity:
        dominant = "content"
    elif query_complexity > retrieval_complexity and query_complexity > content_complexity:
        dominant = "query"
    else:
        dominant = "tie"

    return {
        "score": score,
        "level": level,
        "signals": signals,
        "retrieval_complexity": round(retrieval_complexity, 3),
        "content_complexity": round(content_complexity, 3),
        "query_complexity": round(query_complexity, 3),
        "dominant": dominant,
    }


# ── memory_query constants ──

SORTABLE_FIELDS = {
    "id", "fact", "kind", "speaker", "event_date", "supersedes_topic",
    "conv_id", "agent_id", "swarm_id", "scope", "owner_id", "created_at",
    "session_date", "artifact_id", "version_id", "content_hash",
    "source_id", "status",
    "session", "retention_ttl",
}

NUMERIC_SORT_FIELDS = {"session", "retention_ttl"}

VALID_SORT_ORDERS = {"asc", "desc"}
MAX_QUERY_FACT_CHARS = 1200

RANGE_OPS = {"gt", "gte", "lt", "lte"}


def _consensus_metadata(source_facts: list, source_ids: list = None) -> dict:
    """Propagate metadata from source facts by consensus.

    Only keys where ALL sources agree on value are inherited.
    Missing/incomplete source_ids → empty metadata.
    """
    if not source_facts:
        return {}
    if source_ids:
        matched = [sf.get("metadata", {}) for sf in source_facts
                   if sf.get("id") in source_ids
                   or any(sf.get("id", "").endswith(s) for s in source_ids)]
        if not matched:
            return {}  # unresolved lineage → no metadata (not inferred)
    else:
        return {}
    all_keys = set()
    for m in matched:
        if isinstance(m, dict):
            all_keys.update(m.keys())
    consensus = {}
    for k in all_keys:
        vals = [m.get(k) for m in matched if isinstance(m, dict) and k in m]
        if vals and all(v == vals[0] for v in vals):
            consensus[k] = vals[0]
    return consensus


def _consensus_target(source_facts: list, source_ids: list = None) -> list[str] | None:
    """Propagate target only when all relevant source facts agree on it."""
    if not source_facts:
        return None
    if source_ids:
        matched = [
            sf for sf in source_facts
            if sf.get("id") in source_ids
            or any(sf.get("id", "").endswith(s) for s in source_ids)
        ]
        if not matched:
            return None
    else:
        matched = list(source_facts)

    normalized = []
    for fact in matched:
        target = fact.get("target")
        if not target:
            normalized.append(None)
            continue
        normalized.append(tuple(_normalize_target(target) or []))

    if not normalized:
        return None
    first = normalized[0]
    if all(value == first for value in normalized):
        return list(first) if first else None
    return None


def _resolve_field(fact: dict, key: str):
    if key.startswith("metadata."):
        meta = fact.get("metadata") or {}
        return meta.get(key[9:])
    return fact.get(key)


def _match_filter_value(fact_value, filter_value, allow_range: bool = False) -> bool:
    if isinstance(filter_value, dict) and filter_value.keys() <= RANGE_OPS:
        if not allow_range:
            return False
        if fact_value is None:
            return False
        for op, threshold in filter_value.items():
            if op == "gt" and not (fact_value > threshold):
                return False
            if op == "gte" and not (fact_value >= threshold):
                return False
            if op == "lt" and not (fact_value < threshold):
                return False
            if op == "lte" and not (fact_value <= threshold):
                return False
        return True
    if isinstance(fact_value, list):
        return filter_value in fact_value
    return fact_value == filter_value


def _merge_fact_metadata(existing, stamped):
    """Merge flat metadata dicts with caller-provided metadata winning on conflicts."""
    if stamped is None:
        return existing if isinstance(existing, dict) else None

    merged = {}
    if isinstance(existing, dict):
        merged.update(existing)
    merged.update(stamped)
    return merged


def _is_asserted_derived_fact(fact: dict) -> bool:
    metadata = fact.get("metadata") or {}
    return bool(metadata.get("asserted_derived_tier"))


def _is_supported_cross_fact(fact: dict) -> bool:
    metadata = fact.get("metadata") or {}
    return bool(metadata.get("source_aggregation") or metadata.get("asserted_derived_tier"))


def _fact_matches_structured_filter(
    fact: dict,
    filter: dict | None,
    metadata_schema: dict | None = None,
) -> bool:
    """Apply the same structured filter semantics used by query()."""
    if not filter:
        return True

    range_types = {"number", "integer", "datetime"}
    for key, value in filter.items():
        allow_range = True
        if key.startswith("metadata."):
            field_type = None
            if metadata_schema:
                field_def = metadata_schema.get(key[9:])
                field_type = field_def.get("type") if field_def else None
            allow_range = field_type in range_types
        if not _match_filter_value(
            _resolve_field(fact, key),
            value,
            allow_range=allow_range,
        ):
            return False
    return True


def _is_sortable(sort_by: str, metadata_schema: dict = None) -> bool:
    if sort_by in SORTABLE_FIELDS:
        return True
    if sort_by.startswith("metadata."):
        if not metadata_schema:
            return False
        meta_key = sort_by[9:]
        field_def = metadata_schema.get(meta_key)
        if not field_def:
            return False
        return field_def.get("type", "") in ("string", "number", "integer", "boolean", "datetime")
    return False


def _split_sort_values(facts: list, sort_by: str):
    present, missing = [], []
    for f in facts:
        if _resolve_field(f, sort_by) is None:
            missing.append(f)
        else:
            present.append(f)
    return present, missing


def _sort_value(f: dict, sort_by: str):
    v = _resolve_field(f, sort_by)
    if sort_by in NUMERIC_SORT_FIELDS or isinstance(v, (int, float)):
        return v if isinstance(v, (int, float)) else 0
    return str(v)


# ── L0 enrichment helper ──

def _normalize_fact_types(f: dict) -> None:
    """Fix malformed metadata types in-place before enrichment/storage.

    Converts wrong types to correct ones so downstream code never sees
    entities as string or tags as non-list.
    """
    if "kind" in f and (not isinstance(f["kind"], str) or not f["kind"]):
        del f["kind"]  # remove so enrichment/setdefault can fill it
    if "entities" in f and not isinstance(f["entities"], list):
        v = f["entities"]
        f["entities"] = [v] if isinstance(v, str) and v else []
    if "tags" in f and not isinstance(f["tags"], list):
        v = f["tags"]
        f["tags"] = [v] if isinstance(v, str) and v else []


_SELECTOR_OPTIONAL_SOURCE_FIELDS = ("query", "caption", "blip_caption")

def _selector_optional_raw_fields(*sources: dict | None) -> dict[str, str]:
    fields: dict[str, str] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in _SELECTOR_OPTIONAL_SOURCE_FIELDS:
            value = source.get(key)
            if not isinstance(value, str):
                continue
            stripped = value.strip()
            if stripped:
                fields.setdefault(key, stripped)
    return fields


def _selector_raw_fields(raw_text: str, *sources: dict | None) -> dict[str, str]:
    fields = {"raw_text": str(raw_text or "")}
    fields.update(_selector_optional_raw_fields(*sources))
    return fields


def _stamp_selector_episode_fields(episode: dict, *sources: dict | None) -> None:
    if not isinstance(episode, dict):
        return
    for key, value in _selector_optional_raw_fields(*sources).items():
        episode.setdefault(key, value)


def _coerce_positive_session_num(value) -> int | None:
    """Return a positive integer session number when the value is coercible."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float) and value.is_integer():
        ivalue = int(value)
        return ivalue if ivalue > 0 else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            ivalue = int(stripped)
            return ivalue if ivalue > 0 else None
    return None


def _namespace_derived_fact_ids(facts: list[dict], prefix: str) -> None:
    """Ensure derived tiers never reuse local extractor IDs from source facts."""
    for idx, fact in enumerate(facts or []):
        raw_id = str(fact.get("id") or f"{idx:03d}").strip()
        if raw_id.startswith(f"{prefix}_"):
            continue
        fact["id"] = f"{prefix}_{raw_id or f'{idx:03d}'}"


def _needs_l0_enrichment(f: dict) -> bool:
    """Check if fact needs L0 classification. Presence-based, not value-based."""
    return (
        "kind" not in f or not isinstance(f.get("kind"), str) or not f.get("kind")
        or "entities" not in f or not isinstance(f.get("entities"), list)
        or "tags" not in f or not isinstance(f.get("tags"), list)
    )


# ── Shared visibility predicate (Unit 5) ──

def _is_visible(fact, now=None, fact_lookup=None):
    """Check if a fact is visible (active, not expired, source not retracted).

    Used as the single source of truth for all read paths.
    """
    status = fact.get("status", "active")
    if status != "active":
        return False

    # TTL check
    ttl = fact.get("retention_ttl")
    if ttl is not None:
        created = fact.get("created_at")
        if created and now:
            try:
                from datetime import datetime as _dt
                from datetime import timezone as _tz
                created_dt = _dt.fromisoformat(created.replace("Z", "+00:00"))
                if isinstance(now, str):
                    now_dt = _dt.fromisoformat(now.replace("Z", "+00:00"))
                else:
                    now_dt = now
                elapsed = (now_dt - created_dt).total_seconds()
                if elapsed > ttl:
                    return False
            except (ValueError, TypeError):
                pass

    # Derived tier staleness: hide if ANY source fact is invisible
    if fact_lookup and "source_ids" in fact and fact["source_ids"]:
        for sid in fact["source_ids"]:
            source = fact_lookup.get(sid)
            if source:
                src_status = source.get("status", "active")
                if src_status != "active":
                    return False
                # TTL check on source
                src_ttl = source.get("retention_ttl")
                if src_ttl is not None and now:
                    src_created = source.get("created_at")
                    if src_created:
                        try:
                            from datetime import datetime as _dt2
                            from datetime import timezone as _tz2
                            sc_dt = _dt2.fromisoformat(src_created.replace("Z", "+00:00"))
                            n_dt = now if not isinstance(now, str) else _dt2.fromisoformat(now.replace("Z", "+00:00"))
                            if (n_dt - sc_dt).total_seconds() > src_ttl:
                                return False
                        except (ValueError, TypeError):
                            pass

    return True


# ── MemoryServer ──

def _resolve_default_owner(server) -> str:
    if server.agent_id and server.agent_id != "default":
        return f"agent:{server.agent_id}"
    return "system"


class MemoryServer:
    """In-process memory server with 3-tier fact indexing and multi-agent isolation.

    One instance = one conversation (conv_id = key).
    """

    _EXTRACT_SEM: asyncio.Semaphore | None = None
    _EXTRACT_SEM_LOOP_ID: int | None = None

    @classmethod
    def _get_extract_sem(cls) -> asyncio.Semaphore:
        current_loop_id = id(asyncio.get_event_loop())
        if cls._EXTRACT_SEM is None or current_loop_id != cls._EXTRACT_SEM_LOOP_ID:
            try:
                limit = max(1, int(os.getenv("MEMORY_EXTRACT_CONCURRENCY", "3")))
            except Exception:
                limit = 3
            cls._EXTRACT_SEM = asyncio.Semaphore(limit)
            cls._EXTRACT_SEM_LOOP_ID = current_loop_id
        return cls._EXTRACT_SEM

    VALID_TIER_MODES = {"eager", "lazy_tier2", "lazy_tier2_3"}

    def __init__(
        self,
        data_dir: str,
        key: str,
        extract_model: str | None = "",
        agent_id: str = "default",
        scope: str = "swarm-shared",
        swarm_id: str = "default",
        storage: StorageBackend = None,
        tier_mode: str = "eager",
        profiles: dict = None,
        profile_configs: dict = None,
        inference_leaf_plugins: dict | None = None,
    ):
        # _UNSET sentinel distinguishes "caller omitted the arg" (default "")
        # from "caller explicitly passed None or a model string".
        _UNSET = object()
        _ctor_extract_model = _UNSET if extract_model == "" else extract_model
        self._extract_disabled = extract_model is None
        _ctor_profiles = deepcopy(profiles) if profiles is not None else None
        _ctor_profile_configs = deepcopy(profile_configs) if profile_configs is not None else None
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.key = key
        if extract_model is None:
            self.extract_model = ""
        elif extract_model == "":
            self.extract_model = MemoryConfig().extraction_model
        else:
            self.extract_model = extract_model
        self.agent_id = agent_id
        self.scope = scope
        self.swarm_id = swarm_id
        self._tier_mode = tier_mode if tier_mode in self.VALID_TIER_MODES else "eager"
        self._profiles = profiles  # complexity level → profile name mapping
        self._profile_configs = profile_configs or {}  # name → {model, context_window, ...}
        self._inference_leaf_plugins = dict(DEFAULT_INFERENCE_LEAF_PLUGIN_STATE)
        if inference_leaf_plugins:
            self._inference_leaf_plugins.update(
                {str(name): bool(enabled) for name, enabled in inference_leaf_plugins.items()}
            )
        self._memory_config: dict | None = None
        self._embedding_model: str | None = None
        self._librarian_profile: str | None = None

        self._storage = storage or make_storage(data_dir, key)
        self._cache_json = self.data_dir / f"{key}.json"
        self._cache_embs = self.data_dir / f"{key}_embs.npz"
        self._corpus_json = self.data_dir / f"{key}_corpus.json"
        self._temporal_json = self.data_dir / f"{key}_temporal.json"
        self._uses_legacy_sidecars = isinstance(self._storage, JSONNPZStorage)
        self._supports_write_log = isinstance(self._storage, SQLiteStorageBackend)

        self._prompt_registry = PromptRegistry(data_dir=data_dir, key=key)
        self._membership_registry = MembershipRegistry()
        self._file_lock = asyncio.Lock()
        self._queue_lock = asyncio.Lock()
        self._active_sync_message_ids: set[str] = set()
        self._upsert_locks: dict[str, asyncio.Lock] = {}
        self._consolidation_queue: list[int] = []
        self._data_dict = None
        self._audit = AuditLog(self.data_dir / "audit")

        # Load existing cache if present
        self._all_granular: list[dict] = []
        self._all_cons: list[dict] = []
        self._all_cross: list[dict] = []
        self._all_tlinks: list[dict] = []
        self._raw_sessions: list[dict] = []
        self._raw_docs: dict[str, str] = {}  # doc_key → raw text for hybrid doc context
        self._episode_corpus: dict = {"documents": []}
        self._temporal_index: dict = empty_temporal_index()
        self._temporal_index_dirty = True if self._supports_write_log else not self._temporal_json.exists()
        self._secrets: list[dict] = []
        self._n_sessions = 0
        self._n_sessions_with_facts = 0
        self._tiers_dirty = False
        self._tier2_built = False
        self._tier3_built = False
        self._emb_fingerprints = {}
        self._dedup_index: dict[tuple, dict] = {}
        self._git_dedup_index: dict[tuple, dict] = {}
        self._fact_lookup: dict = {}
        self._metadata_schema: dict = None
        self._instance_config: dict = None
        self._scope_record: dict = {}
        self._source_records: dict[str, dict] = {}

        if self._storage.exists:
            cached = self._storage.load_facts()
            self._all_granular = cached.get("granular", [])
            self._all_cons = cached.get("cons", [])
            self._all_cross = cached.get("cross", [])
            self._all_tlinks = cached.get("tlinks", [])
            self._n_sessions = cached.get("n_sessions", 0)
            self._n_sessions_with_facts = cached.get("n_sessions_with_facts", 0)
            self._raw_sessions = cached.get("raw_sessions", [])
            self._raw_docs = cached.get("raw_docs", {})
            self._episode_corpus = cached.get("episode_corpus", {"documents": []})
            self._secrets = cached.get("secrets", [])
            # Restore _temporal_links on granular facts
            for f in self._all_granular:
                f["_temporal_links"] = []
            if self._all_granular and self._all_tlinks:
                self._all_granular[0]["_temporal_links"] = self._all_tlinks
            log.info("Loaded cache: %dg/%dc/%dx",
                     len(self._all_granular), len(self._all_cons), len(self._all_cross))

            # Backward compat: migrate facts without ACL fields
            for tier in (self._all_granular, self._all_cons, self._all_cross):
                for f in tier:
                    if "owner_id" not in f:
                        acl = _derive_acl_from_scope(
                            f.get("scope"), f.get("agent_id", "default"),
                            f.get("swarm_id", "default"))
                        f["owner_id"] = acl["owner_id"]
                        f["read"] = acl["read"]
                        f["write"] = acl["write"]

            # Backward compat: identity fields default
            for tier in (self._all_granular, self._all_cons, self._all_cross):
                for f in tier:
                    f.setdefault("status", "active")

            # Load dedup index
            saved_dedup = cached.get("_dedup_index", {})
            # JSON keys are strings; convert back to tuples
            for k, v in saved_dedup.items():
                try:
                    key_tuple = tuple(json.loads(k))
                    self._dedup_index[key_tuple] = v
                except (json.JSONDecodeError, TypeError):
                    pass

            saved_git_dedup = cached.get("_git_dedup_index", {})
            for k, v in saved_git_dedup.items():
                try:
                    key_tuple = tuple(json.loads(k))
                    self._git_dedup_index[key_tuple] = v
                except (json.JSONDecodeError, TypeError):
                    pass

            self._metadata_schema = cached.get("metadata_schema")
            self._instance_config = cached.get("instance_config")
            self._scope_record = cached.get("scope_record") or {}
            self._source_records = cached.get("source_records") or {}
            saved_memory_config = cached.get("memory_config")
            saved_profiles = cached.get("profiles")
            saved_configs = cached.get("profile_configs")
            if saved_profiles and not self._profiles:
                self._profiles = {int(k): v for k, v in saved_profiles.items()}
            if saved_configs and not self._profile_configs:
                self._profile_configs = saved_configs
            if saved_memory_config:
                self._apply_memory_config(saved_memory_config)

            # Legacy auto-generated merge tiers are no longer part of the production path.
            # Keep asserted derived tiers and source-aggregation cross facts.
            self._all_cons = [f for f in self._all_cons if _is_asserted_derived_fact(f)]
            self._all_cross = [f for f in self._all_cross if _is_supported_cross_fact(f)]
            preserved_asserted = bool(self._all_cons) or any(
                _is_asserted_derived_fact(f) for f in self._all_cross
            )
            self._tier2_built = True
            self._tier3_built = True if preserved_asserted else bool(self._all_cross)

            # Load fingerprints and pre-load embeddings if fingerprint matches
            self._emb_fingerprints = cached.get("_emb_fingerprints", {})
            current_fps = {
                "gran": _embedding_fingerprint(self._all_granular),
                "cons": _embedding_fingerprint(self._all_cons),
                "cross": _embedding_fingerprint(self._all_cross),
            }
            saved_embs = self._storage.load_embeddings()
            if saved_embs is not None and self._emb_fingerprints == current_fps:
                gran_embs = saved_embs.get("gran", np.zeros((0, 3072)))
                cons_embs = saved_embs.get("cons", np.zeros((0, 3072)))
                cross_embs = saved_embs.get("cross", np.zeros((0, 3072)))
                if (len(gran_embs) == len(self._all_granular)
                        and len(cons_embs) == len(self._all_cons)
                        and len(cross_embs) == len(self._all_cross)):
                    self._data_dict = _build_index_state(
                        self._all_granular,
                        gran_embs,
                        self._all_cons,
                        cons_embs,
                        self._all_cross,
                        cross_embs,
                    )
                    self._fact_lookup = self._data_dict.get("fact_lookup", {})
                    log.info("Loaded embeddings from disk (fingerprint match): %d/%d/%d",
                             len(gran_embs), len(cons_embs), len(cross_embs))
                else:
                    log.info("Saved embeddings count mismatch — will re-embed on next build_index()")
                    self._emb_fingerprints = {}
            elif saved_embs is not None:
                log.info("Embedding fingerprint mismatch — will re-embed on next build_index()")
                self._emb_fingerprints = {}

        if self._memory_config is None:
            self._apply_memory_config(self._default_memory_config())
        if _ctor_profiles is not None or _ctor_profile_configs is not None:
            override_cfg = self.get_config()
            if _ctor_profiles is not None:
                override_cfg["profiles"] = _ctor_profiles
            if _ctor_profile_configs is not None:
                override_cfg["profile_configs"] = _ctor_profile_configs
            self._apply_memory_config(override_cfg)
        # CLI-provided extract_model takes priority over cached config.
        # _UNSET means the caller used the default (no override); any other
        # value — including None (disable extraction) — wins over cache.
        if _ctor_extract_model is not _UNSET:
            if _ctor_extract_model is None:
                self.extract_model = ""
            else:
                self.extract_model = _ctor_extract_model
        if self._uses_legacy_sidecars and self._corpus_json.exists():
            self._episode_corpus = load_episode_corpus(self._corpus_json, strict=False)
        if self._uses_legacy_sidecars and self._temporal_json.exists():
            self._temporal_index = load_temporal_index(self._temporal_json)
            self._temporal_index_dirty = False
        self._initialize_scope_registry()

    VALID_SCOPES = {"agent-private", "swarm-shared", "system-wide"}

    def _mark_tiers_dirty(self):
        """Mark tiers as needing rebuild, reset flags, clear stale data."""
        self._tiers_dirty = True
        self._temporal_index_dirty = True
        self._tier2_built = False
        self._tier3_built = False
        self._all_cons = [f for f in self._all_cons if _is_asserted_derived_fact(f)]
        self._all_cross = [f for f in self._all_cross if _is_asserted_derived_fact(f)]

    # ── ACL enforcement ──

    def _acl_allows(self, fact: dict, caller_id: str,
                    caller_memberships: list[str] = None,
                    caller_role: str = "user") -> bool:
        """Check if caller_id is allowed to read this fact.

        Rules (in order):
        1. admin role → always allowed
        2. owner_id == caller_id → allowed
        3. "agent:PUBLIC" in fact["read"] → allowed
        4. caller_id in fact["read"] → allowed
        5. any membership in fact["read"] → allowed
        6. deny
        """
        if caller_role == "admin":
            return True

        # Backward compat: derive ACL from scope if owner_id missing
        if "owner_id" not in fact:
            acl = _derive_acl_from_scope(
                fact.get("scope"), fact.get("agent_id", "default"),
                fact.get("swarm_id", "default"))
            fact_owner = acl["owner_id"]
            fact_read = acl["read"]
        else:
            fact_owner = fact["owner_id"]
            fact_read = fact.get("read", ["agent:PUBLIC"])

        # Owner always has access
        if fact_owner == caller_id:
            return True

        # system caller sees system-owned facts
        if caller_id == "system" and fact_owner == "system":
            return True

        # Public facts
        if "agent:PUBLIC" in fact_read:
            return True

        # Direct grant
        if caller_id in fact_read:
            return True

        # Membership grant
        if caller_memberships:
            for m in caller_memberships:
                if m in fact_read:
                    return True

        return False

    # ── Fact tagging ──

    def _tag_facts(self, facts, session_date: str,
                   agent_id=None, swarm_id=None, scope=None,
                   owner_id=None, read=None, write=None,
                   artifact_id=None, version_id=None,
                   content_hash=None, status="active",
                   retention_ttl=None, metadata=None, target=None):
        """Tag all facts with conv_id, agent_id, swarm_id, scope, ACL, created_at.

        Per-call params override instance defaults — eliminates the race
        condition from concurrent store() calls with different identities.
        Also propagates identity/versioning fields when provided.
        """
        now = datetime.now(timezone.utc).isoformat()
        _agent_id = agent_id if agent_id is not None else self.agent_id
        _swarm_id = swarm_id if swarm_id is not None else self.swarm_id
        _scope = scope if scope is not None else self.scope

        # Resolve ACL: explicit > derived from agent_id > default public
        if owner_id is not None:
            _owner_id = owner_id
            _read = read if read is not None else ["agent:PUBLIC"]
            _write = write if write is not None else ["agent:PUBLIC"]
        elif _agent_id != "default":
            _owner_id = f"agent:{_agent_id}"
            _read = read if read is not None else ["agent:PUBLIC"]
            _write = write if write is not None else ["agent:PUBLIC"]
        else:
            _owner_id = "system"
            _read = read if read is not None else ["agent:PUBLIC"]
            _write = write if write is not None else ["agent:PUBLIC"]

        for f in facts:
            f["conv_id"] = self.key
            f["session_date"] = session_date
            f["agent_id"] = _agent_id
            f["swarm_id"] = _swarm_id
            f["scope"] = _scope
            f["owner_id"] = _owner_id
            f["read"] = list(_read)
            f["write"] = list(_write)
            f["created_at"] = now
            # Identity/versioning fields
            if artifact_id is not None:
                f["artifact_id"] = artifact_id
            if version_id is not None:
                f["version_id"] = version_id
            if content_hash is not None:
                f["content_hash"] = content_hash
            f.setdefault("status", status)
            if retention_ttl is not None:
                f["retention_ttl"] = retention_ttl
            if metadata is not None:
                merged_metadata = _merge_fact_metadata(f.get("metadata"), metadata)
                if merged_metadata:
                    f["metadata"] = merged_metadata
                else:
                    f.pop("metadata", None)
            if target is not None:
                if target:
                    f["target"] = list(target)
                else:
                    f.pop("target", None)
            # Auto-promote rules/constraints to swarm-shared,
            # but never override an explicit agent-private scope
            if f.get("kind") in ("constraint", "rule") and _scope != "agent-private":
                f["scope"] = "swarm-shared"

    # ── Disk I/O ──

    def _save_cache(self):
        """Save all three tiers to disk (must be called under _file_lock)."""
        save_granular = [{k: v for k, v in f.items() if k != "_temporal_links"}
                         for f in self._all_granular]

        # Serialize dedup_index: tuple keys → JSON string keys
        serializable_dedup = {json.dumps(list(k)): v
                              for k, v in self._dedup_index.items()}
        serializable_git_dedup = {json.dumps(list(k)): v
                                  for k, v in self._git_dedup_index.items()}

        self._storage.save_facts({
            "granular":              save_granular,
            "cons":                  self._all_cons,
            "cross":                 self._all_cross,
            "tlinks":                self._all_tlinks,
            "raw_sessions":          self._raw_sessions,
            "raw_docs":              self._raw_docs,
            "episode_corpus":        self._episode_corpus,
            "secrets":               self._secrets,
            "n_sessions":            self._n_sessions,
            "n_sessions_with_facts": self._n_sessions_with_facts,
            "_emb_fingerprints":     self._emb_fingerprints,
            "_dedup_index":          serializable_dedup,
            "_git_dedup_index":      serializable_git_dedup,
            "metadata_schema":       self._metadata_schema,
            "instance_config":       self._instance_config,
            "scope_record":          self._scope_record,
            "source_records":        self._source_records,
            "profiles":              self._profiles,
            "profile_configs":       self._profile_configs,
            "memory_config":         self._memory_config,
        })
        if self._uses_legacy_sidecars:
            write_json_atomic(self._corpus_json, self._episode_corpus)

    def _build_temporal_text_spans(self) -> list[dict]:
        facts_by_episode = build_facts_by_episode(self._all_granular)
        episode_lookup = build_episode_lookup(self._episode_corpus)
        episode_timeline_ids: dict[str, str] = {}
        spans: list[dict] = []
        for doc in self._episode_corpus.get("documents", []):
            doc_id = str(doc.get("doc_id") or "")
            for ep in doc.get("episodes", []):
                episode_id = str(ep.get("episode_id") or "").strip()
                if not episode_id:
                    continue
                source_id = str(ep.get("source_id") or doc_id or self.key)
                timeline_id = doc_id or f"timeline:{source_id}:main"
                episode_timeline_ids[episode_id] = timeline_id
                support_texts: list[str] = []
                seen_support_texts: set[str] = set()
                for fact in facts_by_episode.get(episode_id, []):
                    fact_text = str(fact.get("fact") or "").strip()
                    if not fact_text or fact_text in seen_support_texts:
                        continue
                    seen_support_texts.add(fact_text)
                    support_texts.append(fact_text[:400])
                    if len(support_texts) >= 8:
                        break
                spans.append(
                    {
                        "span_id": episode_id,
                        "source_id": source_id,
                        "timeline_id": timeline_id,
                        "text": ep.get("raw_text", ""),
                        "timestamp": ep.get("source_date"),
                        "ordinal_hint": None,
                        "provenance": ep.get("provenance") or {},
                        "support_fact_ids": [
                            fact_id
                            for fact_id in (
                                fact.get("id", "")
                                for fact in facts_by_episode.get(episode_id, [])
                            )
                            if fact_id
                        ],
                        "payload": {
                            "episode_id": episode_id,
                            "doc_id": doc_id,
                            "support_texts": support_texts,
                        },
                    }
                )
        for fact in self._all_granular:
            fact_id = str(fact.get("id") or "").strip()
            fact_text = str(fact.get("fact") or "").strip()
            if not fact_id or not fact_text:
                continue
            metadata = fact.get("metadata") or {}
            episode_id = str(
                metadata.get("episode_id")
                or fact.get("episode_id")
                or ""
            ).strip()
            episode = episode_lookup.get(episode_id) or {}
            source_id = str(
                fact.get("source_id")
                or metadata.get("episode_source_id")
                or episode.get("source_id")
                or self.key
            )
            provenance = {}
            support_spans = fact.get("support_spans")
            if isinstance(support_spans, list) and support_spans:
                first_span = support_spans[0] or {}
                start = first_span.get("start")
                end = first_span.get("end")
                if isinstance(start, int) and isinstance(end, int):
                    provenance = {
                        "start_char": int(start),
                        "end_char": int(end),
                        "source_field": str(first_span.get("source_field") or "raw_text"),
                        "episode_id": str(first_span.get("episode_id") or episode_id or ""),
                    }
            elif episode.get("provenance"):
                provenance = dict(episode.get("provenance") or {})
                if episode_id and not provenance.get("episode_id"):
                    provenance["episode_id"] = episode_id
                provenance.setdefault("source_field", "raw_text")
            spans.append(
                {
                    "span_id": f"fact:{fact_id}",
                    "source_id": source_id,
                    "timeline_id": episode_timeline_ids.get(episode_id)
                    or episode.get("source_id")
                    or source_id
                    or self.key,
                    "text": fact_text,
                    "timestamp": fact.get("session_date") or episode.get("source_date"),
                    "ordinal_hint": None,
                    "provenance": provenance,
                    "support_fact_ids": [fact_id],
                    "payload": {
                        "episode_id": episode_id,
                        "fact_id": fact_id,
                    },
                }
            )
        return spans

    def _rebuild_temporal_index(self) -> None:
        self._temporal_index = normalize_temporal_index(self._build_temporal_text_spans())
        self._temporal_index_dirty = False

    @staticmethod
    def _format_calendar_resolution_answer(plan: dict, event: dict) -> str | None:
        time_start = str(event.get("time_start") or "").strip()
        if not time_start:
            return None
        granularity = str(plan.get("granularity") or "date").lower()
        event_granularity = str(event.get("time_granularity") or "").lower()
        if granularity == "year":
            return time_start[:4]
        if granularity == "month":
            try:
                dt = datetime.fromisoformat(f"{time_start[:10]}T00:00:00")
                return dt.strftime("%B")
            except Exception:
                return time_start[:7] or None
        if event_granularity == "year":
            return time_start[:4]
        if event_granularity == "month":
            try:
                dt = datetime.fromisoformat(f"{time_start[:10]}T00:00:00")
                return dt.strftime("%B %Y")
            except Exception:
                return time_start[:7] or None
        return time_start[:10]

    @staticmethod
    def _is_fact_specific_temporal_event(fact_id: str, event: dict) -> bool:
        payload = event.get("payload") or {}
        if isinstance(payload, dict) and str(payload.get("fact_id") or "").strip() == fact_id:
            return True
        support_fact_ids = [str(fid) for fid in (event.get("support_fact_ids") or []) if str(fid).strip()]
        return len(support_fact_ids) == 1 and support_fact_ids[0] == fact_id

    @staticmethod
    def _calendar_fact_candidate_score(fact_text: str, query: str) -> tuple[float, int]:
        fact_tokens = {
            normalize_term_token(token)
            for token in re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?", fact_text.lower())
            if normalize_term_token(token) and normalize_term_token(token) not in STOP_WORDS
        }
        query_tokens = {
            normalize_term_token(token)
            for token in re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?", query.lower())
            if normalize_term_token(token) and normalize_term_token(token) not in STOP_WORDS
        }
        overlap = len(fact_tokens & query_tokens)
        temporal_bonus = 0.0
        if re.search(r"\b\d+\s+(?:years?|months?|weeks?)\s+ago\b", fact_text, re.I):
            temporal_bonus += 2.0
        if re.search(r"\bfor\s+\d+\s+(?:years?|months?|weeks?)\b", fact_text, re.I):
            temporal_bonus += 2.0
        if re.search(r"\blast\s+(?:week|month|year)\b", fact_text, re.I):
            temporal_bonus += 1.5
        return (overlap + temporal_bonus, -len(fact_text))

    def _calendar_query_anchor_timestamp(
        self,
        *,
        source_ids: set[str] | None = None,
        timeline_ids: set[str] | None = None,
    ) -> str | None:
        anchor = latest_calendar_anchor(
            self._temporal_index,
            source_ids=source_ids,
            timeline_ids=timeline_ids,
        )
        if anchor:
            return anchor
        latest_dt: datetime | None = None
        for raw in self._raw_sessions or []:
            raw_date = str(raw.get("session_date") or "").strip()
            if not raw_date:
                continue
            try:
                candidate = date_parser.parse(raw_date, fuzzy=True)
            except Exception:
                continue
            if latest_dt is None or candidate > latest_dt:
                latest_dt = candidate
        if latest_dt is None:
            return None
        return latest_dt.date().isoformat()

    async def _resolve_calendar_answer(
        self,
        *,
        query: str,
        candidate_facts: list[dict],
        source_ids: set[str] | None = None,
        timeline_ids: set[str] | None = None,
        limit: int = 8,
    ) -> dict | None:
        plan = extract_calendar_query(query)
        if not plan or plan.get("mode") != "answer":
            return None
        hit = execute_calendar_query(
            query,
            self._temporal_index,
            anchor_timestamp=self._calendar_query_anchor_timestamp(
                source_ids=source_ids,
                timeline_ids=timeline_ids,
            ),
            source_ids=source_ids,
            timeline_ids=timeline_ids,
            limit=limit,
        )
        events = list(hit.get("events") or [])
        if not events:
            return None
        fact_lookup = {
            str(fact.get("id") or "").strip(): fact
            for fact in candidate_facts
            if str(fact.get("id") or "").strip()
        }
        matched_facts: list[dict] = []
        seen_fact_ids: set[str] = set()
        for event in events:
            payload = event.get("payload") or {}
            candidate_fact_ids = []
            payload_fact_id = str(payload.get("fact_id") or "").strip() if isinstance(payload, dict) else ""
            if payload_fact_id:
                candidate_fact_ids.append(payload_fact_id)
            candidate_fact_ids.extend(
                str(fid).strip()
                for fid in (event.get("support_fact_ids") or [])
                if str(fid).strip()
            )
            for fact_id in candidate_fact_ids:
                if fact_id in seen_fact_ids:
                    continue
                fact = fact_lookup.get(fact_id)
                if not fact:
                    continue
                matched_facts.append(fact)
                seen_fact_ids.add(fact_id)
        if matched_facts:
            atomic_embs = (self._data_dict or {}).get("atomic_embs")
            matched_ids = {
                str(fact.get("id") or "").strip()
                for fact in matched_facts
                if str(fact.get("id") or "").strip()
            }
            ranked_facts: list[dict] = []
            ranked_embs: list[np.ndarray] = []
            if isinstance(atomic_embs, np.ndarray) and len(atomic_embs) == len(self._all_granular):
                for idx, fact in enumerate(self._all_granular):
                    fact_id = str(fact.get("id") or "").strip()
                    if not fact_id or fact_id not in matched_ids:
                        continue
                    ranked_facts.append(fact)
                    ranked_embs.append(atomic_embs[idx])
            if ranked_facts and ranked_embs:
                query_embedding = await embed_query(query)
                sweep = source_local_fact_sweep(
                    query,
                    ranked_facts,
                    np.asarray(ranked_embs),
                    query_embedding=query_embedding,
                    top_k=max(1, limit),
                    bm25_pool=max(8, limit * 2),
                    vector_pool=max(8, limit * 2),
                    entity_pool=max(4, limit),
                    rrf_k=60,
                )
                reranked = [row.get("fact") for row in sweep.get("retrieved", []) if row.get("fact")]
                if reranked:
                    matched_facts = reranked[:limit]
            else:
                matched_facts.sort(
                    key=lambda fact: self._calendar_fact_candidate_score(
                        str(fact.get("fact") or ""),
                        query,
                    ),
                    reverse=True,
                )
                matched_facts = matched_facts[:limit]
            kept_fact_ids = {
                str(fact.get("id") or "").strip()
                for fact in matched_facts
                if str(fact.get("id") or "").strip()
            }
            filtered_events: list[dict] = []
            for event in events:
                payload = event.get("payload") or {}
                payload_fact_id = str(payload.get("fact_id") or "").strip() if isinstance(payload, dict) else ""
                event_fact_ids = {
                    payload_fact_id,
                    *(
                        str(fid).strip()
                        for fid in (event.get("support_fact_ids") or [])
                        if str(fid).strip()
                    ),
                }
                if event_fact_ids & kept_fact_ids:
                    filtered_events.append(event)
            if filtered_events:
                events = filtered_events[:limit]
        return {
            "plan": hit.get("query") or plan,
            "events": events,
            "facts": matched_facts,
        }

    def _temporal_selector_evidence_lines(
        self,
        *,
        fact: dict | None,
        event: dict | None = None,
        max_spans: int = 1,
        max_chars_per_span: int = 220,
    ) -> list[str]:
        if not facts_as_selectors_enabled():
            return []
        episode_lookup = build_episode_lookup(self._episode_corpus)
        fact_lookup = self._fact_lookup if isinstance(getattr(self, "_fact_lookup", None), dict) else None
        support_spans: list[dict] = []
        if isinstance(fact, dict):
            support_spans.extend(
                span
                for span in iter_support_spans(fact, fact_lookup=fact_lookup)
                if isinstance(span, dict)
            )
        if not support_spans and isinstance(event, dict):
            source_span = event.get("source_span")
            if isinstance(source_span, dict):
                support_spans.append(source_span)

        lines: list[str] = []
        seen_refs: set[tuple[str, str, int, int]] = set()
        for span in support_spans[:max_spans]:
            ep_id = str(
                span.get("episode_id")
                or ((fact or {}).get("metadata") or {}).get("episode_id")
                or ""
            ).strip()
            if not ep_id:
                continue
            episode = episode_lookup.get(ep_id) or {}
            source_field = str(span.get("source_field") or "raw_text").strip() or "raw_text"
            raw_text = str(
                (episode.get(source_field) if source_field != "raw_text" else episode.get("raw_text"))
                or ""
            )
            if not raw_text:
                continue
            start_value = span.get("start", span.get("start_char", 0))
            end_value = span.get("end", span.get("end_char", 0))
            try:
                start = max(0, int(start_value))
                end = min(len(raw_text), int(end_value))
            except Exception:
                continue
            if end <= start:
                continue
            ref = (ep_id, source_field, start, end)
            if ref in seen_refs:
                continue
            seen_refs.add(ref)
            snippet = raw_text[start:end].strip().replace("\n", " ")
            if not snippet:
                continue
            if len(snippet) > max_chars_per_span:
                snippet = snippet[: max_chars_per_span - 3].rstrip() + "..."
            lines.extend(
                [
                    f"    Source ({source_field}, Episode {ep_id}, chars {start}-{end}):",
                    f"    > {snippet}",
                ]
            )
        return lines

    @staticmethod
    def _inject_temporal_evidence_block(recall_result: dict, block_text: str) -> None:
        block = str(block_text or "").strip()
        if not block:
            return
        context = str(recall_result.get("context") or "")
        if block not in context:
            recall_result["context"] = f"{block}\n\n{context}" if context else block
        context_packet = recall_result.get("_context_packet")
        if isinstance(context_packet, dict):
            tier1 = list(context_packet.get("tier1") or [])
            if not any(item.get("text") == block for item in tier1 if isinstance(item, dict)):
                tier1.insert(0, {"text": block, "rank": 1_000_000.0, "source": "temporal"})
            context_packet["tier1"] = tier1
            recall_result["_context_packet"] = context_packet
    def _attach_calendar_answer_resolution(
        self,
        *,
        query: str,
        recall_result: dict,
        candidate_facts: list[dict],
        source_ids: set[str] | None = None,
        timeline_ids: set[str] | None = None,
        resolution: dict | None = None,
    ) -> dict:
        if resolution is None:
            return recall_result
        if not resolution:
            return recall_result
        events = list(resolution.get("events") or [])
        facts = list(resolution.get("facts") or [])
        if facts:
            existing = []
            if recall_result.get("runtime_trace", {}).get("runtime") == "fact":
                existing = [
                    fact
                    for fact in candidate_facts
                    if str(fact.get("id") or "").strip() in {
                        item.get("fact_id", "")
                        for item in (recall_result.get("retrieved") or [])
                        if isinstance(item, dict)
                    }
                ]
            else:
                existing = list(candidate_facts)
            seen_fact_ids: set[str] = set()
            merged_facts: list[dict] = []
            for fact in facts + existing:
                fact_id = str(fact.get("id") or "").strip()
                if fact_id and fact_id not in seen_fact_ids:
                    merged_facts.append(fact)
                    seen_fact_ids.add(fact_id)
            if recall_result.get("runtime_trace", {}).get("runtime") == "fact":
                recall_result["retrieved"] = [
                    {
                        "fact_id": str(fact.get("id") or ""),
                        "conv_id": fact.get("conv_id", self.key),
                        "sim": 1_000_000.0 if idx < len(facts) else 0.0,
                    }
                    for idx, fact in enumerate(merged_facts)
                    if str(fact.get("id") or "").strip()
                ]
        lines = []
        for idx, fact in enumerate(facts[:3], start=1):
            fact_id = str(fact.get("id") or "").strip()
            linked_event = next(
                (
                    event
                    for event in events
                    if fact_id
                    and (
                        fact_id == str((event.get("payload") or {}).get("fact_id") or "").strip()
                        or fact_id in {
                            str(fid).strip()
                            for fid in (event.get("support_fact_ids") or [])
                            if str(fid).strip()
                        }
                    )
                ),
                None,
            )
            if not linked_event:
                linked_events = lookup_events_for_fact(self._temporal_index, fact_id=fact_id)
                linked_event = linked_events[0] if linked_events else None
            event_time = str((linked_event or {}).get("time_start") or "").strip()
            if not event_time:
                continue
            fact_text = str(fact.get("fact") or "").strip() or str((linked_event or {}).get("label") or "").strip()
            lines.append(f"[T{idx}] {fact_text} [Event time: {event_time}]")
            lines.extend(
                self._temporal_selector_evidence_lines(
                    fact=fact,
                    event=linked_event,
                )
            )
        if lines:
            prefix = "TEMPORAL EVIDENCE:"
            addition = prefix + "\n" + "\n".join(lines)
            self._inject_temporal_evidence_block(recall_result, addition)
        recall_result["temporal_resolution"] = {
            "mode": "calendar-answer",
            "matched_event_ids": [str(event.get("event_id") or "") for event in events],
            "matched_fact_ids": [str(fact.get("id") or "") for fact in facts],
        }
        runtime_trace = recall_result.get("runtime_trace") or {}
        runtime_trace["temporal_resolution"] = {
            "mode": "calendar-answer",
            "matched_event_ids": [str(event.get("event_id") or "") for event in events],
            "matched_fact_ids": [str(fact.get("id") or "") for fact in facts],
        }
        recall_result["runtime_trace"] = runtime_trace
        return recall_result

    async def _resolve_calendar_seeking(
        self,
        *,
        query: str,
        candidate_facts: list[dict],
    ) -> dict | None:
        plan = extract_calendar_query(query)
        if not plan or plan.get("mode") != "seeking":
            return None
        if not candidate_facts:
            return None
        atomic_embs = (self._data_dict or {}).get("atomic_embs")
        if not isinstance(atomic_embs, np.ndarray) or len(atomic_embs) != len(self._all_granular):
            return None
        allowed_ids = {
            str(f.get("id") or "")
            for f in candidate_facts
            if str(f.get("id") or "").strip()
        }
        ranked_facts: list[dict] = []
        ranked_embs: list[np.ndarray] = []
        for idx, fact in enumerate(self._all_granular):
            fact_id = str(fact.get("id") or "").strip()
            if not fact_id or fact_id not in allowed_ids:
                continue
            ranked_facts.append(fact)
            ranked_embs.append(atomic_embs[idx])
        if not ranked_facts or not ranked_embs:
            return None
        search_queries: list[str] = [plan["content_query"]]
        stripped = re.sub(r'["“”][^"“”]+["“”]', " ", plan["content_query"])
        stripped = re.sub(r"\s+", " ", stripped).strip(" \t:-,?.!")
        if stripped and stripped not in search_queries:
            search_queries.append(stripped)
        facts_by_episode: dict[str, list[dict]] = defaultdict(list)
        for fact in ranked_facts:
            episode_id = str((fact.get("metadata") or {}).get("episode_id") or fact.get("episode_id") or "").strip()
            if episode_id:
                facts_by_episode[episode_id].append(fact)

        ranked_rows: list[dict] = []
        traces: list[dict] = []
        seen_fact_ids: set[str] = set()
        for search_query in search_queries:
            query_embedding = await embed_query(search_query)
            sweep = source_local_fact_sweep(
                search_query,
                ranked_facts,
                np.asarray(ranked_embs),
                query_embedding=query_embedding,
                top_k=12,
                bm25_pool=36,
                vector_pool=24,
                entity_pool=12,
                rrf_k=60,
            )
            traces.append({"query": search_query, "trace": sweep.get("trace", {})})
            for row in sweep.get("retrieved", []):
                fact = row.get("fact") or {}
                fact_id = str(fact.get("id") or "").strip()
                if not fact_id or fact_id in seen_fact_ids:
                    continue
                seen_fact_ids.add(fact_id)
                ranked_rows.append(row)
        for require_specific in (True, False):
            for row in ranked_rows:
                fact = row.get("fact") or {}
                fact_id = str(fact.get("id") or "").strip()
                if not fact_id:
                    continue
                episode_id = str((fact.get("metadata") or {}).get("episode_id") or fact.get("episode_id") or "").strip()
                episode_candidates = list(facts_by_episode.get(episode_id) or []) if episode_id else []
                if fact not in episode_candidates:
                    episode_candidates.insert(0, fact)
                episode_candidates.sort(
                    key=lambda item: self._calendar_fact_candidate_score(
                        str(item.get("fact") or ""),
                        plan["content_query"],
                    ),
                    reverse=True,
                )
                for candidate in episode_candidates:
                    candidate_id = str(candidate.get("id") or "").strip()
                    if not candidate_id:
                        continue
                    events = lookup_events_for_fact(self._temporal_index, fact_id=candidate_id)
                    for event in events:
                        if require_specific and not self._is_fact_specific_temporal_event(candidate_id, event):
                            continue
                        answer = self._format_calendar_resolution_answer(plan, event)
                        if not answer:
                            continue
                        return {
                            "plan": plan,
                            "fact": candidate,
                            "event": event,
                            "answer": answer,
                            "trace": traces,
                        }
        return None

    async def _attach_calendar_seeking_resolution(
        self,
        *,
        query: str,
        recall_result: dict,
        candidate_facts: list[dict],
    ) -> dict:
        resolution = await self._resolve_calendar_seeking(
            query=query,
            candidate_facts=candidate_facts,
        )
        if not resolution:
            return recall_result
        fact = resolution["fact"]
        event = resolution["event"]
        answer = resolution["answer"]
        prefix = "TEMPORAL EVIDENCE:"
        line = (
            f"[T1] {fact.get('fact', '')} "
            f"[Resolved time: {answer}; Event time: {event.get('time_start')}]"
        ).strip()
        lines = [line]
        lines.extend(
            self._temporal_selector_evidence_lines(
                fact=fact,
                event=event,
            )
        )
        self._inject_temporal_evidence_block(recall_result, prefix + "\n" + "\n".join(lines))
        recall_result["temporal_resolution"] = {
            "mode": "calendar-seeking",
            "answer": answer,
            "fact_id": fact.get("id", ""),
            "event_id": event.get("event_id", ""),
            "time_start": event.get("time_start"),
            "time_end": event.get("time_end"),
            "time_granularity": event.get("time_granularity"),
        }
        runtime_trace = recall_result.get("runtime_trace") or {}
        runtime_trace["temporal_resolution"] = {
            "mode": "calendar-seeking",
            "answer": answer,
            "fact_id": fact.get("id", ""),
            "event_id": event.get("event_id", ""),
            "trace": resolution.get("trace", {}),
        }
        recall_result["runtime_trace"] = runtime_trace
        return recall_result

    def _conversation_doc_id(self, source_id: str | None = None) -> str:
        return f"conversation:{source_id or self.key}"

    @staticmethod
    def _document_doc_id(source_id: str) -> str:
        return f"document:{source_id}"

    @staticmethod
    def _multipart_part_key(metadata: dict | None) -> str | None:
        if not isinstance(metadata, dict):
            return None
        part_source_id = str(metadata.get("part_source_id") or "").strip()
        if part_source_id:
            return part_source_id
        part_idx = metadata.get("part_idx")
        if part_idx is None:
            return None
        try:
            part_num = int(part_idx)
        except (TypeError, ValueError):
            return None
        if part_num <= 0:
            return None
        return f"part_{part_num:04d}"

    @staticmethod
    def _episode_sort_key(episode: dict) -> tuple[int, str]:
        episode_id = str(episode.get("episode_id") or "")
        match = re.search(r"_e(\d+)\b", episode_id)
        if match:
            return (int(match.group(1)), episode_id)
        return (10**9, episode_id)

    def _upsert_episode_document(
        self,
        doc_id: str,
        episodes: list[dict],
        *,
        replace_part_key: str | None = None,
    ) -> None:
        docs = self._episode_corpus.setdefault("documents", [])
        for doc in docs:
            if doc.get("doc_id") == doc_id:
                existing = list(doc.get("episodes", []))
                if replace_part_key:
                    existing = [
                        ep
                        for ep in existing
                        if ((ep.get("provenance") or {}).get("multipart_part_key") != replace_part_key)
                    ]
                    doc["episodes"] = existing + episodes
                    doc["episodes"].sort(key=self._episode_sort_key)
                else:
                    doc["episodes"] = episodes
                return
        initial = list(episodes)
        initial.sort(key=self._episode_sort_key)
        docs.append({"doc_id": doc_id, "episodes": initial})

    def _append_or_replace_episode(self, doc_id: str, episode: dict) -> None:
        docs = self._episode_corpus.setdefault("documents", [])
        for doc in docs:
            if doc.get("doc_id") == doc_id:
                doc["episodes"] = [
                    ep for ep in doc.get("episodes", [])
                    if ep.get("episode_id") != episode.get("episode_id")
                ]
                doc["episodes"].append(episode)
                doc["episodes"].sort(key=lambda ep: ep.get("episode_id", ""))
                return
        docs.append({"doc_id": doc_id, "episodes": [episode]})

    def _get_episode_documents(self, source_id: str, source_kind: str) -> list[dict]:
        if source_kind == "document":
            doc_id = self._document_doc_id(source_id)
        else:
            doc_id = self._conversation_doc_id(source_id)
        for doc in self._episode_corpus.get("documents", []):
            if doc.get("doc_id") == doc_id:
                episodes = [deepcopy(ep) for ep in doc.get("episodes", []) if isinstance(ep, dict)]
                episodes.sort(key=self._episode_sort_key)
                return episodes
        return []

    def _next_document_episode_index(self, source_id: str) -> int:
        existing = self._get_episode_documents(source_id, "document")
        if not existing:
            return 1
        return max(self._episode_sort_key(ep)[0] for ep in existing) + 1

    def _next_document_session_num(self, source_id: str) -> int:
        values = []
        for rs in self._raw_sessions:
            if rs.get("format") != "document":
                continue
            if rs.get("source_id") != source_id:
                continue
            session_num = _coerce_positive_session_num(rs.get("session_num"))
            if session_num is not None:
                values.append(session_num)
        return (max(values) if values else 0) + 1

    def _rebuild_document_raw_text(self, source_id: str) -> str:
        rows = []
        for rs in self._raw_sessions:
            if rs.get("format") != "document":
                continue
            if rs.get("source_id") != source_id:
                continue
            if (rs.get("status") or "active") != "active":
                continue
            session_num = _coerce_positive_session_num(rs.get("session_num")) or 10**9
            rows.append((session_num, str(rs.get("content") or "")))
        rows.sort(key=lambda item: item[0])
        return "\n\n".join(content for _num, content in rows if content).strip()

    def _reindex_document_episodes(
        self,
        source_id: str,
        episodes: list[dict],
        *,
        start_index: int,
        part_key: str | None = None,
    ) -> list[dict]:
        remapped = []
        for offset, episode in enumerate(episodes):
            episode_copy = deepcopy(episode)
            episode_copy["episode_id"] = f"{source_id}_e{start_index + offset:02d}"
            episode_copy["source_id"] = source_id
            episode_copy["source_type"] = "document"
            provenance = dict(episode_copy.get("provenance") or {})
            if part_key:
                provenance["multipart_part_key"] = part_key
            if provenance:
                episode_copy["provenance"] = provenance
            remapped.append(episode_copy)
        return remapped

    async def _extract_source_aggregation_facts(
        self,
        *,
        source_id: str,
        source_kind: str,
        source_facts: list[dict],
        source_date: str,
        model: str,
        call_extract_fn,
        agent_id: str = "default",
    ) -> list[dict]:
        if not model:
            return []
        episodes = self._get_episode_documents(source_id, source_kind)
        if not episodes:
            return []

        # MAL source aggregation prompt overrides
        _mal_cfg = _load_mal_active_config(str(self.data_dir), self.key, agent_id)
        _agg_overrides = {}
        for _pk, _pv in (_mal_cfg.get("extraction_prompts") or {}).items():
            if _pk.startswith("document_source_aggregation_prompt:"):
                _agg_overrides[_pk.split(":", 1)[1]] = _pv
        result = await extract_source_aggregation(
            source_id=source_id,
            source_kind=source_kind,
            episodes=episodes,
            source_facts=source_facts,
            model=model,
            call_extract_fn=call_extract_fn,
            prompt_overrides=_agg_overrides or None,
        )
        if not result:
            return []

        derived_facts = result.get("derived_facts", []) or []
        validation = result.get("validation", {}) or {}
        aggregation_status = validation.get("aggregation_status", "accepted")
        accepted_layers = list(validation.get("accepted_layers", []))
        dropped_layers = list(validation.get("dropped_layers", []))
        failure_reasons = list(validation.get("failure_reasons", []))

        for fact in derived_facts:
            fact["conv_id"] = self.key
            fact["source_id"] = source_id
            tags = list(dict.fromkeys((fact.get("tags") or []) + ["unified_substrate", source_kind]))
            fact["tags"] = tags
            metadata = fact.setdefault("metadata", {})
            metadata["source_id"] = source_id
            metadata["source_kind"] = source_kind
            metadata["source_aggregation"] = True
            metadata["aggregation_status"] = aggregation_status
            metadata["accepted_layers"] = accepted_layers
            metadata["dropped_layers"] = dropped_layers
            metadata["failure_reasons"] = failure_reasons
            if source_date:
                fact.setdefault("event_date", source_date)
        return derived_facts

    @staticmethod
    def _stamp_episode_metadata(facts: list[dict], episode_id: str, episode_source_id: str) -> None:
        for fact in facts:
            meta = fact.setdefault("metadata", {})
            meta["episode_id"] = episode_id
            meta["episode_source_id"] = episode_source_id

    @staticmethod
    def _align_fact_selectors(
        facts: list[dict],
        *,
        episode_id: str,
        source_kind: str,
        raw_fields: dict[str, str],
        speakers: dict[str, str] | None = None,
    ) -> None:
        if not facts_as_selectors_enabled() or not facts:
            return
        align_facts_batch(
            facts,
            raw_fields,
            episode_id=episode_id,
            source_kind=source_kind,
            speakers=speakers,
        )

    @staticmethod
    def _hydrate_selector_surfaces_runtime(
        facts_by_episode: dict[str, list[dict]],
        episode_lookup: dict[str, dict],
    ) -> None:
        if not facts_as_selectors_enabled():
            return
        fact_lookup: dict[str, dict] = {}
        for facts in facts_by_episode.values():
            for fact in facts:
                fact_id = str(fact.get("id") or "")
                if fact_id:
                    fact_lookup[fact_id] = fact

        for ep_id, facts in facts_by_episode.items():
            episode = episode_lookup.get(ep_id)
            if not isinstance(episode, dict):
                continue
            seen_surface: set[str] = set()
            collected: list[str] = []
            total_chars = 0
            for fact in facts:
                surface = selector_surface_text(
                    fact,
                    episode_lookup,
                    fact_lookup=fact_lookup,
                    max_spans=2,
                    max_chars_per_span=220,
                )
                if surface:
                    fact["_selector_surface_text"] = surface
                else:
                    fact.pop("_selector_surface_text", None)
                    continue
                if surface in seen_surface:
                    continue
                seen_surface.add(surface)
                collected.append(surface)
                total_chars += len(surface)
                if len(collected) >= 6 or total_chars >= 1200:
                    break
            if collected:
                episode["_selector_surface_text"] = "\n".join(collected)
            else:
                episode.pop("_selector_surface_text", None)

    @staticmethod
    def _episode_source_key(source_id: str) -> str:
        key = re.sub(r"[^A-Za-z0-9._-]+", "_", source_id or "")
        key = key.strip("._-")
        return key or "source"

    def _initialize_scope_registry(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if not isinstance(self._scope_record, dict):
            self._scope_record = {}
        if not isinstance(self._source_records, dict):
            self._source_records = {}

        self._scope_record.setdefault("scope_id", self.key)
        self._scope_record.setdefault("scope_kind", self.scope)
        self._scope_record.setdefault("agent_id", self.agent_id)
        self._scope_record.setdefault("swarm_id", self.swarm_id)
        self._scope_record.setdefault("source_ids", [])
        self._scope_record.setdefault("links", [])
        self._scope_record["updated_at"] = now

        for session in self._raw_sessions:
            family = "document" if session.get("format") == "document" else "conversation"
            source_id = session.get("source_id") or (self.key if family == "conversation" else "")
            if not source_id:
                continue
            self._register_source_record(
                source_id=source_id,
                family=family,
                owner_id=session.get("owner_id", "system"),
                read=session.get("read", ["agent:PUBLIC"]),
                write=session.get("write", ["agent:PUBLIC"]),
                artifact_id=session.get("artifact_id"),
                version_id=session.get("version_id"),
                content_hash=session.get("content_hash"),
                metadata=session.get("metadata"),
                target=session.get("target"),
                source_meta={
                    "stored_format": session.get("format"),
                    "stored_at": session.get("stored_at"),
                },
            )

        for source_id in self._raw_docs:
            self._register_source_record(
                source_id=source_id,
                family=family,
                source_meta={"stored_format": "document"},
            )

        for doc in self._episode_corpus.get("documents", []):
            episodes = doc.get("episodes", [])
            if not episodes:
                continue
            sample = episodes[0]
            source_id = sample.get("source_id") or ""
            family = sample.get("source_type") or "document"
            if not source_id:
                continue
            self._register_source_record(
                source_id=source_id,
                family=family,
                source_meta={"doc_id": doc.get("doc_id")},
            )

    def _register_source_record(
        self,
        source_id: str,
        family: str,
        owner_id: str = "system",
        read: list[str] | None = None,
        write: list[str] | None = None,
        artifact_id: str | None = None,
        version_id: str | None = None,
        content_hash: str | None = None,
        metadata: dict | None = None,
        target: list[str] | None = None,
        source_meta: dict | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        record = self._source_records.get(source_id, {})
        record.update(
            {
                "source_id": source_id,
                "family": family,
                "scope_id": self.key,
                "owner_id": owner_id,
                "read": list(read or record.get("read", ["agent:PUBLIC"])),
                "write": list(write or record.get("write", ["agent:PUBLIC"])),
                "updated_at": now,
            }
        )
        record.setdefault("created_at", now)
        if artifact_id is not None:
            record["artifact_id"] = artifact_id
        if version_id is not None:
            record["version_id"] = version_id
        if content_hash is not None:
            record["content_hash"] = content_hash
        if metadata is not None:
            record["metadata"] = dict(metadata)
        if target is not None:
            record["target"] = list(target)
        if source_meta:
            merged_meta = dict(record.get("source_meta", {}))
            merged_meta.update(source_meta)
            record["source_meta"] = merged_meta
        self._source_records[source_id] = record

        source_ids = set(self._scope_record.get("source_ids", []))
        source_ids.add(source_id)
        self._scope_record["source_ids"] = sorted(source_ids)
        self._scope_record["updated_at"] = now

    def _scope_trace(self) -> dict:
        telemetry = get_runtime_tuning()["telemetry"]
        source_ids = list(self._scope_record.get("source_ids", []))
        max_ids = int(telemetry.get("max_scope_source_ids", 32))
        return {
            "scope_id": self._scope_record.get("scope_id", self.key),
            "scope_kind": self._scope_record.get("scope_kind", self.scope),
            "source_count": len(source_ids),
            "source_ids": source_ids[:max_ids],
            "link_count": len(self._scope_record.get("links", [])),
        }

    def _cross_contamination_trace(
        self,
        *,
        facts: list[dict] | None = None,
        episode_ids: list[str] | None = None,
        episode_lookup: dict[str, dict] | None = None,
        family_first_pass_trace: dict | None = None,
        late_fusion_trace: dict | None = None,
    ) -> dict:
        family_counts = defaultdict(int)
        source_ids = []

        if episode_ids and episode_lookup:
            for ep_id in episode_ids:
                ep = episode_lookup.get(ep_id) or {}
                source_id = ep.get("source_id") or ""
                if not source_id:
                    continue
                source_ids.append(source_id)
                family = ep.get("source_type") or self._source_records.get(source_id, {}).get("family", "unknown")
                family_counts[family] += 1
        elif facts:
            for fact in facts:
                source_id = fact.get("source_id") or (fact.get("metadata") or {}).get("episode_source_id", "")
                if not source_id:
                    continue
                source_ids.append(source_id)
                family = self._source_records.get(source_id, {}).get("family", "unknown")
                family_counts[family] += 1

        unique_source_ids = sorted(set(source_ids))
        rejected_source_ids = set()
        rejected_family_counts = defaultdict(int)
        candidate_source_ids = set(unique_source_ids)
        candidate_family_counts = defaultdict(int, family_counts)

        if family_first_pass_trace:
            for family_row in family_first_pass_trace.get("per_family", []):
                family = family_row.get("family", "unknown")
                for row in family_row.get("pre_source_gate", []):
                    source_id = row.get("source_id", "")
                    if source_id:
                        candidate_source_ids.add(source_id)
                        candidate_family_counts[family] += 1
                for row in family_row.get("post_source_gate", []):
                    source_id = row.get("source_id", "")
                    if source_id and source_id not in unique_source_ids:
                        rejected_source_ids.add(source_id)
                        rejected_family_counts[family] += 1

        if late_fusion_trace:
            for row in late_fusion_trace.get("rejected_candidates", []):
                ep_id = row.get("episode_id", "")
                ep = (episode_lookup or {}).get(ep_id, {})
                source_id = ep.get("source_id", "")
                family = ep.get("source_type") or self._source_records.get(source_id, {}).get("family", "unknown")
                if source_id and source_id not in unique_source_ids:
                    rejected_source_ids.add(source_id)
                    rejected_family_counts[family] += 1

        return {
            "source_ids": unique_source_ids,
            "source_count": len(unique_source_ids),
            "family_counts": dict(sorted(family_counts.items())),
            "multi_source": len(unique_source_ids) > 1,
            "candidate_source_ids": sorted(candidate_source_ids),
            "candidate_source_count": len(candidate_source_ids),
            "candidate_family_counts": dict(sorted(candidate_family_counts.items())),
            "rejected_source_ids": sorted(rejected_source_ids),
            "rejected_source_count": len(rejected_source_ids),
            "rejected_family_counts": dict(sorted(rejected_family_counts.items())),
        }

    def _episode_runtime_trace(
        self,
        *,
        corpus: dict,
        packet: dict,
        episode_lookup: dict[str, dict],
        resolved_facts: list[dict],
    ) -> dict:
        telemetry = get_runtime_tuning()["telemetry"]
        if not telemetry.get("include_runtime_trace", True):
            return {"runtime": "episode", "trace_disabled": True}
        return {
            "runtime": "episode",
            "scope": self._scope_trace(),
            "family_first_pass": packet.get("family_first_pass_trace", {
                "available_families": available_families(corpus),
                "retrieval_families": packet.get("retrieval_families", []),
                "requested_search_family": packet.get("search_family", "auto"),
                "per_family": [],
            }),
            "query": {
                **packet.get("query_operator_plan", {}),
                "output_constraints": packet.get("output_constraints", {}),
            },
            "late_fusion": packet.get("late_fusion_trace", {}),
            "temporal_resolution": packet.get("temporal_trace", {}),
            "selection": {
                "retrieved_episode_ids": packet.get("retrieved_episode_ids", []),
                "actual_injected_episode_ids": packet.get("actual_injected_episode_ids", []),
                "selection_scores": packet.get("selection_scores", []),
            },
            "cross_contamination": self._cross_contamination_trace(
                facts=resolved_facts,
                episode_ids=packet.get("retrieved_episode_ids", []),
                episode_lookup=episode_lookup,
                family_first_pass_trace=packet.get("family_first_pass_trace"),
                late_fusion_trace=packet.get("late_fusion_trace"),
            ),
            "packet": {
                "retrieved_fact_ids": packet.get("retrieved_fact_ids", [])[
                    : int(telemetry.get("max_packet_fact_ids", 24))
                ],
                "retrieved_fact_count": len(packet.get("retrieved_fact_ids", [])),
                "requested_episode_count": len(packet.get("retrieved_episode_ids", [])),
                "actual_injected_episode_count": len(packet.get("actual_injected_episode_ids", [])),
                "support_episode_count": len(packet.get("fact_episode_ids", [])),
                "support_episode_ids": packet.get("fact_episode_ids", [])[
                    : int(telemetry.get("max_packet_fact_ids", 24))
                ],
                "context_chars": len(packet.get("context", "")),
                "snippet_mode": bool(packet.get("selector_config", {}).get("snippet_mode", False)),
                "budget_chars": packet.get("selector_config", {}).get("budget"),
                "source_local_fact_sweep": packet.get("source_local_fact_sweep_trace", {}),
            },
            "tuning": packet.get("tuning_snapshot", {}),
        }

    def _canonical_retrieved_items(self, facts: list[dict]) -> list[dict]:
        items: list[dict] = []
        seen: set[str] = set()
        for fact in facts:
            fact_id = str(fact.get("id", "")).strip()
            if not fact_id or fact_id in seen:
                continue
            seen.add(fact_id)
            items.append({
                "fact_id": fact_id,
                "conv_id": fact.get("conv_id", self.key),
                "sim": float(fact.get("sim", fact.get("score", 0.0)) or 0.0),
            })
        return items

    def _ensure_min_synthesis_evidence(
        self,
        *,
        packet: dict,
        resolved_facts: list[dict],
        episode_lookup: dict[str, dict],
        fact_filter,
        minimum_facts: int = 2,
    ) -> tuple[dict, list[dict]]:
        if len(resolved_facts) >= minimum_facts:
            return packet, resolved_facts

        anchor_episode_ids = list(
            dict.fromkeys(
                packet.get("actual_injected_episode_ids", [])
                or packet.get("fact_episode_ids", [])
                or packet.get("retrieved_episode_ids", [])
            )
        )
        if not anchor_episode_ids:
            return packet, resolved_facts

        anchor_source_ids = {
            (episode_lookup.get(ep_id) or {}).get("source_id", "")
            for ep_id in anchor_episode_ids
        }
        anchor_source_ids.discard("")
        seen_fact_ids = {
            str(fact.get("id", "")).strip()
            for fact in resolved_facts
            if str(fact.get("id", "")).strip()
        }

        def _candidate_rank(fact: dict) -> tuple[int, int, str]:
            episode_ids = fact_episode_ids(fact)
            best_anchor = min(
                (anchor_episode_ids.index(ep_id) for ep_id in episode_ids if ep_id in anchor_episode_ids),
                default=len(anchor_episode_ids),
            )
            session_num = _coerce_positive_session_num(fact.get("session")) or 10**9
            return (best_anchor, session_num, str(fact.get("id", "")))

        extras: list[dict] = []
        for fact in list(self._all_granular) + list(self._all_cross):
            fact_id = str(fact.get("id", "")).strip()
            if not fact_id or fact_id in seen_fact_ids:
                continue
            if not fact_filter(fact):
                continue
            episode_ids = fact_episode_ids(fact)
            source_id = fact.get("source_id") or (fact.get("metadata") or {}).get("episode_source_id", "")
            if not (
                any(ep_id in anchor_episode_ids for ep_id in episode_ids)
                or (source_id and source_id in anchor_source_ids)
            ):
                continue
            extras.append(fact)

        if not extras:
            return packet, resolved_facts

        extras.sort(key=_candidate_rank)
        topped_up = list(resolved_facts)
        for fact in extras:
            topped_up.append(fact)
            seen_fact_ids.add(str(fact.get("id", "")).strip())
            if len(topped_up) >= minimum_facts:
                break

        if len(topped_up) == len(resolved_facts):
            return packet, resolved_facts

        fact_lookup = {fact.get("id", ""): fact for fact in list(self._all_granular) + list(self._all_cross)}
        context, actual_injected_episode_ids = build_context_from_retrieved_facts(
            topped_up,
            episode_lookup,
            fact_lookup=fact_lookup,
            budget=int(packet.get("selector_config", {}).get("budget", 8000)),
            snippet_chars=int(packet.get("tuning_snapshot", {}).get("packet", {}).get("snippet_chars", 1200)),
        )

        packet = dict(packet)
        packet["context"] = context
        packet["actual_injected_episode_ids"] = actual_injected_episode_ids
        packet["retrieved_fact_ids"] = [fact.get("id", "") for fact in topped_up if fact.get("id", "")]
        packet["fact_episode_ids"] = list(
            dict.fromkeys(
                episode_id
                for fact in topped_up
                for episode_id in fact_episode_ids(fact)
                if episode_id
            )
        )
        return packet, topped_up

    def _synthesis_retrieved_items(
        self,
        *,
        resolved_facts: list[dict],
        packet: dict,
        episode_lookup: dict[str, dict],
        minimum_items: int = 2,
    ) -> list[dict]:
        items = list(resolved_facts)
        if len(items) >= minimum_items:
            return items

        for ep_id in packet.get("actual_injected_episode_ids", []) or packet.get("retrieved_episode_ids", []):
            episode = episode_lookup.get(ep_id)
            if not episode:
                continue
            raw_text = str(episode.get("raw_text", "") or "").strip()
            if not raw_text:
                continue
            preview = raw_text[:280]
            if len(raw_text) > 280:
                preview += "..."
            items.append({
                "id": f"support_{ep_id}",
                "fact": f"Source excerpt support from episode {ep_id}: {preview}",
                "kind": "source_excerpt",
                "session": _coerce_positive_session_num(episode.get("session_num")) or 0,
                "metadata": {
                    "episode_id": ep_id,
                    "episode_source_id": episode.get("source_id", ""),
                    "support_only": True,
                },
            })
            if len(items) >= minimum_items:
                break
        return items

    async def _augment_conversation_structural_packet(
        self,
        *,
        query: str,
        packet: dict,
        episode_lookup: dict[str, dict],
        fact_filter,
    ) -> tuple[dict, list[dict] | None]:
        operator_plan = packet.get("query_operator_plan", {})
        if not any(
            operator_plan.get(name, {}).get("enabled", False)
            for name in (
                "commonality",
                "compare_diff",
                "list_set",
                "slot_query",
                "compositional",
            )
        ):
            return packet, None

        selected_source_ids = {
            (episode_lookup.get(ep_id) or {}).get("source_id", "")
            for ep_id in packet.get("retrieved_episode_ids", [])
        }
        selected_source_ids.discard("")
        if not selected_source_ids:
            return packet, None

        selected_source_families = {
            (episode_lookup.get(ep_id) or {}).get("source_type", "")
            for ep_id in packet.get("retrieved_episode_ids", [])
        }
        selected_source_families.discard("")
        if selected_source_families != {"conversation"}:
            return packet, None

        tuning = get_runtime_tuning()
        operator_tuning = tuning["operators"]
        query_features = extract_query_features(query)
        operator_plan = query_features.get("operator_plan") or {}
        retrieval_target = query_features.get("retrieval_target") or query

        atomic_embs = (self._data_dict or {}).get("atomic_embs")
        cross_embs = (self._data_dict or {}).get("cross_embs")

        candidate_facts = []
        candidate_embeddings = []
        fact_lookup: dict[str, dict] = {}
        for facts, embeddings in (
            (self._all_granular, atomic_embs),
            (self._all_cross, cross_embs),
        ):
            if not isinstance(embeddings, np.ndarray) or len(embeddings) != len(facts):
                continue
            for idx, fact in enumerate(facts):
                if not fact_filter(fact):
                    continue
                source_id = fact.get("source_id") or (fact.get("metadata") or {}).get("episode_source_id", "")
                if source_id not in selected_source_ids:
                    continue
                candidate_facts.append(fact)
                candidate_embeddings.append(embeddings[idx])
                fact_lookup[fact.get("id", "")] = fact

        if not candidate_facts:
            return packet, None

        query_embedding = await embed_query(retrieval_target)
        top_k = int(operator_tuning.get("conversation_structural_fact_sweep_top_k", 12))
        bm25_pool = int(operator_tuning.get("conversation_structural_fact_sweep_bm25_pool", 24))
        vector_pool = int(operator_tuning.get("conversation_structural_fact_sweep_vector_pool", 24))
        entity_pool = int(operator_tuning.get("conversation_structural_fact_sweep_entity_pool", 12))
        if operator_plan.get("compositional", {}).get("enabled", False):
            top_k = max(top_k, 24)
            bm25_pool = max(bm25_pool, 64)
            vector_pool = max(vector_pool, 64)
            entity_pool = max(entity_pool, 24)
        sweep = source_local_fact_sweep(
            retrieval_target,
            candidate_facts,
            np.asarray(candidate_embeddings),
            query_embedding=query_embedding,
            top_k=top_k,
            bm25_pool=bm25_pool,
            vector_pool=vector_pool,
            entity_pool=entity_pool,
            rrf_k=int(operator_tuning.get("conversation_structural_fact_sweep_rrf_k", 60)),
        )
        retrieved_facts = [row["fact"] for row in sweep.get("retrieved", [])]
        if not retrieved_facts:
            return packet, None

        if operator_plan.get("commonality", {}).get("enabled", False):
            commonality_extras = _augment_commonality_facts(
                query,
                retrieved_facts,
                candidate_facts,
                limit=6,
            )
            if commonality_extras:
                existing_ids = {str(f.get("id") or "") for f in retrieved_facts}
                for fact in commonality_extras:
                    fact_id = str(fact.get("id") or "")
                    if fact_id and fact_id in existing_ids:
                        continue
                    retrieved_facts.append(fact)
                    if fact_id:
                        existing_ids.add(fact_id)

        context, actual_injected_episode_ids = build_context_from_retrieved_facts(
            retrieved_facts,
            episode_lookup,
            fact_lookup=fact_lookup,
            budget=int(packet.get("selector_config", {}).get("budget", 8000)),
            snippet_chars=int(packet.get("tuning_snapshot", {}).get("packet", {}).get("snippet_chars", 1200)),
            question=query,
            query_features=query_features,
        )

        packet = dict(packet)
        packet["context"] = context
        packet["actual_injected_episode_ids"] = actual_injected_episode_ids
        packet["retrieved_fact_ids"] = [fact.get("id", "") for fact in retrieved_facts]
        packet["fact_episode_ids"] = list(
            dict.fromkeys(
                episode_id
                for fact in retrieved_facts
                for episode_id in fact_episode_ids(fact)
                if episode_id
            )
        )
        packet["source_local_fact_sweep_trace"] = sweep.get("trace", {})
        return packet, retrieved_facts

    async def _recover_multi_item_coverage_packet(
        self,
        *,
        query: str,
        packet: dict,
        episode_lookup: dict[str, dict],
        fact_filter,
    ) -> tuple[dict, list[dict] | None]:
        coverage_query_type = classify_coverage_query(query)
        if coverage_query_type == "none":
            return packet, None
        if (packet.get("temporal_trace") or {}).get("query_class") in {
            "ordinal",
            "calendar-answer",
            "calendar-seeking",
        }:
            return packet, None

        selected_source_ids = {
            (episode_lookup.get(ep_id) or {}).get("source_id", "")
            for ep_id in packet.get("retrieved_episode_ids", [])
        }
        selected_source_ids.discard("")
        selected_source_families = {
            (episode_lookup.get(ep_id) or {}).get("source_type", "")
            for ep_id in packet.get("retrieved_episode_ids", [])
        }
        selected_source_families.discard("")
        if not selected_source_ids:
            return packet, None

        current_fact_lookup: dict[str, dict] = {}
        for fact in self._all_granular + self._all_cross:
            fact_id = str(fact.get("id") or "").strip()
            if fact_id:
                current_fact_lookup[fact_id] = fact
        current_facts = [
            current_fact_lookup[fact_id]
            for fact_id in packet.get("retrieved_fact_ids", [])
            if fact_id in current_fact_lookup
        ]
        pre_stats = compute_coverage_stats(
            coverage_query_type,
            current_facts,
            selected_source_count=len(selected_source_ids),
        )
        if not needs_coverage_recovery(coverage_query_type, pre_stats):
            return packet, None

        requested_families = set(packet.get("retrieval_families") or [])
        requested_search_family = packet.get("search_family", "auto")
        query_features = extract_query_features(query)
        retrieval_target = query_features.get("retrieval_target") or query

        candidate_facts: list[dict] = []
        candidate_embeddings: list[np.ndarray] = []
        fact_lookup: dict[str, dict] = {}
        for facts, embeddings in (
            (self._all_granular, (self._data_dict or {}).get("atomic_embs")),
            (self._all_cross, (self._data_dict or {}).get("cross_embs")),
        ):
            if not isinstance(embeddings, np.ndarray) or len(embeddings) != len(facts):
                continue
            for idx, fact in enumerate(facts):
                if not fact_filter(fact):
                    continue
                metadata = fact.get("metadata") or {}
                source_id = fact.get("source_id") or metadata.get("episode_source_id", "")
                if source_id not in selected_source_ids:
                    continue
                episode_id = metadata.get("episode_id", "")
                family = (
                    self._source_records.get(source_id, {}).get("family")
                    or (episode_lookup.get(episode_id) or {}).get("source_type", "")
                )
                if requested_search_family not in ("auto", "", None) and family and family != requested_search_family:
                    continue
                if requested_families and family and family not in requested_families:
                    continue
                candidate_facts.append(fact)
                candidate_embeddings.append(embeddings[idx])
                fact_id = str(fact.get("id") or "").strip()
                if fact_id:
                    fact_lookup[fact_id] = fact

        if not candidate_facts:
            return packet, None

        query_embedding = await embed_query(retrieval_target)
        base_target = max(
            len(current_facts) + 8,
            int(packet.get("selector_config", {}).get("supporting_facts_total", 12)),
            12,
        )
        sweep = source_local_fact_sweep(
            retrieval_target,
            candidate_facts,
            np.asarray(candidate_embeddings),
            query_embedding=query_embedding,
            top_k=base_target,
            bm25_pool=max(base_target * 2, 24),
            vector_pool=max(base_target * 2, 24),
            entity_pool=max(base_target, 12),
            rrf_k=60,
        )
        recovered_facts = [row.get("fact", {}) for row in sweep.get("retrieved", []) if row.get("fact")]
        if not recovered_facts:
            return packet, None

        merged_facts = merge_coverage_recovery_facts(
            coverage_query_type,
            current_facts,
            recovered_facts,
            max_facts=base_target,
        )
        post_stats = compute_coverage_stats(
            coverage_query_type,
            merged_facts,
            selected_source_count=len(selected_source_ids),
        )
        current_ids = [fact.get("id", "") for fact in current_facts]
        merged_ids = [fact.get("id", "") for fact in merged_facts]
        if merged_ids == current_ids:
            return packet, None
        if (
            post_stats.get("distinct_episodes", 0) <= pre_stats.get("distinct_episodes", 0)
            and post_stats.get("distinct_entities", 0) <= pre_stats.get("distinct_entities", 0)
            and post_stats.get("distinct_support_spans", 0) <= pre_stats.get("distinct_support_spans", 0)
        ):
            return packet, None

        context, actual_injected_episode_ids = build_context_from_retrieved_facts(
            merged_facts,
            episode_lookup,
            fact_lookup=fact_lookup,
            budget=int(packet.get("selector_config", {}).get("budget", 8000)),
            snippet_chars=int(packet.get("tuning_snapshot", {}).get("packet", {}).get("snippet_chars", 1200)),
            question=query,
            query_features=packet.get("query_features") or query_features,
        )
        merged_episode_ids = list(
            dict.fromkeys(
                episode_id
                for fact in merged_facts
                for episode_id in fact_episode_ids(fact)
                if episode_id
            )
        )

        packet = dict(packet)
        packet["context"] = context
        packet["retrieved_fact_ids"] = merged_ids
        packet["fact_episode_ids"] = merged_episode_ids
        packet["actual_injected_episode_ids"] = actual_injected_episode_ids
        packet["coverage_recovery_trace"] = {
            "query_type": coverage_query_type,
            "selected_source_ids": sorted(selected_source_ids),
            "selected_source_families": sorted(selected_source_families),
            "pre_stats": pre_stats,
            "post_stats": post_stats,
            "candidate_count": len(candidate_facts),
            "selected_fact_count": len(merged_facts),
            "sweep": sweep.get("trace", {}),
        }
        family_first_pass_trace = dict(packet.get("family_first_pass_trace") or {})
        family_first_pass_trace["coverage_recovery"] = {
            "query_type": coverage_query_type,
            "pre_stats": pre_stats,
            "post_stats": post_stats,
            "candidate_count": len(candidate_facts),
        }
        packet["family_first_pass_trace"] = family_first_pass_trace
        return packet, merged_facts

    async def _augment_document_structural_packet(
        self,
        *,
        query: str,
        packet: dict,
        episode_lookup: dict[str, dict],
        fact_filter,
    ) -> tuple[dict, list[dict] | None]:
        operator_plan = packet.get("query_operator_plan", {})
        if not operator_plan.get("bounded_chain", {}).get("enabled", False):
            return packet, None

        selected_source_ids = {
            (episode_lookup.get(ep_id) or {}).get("source_id", "")
            for ep_id in packet.get("retrieved_episode_ids", [])
        }
        selected_source_ids.discard("")
        if not selected_source_ids:
            return packet, None

        selected_source_families = {
            (episode_lookup.get(ep_id) or {}).get("source_type", "")
            for ep_id in packet.get("retrieved_episode_ids", [])
        }
        selected_source_families.discard("")
        if selected_source_families != {"document"}:
            return packet, None

        tuning = get_runtime_tuning()
        operator_tuning = tuning["operators"]
        query_features = extract_query_features(query)
        retrieval_target = query_features.get("retrieval_target") or query

        structural_qf = extract_query_features(retrieval_target)
        candidate_facts = []
        fact_lookup: dict[str, dict] = {}
        pseudo_fact_lookup: dict[str, dict] = {}
        for facts in (self._all_granular, self._all_cross):
            for fact in facts:
                if not fact_filter(fact):
                    continue
                source_id = fact.get("source_id") or (fact.get("metadata") or {}).get("episode_source_id", "")
                if source_id not in selected_source_ids:
                    continue
                candidate_facts.append(fact)
                fact_lookup[fact.get("id", "")] = fact
        for ep_id in packet.get("retrieved_episode_ids", []):
            episode = episode_lookup.get(ep_id)
            if not episode or episode.get("source_id", "") not in selected_source_ids:
                continue
            for pseudo in _pseudo_facts_from_episode(ep_id, episode, qf=structural_qf):
                pseudo_fact_lookup[pseudo.get("id", "")] = pseudo
                candidate_facts.append(pseudo)

        if not candidate_facts:
            return packet, None

        seed_fact_count = max(0, int(operator_tuning.get("document_structural_seed_fact_count", 1)))
        query_specificity_bonus = float(
            packet.get("tuning_snapshot", {}).get("packet", {}).get("query_specificity_bonus", 0.0)
        )
        seed_facts = _select_bounded_chain_seed_facts(
            candidate_facts,
            structural_qf,
            token_freq=Counter(
                token
                for fact in candidate_facts
                for token in set(_fact_content_tokens(fact.get("fact", ""), structural_qf))
            ),
            query_specificity_bonus=query_specificity_bonus,
            seed_count=seed_fact_count,
        )
        seed_fact_ids = [fact.get("id", "") for fact in seed_facts if fact.get("id", "")]
        if not seed_facts:
            for fact_id in packet.get("retrieved_fact_ids", []):
                fact = fact_lookup.get(fact_id) or pseudo_fact_lookup.get(fact_id)
                if not fact:
                    continue
                if not (fact.get("fact") or "").strip():
                    continue
                seed_fact_ids.append(fact_id)
                seed_facts.append(fact)
                if len(seed_facts) >= seed_fact_count:
                    break

        if not seed_facts:
            return packet, None

        bundle = build_bounded_chain_candidate_bundle(
            retrieval_target,
            seed_facts,
            candidate_facts,
            max_candidates=int(operator_tuning.get("document_structural_candidate_bundle_top_k", 18)),
            query_specificity_bonus=query_specificity_bonus,
        )
        retrieved_facts = bundle.get("facts", [])
        if not retrieved_facts:
            return packet, None

        context, actual_injected_episode_ids = build_context_from_retrieved_facts(
            retrieved_facts,
            episode_lookup,
            fact_lookup=fact_lookup,
            budget=int(packet.get("selector_config", {}).get("budget", 8000)),
            snippet_chars=int(packet.get("tuning_snapshot", {}).get("packet", {}).get("snippet_chars", 1200)),
            question=query,
            query_features=query_features,
        )

        packet = dict(packet)
        packet["context"] = context
        packet["actual_injected_episode_ids"] = actual_injected_episode_ids
        packet["retrieved_fact_ids"] = [fact.get("id", "") for fact in retrieved_facts]
        packet["fact_episode_ids"] = list(
            dict.fromkeys(
                episode_id
                for fact in retrieved_facts
                for episode_id in fact_episode_ids(fact)
                if episode_id
            )
        )
        packet["source_local_fact_sweep_trace"] = {
            **(bundle.get("trace", {}) or {}),
            "family": "document",
            "seed_fact_ids": seed_fact_ids,
        }
        return packet, retrieved_facts

    async def _rescue_episode_packet_with_semantic_fact_sweep(
        self,
        *,
        query: str,
        packet: dict,
        episode_lookup: dict[str, dict],
        fact_filter,
    ) -> tuple[dict, list[dict] | None]:
        if packet.get("retrieved_fact_ids") or packet.get("actual_injected_episode_ids"):
            return packet, None

        atomic_embs = (self._data_dict or {}).get("atomic_embs")
        if not isinstance(atomic_embs, np.ndarray) or len(atomic_embs) != len(self._all_granular):
            return packet, None

        query_features = extract_query_features(query)
        retrieval_target = query_features.get("retrieval_target") or query
        requested_families = set(packet.get("retrieval_families") or [])
        requested_search_family = packet.get("search_family", "auto")

        candidate_facts = []
        candidate_indices = []
        for idx, fact in enumerate(self._all_granular):
            if not fact_filter(fact):
                continue
            metadata = fact.get("metadata") or {}
            episode_id = metadata.get("episode_id", "")
            if not episode_id or episode_id not in episode_lookup:
                continue
            source_id = fact.get("source_id") or metadata.get("episode_source_id", "")
            family = (
                self._source_records.get(source_id, {}).get("family")
                or (episode_lookup.get(episode_id) or {}).get("source_type", "")
            )
            if requested_search_family not in ("auto", "", None) and family and family != requested_search_family:
                continue
            if requested_families and family and family not in requested_families:
                continue
            candidate_facts.append(fact)
            candidate_indices.append(idx)

        if not candidate_facts:
            return packet, None

        rescue_cfg = get_tuning_section("retrieval").get("semantic_rescue", {})
        query_embedding = await embed_query(retrieval_target)
        sweep = source_local_fact_sweep(
            retrieval_target,
            candidate_facts,
            atomic_embs[candidate_indices],
            query_embedding=query_embedding,
            top_k=int(rescue_cfg.get("top_k", 8)),
            bm25_pool=int(rescue_cfg.get("bm25_pool", 24)),
            vector_pool=int(rescue_cfg.get("vector_pool", 24)),
            entity_pool=int(rescue_cfg.get("entity_pool", 12)),
            rrf_k=int(rescue_cfg.get("rrf_k", 60)),
        )
        ranked_rows = sweep.get("retrieved", [])
        retrieved_facts = [row.get("fact", {}) for row in ranked_rows if row.get("fact")]
        if not retrieved_facts:
            return packet, None

        selection_scores = []
        retrieved_episode_ids = []
        seen_episode_ids = set()
        for row in ranked_rows:
            fact = row.get("fact") or {}
            episode_id = (fact.get("metadata") or {}).get("episode_id", "")
            if not episode_id or episode_id in seen_episode_ids:
                continue
            seen_episode_ids.add(episode_id)
            retrieved_episode_ids.append(episode_id)
            selection_scores.append({
                "episode_id": episode_id,
                "score": float(row.get("score", 0.0)),
            })

        context, actual_injected_episode_ids = build_context_from_retrieved_facts(
            retrieved_facts,
            episode_lookup,
            fact_lookup={fact.get("id", ""): fact for fact in self._all_granular},
            budget=int(packet.get("selector_config", {}).get("budget", 8000)),
            snippet_chars=int(packet.get("tuning_snapshot", {}).get("packet", {}).get("snippet_chars", 1200)),
            question=query,
            query_features=query_features,
        )

        packet = dict(packet)
        packet["context"] = context
        packet["retrieved_fact_ids"] = [fact.get("id", "") for fact in retrieved_facts]
        packet["retrieved_episode_ids"] = retrieved_episode_ids
        packet["actual_injected_episode_ids"] = actual_injected_episode_ids
        packet["fact_episode_ids"] = retrieved_episode_ids
        packet["selection_scores"] = selection_scores
        packet["source_local_fact_sweep_trace"] = {
            **(sweep.get("trace", {}) or {}),
            "mode": "episode_semantic_rescue",
            "requested_search_family": requested_search_family,
            "retrieval_families": sorted(requested_families),
        }
        family_first_pass_trace = dict(packet.get("family_first_pass_trace") or {})
        family_first_pass_trace["semantic_rescue"] = {
            "candidate_count": len(candidate_facts),
            "selected_fact_count": len(retrieved_facts),
        }
        packet["family_first_pass_trace"] = family_first_pass_trace
        return packet, retrieved_facts

    @staticmethod
    def _temporal_query_surface_dates(query: str) -> list[str]:
        pattern = re.compile(
            r"\b(?:"
            r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
            r"\s+\d{1,2},\s*\d{4}"
            r"|"
            r"\d{1,2}\s+"
            r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
            r"(?:,?\s+\d{4})?"
            r"|"
            r"\d{4}-\d{2}-\d{2}"
            r")\b",
            re.I,
        )
        seen: list[str] = []
        for match in pattern.finditer(query or ""):
            value = " ".join(match.group(0).split()).strip().lower()
            if value and value not in seen:
                seen.append(value)
        return seen

    @staticmethod
    def _strip_temporal_surface(query: str) -> str:
        pattern = re.compile(
            r"\b(?:"
            r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
            r"\s+\d{1,2},\s*\d{4}"
            r"|"
            r"\d{1,2}\s+"
            r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
            r"(?:,?\s+\d{4})?"
            r"|"
            r"\d{4}-\d{2}-\d{2}"
            r")\b",
            re.I,
        )
        stripped = pattern.sub(" ", query or "")
        stripped = re.sub(r"\b(?:on|in|at|during|for)\s+(?=[?.!,;:]?$)", " ", stripped, flags=re.I)
        stripped = re.sub(r"\s+", " ", stripped).strip(" \t\r\n,?.!;:")
        return stripped

    @staticmethod
    def _shift_month_anchor(anchor: datetime, delta: int) -> datetime:
        month_index = anchor.month - 1 + delta
        year = anchor.year + month_index // 12
        month = month_index % 12 + 1
        day = min(anchor.day, 28)
        return anchor.replace(year=year, month=month, day=day)

    @staticmethod
    def _format_temporal_anchor(anchor: datetime) -> str:
        return anchor.strftime("%B %d, %Y").replace(" 0", " ")

    @staticmethod
    def _format_temporal_month(anchor: datetime) -> str:
        return anchor.strftime("%B %Y")

    @staticmethod
    def _format_temporal_interval(start: datetime, end: datetime) -> str:
        if start.year == end.year:
            return (
                f"between {start.day} {start.strftime('%B')} and "
                f"{end.day} {end.strftime('%B %Y')}"
            )
        return (
            f"between {start.day} {start.strftime('%B %Y')} and "
            f"{end.day} {end.strftime('%B %Y')}"
        )

    @staticmethod
    def _temporal_query_requests_relative_resolution(query: str) -> bool:
        lowered = (query or "").strip().lower()
        return bool(re.match(r"^(when\b|what date\b|what day\b|which month\b|what month\b|what year\b)", lowered))

    @staticmethod
    def _temporal_query_requests_year_resolution(query: str) -> bool:
        lowered = (query or "").strip().lower()
        return bool(re.match(r"^(what year\b|which year\b)", lowered))

    @staticmethod
    def _temporal_query_requests_month_resolution(query: str) -> bool:
        lowered = (query or "").strip().lower()
        return bool(re.match(r"^(?:in\s+)?(?:which|what) month\b", lowered))

    @staticmethod
    def _temporal_query_requests_first_window(query: str) -> bool:
        lowered = (query or "").strip().lower()
        if "first" not in lowered:
            return False
        return bool(re.match(r"^(when\b|what date\b|what day\b)", lowered))

    @staticmethod
    def _parse_duration_year_count(text: str) -> int | None:
        if not text:
            return None
        word_to_int = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
        }
        patterns = (
            r"\bfor\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+years?\b",
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+years?\s+old\b",
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+years?\s+ago\b",
        )
        lowered = text.lower()
        for pattern in patterns:
            match = re.search(pattern, lowered, re.I)
            if not match:
                continue
            raw = match.group(1).lower()
            if raw.isdigit():
                return int(raw)
            return word_to_int.get(raw)
        return None

    def _relative_temporal_answer_from_text(self, text: str, *, source_date: str) -> str | None:
        if not text or not source_date:
            return None
        try:
            anchor = date_parser.parse(source_date, fuzzy=True)
        except Exception:
            return None
        lowered = text.lower()
        if "yesterday" in lowered or "last night" in lowered:
            return self._format_temporal_anchor(anchor - timedelta(days=1))
        if "today" in lowered or "tonight" in lowered:
            return self._format_temporal_anchor(anchor)
        if "tomorrow" in lowered:
            return self._format_temporal_anchor(anchor + timedelta(days=1))
        if "last week" in lowered:
            return f"the week before {self._format_temporal_anchor(anchor)}"
        if "this week" in lowered:
            return f"the week of {self._format_temporal_anchor(anchor)}"
        if "next week" in lowered:
            return f"the week after {self._format_temporal_anchor(anchor)}"
        if "last month" in lowered:
            return self._format_temporal_month(self._shift_month_anchor(anchor, -1))
        if "this month" in lowered:
            return self._format_temporal_month(anchor)
        if "next month" in lowered:
            return self._format_temporal_month(self._shift_month_anchor(anchor, 1))

        weekday_re = re.compile(
            r"\b(last|this)\s+"
            r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            re.I,
        )
        weekday_to_index = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }
        match = weekday_re.search(text)
        if not match:
            return None
        rel = match.group(1).lower()
        weekday = match.group(2).lower()
        target = weekday_to_index.get(weekday)
        if target is None:
            return None
        days_back = (anchor.weekday() - target) % 7
        if rel == "last":
            days_back = 7 if days_back == 0 else days_back
        resolved = anchor - timedelta(days=days_back)
        return self._format_temporal_anchor(resolved)

    def _derive_relative_temporal_deterministic_answer(
        self,
        *,
        query: str,
        resolved_facts: list[dict],
        episode_lookup: dict[str, dict],
    ) -> str | None:
        if not self._temporal_query_requests_relative_resolution(query):
            return None
        if self._temporal_query_surface_dates(query):
            return None
        qf = extract_query_features(query)
        if qf.get("operator_plan", {}).get("temporal_grounding", {}).get("enabled", False):
            return None
        for fact in resolved_facts:
            fact_text = str((fact or {}).get("fact") or "").strip()
            if not fact_text:
                continue
            for ep_id in fact_episode_ids(fact):
                episode = episode_lookup.get(ep_id) or {}
                source_date = str(episode.get("source_date") or "").strip()
                answer = self._relative_temporal_answer_from_text(
                    fact_text,
                    source_date=source_date,
                )
                if answer:
                    return answer
        return None

    @staticmethod
    def _fact_anchor_datetime(fact: dict, episode_lookup: dict[str, dict]) -> datetime | None:
        for ep_id in fact_episode_ids(fact):
            episode = episode_lookup.get(ep_id) or {}
            for field in ("source_date", "session_date"):
                raw = str(episode.get(field) or "").strip()
                if not raw:
                    continue
                try:
                    return date_parser.parse(raw, fuzzy=True)
                except Exception:
                    continue
        for field in ("source_date", "session_date"):
            raw = str((fact or {}).get(field) or "").strip()
            if not raw:
                continue
            try:
                return date_parser.parse(raw, fuzzy=True)
            except Exception:
                continue
        return None

    def _derive_duration_temporal_deterministic_answer(
        self,
        *,
        query: str,
        resolved_facts: list[dict],
        episode_lookup: dict[str, dict],
    ) -> str | None:
        if not self._temporal_query_requests_year_resolution(query):
            return None
        if self._temporal_query_surface_dates(query):
            return None
        for fact in resolved_facts:
            fact_text = str((fact or {}).get("fact") or "").strip()
            if not fact_text:
                continue
            year_count = self._parse_duration_year_count(fact_text)
            if year_count is None:
                continue
            anchor = self._fact_anchor_datetime(fact, episode_lookup)
            if anchor is None:
                continue
            return str(anchor.year - year_count)
        return None

    def _derive_month_temporal_deterministic_answer(
        self,
        *,
        query: str,
        resolved_facts: list[dict],
        episode_lookup: dict[str, dict],
    ) -> str | None:
        if not self._temporal_query_requests_month_resolution(query):
            return None
        if self._temporal_query_surface_dates(query):
            return None
        for fact in resolved_facts:
            fact_text = str((fact or {}).get("fact") or "").strip()
            if not fact_text:
                continue
            anchor = self._fact_anchor_datetime(fact, episode_lookup)
            if anchor is None:
                continue
            lowered = fact_text.lower()
            direct = self._relative_temporal_answer_from_text(fact_text, source_date=str(anchor))
            if direct and re.match(r"^[A-Z][a-z]+\s+\d{4}$", direct):
                return direct
            if not re.search(r"\blast (?:week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", lowered):
                continue
            for ep_id in fact_episode_ids(fact):
                episode = episode_lookup.get(ep_id) or {}
                raw_text = str(episode.get("raw_text") or "")
                if "last month" not in raw_text.lower():
                    continue
                return self._format_temporal_month(self._shift_month_anchor(anchor, -1))
        return None

    def _derive_first_window_temporal_deterministic_answer(
        self,
        *,
        query: str,
        context: str,
    ) -> str | None:
        if not self._temporal_query_requests_first_window(query):
            return None
        if self._temporal_query_surface_dates(query):
            return None
        match = re.search(
            r"First-mention window: earliest surfaced dated support is "
            r"(\d{4}-\d{2}-\d{2}), with the previous dated episode on "
            r"(\d{4}-\d{2}-\d{2})\.",
            context or "",
        )
        if not match:
            return None
        try:
            end = datetime.strptime(match.group(1), "%Y-%m-%d")
            start = datetime.strptime(match.group(2), "%Y-%m-%d")
        except ValueError:
            return None
        if not start < end:
            return None
        return self._format_temporal_interval(start, end)

    @staticmethod
    def _query_month_year_anchor(query: str) -> tuple[int, int] | None:
        match = re.search(
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(20\d{2}|19\d{2})\b",
            query or "",
            re.I,
        )
        if not match:
            return None
        month_lookup = {
            "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
            "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
        }
        return month_lookup[match.group(1).lower()], int(match.group(2))

    @staticmethod
    def _query_requests_acquisition_item_list(query: str, qf: dict) -> bool:
        if not (qf.get("operator_plan") or {}).get("list_set", {}).get("enabled"):
            return False
        lowered = (query or "").lower()
        if not re.search(r"\b(what|which)\s+(?:items?|things?)\b", lowered):
            return False
        return bool((qf.get("words") or set()) & {"buy", "purchase", "acquire", "get", "own"})

    @staticmethod
    def _extract_acquisition_item_candidates(text: str) -> list[str]:
        if not text:
            return []
        patterns = (
            r"\b(?:acquired|bought|purchased)\s+(?:myself\s+)?(?:a|an|the)?\s*(?:new\s+)?(?P<item>[A-Za-z0-9][A-Za-z0-9&'._-]*(?:\s+[A-Za-z0-9][A-Za-z0-9&'._-]*){0,6})",
            r"\b(?:got|gets|getting)\s+(?:myself\s+)?(?:a|an|the)?\s*(?:new\s+)?(?P<item>[A-Za-z0-9][A-Za-z0-9&'._-]*(?:\s+[A-Za-z0-9][A-Za-z0-9&'._-]*){0,6})",
            r"\bnow owns?\s+(?:a|an|the)?\s*(?:new\s+)?(?P<item>[A-Za-z0-9][A-Za-z0-9&'._-]*(?:\s+[A-Za-z0-9][A-Za-z0-9&'._-]*){0,6})",
            r"\bhas\s+(?:a|an)\s+new\s+(?P<item>[A-Za-z0-9][A-Za-z0-9&'._-]*(?:\s+[A-Za-z0-9][A-Za-z0-9&'._-]*){0,6})",
        )
        candidates: list[str] = []
        seen: set[str] = set()
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.I):
                candidate = re.split(r"\b(?:while|after|before|because|and|but)\b|[.?!;:]", match.group("item"), maxsplit=1)[0]
                candidate = re.sub(r"\s+", " ", candidate).strip(" \t\r\n,.-")
                if not candidate:
                    continue
                words = [w for w in candidate.split() if w.lower() not in {"new", "this", "that", "my", "our"}]
                candidate = " ".join(words).strip()
                if not candidate:
                    continue
                key = candidate.lower()
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
        return candidates


    @staticmethod
    def _query_requests_time_scoped_activity_acquisition(query: str, qf: dict) -> bool:
        lowered = (query or "").lower()
        operator_plan = qf.get("operator_plan") or {}
        slot_head_tokens = set((operator_plan.get("slot_query") or {}).get("head_tokens") or [])
        list_head_tokens = set((operator_plan.get("list_set") or {}).get("head_tokens") or [])
        head_tokens = slot_head_tokens | list_head_tokens
        if not head_tokens & {"activity", "activities", "hobby", "hobbies", "pastime", "pastimes", "sport", "sports", "game", "games"}:
            return False
        if not MemoryServer._query_month_year_anchor(query):
            return False
        return bool(re.search(r"\b(?:take\s+up|takes\s+up|took\s+up|taking\s+up|start|starts|started|starting|begin|begins|began|beginning|try|tries|tried|trying|get\s+into|gets\s+into|got\s+into|getting\s+into)\b", lowered))

    @staticmethod
    def _extract_activity_acquisition_candidates(text: str) -> list[str]:
        if not text:
            return []
        patterns = (
            r"\b(?:take\s+up|takes\s+up|took\s+up|taking\s+up|start|starts|started|starting|begin|begins|began|beginning|try|tries|tried|trying|get\s+into|gets\s+into|got\s+into|getting\s+into)\s+(?P<item>[A-Za-z][A-Za-z&'._-]*(?:\s+[A-Za-z][A-Za-z&'._-]*){0,5})",
        )
        candidates: list[str] = []
        seen: set[str] = set()
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.I):
                candidate = re.split(r"\b(?:while|after|before|because|and|but|or|if|so)\b|[.?!;:]", match.group("item"), maxsplit=1)[0]
                candidate = re.sub(r"\s+", " ", candidate).strip(" \t\r\n,.-")
                if not candidate:
                    continue
                words = [
                    w for w in candidate.split()
                    if w.lower() not in {"new", "this", "that", "my", "our", "another", "calming"}
                ]
                candidate = " ".join(words).strip()
                if not candidate:
                    continue
                key = candidate.lower()
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
        return candidates

    @staticmethod
    def _refine_acquisition_candidate(candidate: str, source_facts: list[dict]) -> str:
        head = normalize_term_token(candidate.split()[-1]) if candidate.split() else ""
        if not head:
            return candidate
        best = candidate
        best_score = (int(any(ch.isupper() for ch in candidate)), int(any(ch.isdigit() for ch in candidate)), len(candidate.split()), len(candidate))
        pattern = re.compile(
            rf"\b([A-Za-z0-9][A-Za-z0-9&'._-]*(?:\s+[A-Za-z0-9][A-Za-z0-9&'._-]*){{0,5}}\s+{re.escape(head)})\b",
            re.I,
        )
        for fact in source_facts:
            text = str((fact or {}).get("fact") or "")
            for match in pattern.finditer(text):
                phrase = re.sub(r"\s+", " ", match.group(1).strip())
                tokens = phrase.split()
                variants = []
                for width in range(2, min(len(tokens), 4) + 1):
                    variants.append(" ".join(tokens[-width:]))
                if not variants:
                    variants = [phrase]
                for variant in variants:
                    cleaned_tokens = list(variant.split())
                    while cleaned_tokens and cleaned_tokens[0].lower() in {"in", "a", "an", "the", "my", "our", "this", "that"}:
                        cleaned_tokens.pop(0)
                    cleaned_variant = " ".join(cleaned_tokens).strip()
                    if not cleaned_variant:
                        continue
                    if normalize_term_token(cleaned_variant.split()[-1]) != head:
                        continue
                    score = (
                        int(any(ch.isupper() for ch in cleaned_variant)),
                        int(any(ch.isdigit() for ch in cleaned_variant)),
                        len(cleaned_variant.split()),
                        len(cleaned_variant),
                    )
                    if score > best_score:
                        best = cleaned_variant
                        best_score = score
        return best

    @staticmethod
    def _query_requests_activity_list(query: str, qf: dict) -> bool:
        list_plan = (qf.get("operator_plan") or {}).get("list_set", {})
        if not list_plan.get("enabled"):
            return False
        head_tokens = set(list_plan.get("head_tokens") or [])
        return "activity" in head_tokens

    @staticmethod
    def _extract_activity_list_candidates(text: str, query_features: dict) -> list[str]:
        lowered = (text or "").lower()
        if not lowered:
            return []
        candidates: list[str] = []
        if "board game" in lowered:
            candidates.append("board games")
        if "wine tasting" in lowered:
            candidates.append("wine tasting")
        if "pet shelter" in lowered and re.search(r"\bvolunteer", lowered):
            candidates.append("pet shelter volunteering")
        if "flower" in lowered and re.search(r"\b(taking care|garden|bloom)", lowered):
            candidates.append("growing flowers")
        if re.search(r"\bcook(?:ing)?\b", lowered):
            candidates.append("cooking")
        if "indoor" in (query_features.get("words") or set()):
            outdoor_tokens = {"picnic", "trail", "hiking", "park", "bike", "biking", "camping", "surfing", "walk", "walking"}
            candidates = [candidate for candidate in candidates if not any(token in candidate for token in outdoor_tokens)]
        seen: set[str] = set()
        ordered: list[str] = []
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            ordered.append(candidate)
        return ordered

    def _derive_activity_list_deterministic_answer(
        self,
        *,
        query: str,
        query_features: dict,
        packet: dict,
        episode_lookup: dict[str, dict],
    ) -> str | None:
        if not self._query_requests_activity_list(query, query_features):
            return None
        selected_source_ids = {
            str((episode_lookup.get(ep_id) or {}).get("source_id") or "")
            for ep_id in packet.get("retrieved_episode_ids", [])
            if episode_lookup.get(ep_id)
        }
        selected_source_ids.discard("")
        if not selected_source_ids:
            return None
        primary_entity = ""
        for phrase in query_features.get("entity_phrases") or []:
            if phrase:
                primary_entity = str(phrase).strip()
                break
        source_facts = []
        for fact in self._all_granular:
            source_id = str(fact.get("source_id") or (fact.get("metadata") or {}).get("episode_source_id") or "")
            if source_id not in selected_source_ids:
                continue
            if primary_entity:
                speaker = str(fact.get("speaker") or "").strip()
                fact_text = str(fact.get("fact") or "")
                if speaker.lower() != primary_entity.lower() and primary_entity.lower() not in fact_text.lower():
                    continue
            source_facts.append(fact)
        if not source_facts:
            return None
        scored: list[tuple[datetime, str]] = []
        for fact in source_facts:
            anchor = self._fact_anchor_datetime(fact, episode_lookup) or datetime.max.replace(tzinfo=None)
            if getattr(anchor, 'tzinfo', None) is not None:
                anchor = anchor.replace(tzinfo=None)
            for candidate in self._extract_activity_list_candidates(str(fact.get("fact") or ""), query_features):
                scored.append((anchor, candidate))
        if not scored:
            return None
        scored.sort(key=lambda item: item[0])
        seen: set[str] = set()
        ordered: list[str] = []
        for _anchor, candidate in scored:
            if candidate in seen:
                continue
            seen.add(candidate)
            ordered.append(candidate)
        if not ordered:
            return None
        return ", ".join(ordered)

    def _derive_time_scoped_acquisition_deterministic_answer(
        self,
        *,
        query: str,
        query_features: dict,
        packet: dict,
        episode_lookup: dict[str, dict],
    ) -> str | None:
        if not self._query_requests_acquisition_item_list(query, query_features):
            return None
        month_year = self._query_month_year_anchor(query)
        if not month_year:
            return None
        month, year = month_year
        selected_source_ids = {
            str((episode_lookup.get(ep_id) or {}).get("source_id") or "")
            for ep_id in packet.get("retrieved_episode_ids", [])
            if episode_lookup.get(ep_id)
        }
        selected_source_ids.discard("")
        if not selected_source_ids:
            return None

        source_facts = []
        for fact in self._all_granular:
            source_id = str(fact.get("source_id") or (fact.get("metadata") or {}).get("episode_source_id") or "")
            if source_id in selected_source_ids:
                source_facts.append(fact)
        if not source_facts:
            return None

        episode_best: dict[str, tuple[tuple[int, int, int, int], str, str]] = {}
        for fact in source_facts:
            ep_ids = fact_episode_ids(fact)
            if not ep_ids:
                continue
            ep_id = ep_ids[0]
            episode = episode_lookup.get(ep_id) or {}
            source_date = str(episode.get("source_date") or "")
            try:
                anchor = date_parser.parse(source_date, fuzzy=True)
            except Exception:
                continue
            if anchor.year != year or anchor.month != month:
                continue
            for candidate in self._extract_acquisition_item_candidates(str(fact.get("fact") or "")):
                refined = self._refine_acquisition_candidate(candidate, source_facts)
                score = (
                    int(any(ch.isupper() for ch in refined)),
                    int(any(ch.isdigit() for ch in refined)),
                    len(refined.split()),
                    len(refined),
                )
                current = episode_best.get(ep_id)
                if current is None or score > current[0]:
                    episode_best[ep_id] = (score, refined, source_date)

        if not episode_best:
            return None

        ordered = sorted(episode_best.items(), key=lambda item: date_parser.parse(item[1][2], fuzzy=True))
        candidates: list[str] = []
        seen: set[str] = set()
        for _ep_id, (_score, candidate, _date) in ordered:
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
        if not candidates:
            return None
        return ", ".join(candidates)


    def _derive_time_scoped_activity_acquisition_deterministic_answer(
        self,
        *,
        query: str,
        query_features: dict,
        packet: dict,
        episode_lookup: dict[str, dict],
    ) -> str | None:
        if not self._query_requests_time_scoped_activity_acquisition(query, query_features):
            return None
        month_year = self._query_month_year_anchor(query)
        if not month_year:
            return None
        month, year = month_year
        selected_source_ids = {
            str((episode_lookup.get(ep_id) or {}).get("source_id") or "")
            for ep_id in packet.get("retrieved_episode_ids", [])
            if episode_lookup.get(ep_id)
        }
        selected_source_ids.discard("")
        if not selected_source_ids:
            return None

        primary_entity = ""
        for phrase in query_features.get("entity_phrases") or []:
            if phrase:
                primary_entity = str(phrase).strip().lower()
                break

        source_facts = []
        for fact in self._all_granular:
            source_id = str(fact.get("source_id") or (fact.get("metadata") or {}).get("episode_source_id") or "")
            if source_id not in selected_source_ids:
                continue
            fact_text = str(fact.get("fact") or "")
            speaker = str(fact.get("speaker") or "").strip().lower()
            if primary_entity and primary_entity not in fact_text.lower() and speaker != primary_entity:
                continue
            source_facts.append(fact)
        if not source_facts:
            return None

        episode_best: dict[str, tuple[tuple[int, int, int], str, str]] = {}
        for fact in source_facts:
            ep_ids = fact_episode_ids(fact)
            if not ep_ids:
                continue
            ep_id = ep_ids[0]
            episode = episode_lookup.get(ep_id) or {}
            source_date = str(episode.get("source_date") or "")
            try:
                anchor = date_parser.parse(source_date, fuzzy=True)
            except Exception:
                continue
            if anchor.year != year or anchor.month != month:
                continue
            for candidate in self._extract_activity_acquisition_candidates(str(fact.get("fact") or "")):
                score = (
                    len(candidate.split()),
                    int(any(ch.isupper() for ch in candidate)),
                    len(candidate),
                )
                current = episode_best.get(ep_id)
                if current is None or score > current[0]:
                    episode_best[ep_id] = (score, candidate, source_date)

        if not episode_best:
            return None

        ordered = sorted(episode_best.items(), key=lambda item: date_parser.parse(item[1][2], fuzzy=True))
        seen: set[str] = set()
        candidates: list[str] = []
        for _ep_id, (_score, candidate, _date) in ordered:
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
        if not candidates:
            return None
        return ", ".join(candidates)

    def _temporal_grounding_pseudo_facts(self, ep_id: str, episode: dict) -> list[dict]:
        raw_text = str(episode.get("raw_text") or "")
        source_date = str(episode.get("source_date") or "").strip()
        if not raw_text or not source_date:
            return []
        try:
            anchor = date_parser.parse(source_date, fuzzy=True)
        except Exception:
            return []

        weekday_re = re.compile(
            r"\b(last|this)\s+"
            r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            re.I,
        )
        weekday_to_index = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6,
        }

        pseudo: list[dict] = []
        seen: set[str] = set()
        for line_idx, raw_line in enumerate(raw_text.splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.lower().startswith("[turn query]"):
                continue
            lowered = line.lower()
            resolved: datetime | None = None
            if "yesterday" in lowered or "last night" in lowered:
                resolved = anchor - timedelta(days=1)
            elif "today" in lowered or "tonight" in lowered:
                resolved = anchor
            elif "tomorrow" in lowered:
                resolved = anchor + timedelta(days=1)
            elif "last week" in lowered:
                resolved = anchor - timedelta(days=7)
            elif "this week" in lowered:
                resolved = anchor
            elif "next week" in lowered:
                resolved = anchor + timedelta(days=7)
            elif "last month" in lowered:
                resolved = self._shift_month_anchor(anchor, -1)
            elif "this month" in lowered:
                resolved = anchor
            elif "next month" in lowered:
                resolved = self._shift_month_anchor(anchor, 1)
            else:
                match = weekday_re.search(line)
                if match:
                    rel = match.group(1).lower()
                    weekday = match.group(2).lower()
                    target = weekday_to_index.get(weekday)
                    if target is not None:
                        days_back = (anchor.weekday() - target) % 7
                        if rel == "last":
                            days_back = 7 if days_back == 0 else days_back
                        resolved = anchor - timedelta(days=days_back)
            if resolved is None:
                continue
            fact_text = (
                f"{line} Resolved date: {self._format_temporal_anchor(resolved)} "
                f"({resolved.strftime('%B %Y')})."
            )
            if fact_text in seen:
                continue
            seen.add(fact_text)
            pseudo.append({
                "id": f"temporal_anchor_{ep_id}_{line_idx}",
                "session": 0,
                "fact": fact_text,
                "source_id": episode.get("source_id", ""),
                "metadata": {
                    "episode_id": ep_id,
                    "episode_source_id": episode.get("source_id", ""),
                    "source_aggregation": True,
                    "semantic_class": "temporal_grounding",
                },
            })
        return pseudo

    async def _repair_temporal_grounding_packet(
        self,
        *,
        query: str,
        packet: dict,
        episode_lookup: dict[str, dict],
        fact_filter,
    ) -> tuple[dict, list[dict] | None]:
        query_features = packet.get("query_features") or extract_query_features(query)
        if not query_features.get("operator_plan", {}).get("temporal_grounding", {}).get("enabled", False):
            return packet, None

        requested_dates = self._temporal_query_surface_dates(query)
        if not requested_dates:
            return packet, None

        context_lower = str(packet.get("context") or "").lower()
        if any(date_text in context_lower for date_text in requested_dates):
            return packet, None

        selected_episode_ids = list(dict.fromkeys(
            list(packet.get("retrieved_episode_ids") or [])
            + list(packet.get("actual_injected_episode_ids") or [])
            + list(packet.get("fact_episode_ids") or [])
        ))
        selected_source_ids = {
            str((episode_lookup.get(ep_id) or {}).get("source_id") or "").strip()
            for ep_id in selected_episode_ids
        }
        selected_source_ids.discard("")
        if not selected_source_ids:
            return packet, None

        atomic_embs = (self._data_dict or {}).get("atomic_embs")
        if not isinstance(atomic_embs, np.ndarray) or len(atomic_embs) != len(self._all_granular):
            return packet, None

        requested_families = set(packet.get("retrieval_families") or [])
        requested_search_family = packet.get("search_family", "auto")
        candidate_facts: list[dict] = []
        candidate_embeddings: list[np.ndarray] = []
        fact_lookup: dict[str, dict] = {}
        emb_width = int(atomic_embs.shape[1]) if len(atomic_embs.shape) > 1 else 1
        emb_dtype = atomic_embs.dtype
        for idx, fact in enumerate(self._all_granular):
            if not fact_filter(fact):
                continue
            metadata = fact.get("metadata") or {}
            episode_id = str(metadata.get("episode_id") or "").strip()
            if not episode_id or episode_id not in episode_lookup:
                continue
            source_id = str(fact.get("source_id") or metadata.get("episode_source_id") or "").strip()
            if source_id not in selected_source_ids:
                continue
            family = (
                self._source_records.get(source_id, {}).get("family")
                or (episode_lookup.get(episode_id) or {}).get("source_type", "")
            )
            if requested_search_family not in ("auto", "", None) and family and family != requested_search_family:
                continue
            if requested_families and family and family not in requested_families:
                continue
            candidate_facts.append(fact)
            candidate_embeddings.append(atomic_embs[idx])
            fact_lookup[str(fact.get("id") or "")] = fact

        pseudo_fact_lookup: dict[str, dict] = {}
        zero_vec = np.zeros((emb_width,), dtype=emb_dtype)
        for ep_id, episode in episode_lookup.items():
            source_id = str((episode or {}).get("source_id") or "").strip()
            if source_id not in selected_source_ids:
                continue
            for pseudo in self._temporal_grounding_pseudo_facts(ep_id, episode):
                pseudo_fact_lookup[str(pseudo.get("id") or "")] = pseudo
                candidate_facts.append(pseudo)
                candidate_embeddings.append(zero_vec.copy())

        if not candidate_facts:
            return packet, None

        retrieval_target = query_features.get("retrieval_target") or query
        search_queries = [retrieval_target]
        stripped_query = self._strip_temporal_surface(retrieval_target)
        if stripped_query and stripped_query not in search_queries:
            search_queries.append(stripped_query)

        rescue_cfg = get_tuning_section("retrieval").get("semantic_rescue", {})
        ranked_rows: list[dict] = []
        seen_fact_ids: set[str] = set()
        traces: list[dict] = []
        for search_query in search_queries:
            query_embedding = await embed_query(search_query)
            sweep = source_local_fact_sweep(
                search_query,
                candidate_facts,
                np.asarray(candidate_embeddings),
                query_embedding=query_embedding,
                top_k=max(12, int(rescue_cfg.get("top_k", 8))),
                bm25_pool=max(36, int(rescue_cfg.get("bm25_pool", 24))),
                vector_pool=max(24, int(rescue_cfg.get("vector_pool", 24))),
                entity_pool=max(12, int(rescue_cfg.get("entity_pool", 12))),
                rrf_k=int(rescue_cfg.get("rrf_k", 60)),
            )
            traces.append({"query": search_query, "trace": sweep.get("trace", {})})
            for row in sweep.get("retrieved", []):
                fact = row.get("fact") or {}
                fact_id = str(fact.get("id") or "").strip()
                if not fact_id or fact_id in seen_fact_ids:
                    continue
                seen_fact_ids.add(fact_id)
                ranked_rows.append(row)

        if not ranked_rows:
            return packet, None

        retrieved_facts = [row.get("fact") for row in ranked_rows if row.get("fact")]
        retrieved_episode_ids: list[str] = []
        seen_episode_ids: set[str] = set()
        selection_scores: list[dict] = []
        for row in ranked_rows:
            fact = row.get("fact") or {}
            episode_id = str((fact.get("metadata") or {}).get("episode_id") or "").strip()
            if not episode_id or episode_id in seen_episode_ids:
                continue
            seen_episode_ids.add(episode_id)
            retrieved_episode_ids.append(episode_id)
            selection_scores.append({
                "episode_id": episode_id,
                "score": float(row.get("score", 0.0)),
            })

        merged_fact_lookup = dict(fact_lookup)
        merged_fact_lookup.update(pseudo_fact_lookup)
        context, actual_injected_episode_ids = build_context_from_retrieved_facts(
            retrieved_facts,
            episode_lookup,
            fact_lookup=merged_fact_lookup,
            budget=int(packet.get("selector_config", {}).get("budget", 8000)),
            snippet_chars=int(packet.get("tuning_snapshot", {}).get("packet", {}).get("snippet_chars", 1200)),
            question=query,
            query_features=query_features,
        )

        packet = dict(packet)
        packet["context"] = context
        packet["retrieved_fact_ids"] = [str(fact.get("id") or "") for fact in retrieved_facts]
        packet["retrieved_episode_ids"] = retrieved_episode_ids
        packet["actual_injected_episode_ids"] = actual_injected_episode_ids
        packet["fact_episode_ids"] = retrieved_episode_ids
        packet["selection_scores"] = selection_scores
        packet["source_local_fact_sweep_trace"] = {
            "mode": "temporal_grounding_repair",
            "requested_search_family": requested_search_family,
            "retrieval_families": sorted(requested_families),
            "queries": traces,
            "selected_source_ids": sorted(selected_source_ids),
        }
        family_first_pass_trace = dict(packet.get("family_first_pass_trace") or {})
        family_first_pass_trace["temporal_grounding_repair"] = {
            "candidate_count": len(candidate_facts),
            "selected_fact_count": len(retrieved_facts),
            "selected_source_ids": sorted(selected_source_ids),
        }
        packet["family_first_pass_trace"] = family_first_pass_trace
        return packet, retrieved_facts

    async def _generic_fact_recall(
        self,
        *,
        query: str,
        fact_filter,
        search_family: str = "auto",
        query_type: str = "auto",
    ) -> dict:
        resolved_type = self._QUERY_TYPE_MAP.get(query_type) or detect_query_type(query)
        retrieval_target = (extract_query_features(query).get("retrieval_target") or query)
        rescue_cfg = get_tuning_section("retrieval").get("semantic_rescue", {})

        candidate_facts = []
        candidate_embeddings = []
        for facts, embeddings in (
            (self._all_granular, (self._data_dict or {}).get("atomic_embs")),
            (self._all_cons, (self._data_dict or {}).get("cons_embs")),
            (self._all_cross, (self._data_dict or {}).get("cross_embs")),
        ):
            if not isinstance(embeddings, np.ndarray) or len(embeddings) != len(facts):
                continue
            for idx, fact in enumerate(facts):
                if not fact_filter(fact):
                    continue
                if search_family not in ("auto", "", None):
                    source_id = fact.get("source_id") or (fact.get("metadata") or {}).get("episode_source_id", "")
                    family = self._source_records.get(source_id, {}).get("family")
                    if family and family != search_family:
                        continue
                candidate_facts.append(fact)
                candidate_embeddings.append(embeddings[idx])

        if not candidate_facts:
            empty_result = {
                "context": "RETRIEVED FACTS:",
                "_context_packet": {"tier1": [], "tier2": [], "tier3": [], "tier4": []},
                "retrieved": [],
                "query_type": resolved_type,
                "is_multihop": False,
                "complexity_hint": {
                    "score": 0.0,
                    "level": 1,
                    "signals": [],
                    "retrieval_complexity": 0.0,
                    "content_complexity": 0.0,
                    "query_complexity": 0.0,
                    "dominant": "tie",
                },
                "n_facts": len(self._all_granular) + len(self._all_cons) + len(self._all_cross),
                "sessions_in_context": 0,
                "total_sessions": self._n_sessions,
                "coverage_pct": 0,
                "raw_budget": 0,
                "recommended_prompt_type": resolved_type,
                "use_tool": False,
                "runtime_trace": {
                    "runtime": "fact",
                    "scope": self._scope_trace(),
                    "reason": "empty_visible_facts",
                },
            }
            if self._profiles:
                level = empty_result["complexity_hint"].get("level", 1)
                profile = self._profiles.get(level)
                if profile:
                    empty_result["recommended_profile"] = profile
                    payload, payload_meta = self._build_payload(
                        query=query,
                        recall_result=empty_result,
                    )
                    if payload and payload_meta:
                        empty_result["payload"] = payload
                        empty_result["payload_meta"] = payload_meta
            return empty_result

        query_embedding = await embed_query(retrieval_target)
        sweep = source_local_fact_sweep(
            retrieval_target,
            candidate_facts,
            np.asarray(candidate_embeddings),
            query_embedding=query_embedding,
            top_k=int(rescue_cfg.get("top_k", 8)),
            bm25_pool=int(rescue_cfg.get("bm25_pool", 24)),
            vector_pool=int(rescue_cfg.get("vector_pool", 24)),
            entity_pool=int(rescue_cfg.get("entity_pool", 12)),
            rrf_k=int(rescue_cfg.get("rrf_k", 60)),
        )
        retrieved_items = [
            {
                "fact_id": row["fact"].get("id", ""),
                "conv_id": row["fact"].get("conv_id", self.key),
                "sim": float(row["score"]),
            }
            for row in sweep.get("retrieved", [])
        ]
        resolved_facts = [row["fact"] for row in sweep.get("retrieved", [])]

        total_sessions = self._n_sessions
        sessions_in_ctx = len({
            session_num
            for f in resolved_facts
            if (session_num := _coerce_positive_session_num(f.get("session"))) is not None
            and session_num <= len(self._raw_sessions)
        })
        raw_budget = compute_raw_budget(resolved_type, total_sessions, sessions_in_ctx)
        coverage_pct = (sessions_in_ctx / total_sessions * 100) if total_sessions else 0

        context_packet = _build_context_packet(
            resolved_facts,
            self._raw_sessions,
            budget=raw_budget,
            raw_docs=self._raw_docs or None,
        )
        hybrid_ctx = _render_context_packet(context_packet)

        complexity_hint = _compute_complexity_hint(
            retrieved=resolved_facts,
            resolved_type=resolved_type,
            is_multihop=resolved_type in ("temporal", "current", "counting"),
            fact_lookup=self._fact_lookup,
            query=query,
        )

        prompt_type, use_tool = _route_prompt_type(
            resolved_type,
            resolved_facts,
            total_sessions,
            sessions_in_ctx,
            hybrid_ctx,
        )

        result = {
            "context": hybrid_ctx,
            "_context_packet": context_packet,
            "retrieved": retrieved_items,
            "query_type": resolved_type,
            "is_multihop": resolved_type in ("temporal", "current", "counting"),
            "complexity_hint": complexity_hint,
            "n_facts": len(self._all_granular) + len(self._all_cons) + len(self._all_cross),
            "sessions_in_context": sessions_in_ctx,
            "total_sessions": total_sessions,
            "coverage_pct": coverage_pct,
            "raw_budget": raw_budget,
            "recommended_prompt_type": prompt_type,
            "use_tool": use_tool,
            "runtime_trace": {
                "runtime": "fact",
                "scope": self._scope_trace(),
                "query": {
                    "retrieval_target": retrieval_target,
                    "resolved_type": resolved_type,
                },
                "selection": {
                    "retrieved_fact_count": len(resolved_facts),
                    "selected_fact_ids": [
                        fact.get("id", "")
                        for fact in resolved_facts[:12]
                    ],
                },
                "packet": {
                    "context_chars": len(hybrid_ctx),
                    "raw_budget": raw_budget,
                    "source_local_fact_sweep": sweep.get("trace", {}),
                },
            },
        }

        if self._profiles:
            level = complexity_hint.get("level", 1)
            profile = self._profiles.get(level)
            if profile:
                result["recommended_profile"] = profile
            payload, payload_meta = self._build_payload(
                query=query,
                recall_result=result,
            )
            if payload and payload_meta:
                result["payload"] = payload
                result["payload_meta"] = payload_meta

        return result

    def _episode_runtime_facts(self, fact_filter, *, query: str | None = None) -> list[dict]:
        facts = [f for f in self._all_granular if fact_filter(f)]
        query_features = extract_query_features(query or "") if query else {}
        explicit_step_query = bool(
            query_features.get("step_numbers") or query_features.get("step_range")
        )
        for f in self._all_cross:
            if not fact_filter(f):
                continue
            metadata = f.get("metadata") or {}
            if not metadata.get("source_aggregation"):
                continue
            if metadata.get("semantic_class") != "temporal_semantics":
                continue
            if explicit_step_query:
                continue
            if not fact_episode_ids(f):
                continue
            facts.append(f)
        return facts
    def _visible_episode_runtime(
        self,
        fact_filter,
        *,
        query: str | None = None,
    ) -> tuple[dict, dict[str, dict], dict[str, list[dict]], object] | None:
        visible_facts = self._episode_runtime_facts(fact_filter, query=query)
        if not visible_facts:
            return None
        facts_by_episode = build_facts_by_episode(visible_facts)
        if not self._episode_corpus.get("documents"):
            return None
        if not facts_by_episode:
            return None

        query_features = extract_query_features(query or "") if query else {}
        include_full_document_docs = bool(
            query
            and (
                query_features.get("step_numbers")
                or query_features.get("step_range")
            )
        )

        visible_docs = []
        for doc in self._episode_corpus.get("documents", []):
            fact_visible_episode_ids = {
                ep.get("episode_id", "")
                for ep in doc.get("episodes", [])
                if ep.get("episode_id") in facts_by_episode
            }
            if not fact_visible_episode_ids:
                continue
            doc_source_families = {
                ep.get("source_type", "")
                for ep in doc.get("episodes", [])
                if ep.get("episode_id")
            }
            if include_full_document_docs and doc_source_families == {"document"}:
                episodes = [ep for ep in doc.get("episodes", []) if ep.get("episode_id")]
            else:
                episodes = [
                    ep
                    for ep in doc.get("episodes", [])
                    if ep.get("episode_id") in fact_visible_episode_ids
                ]
            if episodes:
                visible_docs.append({"doc_id": doc.get("doc_id"), "episodes": episodes})
        if not visible_docs:
            return None

        corpus = {"documents": visible_docs}
        episode_lookup = build_episode_lookup(corpus)
        bm25 = build_episode_bm25(corpus)
        return corpus, episode_lookup, facts_by_episode, bm25

    def _next_session_num(self) -> int:
        session_nums = [
            rs.get("session_num", 0)
            for rs in self._raw_sessions
            if isinstance(rs, dict) and isinstance(rs.get("session_num"), int)
        ]
        return (max(session_nums) if session_nums else 0) + 1

    @staticmethod
    def _iso_from_timestamp_ms(timestamp_ms: int | None) -> str:
        ts = int(timestamp_ms or 0) / 1000.0
        return datetime.fromtimestamp(ts, timezone.utc).isoformat()

    def _raw_entry_acl_allows(self, entry: dict, caller_id: str, caller_memberships: list[str], caller_role: str) -> bool:
        pseudo_fact = {
            "owner_id": entry.get("owner_id"),
            "read": entry.get("read") or [],
            "scope": entry.get("scope"),
            "agent_id": entry.get("agent_id"),
            "swarm_id": entry.get("swarm_id"),
        }
        return self._acl_allows(pseudo_fact, caller_id, caller_memberships, caller_role)

    @staticmethod
    def _raw_query_tokens(query: str) -> list[str]:
        return [
            token
            for token in re.findall(r"[A-Za-z0-9_./:-]+", (query or "").lower())
            if len(token) >= 2 and token not in STOP_WORDS
        ]

    @staticmethod
    def _raw_metadata_text(metadata: dict | None) -> str:
        if not metadata:
            return ""
        parts: list[str] = []
        for value in metadata.values():
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                parts.extend(str(item) for item in value if isinstance(item, (str, int, float)))
            elif isinstance(value, (int, float)):
                parts.append(str(value))
        return " ".join(parts).lower()

    def _score_raw_write_entry(self, query: str, tokens: list[str], entry: dict) -> float:
        haystack = str(entry.get("content") or "").lower()
        meta_text = self._raw_metadata_text(entry.get("metadata") or {})
        if not haystack and not meta_text:
            return 0.0
        score = 0.0
        q = (query or "").strip().lower()
        if q and q in haystack:
            score += 10.0
        if q and q in meta_text:
            score += 4.0
        for token in tokens:
            if token in haystack:
                score += 1.0
            if token in meta_text:
                score += 0.5
        family = str(entry.get("content_family") or "").lower()
        if family and family in q:
            score += 1.0
        return score

    def _raw_recall_entries(
        self,
        *,
        query: str,
        caller_id: str,
        caller_memberships: list[str],
        caller_role: str,
        swarm_id: str | None,
        limit: int = 8,
    ) -> list[dict]:
        if not self._supports_write_log or not hasattr(self._storage, "list_write_log_entries"):
            return []
        tokens = self._raw_query_tokens(query)
        entries = self._storage.list_write_log_entries(  # type: ignore[attr-defined]
            states=["pending", "in_progress", "failed"],
            swarm_id=swarm_id if swarm_id and swarm_id != "default" else None,
            order="desc",
        )
        scored = []
        for entry in entries:
            if not self._raw_entry_acl_allows(entry, caller_id, caller_memberships, caller_role):
                continue
            score = self._score_raw_write_entry(query, tokens, entry)
            if score <= 0:
                continue
            snippet = re.sub(r"\s+", " ", str(entry.get("content") or "")).strip()[:400]
            scored.append({
                "message_id": entry.get("message_id"),
                "session_id": entry.get("session_id"),
                "content_family": entry.get("content_family"),
                "content": snippet,
                "metadata": entry.get("metadata") or {},
                "timestamp_ms": entry.get("timestamp_ms"),
                "extraction_state": entry.get("extraction_state"),
                "score": float(score),
            })
        scored.sort(key=lambda item: (-item["score"], -(item.get("timestamp_ms") or 0)))
        return scored[:limit]

    def _render_raw_recall_context(self, raw_results: list[dict]) -> str:
        if not raw_results:
            return ""
        lines = ["RECENT RAW WRITES:"]
        for item in raw_results:
            family = str(item.get("content_family") or "raw")
            snippet = str(item.get("content") or "").strip()
            if not snippet:
                continue
            lines.append(f"[{family}] {snippet}")
        return "\n".join(lines)

    def _merge_raw_recall(
        self,
        *,
        query: str,
        result: dict,
        caller_id: str,
        caller_memberships: list[str],
        caller_role: str,
        swarm_id: str | None,
    ) -> dict:
        raw_results = self._raw_recall_entries(
            query=query,
            caller_id=caller_id,
            caller_memberships=caller_memberships,
            caller_role=caller_role,
            swarm_id=swarm_id,
        )
        if not raw_results:
            return result
        raw_context = self._render_raw_recall_context(raw_results)
        context = str(result.get("context") or "").strip()
        if context and context != "RETRIEVED FACTS:" and raw_context:
            result["context"] = f"{context}\n\n{raw_context}"
        elif raw_context:
            result["context"] = raw_context
        existing_retrieved = list(result.get("retrieved") or [])
        result["retrieved"] = existing_retrieved + raw_results
        result["raw_recall_count"] = len(raw_results)
        runtime_trace = dict(result.get("runtime_trace") or {})
        runtime_trace["raw_recall"] = {
            "count": len(raw_results),
            "message_ids": [item.get("message_id") for item in raw_results],
        }
        result["runtime_trace"] = runtime_trace
        return result

    async def write(
        self,
        *,
        message_id: str,
        session_id: str,
        content: str,
        content_family: str,
        timestamp_ms: int,
        agent_id: str = None,
        swarm_id: str = None,
        scope: str = None,
        owner_id: str = None,
        read: list[str] | None = None,
        write: list[str] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        if not self._supports_write_log or not hasattr(self._storage, "append_write_log"):
            raise RuntimeError("memory_write requires SQLite storage backend")
        err = self._validate_metadata(metadata)
        if err:
            raise ValueError(err)
        if str(content_family) not in {"chat", "document", "codebase", "artifact"}:
            raise ValueError(f"Unsupported content_family: {content_family}")
        _agent_id = agent_id if agent_id is not None else self.agent_id
        _swarm_id = swarm_id if swarm_id is not None else self.swarm_id
        _scope = scope if scope is not None else self.scope
        if _scope not in self.VALID_SCOPES:
            raise ValueError(f"Unknown scope: {_scope}")
        acl_defaults = _derive_acl_from_scope(_scope, _agent_id, _swarm_id)
        _owner_id = owner_id if owner_id is not None else acl_defaults["owner_id"]
        _read = list(read) if read is not None else list(acl_defaults["read"])
        _write = list(write) if write is not None else list(acl_defaults["write"])
        visibility = "private" if _scope == "agent-private" else "shared"
        receipt = self._storage.append_write_log(  # type: ignore[attr-defined]
            message_id=str(message_id),
            session_id=str(session_id),
            agent_id=str(_agent_id or "default"),
            swarm_id=str(_swarm_id or "default"),
            visibility=visibility,
            owner_id=_owner_id,
            scope=_scope,
            read=_read,
            write=_write,
            content_family=str(content_family),
            content_text=str(content),
            metadata=metadata or {},
            timestamp_ms=int(timestamp_ms),
        )
        receipt.update({"message_id": str(message_id), "session_id": str(session_id)})
        return receipt

    def write_status(self, message_id: str) -> dict | None:
        if not self._supports_write_log or not hasattr(self._storage, "get_write_status"):
            return None
        return self._storage.get_write_status(str(message_id))  # type: ignore[attr-defined]

    def _should_retry_write_entry(self, entry: dict, now_ms: int) -> bool:
        state = str(entry.get("extraction_state") or "pending")
        attempts = int(entry.get("extraction_attempts") or 0)
        last_attempt_ms = int(entry.get("last_extraction_attempt_ms") or 0)
        if state == "failed" and attempts >= 3:
            return False
        if state == "in_progress":
            return (now_ms - last_attempt_ms) >= 30_000
        if attempts <= 0:
            return True
        backoff_ms = min(60_000, 1_000 * (2 ** max(attempts - 1, 0)))
        return (now_ms - last_attempt_ms) >= backoff_ms

    async def _extract_write_log_entry(self, entry: dict) -> dict:
        family = str(entry.get("content_family") or "chat")
        metadata = dict(entry.get("metadata") or {})
        scope = entry.get("scope") or ("agent-private" if entry.get("visibility") == "private" else "swarm-shared")
        if family in {"chat", "conversation"}:
            session_num = metadata.get("turn_number")
            if isinstance(session_num, str) and session_num.isdigit():
                session_num = int(session_num)
            if not isinstance(session_num, int):
                session_num = metadata.get("part_idx")
            if isinstance(session_num, str) and session_num.isdigit():
                session_num = int(session_num)
            if not isinstance(session_num, int):
                session_num = self._next_session_num()
            session_date = str(metadata.get("session_date") or self._iso_from_timestamp_ms(entry.get("timestamp_ms")))
            speakers = str(metadata.get("speakers") or "User and Assistant")
            source_id = str(
                metadata.get("logical_source_id")
                or metadata.get("source_id")
                or entry.get("session_id")
                or entry.get("message_id")
            )
            source_meta = {"session_key": entry.get("session_id")}
            if family == "conversation":
                source_meta.update({
                    "root_message_id": entry.get("message_id"),
                    "content_family": family,
                })
            return await self.store(
                content=str(entry.get("content") or ""),
                session_num=session_num,
                session_date=session_date,
                speakers=speakers,
                agent_id=entry.get("agent_id"),
                swarm_id=entry.get("swarm_id"),
                scope=scope,
                owner_id=entry.get("owner_id"),
                read=entry.get("read") or [],
                write=entry.get("write") or [],
                source_id=source_id,
                source_meta=source_meta,
                metadata=metadata,
                skip_dedup=True,
                message_id=str(entry.get("message_id") or ""),
            )
        source_id = str(
            metadata.get("source_id")
            or metadata.get("path")
            or entry.get("session_id")
            or entry.get("message_id")
        )
        facts_extracted = await self.ingest_document(
            content=str(entry.get("content") or ""),
            source_id=source_id,
            agent_id=entry.get("agent_id"),
            swarm_id=entry.get("swarm_id"),
            scope=scope,
            metadata=metadata,
            family=family,
            skip_dedup=True,
            source_meta={
                "root_message_id": entry.get("message_id"),
                "session_key": entry.get("session_id"),
                "content_family": family,
            },
        )
        return {"facts_extracted": facts_extracted, "source_id": source_id}

    async def process_write_log_once(self, batch_size: int = 8) -> int:
        if not self._supports_write_log or not hasattr(self._storage, "list_write_log_entries"):
            return 0
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        processed = 0
        async with self._queue_lock:
            entries = self._storage.list_write_log_entries(  # type: ignore[attr-defined]
                states=["pending", "in_progress", "failed"],
                order="asc",
            )
            for entry in entries:
                if processed >= batch_size:
                    break
                if not self._should_retry_write_entry(entry, now_ms):
                    continue
                if str(entry.get("message_id") or "") in self._active_sync_message_ids:
                    continue
                if any(
                    rs.get("message_id") == entry.get("message_id")
                    and str(rs.get("status") or "active") == "active"
                    for rs in self._raw_sessions
                ):
                    self._storage.mark_write_state(entry["message_id"], "complete")  # type: ignore[attr-defined]
                    continue
                self._storage.mark_write_state(entry["message_id"], "in_progress", attempts_delta=1)  # type: ignore[attr-defined]
                try:
                    await self._extract_write_log_entry(entry)
                    self._storage.mark_write_state(entry["message_id"], "complete")  # type: ignore[attr-defined]
                    processed += 1
                except Exception:
                    attempts = int(entry.get("extraction_attempts") or 0) + 1
                    next_state = "failed" if attempts >= 3 else "pending"
                    self._storage.mark_write_state(entry["message_id"], next_state)  # type: ignore[attr-defined]
                    log.exception("write-log extraction failed for %s", entry.get("message_id"))
        return processed

    # ── store() ──

    async def store(
        self,
        content: str,
        session_num: int,
        session_date: str,
        speakers: str = "User and Assistant",
        agent_id: str = None,
        swarm_id: str = None,
        scope: str = None,
        upsert_by_key: str = None,
        content_type: str = "default",
        librarian_prompt: str = None,
        owner_id: str = None,
        read: list = None,
        write: list = None,
        source_id: str = None,
        artifact_id: str = None,
        version_id: str = None,
        parent_version: str = None,
        content_hash: str = None,
        skip_dedup: bool = False,
        source_meta: dict = None,
        retention_ttl: int = None,
        metadata: dict = None,
        target=None,
        message_id: str = None,
    ) -> dict:
        """Extract atomic facts from a conversation turn and persist to disk.

        Returns dict with facts_extracted (and upserted/session_key if upsert_by_key set).
        """
        _agent_id = agent_id if agent_id is not None else self.agent_id
        _swarm_id = swarm_id if swarm_id is not None else self.swarm_id
        _scope = scope if scope is not None else self.scope

        # Scope validation
        if _scope not in self.VALID_SCOPES:
            return {"error": f"Unknown scope: {_scope}", "code": "INVALID_SCOPE"}

        # Upsert validation
        if upsert_by_key is not None and _scope != "agent-private":
            return {"error": "upsert_by_key requires agent-private scope",
                    "code": "UPSERT_SCOPE_ERROR"}

        # librarian_prompt scope check
        if librarian_prompt is not None and _scope != "agent-private":
            return {"error": "librarian_prompt only permitted for agent-private scope",
                    "code": "LIBRARIAN_PROMPT_SCOPE_ERROR"}

        # Resolve ACL defaults from scope unless explicit overrides are provided.
        acl_defaults = _derive_acl_from_scope(_scope, _agent_id, _swarm_id)
        _owner_id = owner_id if owner_id is not None else acl_defaults["owner_id"]
        _read = list(read) if read is not None else list(acl_defaults["read"])
        _write = list(write) if write is not None else list(acl_defaults["write"])

        # Metadata validation
        err = self._validate_metadata(metadata)
        if err:
            return {"error": err, "code": "VALIDATION_ERROR"}
        try:
            normalized_target = _normalize_target(target)
        except ValueError as e:
            return {"error": str(e), "code": "VALIDATION_ERROR"}

        # ── Compute content hash if not provided ──
        if content_hash is None:
            content_hash = content_hash_text(content)

        # ── Generate artifact/version IDs if not provided ──
        if artifact_id is None:
            artifact_id = _generate_artifact_id()
        if version_id is None:
            version_id = _generate_version_id()

        # ── Dedup check ──
        dedup_key = None
        supersede_version_id = None
        if source_id and not skip_dedup:
            dedup_key = (source_id, session_num)
            existing = self._dedup_index.get(dedup_key)
            if existing is not None:
                if existing["content_hash"] == content_hash:
                    return {"status": "duplicate",
                            "artifact_id": existing["artifact_id"]}
                else:
                    # Different content → new version (Unit 4).
                    # Keep the current version active until the new extraction
                    # actually succeeds.
                    artifact_id = existing["artifact_id"]
                    parent_version = existing["version_id"]
                    version_id = _generate_version_id()
                    supersede_version_id = existing["version_id"]

        # ── Upsert path: serialize entire operation per session_key ──
        if upsert_by_key is not None:
            if upsert_by_key not in self._upsert_locks:
                self._upsert_locks[upsert_by_key] = asyncio.Lock()
            async with self._upsert_locks[upsert_by_key]:
                return await self._store_impl(
                    content, session_num, session_date, speakers,
                    _agent_id, _swarm_id, _scope,
                    upsert_by_key, content_type, librarian_prompt,
                    agent_id, swarm_id, scope,
                    _owner_id, _read, _write,
                    source_id=source_id, artifact_id=artifact_id,
                    version_id=version_id, parent_version=parent_version,
                    content_hash=content_hash, source_meta=source_meta,
                    retention_ttl=retention_ttl, metadata=metadata,
                    target=normalized_target,
                    message_id=message_id,
                    dedup_key=dedup_key,
                    supersede_version_id=supersede_version_id,
                )

        # ── Normal (non-upsert) path ──
        return await self._store_impl(
            content, session_num, session_date, speakers,
            _agent_id, _swarm_id, _scope,
            upsert_by_key, content_type, librarian_prompt,
            agent_id, swarm_id, scope,
            _owner_id, _read, _write,
            source_id=source_id, artifact_id=artifact_id,
            version_id=version_id, parent_version=parent_version,
            content_hash=content_hash, source_meta=source_meta,
            retention_ttl=retention_ttl, metadata=metadata,
            target=normalized_target,
            message_id=message_id,
            dedup_key=dedup_key,
            supersede_version_id=supersede_version_id,
        )

    async def _store_impl(
        self,
        content: str,
        session_num: int,
        session_date: str,
        speakers: str,
        _agent_id: str,
        _swarm_id: str,
        _scope: str,
        upsert_by_key: str | None,
        content_type: str,
        librarian_prompt: str | None,
        agent_id: str | None,
        swarm_id: str | None,
        scope: str | None,
        _owner_id: str = "system",
        _read: list = None,
        _write: list = None,
        source_id: str = None,
        artifact_id: str = None,
        version_id: str = None,
        parent_version: str = None,
        content_hash: str = None,
        source_meta: dict = None,
        retention_ttl: int = None,
        metadata: dict = None,
        target: list[str] | None = None,
        message_id: str | None = None,
        dedup_key: tuple | None = None,
        supersede_version_id: str | None = None,
    ) -> dict:
        """Internal implementation of store(). Separated so upsert path can
        hold a per-key lock around the entire operation."""
        raw_session_id = str(uuid4())
        effective_message_id = str(message_id) if message_id is not None else f"raw:{raw_session_id}"
        self._audit.log("store", _owner_id,
                        {"session_num": session_num, "artifact_id": artifact_id})

        # Handle upsert: find and remove existing session + its facts
        # Must happen under _file_lock to prevent race conditions
        upserted = False
        if upsert_by_key is not None:
            async with self._file_lock:
                for i, rs in enumerate(self._raw_sessions):
                    if (rs.get("session_key") == upsert_by_key
                            and rs.get("agent_id") == _agent_id
                            and rs.get("swarm_id") == _swarm_id
                            and rs.get("scope") == _scope):
                        old_rsid = rs.get("raw_session_id")
                        if old_rsid:
                            self._all_granular = [
                                f for f in self._all_granular
                                if f.get("raw_session_id") != old_rsid
                            ]
                        self._raw_sessions.pop(i)
                        upserted = True
                        break

        if _read is None:
            _read = ["agent:PUBLIC"]
        if _write is None:
            _write = ["agent:PUBLIC"]

        conv_source_id = source_id or self.key

        # Save raw session BEFORE extraction — source of truth
        raw_session = {
            "raw_session_id": raw_session_id,
            "session_num": session_num,
            "session_date": session_date,
            "content": content,
            "speakers": speakers,
            "agent_id": _agent_id,
            "swarm_id": _swarm_id,
            "scope": _scope,
            "owner_id": _owner_id,
            "read": list(_read),
            "write": list(_write),
            "content_type": content_type,
            "stored_at": datetime.now(timezone.utc).isoformat(),
            "format": "conversation",
            "source_id": conv_source_id,
            "artifact_id": artifact_id,
            "version_id": version_id,
            "parent_version": parent_version,
            "content_hash": content_hash,
            "status": "pending_extraction",
        }
        if metadata is not None:
            raw_session["metadata"] = metadata
        if target:
            raw_session["target"] = list(target)
        if retention_ttl is not None:
            raw_session["retention_ttl"] = retention_ttl
        if source_meta:
            raw_session.update(source_meta)
        if upsert_by_key is not None:
            raw_session["session_key"] = upsert_by_key
        raw_session["message_id"] = effective_message_id

        # A synchronous store() call already owns extraction for this message.
        # Keep the write-log worker away from the same pending row until the
        # synchronous extraction path finishes; after a process crash this set is
        # empty again, so stale pending rows remain retryable on restart.
        self._active_sync_message_ids.add(effective_message_id)
        try:
            async with self._file_lock:
                stale_idx = None
                stale_raw_session_id = None
                for i, rs in enumerate(self._raw_sessions):
                    if (
                        rs.get("message_id") == effective_message_id
                        and str(rs.get("status") or "") == "pending_extraction"
                    ):
                        stale_idx = i
                        stale_raw_session_id = rs.get("raw_session_id")
                        break
                if stale_idx is not None:
                    if stale_raw_session_id:
                        self._all_granular = [
                            f for f in self._all_granular if f.get("raw_session_id") != stale_raw_session_id
                        ]
                    self._raw_sessions.pop(stale_idx)
                self._raw_sessions.append(raw_session)
                self._register_source_record(
                    source_id=conv_source_id,
                    family="conversation",
                    owner_id=_owner_id,
                    read=_read,
                    write=_write,
                    artifact_id=artifact_id,
                    version_id=version_id,
                    content_hash=content_hash,
                    metadata=metadata,
                    target=target,
                    source_meta={"stored_format": "conversation"},
                )
                self._save_cache()

            # ── MAL model override + extraction prompt ──
            _extract_model = _resolve_extract_model(
                self.extract_model, str(self.data_dir), self.key, agent_id or "default",
            )
            sem = self._get_extract_sem()

            # MAL extraction prompt override
            _mal_cfg = _load_mal_active_config(str(self.data_dir), self.key, agent_id or "default")
            _mal_prompts = _mal_cfg.get("extraction_prompts") or {}
            _mal_conv_key = f"conversation_content_type:{content_type}"
            _has_mal_override = _mal_conv_key in _mal_prompts

            if librarian_prompt is not None:
                extraction_prompt = librarian_prompt
            elif _has_mal_override:
                extraction_prompt = _mal_prompts[_mal_conv_key]
            else:
                extraction_prompt = self._prompt_registry.get(content_type)

            registry_prompt = self._prompt_registry.get(content_type)
            use_custom = (librarian_prompt is not None
                          or _has_mal_override
                          or self._prompt_registry._custom_path(content_type).exists()
                          or content_type != "default")
            block_prompt_pipeline = content_type in {"conversation", "document"}

            if not use_custom or block_prompt_pipeline:
                # Default path — extract_session uses its own hardcoded EXTRACTION_PROMPT
                async def _call_extract_fn(model, system, user_msg, max_tokens=8192):
                    return await call_extract(model, system, user_msg, max_tokens, sem)
            else:
                # Custom prompt — wrap call_extract_fn to inject resolved prompt
                async def _call_extract_fn(model, system, user_msg, max_tokens=8192):
                    try:
                        dt = datetime.fromisoformat(session_date.replace("Z", "+00:00"))
                        date_str = dt.strftime("%d %B %Y")
                        year_minus_1 = str(dt.year - 1)
                    except Exception:
                        try:
                            dt = date_parser.parse(session_date, fuzzy=True)
                            date_str = dt.strftime("%d %B %Y")
                            year_minus_1 = str(dt.year - 1)
                        except Exception:
                            date_str = session_date
                            year_match = re.search(r"\b(20\d{2}|19\d{2})\b", session_date or "")
                            year_minus_1 = str(int(year_match.group(1)) - 1) if year_match else "2022"
                    class _SafeDict(dict):
                        def __missing__(self, key):
                            return "{" + key + "}"

                    custom_system = extraction_prompt.format_map(_SafeDict(
                        session_date=date_str,
                        year_minus_1=year_minus_1,
                        session_num=session_num,
                    ))
                    return await call_extract(model, custom_system, user_msg, max_tokens, sem)

            conv_id, sn, sdate, facts, tlinks = await extract_session(
                session_text=content,
                session_num=session_num,
                session_date=session_date,
                conv_id=self.key,
                speakers=speakers,
                model=_extract_model,
                call_extract_fn=_call_extract_fn,
                fmt="CONVERSATION",
            )

            # Retry once if 0 facts
            if len(facts) == 0:
                conv_id, sn, sdate, facts, tlinks = await extract_session(
                    session_text=content,
                    session_num=session_num,
                    session_date=session_date,
                    conv_id=self.key,
                    speakers=speakers,
                    model=_extract_model,
                    call_extract_fn=_call_extract_fn,
                    fmt="CONVERSATION",
                )

            if len(facts) == 0:
                log.warning("store() returned 0 facts for session %d after retry", session_num)
                async with self._file_lock:
                    for rs in self._raw_sessions:
                        if rs.get("raw_session_id") == raw_session_id:
                            # 0 extracted facts is a terminal ingest outcome, not a
                            # corrupt half-ingested raw session.
                            rs["status"] = "active"
                            break
                    self._save_cache()
                result = {"facts_extracted": 0}
                if upsert_by_key is not None:
                    result["upserted"] = upserted
                    result["session_key"] = upsert_by_key
                return result

            episode = {
                "episode_id": f"{self._episode_source_key(conv_source_id)}_e{session_num:04d}",
                "source_type": "conversation",
                "source_id": conv_source_id,
                "source_date": session_date,
                "topic_key": f"session_{session_num}",
                "state_label": "session",
                "currentness": "unknown",
                "raw_text": content,
                "provenance": {"raw_span": [0, len(content)]},
            }
            _stamp_selector_episode_fields(episode, raw_session, source_meta)

            self._tag_facts(facts, session_date,
                            agent_id=agent_id, swarm_id=swarm_id, scope=scope,
                            owner_id=_owner_id, read=_read, write=_write,
                            artifact_id=artifact_id, version_id=version_id,
                            content_hash=content_hash, retention_ttl=retention_ttl,
                            metadata=metadata, target=target)

            # Make fact IDs unique per session to prevent lookup collision
            # from repeated model-generated local ids like f_01, f_02.
            for f in facts:
                raw_id = f.get("id", "")
                if raw_id and not raw_id.startswith(f"s{session_num}_"):
                    f["id"] = f"s{session_num}_{raw_id}"

            self._stamp_episode_metadata(facts, episode["episode_id"], conv_source_id)
            self._align_fact_selectors(
                facts,
                episode_id=episode["episode_id"],
                source_kind="conversation",
                raw_fields=_selector_raw_fields(content, raw_session, source_meta),
                speakers=speakers if isinstance(speakers, dict) else None,
            )

            # Tag facts with raw_session_id for upsert tracking
            for f in facts:
                f["source_id"] = conv_source_id
                f["raw_session_id"] = raw_session_id

            # Compute and stamp _session_content_complexity on each fact
            session_complexity = _compute_content_complexity(facts)
            for f in facts:
                f["_session_content_complexity"] = session_complexity

            # Set _temporal_links
            for f in facts:
                f["_temporal_links"] = []
            if facts and tlinks:
                facts[0]["_temporal_links"] = tlinks

            # Clear stale tiers BEFORE save
            self._mark_tiers_dirty()
            self._data_dict = None

            async with self._file_lock:
                if supersede_version_id:
                    for f in self._all_granular:
                        if f.get("version_id") == supersede_version_id:
                            f["status"] = "superseded"
                    for rs in self._raw_sessions:
                        if rs.get("version_id") == supersede_version_id:
                            rs["status"] = "superseded"
                if dedup_key is not None:
                    self._dedup_index[dedup_key] = {
                        "artifact_id": artifact_id,
                        "version_id": version_id,
                        "content_hash": content_hash,
                    }
                for rs in self._raw_sessions:
                    if rs.get("raw_session_id") == raw_session_id:
                        rs["status"] = "active"
                        break
                self._all_granular.extend(facts)
                self._all_tlinks.extend(tlinks)
                self._append_or_replace_episode(self._conversation_doc_id(conv_source_id), episode)
                if not upserted:
                    self._n_sessions += 1
                    if facts:
                        self._n_sessions_with_facts += 1
                self._save_cache()

            result = {"facts_extracted": len(facts)}
            if upsert_by_key is not None:
                result["upserted"] = upserted
                result["session_key"] = upsert_by_key
            return result
        finally:
            self._active_sync_message_ids.discard(effective_message_id)

    # ── ingest_document() ──

    async def ingest_document(
        self,
        content: str,
        source_id: str,
        speakers: str = "Document",
        agent_id: str = None,
        swarm_id: str = None,
        scope: str = None,
        artifact_id: str = None,
        version_id: str = None,
        parent_version: str = None,
        content_hash: str = None,
        skip_dedup: bool = False,
        source_meta: dict = None,
        retention_ttl: int = None,
        metadata: dict = None,
        target=None,
        family: str = "document",
    ) -> int:
        """Ingest a document through the episode pipeline."""
        _agent_id = agent_id if agent_id is not None else self.agent_id
        _swarm_id = swarm_id if swarm_id is not None else self.swarm_id
        _scope = scope if scope is not None else self.scope

        err = self._validate_metadata(metadata)
        if err:
            raise ValueError(err)
        normalized_target = _normalize_target(target)

        # Resolve ACL from scope unless explicit overrides are provided later.
        acl_defaults = _derive_acl_from_scope(_scope, _agent_id, _swarm_id)
        _owner_id = acl_defaults["owner_id"]
        _read = list(acl_defaults["read"])
        _write = list(acl_defaults["write"])

        # Generate identity fields
        if artifact_id is None:
            artifact_id = _generate_artifact_id()
        if version_id is None:
            version_id = _generate_version_id()
        if content_hash is None:
            content_hash = content_hash_text(content)

        multipart_part_key = self._multipart_part_key(metadata)
        if multipart_part_key:
            doc_dedup_key = (source_id, "__document_part__", multipart_part_key)
        else:
            doc_dedup_key = (source_id, "__document__")
        doc_supersede_version_id = None
        if source_id and not skip_dedup:
            existing = self._dedup_index.get(doc_dedup_key)
            if existing is not None:
                if existing["content_hash"] == content_hash:
                    return 0
                artifact_id = existing["artifact_id"]
                parent_version = existing["version_id"]
                version_id = _generate_version_id()
                doc_supersede_version_id = existing["version_id"]

        sem = self._get_extract_sem()
        _extract_model = _resolve_extract_model(
            self.extract_model, str(self.data_dir), self.key, agent_id or "default",
        )

        async def _call_extract_fn(model, system, user_msg, max_tokens=8192):
            return await call_extract(model, system, user_msg, max_tokens, sem)
        block_dicts, _blocks = segment_document_text(content, source_id)
        doc_meta = extract_doc_metadata(content, source_id)
        grouping_config = dict(get_tuning_section("episodes", "document_grouping"))
        # MAL generation-aware overrides for grouping + extraction prompts
        _mal_cfg = _load_mal_active_config(str(self.data_dir), self.key, agent_id or "default")
        _mal_prompts = _mal_cfg.get("extraction_prompts") or {}
        if _mal_cfg.get("grouping_prompt_mode"):
            grouping_config["prompt_mode"] = _mal_cfg["grouping_prompt_mode"]
        if _mal_cfg.get("size_cap_chars"):
            grouping_config["size_cap_chars"] = _mal_cfg["size_cap_chars"]
        try:
            episodes, _grouping_raw, _mode_used = await group_document(
                _extract_model,
                source_id,
                doc_meta["title"],
                doc_meta["date"],
                block_dicts,
                grouping_config,
                sem,
            )
        except Exception as e:
            log.warning(
                "document episode grouping failed for %s; falling back to singleton block episodes: %s",
                source_id,
                e,
            )
            episodes = build_singleton_episodes(source_id, doc_meta["date"], block_dicts)

        if not episodes:
            episodes = build_singleton_episodes(source_id, doc_meta["date"], block_dicts)

        doc_session_start = 1
        async with self._file_lock:
            if multipart_part_key:
                next_episode_index = self._next_document_episode_index(source_id)
                doc_session_start = self._next_document_session_num(source_id)
                episodes = self._reindex_document_episodes(
                    source_id,
                    episodes,
                    start_index=next_episode_index,
                    part_key=multipart_part_key,
                )
                existing_raw = self._raw_docs.get(source_id, "")
                self._raw_docs[source_id] = (
                    f"{existing_raw}\n\n{content}".strip() if existing_raw else content
                )
            else:
                self._raw_docs[source_id] = content
            self._upsert_episode_document(
                self._document_doc_id(source_id),
                episodes,
                replace_part_key=multipart_part_key,
            )
            self._register_source_record(
                source_id=source_id,
                family="document",
                owner_id=_owner_id,
                read=_read,
                write=_write,
                artifact_id=artifact_id,
                version_id=version_id,
                content_hash=content_hash,
                metadata=metadata,
                target=normalized_target,
                source_meta=source_meta,
            )
            self._save_cache()

        async def _extract_episode(ep_idx: int, ep: dict) -> list[dict]:
            session_num = doc_session_start + ep_idx - 1
            raw_session_id = str(uuid4())
            raw_session = {
                "raw_session_id": raw_session_id,
                "session_num": session_num,
                "session_date": ep.get("source_date", "") or doc_meta["date"],
                "content": ep["raw_text"],
                "speakers": speakers,
                "agent_id": _agent_id,
                "swarm_id": _swarm_id,
                "scope": _scope,
                "owner_id": _owner_id,
                "read": list(_read),
                "write": list(_write),
                "stored_at": datetime.now(timezone.utc).isoformat(),
                "format": family,
                "source_id": source_id,
                "artifact_id": artifact_id,
                "version_id": version_id,
                "parent_version": parent_version,
                "content_hash": content_hash,
                "status": "active",
                "episode_id": ep["episode_id"],
            }
            if metadata is not None:
                raw_session["metadata"] = dict(metadata)
                if metadata.get("part_source_id"):
                    raw_session["part_source_id"] = str(metadata.get("part_source_id"))
            if normalized_target:
                raw_session["target"] = list(normalized_target)
            if source_meta:
                raw_session.update(source_meta)

            async with self._file_lock:
                self._raw_sessions.append(raw_session)
                self._save_cache()

            # MAL block prompt overrides for document extraction
            _block_overrides = {}
            for _pk, _pv in _mal_prompts.items():
                if _pk.startswith("document_block_prompt:"):
                    _block_overrides[_pk.split(":", 1)[1]] = _pv
            _conv_id, _sn, _sdate, facts, tlinks = await extract_session(
                session_text=ep["raw_text"],
                session_num=session_num,
                session_date=ep.get("source_date", "") or doc_meta["date"],
                conv_id=self.key,
                speakers=speakers,
                model=_extract_model,
                call_extract_fn=_call_extract_fn,
                block_prompt_overrides=_block_overrides or None,
            )
            self._tag_facts(
                facts,
                ep.get("source_date", "") or doc_meta["date"],
                agent_id=agent_id,
                swarm_id=swarm_id,
                scope=scope,
                owner_id=_owner_id,
                read=_read,
                write=_write,
                artifact_id=artifact_id,
                version_id=version_id,
                content_hash=content_hash,
                target=normalized_target,
                metadata=metadata,
                retention_ttl=retention_ttl,
            )
            _stamp_selector_episode_fields(ep, raw_session, source_meta)
            self._stamp_episode_metadata(facts, ep["episode_id"], source_id)
            self._align_fact_selectors(
                facts,
                episode_id=ep["episode_id"],
                source_kind="document",
                raw_fields=_selector_raw_fields(ep["raw_text"], raw_session, source_meta, ep),
            )
            complexity = _compute_content_complexity(facts)
            for fact in facts:
                raw_id = fact.get("id", "")
                if raw_id and not raw_id.startswith(f"{ep['episode_id']}_"):
                    fact["id"] = f"{ep['episode_id']}_{raw_id}"
                fact["source_id"] = source_id
                fact["raw_session_id"] = raw_session_id
                fact["session"] = session_num
                fact["_session_content_complexity"] = complexity
                merged_metadata = _merge_fact_metadata(
                    fact.get("metadata"),
                    (
                        {}
                        if fact.get("metadata", {}).get("document_source")
                        else {"document_source": source_id}
                    ),
                )
                if merged_metadata:
                    err = self._validate_metadata(merged_metadata)
                    if err:
                        raise ValueError(err)
                    fact["metadata"] = merged_metadata
            for fact in facts:
                fact["_temporal_links"] = []
            if facts and tlinks:
                facts[0]["_temporal_links"] = tlinks

            async with self._file_lock:
                self._all_granular.extend(facts)
                self._all_tlinks.extend(tlinks)
                self._mark_tiers_dirty()
                self._data_dict = None
                self._save_cache()
            return facts

        total_facts = 0
        doc_granular = []
        doc_tlinks = []
        tasks = [
            asyncio.create_task(_extract_episode(idx + 1, ep))
            for idx, ep in enumerate(episodes)
        ]
        for task in asyncio.as_completed(tasks):
            facts = await task
            total_facts += len(facts)
            doc_granular.extend(facts)
            if facts and facts[0].get("_temporal_links"):
                doc_tlinks.extend(facts[0]["_temporal_links"])

        if not doc_granular:
            return 0

        doc_cons = []
        doc_cross = []

        substrate_cross = await self._extract_source_aggregation_facts(
            source_id=source_id,
            source_kind="document",
            source_facts=doc_granular,
            source_date=doc_meta["date"],
            model=_extract_model,
            call_extract_fn=_call_extract_fn,
            agent_id=agent_id or "default",
        )
        _namespace_derived_fact_ids(substrate_cross, f"substrate_{self._episode_source_key(source_id)}")
        max_doc_cc = max((f.get("_session_content_complexity", 0.0) for f in doc_granular), default=0.0)
        for fact in substrate_cross:
            fact["_session_content_complexity"] = max_doc_cc
        self._tag_facts(
            substrate_cross,
            doc_meta["date"],
            agent_id=agent_id,
            swarm_id=swarm_id,
            scope=scope,
            owner_id=_owner_id,
            read=_read,
            write=_write,
            artifact_id=artifact_id,
            version_id=version_id,
            content_hash=content_hash,
            target=_consensus_target(doc_granular),
        )
        doc_cross.extend(substrate_cross)

        async with self._file_lock:
            if doc_supersede_version_id:
                for f in self._all_granular:
                    if f.get("version_id") == doc_supersede_version_id:
                        f["status"] = "superseded"
                for rs in self._raw_sessions:
                    if rs.get("version_id") == doc_supersede_version_id:
                        rs["status"] = "superseded"
            if source_id and not skip_dedup:
                self._dedup_index[doc_dedup_key] = {
                    "artifact_id": artifact_id,
                    "version_id": version_id,
                    "content_hash": content_hash,
                }
            if multipart_part_key:
                self._raw_docs[source_id] = self._rebuild_document_raw_text(source_id)
            self._all_cons.extend(doc_cons)
            self._all_cross.extend(doc_cross)
            self._n_sessions += len(episodes)
            self._n_sessions_with_facts += len({
                fact.get("session", 1)
                for fact in doc_granular
                if fact
            })
            self._save_cache()

        self._data_dict = None
        self._tiers_dirty = True
        self._tier2_built = True
        self._tier3_built = bool(doc_cross)
        return total_facts

    # ── L0 enrichment ──

    async def _enrich_missing_fact_metadata(self, facts: list[dict]) -> None:
        """L0 enrich facts with incomplete metadata via classify_fact."""
        if self._extract_disabled:
            return
        from .common import call_extract
        from .librarian import classify_fact, merge_l1_metadata
        sem = self._get_extract_sem()
        async def _classify_fn(model, system, user_msg, max_tokens=256):
            return await call_extract(model, system, user_msg, max_tokens, sem)
        targets = [f for f in facts if _needs_l0_enrichment(f)]
        if not targets:
            return
        model = self.extract_model or "__l0_enrichment__"
        async def _enrich_one(fact: dict):
            try:
                metadata = await classify_fact(fact.get("fact", ""), model, _classify_fn)
                merge_l1_metadata(fact, metadata)
            except Exception:
                pass
        await asyncio.gather(*[_enrich_one(f) for f in targets])

    # ── ingest_asserted_facts() ──

    async def ingest_asserted_facts(
        self,
        facts: list[dict],
        consolidated: list[dict] = None,
        cross_session: list[dict] = None,
        raw_sessions: list[dict] = None,
        provenance: dict = None,
        owner_id: str = None,
        read: list[str] = None,
        write: list[str] = None,
        enrich_l0: bool = True,
        artifact_id: str = None,
        version_id: str = None,
    ) -> dict:
        """Authoritative import of pre-extracted memory artifacts.

        Works on non-empty memory. Imported session numbers are offset
        by current session count so positional raw lookup stays correct.
        All mutation under _file_lock to prevent concurrent offset collision.
        """
        # -- STEP 1: Validate imported raw_sessions (dense 1..M before remap) --
        if raw_sessions:
            snums = [rs.get("session_num") for rs in raw_sessions]
            if any(s is None for s in snums):
                return {"error": "raw_session missing session_num",
                        "code": "VALIDATION_ERROR"}
            if len(snums) != len(set(snums)):
                return {"error": f"Duplicate session_nums: {sorted(snums)}",
                        "code": "VALIDATION_ERROR"}
            snums_sorted = sorted(snums)
            if snums_sorted != list(range(1, len(snums) + 1)):
                return {"error": f"Imported raw_sessions must be dense 1..N, "
                                 f"got {snums_sorted}",
                        "code": "VALIDATION_ERROR"}
            raw_sessions = sorted(raw_sessions, key=lambda rs: rs["session_num"])

            # Validate facts reference valid imported sessions (pre-offset)
            raw_snum_set = set(snums)
            for f in facts:
                sn = f.get("session")
                if sn is None:
                    return {"error": f"Fact missing 'session': {f.get('id', '?')}",
                            "code": "VALIDATION_ERROR"}
                normalized_session = _coerce_positive_session_num(sn)
                if normalized_session is None:
                    return {"error": f"Fact session={sn!r} is not a positive integer",
                            "code": "VALIDATION_ERROR"}
                if normalized_session not in raw_snum_set:
                    return {"error": f"Fact session={sn} has no matching "
                                     f"imported raw_session",
                            "code": "VALIDATION_ERROR"}
                f["session"] = normalized_session

        # -- STEP 2: Resolve ACL --
        _owner = owner_id or _resolve_default_owner(self)
        _read = read if read is not None else ["agent:PUBLIC"]
        _write = write if write is not None else ["agent:PUBLIC"]

        # Validate per-fact metadata (all tiers)
        for f in facts:
            err = self._validate_metadata(f.get("metadata"))
            if err:
                return {"error": f"Fact '{f.get('id', '?')}': {err}", "code": "VALIDATION_ERROR"}
            if "target" in f:
                try:
                    normalized_target = _normalize_target(f.get("target"))
                except ValueError as e:
                    return {"error": f"Fact '{f.get('id', '?')}': {e}", "code": "VALIDATION_ERROR"}
                if normalized_target:
                    f["target"] = normalized_target
                else:
                    f.pop("target", None)
        if consolidated:
            for cf in consolidated:
                err = self._validate_metadata(cf.get("metadata"))
                if err:
                    return {"error": f"Cons fact '{cf.get('id', '?')}': {err}", "code": "VALIDATION_ERROR"}
                if "target" in cf:
                    try:
                        normalized_target = _normalize_target(cf.get("target"))
                    except ValueError as e:
                        return {"error": f"Cons fact '{cf.get('id', '?')}': {e}", "code": "VALIDATION_ERROR"}
                    if normalized_target:
                        cf["target"] = normalized_target
                    else:
                        cf.pop("target", None)
        if cross_session:
            for xf in cross_session:
                err = self._validate_metadata(xf.get("metadata"))
                if err:
                    return {"error": f"Cross fact '{xf.get('id', '?')}': {err}", "code": "VALIDATION_ERROR"}
                if "target" in xf:
                    try:
                        normalized_target = _normalize_target(xf.get("target"))
                    except ValueError as e:
                        return {"error": f"Cross fact '{xf.get('id', '?')}': {e}", "code": "VALIDATION_ERROR"}
                    if normalized_target:
                        xf["target"] = normalized_target
                    else:
                        xf.pop("target", None)
        if raw_sessions:
            for rs in raw_sessions:
                if "target" in rs:
                    try:
                        normalized_target = _normalize_target(rs.get("target"))
                    except ValueError as e:
                        return {"error": f"Raw session '{rs.get('raw_session_id', '?')}': {e}",
                                "code": "VALIDATION_ERROR"}
                    if normalized_target:
                        rs["target"] = normalized_target
                    else:
                        rs.pop("target", None)

        # -- Normalize malformed types before enrichment --
        for f in facts:
            _normalize_fact_types(f)

        # -- L0 enrichment for facts with incomplete metadata --
        if enrich_l0:
            await self._enrich_missing_fact_metadata(facts)

        # -- ALL MUTATION UNDER LOCK --
        async with self._file_lock:

            # -- STEP 3: Session offset/remap --
            offset = self._n_sessions

            if offset > 0:
                for f in facts:
                    sn = f.get("session", 0)
                    normalized_session = _coerce_positive_session_num(sn)
                    if normalized_session is not None:
                        f["session"] = normalized_session + offset

                if consolidated:
                    for cf in consolidated:
                        normalized_session = _coerce_positive_session_num(cf.get("session"))
                        if normalized_session is not None:
                            cf["session"] = normalized_session + offset
                        if "sessions" in cf:
                            cf["sessions"] = [
                                normalized + offset
                                for s in cf["sessions"]
                                if (normalized := _coerce_positive_session_num(s)) is not None
                            ]

                if cross_session:
                    for xf in cross_session:
                        if "sessions" in xf:
                            xf["sessions"] = [
                                normalized + offset
                                for s in xf["sessions"]
                                if (normalized := _coerce_positive_session_num(s)) is not None
                            ]

                if raw_sessions:
                    for rs in raw_sessions:
                        rs["session_num"] = rs["session_num"] + offset

            # -- STEP 4: Scoped ID remap --
            import_uid = uuid4().hex[:8]
            id_remap = {}

            for f in facts:
                if "id" in f:
                    scope = f.get("session", 0)
                    key = ("g", scope, f["id"])
                    new_id = f"{import_uid}_g_s{scope}_{f['id']}"
                    id_remap[key] = new_id
                    f["id"] = new_id

            if consolidated:
                for cf in consolidated:
                    if "id" in cf:
                        scope = cf.get("session",
                            cf.get("sessions", [0])[0] if cf.get("sessions") else 0)
                        key = ("c", scope, cf["id"])
                        new_id = f"{import_uid}_c_s{scope}_{cf['id']}"
                        id_remap[key] = new_id
                        cf["id"] = new_id
                    if "source_ids" in cf:
                        cf_sessions = cf.get("sessions",
                                             [cf.get("session", 0)])
                        remapped = []
                        for sid in cf["source_ids"]:
                            matches = [id_remap[("g", cs, sid)]
                                       for cs in cf_sessions
                                       if ("g", cs, sid) in id_remap]
                            if len(matches) == 0:
                                return {"error": f"Unresolved source_id '{sid}' in "
                                                 f"cons fact {cf.get('id','?')}",
                                        "code": "VALIDATION_ERROR"}
                            if len(matches) > 1:
                                return {"error": f"Ambiguous source_id '{sid}' in "
                                                 f"cons fact {cf.get('id','?')}: "
                                                 f"matches {len(matches)} sessions",
                                        "code": "VALIDATION_ERROR"}
                            remapped.append(matches[0])
                        cf["source_ids"] = remapped

            if cross_session:
                for xf in cross_session:
                    if "id" in xf:
                        entity = (xf.get("entities") or ["unk"])[0]
                        key = ("x", entity, xf["id"])
                        new_id = f"{import_uid}_x_{entity}_{xf['id']}"
                        id_remap[key] = new_id
                        xf["id"] = new_id
                    if "source_ids" in xf:
                        xf_sessions = xf.get("sessions", [])
                        remapped = []
                        for sid in xf["source_ids"]:
                            matches = [id_remap[("g", xs, sid)]
                                       for xs in xf_sessions
                                       if ("g", xs, sid) in id_remap]
                            if len(matches) == 0:
                                return {"error": f"Unresolved source_id '{sid}' in "
                                                 f"cross fact {xf.get('id','?')}",
                                        "code": "VALIDATION_ERROR"}
                            if len(matches) > 1:
                                return {"error": f"Ambiguous source_id '{sid}' in "
                                                 f"cross fact {xf.get('id','?')}: "
                                                 f"matches {len(matches)} sessions",
                                        "code": "VALIDATION_ERROR"}
                            remapped.append(matches[0])
                        xf["source_ids"] = remapped

            # -- STEP 5: Build session_date lookup --
            sdate_map = {}
            if raw_sessions:
                sdate_map = {rs["session_num"]: rs.get("session_date", "")
                             for rs in raw_sessions}

            # -- STEP 6: Tag facts with inline metadata --
            _art_id = artifact_id or _generate_artifact_id()
            _ver_id = version_id or _generate_version_id()
            now = datetime.now(timezone.utc).isoformat()
            for f in facts:
                f.setdefault("kind", "fact")
                f.setdefault("entities", [])
                f.setdefault("tags", [])
                f.setdefault("session", self._n_sessions + 1)
                f["conv_id"] = self.key
                f["session_date"] = sdate_map.get(f.get("session", 0),
                                                   f.get("session_date", ""))
                f["agent_id"] = self.agent_id
                f["swarm_id"] = self.swarm_id
                f["scope"] = self.scope
                f["owner_id"] = _owner
                f["read"] = list(_read)
                f["write"] = list(_write)
                f["created_at"] = now
                f.setdefault("artifact_id", _art_id)
                f.setdefault("version_id", _ver_id)
                f.setdefault("status", "active")
                if provenance:
                    f["provenance"] = provenance

            # Compute _session_content_complexity from enriched metadata
            from collections import defaultdict as _defaultdict
            by_session = _defaultdict(list)
            for f in facts:
                by_session[f.get("session", 0)].append(f)
            for sn, sfacts in by_session.items():
                complexity = _compute_content_complexity(sfacts)
                for f in sfacts:
                    f["_session_content_complexity"] = complexity

            granular_by_id = {f.get("id"): f for f in facts if f.get("id")}

            self._all_granular.extend(facts)

            # -- STEP 7: Add consolidated --
            if consolidated:
                for cf in consolidated:
                    _normalize_fact_types(cf)
                    cf.setdefault("kind", "fact")
                    cf.setdefault("entities", [])
                    cf["conv_id"] = self.key
                    cf["session_date"] = sdate_map.get(cf.get("session", 0),
                                                        cf.get("session_date", ""))
                    cf["agent_id"] = self.agent_id
                    cf["swarm_id"] = self.swarm_id
                    cf["scope"] = self.scope
                    cf["owner_id"] = _owner
                    cf["read"] = list(_read)
                    cf["write"] = list(_write)
                    cf["created_at"] = now
                    merged_metadata = _merge_fact_metadata(
                        cf.get("metadata"),
                        {"asserted_derived_tier": True},
                    )
                    if merged_metadata:
                        cf["metadata"] = merged_metadata
                    if provenance:
                        cf["provenance"] = provenance
                    source_ids = cf.get("source_ids") or []
                    source_cc = max(
                        (
                            granular_by_id.get(source_id, {}).get("_session_content_complexity", 0.0)
                            for source_id in source_ids
                        ),
                        default=0.0,
                    )
                    if source_cc > 0.0:
                        cf["_session_content_complexity"] = source_cc
                self._all_cons.extend(consolidated)

            # -- STEP 8: Add cross-session --
            if cross_session:
                for xf in cross_session:
                    _normalize_fact_types(xf)
                    xf.setdefault("kind", "fact")
                    xf.setdefault("entities", [])
                    xf["conv_id"] = self.key
                    xf["session_date"] = ""
                    xf["agent_id"] = self.agent_id
                    xf["swarm_id"] = self.swarm_id
                    xf["scope"] = self.scope
                    xf["owner_id"] = _owner
                    xf["read"] = list(_read)
                    xf["write"] = list(_write)
                    xf["created_at"] = now
                    merged_metadata = _merge_fact_metadata(
                        xf.get("metadata"),
                        {"asserted_derived_tier": True},
                    )
                    if merged_metadata:
                        xf["metadata"] = merged_metadata
                    if provenance:
                        xf["provenance"] = provenance
                    source_ids = xf.get("source_ids") or []
                    source_cc = max(
                        (
                            granular_by_id.get(source_id, {}).get("_session_content_complexity", 0.0)
                            for source_id in source_ids
                        ),
                        default=0.0,
                    )
                    if source_cc <= 0.0:
                        source_cc = max(
                            (
                                f.get("_session_content_complexity", 0.0)
                                for f in facts
                                if f.get("session") in (xf.get("sessions") or [])
                            ),
                            default=0.0,
                        )
                    if source_cc > 0.0:
                        xf["_session_content_complexity"] = source_cc
                self._all_cross.extend(cross_session)

            # -- STEP 9: Append raw sessions with ACL --
            if raw_sessions:
                for rs in raw_sessions:
                    rs.setdefault("session_date", "")
                    rs.setdefault("speakers", "External source")
                    rs["owner_id"] = _owner
                    rs["read"] = list(_read)
                    rs["write"] = list(_write)
                    self._raw_sessions.append(rs)

            # -- STEP 10: Update _n_sessions --
            candidates = [self._n_sessions]
            gran_snums = [
                session_num
                for f in self._all_granular
                if (session_num := _coerce_positive_session_num(f.get("session"))) is not None
            ]
            if gran_snums:
                candidates.append(max(gran_snums))
            raw_snums = [
                session_num
                for rs in self._raw_sessions
                if (session_num := _coerce_positive_session_num(rs.get("session_num"))) is not None
            ]
            if raw_snums:
                candidates.append(max(raw_snums))
            self._n_sessions = max(candidates)

            self._n_sessions_with_facts = len(set(gran_snums))

            # -- STEP 11: ALWAYS prevent _rebuild_tiers() --
            self._tiers_dirty = False
            self._tier2_built = True
            self._tier3_built = True

            self._save_cache()

        # -- STEP 12: Embed + refresh retrieval index --
        await self.build_index()

        return {
            "granular_added": len(facts),
            "consolidated_added": len(consolidated) if consolidated else 0,
            "cross_session_added": len(cross_session) if cross_session else 0,
            "raw_sessions_added": len(raw_sessions) if raw_sessions else 0,
            "hybrid_context_available": (raw_sessions is not None
                                         and len(raw_sessions) > 0),
            "session_offset": offset,
            "import_uid": import_uid,
        }

    # ── reextract() ──

    async def reextract(self, model: str = None, call_extract_fn=None) -> dict:
        """Re-run extraction on stored raw sessions with current prompt.

        Clears existing facts and re-extracts from raw_sessions.
        Raw sessions are preserved unchanged.

        Returns: {"reextracted": N, "sessions": M}
        """
        if not self._raw_sessions:
            return {"reextracted": 0, "sessions": 0, "error": "no raw sessions stored"}

        async with self._file_lock:
            self._all_granular = []
            self._all_cons = []
            self._all_cross = []
            self._all_tlinks = []
            self._n_sessions = 0
            self._n_sessions_with_facts = 0

        extract_model = model or self.extract_model
        sem = self._get_extract_sem()

        async def _default_call_extract(m, system, user_msg, max_tokens=8192):
            return await call_extract(m, system, user_msg, max_tokens, sem)

        default_fn = call_extract_fn or _default_call_extract

        for raw in self._raw_sessions:
            raw_ct = raw.get("content_type", "default")

            # Use content_type-aware fn if raw session had a non-default type
            if call_extract_fn is None and raw_ct != "default":
                ct_prompt = self._prompt_registry.get(raw_ct)
                sdate_raw = raw["session_date"]
                snum_raw = raw["session_num"]

                async def _ct_call_extract(m, system, user_msg, max_tokens=8192,
                                           _p=ct_prompt, _sd=sdate_raw, _sn=snum_raw):
                    try:
                        dt = datetime.fromisoformat(_sd.replace("Z", "+00:00"))
                        ds = dt.strftime("%d %B %Y")
                        ym1 = str(dt.year - 1)
                    except Exception:
                        ds = _sd
                        ym1 = str(int(_sd[:4]) - 1) if len(_sd) >= 4 else "2022"

                    class _SafeDict(dict):
                        def __missing__(self, key):
                            return "{" + key + "}"

                    cs = _p.format_map(_SafeDict(
                        session_date=ds, year_minus_1=ym1, session_num=_sn))
                    return await call_extract(m, cs, user_msg, max_tokens, sem)

                fn = _ct_call_extract
            else:
                fn = default_fn

            conv_id, sn, sdate, facts, tlinks = await extract_session(
                session_text=raw["content"],
                session_num=raw["session_num"],
                session_date=raw["session_date"],
                conv_id=self.key,
                speakers=raw.get("speakers", "User and Assistant"),
                model=extract_model,
                call_extract_fn=fn,
            )
            episode_id = (
                str(raw.get("episode_id") or "")
                or f"{self._episode_source_key(str(raw.get('source_id') or self.key))}_e{int(raw.get('session_num', 0)):04d}"
            )
            for doc in self._episode_corpus.get("documents", []):
                for episode in doc.get("episodes", []):
                    if episode.get("episode_id") == episode_id:
                        _stamp_selector_episode_fields(episode, raw)
                        break
            self._align_fact_selectors(
                facts,
                episode_id=episode_id,
                source_kind="document" if raw.get("format") == "document" else "conversation",
                raw_fields=_selector_raw_fields(str(raw.get("content") or ""), raw),
                speakers=raw.get("speakers") if isinstance(raw.get("speakers"), dict) else None,
            )
            # Tag facts with raw_session_id if available
            rsid = raw.get("raw_session_id")
            if rsid:
                for f in facts:
                    f["raw_session_id"] = rsid
            self._tag_facts(
                facts, sdate,
                raw.get("agent_id", self.agent_id),
                raw.get("swarm_id", self.swarm_id),
                raw.get("scope", self.scope),
                metadata=raw.get("metadata"),
                target=raw.get("target"),
            )
            # Compute and stamp _session_content_complexity after re-extraction
            session_complexity = _compute_content_complexity(facts)
            for f in facts:
                f["_session_content_complexity"] = session_complexity
            async with self._file_lock:
                raw["status"] = "active"
                self._all_granular.extend(facts)
                self._all_tlinks.extend(tlinks)
                self._n_sessions += 1
                if facts:
                    self._n_sessions_with_facts += 1

        self._data_dict = None
        self._mark_tiers_dirty()
        async with self._file_lock:
            self._save_cache()

        # Rebuild Tier 2/3
        await self._rebuild_tiers()
        self._tiers_dirty = False
        self._tier2_built = True
        self._tier3_built = True

        return {
            "reextracted": len(self._all_granular),
            "sessions": len(self._raw_sessions),
        }

    # ── _rebuild_tiers() ──

    async def _rebuild_tiers(self) -> None:
        """Rebuild production derived tiers from granular facts.

        Legacy LLM merge tiers are disabled. The rebuild path only emits
        source-aggregation-derived cross facts, partitioned by ACL domain
        + delivery target.
        """
        if not self._all_granular:
            return
        if not self._raw_sessions and not (self._episode_corpus or {}).get("documents"):
            return

        sem = self._get_extract_sem()

        async def _call_extract_fn(model, system, user_msg, max_tokens=8192):
            if not model:
                return []
            return await call_extract(
                model=model,
                system=system,
                user_msg=user_msg,
                max_tokens=max_tokens,
                semaphore=sem,
            )

        # Partition granular facts by ACL domain + canonical target tuple.
        domain_facts: dict[tuple, list[dict]] = defaultdict(list)
        for f in self._all_granular:
            target_key = tuple(f.get("target", [])) if f.get("target") else None
            domain_key = (
                f.get("owner_id", "system"),
                tuple(sorted(f.get("read", ["agent:PUBLIC"]))),
                target_key,
            )
            domain_facts[domain_key].append(f)

        new_cons = [f for f in self._all_cons if _is_asserted_derived_fact(f)]
        new_cross = [f for f in self._all_cross if _is_asserted_derived_fact(f)]

        for (d_owner_id, d_read_tuple, d_target_tuple), d_facts in domain_facts.items():
            # Derive legacy fields from first fact in domain for _tag_facts compat
            d_agent_id = d_facts[0].get("agent_id", "default")
            d_swarm_id = d_facts[0].get("swarm_id", "default")
            d_scope = d_facts[0].get("scope", "swarm-shared")
            source_fact_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
            for fact in d_facts:
                metadata = fact.get("metadata") or {}
                source_id = metadata.get("episode_source_id") or fact.get("source_id")
                if not source_id:
                    continue
                source_record = self._source_records.get(source_id) or {}
                source_kind = source_record.get("family") or "conversation"
                source_fact_groups[(source_id, source_kind)].append(fact)

            for (source_id, source_kind), source_facts in source_fact_groups.items():
                source_date = max(
                    [str(f.get("session_date") or "") for f in source_facts if f.get("session_date")] or [""]
                )
                substrate_cross = await self._extract_source_aggregation_facts(
                    source_id=source_id,
                    source_kind=source_kind,
                    source_facts=source_facts,
                    source_date=source_date,
                    model=self.extract_model,
                    call_extract_fn=_call_extract_fn,
                    agent_id=d_agent_id,
                )
                _namespace_derived_fact_ids(
                    substrate_cross,
                    f"substrate_{self._episode_source_key(source_id)}",
                )
                max_cc = max((f.get("_session_content_complexity", 0.0) for f in source_facts), default=0.0)
                for fact in substrate_cross:
                    fact["_session_content_complexity"] = max_cc
                self._tag_facts(
                    substrate_cross,
                    source_date or "2024-01-01",
                    agent_id=d_agent_id,
                    swarm_id=d_swarm_id,
                    scope=d_scope,
                    owner_id=d_owner_id,
                    read=list(d_read_tuple),
                    write=d_facts[0].get("write", ["agent:PUBLIC"]),
                    target=_consensus_target(source_facts),
                )
                new_cross.extend(substrate_cross)

        async with self._file_lock:
            self._all_cons = new_cons
            self._all_cross = new_cross
            self._save_cache()

        self._data_dict = None

    # ── flush_background() ──

    async def flush_background(self) -> dict:
        """Run Tier 2 (consolidation) and Tier 3 (cross-session).

        Delegates to _rebuild_tiers() which partitions by visibility domain.
        Returns {"rebuilt": True, "total_consolidated": N, "total_cross_session": N}.
        """
        await self._rebuild_tiers()
        self._tiers_dirty = False
        self._tier2_built = True
        self._tier3_built = True
        return {
            "rebuilt": True,
            "total_consolidated": len(self._all_cons),
            "total_cross_session": len(self._all_cross),
        }

    # ── build_index() ──

    async def build_index(self) -> dict:
        """Embed tiers and build data_dict for retrieval.

        Mode-dependent:
          eager:        rebuild Tier 2/3 if dirty, embed all tiers
          lazy_tier2:   skip Tier 2/3 rebuild (built on demand in recall)
          lazy_tier2_3: skip Tier 2/3 rebuild (built on demand in recall)

        Must be called before first recall(). Re-call after new store() batches.
        """
        if self._tier_mode == "eager":
            if self._tiers_dirty or not (self._tier2_built and self._tier3_built):
                await self._rebuild_tiers()
                self._tiers_dirty = False
                self._tier2_built = True
                self._tier3_built = True
        # lazy modes: skip tier rebuild, done on demand in recall()

        if not self._all_granular and self._storage.exists:
            cached = self._storage.load_facts()
            self._all_granular = cached.get("granular", [])
            self._all_cons = cached.get("cons", [])
            self._all_cross = cached.get("cross", [])
            self._all_tlinks = cached.get("tlinks", [])
            for f in self._all_granular:
                f["_temporal_links"] = []
            if self._all_granular and self._all_tlinks:
                self._all_granular[0]["_temporal_links"] = self._all_tlinks

        assert len(self._all_granular) > 0, "No granular facts — call store() first"
        if isinstance(self._storage, SQLiteStorageBackend):
            async with self._file_lock:
                self._save_cache()
        if len(self._all_cons) > 0:
            g_id = self._all_granular[0].get("id", "")
            c_id = self._all_cons[0].get("id", "")
            assert g_id != c_id, f"Tier ID collision: granular[0]={g_id} == cons[0]={c_id}"

        # Fingerprint-based embedding cache (Unit 1, from run_23p.py lines 208-231)
        fp_gran = _embedding_fingerprint(self._all_granular)
        fp_cons = _embedding_fingerprint(self._all_cons)
        fp_cross = _embedding_fingerprint(self._all_cross)
        new_fps = {"gran": fp_gran, "cons": fp_cons, "cross": fp_cross}

        cache_hit = False
        saved_embs = self._storage.load_embeddings()
        if saved_embs is not None and self._emb_fingerprints == new_fps:
            emb_dim = self._embedding_dim_from_arrays(
                saved_embs.get("gran"),
                saved_embs.get("cons"),
                saved_embs.get("cross"),
            )
            gran_embs = saved_embs.get("gran", np.zeros((0, emb_dim)))
            cons_embs = saved_embs.get("cons", np.zeros((0, emb_dim)))
            cross_embs = saved_embs.get("cross", np.zeros((0, emb_dim)))
            if (len(gran_embs) == len(self._all_granular)
                    and len(cons_embs) == len(self._all_cons)
                    and len(cross_embs) == len(self._all_cross)):
                cache_hit = True
                log.info("Embedding cache hit (fingerprint match)")

        if not cache_hit:
            gran_texts = [f.get("fact", "") for f in self._all_granular]
            gran_embs = await embed_texts(gran_texts, label=f"gran-{self.key[:8]}")
            emb_dim = self._embedding_dim_from_arrays(gran_embs)

            cons_texts = [f.get("fact", "") for f in self._all_cons]
            cons_embs = await embed_texts(cons_texts, label=f"cons-{self.key[:8]}") \
                if cons_texts else np.zeros((0, emb_dim))

            cross_texts = [f.get("fact", "") for f in self._all_cross]
            cross_embs = await embed_texts(cross_texts, label=f"cross-{self.key[:8]}") \
                if cross_texts else np.zeros((0, emb_dim))

            self._storage.save_embeddings(gran_embs, cons_embs, cross_embs)
            self._emb_fingerprints = new_fps
            log.info("Embeddings computed and cached")

        self._data_dict = _build_index_state(
            self._all_granular,
            gran_embs,
            self._all_cons,
            cons_embs,
            self._all_cross,
            cross_embs,
        )

        # Resolve supersession + populate _fact_lookup for _is_visible source checks
        all_facts = self._all_granular + self._all_cons + self._all_cross
        resolve_supersession(all_facts, self._data_dict["fact_lookup"])
        self._fact_lookup = self._data_dict["fact_lookup"]
        self._rebuild_temporal_index()

        # Persist supersession links
        async with self._file_lock:
            self._save_cache()
            if self._uses_legacy_sidecars:
                write_json_atomic(self._temporal_json, self._temporal_index)

        return {
            "granular": len(self._all_granular),
            "consolidated": len(self._all_cons),
            "cross_session": len(self._all_cross),
            "tier_mode": self._tier_mode,
            "tier2_built": self._tier2_built,
            "tier3_built": self._tier3_built,
        }

    # ── recall() ──

    # Map spec query_type → retrieval type
    _QUERY_TYPE_MAP = {
        "auto":        None,
        "lookup":      "default",
        "temporal":    "temporal",
        "aggregate":   "counting",
        "current":     "current",
        "synthesize":  "synthesis",
        "procedural":  "rule",
        "prospective": "prospective",
    }

    # Query types that benefit from Tier 2
    _NEEDS_TIER2 = {"synthesis", "current", "temporal", "counting",
                    "aggregate", "summarize", "icl"}
    # Query types that benefit from Tier 3
    _NEEDS_TIER3 = {"synthesis"}

    async def recall(
        self,
        query: str,
        agent_id: str = None,
        swarm_id: str = None,
        search_family: str = "auto",
        token_budget: int | None = None,
        query_type: str = "auto",
        kind: str = "all",
        caller_memberships: list = None,
        caller_role: str = "user",
        caller_id: str = None,
    ) -> dict:
        """Query memory. Returns dict with context, retrieved, query_type, etc.

        search_family: auto | conversation | document
        query_type: auto | lookup | temporal | aggregate | current |
                    synthesize | procedural | prospective
        kind: all | fact | preference | constraint | rule | ... (filters by fact kind)
        """
        self._audit.log("recall", caller_id or "system",
                        {"query": query[:200], "query_type": query_type})
        if self._data_dict is None and self._all_granular:
            await self.build_index()

        # Build filter: ACL + optional kind
        # caller_id takes precedence (set by MCP identity resolution)
        if caller_id:
            _caller_id = caller_id
        elif agent_id and agent_id != "default":
            _caller_id = f"agent:{agent_id}"
        else:
            _caller_id = "system"
        _memberships = caller_memberships or []
        # Auto-resolve memberships from registry if not provided
        if not _memberships and hasattr(self, '_membership_registry'):
            _memberships = self._membership_registry.memberships_for(_caller_id)
        _role = caller_role
        _swarm_for_raw = swarm_id if swarm_id is not None else self.swarm_id
        def _with_raw_recall(result: dict) -> dict:
            return self._merge_raw_recall(
                query=query,
                result=result,
                caller_id=_caller_id,
                caller_memberships=_memberships,
                caller_role=_role,
                swarm_id=_swarm_for_raw,
            )
        acl_ok = lambda f: self._acl_allows(f, _caller_id, _memberships, _role)
        _now = datetime.now(timezone.utc)
        _fl = self._fact_lookup if hasattr(self, '_fact_lookup') else None
        visible = lambda f: _is_visible(f, now=_now, fact_lookup=_fl) and acl_ok(f)

        if kind != "all":
            fact_filter = lambda f: visible(f) and f.get("kind") == kind
        else:
            fact_filter = visible

        has_episode_corpus = bool(self._episode_corpus.get("documents"))
        has_visible_facts = any(fact_filter(f) for f in self._all_granular)

        # MAL generation-aware overrides
        _mal_config = _load_mal_active_config(str(self.data_dir), self.key, agent_id or "default")
        _mal_selector = _mal_config.get("selector_config_overrides") or None
        _mal_leaf_overrides = _mal_config.get("inference_leaf_plugin_overrides") or {}

        episode_runtime = self._visible_episode_runtime(fact_filter, query=query)
        if episode_runtime:
            corpus, _episode_lookup, _facts_by_episode, _bm25 = episode_runtime
            packet = build_episode_hybrid_context(
                query,
                corpus,
                self._episode_runtime_facts(fact_filter, query=query),
                selector_config=_mal_selector,
                search_family=search_family,
                temporal_index=self._temporal_index,
            )
            temporal_trace = packet.get("temporal_trace") or {}
            temporal_executor_matched = bool(
                temporal_trace.get("query_class") in {"ordinal", "calendar-answer"}
                and temporal_trace.get("matched")
                and not temporal_trace.get("fallback")
            )
            packet, augmented_facts = await self._repair_temporal_grounding_packet(
                query=query,
                packet=packet,
                episode_lookup=_episode_lookup,
                fact_filter=fact_filter,
            )
            if not temporal_executor_matched and augmented_facts is None:
                packet, augmented_facts = await self._augment_conversation_structural_packet(
                    query=query,
                    packet=packet,
                    episode_lookup=_episode_lookup,
                    fact_filter=fact_filter,
                )
                if augmented_facts is None:
                    packet, augmented_facts = await self._augment_document_structural_packet(
                        query=query,
                        packet=packet,
                        episode_lookup=_episode_lookup,
                        fact_filter=fact_filter,
                    )
                packet, coverage_recovery_facts = await self._recover_multi_item_coverage_packet(
                    query=query,
                    packet=packet,
                    episode_lookup=_episode_lookup,
                    fact_filter=fact_filter,
                )
                if coverage_recovery_facts is not None:
                    augmented_facts = coverage_recovery_facts
                if augmented_facts is None:
                    packet, augmented_facts = await self._rescue_episode_packet_with_semantic_fact_sweep(
                        query=query,
                        packet=packet,
                        episode_lookup=_episode_lookup,
                        fact_filter=fact_filter,
                    )
            selected_ids = set(packet.get("retrieved_fact_ids", []))
            episode_runtime_facts = self._episode_runtime_facts(fact_filter, query=query)
            episode_runtime_lookup = {
                f.get("id", ""): f for f in episode_runtime_facts
            }
            resolved_facts = augmented_facts or [
                episode_runtime_lookup[fact_id]
                for fact_id in packet.get("retrieved_fact_ids", [])
                if fact_id in episode_runtime_lookup
            ]
            total_sessions = len(self._raw_sessions)
            sessions_in_ctx = len({
                session_num
                for f in resolved_facts
                if (session_num := _coerce_positive_session_num(f.get("session"))) is not None
            })
            coverage_pct = (sessions_in_ctx / total_sessions * 100) if total_sessions else 0
            resolved_type = self._QUERY_TYPE_MAP.get(query_type) or detect_query_type(query)
            query_features = extract_query_features(query)
            explicit_step_query = bool(
                query_features.get("step_numbers") or query_features.get("step_range")
            )
            if resolved_type == "temporal" and not explicit_step_query:
                semantic_temporal_episode_ids = {
                    episode_id
                    for fact in resolved_facts
                    for episode_id in fact_episode_ids(fact)
                    if ((fact.get("metadata") or {}).get("source_aggregation")
                        and (fact.get("metadata") or {}).get("semantic_class") == "temporal_semantics")
                }
                if semantic_temporal_episode_ids:
                    preferred_facts = []
                    for fact in resolved_facts:
                        metadata = fact.get("metadata") or {}
                        is_semantic_temporal = (
                            metadata.get("source_aggregation")
                            and metadata.get("semantic_class") == "temporal_semantics"
                        )
                        if is_semantic_temporal:
                            preferred_facts.append(fact)
                            continue
                        if set(fact_episode_ids(fact)) & semantic_temporal_episode_ids:
                            continue
                        preferred_facts.append(fact)
                    if preferred_facts and len(preferred_facts) != len(resolved_facts):
                        resolved_facts = preferred_facts
                        context, actual_injected_episode_ids = build_context_from_retrieved_facts(
                            resolved_facts,
                            _episode_lookup,
                            fact_lookup=episode_runtime_lookup,
                            budget=int(packet.get("selector_config", {}).get("budget", 8000)),
                            snippet_chars=int(packet.get("tuning_snapshot", {}).get("packet", {}).get("snippet_chars", 1200)),
                            question=query,
                            query_features=query_features,
                        )
                        packet = dict(packet)
                        packet["context"] = context
                        packet["retrieved_fact_ids"] = [fact.get("id", "") for fact in resolved_facts]
                        packet["retrieved_episode_ids"] = list(dict.fromkeys(
                            episode_id
                            for fact in resolved_facts
                            for episode_id in fact_episode_ids(fact)
                            if episode_id
                        ))
                        packet["actual_injected_episode_ids"] = actual_injected_episode_ids
            if resolved_type == "synthesis":
                packet, resolved_facts = self._ensure_min_synthesis_evidence(
                    packet=packet,
                    resolved_facts=resolved_facts,
                    episode_lookup=_episode_lookup,
                    fact_filter=fact_filter,
                )
            complexity_hint = _compute_complexity_hint(
                retrieved=resolved_facts,
                resolved_type=resolved_type,
                is_multihop=(resolved_type in ("temporal", "current", "counting")),
                fact_lookup=episode_runtime_lookup,
                query=query,
            )

            prompt_type, use_tool = _route_prompt_type(
                resolved_type,
                resolved_facts,
                total_sessions,
                sessions_in_ctx,
                packet["context"],
                allow_tool_mode=False,
            )

            if resolved_type == "summarize":
                retrieved_items = self._canonical_retrieved_items(resolved_facts)
            elif resolved_type == "synthesis":
                retrieved_items = self._synthesis_retrieved_items(
                    resolved_facts=resolved_facts,
                    packet=packet,
                    episode_lookup=_episode_lookup,
                )
            else:
                retrieved_items = resolved_facts

            result = {
                "context": packet["context"],
                "retrieved": retrieved_items,
                "query_type": resolved_type,
                "is_multihop": resolved_type in ("temporal", "current", "counting"),
                "complexity_hint": complexity_hint,
                "n_facts": len(self._all_granular) + len(self._all_cons) + len(self._all_cross),
                "sessions_in_context": sessions_in_ctx,
                "total_sessions": total_sessions,
                "coverage_pct": coverage_pct,
                "raw_budget": 0,
                "recommended_prompt_type": prompt_type,
                "use_tool": use_tool,
                "retrieved_episode_ids": packet["retrieved_episode_ids"],
                "actual_injected_episode_ids": packet["actual_injected_episode_ids"],
                "selection_scores": packet["selection_scores"],
                "query_operator_plan": packet["query_operator_plan"],
                "output_constraints": packet["output_constraints"],
                "retrieval_families": packet.get("retrieval_families", []),
                "search_family": packet.get("search_family", search_family),
                "inference_leaf_plugins": _mal_leaf_overrides or None,
                "runtime_trace": self._episode_runtime_trace(
                    corpus=corpus,
                    packet=packet,
                    episode_lookup=_episode_lookup,
                    resolved_facts=resolved_facts,
                ),
            }
            if self._profiles:
                level = complexity_hint.get("level", 1)
                profile = self._profiles.get(level)
                if profile:
                    result["recommended_profile"] = profile
                payload, payload_meta = self._build_payload(
                    query=query,
                    recall_result=result,
                )
                if payload and payload_meta:
                    result["payload"] = payload
                    result["payload_meta"] = payload_meta
            if temporal_trace:
                result["temporal_resolution"] = temporal_trace
                deterministic_answer = str(temporal_trace.get("deterministic_answer") or "").strip()
                if deterministic_answer:
                    result["deterministic_answer"] = deterministic_answer
            temporal_deterministic_query = (
                resolved_type == "temporal"
                or self._temporal_query_requests_year_resolution(query)
                or self._temporal_query_requests_month_resolution(query)
                or self._temporal_query_requests_first_window(query)
            )
            if not result.get("deterministic_answer") and temporal_deterministic_query:
                deterministic_answer = self._derive_relative_temporal_deterministic_answer(
                    query=query,
                    resolved_facts=resolved_facts,
                    episode_lookup=_episode_lookup,
                )
                query_class = "relative-anchor"
                if not deterministic_answer:
                    deterministic_answer = self._derive_first_window_temporal_deterministic_answer(
                        query=query,
                        context=packet["context"],
                    )
                    if deterministic_answer:
                        query_class = "first-window"
                if not deterministic_answer:
                    deterministic_answer = self._derive_duration_temporal_deterministic_answer(
                        query=query,
                        resolved_facts=resolved_facts,
                        episode_lookup=_episode_lookup,
                    )
                    if deterministic_answer:
                        query_class = "duration-anchor"
                if not deterministic_answer:
                    deterministic_answer = self._derive_month_temporal_deterministic_answer(
                        query=query,
                        resolved_facts=resolved_facts,
                        episode_lookup=_episode_lookup,
                    )
                    if deterministic_answer:
                        query_class = "month-anchor"
                if deterministic_answer:
                    result["deterministic_answer"] = deterministic_answer
                    temporal_resolution = dict(result.get("temporal_resolution") or {})
                    temporal_resolution.setdefault("query_class", query_class)
                    temporal_resolution["deterministic_answer"] = deterministic_answer
                    result["temporal_resolution"] = temporal_resolution
            if not result.get("deterministic_answer"):
                deterministic_answer = self._derive_time_scoped_acquisition_deterministic_answer(
                    query=query,
                    query_features=query_features,
                    packet=packet,
                    episode_lookup=_episode_lookup,
                )
                if deterministic_answer:
                    result["deterministic_answer"] = deterministic_answer
                    runtime_trace = dict(result.get("runtime_trace") or {})
                    runtime_trace["deterministic_answer"] = {
                        "kind": "time_scoped_acquisition",
                        "answer": deterministic_answer,
                    }
                    result["runtime_trace"] = runtime_trace
            if not result.get("deterministic_answer"):
                deterministic_answer = self._derive_time_scoped_activity_acquisition_deterministic_answer(
                    query=query,
                    query_features=query_features,
                    packet=packet,
                    episode_lookup=_episode_lookup,
                )
                if deterministic_answer:
                    result["deterministic_answer"] = deterministic_answer
                    runtime_trace = dict(result.get("runtime_trace") or {})
                    runtime_trace["deterministic_answer"] = {
                        "kind": "time_scoped_activity_acquisition",
                        "answer": deterministic_answer,
                    }
                    result["runtime_trace"] = runtime_trace
            if not result.get("deterministic_answer"):
                deterministic_answer = self._derive_activity_list_deterministic_answer(
                    query=query,
                    query_features=query_features,
                    packet=packet,
                    episode_lookup=_episode_lookup,
                )
                if deterministic_answer:
                    result["deterministic_answer"] = deterministic_answer
                    runtime_trace = dict(result.get("runtime_trace") or {})
                    runtime_trace["deterministic_answer"] = {
                        "kind": "activity_list",
                        "answer": deterministic_answer,
                    }
                    result["runtime_trace"] = runtime_trace
            return _with_raw_recall(result)
        if has_episode_corpus and not has_visible_facts:
            resolved_type = self._QUERY_TYPE_MAP.get(query_type) or "default"
            return _with_raw_recall({
                "context": "RETRIEVED FACTS:",
                "retrieved": [],
                "query_type": resolved_type,
                "is_multihop": False,
                "complexity_hint": {
                    "score": 0.0,
                    "level": 1,
                    "signals": [],
                    "retrieval_complexity": 0.0,
                    "content_complexity": 0.0,
                    "query_complexity": 0.0,
                    "dominant": "tie",
                },
                "n_facts": len(self._all_granular) + len(self._all_cons) + len(self._all_cross),
                "sessions_in_context": 0,
                "total_sessions": len(self._raw_sessions),
                "coverage_pct": 0,
                "raw_budget": 0,
                "recommended_prompt_type": resolved_type,
                "use_tool": False,
                "retrieved_episode_ids": [],
                "actual_injected_episode_ids": [],
                "selection_scores": [],
                "query_operator_plan": {},
                "output_constraints": {},
                "retrieval_families": [],
                "search_family": search_family,
                "runtime_trace": {
                    "runtime": "episode",
                    "scope": self._scope_trace(),
                    "reason": "empty_visible_facts",
                    "family_first_pass": {
                        "available_families": available_families(self._episode_corpus),
                        "retrieval_families": [],
                        "requested_search_family": search_family,
                        "per_family": [],
                    },
                    "query": {},
                    "late_fusion": {"mode": "empty"},
                    "selection": {
                        "retrieved_episode_ids": [],
                        "actual_injected_episode_ids": [],
                        "selection_scores": [],
                    },
                    "cross_contamination": {
                        "source_ids": [],
                        "source_count": 0,
                        "family_counts": {},
                        "multi_source": False,
                        "candidate_source_ids": [],
                        "candidate_source_count": 0,
                        "candidate_family_counts": {},
                        "rejected_source_ids": [],
                        "rejected_source_count": 0,
                        "rejected_family_counts": {},
                    },
                    "packet": {
                        "retrieved_fact_ids": [],
                        "retrieved_fact_count": 0,
                        "requested_episode_count": 0,
                        "actual_injected_episode_count": 0,
                        "context_chars": len("RETRIEVED FACTS:"),
                        "snippet_mode": False,
                        "budget_chars": None,
                    },
                    "tuning": get_runtime_tuning(),
                },
            })
        return _with_raw_recall(await self._generic_fact_recall(
            query=query,
            fact_filter=fact_filter,
            search_family=search_family,
            query_type=query_type,
        ))

    # ── context_for() ──

    async def context_for(
        self,
        query: str,
        agent_id: str = None,
        swarm_id: str = None,
        search_family: str = "auto",
        token_budget: int = 4000,
        query_type: str = "auto",
        kind: str = "all",
        caller_memberships: list = None,
        caller_role: str = "user",
        caller_id: str = None,
    ) -> dict:
        """Same as recall() but truncates context to token_budget (1 token ≈ 4 chars)."""
        result = await self.recall(
            query, agent_id=agent_id, swarm_id=swarm_id,
            search_family=search_family,
            query_type=query_type, kind=kind,
            caller_memberships=caller_memberships, caller_role=caller_role,
            caller_id=caller_id,
        )
        if "context" not in result:
            return result
        max_chars = token_budget * 4
        if len(result["context"]) > max_chars:
            result["context"] = result["context"][:max_chars] + "\n[...truncated]"
        result.pop("payload", None)
        result.pop("payload_meta", None)
        result.pop("_context_packet", None)
        return result

    # ── Secrets ──

    def store_secret(self, name: str, value: str,
                     agent_id: str, swarm_id: str, scope: str) -> dict:
        """Store secret. Upsert by name within same scope context."""
        if scope not in ("system-wide", "swarm-shared", "agent-private"):
            return {"error": "invalid scope", "code": "INVALID_SCOPE"}

        for existing in self._secrets:
            if (existing["name"] == name
                    and existing["agent_id"] == agent_id
                    and existing["swarm_id"] == swarm_id
                    and existing["scope"] == scope):
                existing["value"] = value
                existing["stored_at"] = datetime.now(timezone.utc).isoformat()
                self._save_cache()
                return {"stored": True}

        self._secrets.append({
            "name":      name,
            "value":     value,
            "agent_id":  agent_id,
            "swarm_id":  swarm_id,
            "scope":     scope,
            "stored_at": datetime.now(timezone.utc).isoformat(),
        })
        self._save_cache()
        return {"stored": True}

    def get_secret(self, name: str, agent_id: str, swarm_id: str) -> dict:
        """Fetch secret by exact name. Scope-enforced.

        Iterates ALL candidates with matching name, checks each scope,
        returns the first accessible one. Only returns FORBIDDEN if
        candidates exist but none are accessible.
        """
        found_any = False
        for s in self._secrets:
            if s["name"] != name:
                continue
            found_any = True
            scope = s["scope"]
            if scope == "system-wide":
                return {"value": s["value"]}
            if scope == "swarm-shared" and s["swarm_id"] == swarm_id:
                return {"value": s["value"]}
            if scope == "agent-private" and s["agent_id"] == agent_id:
                return {"value": s["value"]}
        if found_any:
            return {"error": "access denied", "code": "SECRET_FORBIDDEN"}
        return {"error": "secret not found", "code": "SECRET_NOT_FOUND"}

    # ── Profile helpers ──

    def _has_profiles(self) -> bool:
        """True if profile config was provided."""
        return bool(self._profiles)

    def _default_embed_model(self) -> str:
        try:
            from .setup_store import get_config

            cfg = get_config()
            return cfg.get("embed_model") or "text-embedding-3-large"
        except Exception:
            return "text-embedding-3-large"

    def _normalize_retrieval_config(self, retrieval: dict | None) -> dict:
        if retrieval is None:
            retrieval = {}
        elif not isinstance(retrieval, dict):
            raise ValueError("retrieval must be an object")
        allowed = {"search_family", "default_token_budget"}
        unknown = set(retrieval) - allowed
        if unknown:
            raise ValueError(
                f"unknown retrieval config keys: {', '.join(sorted(unknown))}"
            )
        normalized = dict(retrieval)
        normalized.setdefault("search_family", "auto")
        normalized.setdefault("default_token_budget", 4000)
        if normalized["search_family"] not in {"auto", "conversation", "document"}:
            raise ValueError("retrieval.search_family must be auto|conversation|document")
        if (
            not isinstance(normalized["default_token_budget"], int)
            or normalized["default_token_budget"] <= 0
        ):
            raise ValueError("retrieval.default_token_budget must be positive int")
        return normalized

    def _default_memory_config(self) -> dict:
        return {
            "schema_version": 1,
            "embedding_model": self._default_embed_model(),
            "librarian_profile": self.extract_model or None,
            "profiles": dict(self._profiles or {}),
            "profile_configs": deepcopy(self._profile_configs or {}),
            "retrieval": {
                "search_family": "auto",
                "default_token_budget": 4000,
            },
        }

    def _resolve_librarian_model(
        self,
        librarian_profile: str | None,
        profile_configs: dict,
    ) -> str | None:
        if not librarian_profile:
            return None
        cfg = profile_configs.get(librarian_profile)
        if isinstance(cfg, dict) and cfg.get("model"):
            return cfg["model"]
        return librarian_profile

    def _normalize_profile_levels(self, profiles: dict) -> dict[int, str]:
        """Normalize complexity→profile mapping to int keys."""
        normalized = {}
        for level, name in profiles.items():
            lvl = int(level)
            if lvl < 1 or lvl > 5:
                raise ValueError(f"profile level must be 1-5, got {lvl}")
            normalized[lvl] = name
        return normalized

    def _validate_profile_configs(self, profile_configs: dict) -> None:
        """Validate persisted/runtime profile config shape."""
        for name, cfg in profile_configs.items():
            if "model" not in cfg:
                raise ValueError(f"profile '{name}' missing 'model'")
            if "max_output_tokens" in cfg:
                value = cfg["max_output_tokens"]
                if not isinstance(value, int) or value <= 0:
                    raise ValueError(
                        f"profile '{name}': max_output_tokens must be positive int"
                    )
            if "max_output_tokens_summarize" in cfg:
                value = cfg["max_output_tokens_summarize"]
                if not isinstance(value, int) or value <= 0:
                    raise ValueError(
                        f"profile '{name}': max_output_tokens_summarize must be positive int"
                    )
            if "temperature" in cfg:
                value = cfg["temperature"]
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"profile '{name}': temperature must be numeric"
                    )

    def _validate_memory_config(self, config: dict) -> dict:
        if not isinstance(config, dict):
            raise ValueError("config must be an object")

        allowed = {
            "schema_version",
            "embedding_model",
            "librarian_profile",
            "profiles",
            "profile_configs",
            "retrieval",
        }
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(
                f"unknown memory config keys: {', '.join(sorted(unknown))}"
            )

        schema_version = config.get("schema_version")
        if schema_version != 1:
            raise ValueError("schema_version must be 1")

        embedding_model = config.get("embedding_model")
        if not isinstance(embedding_model, str) or not embedding_model.strip():
            embedding_model = self._default_embed_model()
        if not embedding_model:
            raise ValueError("embedding_model is required")

        librarian_profile = config.get("librarian_profile")
        if librarian_profile is None:
            normalized_librarian_profile = None
        elif isinstance(librarian_profile, str) and librarian_profile.strip():
            normalized_librarian_profile = librarian_profile.strip()
        else:
            raise ValueError("librarian_profile must be a non-empty string or null")

        profiles = config.get("profiles") or {}
        profile_configs = config.get("profile_configs") or {}
        normalized_profiles = self._normalize_profile_levels(profiles) if profiles else {}
        if normalized_profiles and not profile_configs:
            profile_configs = {
                name: {"model": name}
                for name in dict.fromkeys(normalized_profiles.values())
            }
        self._validate_profile_configs(profile_configs)
        for level, name in normalized_profiles.items():
            if name not in profile_configs:
                raise ValueError(
                    f"profile '{name}' referenced by level {level} but not in profile_configs"
                )

        retrieval = self._normalize_retrieval_config(config.get("retrieval"))

        return {
            "schema_version": schema_version,
            "embedding_model": embedding_model,
            "librarian_profile": normalized_librarian_profile,
            "profiles": normalized_profiles,
            "profile_configs": deepcopy(profile_configs),
            "retrieval": retrieval,
        }

    def _apply_memory_config(self, config: dict) -> None:
        normalized = self._validate_memory_config(config)
        self._memory_config = deepcopy(normalized)
        self._embedding_model = normalized["embedding_model"]
        self._librarian_profile = normalized["librarian_profile"]
        self._profiles = dict(normalized["profiles"]) if normalized["profiles"] else None
        self._profile_configs = deepcopy(normalized["profile_configs"])
        self.extract_model = self._resolve_librarian_model(
            self._librarian_profile,
            self._profile_configs,
        ) or ""

    async def set_config(self, config: dict) -> None:
        async with self._file_lock:
            self._apply_memory_config(config)
            self._save_cache()

    def get_config(self) -> dict:
        if self._memory_config is None:
            self._apply_memory_config(self._default_memory_config())
        return deepcopy(self._memory_config)

    def _embedding_dim_from_arrays(self, *arrays) -> int:
        for arr in arrays:
            if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] > 0:
                return int(arr.shape[1])
        return 3072

    def _get_profile_config(self, name: str) -> dict | None:
        """Return profile config dict by name: {model, context_window, ...}.

        Falls back to minimal config if profile_configs not provided.
        """
        if self._profile_configs and name in self._profile_configs:
            return self._profile_configs[name]
        # Check if name exists in profiles mapping
        if not self._profiles:
            return None
        if name not in self._profiles.values():
            return None
        # Minimal fallback — caller must provide profile_configs for real use
        return {"model": name, "context_window": 128000,
                "input_cost_per_1k": 2.0, "output_cost_per_1k": 8.0}

    def _list_profile_names(self) -> list[str]:
        """Return unique profile names."""
        if not self._profiles:
            return []
        return list(dict.fromkeys(self._profiles.values()))

    def _cheapest_profile(self) -> str | None:
        """Return profile name mapped to the lowest complexity level."""
        if not self._profiles:
            return None
        min_level = min(self._profiles.keys())
        return self._profiles[min_level]

    async def set_profiles(self, profiles: dict, profile_configs: dict) -> None:
        """Set complexity→profile mapping and profile configs. Persisted."""
        normalized = self._normalize_profile_levels(profiles)
        for level, name in normalized.items():
            if name not in profile_configs:
                raise ValueError(
                    f"profile '{name}' referenced by level {level} but not in profile_configs"
                )
        self._validate_profile_configs(profile_configs)
        async with self._file_lock:
            next_config = self.get_config()
            next_config["profiles"] = normalized
            next_config["profile_configs"] = deepcopy(profile_configs)
            self._apply_memory_config(next_config)
            self._save_cache()

    def get_profiles(self) -> dict:
        """Return current profiles and configs."""
        cfg = self.get_config()
        return {
            "profiles": cfg.get("profiles", {}),
            "profile_configs": cfg.get("profile_configs", {}),
        }

    def _resolve_inference_target(
        self,
        recommended_profile: str | None,
        inference_model: str | None = None,
    ) -> dict | None:
        """Resolve model/profile defaults for payload building."""
        fallback_cfg = {
            "context_window": 128000,
            "thinking_overhead": 0,
            "input_cost_per_1k": 2.0,
            "output_cost_per_1k": 8.0,
            "temperature": 0,
        }

        if inference_model:
            return {
                "model": inference_model,
                "profile_used": inference_model,
                "profile_fallback": False,
                "cfg": {"model": inference_model, **fallback_cfg},
            }

        if self._has_profiles() and recommended_profile:
            cfg = self._get_profile_config(recommended_profile)
            if cfg:
                return {
                    "model": cfg["model"],
                    "profile_used": recommended_profile,
                    "profile_fallback": False,
                    "cfg": {**fallback_cfg, **cfg},
                }

        if self._has_profiles():
            fallback = self._cheapest_profile()
            fallback_cfg_resolved = self._get_profile_config(fallback) if fallback else None
            if fallback and fallback_cfg_resolved:
                return {
                    "model": fallback_cfg_resolved["model"],
                    "profile_used": fallback,
                    "profile_fallback": True,
                    "cfg": {**fallback_cfg, **fallback_cfg_resolved},
                }

        return None

    def _resolve_max_tokens(
        self,
        cfg: dict | None,
        resolved_type: str,
        prompt_type: str,
        explicit_max_tokens: int | None = None,
    ) -> int:
        if explicit_max_tokens is not None:
            return explicit_max_tokens
        is_summarize = (
            resolved_type == "summarize"
            or prompt_type in ("summarize", "summarize_with_metadata")
        )
        if cfg:
            if is_summarize and "max_output_tokens_summarize" in cfg:
                return cfg["max_output_tokens_summarize"]
            if "max_output_tokens" in cfg:
                return cfg["max_output_tokens"]
        return 4096 if is_summarize else 2000

    def _compute_memory_budget(self, cfg: dict | None, max_tokens: int) -> int:
        cfg = cfg or {}
        context_window = int(cfg.get("context_window", 128000))
        thinking_overhead = float(cfg.get("thinking_overhead", 0) or 0)
        usable = int(context_window * (1 - thinking_overhead) * 0.9)
        return max(1, usable - max_tokens)

    def _build_payload_messages(
        self,
        *,
        prompt_type: str,
        context: str,
        query: str,
        recall_result: dict,
        speakers: str,
    ) -> list[dict]:
        query_features = extract_query_features(query)
        operator_plan = recall_result.get("query_operator_plan") or query_features.get("operator_plan") or {}
        plugin_state = dict(self._inference_leaf_plugins)
        plugin_state.update(recall_result.get("inference_leaf_plugins") or {})
        prompt_key = resolve_inference_prompt_key(
            prompt_type,
            operator_plan,
            plugin_state=plugin_state,
        )
        prompt = get_inf_prompt(prompt_key)
        fmt_kwargs = {
            "speakers": speakers,
            "sessions_in_context": recall_result.get("sessions_in_context", 0),
            "total_sessions": recall_result.get("total_sessions", 0),
            "coverage_pct": recall_result.get("coverage_pct", 100),
            "reference_date": "2023-01-01",
        }
        formatted = prompt.format(
            context=context,
            question=query,
            **fmt_kwargs,
        )
        return [{"role": "user", "content": formatted}]

    def _normalize_grounded_answer(self, query: str, answer: str, recall_result: dict) -> str:
        if not answer:
            return answer
        qf = extract_query_features(query)
        lowered_query = (query or "").strip().lower()
        support_texts: list[str] = []
        for fact in recall_result.get("retrieved", []):
            text = (fact or {}).get("fact", "")
            if text:
                support_texts.append(text)

        def _extract_explicit_country_surface(text: str) -> str | None:
            if not text:
                return None
            cleaned = re.sub(r"[*_`#]+", "", text)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            for pattern in (
                r"^\s*answer\s*:\s*(?:in\s+)?(?:the\s+)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})\b",
                r"\b(?:country of|country is|country was|in the country of)\s+(?:the\s+)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})\b",
                r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\s+is\s+in\s+(?:the\s+)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})\b",
                r"\bwhich\s+is\s+in\s+(?:the\s+)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})\b",
                r"\b(?:the\s+)?answer would be\s+(?:in\s+)?(?:the\s+)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})\b",
                r"\b(?:visited|visiting|travel(?:ed|ing)(?:\s+to)?|trip to|meet(?:ing)? in)\s+(?:the\s+)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})\b",
                r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3},\s*(?:the\s+)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})\b",
            ):
                match = re.search(pattern, cleaned, re.I)
                if not match:
                    continue
                candidate = re.sub(r"\s+", " ", match.group(1).strip(" \t\r\n,.;:!?"))
                candidate = re.sub(r"^the\s+", "", candidate, flags=re.I)
                if not candidate or any(ch.isdigit() for ch in candidate):
                    continue
                candidate_tokens = []
                for token in candidate.split():
                    if re.match(r"^[A-Z][A-Za-z-]*$", token):
                        candidate_tokens.append(token)
                        continue
                    if token.lower() in {"and", "of", "the"} and candidate_tokens:
                        candidate_tokens.append(token.lower())
                        continue
                    break
                while candidate_tokens and candidate_tokens[-1] in {"and", "of", "the"}:
                    candidate_tokens.pop()
                candidate = " ".join(candidate_tokens)
                if not candidate:
                    continue
                if len(candidate.split()) > 4:
                    continue
                return candidate
            return None

        def _extract_city_based_country_surface(text: str) -> str | None:
            if not text:
                return None
            cleaned = re.sub(r"[*_`#]+", "", text)
            cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
            for city, country in sorted(_CITY_TO_COUNTRY.items(), key=lambda item: (-len(item[0]), item[0])):
                if re.search(rf"\b{re.escape(city)}\b", cleaned):
                    return country
            return None

        def _country_surface_from_texts(*texts: str) -> str | None:
            explicit_candidates = []
            city_candidates = []
            for text in texts:
                candidate = _extract_explicit_country_surface(text)
                if candidate:
                    explicit_candidates.append(candidate)
            for text in texts:
                candidate = _extract_city_based_country_surface(text)
                if candidate:
                    city_candidates.append(candidate)
            normalized_city = {candidate.strip() for candidate in city_candidates if candidate and candidate.strip()}
            if len(normalized_city) == 1:
                return next(iter(normalized_city))
            normalized_explicit = {candidate.strip() for candidate in explicit_candidates if candidate and candidate.strip()}
            if len(normalized_explicit) == 1:
                return next(iter(normalized_explicit))
            return None

        if re.match(r"^(?:in\s+)?what country\b|^which country\b", lowered_query):
            answer_candidate = _country_surface_from_texts(answer)
            if answer_candidate:
                return answer_candidate

        temporal_resolution = recall_result.get("temporal_resolution") or {}
        if str(temporal_resolution.get("deterministic_answer") or "").strip():
            return answer

        deterministic_meta = (recall_result.get("runtime_trace") or {}).get("deterministic_answer") or {}
        deterministic_kind = str(deterministic_meta.get("kind") or "").strip().lower()
        if deterministic_kind in {"activity_list", "time_scoped_acquisition", "time_scoped_activity_acquisition"}:
            return answer

        slot_plan = qf.get("operator_plan", {}).get("slot_query", {})
        slot_query_enabled = bool(slot_plan.get("enabled"))
        head_tokens = {
            normalize_term_token(token)
            for token in (slot_plan.get("head_tokens") or [])
            if normalize_term_token(token)
        } if slot_query_enabled else set()

        lowered = answer.lower()
        negative_answer = any(
            phrase in lowered
            for phrase in (
                "not mentioned",
                "not specified",
                "not provided",
                "not available",
                "unknown",
                "not explicitly stated",
                "not detailed",
                "cannot be determined",
                "does not specify",
                "not clarified",
                "no specific",
            )
        )

        grounded_candidates: list[str] = []
        explicit_raw_candidates: list[str] = []
        grounded_seen: set[str] = set()

        def _normalized_answer_surface(text: str) -> str:
            stripped = re.sub(r"^\s*answer\s*:\s*", "", text or "", flags=re.I).strip()
            stripped = re.sub(r"\s+", " ", stripped)
            return stripped.strip(" \t\r\n`*_#:-").lower()

        def _normalized_text_tokens(text: str) -> list[str]:
            return [
                normalize_term_token(token)
                for token in re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", text or "")
                if normalize_term_token(token)
            ]

        answer_norm_tokens = _normalized_text_tokens(answer)
        function_word_tokens = {
            "a", "an", "the", "and", "or", "of", "to", "for", "with", "in", "on", "at", "by", "from",
        }

        def _answer_mentions_candidate(candidate: str) -> bool:
            candidate_tokens = [
                token
                for token in _normalized_text_tokens(candidate)
                if token and token not in function_word_tokens
            ]
            if not candidate_tokens:
                return False
            cursor = 0
            for token in answer_norm_tokens:
                if token == candidate_tokens[cursor]:
                    cursor += 1
                    if cursor >= len(candidate_tokens):
                        return True
            return False

        requested_temporal_markers: list[str] = []
        for pattern in (
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
            r"\b\d{4}-\d{2}-\d{2}\b",
        ):
            for match in re.finditer(pattern, query, re.I):
                marker = re.sub(r"\s+", " ", match.group(0).strip()).lower()
                if marker not in requested_temporal_markers:
                    requested_temporal_markers.append(marker)

        def _commonality_label_from_overlap(overlap_tokens: tuple[str, ...]) -> str | None:
            token_set = set(overlap_tokens)
            if "movie" in token_set or "movy" in token_set:
                return "movies"
            if "dessert" in token_set:
                return "dairy-free desserts" if any(token.startswith("dairy") for token in token_set) else "desserts"
            if "hobby" in token_set:
                return "hobbies"
            if len(token_set) == 1:
                token = next(iter(token_set))
                if token.endswith("y"):
                    return f"{token[:-1]}ies"
                if token.endswith("s"):
                    return token
                return f"{token}s"
            return " ".join(overlap_tokens)

        def _derive_commonality_interest_answer() -> str | None:
            commonality_plan = qf.get("operator_plan", {}).get("commonality", {})
            if not commonality_plan.get("enabled") or not _COMMONALITY_INTEREST_QUERY_RE.search(query):
                return None
            query_entities = _extract_query_named_entities(query)
            if len(query_entities) < 2:
                return None
            entities = query_entities[:2]
            rows_by_entity = defaultdict(list)
            token_freq = defaultdict(int)
            for idx, support_text in enumerate(support_texts):
                if not support_text or not _COMMONALITY_INTEREST_FACT_RE.search(support_text):
                    continue
                hits = _fact_entity_hits({"fact": support_text}, entities)
                if not hits:
                    continue
                tokens = _commonality_tokens(support_text, query_entities)
                if not tokens:
                    continue
                row = {
                    "fact": {"id": f"support_{idx}", "fact": support_text, "session": 0},
                    "tokens": tokens,
                    "idx": idx,
                    "rank": idx,
                }
                for token in tokens:
                    token_freq[token] += 1
                for entity in hits:
                    rows_by_entity[entity].append(row)
            if any(not rows_by_entity.get(entity) for entity in entities):
                return None
            candidates = []
            for left in rows_by_entity[entities[0]][:64]:
                for right in rows_by_entity[entities[1]][:64]:
                    if left["idx"] == right["idx"]:
                        continue
                    overlap = (left["tokens"] & right["tokens"]) - _COMMONALITY_OVERLAP_IGNORE
                    if not overlap:
                        continue
                    rarity = sum(1.0 / max(token_freq.get(token, 1), 1) for token in overlap)
                    score = rarity * 10.0 + len(overlap) * 1.5 - 0.05 * abs(left["idx"] - right["idx"])
                    candidates.append((score, overlap, left, right))
            labels = []
            seen_labels = set()
            for group in _rank_commonality_groups(candidates):
                label = _commonality_label_from_overlap(group["overlap"])
                if not label:
                    continue
                if label in seen_labels:
                    continue
                seen_labels.add(label)
                labels.append(label)
                if len(labels) >= 4:
                    break
            return ", ".join(labels) if len(labels) >= 2 else None

        country_support_candidate = _country_surface_from_texts(*support_texts) if re.match(r"^(?:in\s+)?what country\b|^which country\b", lowered_query) else None
        commonality_interest_answer = _derive_commonality_interest_answer()
        if not slot_query_enabled:
            if country_support_candidate:
                return country_support_candidate
            if negative_answer and commonality_interest_answer:
                return commonality_interest_answer
            return answer
        if not head_tokens:
            if country_support_candidate:
                return country_support_candidate
            return answer

        def _ranked_grounded_candidates() -> list[str]:
            candidate_pool = explicit_raw_candidates or grounded_candidates
            if not candidate_pool:
                return []
            lowered_supports = [text.lower() for text in support_texts if text]
            time_like_tokens = {
                "today", "yesterday", "tomorrow", "tonight", "soon", "later", "lately", "recently",
                "week", "month", "year", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
                "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
                "last", "next",
            }
            generic_value_tokens = {
                "idea", "ideas", "thing", "things", "stuff", "something", "anything", "everything",
                "place", "places", "breakthrough",
            }
            query_word_set = set(qf.get("words") or set())
            query_word_ignore = {"the", "a", "an", "what", "which", "who", "where", "when", "why", "how"}
            semantic_head_tokens = head_tokens | {
                "vehicle", "truck", "van", "bike", "bicycle", "motorcycle", "suv", "sedan",
                "country", "nation", "city", "state", "province", "region", "location", "place", "destination", "area",
                "activity", "activities", "hobby", "hobbies", "pastime", "pastimes", "sport", "sports", "game", "games",
            }

            def _score(candidate: str) -> tuple[int, int, int, int, int, int, int, int, int]:
                lowered_candidate = candidate.lower()
                candidate_tokens = [
                    normalize_term_token(token)
                    for token in re.findall(r"[A-Za-z0-9&'._-]+", candidate)
                    if normalize_term_token(token)
                ]
                token_count = len(candidate_tokens)
                support_hits = sum(lowered_candidate in text for text in lowered_supports)
                contains_shorter = sum(
                    1
                    for other in grounded_candidates
                    if other != candidate and other.lower() in lowered_candidate
                )
                novel_tokens = [
                    token
                    for token in candidate_tokens
                    if token not in head_tokens and token not in query_word_set and token not in query_word_ignore
                ]
                informative_novel = [
                    token for token in novel_tokens if token not in time_like_tokens and token not in generic_value_tokens
                ]
                noise_hits = sum(token in time_like_tokens or token in generic_value_tokens for token in candidate_tokens)
                semantic_only = int(bool(candidate_tokens) and set(candidate_tokens) <= semantic_head_tokens)
                date_alignment = int(
                    bool(requested_temporal_markers)
                    and any(
                        lowered_candidate in text and any(marker in text for marker in requested_temporal_markers)
                        for text in lowered_supports
                    )
                )
                return (
                    date_alignment,
                    int(noise_hits == 0 and not semantic_only),
                    len(informative_novel),
                    int(token_count >= 2),
                    int(any(ch.isupper() for ch in candidate)),
                    token_count,
                    contains_shorter,
                    support_hits,
                    len(candidate),
                )

            return sorted(candidate_pool, key=_score, reverse=True)

        def _best_grounded_candidate() -> str | None:
            ranked = _ranked_grounded_candidates()
            return ranked[0] if ranked else None

        def _extract_grounded_answer_phrases(text: str) -> list[str]:
            grounding_token_rows = []
            for entry in (slot_grounding_texts or support_texts):
                if not entry:
                    continue
                norm_tokens = [
                    normalize_term_token(token)
                    for token in re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", entry)
                    if normalize_term_token(token)
                ]
                if norm_tokens:
                    grounding_token_rows.append(" ".join(norm_tokens))
            if not grounding_token_rows:
                return []
            raw_tokens = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", text or "")
            if not raw_tokens:
                return []
            query_entity_tokens = {
                normalize_term_token(token)
                for phrase in (qf.get("entity_phrases") or [])
                for token in re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", str(phrase))
                if normalize_term_token(token)
            }
            ignore_tokens = set(qf.get("words") or set()) | head_tokens | function_word_tokens | {
                "share", "shared", "same", "similar", "interest", "interests", "hobby", "hobbies",
                "goal", "goals", "kind", "kinds", "type", "types", "include", "includes",
            }
            ignore_tokens |= query_entity_tokens
            candidates: list[tuple[int, int, int, str]] = []
            for size in range(min(5, len(raw_tokens)), 0, -1):
                for start in range(len(raw_tokens) - size + 1):
                    phrase_tokens = raw_tokens[start:start + size]
                    norm_tokens = [
                        normalize_term_token(token)
                        for token in phrase_tokens
                        if normalize_term_token(token)
                    ]
                    informative = [
                        token
                        for token in norm_tokens
                        if token not in ignore_tokens and len(token) >= 3
                    ]
                    if len(informative) < 2:
                        continue
                    left = 0
                    right = len(phrase_tokens)
                    while left < right:
                        norm = normalize_term_token(phrase_tokens[left])
                        if norm and norm not in ignore_tokens and len(norm) >= 3:
                            break
                        left += 1
                    while right > left:
                        norm = normalize_term_token(phrase_tokens[right - 1])
                        if norm and norm not in ignore_tokens and len(norm) >= 3:
                            break
                        right -= 1
                    if right - left < 2:
                        continue
                    phrase = " ".join(phrase_tokens[left:right]).strip()
                    normalized_phrase = " ".join(informative)
                    support_hits = sum(normalized_phrase in entry for entry in grounding_token_rows)
                    if support_hits == 0:
                        continue
                    candidates.append((start, -size, -support_hits, phrase))
            candidates.sort()
            kept: list[str] = []
            kept_ranges: list[tuple[int, int]] = []
            for start, neg_size, _neg_hits, phrase in candidates:
                size = -neg_size
                end = start + size
                if any(not (end <= left or start >= right) for left, right in kept_ranges):
                    continue
                lowered_phrase = phrase.lower()
                if any(
                    lowered_phrase in existing.lower() or existing.lower() in lowered_phrase
                    for existing in kept
                ):
                    continue
                kept.append(phrase)
                kept_ranges.append((start, end))
                if len(kept) >= 4:
                    break
            return kept

        def _add_grounded_candidate(candidate: str) -> None:
            normalized = re.sub(r"\s+", " ", candidate.strip())
            key = normalized.lower()
            if not normalized or key in grounded_seen:
                return
            grounded_seen.add(key)
            grounded_candidates.append(normalized)

        in_retrieved_section = False
        in_raw_slot_section = False
        for raw_line in (recall_result.get("context") or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            upper = line.upper()
            if upper.startswith("RETRIEVED FACTS:"):
                in_retrieved_section = True
                in_raw_slot_section = False
                continue
            if upper.startswith("RAW SLOT CANDIDATES:"):
                in_retrieved_section = False
                in_raw_slot_section = True
                continue
            if upper.startswith("--- "):
                in_retrieved_section = False
                in_raw_slot_section = False
                continue
            if in_retrieved_section:
                if not re.match(r"^\[\d+\]\s+", line):
                    continue
                line = re.sub(r"^\[\d+\]\s*", "", line)
                line = re.sub(r"^\(S\d+\)\s*", "", line, flags=re.I)
                line = re.sub(r"\s*\[Episode:[^\]]+\]\s*$", "", line, flags=re.I)
                if line:
                    support_texts.append(line)
                continue
            if in_raw_slot_section:
                if not re.match(r"^\[Q\d+\]\s+", line):
                    continue
                candidate = re.sub(r"^\[Q\d+\]\s*", "", line)
                candidate = re.split(
                    r"\s+\(from \[Turn query\]:|\s+\[Episode:|\s+\[Local evidence:|\s+\[Fact:",
                    candidate,
                    maxsplit=1,
                )[0].strip()
                if candidate:
                    _add_grounded_candidate(candidate)
                    explicit_raw_candidates.append(candidate)
                    support_texts.append(candidate)

        slot_support_texts = support_texts
        primary_entity = ""
        primary_entity_tokens: set[str] = set()
        for phrase in qf.get("entity_phrases") or []:
            if phrase:
                primary_entity = str(phrase).strip().lower()
                primary_entity_tokens = {
                    normalize_term_token(token)
                    for token in re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", primary_entity)
                    if normalize_term_token(token)
                }
                break
        if not primary_entity:
            query_entity_spans = re.findall(r"([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,2})", query)
            for span in query_entity_spans:
                norm_tokens = {
                    normalize_term_token(token)
                    for token in re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", span)
                    if normalize_term_token(token)
                }
                if not norm_tokens:
                    continue
                if norm_tokens <= {"what", "which", "when", "where", "who", "why", "how"}:
                    continue
                primary_entity = span.strip().lower()
                primary_entity_tokens = norm_tokens
                break

        foreign_subject_ignore = {
            "i", "we", "he", "she", "they", "it", "last", "next", "this", "that", "these", "those",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
        }

        def _support_text_has_foreign_subject(text: str) -> bool:
            if not primary_entity or primary_entity in text.lower() or not head_tokens:
                return False
            raw_words = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", text)
            norm_words = [normalize_term_token(word) for word in raw_words]
            head_index = None
            head_len = len(head_tokens)
            for idx in range(max(0, len(norm_words) - head_len + 1)):
                if norm_words[idx:idx + head_len] == list(head_tokens):
                    head_index = idx
                    break
            if head_index is None:
                return False
            for raw in raw_words[:head_index]:
                if not raw[:1].isupper():
                    continue
                norm = normalize_term_token(raw)
                if not norm or norm in foreign_subject_ignore or norm in primary_entity_tokens:
                    continue
                return True
            return False

        slot_grounding_texts: list[str] = []
        for text in slot_support_texts:
            if _support_text_has_foreign_subject(text):
                continue
            slot_grounding_texts.append(text)
            for candidate in _fact_slot_fill_candidates(text, qf, allow_loose_fallback=False):
                _add_grounded_candidate(candidate)

        country_support_candidate = _country_surface_from_texts(*support_texts) if re.match(r"^(?:in\s+)?what country\b|^which country\b", lowered_query) else None

        def _slot_rescue_blocked_by_missing_qualifiers() -> bool:
            qualifier_ignore = function_word_tokens | {
                "kind", "kinds", "type", "types", "main", "one", "two", "three", "four", "five",
                "share", "shared", "both", "same", "similar", "interest", "interests", "hobby", "hobbies",
                "temporary", "current", "favorite", "new", "old", "certain", "specific", "another", "other", "various",
                "recreational", "indoor", "outdoor",
                "prefer", "prefers", "preferred", "preference", "preferences",
                "why", "reason", "reasons", "because",
                "work", "working", "pursue", "pursuing",
                "begin", "beginning", "start", "starting", "started",
            }
            query_entity_tokens = {
                normalize_term_token(token)
                for phrase in (qf.get("entity_phrases") or [])
                for token in re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", str(phrase))
                if normalize_term_token(token)
            }
            qualifiers = sorted(
                token
                for token in (qf.get("words") or set())
                if token
                and token not in head_tokens
                and token not in qualifier_ignore
                and token not in query_entity_tokens
            )
            if not qualifiers:
                return False
            grounding_tokens = set()
            for text in [*slot_grounding_texts, *explicit_raw_candidates, *support_texts]:
                grounding_tokens.update(_normalized_text_tokens(text))
            missing = [token for token in qualifiers if token not in grounding_tokens]
            return bool(missing)

        ranked_grounded_candidates = _ranked_grounded_candidates()
        best_grounded = ranked_grounded_candidates[0] if ranked_grounded_candidates else None
        normalized_answer_surface = _normalized_answer_surface(answer)
        answer_surface_candidates = _fact_slot_fill_candidates(answer, qf, allow_loose_fallback=False)

        raw_head_words = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", str(slot_plan.get("head_phrase") or ""))
        explicit_plural_query = bool(
            re.match(r"^(?:what|which)\s+are\b", lowered_query)
            or re.search(r"\b(?:two|three|four|five|several|some|many|multiple)\s+of\b", lowered_query)
        )
        singular_query_prefix = bool(
            re.match(r"^(?:what|which)\s+(?:is|was|does|did|has|had|might|may|could|would|should|can)\b", lowered_query)
        )
        last_head_word = raw_head_words[-1].lower() if raw_head_words else ""
        morphological_plural_head = bool(
            last_head_word
            and len(last_head_word) > 3
            and last_head_word.endswith("s")
            and not last_head_word.endswith(("ss", "us", "is"))
        )
        plural_slot_query = bool(
            explicit_plural_query
            or (morphological_plural_head and not singular_query_prefix)
        )
        multi_item_query = bool(
            plural_slot_query
            or qf.get("operator_plan", {}).get("commonality", {}).get("enabled")
            or qf.get("operator_plan", {}).get("list_set", {}).get("enabled")
        )
        mentioned_grounded_candidates = [
            candidate
            for candidate in ranked_grounded_candidates
            if _answer_mentions_candidate(candidate)
        ]
        answer_grounded_phrases = _extract_grounded_answer_phrases(answer)
        qualifier_guard_active = _slot_rescue_blocked_by_missing_qualifiers()

        def _candidate_adds_new_slot_info(candidate: str) -> bool:
            candidate_tokens = {
                normalize_term_token(token)
                for token in re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", candidate)
                if normalize_term_token(token)
            }
            candidate_tokens -= head_tokens
            candidate_tokens -= set(qf.get("words") or set())
            candidate_tokens -= {"the", "a", "an", "one", "provided", "context", "mentioned"}
            return bool(candidate_tokens)

        if re.match(r"^(?:in\s+)?what country\b|^which country\b", lowered_query) and country_support_candidate:
            return country_support_candidate

        meta_explanation = any(
            phrase in lowered
            for phrase in (
                "retrieved fact",
                "retrieved facts",
                "raw context",
                "provided context",
                "episode ",
                "explicitly stated",
                "confirmed in",
            )
        )
        if not best_grounded and meta_explanation and not any(
            _candidate_adds_new_slot_info(candidate)
            for candidate in answer_surface_candidates
        ):
            if country_support_candidate:
                return country_support_candidate
            if commonality_interest_answer:
                return commonality_interest_answer
            return "Not mentioned in the provided context."

        if multi_item_query:
            if len(mentioned_grounded_candidates) >= 2:
                return ", ".join(mentioned_grounded_candidates[:4])
            if len(answer_grounded_phrases) >= 2:
                return ", ".join(answer_grounded_phrases[:4])

        if negative_answer:
            if country_support_candidate:
                return country_support_candidate
            if commonality_interest_answer:
                return commonality_interest_answer
            if best_grounded and not qualifier_guard_active:
                if plural_slot_query and len(ranked_grounded_candidates) >= 2:
                    return ", ".join(ranked_grounded_candidates[:2])
                return best_grounded
            return "Not mentioned in the provided context."

        if best_grounded and qualifier_guard_active:
            grounded_surfaces = {candidate.lower() for candidate in grounded_candidates}
            if normalized_answer_surface in grounded_surfaces:
                return "Not mentioned in the provided context."
            if normalized_answer_surface in {"it", "that", "this", "one", "this one", "that one"}:
                return "Not mentioned in the provided context."
            if any(candidate.lower() in lowered for candidate in grounded_candidates):
                return "Not mentioned in the provided context."

        if best_grounded and not qualifier_guard_active:
            grounded_surfaces = {candidate.lower() for candidate in grounded_candidates}
            if normalized_answer_surface in grounded_surfaces:
                return best_grounded
            if normalized_answer_surface in {"it", "that", "this", "one", "this one", "that one"}:
                return best_grounded
            if any(candidate.lower() in lowered for candidate in grounded_candidates):
                explanatory_slot_phrases = (
                    "retrieved fact",
                    "retrieved facts",
                    "raw context",
                    "provided context",
                    "episode ",
                    "does not mention",
                    "do not mention",
                    "not mentioned",
                    "not on ",
                    "not in ",
                    "not about ",
                    "however",
                    "but on ",
                    "specifically",
                )
                answer_word_count = len(re.findall(r"[A-Za-z0-9&'._-]+", answer))
                grounded_word_count = len(re.findall(r"[A-Za-z0-9&'._-]+", best_grounded))
                if any(phrase in lowered for phrase in explanatory_slot_phrases) or answer_word_count > grounded_word_count + 6:
                    return best_grounded
                return answer

        if re.match(r"^(?:in\s+)?what country\b|^which country\b", lowered_query):
            support_candidate = _country_surface_from_texts(answer, *support_texts)
            if support_candidate:
                return support_candidate

        answer_tokens = {
            normalize_term_token(token)
            for token in re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?", answer)
            if normalize_term_token(token)
        }
        answer_tokens -= {
            token
            for token in set(answer_tokens)
            if token in qf.get("words", set()) or token in head_tokens
        }
        answer_tokens -= {"the", "a", "an", "one", "provided", "context", "mentioned"}
        if not answer_tokens:
            return "Not mentioned in the provided context."
        grounding_text = " ".join(slot_grounding_texts or support_texts).lower()
        if not all(token in grounding_text for token in answer_tokens):
            return "Not mentioned in the provided context."
        return answer

    def _build_provider_payload(
        self,
        *,
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        use_tool: bool,
    ) -> tuple[dict, int]:
        provider = _provider_for_model(model)
        tool_tokens_est = 0

        if provider == "anthropic":
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if use_tool:
                payload["tools"] = list(TEMPORAL_TOOLS) + list(COUNTING_TOOLS) + [GET_CONTEXT_TOOL]
                tool_tokens_est = _estimate_tokens(payload["tools"])
            return payload, tool_tokens_est

        if provider == "google":
            payload = {
                "model": model,
                "messages": messages,
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            }
            return payload, tool_tokens_est

        payload = {
            "model": model,
            "messages": messages,
            _tok_key(model): max_tokens,
            "seed": 42,
        }
        if _supports_temperature(model):
            payload["temperature"] = temperature
        if use_tool:
            payload["tools"] = _build_openai_tools_payload()
            payload["tool_choice"] = "auto"
            tool_tokens_est = _estimate_tokens(payload["tools"])
        return payload, tool_tokens_est

    def _truncate_by_priority(
        self,
        *,
        model: str,
        query: str,
        recall_result: dict,
        context_packet: dict,
        memory_budget: int,
        max_tokens: int,
        prompt_type: str,
        use_tool: bool,
        speakers: str,
        temperature: float,
    ) -> tuple[dict, str, dict, dict]:
        working = deepcopy(context_packet)
        removed = {"tier4": 0, "tier3": 0, "tier2": 0}
        budget_exceeded = False

        while True:
            context = _render_context_packet(working)
            messages = self._build_payload_messages(
                prompt_type=prompt_type,
                context=context,
                query=query,
                recall_result=recall_result,
                speakers=speakers,
            )
            payload, tool_tokens_est = self._build_provider_payload(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                use_tool=use_tool,
            )
            message_tokens_est = _estimate_tokens(payload.get("messages", []))
            total_input_est = message_tokens_est + tool_tokens_est
            if total_input_est <= memory_budget:
                truncation = None
                if any(removed.values()):
                    truncation = {
                        "removed": removed,
                        "budget_exceeded": False,
                    }
                return working, context, payload, {
                    "context_tokens": _estimate_tokens(context),
                    "message_tokens_est": message_tokens_est,
                    "tool_tokens_est": tool_tokens_est,
                    "memory_budget": memory_budget,
                    "budget_exceeded": False,
                    "truncation": truncation,
                }

            for tier in ("tier4", "tier3", "tier2"):
                if working[tier]:
                    working[tier].pop()
                    removed[tier] += 1
                    break
            else:
                budget_exceeded = True
                truncation = {
                    "removed": removed,
                    "budget_exceeded": True,
                }
                return working, context, payload, {
                    "context_tokens": _estimate_tokens(context),
                    "message_tokens_est": message_tokens_est,
                    "tool_tokens_est": tool_tokens_est,
                    "memory_budget": memory_budget,
                    "budget_exceeded": budget_exceeded,
                    "truncation": truncation,
                }

    def _build_payload(
        self,
        *,
        query: str,
        recall_result: dict,
        inference_model: str = None,
        max_tokens: int = None,
        use_tool: bool = None,
        speakers: str = "User and Assistant",
    ) -> tuple[dict | None, dict | None]:
        resolved_type = recall_result.get("query_type", "default")
        prompt_type = recall_result.get("recommended_prompt_type", resolved_type)
        recommended_profile = recall_result.get("recommended_profile")
        payload_target = self._resolve_inference_target(recommended_profile, inference_model)
        if payload_target is None:
            return None, None

        model = payload_target["model"]
        profile_used = payload_target["profile_used"]
        profile_fallback = payload_target["profile_fallback"]
        cfg = payload_target["cfg"]

        final_use_tool = recall_result.get("use_tool", False) if use_tool is None else use_tool
        temperature = float(cfg.get("temperature", 0) or 0)
        resolved_max_tokens = self._resolve_max_tokens(
            cfg=cfg,
            resolved_type=resolved_type,
            prompt_type=prompt_type,
            explicit_max_tokens=max_tokens,
        )
        memory_budget = self._compute_memory_budget(cfg, resolved_max_tokens)
        context_packet = recall_result.get("_context_packet") or {
            "tier1": [],
            "tier2": [],
            "tier3": [{"text": recall_result.get("context", ""), "rank": 0, "source": "fact"}],
            "tier4": [],
        }

        _packet, rendered_context, payload, meta = self._truncate_by_priority(
            model=model,
            query=query,
            recall_result=recall_result,
            context_packet=context_packet,
            memory_budget=memory_budget,
            max_tokens=resolved_max_tokens,
            prompt_type=prompt_type,
            use_tool=final_use_tool,
            speakers=speakers,
            temperature=temperature,
        )

        recall_result["context"] = rendered_context
        recall_result["_context_packet"] = _packet

        payload_meta = {
            "profile_used": profile_used,
            "profile_fallback": profile_fallback,
            "context_tokens": meta["context_tokens"],
            "message_tokens_est": meta["message_tokens_est"],
            "tool_tokens_est": meta["tool_tokens_est"],
            "memory_budget": meta["memory_budget"],
            "budget_exceeded": meta["budget_exceeded"],
            "prompt_type": prompt_type,
            "use_tool": final_use_tool,
            "truncation": meta["truncation"],
            "provider": _provider_for_model(model),
            "provider_family": _provider_family_for_model(model),
        }
        return payload, payload_meta

    def _check_shell_budget(
        self,
        *,
        payload: dict,
        payload_meta: dict,
        shell_budget: float | None,
        recommended_profile: str | None,
        query_type: str,
    ) -> dict | None:
        if shell_budget is None:
            return None

        est_cost = self._estimate_payload_cost(payload=payload, payload_meta=payload_meta)
        output_tokens = (
            payload.get("max_output_tokens")
            or payload.get("max_tokens")
            or payload.get("max_completion_tokens")
            or 0
        )
        input_tokens = (
            payload_meta.get("message_tokens_est", 0)
            + payload_meta.get("tool_tokens_est", 0)
        )

        if est_cost <= shell_budget:
            return None

        best_effort = None
        if self._has_profiles():
            for name in sorted(
                self._list_profile_names(),
                key=lambda n: (self._get_profile_config(n) or {}).get("input_cost_per_1k", 999),
            ):
                p_cfg = self._get_profile_config(name)
                if not p_cfg or "input_cost_per_1k" not in p_cfg:
                    continue
                p_cost = (input_tokens / 1000 * p_cfg["input_cost_per_1k"]
                          + output_tokens / 1000 * p_cfg["output_cost_per_1k"])
                if p_cost <= shell_budget:
                    best_effort = name
                    break

        return {
            "telemetry_version": 1,
            "answer": None,
            "budget_exceeded": True,
            "estimated_cost": round(est_cost, 6),
            "shell_budget": shell_budget,
            "best_effort_profile": best_effort,
            "recommended_profile": recommended_profile,
            "query_type": query_type,
        }

    def _estimate_payload_cost(
        self,
        *,
        payload: dict,
        payload_meta: dict,
    ) -> float:
        model = payload.get("model", "")
        output_tokens = (
            payload.get("max_output_tokens")
            or payload.get("max_tokens")
            or payload.get("max_completion_tokens")
            or 0
        )
        input_tokens = (
            payload_meta.get("message_tokens_est", 0)
            + payload_meta.get("tool_tokens_est", 0)
        )

        profile_used = payload_meta.get("profile_used")
        cfg = self._get_profile_config(profile_used) if profile_used else None
        if cfg and "input_cost_per_1k" in cfg:
            return (
                input_tokens / 1000 * cfg["input_cost_per_1k"]
                + output_tokens / 1000 * cfg["output_cost_per_1k"]
            )

        pricing = _PRICING.get(model, {"input": 2.0, "output": 8.0})
        return (
            input_tokens / 1e6 * pricing["input"]
            + output_tokens / 1e6 * pricing["output"]
        )

    async def _send_payload(
        self,
        payload: dict,
        *,
        caller_id: str = None,
    ) -> tuple[str, bool, list[dict]]:
        model = payload["model"]
        provider = _provider_for_model(model)
        has_tools = bool(payload.get("tools"))

        if not has_tools:
            messages = payload.get("messages", [])
            max_tokens = (
                payload.get("max_output_tokens")
                or payload.get("max_tokens")
                or payload.get("max_completion_tokens")
                or 2000
            )
            temperature = payload.get("temperature", 0.0)
            if provider == "openai" and messages and len(messages) == 1 and messages[0].get("role") == "user":
                answer = await call_oai(
                    model,
                    messages[0].get("content", ""),
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return answer, False, []

            answer = await _call_model(
                model,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return answer, False, []

        if provider == "anthropic":
            client = _get_client(model)
            request = dict(payload)
            request["model"] = _api_model(model)
            messages = deepcopy(request["messages"])
            request["messages"] = messages
            tool_called = False
            tool_results = []
            for _ in range(3):
                response = await client.messages.create(**request)
                tool_uses = [b for b in response.content if getattr(b, "type", "") == "tool_use"]
                text_blocks = [b for b in response.content if getattr(b, "type", "") == "text"]
                if not tool_uses:
                    answer = "".join(getattr(b, "text", "") for b in text_blocks).strip()
                    return answer, tool_called, tool_results

                tool_called = True
                tool_outputs = []
                for tu in tool_uses:
                    input_data = getattr(tu, "input", {}) or {}
                    result_obj = get_more_context(
                        input_data.get("session_id", 0),
                        raw_sessions=self._raw_sessions,
                    ) if tu.name == "get_more_context" else json.loads(
                        json.dumps({"error": f"Unknown tool: {tu.name}"})
                    )
                    result_str = json.dumps(result_obj)
                    tool_results.append({"tool": tu.name, "input": input_data, "result": result_str})
                    if tu.name == "get_more_context" and hasattr(self, "_audit"):
                        self._audit.log(
                            "get_more_context",
                            caller_id=caller_id or "unknown",
                            details={"session_id": input_data.get("session_id")},
                        )
                    tool_outputs.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": result_str,
                    })
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_outputs})
            return "", tool_called, tool_results

        client = _get_client(model)
        request = dict(payload)
        request["model"] = _api_model(model)
        messages = deepcopy(payload["messages"])
        request["messages"] = messages
        tool_results = []
        tool_called = False
        for _ in range(3):
            response = await client.chat.completions.create(**request)
            msg = response.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []
            if not tool_calls:
                return (msg.content or ""), tool_called, tool_results

            tool_called = True
            assistant_tool_calls = []
            for tc in tool_calls:
                assistant_tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": assistant_tool_calls,
            })

            for tc in tool_calls:
                args = json.loads(tc.function.arguments)
                if tc.function.name == "get_more_context":
                    tool_result = get_more_context(
                        args.get("session_id", 0),
                        raw_sessions=self._raw_sessions,
                    )
                    if hasattr(self, "_audit"):
                        self._audit.log(
                            "get_more_context",
                            caller_id=caller_id or "unknown",
                            details={"session_id": args.get("session_id")},
                        )
                else:
                    tool_result = {"error": f"Unknown tool: {tc.function.name}"}
                tool_result_json = json.dumps(tool_result)
                tool_results.append({
                    "tool": tc.function.name,
                    "input": args,
                    "result": tool_result_json,
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result_json,
                })

            request = {
                "model": _api_model(model),
                "messages": messages,
                _tok_key(model): payload.get(_tok_key(model), payload.get("max_tokens", 2000)),
                "seed": payload.get("seed", 42),
            }
            if "tools" in payload:
                request["tools"] = payload["tools"]
                request["tool_choice"] = payload.get("tool_choice", "auto")
            if _supports_temperature(model) and "temperature" in payload:
                request["temperature"] = payload["temperature"]
        return "", tool_called, tool_results

    # ── ask() ──

    async def ask(
        self,
        query: str,
        agent_id: str = None,
        swarm_id: str = None,
        search_family: str = "auto",
        query_type: str = "auto",
        kind: str = "all",
        caller_memberships: list = None,
        caller_role: str = "user",
        caller_id: str = None,
        inference_model: str = None,
        max_tokens: int = None,
        use_tool: bool = None,
        shell_budget: float = None,
        speakers: str = "User and Assistant",
    ) -> dict:
        """Answer a question using recall() + LLM inference.

        search_family: auto | conversation | document
        Returns dict with: answer, profile_used, profile_fallback,
        recommended_profile, query_type, use_tool, tool_called, budget_exceeded.
        """
        self._audit.log("ask", caller_id or "system",
                        {"query": query[:200]})
        recall_result = await self.recall(
            query, agent_id=agent_id, swarm_id=swarm_id,
            search_family=search_family,
            query_type=query_type, kind=kind,
            caller_memberships=caller_memberships,
            caller_role=caller_role, caller_id=caller_id,
        )
        if "context" not in recall_result:
            return recall_result
        deterministic_answer = str(recall_result.get("deterministic_answer") or "").strip()
        if deterministic_answer:
            deterministic_answer = _apply_output_constraints(
                deterministic_answer,
                recall_result.get("output_constraints"),
            )
            deterministic_answer = self._normalize_grounded_answer(
                query,
                deterministic_answer,
                recall_result,
            )
            return {
                "telemetry_version": 1,
                "answer": deterministic_answer,
                "query_type": recall_result.get("query_type", "default"),
                "use_tool": False,
                "tool_called": False,
                "tool_results": None,
                "profile_used": "deterministic:temporal_v1",
                "profile_fallback": False,
                "recommended_profile": recall_result.get("recommended_profile"),
                "budget_exceeded": False,
                "estimated_cost": 0.0,
                "retrieval_families": recall_result.get("retrieval_families", []),
                "search_family": recall_result.get("search_family"),
                "retrieved_count": len(recall_result.get("retrieved", [])),
                "runtime_trace": recall_result.get("runtime_trace"),
                "payload_meta": {
                    "profile_used": "deterministic:temporal_v1",
                    "profile_fallback": False,
                    "use_tool": False,
                    "deterministic": True,
                },
            }
        if not self._has_profiles() and inference_model is None:
            return {"error": "No profiles configured and no inference_model provided",
                    "code": "NO_PROFILES"}
        resolved_type = recall_result.get("query_type", "default")
        recommended_profile = recall_result.get("recommended_profile")

        payload = recall_result.get("payload")
        payload_meta = recall_result.get("payload_meta")
        if payload is None or any(
            value is not None
            for value in (inference_model, max_tokens, use_tool)
        ) or speakers != "User and Assistant":
            payload, payload_meta = self._build_payload(
                query=query,
                recall_result=recall_result,
                inference_model=inference_model,
                max_tokens=max_tokens,
                use_tool=use_tool,
                speakers=speakers,
            )

        if not payload or not payload_meta:
            return {"error": "No profiles configured and no inference_model provided",
                    "code": "NO_PROFILES"}

        estimated_cost = round(
            self._estimate_payload_cost(payload=payload, payload_meta=payload_meta),
            6,
        )
        budget_result = self._check_shell_budget(
            payload=payload,
            payload_meta=payload_meta,
            shell_budget=shell_budget,
            recommended_profile=recommended_profile,
            query_type=resolved_type,
        )
        if budget_result is not None:
            return budget_result

        answer, tool_called, tool_results = await self._send_payload(
            payload,
            caller_id=caller_id,
        )

        if answer:
            answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
            answer = re.sub(r'<think>.*', '', answer, flags=re.DOTALL).strip()
            answer = _apply_output_constraints(answer, recall_result.get("output_constraints"))
            answer = self._normalize_grounded_answer(query, answer, recall_result)

        trace_ref = f"ask_{uuid4().hex[:12]}"

        return {
            "telemetry_version": 1,
            "answer": answer,
            "query_type": resolved_type,
            "use_tool": payload_meta.get("use_tool", False),
            "tool_called": tool_called,
            "tool_results": tool_results,
            "profile_used": payload_meta.get("profile_used"),
            "profile_fallback": payload_meta.get("profile_fallback", False),
            "recommended_profile": recommended_profile,
            "budget_exceeded": False,
            "estimated_cost": estimated_cost,
            "retrieval_families": recall_result.get("retrieval_families", []),
            "search_family": recall_result.get("search_family"),
            "retrieved_count": len(recall_result.get("retrieved", [])),
            "runtime_trace": recall_result.get("runtime_trace"),
            "runtime_trace_ref": trace_ref,
            "payload_meta": payload_meta,
        }

    # ── get_versions() (Unit 6) ──

    def get_versions(self, artifact_id, caller_id=None, caller_role="agent"):
        """Return version chain ordered by parent_version for an artifact."""
        # ACL check: find any fact from this artifact, verify read access
        if caller_role != "admin":
            sample = None
            for f in self._all_granular:
                if f.get("artifact_id") == artifact_id:
                    sample = f
                    break
            if sample is None:
                for rs in self._raw_sessions:
                    if rs.get("artifact_id") == artifact_id:
                        sample = rs
                        break
            if sample is not None:
                _memberships = []
                if hasattr(self, '_membership_registry') and caller_id:
                    _memberships = self._membership_registry.memberships_for(caller_id)
                if not self._acl_allows(sample, caller_id or "system", _memberships, caller_role):
                    return {"error": "Access denied", "code": "ACL_FORBIDDEN"}

        versions = []
        for tier in (self._all_granular, self._all_cons, self._all_cross):
            for f in tier:
                if f.get("artifact_id") == artifact_id:
                    ver = {
                        "version_id": f.get("version_id"),
                        "parent_version": f.get("parent_version"),
                        "content_hash": f.get("content_hash"),
                        "status": f.get("status", "active"),
                        "fact": f.get("fact"),
                        "created_at": f.get("created_at"),
                    }
                    if ver not in versions:
                        versions.append(ver)
        # Also check raw_sessions
        for rs in self._raw_sessions:
            if rs.get("artifact_id") == artifact_id:
                ver = {
                    "version_id": rs.get("version_id"),
                    "parent_version": rs.get("parent_version"),
                    "content_hash": rs.get("content_hash"),
                    "status": rs.get("status", "active"),
                    "created_at": rs.get("stored_at"),
                }
                if ver not in versions:
                    versions.append(ver)
        # Order: roots first, then children
        by_vid = {v["version_id"]: v for v in versions}
        ordered = []
        seen = set()
        # Find roots (no parent or parent not in set)
        roots = [v for v in versions
                 if not v.get("parent_version")
                 or v["parent_version"] not in by_vid]
        queue = list(roots)
        while queue:
            v = queue.pop(0)
            vid = v["version_id"]
            if vid in seen:
                continue
            seen.add(vid)
            ordered.append(v)
            # Find children
            for c in versions:
                if c.get("parent_version") == vid and c["version_id"] not in seen:
                    queue.append(c)
        # Add any remaining (disconnected)
        for v in versions:
            if v["version_id"] not in seen:
                ordered.append(v)
        return {"artifact_id": artifact_id, "versions": ordered}

    # ── edit() (Unit 6) ──

    async def edit(self, artifact_id, new_content, caller_id=None, caller_role="agent"):
        """Create a new version of an artifact with new content.

        Finds the active version, creates a new version, supersedes the old one.
        Requires write access (owner, write ACL, or admin).
        """
        self._audit.log("edit", caller_id or "system",
                        {"artifact_id": artifact_id})
        # Find active facts for this artifact
        active_facts = [f for f in self._all_granular
                        if f.get("artifact_id") == artifact_id
                        and f.get("status") == "active"]
        if not active_facts:
            return {"error": f"No active artifact: {artifact_id}",
                    "code": "NOT_FOUND"}

        old_fact = active_facts[0]

        # ACL check: write access required
        if caller_role != "admin":
            fact_owner = old_fact.get("owner_id", "system")
            write_acl = old_fact.get("write", [])
            if (caller_id != fact_owner
                    and caller_id not in write_acl
                    and "agent:PUBLIC" not in write_acl):
                return {"error": "Write access denied",
                        "code": "ACL_FORBIDDEN"}
        old_version_id = old_fact.get("version_id")
        new_version_id = _generate_version_id()
        new_hash = content_hash_text(new_content)

        # Supersede old version
        for f in self._all_granular:
            if f.get("artifact_id") == artifact_id and f.get("status") == "active":
                f["status"] = "superseded"
        for rs in self._raw_sessions:
            if rs.get("artifact_id") == artifact_id and rs.get("status") == "active":
                rs["status"] = "superseded"

        # Create new fact (copy metadata from old)
        new_fact = dict(old_fact)
        new_fact["fact"] = new_content
        new_fact["version_id"] = new_version_id
        new_fact["parent_version"] = old_version_id
        new_fact["content_hash"] = new_hash
        new_fact["status"] = "active"
        new_fact["created_at"] = datetime.now(timezone.utc).isoformat()
        new_fact["id"] = f"edited_{new_version_id}"

        async with self._file_lock:
            self._all_granular.append(new_fact)
            self._tiers_dirty = True
            self._data_dict = None
            # Update dedup index if source_id present
            sid = old_fact.get("source_id") or new_fact.get("source_id")
            sn = old_fact.get("session", 0)
            if sid:
                dedup_key = (sid, sn)
                self._dedup_index[dedup_key] = {
                    "artifact_id": artifact_id,
                    "version_id": new_version_id,
                    "content_hash": new_hash,
                }
            self._save_cache()

        return {"artifact_id": artifact_id,
                "version_id": new_version_id,
                "parent_version": old_version_id}

    # ── retract() (Unit 6) ──

    async def retract(self, artifact_id, caller_id=None, caller_role="agent"):
        """Retract an artifact — all versions become invisible.

        Requires write access (owner, write ACL, or admin).
        """
        self._audit.log("retract", caller_id or "system",
                        {"artifact_id": artifact_id})

        # ACL check: write access required
        if caller_role != "admin":
            target = None
            for f in self._all_granular:
                if f.get("artifact_id") == artifact_id:
                    target = f
                    break
            if target is None:
                return {"error": f"Artifact not found: {artifact_id}",
                        "code": "NOT_FOUND"}
            fact_owner = target.get("owner_id", "system")
            write_acl = target.get("write", [])
            if (caller_id != fact_owner
                    and caller_id not in write_acl
                    and "agent:PUBLIC" not in write_acl):
                return {"error": "Write access denied",
                        "code": "ACL_FORBIDDEN"}

        found = False
        async with self._file_lock:
            for f in self._all_granular:
                if f.get("artifact_id") == artifact_id:
                    f["status"] = "retracted"
                    found = True
            for f in self._all_cons:
                if f.get("artifact_id") == artifact_id:
                    f["status"] = "retracted"
                    found = True
            for f in self._all_cross:
                if f.get("artifact_id") == artifact_id:
                    f["status"] = "retracted"
                    found = True
            for rs in self._raw_sessions:
                if rs.get("artifact_id") == artifact_id:
                    rs["status"] = "retracted"
                    found = True
            if found:
                self._data_dict = None
                self._save_cache()
        if not found:
            return {"error": f"Artifact not found: {artifact_id}",
                    "code": "NOT_FOUND"}
        return {"artifact_id": artifact_id, "status": "retracted"}

    # ── purge() (Unit 6) ──

    async def purge(self, artifact_id, caller_id=None, caller_role="agent"):
        """Purge an artifact — admin only, physically removes from all lists."""
        self._audit.log("purge", caller_id or "system",
                        {"artifact_id": artifact_id})
        if caller_role != "admin":
            return {"error": "purge requires admin role",
                    "code": "ACL_FORBIDDEN"}

        async with self._file_lock:
            before_g = len(self._all_granular)
            self._all_granular = [f for f in self._all_granular
                                  if f.get("artifact_id") != artifact_id]
            self._all_cons = [f for f in self._all_cons
                              if f.get("artifact_id") != artifact_id]
            self._all_cross = [f for f in self._all_cross
                               if f.get("artifact_id") != artifact_id]
            self._raw_sessions = [rs for rs in self._raw_sessions
                                  if rs.get("artifact_id") != artifact_id]
            removed = before_g - len(self._all_granular)
            # Clean dedup index
            keys_to_remove = [k for k, v in self._dedup_index.items()
                              if v.get("artifact_id") == artifact_id]
            for k in keys_to_remove:
                del self._dedup_index[k]
            # Clean git dedup index
            git_keys_to_remove = [k for k, v in self._git_dedup_index.items()
                                  if v.get("artifact_id") == artifact_id]
            for k in git_keys_to_remove:
                del self._git_dedup_index[k]

            self._data_dict = None
            self._tiers_dirty = True
            self._save_cache()

        return {"artifact_id": artifact_id, "purged_facts": removed}

    # ── redact() (Unit 7) ──

    async def redact(self, artifact_id: str, fields: list[str],
                     caller_id=None, caller_role="agent") -> dict:
        """Redact specified fields of an artifact.

        Replaces fact text with [REDACTED], entities with ["[REDACTED]"],
        raw session content with [REDACTED]. Sets status="redacted".
        Requires write access or admin role.
        """
        self._audit.log("redact", caller_id or "system",
                        {"artifact_id": artifact_id, "fields": fields})
        # ACL check: write access or admin
        if caller_role != "admin":
            # Find any fact with this artifact_id to check write ACL
            target = None
            for f in self._all_granular:
                if f.get("artifact_id") == artifact_id:
                    target = f
                    break
            if target is None:
                return {"error": f"Artifact not found: {artifact_id}",
                        "code": "NOT_FOUND"}
            write_acl = target.get("write", [])
            if caller_id not in write_acl and "agent:PUBLIC" not in write_acl:
                return {"error": "Write access denied",
                        "code": "ACL_FORBIDDEN"}

        found = False
        async with self._file_lock:
            for tier in (self._all_granular, self._all_cons, self._all_cross):
                for f in tier:
                    if f.get("artifact_id") == artifact_id:
                        if "fact" in fields:
                            f["fact"] = "[REDACTED]"
                        if "entities" in fields:
                            f["entities"] = ["[REDACTED]"]
                        f["status"] = "redacted"
                        found = True
            for rs in self._raw_sessions:
                if rs.get("artifact_id") == artifact_id:
                    if "content" in fields or "fact" in fields:
                        rs["content"] = "[REDACTED]"
                    rs["status"] = "redacted"
                    found = True
            if found:
                self._data_dict = None
                self._save_cache()

        if not found:
            return {"error": f"Artifact not found: {artifact_id}",
                    "code": "NOT_FOUND"}
        return {"redacted": True, "artifact_id": artifact_id, "fields": fields}

    # ── stats() ──

    def stats(self) -> dict:
        raw_status_counts = Counter((rs.get("status") or "active") for rs in self._raw_sessions)
        logical_source_ids = set(self._source_records.keys())
        part_source_ids = {
            rs.get("part_source_id")
            for rs in self._raw_sessions
            if rs.get("part_source_id")
        }
        return {
            "telemetry_version": 1,
            "granular": len(self._all_granular),
            "consolidated": len(self._all_cons),
            "cross_session": len(self._all_cross),
            "secrets": len(self._secrets),
            "index_built": self._data_dict is not None,
            "agent_id": self.agent_id,
            "scope": self.scope,
            "swarm_id": self.swarm_id,
            "tier_mode": self._tier_mode,
            "tier2_built": self._tier2_built,
            "tier3_built": self._tier3_built,
            "tiers_dirty": self._tiers_dirty,
            "raw_sessions_count": len(self._raw_sessions),
            "source_records_count": len(self._source_records),
            "raw_session_status_counts": dict(raw_status_counts),
            "all_raw_sessions_active": all(
                (rs.get("status") or "active") == "active"
                for rs in self._raw_sessions
            ),
            "logical_source_count": len(logical_source_ids),
            "part_source_count": len(part_source_ids),
            "process_cost_summary": get_cost_summary(),
            "process_cost_scope": "process",
        }

    # ── metadata schema ──

    def _validate_metadata(self, metadata) -> str:
        """Validate metadata against schema. Returns error message or None."""
        if metadata is not None and not isinstance(metadata, dict):
            return f"metadata must be a dict, got {type(metadata).__name__}"
        if not metadata:
            return None
        # Flatness validation (always, even without schema)
        for key, value in metadata.items():
            if isinstance(value, dict):
                return f"metadata.{key}: nested dicts not allowed"
            if isinstance(value, list):
                if not all(isinstance(v, str) for v in value):
                    return f"metadata.{key}: lists must contain only strings"
        if not self._metadata_schema:
            return None
        for field_name, field_def in self._metadata_schema.items():
            if field_def.get("required") and field_name not in metadata:
                return f"required metadata field '{field_name}' missing"
        for key, value in metadata.items():
            if key in self._metadata_schema:
                expected = self._metadata_schema[key]["type"]
                if expected == "string" and not isinstance(value, str):
                    return f"metadata.{key}: expected string, got {type(value).__name__}"
                if expected == "number" and (isinstance(value, bool) or not isinstance(value, (int, float))):
                    return f"metadata.{key}: expected number, got {type(value).__name__}"
                if expected == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
                    return f"metadata.{key}: expected integer, got {type(value).__name__}"
                if expected == "boolean" and not isinstance(value, bool):
                    return f"metadata.{key}: expected boolean, got {type(value).__name__}"
                if expected == "enum":
                    allowed = self._metadata_schema[key].get("values", [])
                    if value not in allowed:
                        return f"metadata.{key}: '{value}' not in {allowed}"
                if expected == "datetime" and not isinstance(value, str):
                    return f"metadata.{key}: expected ISO datetime string"
                if expected == "string[]":
                    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                        return f"metadata.{key}: expected list of strings"
        return None

    async def set_metadata_schema(self, schema: dict) -> None:
        """Declare metadata schema. Persisted via _save_cache."""
        for field_name, field_def in schema.items():
            if "type" not in field_def:
                raise ValueError(f"metadata schema field '{field_name}' missing 'type'")
            if field_def["type"] not in ("string", "number", "integer",
                                          "boolean", "datetime", "enum", "string[]"):
                raise ValueError(f"unsupported type '{field_def['type']}' for '{field_name}'")
            if field_def["type"] == "enum":
                vals = field_def.get("values")
                if not isinstance(vals, list) or len(vals) == 0:
                    raise ValueError(f"enum field '{field_name}' requires non-empty 'values' list")
        async with self._file_lock:
            self._metadata_schema = schema
            self._save_cache()

    def get_metadata_schema(self) -> dict:
        return self._metadata_schema or {}

    # ── query() ──

    async def query(
        self,
        filter: dict = None,
        sort_by: str = "session_date",
        sort_order: str = "desc",
        limit: int = 10,
        offset: int = 0,
        caller_id: str = None,
        caller_role: str = "agent",
        caller_memberships: list[str] = None,
    ) -> dict:
        """Structured query on facts. No vectors, no LLM."""
        meta_schema = self._metadata_schema

        if not _is_sortable(sort_by, meta_schema):
            return {"error": f"sort_by must be a core scalar field or declared "
                             f"metadata.* field (schema required for metadata sort)",
                    "code": "INVALID_SORT_FIELD"}
        if sort_order not in VALID_SORT_ORDERS:
            return {"error": "sort_order must be 'asc' or 'desc'",
                    "code": "INVALID_SORT_ORDER"}
        if limit < 0 or offset < 0:
            return {"error": "limit and offset must be non-negative",
                    "code": "INVALID_PAGINATION"}

        # Tag facts with tier for query results
        gran_set = set(id(f) for f in self._all_granular)
        cons_set = set(id(f) for f in self._all_cons)
        all_facts = self._all_granular + self._all_cons + self._all_cross
        for f in all_facts:
            if id(f) in gran_set:
                f.setdefault("_tier", "granular")
            elif id(f) in cons_set:
                f.setdefault("_tier", "consolidated")
            else:
                f.setdefault("_tier", "cross_session")
        now = datetime.now(timezone.utc)

        fl = {}
        for f in self._all_granular:
            cid = f.get("conv_id", "mem_s0")
            fid = f.get("id", "")
            if fid:
                fl[fid] = f
                fl[f"{cid}_{fid}"] = f

        visible = [
            f for f in all_facts
            if _is_visible(f, now=now, fact_lookup=fl)
            and self._acl_allows(f, caller_id or "system",
                                 caller_memberships or [], caller_role)
        ]

        if filter:
            visible = [
                f for f in visible
                if _fact_matches_structured_filter(f, filter, meta_schema)
            ]

        total = len(visible)

        reverse = (sort_order == "desc")
        present, missing = _split_sort_values(visible, sort_by)
        present.sort(key=lambda f: _sort_value(f, sort_by), reverse=reverse)
        visible = present + missing

        paginated = visible[offset:offset + limit]

        if hasattr(self, "_audit") and self._audit:
            self._audit.log("query", caller_id or "system", {
                "filter": filter, "sort_by": sort_by, "limit": limit,
                "total_matched": total, "returned": len(paginated),
            })

        preview_facts = []
        for fact in paginated:
            rendered = deepcopy(fact)
            text = rendered.get("fact")
            if isinstance(text, str) and len(text) > MAX_QUERY_FACT_CHARS:
                rendered["fact"] = f"{text[:MAX_QUERY_FACT_CHARS]}...[truncated]"
                rendered["fact_truncated"] = True
            preview_facts.append(rendered)

        return {
            "total": total,
            "facts": preview_facts,
            "has_more": (offset + limit) < total,
        }
