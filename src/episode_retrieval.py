#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Episode-native retrieval and deterministic selection."""

from __future__ import annotations

import re
from collections import defaultdict

from .common import normalize_term_token
from .episode_features import (
    NUMBER_RE,
    QUERY_WORDS_TO_IGNORE,
    episode_chainages,
    episode_codes,
    episode_sentences,
    episode_text,
    episode_text_lower,
    extract_query_features,
    generic_episode_penalty,
    has_exact_step_mention,
    step_range_overlap,
    word_token_set,
)
from .episode_packet import _fact_list_item_keys, _fact_plan_commitment_score
from .retrieval import BM25Index
from .tuning import get_tuning_section

DEFAULT_SELECTION_CONFIG = get_tuning_section("retrieval", "selector")

_CONVERSATION_QUERY_RE = re.compile(
    r"\b("
    r"did i|did we|what did .* say|who said|who told|when did .* mention|"
    r"remember|conversation|chat|we talked|we discussed|told me|mentioned to me"
    r")\b",
    re.I,
)
_DOCUMENT_QUERY_RE = re.compile(
    r"\b("
    r"document|report|contract|clause|section|appendix|spec|proposal|policy|"
    r"manual|drawing|figure|table|permit|memo|pdf"
    r")\b",
    re.I,
)
_LOCATION_EVENT_TOKENS = {
    "plac",
    "event",
    "venue",
    "bar",
    "pub",
    "club",
    "cafe",
    "caf",
    "coffee",
    "stadium",
    "arena",
    "theater",
    "cinema",
    "restaurant",
    "park",
}
_GENERIC_LIST_SET_HEAD_TOKENS = {
    "plac",
    "place",
    "event",
    "item",
    "thing",
    "venu",
    "location",
    "activ",
    "activit",
    "kind",
    "type",
    "name",
}

def resolve_selection_config(config: dict | None = None) -> dict:
    """Merge caller overrides onto the conservative production default."""
    merged = get_tuning_section("retrieval", "selector")
    if config:
        merged.update(config)
    return merged


def _effective_retrieval_target(question: str, qf: dict) -> str:
    target = qf.get("retrieval_target", question)
    list_plan = qf.get("operator_plan", {}).get("list_set", {}) or {}
    head_tokens = set(list_plan.get("head_tokens") or [])
    if not list_plan.get("enabled") or not head_tokens:
        return target
    if not head_tokens <= _GENERIC_LIST_SET_HEAD_TOKENS:
        return target

    reduced_terms: list[str] = []
    for phrase in qf.get("entity_phrases", []):
        if phrase not in reduced_terms:
            reduced_terms.append(phrase)
    for word in sorted(qf.get("words", set()) - head_tokens):
        if word not in reduced_terms:
            reduced_terms.append(word)
    reduced = " ".join(reduced_terms).strip()
    return reduced or target


def build_episode_bm25(corpus: dict) -> BM25Index:
    flat = []
    ids = []
    for doc in corpus.get("documents", []):
        for ep in doc.get("episodes", []):
            flat.append(episode_text(ep))
            ids.append(ep["episode_id"])
    return BM25Index(flat, ids)


def available_families(corpus: dict) -> list[str]:
    families = []
    seen = set()
    for doc in corpus.get("documents", []):
        for ep in doc.get("episodes", []):
            family = ep.get("source_type") or "document"
            if family not in seen:
                seen.add(family)
                families.append(family)
    return families


def route_retrieval_families(
    query: str,
    available: list[str],
    explicit_family: str | None = None,
) -> list[str]:
    """Choose retrieval families for first-pass candidate generation.

    Rules:
    - explicit family wins
    - one available family -> use it
    - otherwise, only route to one family when the question is clearly typed
    - ambiguous questions retrieve separately from each available family
    """
    allowed = [family for family in available if family in {"conversation", "document"}]
    routing = get_tuning_section("routing")
    if explicit_family and explicit_family != "auto":
        return [family for family in allowed if family == explicit_family]
    if len(allowed) <= 1:
        return allowed

    conv_match = bool(_CONVERSATION_QUERY_RE.search(query))
    doc_match = bool(_DOCUMENT_QUERY_RE.search(query))
    if conv_match and not doc_match and "conversation" in allowed:
        return ["conversation"]
    if doc_match and not conv_match and "document" in allowed:
        return ["document"]
    max_plausible = max(1, int(routing.get("max_plausible_families", len(allowed))))
    fanout = max(1, int(routing.get("ambiguous_family_fanout", len(allowed))))
    limit = min(len(allowed), max_plausible, fanout)
    return allowed[:limit]


def partition_corpus_by_family(corpus: dict) -> dict[str, dict]:
    families: dict[str, list[dict]] = defaultdict(list)
    for doc in corpus.get("documents", []):
        by_family: dict[str, list[dict]] = defaultdict(list)
        for ep in doc.get("episodes", []):
            family = ep.get("source_type") or "document"
            by_family[family].append(ep)
        for family, episodes in by_family.items():
            families[family].append({
                "doc_id": doc.get("doc_id"),
                "episodes": episodes,
            })
    return {family: {"documents": docs} for family, docs in families.items()}


def _query_word_overlap_score(ep: dict, qf: dict, weight: float) -> float:
    if weight <= 0:
        return 0.0
    tokens = word_token_set(episode_text(ep))
    return weight * sum(1 for word in qf["words"] if word in tokens)


def _number_overlap_score(ep: dict, qf: dict, weight: float) -> float:
    if weight <= 0:
        return 0.0
    numbers = set(NUMBER_RE.findall(episode_text(ep)))
    return weight * sum(1 for number in qf["numbers"] if number in numbers)


def _km_overlap_score(ep: dict, qf: dict, weight: float) -> float:
    if weight <= 0 or not qf["kms"]:
        return 0.0
    return weight * len(episode_chainages(ep) & qf["kms"])


def _code_overlap_score(ep: dict, qf: dict, weight: float) -> float:
    if weight <= 0 or not qf["codes"]:
        return 0.0
    return weight * len(episode_codes(ep) & qf["codes"])


def _entity_phrase_score(ep: dict, qf: dict, weight: float) -> float:
    if weight <= 0:
        return 0.0
    lower = episode_text_lower(ep)
    tokens = word_token_set(episode_text(ep))
    score = 0.0
    for phrase in qf.get("entity_phrases", []):
        phrase_lower = phrase.lower()
        phrase_tokens = qf.get("entity_phrase_tokens", {}).get(phrase_lower) or word_token_set(phrase)
        if (" " in phrase_lower and phrase_lower in lower) or (
            " " not in phrase_lower and phrase_tokens and phrase_tokens <= tokens
        ):
            score += weight * (3.0 if " " in phrase_lower else 1.0)
    return score


def _step_number_score(ep: dict, qf: dict, weight: float) -> float:
    if weight <= 0 or not (qf.get("step_numbers") or qf.get("step_range")):
        return 0.0
    text = episode_text(ep)
    score = 0.0
    explicit_steps = set(qf.get("step_numbers", set()) or set())
    for step in explicit_steps:
        if has_exact_step_mention(text, step):
            score += weight
    interior_hits = step_range_overlap(text, qf.get("step_range")) - explicit_steps
    if interior_hits:
        score += weight * min(1.5, 0.75 * len(interior_hits))
    return score


def _currentness_score(ep: dict, qf: dict, weight: float) -> float:
    if weight <= 0:
        return 0.0
    currentness = (ep.get("currentness") or "").lower()
    lower = qf["lower"]
    if "current" in lower and currentness == "current":
        return weight
    if "latest" in lower and currentness == "current":
        return weight
    if "historical" in lower and currentness == "historical":
        return weight
    if "outdated" in lower and currentness == "outdated":
        return weight
    return 0.0


def _query_requests_location_event(qf: dict) -> bool:
    words = set(qf.get("words", set()))
    return bool(qf.get("asks_where")) or bool(words & _LOCATION_EVENT_TOKENS)


def _episode_location_event_score(ep: dict, qf: dict, weight: float) -> float:
    if weight <= 0 or not _query_requests_location_event(qf):
        return 0.0
    tokens = word_token_set(episode_text(ep))
    return weight * float(len(tokens & _LOCATION_EVENT_TOKENS))


def _query_requests_plan_meetup(qf: dict) -> bool:
    words = set(qf.get("words", set()))
    if not (qf.get("operator_plan") or {}).get("list_set", {}).get("enabled"):
        return False
    return bool(words & {"plan", "plann", "future", "upcom", "schedul"})


def _episode_plan_commitment_score(
    ep: dict,
    qf: dict,
    *,
    commit_weight: float,
    tentative_penalty: float,
) -> float:
    if not _query_requests_plan_meetup(qf):
        return 0.0
    raw_score = _fact_plan_commitment_score(episode_text(ep), qf)
    if raw_score > 0:
        return commit_weight * raw_score
    if _episode_location_event_score(ep, qf, 1.0) > 0:
        return -tentative_penalty
    return raw_score


def _episode_latent_attribute_score(ep: dict, qf: dict, weight: float) -> float:
    return 0.0


def _mega_penalty(ep: dict, weight: float) -> float:
    raw_len = len(ep.get("raw_text", ""))
    if raw_len <= 12000:
        return 0.0
    if raw_len <= 50000:
        return weight * 0.5
    return weight


def score_episode_with_breakdown(
    ep: dict,
    qf: dict,
    bm25_score: float,
    config: dict | None = None,
) -> tuple[float, dict]:
    config = resolve_selection_config(config)
    breakdown = {
        "bm25": bm25_score,
        "word_overlap": _query_word_overlap_score(ep, qf, config["word_overlap_bonus"]),
        "number_overlap": _number_overlap_score(ep, qf, config["number_overlap_bonus"]),
        "chainage_overlap": _km_overlap_score(ep, qf, config.get("chainage_overlap_bonus", 0.0)),
        "identifier_overlap": _code_overlap_score(ep, qf, config.get("identifier_overlap_bonus", 0.0)),
        "entity_phrase": _entity_phrase_score(ep, qf, config["entity_phrase_bonus"]),
        "step": _step_number_score(ep, qf, config["step_bonus"]),
        "currentness": _currentness_score(ep, qf, config["currentness_bonus"]),
        "location_event": _episode_location_event_score(ep, qf, config.get("location_event_bonus", 1.25)),
        "plan_commitment": _episode_plan_commitment_score(
            ep,
            qf,
            commit_weight=config.get("plan_commitment_bonus", 2.0),
            tentative_penalty=config.get("plan_tentative_penalty", 1.0),
        ),
        "latent_attribute": _episode_latent_attribute_score(
            ep,
            qf,
            config.get("latent_attribute_bonus", 1.0),
        ),
        "generic_penalty": -config["generic_penalty"] * generic_episode_penalty(
            ep.get("topic_key", ""),
            ep.get("state_label", ""),
        ),
        "mega_penalty": -_mega_penalty(ep, config["mega_penalty"]),
    }
    total = sum(breakdown.values())
    breakdown["total"] = total
    return total, breakdown


def score_episode(ep: dict, qf: dict, bm25_score: float, config: dict | None = None) -> float:
    return score_episode_with_breakdown(ep, qf, bm25_score, config)[0]


def _episode_order_value(ep: dict, order_basis: str | None) -> int:
    if order_basis == "step_order":
        lower = episode_text_lower(ep)
        match = re.search(r"\[step\s+(\d+)\]|\bstep\s+(\d+)\b", lower)
        if match:
            return int(match.group(1) or match.group(2))
    topic = (ep.get("topic_key") or "").lower()
    match = re.search(r"\bsession_(\d+)\b", topic)
    if match:
        return int(match.group(1))
    match = re.search(r"_e(\d+)\b", ep.get("episode_id", ""))
    if match:
        return int(match.group(1))
    return 10**9


def _text_tokens(text: str) -> set[str]:
    return {
        normalize_term_token(token)
        for token in re.findall(r"[A-Za-z0-9]+", text)
        if len(token) >= 3 and normalize_term_token(token)
    }


def _entity_match_count(ep: dict, qf: dict) -> int:
    lower = episode_text_lower(ep)
    tokens = word_token_set(episode_text(ep))
    matches = 0
    for phrase in qf.get("entity_phrases", []):
        phrase_lower = phrase.lower()
        phrase_tokens = qf.get("entity_phrase_tokens", {}).get(phrase_lower) or word_token_set(phrase)
        if (" " in phrase_lower and phrase_lower in lower) or (
            " " not in phrase_lower and phrase_tokens and phrase_tokens <= tokens
        ):
            matches += 1
    return matches


def _anchor_match_score(ep: dict, qf: dict) -> float:
    text = episode_text(ep)
    lower = text.lower()
    tokens = word_token_set(text)
    numbers = set(NUMBER_RE.findall(text))
    score = 0.0
    explicit_steps = set(qf.get("step_numbers", set()) or set())
    for code in qf.get("codes", set()):
        if code.lower() in lower:
            score += 8.0
    for step in explicit_steps:
        if has_exact_step_mention(text, step):
            score += 6.0
    interior_hits = step_range_overlap(text, qf.get("step_range")) - explicit_steps
    if interior_hits:
        score += min(6.0, 3.0 * len(interior_hits))
    for number in qf.get("numbers", set()):
        if number in numbers:
            score += 2.0
    for phrase in qf.get("entity_phrases", []):
        phrase_lower = phrase.lower()
        phrase_tokens = qf.get("entity_phrase_tokens", {}).get(phrase_lower) or word_token_set(phrase)
        if (" " in phrase_lower and phrase_lower in lower) or (
            " " not in phrase_lower and phrase_tokens and phrase_tokens <= tokens
        ):
            score += 3.0
    return score


def _episode_contrast_score(ep_a: dict, ep_b: dict) -> float:
    score = 0.0
    if ep_a.get("source_id") == ep_b.get("source_id"):
        score += 2.0
    if ep_a.get("state_label") != ep_b.get("state_label"):
        score += 2.0
    if ep_a.get("currentness") != ep_b.get("currentness"):
        score += 2.0
    if ep_a.get("source_date") != ep_b.get("source_date"):
        score += 1.0
    return score


def _chain_tokens(ep: dict, qf: dict) -> set[str]:
    return _chain_tokens_with_frontier(ep, qf, get_tuning_section("operators"), frontier=None)


def _query_entity_tokens(qf: dict) -> set[str]:
    tokens = set()
    for phrase_tokens in (qf.get("entity_phrase_tokens") or {}).values():
        tokens |= set(phrase_tokens)
    return tokens


_IGNORED_CONTENT_TOKENS = {
    normalize_term_token(token)
    for token in QUERY_WORDS_TO_IGNORE
    if normalize_term_token(token)
}


def _content_word_token_set(text: str) -> set[str]:
    return {
        token
        for token in word_token_set(text)
        if token not in _IGNORED_CONTENT_TOKENS
    }


def _sentence_bridge_tokens(text: str, qf: dict) -> set[str]:
    tokens = _content_word_token_set(text)
    tokens -= set(qf.get("words", set()))
    tokens -= _query_entity_tokens(qf)
    lower = text.lower()
    tokens.update(code.lower() for code in episode_codes({"raw_text": text}))
    tokens.update(km for km in episode_chainages({"raw_text": text}))
    tokens.update(number for number in NUMBER_RE.findall(text))
    for phrase in qf.get("entity_phrases", []):
        phrase_lower = phrase.lower()
        phrase_tokens = qf.get("entity_phrase_tokens", {}).get(phrase_lower) or word_token_set(phrase)
        if (" " in phrase_lower and phrase_lower in lower) or (
            " " not in phrase_lower and phrase_tokens and phrase_tokens <= word_token_set(text)
        ):
            tokens.add(phrase_lower)
    return tokens


def _support_sentence_score(
    text: str,
    qf: dict,
    operator_tuning: dict,
    frontier: set[str] | None = None,
    unresolved_words: set[str] | None = None,
) -> float:
    lower = text.lower()
    tokens = _content_word_token_set(text)
    score = 0.0
    entity_weight = float(operator_tuning.get("bounded_chain_sentence_entity_match_weight", 6.0))
    word_weight = float(operator_tuning.get("bounded_chain_sentence_word_match_weight", 1.0))
    unresolved_weight = float(operator_tuning.get("bounded_chain_sentence_unresolved_weight", 2.0))
    number_weight = float(operator_tuning.get("bounded_chain_sentence_number_match_weight", 1.5))
    frontier_weight = float(operator_tuning.get("bounded_chain_sentence_frontier_overlap_weight", 2.0))
    for phrase in qf.get("entity_phrases", []):
        phrase_lower = phrase.lower()
        phrase_tokens = qf.get("entity_phrase_tokens", {}).get(phrase_lower) or word_token_set(phrase)
        if (" " in phrase_lower and phrase_lower in lower) or (
            " " not in phrase_lower and phrase_tokens and phrase_tokens <= tokens
        ):
            score += entity_weight
    for word in qf.get("words", set()):
        if word in tokens:
            score += word_weight
    if unresolved_words:
        for word in unresolved_words:
            if word in tokens:
                score += unresolved_weight
    for number in qf.get("numbers", set()):
        if number in NUMBER_RE.findall(text):
            score += number_weight
    if frontier:
        score += frontier_weight * len(frontier & tokens)
    return score


def _chain_tokens_with_frontier(
    ep: dict,
    qf: dict,
    operator_tuning: dict,
    frontier: set[str] | None,
    unresolved_words: set[str] | None = None,
    sentence_budget: int = 2,
) -> set[str]:
    lower = episode_text_lower(ep)
    tokens = _content_word_token_set(episode_text(ep))
    keys = set()
    keys.update(code.lower() for code in episode_codes(ep))
    keys.update(episode_chainages(ep))
    support_sentences = []
    for sent in episode_sentences(ep.get("raw_text", "")):
        score = _support_sentence_score(
            sent,
            qf,
            operator_tuning,
            frontier,
            unresolved_words,
        )
        if score > 0:
            support_sentences.append((score, sent))
    support_sentences.sort(key=lambda row: (-row[0], row[1]))
    if support_sentences:
        for _score, sent in support_sentences[: max(1, sentence_budget)]:
            keys |= _sentence_bridge_tokens(sent, qf)
    else:
        for phrase in qf.get("entity_phrases", []):
            phrase_lower = phrase.lower()
            phrase_tokens = qf.get("entity_phrase_tokens", {}).get(phrase_lower) or word_token_set(phrase)
            if (" " in phrase_lower and phrase_lower in lower) or (
                " " not in phrase_lower and phrase_tokens and phrase_tokens <= tokens
            ):
                keys.add(phrase_lower)
        for word in qf.get("words", set()):
            if word in tokens:
                keys.add(word)
        for number in qf.get("numbers", set()):
            if number in NUMBER_RE.findall(episode_text(ep)):
                keys.add(number)
    return keys


def _best_support_sentence_score(
    ep: dict,
    qf: dict,
    operator_tuning: dict,
    frontier: set[str] | None,
    unresolved_words: set[str] | None = None,
) -> float:
    best = 0.0
    for sent in episode_sentences(ep.get("raw_text", "")):
        best = max(
            best,
            _support_sentence_score(
                sent,
                qf,
                operator_tuning,
                frontier,
                unresolved_words,
            ),
        )
    return best


def _episode_resolver_tokens(
    ep: dict,
    qf: dict,
    operator_tuning: dict,
    frontier: set[str] | None,
    unresolved_words: set[str] | None,
    sentence_budget: int,
) -> set[str]:
    tokens: set[str] = set()
    support_sentences = []
    for sent in episode_sentences(ep.get("raw_text", "")):
        score = _support_sentence_score(
            sent,
            qf,
            operator_tuning,
            frontier,
            unresolved_words,
        )
        if score > 0:
            support_sentences.append((score, sent))
    support_sentences.sort(key=lambda row: (-row[0], row[1]))
    for _score, sent in support_sentences[: max(1, sentence_budget)]:
        tokens |= _content_word_token_set(sent)
        tokens.update(number for number in NUMBER_RE.findall(sent))
    return tokens


def _episode_relation_tokens(
    ep: dict,
    qf: dict,
    operator_tuning: dict,
    frontier: set[str] | None,
    unresolved_words: set[str] | None,
    sentence_budget: int,
) -> set[str]:
    resolver_tokens = _episode_resolver_tokens(
        ep,
        qf,
        operator_tuning,
        frontier,
        unresolved_words,
        sentence_budget,
    )
    return resolver_tokens & GENERIC_CHAIN_RELATION_TOKENS


def _move_row_to_front(rows: list[dict], episode_id: str | None) -> list[dict]:
    if not episode_id:
        return rows
    front = None
    remainder = []
    for row in rows:
        if row["episode_id"] == episode_id and front is None:
            front = row
        else:
            remainder.append(row)
    if front is None:
        return rows
    return [front] + remainder


def _greedy_commonality_order(rows: list[dict], episode_lookup: dict[str, dict], qf: dict) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            -_entity_match_count(episode_lookup[row["episode_id"]], qf),
            -row["score"],
            row["bm25_rank"],
            row["episode_id"],
        ),
    )


def _apply_compare_diff_front(
    rows: list[dict],
    episode_lookup: dict[str, dict],
    budget: int,
) -> list[dict]:
    pool = rows[: max(2, budget)]
    best_pair = None
    best_score = None
    for i, left in enumerate(pool):
        ep_left = episode_lookup[left["episode_id"]]
        for right in pool[i + 1 :]:
            ep_right = episode_lookup[right["episode_id"]]
            contrast = _episode_contrast_score(ep_left, ep_right)
            if contrast <= 0:
                continue
            pair_score = (contrast, left["score"] + right["score"])
            if best_score is None or pair_score > best_score:
                best_score = pair_score
                best_pair = (left["episode_id"], right["episode_id"])
    if not best_pair:
        return rows
    ordered = _move_row_to_front(rows, best_pair[1])
    return _move_row_to_front(ordered, best_pair[0])


def _apply_ordinal_front(
    rows: list[dict],
    episode_lookup: dict[str, dict],
    ordinal_plan: dict,
    budget: int,
) -> list[dict]:
    pool = rows[: max(1, budget)]
    ordered = sorted(
        pool,
        key=lambda row: (
            _episode_order_value(episode_lookup[row["episode_id"]], ordinal_plan.get("order_basis")),
            row["bm25_rank"],
            row["episode_id"],
        ),
    )
    mode = ordinal_plan.get("mode")
    selected_id = None
    if mode == "nth":
        idx = max(0, int(ordinal_plan.get("index") or 1) - 1)
        if idx < len(ordered):
            selected_id = ordered[idx]["episode_id"]
    elif mode == "last" and ordered:
        selected_id = ordered[-1]["episode_id"]
    elif mode == "penultimate" and len(ordered) >= 2:
        selected_id = ordered[-2]["episode_id"]
    return _move_row_to_front(rows, selected_id)


def _apply_local_anchor_priority(rows: list[dict], episode_lookup: dict[str, dict], qf: dict) -> list[dict]:
    step_range = qf.get("step_range")
    return sorted(
        rows,
        key=lambda row: (
            -_anchor_match_score(episode_lookup[row["episode_id"]], qf),
            _episode_order_value(episode_lookup[row["episode_id"]], "step_order")
            if step_range else -row["score"],
            -row["score"],
            row["bm25_rank"],
            row["episode_id"],
        ),
    )


def _apply_current_priority(rows: list[dict], episode_lookup: dict[str, dict], qf: dict) -> list[dict]:
    if not rows:
        return rows
    if qf.get("query_type") != "current" and "current" not in qf.get("lower", "") and "latest" not in qf.get("lower", ""):
        return rows
    best_id = None
    best_key = None
    for row in rows[:12]:
        ep = episode_lookup[row["episode_id"]]
        breakdown = row.get("score_breakdown") or {}
        if breakdown.get("word_overlap", 0.0) <= 0:
            continue
        if breakdown.get("number_overlap", 0.0) <= 0 and _anchor_match_score(ep, qf) <= 0:
            continue
        currentness = 1 if (ep.get("currentness") or "").lower() == "current" else 0
        key = (
            currentness,
            _episode_order_value(ep, "match_order"),
            row["score"],
            row["bm25_rank"] * -1,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_id = row["episode_id"]
    return _move_row_to_front(rows, best_id)


def _apply_bounded_chain_expansion(
    rows: list[dict],
    episode_lookup: dict[str, dict],
    qf: dict,
    operator_tuning: dict,
    max_hops: int,
) -> list[dict]:
    if not rows or max_hops <= 0:
        return rows
    root_row = rows[0]
    root_ep = episode_lookup[root_row["episode_id"]]
    root_sentence_budget = int(operator_tuning.get("bounded_chain_root_support_sentence_budget", 1))
    sentence_budget = int(operator_tuning.get("bounded_chain_support_sentence_budget", 2))
    same_source_bonus_weight = float(operator_tuning.get("bounded_chain_same_source_bonus", 2.0))
    unresolved_overlap_weight = float(operator_tuning.get("bounded_chain_unresolved_overlap_weight", 1.5))
    novelty_weight = float(operator_tuning.get("bounded_chain_novelty_weight", 0.75))
    bm25_carry_weight = float(operator_tuning.get("bounded_chain_bm25_carry_weight", 0.01))
    relation_novelty_weight = float(operator_tuning.get("bounded_chain_relation_novelty_weight", 1.0))
    relation_repeat_penalty = float(operator_tuning.get("bounded_chain_relation_repeat_penalty", 1.0))
    lookahead_weight = float(operator_tuning.get("bounded_chain_lookahead_weight", 1.0))
    branch_limit = max(1, int(operator_tuning.get("bounded_chain_branch_limit", 24)))
    support_sentence_score_weight = float(
        operator_tuning.get("bounded_chain_support_sentence_score_weight", 1.0)
    )
    same_source_scan_enabled = bool(
        operator_tuning.get("bounded_chain_expand_same_source_episodes", True)
    )
    same_source_scan_limit = max(
        0,
        int(operator_tuning.get("bounded_chain_same_source_scan_limit", 256)),
    )
    candidate_row_pool = list(rows)
    existing_row_ids = {row["episode_id"] for row in rows}
    if same_source_scan_enabled and same_source_scan_limit > 0:
        existing_ids = set(existing_row_ids)
        source_id = root_ep.get("source_id")
        supplemental_rows = []
        for ep_id in sorted(episode_lookup):
            if ep_id in existing_ids or ep_id == root_row["episode_id"]:
                continue
            ep = episode_lookup[ep_id]
            if ep.get("source_id") != source_id:
                continue
            supplemental_rows.append(
                {
                    "episode_id": ep_id,
                    "score": 0.0,
                    "bm25_rank": len(rows) + len(supplemental_rows) + 1,
                    "source_id": ep.get("source_id", ""),
                    "source_type": ep.get("source_type", "document"),
                    "topic_key": ep.get("topic_key", ""),
                    "state_label": ep.get("state_label", ""),
                    "score_breakdown": {"same_source_scan": 1.0},
                }
            )
            if len(supplemental_rows) >= same_source_scan_limit:
                break
        candidate_row_pool.extend(supplemental_rows)
    scanned_only_ids = {
        row["episode_id"]
        for row in candidate_row_pool
        if row["episode_id"] not in existing_row_ids
    }
    frontier = _chain_tokens_with_frontier(
        root_ep,
        qf,
        operator_tuning,
        frontier=None,
        unresolved_words=set(qf.get("words", set())),
        sentence_budget=root_sentence_budget,
    )
    root_resolver_tokens = _episode_resolver_tokens(
        root_ep,
        qf,
        operator_tuning,
        frontier=None,
        unresolved_words=set(qf.get("words", set())),
        sentence_budget=root_sentence_budget,
    )
    seen_relation_tokens = _episode_relation_tokens(
        root_ep,
        qf,
        operator_tuning,
        frontier=None,
        unresolved_words=set(qf.get("words", set())),
        sentence_budget=root_sentence_budget,
    )
    unresolved_words = set(qf.get("words", set())) - frontier - root_resolver_tokens

    def _candidate_rows(
        active_frontier: set[str],
        active_unresolved: set[str],
        active_relations: set[str],
        used_ids: set[str],
    ) -> list[tuple[float, dict, set[str], set[str], set[str]]]:
        rows_out: list[tuple[float, dict, set[str], set[str], set[str]]] = []
        for row in candidate_row_pool:
            ep_id = row["episode_id"]
            if ep_id == root_row["episode_id"] or ep_id in used_ids:
                continue
            ep = episode_lookup[ep_id]
            candidate_tokens = _chain_tokens_with_frontier(
                ep,
                qf,
                operator_tuning,
                frontier=active_frontier,
                unresolved_words=active_unresolved,
                sentence_budget=sentence_budget,
            )
            overlap = len(active_frontier & candidate_tokens)
            if overlap <= 0:
                continue
            resolver_tokens = _episode_resolver_tokens(
                ep,
                qf,
                operator_tuning,
                frontier=active_frontier,
                unresolved_words=active_unresolved,
                sentence_budget=sentence_budget,
            )
            relation_tokens = _episode_relation_tokens(
                ep,
                qf,
                operator_tuning,
                frontier=active_frontier,
                unresolved_words=active_unresolved,
                sentence_budget=sentence_budget,
            )
            novelty = len(candidate_tokens - active_frontier)
            unresolved_overlap = len(active_unresolved & resolver_tokens)
            relation_novelty = len(relation_tokens - active_relations)
            relation_repeat = len(relation_tokens & active_relations)
            same_source_bonus = same_source_bonus_weight if ep.get("source_id") == root_ep.get("source_id") else 0.0
            support_sentence_score = _best_support_sentence_score(
                ep,
                qf,
                operator_tuning,
                active_frontier,
                active_unresolved,
            )
            candidate_score = (
                overlap
                + same_source_bonus
                + (unresolved_overlap_weight * unresolved_overlap)
                + (novelty_weight * novelty)
                + (relation_novelty_weight * relation_novelty)
                - (relation_repeat_penalty * relation_repeat)
                + (support_sentence_score_weight * support_sentence_score)
                + row["score"] * bm25_carry_weight
            )
            candidate_row = dict(row)
            breakdown = dict(candidate_row.get("score_breakdown") or {})
            breakdown["bounded_chain_candidate_score"] = candidate_score
            breakdown["bounded_chain_frontier_overlap"] = overlap
            breakdown["bounded_chain_unresolved_overlap"] = unresolved_overlap
            breakdown["bounded_chain_relation_novelty"] = relation_novelty
            breakdown["bounded_chain_relation_repeat"] = relation_repeat
            if ep_id in scanned_only_ids:
                breakdown["same_source_scan"] = 1.0
            candidate_row["score_breakdown"] = breakdown
            rows_out.append((candidate_score, candidate_row, candidate_tokens, resolver_tokens, relation_tokens))
        rows_out.sort(
            key=lambda item: (
                -item[0],
                -item[1]["score"],
                item[1]["bm25_rank"],
                item[1]["episode_id"],
            ),
        )
        return rows_out[:branch_limit]

    def _best_path(
        active_frontier: set[str],
        active_unresolved: set[str],
        active_relations: set[str],
        used_ids: set[str],
        depth: int,
    ) -> tuple[float, list[str]]:
        if depth <= 0:
            return 0.0, []
        best_score = 0.0
        best_path: list[str] = []
        for local_score, row, candidate_tokens, resolver_tokens, relation_tokens in _candidate_rows(
            active_frontier,
            active_unresolved,
            active_relations,
            used_ids,
        ):
            ep_id = row["episode_id"]
            next_used = set(used_ids)
            next_used.add(ep_id)
            next_frontier = set(active_frontier) | candidate_tokens
            next_unresolved = set(active_unresolved) - candidate_tokens - resolver_tokens
            next_relations = set(active_relations) | relation_tokens
            future_score, future_path = _best_path(
                next_frontier,
                next_unresolved,
                next_relations,
                next_used,
                depth - 1,
            )
            total_score = local_score + (lookahead_weight * future_score if future_path else 0.0)
            if total_score > best_score:
                best_score = total_score
                best_path = [ep_id] + future_path
        return best_score, best_path

    _best_score, inserted = _best_path(
        frontier,
        unresolved_words,
        seen_relation_tokens,
        set(),
        max_hops,
    )
    if not inserted:
        return rows
    row_by_id = {row["episode_id"]: row for row in candidate_row_pool}
    front_ids = [root_row["episode_id"]] + inserted
    front = [row_by_id[ep_id] for ep_id in front_ids if ep_id in row_by_id]
    tail = [row for row in rows if row["episode_id"] not in set(front_ids)]
    return front + tail


def _apply_list_set_dedup(
    rows: list[dict],
    episode_lookup: dict[str, dict],
    qf: dict,
    selection_budget: int,
    overlap_threshold: float,
    preserve_source_order: bool,
) -> list[dict]:
    if overlap_threshold <= 0:
        return rows
    ordered_rows = list(rows)
    kept = []
    rejected = []
    kept_tokens: list[set[str]] = []
    for row in ordered_rows:
        tokens = _text_tokens(episode_lookup[row["episode_id"]].get("raw_text", ""))
        duplicate = False
        for prior_tokens in kept_tokens:
            if not tokens or not prior_tokens:
                continue
            overlap = len(tokens & prior_tokens) / max(1, min(len(tokens), len(prior_tokens)))
            if overlap >= overlap_threshold:
                duplicate = True
                break
        if duplicate:
            rejected.append(row)
            continue
        kept.append(row)
        kept_tokens.append(tokens)
    if preserve_source_order:
        list_plan = qf.get("operator_plan", {}).get("list_set", {})
        head_phrase = (list_plan.get("head_phrase") or "").strip().lower()
        head_tokens = set(list_plan.get("head_tokens") or [])

        def _list_item_coverage(row: dict) -> int:
            ep = episode_lookup[row["episode_id"]]
            keys: set[str] = set()
            for sent in episode_sentences(ep.get("raw_text", "")):
                keys.update(_fact_list_item_keys(sent, qf))
            return len(keys)

        def _list_head_match(row: dict) -> int:
            ep = episode_lookup[row["episode_id"]]
            lower = episode_text_lower(ep)
            tokens = word_token_set(episode_text(ep))
            if head_phrase and head_phrase in lower:
                return 2
            if head_tokens and head_tokens <= tokens:
                return 1
            return 0

        front_pool = kept[: max(0, min(len(kept), max(3, selection_budget + 1)))]
        remainder = kept[len(front_pool):]
        front_pool = sorted(
            front_pool,
            key=lambda row: (
                -_list_head_match(row),
                -_list_item_coverage(row),
                -row["score"],
                _episode_order_value(episode_lookup[row["episode_id"]], "match_order"),
                row["bm25_rank"],
                row["episode_id"],
            ),
        )
        kept = front_pool + remainder
    return kept + rejected


def _apply_query_operators(
    rows: list[dict],
    episode_lookup: dict[str, dict],
    qf: dict,
    operator_tuning: dict,
    config: dict,
) -> list[dict]:
    ordered = list(rows)
    plan = qf.get("operator_plan", {})
    if plan.get("commonality", {}).get("enabled"):
        ordered = _greedy_commonality_order(ordered, episode_lookup, qf)
    if plan.get("local_anchor", {}).get("enabled"):
        ordered = _apply_local_anchor_priority(ordered, episode_lookup, qf)
    ordered = _apply_current_priority(ordered, episode_lookup, qf)
    if plan.get("compare_diff", {}).get("enabled"):
        ordered = _apply_compare_diff_front(
            ordered,
            episode_lookup,
            int(operator_tuning.get("compare_alignment_budget", 12)),
        )
    if plan.get("ordinal", {}).get("enabled"):
        ordered = _apply_ordinal_front(
            ordered,
            episode_lookup,
            plan.get("ordinal", {}),
            int(operator_tuning.get("ordinal_candidate_budget", 12)),
        )
    if plan.get("bounded_chain", {}).get("enabled"):
        ordered = _apply_bounded_chain_expansion(
            ordered,
            episode_lookup,
            qf,
            operator_tuning,
            int(plan.get("bounded_chain", {}).get("max_hops", 2)),
        )
    if plan.get("list_set", {}).get("enabled"):
        ordered = _apply_list_set_dedup(
            ordered,
            episode_lookup,
            qf,
            int(config.get("max_episodes_default", 3)),
            float(operator_tuning.get("list_set_dedup_overlap", 0.9)),
            bool(plan.get("list_set", {}).get("preserve_source_order", True)),
        )
    return ordered


def _candidate_trace_rows(rows: list[dict], limit: int) -> list[dict]:
    traced = []
    for row in rows[:limit]:
        traced.append(
            {
                "episode_id": row["episode_id"],
                "source_id": row["source_id"],
                "source_type": row["source_type"],
                "topic_key": row["topic_key"],
                "state_label": row["state_label"],
                "score": row["score"],
                "bm25_rank": row["bm25_rank"],
                "score_breakdown": row["score_breakdown"],
            }
        )
    return traced


def _rank_sources(
    scored: list[tuple[str, float]],
    episode_lookup: dict[str, dict],
    config: dict,
) -> list[str]:
    by_source: dict[str, list[tuple[int, float]]] = defaultdict(list)
    limit = min(len(scored), max(config["max_candidates"], 1))
    for rank, (ep_id, score) in enumerate(scored[:limit], start=1):
        source_id = episode_lookup[ep_id]["source_id"]
        by_source[source_id].append((rank, score))

    source_rows = []
    for source_id, rows in by_source.items():
        rrf = sum(1.0 / (config["rrf_k"] + rank) for rank, _score in rows[:8])
        best = max(score for _rank, score in rows)
        support = len(rows)
        source_rows.append((source_id, rrf, best, support))
    source_rows.sort(key=lambda row: (-row[1], -row[2], -row[3], row[0]))
    if not source_rows:
        return []
    if len(source_rows) == 1:
        return [source_rows[0][0]]

    first = source_rows[0]
    second = source_rows[1]
    if first[1] >= second[1] * 1.35 or first[2] >= second[2] + 2.5:
        return [first[0]]
    return [row[0] for row in source_rows[: config["max_sources_per_family"]]]


def _latent_attribute_candidate_hits(
    qf: dict,
    episode_lookup: dict[str, dict],
    limit: int,
) -> list[dict]:
    return []


def choose_episode_ids(
    question: str,
    bm25: BM25Index,
    episode_lookup: dict[str, dict],
    config: dict | None = None,
) -> tuple[list[str], list[tuple[str, float]]]:
    result = choose_episode_ids_with_trace(question, bm25, episode_lookup, config)
    return result["selected_ids"], result["scored"]


def choose_episode_ids_with_trace(
    question: str,
    bm25: BM25Index,
    episode_lookup: dict[str, dict],
    config: dict | None = None,
) -> dict:
    config = resolve_selection_config(config)
    operator_tuning = get_tuning_section("operators")
    telemetry = get_tuning_section("telemetry")
    qf = extract_query_features(question)
    retrieval_target = _effective_retrieval_target(question, qf)
    bm_hits = bm25.search(retrieval_target, top_k=config["max_candidates"])
    latent_hits = _latent_attribute_candidate_hits(qf, episode_lookup, config["max_candidates"])
    seen_hit_ids: set[str] = set()
    candidate_hits: list[dict] = []
    for hit in bm_hits + latent_hits:
        hit_id = hit["id"]
        if hit_id in seen_hit_ids:
            continue
        seen_hit_ids.add(hit_id)
        candidate_hits.append(hit)

    scored_rows = []
    for rank, hit in enumerate(candidate_hits, start=1):
        ep = episode_lookup[hit["id"]]
        score, breakdown = score_episode_with_breakdown(ep, qf, hit["s"], config)
        scored_rows.append(
            {
                "episode_id": hit["id"],
                "score": score,
                "bm25_rank": rank,
                "source_id": ep.get("source_id", ""),
                "source_type": ep.get("source_type", "document"),
                "topic_key": ep.get("topic_key", ""),
                "state_label": ep.get("state_label", ""),
                "score_breakdown": breakdown,
            }
        )
    scored_rows.sort(key=lambda row: (-row["score"], row["episode_id"]))
    scored = [(row["episode_id"], row["score"]) for row in scored_rows]
    if not scored:
        return {
            "selected_ids": [],
            "scored": [],
            "trace": {
                "candidate_count": 0,
                "pre_source_gate": [],
                "selected_source_ids": [],
                "post_source_gate": [],
                "selected_ids": [],
            },
        }

    selected_source_ids = set(_rank_sources(scored, episode_lookup, config))
    filtered_rows = [
        row
        for row in scored_rows
        if not selected_source_ids or row["source_id"] in selected_source_ids
    ]
    filtered_rows = _apply_query_operators(filtered_rows, episode_lookup, qf, operator_tuning, config)
    filtered = [(row["episode_id"], row["score"]) for row in filtered_rows]
    selected = [
        row["episode_id"]
        for row in filtered_rows[: config["max_episodes_default"]]
    ]
    limit = int(telemetry.get("max_family_candidates", 8))
    return {
        "selected_ids": selected,
        "scored": filtered,
        "trace": {
            "candidate_count": len(scored_rows),
            "retrieval_target": retrieval_target,
            "pre_source_gate": _candidate_trace_rows(scored_rows, limit),
            "selected_source_ids": sorted(selected_source_ids),
            "post_source_gate": _candidate_trace_rows(filtered_rows, limit),
            "selected_ids": selected,
        },
    }


def select_episode_ids_late_fusion(
    query: str,
    family_results: list[dict],
    episode_lookup: dict[str, dict],
    config: dict | None = None,
) -> tuple[list[str], list[tuple[str, float]]]:
    result = select_episode_ids_late_fusion_with_trace(query, family_results, episode_lookup, config)
    return result["selected_ids"], result["scored"]


def select_episode_ids_late_fusion_with_trace(
    query: str,
    family_results: list[dict],
    episode_lookup: dict[str, dict],
    config: dict | None = None,
) -> dict:
    """Fuse per-family ranked candidates after isolated first-pass retrieval."""
    del query, episode_lookup  # late fusion is rank-based and family-agnostic
    config = resolve_selection_config(config)
    telemetry = get_tuning_section("telemetry")
    if not family_results:
        return {
            "selected_ids": [],
            "scored": [],
            "trace": {
                "mode": "empty",
                "input_families": [],
                "candidates_entering_fusion": [],
                "selected_ids": [],
                "rejected_candidates": [],
            },
        }
    if len(family_results) == 1:
        result = family_results[0]
        scored = result.get("scored", [])
        selected_ids = result.get("selected_ids", [])
        limit = int(telemetry.get("max_rejected_candidates", 8))
        return {
            "selected_ids": selected_ids,
            "scored": scored,
            "trace": {
                "mode": "single_family",
                "input_families": [result.get("family", "unknown")],
                "candidates_entering_fusion": [
                    {"episode_id": ep_id, "score": score}
                    for ep_id, score in scored[:limit]
                ],
                "selected_ids": selected_ids,
                "rejected_candidates": [
                    {"episode_id": ep_id, "score": score}
                    for ep_id, score in scored[len(selected_ids): len(selected_ids) + limit]
                ],
            },
        }

    fused_scores: dict[str, float] = defaultdict(float)
    best_rank: dict[str, int] = {}
    contributor_families: dict[str, set[str]] = defaultdict(set)
    for result in family_results:
        for rank, (ep_id, _score) in enumerate(result.get("scored", [])[: config["late_fusion_per_family"]], start=1):
            fused_scores[ep_id] += 1.0 / (config["rrf_k"] + rank)
            best_rank[ep_id] = min(best_rank.get(ep_id, rank), rank)
            contributor_families[ep_id].add(result.get("family", "unknown"))

    fused = sorted(
        fused_scores.items(),
        key=lambda row: (-row[1], best_rank.get(row[0], 10**9), row[0]),
    )
    selected = [ep_id for ep_id, _score in fused[: config["max_episodes_default"]]]
    limit = int(telemetry.get("max_rejected_candidates", 8))
    return {
        "selected_ids": selected,
        "scored": fused,
        "trace": {
            "mode": "late_fusion",
            "input_families": [result.get("family", "unknown") for result in family_results],
            "candidates_entering_fusion": [
                {
                    "episode_id": ep_id,
                    "score": score,
                    "best_rank": best_rank.get(ep_id),
                    "families": sorted(contributor_families.get(ep_id, set())),
                }
                for ep_id, score in fused[:limit]
            ],
            "selected_ids": selected,
            "rejected_candidates": [
                {
                    "episode_id": ep_id,
                    "score": score,
                    "best_rank": best_rank.get(ep_id),
                    "families": sorted(contributor_families.get(ep_id, set())),
                }
                for ep_id, score in fused[len(selected): len(selected) + limit]
            ],
        },
    }
GENERIC_CHAIN_RELATION_TOKENS = {
    token
    for token in (
        "associated",
        "affiliated",
        "founded",
        "created",
        "plays",
        "play",
        "position",
        "sport",
        "religion",
        "language",
        "city",
        "country",
        "official",
        "documents",
        "written",
        "citizen",
        "citizenship",
        "government",
        "capital",
        "origin",
    )
    for token in word_token_set(token)
}
