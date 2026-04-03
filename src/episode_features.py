#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic episode/query feature helpers for episode-native retrieval."""

from __future__ import annotations

import re
from functools import lru_cache

from .common import normalize_term_token
from .retrieval import detect_query_type

QUERY_WORDS_TO_IGNORE = {
    "what", "which", "when", "where", "who", "why", "how",
    "the", "and", "for", "from", "with", "this", "that", "does",
    "in", "on", "at", "to", "of", "by", "as",
    "should", "would", "could", "about", "after", "before",
    "has", "have", "had", "do", "did", "done", "is", "are", "was", "were",
    "project", "current", "currently", "assigned",
}

GENERIC_TOPIC_TOKENS = {
    "overview", "summary", "context", "framework", "register",
    "document", "metadata", "appendix", "table", "matrix",
    "status", "analysis", "schedule", "reporting", "cross_references",
}

PERMIT_CODE_RE = re.compile(r"\bT-\d+[A-Z]?\b", re.I)
CHAINAGE_RE = re.compile(r"\bkm\s*\d+(?:\.\d+)?\b", re.I)
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
WORD_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)?")
STEP_RE = re.compile(r"\bstep\s+(\d+)\b", re.I)
STEP_FROM_TO_RE = re.compile(r"\bfrom\s+step\s+(\d+)\s+(?:to|through)\s+step\s+(\d+)\b", re.I)
STEP_BETWEEN_RE = re.compile(r"\bbetween\s+steps?\s+(\d+)\s+and\s+(\d+)\b", re.I)
STEP_SPAN_RE = re.compile(r"\bsteps?\s+(\d+)\s*(?:-|to|through)\s*(\d+)\b", re.I)
CAP_ENTITY_RE = re.compile(
    r'(?<!\w)(?:[A-Z][A-Za-z0-9.-]*|"(?:[^"]+)")(?:\s+(?:[A-Z][A-Za-z0-9.-]*|"(?:[^"]+)"))+'
)
SINGLE_CAP_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z0-9.-]*\b")
ORDINAL_NTH_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.I)
COMMONALITY_RE = re.compile(r"\b(in common|both|shared|share|same as)\b", re.I)
LIST_SET_RE = re.compile(
    r"\b(list|enumerate|what are|which are|all(?:\s+the)?)\b",
    re.I,
)
LIST_SET_HEAD_RE = re.compile(
    r"^\s*(?:what|which)\s+(?:kind[s]?\s+of\s+|type[s]?\s+of\s+)?"
    r"(?P<head>[A-Za-z-]+(?:\s+[A-Za-z-]+){0,4})\s+"
    r"(?:has|have|had|does|do|did|are|were|is|was|should|can|could|would)\b",
    re.I,
)
LIST_SET_OF_HEAD_RE = re.compile(
    r"^\s*(?:what|which)\s+of\s+.+?\s+"
    r"(?P<head>[A-Za-z-]+(?:\s+(?:and\s+)?[A-Za-z-]+){0,4})\s+"
    r"(?:has|have|had|does|do|did|are|were|is|was|should|can|could|would)\b",
    re.I,
)
MONTH_NAME_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.I,
)
EXPLICIT_DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}\s+"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"(?:,?\s+\d{4})?)\b",
    re.I,
)
KIND_TYPE_SLOT_RE = re.compile(
    r"^\s*(?:what|which)\s+(?:kind|type)s?\s+of\s+"
    r"(?P<head>[A-Za-z-]+(?:\s+[A-Za-z-]+){0,3})\s+"
    r"(?:has|have|had|does|do|did|are|were|is|was|should|can|could|would)\b",
    re.I,
)
POSSESSIVE_SLOT_RE = re.compile(
    r"^\s*(?:what|which)\s+(?:is|are|was|were|might|may|could|would|should|can)\s+"
    r"(?:(?:one|some)\s+of\s+)?"
    r"(?:(?:[A-Za-z0-9_.-]+(?:\s+[A-Za-z0-9_.-]+){0,5})'s|his|her|their|my|our|your)\s+"
    r"(?P<head>[A-Za-z-]+(?:\s+[A-Za-z-]+){0,3})\b",
    re.I,
)
OF_SLOT_RE = re.compile(
    r"^\s*(?:what|which)\s+(?:is|are|was|were)\s+the\s+"
    r"(?P<head>[A-Za-z-]+(?:\s+[A-Za-z-]+){0,2})\s+of\b",
    re.I,
)
TRAILING_SLOT_FILLER_TOKENS = {"be", "current", "currently", "now", "right", "today"}
SLOT_HEAD_TAIL_BREAK_TOKENS = {"in", "on", "at", "for", "with", "about", "to", "from"}
DIRECT_FIELD_QUERY_RE = re.compile(
    r"^\s*(?:what|which)\s+(?:(?P<modifier>[A-Za-z-]+)\s+)?"
    r"(?P<head>[A-Za-z-]+)\s+"
    r"(?:has|have|had|does|do|did|is|was|are|were|can|could|would|should)\b",
    re.I,
)
COMPARE_DIFF_RE = re.compile(r"\b(compare|difference|different|changed|change|before|after|versus|vs\.?)\b", re.I)
LOCAL_ANCHOR_RE = re.compile(
    r"\b(where|which store|which permit|which step|which section|which clause|which file|which document)\b",
    re.I,
)
GENERIC_WHICH_QUERY_RE = re.compile(r"^\s*which\s+(?:(?:[A-Za-z-]+)\s+)?(?P<head>[A-Za-z-]+)\b", re.I)
COMPOSITIONAL_BOTH_RE = re.compile(r"\bboth\b.+\band\b", re.I)
COMPOSITIONAL_GOAL_RE = re.compile(r"\bso\s+that\b|\bthat\s+would\b", re.I)
COMPOSITIONAL_CONSTRAINT_VERB_RE = re.compile(
    r"\b(make|help|keep|allow|let|enable|support)(?:ing)?\b",
    re.I,
)
TEMPORAL_GROUNDING_RE = re.compile(
    r"\b(first|last|yesterday|tomorrow|tonight|last night|next month|this month|ago|for \d+ years?)\b",
    re.I,
)
RETURN_ONLY_RE = re.compile(r"\b(return only|only return|only answer|do not include any other text)\b", re.I)
JSON_OUTPUT_RE = re.compile(r"\bjson\b", re.I)
PREPEND_RE = re.compile(r"\bprepend(?:\s+(?:the\s+)?)?(?:prefix\s+)?[\"']?([A-Za-z0-9._:-]{2,64})[\"']?\b", re.I)
PREPEND_INSTRUCTION_RE = re.compile(
    r"\bprepend(?:\s+(?:the\s+)?)?(?:prefix\s+)?[\"']?[A-Za-z0-9._:-]{2,64}[\"']?\s+to\s+",
    re.I,
)
JSON_INSTRUCTION_RE = re.compile(
    r"\b(?:return|respond|output|answer)\s+(?:as|in)\s+json\b",
    re.I,
)
ONE_INDEXED_RE = re.compile(r"\b1(?:-|\s)?indexed\b", re.I)
OUTPUT_SUFFIX_INSTRUCTION_RE = re.compile(
    r"[?.!]\s*(?:"
    r"(?:(?:return|respond|answer|output|prepend)\b"
    r"(?=.*\b(?:exact|only|just|json|single|one|token|answer|string|text|value|number|name|word|phrase|id|code|date)\b))"
    r"|"
    r"(?:(?:provide|give|include)\b"
    r"(?=.*\b(?:exact|only|just|json|single|one)\b))"
    r").*$",
    re.I,
)
DIRECTIVE_CAP_TOKENS = {
    "return",
    "respond",
    "answer",
    "output",
    "provide",
    "give",
    "include",
    "prepend",
}
HEAD_TOKEN_IGNORE = {"kind", "kinds", "type", "types", "of", "the", "a", "an", "or", "and"}
FIELD_HEAD_IGNORE = HEAD_TOKEN_IGNORE | {
    "many",
    "much",
    "long",
    "old",
    "new",
}


def _normalize_head_parse_question(question: str) -> str:
    """Normalize lightweight punctuation variants before head parsing."""
    return re.sub(r"(?<=[A-Za-z])/(?=[A-Za-z])", " or ", question)


def _normalize_head_token(token: str) -> str:
    token = (token or "").strip().lower()
    if not token:
        return ""
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("sses", "shes", "ches", "xes", "zes")) and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")) and len(token) > 3:
        return token[:-1]
    return token

ORDINAL_WORD_TO_INDEX = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}


def _semantic_query_expansion_tokens(lower: str) -> set[str]:
    extra = set()
    if "come into existence" in lower:
        extra |= {
            normalize_term_token("founded"),
            normalize_term_token("created"),
        }
    if "country of origin" in lower:
        extra |= {
            normalize_term_token("created"),
            normalize_term_token("founded"),
        }
    return extra


@lru_cache(maxsize=2048)
def episode_sentences(raw_text: str) -> tuple[str, ...]:
    parts = []
    for para in raw_text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        sent_candidates = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+(?=(?:\d+\.\s*)?[A-Z0-9`\"'])", para)
            if s.strip()
        ]
        if len(para) <= 260 and len(sent_candidates) <= 1:
            parts.append(para)
            continue
        for sent in sent_candidates:
            if sent:
                parts.append(sent)
    return tuple(parts)


def episode_text(ep: dict) -> str:
    return " ".join(
        x
        for x in [
            ep.get("topic_key", ""),
            ep.get("state_label", ""),
            ep.get("currentness", ""),
            ep.get("raw_text", ""),
        ]
        if x
    )


def episode_text_lower(ep: dict) -> str:
    return episode_text(ep).lower()


def word_token_set(text: str) -> set[str]:
    return {
        normalize_term_token(token)
        for token in WORD_RE.findall(text)
        if len(token) >= 3 and normalize_term_token(token)
    }


@lru_cache(maxsize=256)
def _exact_step_mention_re(step: int) -> re.Pattern[str]:
    return re.compile(rf"\bstep\s+{int(step)}\b", re.I)


def has_exact_step_mention(text: str, step: int) -> bool:
    if not text:
        return False
    return bool(_exact_step_mention_re(int(step)).search(text))


@lru_cache(maxsize=256)
def step_mentions(text: str) -> tuple[int, ...]:
    if not text:
        return ()
    return tuple(int(m) for m in STEP_RE.findall(text))


def extract_step_range(text: str, step_numbers: set[int] | None = None) -> tuple[int, int] | None:
    if not text:
        return None
    for pattern in (STEP_FROM_TO_RE, STEP_BETWEEN_RE, STEP_SPAN_RE):
        match = pattern.search(text)
        if match:
            start = int(match.group(1))
            end = int(match.group(2))
            if start != end:
                return (min(start, end), max(start, end))
    return None


def step_range_overlap(text: str, step_range: tuple[int, int] | None) -> set[int]:
    if not text or not step_range:
        return set()
    start, end = step_range
    return {
        step
        for step in step_mentions(text)
        if start <= step <= end
    }


def extract_query_features(question: str) -> dict:
    output_constraints = extract_output_constraints(question)
    retrieval_target = build_retrieval_target(question, output_constraints)
    q_lower = retrieval_target.lower()
    words = {
        normalize_term_token(w)
        for w in WORD_RE.findall(retrieval_target)
        if len(w) >= 3 and w.lower() not in QUERY_WORDS_TO_IGNORE and normalize_term_token(w)
    }
    words |= _semantic_query_expansion_tokens(q_lower)
    codes = {m.upper() for m in PERMIT_CODE_RE.findall(retrieval_target)}
    kms = {m.lower().replace(" ", "") for m in CHAINAGE_RE.findall(q_lower)}
    numbers = set(NUMBER_RE.findall(retrieval_target))
    step_numbers = {int(m) for m in STEP_RE.findall(retrieval_target)}
    step_range = extract_step_range(retrieval_target, step_numbers)
    entity_phrases = []
    for match in CAP_ENTITY_RE.findall(retrieval_target):
        cleaned = " ".join(match.split())
        entity_phrases.append(cleaned)
    multi_phrase_tokens = {
        token
        for phrase in entity_phrases
        if len(WORD_RE.findall(phrase)) > 1
        for token in SINGLE_CAP_TOKEN_RE.findall(phrase)
    }
    for token in SINGLE_CAP_TOKEN_RE.findall(retrieval_target):
        lowered = token.lower()
        if (
            len(token) <= 1
            or lowered in QUERY_WORDS_TO_IGNORE
            or lowered in DIRECTIVE_CAP_TOKENS
            or lowered in {"what", "which", "where", "when", "who", "why", "how"}
            or token in multi_phrase_tokens
        ):
            continue
        if token not in entity_phrases:
            entity_phrases.append(token)

    entity_phrase_tokens = {
        phrase.lower(): word_token_set(phrase)
        for phrase in entity_phrases
    }
    entity_tokens = set().union(*entity_phrase_tokens.values()) if entity_phrase_tokens else set()
    words -= entity_tokens

    operator_plan = build_query_operator_plan(
        question,
        retrieval_target=retrieval_target,
        base_query_type=detect_query_type(question),
        step_numbers=step_numbers,
        entity_phrases=entity_phrases,
        codes=codes,
        numbers=numbers,
    )

    mentions_kind = (
        ("what kind of" in q_lower or "which kind of" in q_lower)
        and any(
            tok in q_lower
            for tok in (
                "penalty",
                "clause",
                "contract",
                "variation",
                "scope",
                "permit",
                "rule",
                "constraint",
                "decision",
                "obligation",
                "term",
            )
        )
    )

    return {
        "raw": question,
        "lower": q_lower,
        "query_type": operator_plan["base_query_type"],
        "words": words,
        "codes": codes,
        "kms": kms,
        "numbers": numbers,
        "step_numbers": step_numbers,
        "step_range": step_range,
        "entity_phrases": entity_phrases,
        "entity_phrase_tokens": entity_phrase_tokens,
        "operator_plan": operator_plan,
        "output_constraints": output_constraints,
        "retrieval_target": retrieval_target,
        "asks_where": "where" in q_lower,
        # Compatibility flags kept as passive metadata for callers/tests.
        "mentions_permit": "permit" in q_lower or "permits" in q_lower,
        "mentions_compliance": "compliant" in q_lower or "compliance" in q_lower,
        "mentions_scope": "scope" in q_lower or "project-shared" in q_lower or "agent-private" in q_lower,
        "mentions_kind": mentions_kind,
    }


def build_query_operator_plan(
    question: str,
    *,
    retrieval_target: str | None = None,
    base_query_type: str | None = None,
    step_numbers: set[int] | None = None,
    entity_phrases: list[str] | None = None,
    codes: set[str] | None = None,
    numbers: set[str] | None = None,
) -> dict:
    lower = question.lower()
    step_numbers = step_numbers or set()
    entity_phrases = entity_phrases or []
    codes = codes or set()
    numbers = numbers or set()

    ordinal_mode = None
    ordinal_index = None
    order_basis = None
    for word, index in ORDINAL_WORD_TO_INDEX.items():
        if re.search(rf"\b{word}\b", lower):
            ordinal_mode = "nth"
            ordinal_index = index
            order_basis = "match_order"
            break
    if ordinal_mode is None:
        nth_match = ORDINAL_NTH_RE.search(question)
        if nth_match:
            ordinal_mode = "nth"
            ordinal_index = int(nth_match.group(1))
            order_basis = "match_order"
    if ordinal_mode is None and re.search(r"\bpenultimate\b", lower):
        ordinal_mode = "penultimate"
        order_basis = "match_order"
    if ordinal_mode is None and re.search(r"\blast\b", lower):
        ordinal_mode = "last"
        order_basis = "match_order"
    if ordinal_mode is None and re.search(
        r"\bfinal\s+(?:one|item|entry|candidate|version|step|mention|response|artifact)\b",
        lower,
    ):
        ordinal_mode = "last"
        order_basis = "match_order"

    commonality_enabled = bool(COMMONALITY_RE.search(question))
    list_set_enabled = bool(LIST_SET_RE.search(question))
    head_parse_question = _normalize_head_parse_question(question)
    list_set_head_phrase = None
    list_set_head_tokens: list[str] = []
    slot_query_enabled = False
    slot_head_phrase = None
    slot_head_tokens: list[str] = []
    if not list_set_enabled:
        head_match = LIST_SET_HEAD_RE.search(head_parse_question) or LIST_SET_OF_HEAD_RE.search(head_parse_question)
        if head_match:
            list_set_head_phrase = " ".join(head_match.group("head").lower().split())
            raw_head_tokens = [
                token.lower()
                for token in WORD_RE.findall(head_match.group("head"))
                if token.lower() not in HEAD_TOKEN_IGNORE
            ]
            head_tokens = [
                _normalize_head_token(token)
                for token in raw_head_tokens
                if _normalize_head_token(token)
            ]
            list_set_head_tokens = head_tokens
            list_set_enabled = any(
                token.endswith("s") and len(token) >= 4
                for token in raw_head_tokens
            )
    else:
        head_match = LIST_SET_HEAD_RE.search(head_parse_question) or LIST_SET_OF_HEAD_RE.search(head_parse_question)
        if head_match:
            list_set_head_phrase = " ".join(head_match.group("head").lower().split())
            list_set_head_tokens = [
                _normalize_head_token(token)
                for token in WORD_RE.findall(head_match.group("head"))
                if token.lower() not in HEAD_TOKEN_IGNORE and _normalize_head_token(token)
            ]
    kind_type_match = KIND_TYPE_SLOT_RE.search(question)
    possessive_slot_match = POSSESSIVE_SLOT_RE.search(question)
    of_slot_match = OF_SLOT_RE.search(question)
    literal_field_match = DIRECT_FIELD_QUERY_RE.search(question)

    def _assign_slot_head(raw_head: str, *, disable_list_set: bool = False) -> None:
        nonlocal slot_query_enabled, slot_head_phrase, slot_head_tokens, list_set_enabled
        raw_tokens = [token.lower() for token in WORD_RE.findall(raw_head)]
        for idx, token in enumerate(raw_tokens[1:], start=1):
            if token in SLOT_HEAD_TAIL_BREAK_TOKENS:
                raw_tokens = raw_tokens[:idx]
                break
        while raw_tokens and raw_tokens[-1] in TRAILING_SLOT_FILLER_TOKENS:
            raw_tokens.pop()
        normalized_tokens = [
            token
            for token in raw_tokens
            if token not in HEAD_TOKEN_IGNORE
        ]
        if not normalized_tokens:
            return
        slot_query_enabled = True
        slot_head_phrase = " ".join(raw_tokens)
        slot_head_tokens = normalized_tokens
        if disable_list_set:
            list_set_enabled = False

    if kind_type_match:
        _assign_slot_head(kind_type_match.group("head"))
    elif possessive_slot_match:
        _assign_slot_head(possessive_slot_match.group("head"), disable_list_set=True)
    elif of_slot_match:
        _assign_slot_head(of_slot_match.group("head"), disable_list_set=True)
    elif literal_field_match:
        head = literal_field_match.group("head").lower()
        modifier = (literal_field_match.group("modifier") or "").lower()
        if (
            head not in FIELD_HEAD_IGNORE
            and not (head.endswith("s") and len(head) >= 4)
            and modifier not in {"many", "much"}
        ):
            _assign_slot_head(head)
    compare_diff_enabled = bool(COMPARE_DIFF_RE.search(question))
    local_anchor_enabled = bool(LOCAL_ANCHOR_RE.search(question))
    if not local_anchor_enabled and GENERIC_WHICH_QUERY_RE.search(question):
        local_anchor_enabled = bool(entity_phrases or codes or step_numbers or numbers)
    compositional_enabled = False
    if not any([commonality_enabled, list_set_enabled, compare_diff_enabled]):
        has_both_clause = bool(COMPOSITIONAL_BOTH_RE.search(question))
        has_goal_clause = bool(COMPOSITIONAL_GOAL_RE.search(question))
        has_constraint_while = "while" in lower and bool(COMPOSITIONAL_CONSTRAINT_VERB_RE.search(question))
        compositional_enabled = has_both_clause or has_goal_clause or has_constraint_while
    temporal_grounding_enabled = bool(
        TEMPORAL_GROUNDING_RE.search(lower)
        or MONTH_NAME_RE.search(question)
        or EXPLICIT_DATE_RE.search(question)
    )
    if step_numbers:
        local_anchor_enabled = True

    bounded_chain_enabled = False
    if not any([commonality_enabled, list_set_enabled, compare_diff_enabled]):
        if re.search(r"^\s*(?:(?:in|on|at|from|for|to|of|by)\s+)?(what|which|where)\b", lower) and (
            len(entity_phrases) >= 1 or len(codes) >= 1 or len(numbers) >= 2
        ):
            bounded_chain_enabled = True

    operators = []
    if ordinal_mode is not None:
        operators.append("ordinal")
    if commonality_enabled:
        operators.append("commonality")
    if list_set_enabled:
        operators.append("list_set")
    if compare_diff_enabled:
        operators.append("compare_diff")
    if local_anchor_enabled:
        operators.append("local_anchor")
    if bounded_chain_enabled:
        operators.append("bounded_chain")
    if slot_query_enabled:
        operators.append("slot_query")
    if compositional_enabled:
        operators.append("compositional")
    if temporal_grounding_enabled:
        operators.append("temporal_grounding")

    return {
        "base_query_type": base_query_type or detect_query_type(retrieval_target or question),
        "retrieval_target": retrieval_target or question,
        "output_constraints": extract_output_constraints(question),
        "operators": operators,
        "ordinal": {
            "enabled": ordinal_mode is not None,
            "mode": ordinal_mode,
            "index": ordinal_index,
            "order_basis": order_basis,
        },
        "commonality": {"enabled": commonality_enabled},
        "list_set": {
            "enabled": list_set_enabled,
            "preserve_source_order": True,
            "head_phrase": list_set_head_phrase,
            "head_tokens": list_set_head_tokens,
        },
        "compare_diff": {"enabled": compare_diff_enabled},
        "local_anchor": {"enabled": local_anchor_enabled},
        "bounded_chain": {"enabled": bounded_chain_enabled, "max_hops": 2},
        "slot_query": {
            "enabled": slot_query_enabled,
            "head_phrase": slot_head_phrase,
            "head_tokens": [token for token in slot_head_tokens if token],
        },
        "compositional": {"enabled": compositional_enabled},
        "temporal_grounding": {"enabled": temporal_grounding_enabled},
    }


def extract_output_constraints(question: str) -> dict:
    prepend_match = PREPEND_RE.search(question)
    return {
        "return_only": bool(RETURN_ONLY_RE.search(question)),
        "json_output": bool(JSON_OUTPUT_RE.search(question)),
        "prepend_prefix": prepend_match.group(1) if prepend_match else None,
    }


def build_retrieval_target(question: str, output_constraints: dict | None = None) -> str:
    text = question
    output_constraints = output_constraints or extract_output_constraints(question)

    if output_constraints.get("prepend_prefix"):
        text = PREPEND_INSTRUCTION_RE.sub("", text)
    text = OUTPUT_SUFFIX_INSTRUCTION_RE.sub("", text)
    text = RETURN_ONLY_RE.sub(" ", text)
    if output_constraints.get("json_output"):
        text = JSON_INSTRUCTION_RE.sub(" ", text)
    text = ONE_INDEXED_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n.:;,-")
    return text or question


def generic_episode_penalty(topic_key: str, state_label: str) -> float:
    lower = f"{topic_key} {state_label}".lower()
    return float(sum(1 for tok in GENERIC_TOPIC_TOKENS if tok in lower))


def episode_codes(ep: dict) -> set[str]:
    text = " ".join([
        ep.get("topic_key", ""),
        ep.get("state_label", ""),
        ep.get("raw_text", ""),
    ])
    return {m.upper() for m in PERMIT_CODE_RE.findall(text)}


def episode_chainages(ep: dict) -> set[str]:
    text = " ".join([
        ep.get("topic_key", ""),
        ep.get("state_label", ""),
        ep.get("raw_text", ""),
    ]).lower()
    return {m.lower().replace(" ", "") for m in CHAINAGE_RE.findall(text)}
