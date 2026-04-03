#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Episode packet assembly helpers."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta

from .common import normalize_term_token
from .episode_features import (
    CHAINAGE_RE,
    NUMBER_RE,
    PERMIT_CODE_RE,
    QUERY_WORDS_TO_IGNORE,
    WORD_RE,
    episode_sentences,
    extract_query_features,
    has_exact_step_mention,
    step_mentions,
    step_range_overlap,
    word_token_set,
)
from .tuning import get_tuning_section

GENERIC_FACT_TOKENS = {
    "user", "assistant", "speaker", "people", "person", "thing", "things",
    "something", "anything", "everything", "someone", "anyone", "everyone",
    "community", "project", "experience", "story", "support", "supportive",
    "great", "good", "cool", "awesome", "nice", "fun", "interesting",
    "okay", "today", "yesterday", "tomorrow", "week", "month", "year",
}
GENERIC_CHAIN_BRIDGE_TOKENS = {
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
}
GENERIC_LOCATION_TOKENS = {
    "plac",
    "event",
    "venue",
    "city",
    "country",
    "continent",
    "capital",
    "state",
    "province",
    "territory",
    "region",
    "island",
    "origin",
    "located",
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
GENERIC_LIST_SET_HEAD_TOKENS = {
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
LIST_ITEM_KEY_IGNORE_TOKENS = {
    "next",
    "today",
    "tomorrow",
    "yesterday",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "join",
    "meet",
    "meeting",
    "free",
}
LIST_ITEM_LEADING_IGNORE_TOKENS = {
    "a",
    "an",
    "the",
    "to",
    "at",
    "in",
    "on",
    "for",
    "of",
    "with",
    "into",
    "onto",
    "from",
    "by",
    "about",
    "around",
    "through",
    "up",
    "out",
    "then",
    "just",
    "also",
    "really",
    "pretty",
    "well",
    "yes",
    "yeah",
    "hey",
    "hi",
    "hello",
    "i",
    "we",
    "you",
    "he",
    "she",
    "they",
    "it",
    "im",
    "i'm",
    "its",
    "it's",
    "lets",
    "let's",
    "will",
    "would",
    "should",
    "could",
    "might",
    "must",
    "shall",
}
LIST_ITEM_FRAGMENT_IGNORE_TOKENS = LIST_ITEM_KEY_IGNORE_TOKENS | LIST_ITEM_LEADING_IGNORE_TOKENS | {
    "want",
    "join",
    "see",
    "agree",
    "great",
    "heard",
    "catch",
    "lately",
    "later",
    "together",
    "about",
    "been",
    "being",
    "am",
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "done",
    "doing",
    "go",
    "goes",
    "going",
    "went",
    "train",
    "trained",
    "training",
    "play",
    "played",
    "playing",
    "goe",
    "offer",
    "offers",
    "offering",
    "include",
    "includes",
    "including",
    "thought",
    "can",
    "will",
    "would",
    "should",
    "could",
    "might",
    "must",
    "shall",
    "try",
    "let",
    "what",
    "im",
    "its",
    "doesn",
    "don",
    "didn",
    "won",
}
LIST_ITEM_PLAN_GENERIC_TOKENS = {
    "talk",
    "meet",
    "keep",
    "updat",
    "wait",
    "soon",
    "later",
    "great",
    "good",
    "thing",
    "event",
    "activ",
}
LIST_ITEM_FRAGMENT_SPLIT_RE = re.compile(r"\s*(?:,|/|\band\b|\bor\b)\s*", re.I)
LIST_ITEM_FRAGMENT_STOP_RE = re.compile(
    r"\b("
    r"next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|month)|"
    r"today|tomorrow|yesterday|tonight|"
    r"if|when|because|after|before|later"
    r")\b",
    re.I,
)
GREETING_FACT_RE = re.compile(
    r"^(?:[A-Z][a-z]+:\s*)?(?:hey|hi|hello|how's it going|what's up|long time no see)\b",
    re.I,
)
LIST_SET_ITEM_VERB_RE = re.compile(
    r"\b("
    r"doing|do|did|done|has done|have done|going to do|gonna do|will do|"
    r"practice|practiced|practices|train|trained|training|play|plays|played|playing|"
    r"go|goes|went|going|meet|meets|meeting|join|joins|joined|attend|attends|attended|attending|serve|serves|serving|"
    r"offer|offers|offering|include|includes|including"
    r")\b",
    re.I,
)
LOW_INFORMATION_FACT_RE = re.compile(
    r"^(?:[A-Z][A-Za-z\"' -]+ )?(?:is|was|are|were|agrees|agree|has to|have to|had to|hopes|hope|said|says|heard|looks|looked)\b",
    re.I,
)
FUTURE_TIME_ANCHOR_RE = re.compile(
    r"\b("
    r"tomorrow|tonight|later|this\s+(?:weekend|week|month)|"
    r"next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|month|year)|"
    r"on\s+\d{4}-\d{2}-\d{2}|"
    r"on\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r")\b",
    re.I,
)
FUTURE_COMMITMENT_RE = re.compile(
    r"\b("
    r"let'?s|will|going to|plan(?:ning)? to|scheduled to"
    r")\b",
    re.I,
)
FUTURE_TENTATIVE_RE = re.compile(
    r"\b("
    r"maybe|perhaps|might|could|if"
    r")\b",
    re.I,
)
LIST_ITEM_ANCHOR_RE = re.compile(
    r"\b(?:[A-Z]{2,}|[A-Z][A-Za-z]+(?:'[A-Za-z]+)?(?:\s+[A-Z][A-Za-z]+(?:'[A-Za-z]+)?){0,3})\b"
)

ACTIVITY_SLOT_HEAD_TOKENS = {
    "activity", "activities", "hobby", "hobbies", "pastime", "pastimes",
    "sport", "sports", "game", "games",
}
ACTIVITY_SLOT_VERB_RE = re.compile(
    r"\b(?:go|goes|went|going|play|plays|played|playing|practice|practices|practiced|practicing|"
    r"train|trains|trained|training|pursue|pursues|pursued|pursuing|enjoy|enjoys|enjoyed|"
    r"love|loves|loved|do|does|did|doing|try|tries|tried)\s+(?P<cand>[A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){0,2})",
    re.I,
)
ARTICLE_ITEM_PHRASE_RE = re.compile(
    r"^(?:the|a|an)\s+(?P<phrase>[A-Za-z]+(?:\s+[A-Za-z]+){0,3})\b",
    re.I,
)
ARTICLE_SLOT_CANDIDATE_RE = re.compile(
    r"\b(?:the|a|an)\s+(?P<phrase>[A-Za-z][A-Za-z0-9&'._-]*(?:\s+[A-Za-z][A-Za-z0-9&'._-]*){0,5})",
    re.I,
)
QUOTED_SLOT_CANDIDATE_RE = re.compile(r'"([^"]{2,120})"|\'([^\']{2,120})\'')
TITLEISH_SLOT_CANDIDATE_RE = re.compile(
    r"[A-Z][A-Za-z0-9&'._-]+(?:\s+[A-Z][A-Za-z0-9&'._-]+){0,7}"
)
TURN_QUERY_LINE_RE = re.compile(r"^\[Turn query\]\s*(?P<query>.+?)\s*$", re.I | re.M)
RELATIVE_LAST_NIGHT_RE = re.compile(r"\blast night\b", re.I)
RELATIVE_YESTERDAY_RE = re.compile(r"\byesterday\b", re.I)
RELATIVE_TOMORROW_RE = re.compile(r"\btomorrow\b", re.I)
RELATIVE_FOR_YEARS_RE = re.compile(r"\bfor\s+(\d+)\s+years?\b", re.I)
RELATIVE_AGO_YEARS_RE = re.compile(r"\baround\s+(\d+)\s+years?\s+ago\b", re.I)
SLOT_LINKING_VERB_FRAGMENT = r"(?:is|was|are|were|be|being|been|called|named|titled)"
GENERIC_SLOT_MODIFIER_TOKENS = {
    "big",
    "small",
    "new",
    "old",
    "current",
    "favorite",
    "certain",
    "specific",
    "another",
    "other",
    "various",
    "temp",
    "hard",
    "tough",
    "difficult",
    "easy",
    "good",
    "bad",
}
_SLOT_META_CANDIDATE_TOKENS = {
    "answer", "retrieved", "facts", "fact", "raw", "context", "episode", "source",
    "section", "sections", "provided", "information", "query", "question",
}

GENERIC_SLOT_CONTEXT_TOKENS = {
    "an",
    "the",
    "get",
    "gets",
    "got",
    "getting",
    "take",
    "takes",
    "took",
    "taken",
    "taking",
    "do",
    "does",
    "did",
    "done",
    "doing",
    "work",
    "works",
    "worked",
    "working",
    "make",
    "makes",
    "made",
    "making",
    "help",
    "helps",
    "helping",
    "cover",
    "covers",
    "covered",
    "try",
    "tries",
    "tried",
    "use",
    "uses",
    "used",
    "using",
    "his",
    "her",
    "their",
    "my",
    "our",
    "your",
}
GENERIC_CHAIN_RELATION_TOKENS = {
    token
    for phrase in GENERIC_CHAIN_BRIDGE_TOKENS
    for token in word_token_set(phrase)
}


def _normalize_token_set(tokens: set[str]) -> set[str]:
    normalized: set[str] = set()
    for token in tokens:
        if token:
            normalized.add(token)
        norm = normalize_term_token(token)
        if norm:
            normalized.add(norm)
    return normalized


GENERIC_FACT_TOKENS = _normalize_token_set(GENERIC_FACT_TOKENS)
GENERIC_LOCATION_TOKENS = _normalize_token_set(GENERIC_LOCATION_TOKENS)
GENERIC_LIST_SET_HEAD_TOKENS = _normalize_token_set(GENERIC_LIST_SET_HEAD_TOKENS)
LIST_ITEM_KEY_IGNORE_TOKENS = _normalize_token_set(LIST_ITEM_KEY_IGNORE_TOKENS)
LIST_ITEM_LEADING_IGNORE_TOKENS = _normalize_token_set(LIST_ITEM_LEADING_IGNORE_TOKENS)
LIST_ITEM_FRAGMENT_IGNORE_TOKENS = _normalize_token_set(LIST_ITEM_FRAGMENT_IGNORE_TOKENS)
LIST_ITEM_PLAN_GENERIC_TOKENS = _normalize_token_set(LIST_ITEM_PLAN_GENERIC_TOKENS)
GENERIC_SLOT_MODIFIER_TOKENS = _normalize_token_set(GENERIC_SLOT_MODIFIER_TOKENS)
GENERIC_SLOT_CONTEXT_TOKENS = _normalize_token_set(GENERIC_SLOT_CONTEXT_TOKENS)


def _fact_overlap_score(fact_text: str, qf: dict) -> float:
    score = 0.0
    lower = fact_text.lower()
    tokens = word_token_set(fact_text)
    numbers = set(NUMBER_RE.findall(fact_text))
    explicit_steps = set(qf.get("step_numbers", set()) or set())
    for code in qf["codes"]:
        if code.lower() in lower:
            score += 8.0
    for km in qf["kms"]:
        if km in lower.replace(" ", ""):
            score += 6.0
    for number in qf["numbers"]:
        if number in numbers:
            score += 1.5
    for word in _fact_query_words(qf):
        if word in tokens:
            score += 0.5
    for phrase in qf.get("entity_phrases", []):
        phrase_lower = phrase.lower()
        phrase_tokens = qf.get("entity_phrase_tokens", {}).get(phrase_lower) or word_token_set(phrase)
        if (" " in phrase_lower and phrase_lower in lower) or (
            " " not in phrase_lower and phrase_tokens and phrase_tokens <= tokens
        ):
            score += 2.5 if " " in phrase_lower else 1.0
    for step in explicit_steps:
        if has_exact_step_mention(fact_text, step):
            score += 3.0
    interior_hits = step_range_overlap(fact_text, qf.get("step_range")) - explicit_steps
    if interior_hits:
        score += min(3.0, 1.5 * len(interior_hits))
    return score


def _query_specificity_score(
    fact_text: str,
    qf: dict,
    token_freq: dict[str, int],
    bonus: float,
) -> float:
    if bonus <= 0:
        return 0.0
    tokens = word_token_set(fact_text)
    score = 0.0
    for word in _fact_query_words(qf):
        if word in tokens:
            score += 1.0 / max(1, token_freq.get(word, 1))
    return bonus * score


def _fact_query_words(qf: dict) -> set[str]:
    words = set(qf.get("words", set()))
    operator_plan = qf.get("operator_plan", {})
    if operator_plan.get("list_set", {}).get("enabled"):
        words -= set(operator_plan.get("list_set", {}).get("head_tokens") or [])
    return words


def _entity_anchor_tokens(qf: dict) -> set[str]:
    tokens = set()
    for phrase in qf.get("entity_phrases", []):
        tokens |= qf.get("entity_phrase_tokens", {}).get(phrase.lower(), set())
    return tokens


def _fact_content_tokens(fact_text: str, qf: dict) -> list[str]:
    entity_tokens = _entity_anchor_tokens(qf)
    operator_plan = qf.get("operator_plan", {})
    category_tokens = set(operator_plan.get("list_set", {}).get("head_tokens") or [])
    category_tokens |= set(operator_plan.get("slot_query", {}).get("head_tokens") or [])
    content = []
    for token in WORD_RE.findall(fact_text):
        lowered = token.lower()
        norm = normalize_term_token(token)
        if len(lowered) < 3:
            continue
        if not norm:
            continue
        if norm in QUERY_WORDS_TO_IGNORE:
            continue
        if norm in qf.get("words", set()):
            continue
        if norm in entity_tokens:
            continue
        if norm in category_tokens:
            continue
        if norm in GENERIC_FACT_TOKENS:
            continue
        content.append(norm)
    return content


def _clean_slot_candidate_tokens(
    raw_tokens: list[str],
    qf: dict,
    head_tokens: set[str],
    *,
    keep_coordination: bool = False,
) -> list[str]:
    cleaned: list[str] = []
    query_words = qf.get("words", set())
    entity_tokens = _entity_anchor_tokens(qf)
    primary_entity_tokens: set[str] = set()
    entity_phrases = qf.get("entity_phrases", [])
    if entity_phrases:
        primary_entity_tokens = set((qf.get("entity_phrase_tokens", {}) or {}).get(entity_phrases[0].lower(), set()))
    for raw in raw_tokens:
        norm = normalize_term_token(raw)
        if keep_coordination and raw.lower() == "and" and cleaned:
            cleaned.append(raw)
            continue
        if not norm:
            continue
        if len(norm) < 2:
            continue
        if norm in QUERY_WORDS_TO_IGNORE:
            continue
        if norm in query_words and not (raw[:1].isupper() and norm not in entity_tokens and norm not in head_tokens):
            continue
        if norm in entity_tokens:
            if raw[:1].isupper() and norm not in primary_entity_tokens and norm not in head_tokens:
                cleaned.append(raw)
                continue
            continue
        if norm in head_tokens:
            continue
        if norm in GENERIC_FACT_TOKENS:
            continue
        if norm in GENERIC_SLOT_MODIFIER_TOKENS:
            continue
        if norm in GENERIC_SLOT_CONTEXT_TOKENS:
            continue
        cleaned.append(raw)
    while keep_coordination and cleaned and cleaned[-1].lower() == "and":
        cleaned.pop()
    return cleaned


def _normalize_slot_candidate_fragment(
    fragment: str,
    qf: dict,
    head_tokens: set[str],
    *,
    keep_coordination: bool = False,
) -> str | None:
    text = re.split(r"[,;()]", fragment, maxsplit=1)[0]
    text = text.strip(" \t\r\n:;,.!?-")
    if not text:
        return None

    quoted_matches = [
        first or second
        for first, second in QUOTED_SLOT_CANDIDATE_RE.findall(text)
        if first or second
    ]
    titleish = False
    raw_tokens: list[str]
    if keep_coordination:
        raw_tokens = WORD_RE.findall(text)
    elif quoted_matches:
        raw_tokens = WORD_RE.findall(max(quoted_matches, key=len))
        titleish = True
    else:
        titleish_spans = [match.group(0) for match in TITLEISH_SLOT_CANDIDATE_RE.finditer(text)]
        if titleish_spans:
            raw_tokens = WORD_RE.findall(max(titleish_spans, key=len))
            titleish = True
        else:
            raw_tokens = WORD_RE.findall(text)
    cleaned = _clean_slot_candidate_tokens(
        raw_tokens,
        qf,
        head_tokens,
        keep_coordination=keep_coordination,
    )
    if not cleaned:
        return None
    single_token_ing = len(cleaned) == 1 and cleaned[0].lower().endswith("ing") and not quoted_matches
    activity_head = bool(head_tokens & ACTIVITY_SLOT_HEAD_TOKENS)
    article_scaffold = bool(raw_tokens and raw_tokens[0].lower() in {"a", "an", "the", "my", "our", "your", "his", "her", "their", "its"})
    leading = cleaned[0].lower()
    if leading in {
        "but", "because", "although", "though", "while", "when", "where",
        "which", "that", "if", "then",
    }:
        return None
    if leading in {
        "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did",
    }:
        return None
    if single_token_ing and not activity_head and not article_scaffold:
        return None
    if len(cleaned) == 1 and leading in {"doing", "following", "going", "working", "being", "having"} and not quoted_matches:
        return None
    if len(cleaned) == 1 and not titleish and len(cleaned[0]) < 4:
        return None
    if any(token.lower() in _SLOT_META_CANDIDATE_TOKENS for token in cleaned):
        return None
    return " ".join(cleaned)


def _extract_activity_slot_candidates(fact_text: str, qf: dict, head_token_set: set[str]) -> list[str]:
    if not (head_token_set & ACTIVITY_SLOT_HEAD_TOKENS):
        return []
    candidates: list[str] = []
    seen: set[str] = set()
    for match in ACTIVITY_SLOT_VERB_RE.finditer(fact_text or ""):
        candidate = _normalize_slot_candidate_fragment(
            match.group("cand"),
            qf,
            head_token_set,
        )
        if not candidate:
            continue
        lowered = candidate.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        candidates.append(candidate)
    return candidates


def _fact_slot_fill_candidates(
    fact_text: str,
    qf: dict,
    *,
    allow_loose_fallback: bool = True,
) -> list[str]:
    slot_plan = (qf.get("operator_plan") or {}).get("slot_query") or {}
    if not slot_plan.get("enabled"):
        return []
    head_tokens = [
        normalize_term_token(token)
        for token in (slot_plan.get("head_tokens") or [])
        if normalize_term_token(token)
    ]
    if not head_tokens:
        return []

    head_token_set = set(head_tokens)
    raw_words = WORD_RE.findall(fact_text)
    norm_words = [normalize_term_token(word) for word in raw_words]
    candidates: list[str] = []
    seen: set[str] = set()
    head_len = len(head_tokens)
    article_like_tokens = {"a", "an", "the", "my", "our", "your", "his", "her", "their", "its"}

    def _add_candidate_fragment(fragment: str, *, keep_coordination: bool = False) -> None:
        candidate = _normalize_slot_candidate_fragment(
            fragment,
            qf,
            head_token_set,
            keep_coordination=keep_coordination,
        )
        if candidate and candidate not in seen:
            candidates.append(candidate)
            seen.add(candidate)

    for idx in range(max(0, len(norm_words) - head_len + 1)):
        if norm_words[idx:idx + head_len] != head_tokens:
            continue
        raw_prefix = raw_words[max(0, idx - 3):idx]
        prefix_start = next(
            (pos for pos, token in enumerate(raw_prefix) if token.lower() in article_like_tokens),
            None,
        )
        if prefix_start is not None:
            _add_candidate_fragment(" ".join(raw_prefix[prefix_start:]))
        else:
            search_start = max(0, idx - 8)
            raw_inverse = raw_words[search_start:idx]
            norm_inverse = norm_words[search_start:idx]
            for split_idx in range(len(norm_inverse) - 1, -1, -1):
                raw_token = raw_inverse[split_idx].lower()
                norm_token = norm_inverse[split_idx]
                if raw_token not in {"is", "was", "are", "were", "be", "being", "been", "called", "named", "titled"} and norm_token not in {"be", "call", "name", "title"}:
                    continue
                subject_tokens = raw_inverse[split_idx + 1:]
                subject_lowers = {token.lower() for token in subject_tokens}
                if not (
                    "s" in subject_lowers
                    or subject_lowers & {"his", "her", "their", "my", "our", "your", "its"}
                ):
                    continue
                inverse = _clean_slot_candidate_tokens(
                    raw_inverse[:split_idx],
                    qf,
                    head_token_set,
                    keep_coordination=True,
                )
                if inverse:
                    _add_candidate_fragment(" ".join(raw_inverse[:split_idx]), keep_coordination=True)
                break

        suffix_start = idx + head_len
        saw_linking = False
        while suffix_start < len(raw_words):
            raw_token = raw_words[suffix_start].lower()
            norm_token = norm_words[suffix_start]
            if raw_token in {"is", "was", "are", "were", "be", "being", "been", "called", "named", "titled"} or norm_token in {"be", "call", "name", "title"}:
                saw_linking = True
                suffix_start += 1
                continue
            break
        if saw_linking:
            raw_suffix = raw_words[suffix_start:suffix_start + 16]
            suffix = _clean_slot_candidate_tokens(raw_suffix, qf, head_token_set, keep_coordination=True)
            if suffix:
                _add_candidate_fragment(" ".join(raw_suffix), keep_coordination=True)

    head_phrase = slot_plan.get("head_phrase") or " ".join(head_tokens)
    if head_phrase:
        escaped = re.escape(head_phrase)
        for pattern in (
            rf"(?P<cand>[^.:\n]{{1,100}}?)\b{SLOT_LINKING_VERB_FRAGMENT}\b[^.:\n]{{0,30}}\b{escaped}\b",
            rf"\b{escaped}\b[^.:\n]{{0,30}}\b{SLOT_LINKING_VERB_FRAGMENT}\b(?P<cand>[^.:\n]{{1,100}})",
        ):
            for match in re.finditer(pattern, fact_text, re.I):
                candidate = _normalize_slot_candidate_fragment(
                    match.group("cand"),
                    qf,
                    head_token_set,
                    keep_coordination=True,
                )
                if candidate and candidate not in seen:
                    candidates.append(candidate)
                    seen.add(candidate)
    if not candidates:
        for candidate in _extract_activity_slot_candidates(fact_text, qf, head_token_set):
            if candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)
    if allow_loose_fallback and not candidates:
        for match in TITLEISH_SLOT_CANDIDATE_RE.finditer(fact_text):
            candidate = _normalize_slot_candidate_fragment(
                match.group(0),
                qf,
                head_token_set,
            )
            if candidate and candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)
    if allow_loose_fallback and not candidates:
        for match in ARTICLE_SLOT_CANDIDATE_RE.finditer(fact_text):
            candidate = _normalize_slot_candidate_fragment(
                match.group("phrase"),
                qf,
                head_token_set,
            )
            if candidate and candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)
    return candidates


def _slot_content_head_tokens(qf: dict) -> list[str]:
    slot_plan = (qf.get("operator_plan") or {}).get("slot_query") or {}
    head_tokens = [token for token in slot_plan.get("head_tokens") or [] if token]
    head_phrase = str(slot_plan.get("head_phrase") or "")
    variants: list[str] = []
    seen: set[str] = set()

    def _add(token: str) -> None:
        norm = normalize_term_token(token)
        if not norm or norm in GENERIC_SLOT_MODIFIER_TOKENS or norm in seen:
            return
        seen.add(norm)
        variants.append(norm)

    for token in head_tokens:
        _add(token)
    for raw in WORD_RE.findall(head_phrase):
        _add(raw.lower())

    return variants or head_tokens[-1:]


def _turn_query_slot_candidate(raw_query: str, qf: dict, head_tokens: set[str]) -> str | None:
    raw_words = WORD_RE.findall(raw_query)
    if not raw_words:
        return None
    norm_words = [normalize_term_token(word) for word in raw_words]
    boundary_indexes = [idx for idx, token in enumerate(norm_words) if token in head_tokens]
    if not boundary_indexes:
        return None
    prefix_words = raw_words[:boundary_indexes[0]]
    if len(prefix_words) < 2:
        return None
    return " ".join(word.lower() for word in prefix_words).strip()


def _slot_primary_entity_tokens(qf: dict) -> tuple[str, set[str]]:
    primary_entity = ""
    primary_tokens: set[str] = set()
    for phrase in qf.get("entity_phrases") or []:
        if not phrase:
            continue
        primary_entity = str(phrase).strip().lower()
        primary_tokens = {
            normalize_term_token(token)
            for token in WORD_RE.findall(primary_entity)
            if normalize_term_token(token)
        }
        if primary_tokens:
            break
    return primary_entity, primary_tokens


def _slot_candidate_adds_new_info(candidate: str, qf: dict, head_tokens: set[str]) -> bool:
    candidate_tokens = {
        normalize_term_token(token)
        for token in WORD_RE.findall(candidate)
        if normalize_term_token(token)
    }
    candidate_tokens -= head_tokens
    candidate_tokens -= set(qf.get("words") or set())
    candidate_tokens -= {"the", "a", "an", "one", "provided", "context", "mentioned"}
    return bool(candidate_tokens)


def _clip_headless_slot_fragment(
    fragment: str,
    *,
    max_tokens: int = 6,
) -> str:
    stop_tokens = {
        "and", "or", "but", "because", "while", "when", "where", "which", "that",
        "another", "for", "with", "after", "before", "during", "since", "until",
        "on", "at", "by", "from", "last", "next", "this", "these", "those",
    }
    raw_words = WORD_RE.findall(fragment)
    kept: list[str] = []
    for word in raw_words:
        norm = normalize_term_token(word)
        if kept and norm in stop_tokens:
            break
        kept.append(word)
        if len(kept) >= max_tokens:
            break
    return " ".join(kept).strip()


def _headless_slot_relation_candidates(
    fact_text: str,
    qf: dict,
) -> list[str]:
    slot_plan = (qf.get("operator_plan") or {}).get("slot_query") or {}
    if not slot_plan.get("enabled"):
        return []

    head_tokens = {
        normalize_term_token(token)
        for token in (slot_plan.get("head_tokens") or [])
        if normalize_term_token(token)
    }
    if not head_tokens:
        return []

    vehicle_heads = {"car", "vehicle", "truck", "van", "bike", "bicycle", "motorcycle", "suv", "sedan"}
    country_heads = {"country", "nation"}
    location_heads = country_heads | {"city", "state", "province", "region", "location", "place", "destination", "area"}
    activity_heads = {"activity", "activities", "hobby", "hobbies", "pastime", "pastimes", "sport", "sports", "game", "games"}

    patterns: list[tuple[str, int]] = []
    if head_tokens & vehicle_heads:
        patterns = [
            (r"\b(?:owns?|owned|bought|purchased|gets?|got|drives?|drove|driven|uses?|used|replaced)\b\s+(?P<cand>(?:a|an|the|my|our|your|his|her|their|its)\s+[^.:\n]{1,80})", 6),
        ]
    elif head_tokens & country_heads:
        patterns = [
            (r"\b(?:travel(?:s|ed|ing)?|visit(?:s|ed|ing)?|move(?:s|d|ing)?|live(?:s|d|ing)?|stay(?:s|ed|ing)?)\s+(?:to|in|from|at)\s+(?P<cand>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\b", 4),
        ]
    elif head_tokens & location_heads:
        patterns = [
            (r"\b(?:travel(?:s|ed|ing)?|visit(?:s|ed|ing)?|move(?:s|d|ing)?|live(?:s|d|ing)?|stay(?:s|ed|ing)?|go(?:es|ing|ne|went)?)\s+(?:to|in|from|at)\s+(?P<cand>(?:the\s+)?[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\b", 5),
        ]
    elif head_tokens & activity_heads:
        patterns = [
            (r"\b(?:trying|try|tried|taking\s+up|takes\s+up|took\s+up|pursu(?:e|es|ed|ing)|playing|play(?:s|ed)?|went|signed\s+up\s+for)\s+(?P<cand>[^.:\n]{1,50})", 4),
        ]

    candidates: list[str] = []
    seen: set[str] = set()
    for pattern, max_tokens in patterns:
        for match in re.finditer(pattern, fact_text, re.I):
            fragment = _clip_headless_slot_fragment(match.group("cand"), max_tokens=max_tokens)
            if not fragment:
                continue
            candidate = _normalize_slot_candidate_fragment(fragment, qf, head_tokens)
            if not candidate:
                continue
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
    return candidates


def _best_local_slot_candidate_from_window(
    lines: list[str],
    qf: dict,
    head_tokens: set[str],
) -> str | None:
    for line in lines:
        if TURN_QUERY_LINE_RE.match(line):
            continue
        candidates = _fact_slot_fill_candidates(line, qf, allow_loose_fallback=False)
        if not candidates:
            candidates = _headless_slot_relation_candidates(line, qf)
        for candidate in candidates:
            if _slot_candidate_adds_new_info(candidate, qf, head_tokens):
                return candidate
    return None


def _extract_requested_temporal_markers(question: str) -> list[str]:
    if not question:
        return []
    markers: list[str] = []
    seen: set[str] = set()
    patterns = (
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, question, re.I):
            marker = re.sub(r"\s+", " ", match.group(0).strip()).lower()
            if marker in seen:
                continue
            seen.add(marker)
            markers.append(marker)
    return markers


def _build_fact_slot_candidate_notes(
    supporting_facts: list[dict],
    qf: dict,
    *,
    question: str | None = None,
) -> list[str]:
    slot_plan = (qf.get("operator_plan") or {}).get("slot_query") or {}
    if not slot_plan.get("enabled"):
        return []

    head_tokens = {
        normalize_term_token(token)
        for token in (slot_plan.get("head_tokens") or [])
        if normalize_term_token(token)
    }
    primary_entity, primary_entity_tokens = _slot_primary_entity_tokens(qf)
    requested_temporal_markers = _extract_requested_temporal_markers(question or "")
    notes: list[tuple[int, tuple[int, int, int, int, int, int, int], int, int, str]] = []
    seen: set[str] = set()

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

    foreign_subject_ignore = {
        "i", "we", "he", "she", "they", "it", "last", "next", "this", "that", "these", "those",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
    }

    def _has_primary_entity(text: str) -> bool:
        if not primary_entity_tokens:
            return True
        text_lower = text.lower()
        if primary_entity and primary_entity in text_lower:
            return True
        text_tokens = {
            normalize_term_token(token)
            for token in WORD_RE.findall(text)
            if normalize_term_token(token)
        }
        return bool(primary_entity_tokens & text_tokens)

    def _has_foreign_subject_before_head(text: str) -> bool:
        if not primary_entity_tokens or not head_tokens:
            return False
        raw_words = WORD_RE.findall(text)
        norm_words = [normalize_term_token(word) for word in raw_words]
        head_len = len(head_tokens)
        head_seq = list(head_tokens)
        head_index = None
        for idx in range(max(0, len(norm_words) - head_len + 1)):
            if norm_words[idx:idx + head_len] == head_seq:
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

    semantic_head_tokens = head_tokens | {
        "vehicle", "truck", "van", "bike", "bicycle", "motorcycle", "suv", "sedan",
        "country", "nation", "city", "state", "province", "region", "location", "place", "destination", "area",
        "activity", "activities", "hobby", "hobbies", "pastime", "pastimes", "sport", "sports", "game", "games",
    }

    def _candidate_quality(candidate: str, fact_text: str) -> tuple[int, int, int, int, int, int, int]:
        candidate_tokens = [
            normalize_term_token(token)
            for token in WORD_RE.findall(candidate)
            if normalize_term_token(token)
        ]
        novel_tokens = [
            token
            for token in candidate_tokens
            if token not in head_tokens and token not in set(qf.get("words") or set()) and token not in QUERY_WORDS_TO_IGNORE
        ]
        informative_novel = [
            token for token in novel_tokens if token not in time_like_tokens and token not in generic_value_tokens
        ]
        noise_hits = sum(token in time_like_tokens or token in generic_value_tokens for token in candidate_tokens)
        date_alignment = int(bool(requested_temporal_markers) and any(marker in fact_text.lower() for marker in requested_temporal_markers))
        semantic_only = int(bool(candidate_tokens) and set(candidate_tokens) <= semantic_head_tokens)
        return (
            date_alignment,
            int(noise_hits == 0 and not semantic_only),
            len(informative_novel),
            int(any(ch.isupper() for ch in candidate)),
            int(len(candidate_tokens) >= 2),
            len(candidate_tokens),
            len(candidate),
        )

    for fact_rank, fact in enumerate(supporting_facts):
        fact_text = str(fact.get("fact") or "").strip()
        if not fact_text or not _has_primary_entity(fact_text) or _has_foreign_subject_before_head(fact_text):
            continue

        strict_candidates = _fact_slot_fill_candidates(fact_text, qf, allow_loose_fallback=False)
        relation_candidates: list[str] = []
        if not strict_candidates:
            relation_candidates = [
                candidate
                for candidate in _headless_slot_relation_candidates(fact_text, qf)
                if _slot_candidate_adds_new_info(candidate, qf, head_tokens)
            ]
        candidates = strict_candidates or relation_candidates
        if not candidates:
            continue

        for cand_rank, candidate in enumerate(candidates):
            normalized = candidate.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ep_id = (fact.get("metadata") or {}).get("episode_id", "")
            note = f"{candidate} [Episode: {ep_id}] [Fact: {fact_text}]"
            strict_rank = 0 if strict_candidates else 1
            notes.append((strict_rank, _candidate_quality(candidate, fact_text), fact_rank, cand_rank, note))
            break

    notes.sort(
        key=lambda row: (
            row[0],
            -row[1][0],
            -row[1][1],
            -row[1][2],
            -row[1][3],
            -row[1][4],
            -row[1][5],
            row[2],
            row[3],
            row[4].lower(),
        )
    )
    return [note for *_meta, note in notes[:6]]


def _line_speaker_name(line: str) -> str | None:
    match = re.match(r"^([A-Z][A-Za-z]+):", line)
    if not match:
        return None
    return match.group(1).strip().lower()


def _build_raw_slot_candidate_notes(
    injection_episode_ids: list[str],
    episode_lookup: dict[str, dict],
    qf: dict,
) -> list[str]:
    slot_plan = (qf.get("operator_plan") or {}).get("slot_query") or {}
    if not slot_plan.get("enabled"):
        return []
    content_head_tokens = set(_slot_content_head_tokens(qf))
    if not content_head_tokens:
        return []
    modifier_tokens = {
        token
        for token in (slot_plan.get("head_tokens") or [])
        if token and token not in content_head_tokens
    }
    query_words = {
        token
        for token in (qf.get("words") or set())
        if token and token not in QUERY_WORDS_TO_IGNORE
    }

    subject_speakers = {
        phrase.lower()
        for phrase in (qf.get("entity_phrases") or [])
        if isinstance(phrase, str) and phrase and " " not in phrase.strip()
    }

    episode_order = {ep_id: idx for idx, ep_id in enumerate(injection_episode_ids)}
    ranked: list[tuple[int, int, int, str, str, str, int, str]] = []
    seen: set[tuple[str, str]] = set()
    for ep_id in injection_episode_ids:
        episode = episode_lookup.get(ep_id) or {}
        raw_text = str(episode.get("raw_text") or "")
        if not raw_text:
            continue
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        for idx, line in enumerate(lines):
            match = TURN_QUERY_LINE_RE.match(line)
            if not match:
                continue
            prev_lines = [x for x in lines[max(0, idx - 2):idx] if not TURN_QUERY_LINE_RE.match(x)]
            if subject_speakers and prev_lines:
                prev_speaker = prev_lines[-1].split(":", 1)[0].strip().lower()
                if prev_speaker not in subject_speakers:
                    continue
            raw_query = match.group("query").strip()
            raw_words = WORD_RE.findall(raw_query)
            norm_words = {normalize_term_token(word) for word in raw_words if normalize_term_token(word)}
            if not (content_head_tokens & norm_words):
                continue
            window_lines = lines[max(0, idx - 2): min(len(lines), idx + 4)]
            candidate = _best_local_slot_candidate_from_window(window_lines, qf, content_head_tokens)
            if not candidate:
                candidate = _turn_query_slot_candidate(raw_query, qf, content_head_tokens)
            if not candidate:
                continue
            dedupe_key = (ep_id, candidate.lower())
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            window_lines = lines[max(0, idx - 2): min(len(lines), idx + 4)]
            local_tokens = {
                normalize_term_token(word)
                for word in WORD_RE.findall(" ".join(window_lines))
                if normalize_term_token(word)
            }
            modifier_hits = sum(1 for token in modifier_tokens if token in local_tokens)
            query_overlap = sum(1 for token in query_words if token in local_tokens)
            score = modifier_hits * 3 + query_overlap
            evidence = " / ".join(window_lines)
            ranked.append((score, modifier_hits, episode_order.get(ep_id, 999), ep_id, candidate, raw_query, len(evidence), evidence))

    if not ranked:
        return []
    ranked.sort(key=lambda row: (row[2], -row[0], -row[1], row[6], row[3], row[4]))
    if any(row[1] > 0 for row in ranked):
        ranked = [row for row in ranked if row[1] > 0]
    earliest_episode_order = min(row[2] for row in ranked)
    ranked = [row for row in ranked if row[2] == earliest_episode_order]

    notes: list[str] = []
    for _, _, _, ep_id, candidate, raw_query, _, evidence in ranked[:3]:
        notes.append(
            f"{candidate} (from [Turn query]: {raw_query}) [Episode: {ep_id}] [Local evidence: {evidence}]"
        )
    return notes


def _build_raw_list_candidate_notes(
    injection_episode_ids: list[str],
    episode_lookup: dict[str, dict],
    qf: dict,
) -> list[str]:
    list_plan = (qf.get("operator_plan") or {}).get("list_set") or {}
    if not list_plan.get("enabled") or _list_set_has_generic_head(qf):
        return []

    head_tokens = {token for token in (list_plan.get("head_tokens") or []) if token}
    query_words = {
        token
        for token in (qf.get("words") or set())
        if token and token not in QUERY_WORDS_TO_IGNORE
    }
    subject_speakers = {
        phrase.lower()
        for phrase in (qf.get("entity_phrases") or [])
        if isinstance(phrase, str) and phrase and " " not in phrase.strip()
    }

    episode_order = {ep_id: idx for idx, ep_id in enumerate(injection_episode_ids)}
    ranked: list[tuple[int, int, int, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for ep_id in injection_episode_ids:
        episode = episode_lookup.get(ep_id) or {}
        raw_text = str(episode.get("raw_text") or "")
        if not raw_text:
            continue
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        for line in lines:
            if TURN_QUERY_LINE_RE.match(line):
                continue
            speaker = _line_speaker_name(line)
            if subject_speakers and speaker and speaker not in subject_speakers:
                continue
            local_tokens = {
                normalize_term_token(word)
                for word in WORD_RE.findall(line)
                if normalize_term_token(word)
            }
            query_overlap = sum(1 for token in query_words if token in local_tokens)
            head_overlap = sum(1 for token in head_tokens if token in local_tokens)
            primary_signal = int(_fact_has_primary_list_signal(line, qf))
            if query_overlap <= 0 and primary_signal <= 0:
                continue
            dedupe_key = (ep_id, line.lower())
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            score = (query_overlap * 3) + head_overlap + primary_signal
            ranked.append((score, episode_order.get(ep_id, 999), len(line), ep_id, line))

    if not ranked:
        return []

    ranked.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
    best_by_episode: dict[str, tuple[int, int, int, str, str]] = {}
    for row in ranked:
        ep_id = row[3]
        if ep_id not in best_by_episode:
            best_by_episode[ep_id] = row
    notes: list[str] = []
    for _, _, _, ep_id, evidence in list(best_by_episode.values())[:8]:
        notes.append(f"[Episode: {ep_id}] {evidence}")
    return notes


def _fact_chain_tokens(fact_text: str, qf: dict) -> set[str]:
    tokens = set(_fact_content_tokens(fact_text, qf))
    tokens -= GENERIC_CHAIN_BRIDGE_TOKENS
    lower = fact_text.lower()
    tokens.update(code.lower() for code in PERMIT_CODE_RE.findall(fact_text))
    tokens.update(km for km in CHAINAGE_RE.findall(lower))
    tokens.update(number for number in NUMBER_RE.findall(fact_text))
    for phrase in qf.get("entity_phrases", []):
        phrase_lower = phrase.lower()
        phrase_tokens = qf.get("entity_phrase_tokens", {}).get(phrase_lower) or word_token_set(phrase)
        if (" " in phrase_lower and phrase_lower in lower) or (
            " " not in phrase_lower and phrase_tokens and phrase_tokens <= word_token_set(fact_text)
        ):
            tokens.add(phrase_lower)
    return tokens


def _fact_resolver_tokens(fact_text: str, qf: dict) -> set[str]:
    tokens = set(word_token_set(fact_text))
    lower = fact_text.lower()
    tokens.update(code.lower() for code in PERMIT_CODE_RE.findall(fact_text))
    tokens.update(km for km in CHAINAGE_RE.findall(lower))
    tokens.update(number for number in NUMBER_RE.findall(fact_text))
    return tokens


def _fact_relation_tokens(fact_text: str) -> set[str]:
    return word_token_set(fact_text) & GENERIC_CHAIN_RELATION_TOKENS


def _fact_matches_list_head(fact_text: str, qf: dict) -> bool:
    head_tokens = set((qf.get("operator_plan") or {}).get("list_set", {}).get("head_tokens") or [])
    if not head_tokens:
        return False
    return head_tokens <= _fact_resolver_tokens(fact_text, qf)


def _fact_rank_tuple(
    fact: dict,
    qf: dict,
    token_freq: dict[str, int],
    query_specificity_bonus: float = 0.0,
) -> tuple[float, float, float, float, float, float, int, str]:
    text = fact.get("fact", "")
    lower = text.lower()
    overlap = _fact_overlap_score(text, qf)
    specificity = _query_specificity_score(text, qf, token_freq, query_specificity_bonus)
    content_tokens = _fact_content_tokens(text, qf)
    rarity = 0.0
    if content_tokens:
        rarity = sum(1.0 / max(1, token_freq.get(token, 1)) for token in content_tokens) / len(content_tokens)
    concise = 0.25 if 1 <= len(content_tokens) <= 5 else 0.0
    extracted_bonus = 0.15 if not str(fact.get("id", "")).startswith("raw_") else 0.0
    greeting_penalty = -1.0 if GREETING_FACT_RE.search(text) else 0.0
    list_head_match_bonus = 0.5 if _fact_matches_list_head(text, qf) else 0.0
    location_bonus = 0.75 * _fact_location_hint_score(text, qf)
    plan_commitment_bonus = _fact_plan_commitment_score(text, qf)
    low_information_penalty = 0.0
    distinct_value_token = bool(content_tokens) and (
        qf.get("operator_plan", {}).get("slot_query", {}).get("enabled")
        or _fact_matches_list_head(text, qf)
        or bool(PERMIT_CODE_RE.search(text))
        or bool(CHAINAGE_RE.search(lower))
        or bool(NUMBER_RE.search(text))
    )
    if overlap < 1.5 and len(content_tokens) <= 1 and not distinct_value_token:
        low_information_penalty -= 1.0
        if LOW_INFORMATION_FACT_RE.search(text):
            low_information_penalty -= 1.0
    step_range_action_result_bonus = 0.0
    step_range_availability_penalty = 0.0
    if qf.get("step_range"):
        if re.search(r"\b(?:is|are) (?:currently )?available\b|\bavailable action(?:s)?\b", lower):
            step_range_availability_penalty -= 2.5
        if re.match(
            r"^(?:action\b|you\b|open\b|close\b|move\b|take\b|use\b|push\b|pull\b|drop\b|put\b|go\b|look\b|examine\b)",
            lower,
        ):
            step_range_action_result_bonus += 1.0
        if lower.startswith("you "):
            step_range_action_result_bonus += 1.0
        if any(
            phrase in lower
            for phrase in (
                "you pick up",
                "you move",
                "you opened",
                "you close",
                "you arrive",
                "was moved to",
                "contains ",
            )
        ):
            step_range_action_result_bonus += 0.75
    slot_fill_bonus = _fact_slot_fill_score(text, qf)
    local_idx = _fact_local_index(fact) or 0
    total = (
        overlap
        + specificity
        + rarity
        + concise
        + extracted_bonus
        + list_head_match_bonus
        + location_bonus
        + plan_commitment_bonus
        + greeting_penalty
        + low_information_penalty
        + step_range_action_result_bonus
        + step_range_availability_penalty
        + slot_fill_bonus
    )
    return (
        total,
        overlap,
        specificity,
        rarity,
        concise,
        extracted_bonus
        + list_head_match_bonus
        + location_bonus
        + plan_commitment_bonus
        + slot_fill_bonus
        + low_information_penalty
        + step_range_action_result_bonus
        + step_range_availability_penalty,
        local_idx,
        fact.get("id", ""),
    )


def _fact_slot_fill_score(fact_text: str, qf: dict) -> float:
    candidates = _fact_slot_fill_candidates(fact_text, qf)
    if not candidates:
        return 0.0
    return max(
        min(3.0, 1.5 + 0.5 * max(0, len(candidate.split()) - 1))
        for candidate in candidates
    )


def _parse_support_date(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    patterns = (
        "%Y-%m-%d",
        "%Y/%m/%d (%a) %H:%M",
        "%d %B, %Y",
        "%I:%M %p on %d %B, %Y",
    )
    for pattern in patterns:
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            continue
    return None


def _build_temporal_grounding_notes(
    question: str,
    supporting_facts: list[dict],
    episode_lookup: dict[str, dict],
) -> list[str]:
    qf = extract_query_features(question)
    if not (qf.get("operator_plan", {}).get("temporal_grounding", {}).get("enabled") or qf.get("query_type") == "temporal"):
        return []

    notes: list[str] = []
    seen: set[str] = set()
    support_dates: list[datetime] = []
    corpus_dates = sorted(
        date
        for date in (
            _parse_support_date(ep.get("source_date", ""))
            for ep in episode_lookup.values()
        )
        if date is not None
    )

    for fact in supporting_facts:
        text = fact.get("fact", "")
        ep_id = (fact.get("metadata") or {}).get("episode_id", "")
        ep = episode_lookup.get(ep_id, {})
        anchor = _parse_support_date(ep.get("source_date")) or _parse_support_date(fact.get("session_date"))
        if anchor is None:
            continue
        support_dates.append(anchor)
        derived_note = None
        if RELATIVE_LAST_NIGHT_RE.search(text) or RELATIVE_YESTERDAY_RE.search(text):
            derived_note = f"Relative timing: '{text.strip()}' resolves against episode date {anchor.date().isoformat()} to { (anchor - timedelta(days=1)).date().isoformat() }."
        elif RELATIVE_TOMORROW_RE.search(text):
            derived_note = f"Relative timing: '{text.strip()}' resolves against episode date {anchor.date().isoformat()} to { (anchor + timedelta(days=1)).date().isoformat() }."
        else:
            years_match = RELATIVE_FOR_YEARS_RE.search(text) or RELATIVE_AGO_YEARS_RE.search(text)
            if years_match:
                years = int(years_match.group(1))
                derived_year = anchor.year - years
                derived_note = f"Relative timing: '{text.strip()}' anchored to {anchor.date().isoformat()} implies year {derived_year}."
        if derived_note and derived_note not in seen:
            notes.append(derived_note)
            seen.add(derived_note)

    if re.search(r"\bfirst\b", question, re.I) and support_dates:
        first_date = min(support_dates)
        prev_dates = [date for date in corpus_dates if date < first_date]
        if prev_dates:
            note = (
                "First-mention window: earliest surfaced dated support is "
                f"{first_date.date().isoformat()}, with the previous dated episode on "
                f"{prev_dates[-1].date().isoformat()}."
            )
        else:
            note = f"First-mention window: earliest surfaced dated support is {first_date.date().isoformat()}."
        if note not in seen:
            notes.append(note)
            seen.add(note)
    return notes


def _list_set_item_bonus(
    fact_text: str,
    qf: dict,
    token_freq: dict[str, int],
    *,
    min_freq: int,
    bonus_weight: float,
    max_item_tokens: int,
) -> float:
    if bonus_weight <= 0 or min_freq <= 1:
        return 0.0
    tokens = _fact_content_tokens(fact_text, qf)
    if not tokens or len(tokens) > max_item_tokens:
        return 0.0
    recurring = {
        token
        for token in tokens
        if token_freq.get(token, 0) >= min_freq
    }
    if not recurring:
        return 0.0
    return bonus_weight * float(len(recurring))


def _list_set_compact_item_bonus(
    fact_text: str,
    qf: dict,
    *,
    compact_bonus: float,
    enumeration_bonus: float,
    max_item_tokens: int,
) -> float:
    tokens = _fact_content_tokens(fact_text, qf)
    if not tokens:
        return 0.0
    lower = fact_text.lower()
    score = 0.0
    if compact_bonus > 0 and len(tokens) <= max_item_tokens and LIST_SET_ITEM_VERB_RE.search(lower):
        score += compact_bonus
    if (
        enumeration_bonus > 0
        and len(tokens) <= max(2 * max_item_tokens, max_item_tokens)
        and ("," in fact_text or " and " in lower)
    ):
        score += enumeration_bonus
    return score


def _fact_entity_anchor_score(fact_text: str, qf: dict) -> float:
    lower = fact_text.lower()
    tokens = word_token_set(fact_text)
    score = 0.0
    for phrase in qf.get("entity_phrases", []):
        phrase_lower = phrase.lower()
        phrase_tokens = qf.get("entity_phrase_tokens", {}).get(phrase_lower) or word_token_set(phrase)
        if (" " in phrase_lower and phrase_lower in lower) or (
            " " not in phrase_lower and phrase_tokens and phrase_tokens <= tokens
        ):
            score += 2.0 if " " in phrase_lower else 1.0
    return score


def _query_requests_location(qf: dict) -> bool:
    words = set(qf.get("words", set()))
    return bool(qf.get("asks_where")) or bool(words & GENERIC_LOCATION_TOKENS)


def _list_set_has_generic_head(qf: dict) -> bool:
    head_tokens = set((qf.get("operator_plan") or {}).get("list_set", {}).get("head_tokens") or [])
    return bool(head_tokens) and head_tokens <= GENERIC_LIST_SET_HEAD_TOKENS


def _fact_location_hint_score(fact_text: str, qf: dict) -> float:
    if not _query_requests_location(qf):
        return 0.0
    tokens = word_token_set(fact_text)
    return float(len(tokens & GENERIC_LOCATION_TOKENS))


def _fact_list_item_keys(fact_text: str, qf: dict) -> list[str]:
    if not (qf.get("operator_plan") or {}).get("list_set", {}).get("enabled"):
        return []
    speaker_stripped = re.sub(r"^[A-Z][A-Za-z]+:\s*", "", fact_text)
    entity_phrases = {phrase.lower() for phrase in qf.get("entity_phrases", [])}
    entity_tokens = _entity_anchor_tokens(qf)
    seen: set[str] = set()
    keys: list[str] = []

    def _add_key(fragment: str) -> None:
        text = LIST_ITEM_FRAGMENT_STOP_RE.split(fragment, maxsplit=1)[0]
        text = re.split(r"[.?!;:]", text, maxsplit=1)[0]
        if not text.strip():
            return
        raw_tokens = WORD_RE.findall(text)
        cleaned: list[str] = []
        for raw in raw_tokens:
            lowered = raw.lower()
            norm = normalize_term_token(raw)
            if not norm:
                continue
            if len(norm) <= 1:
                continue
            if norm in entity_tokens or lowered in entity_phrases:
                continue
            if not cleaned and (norm in LIST_ITEM_LEADING_IGNORE_TOKENS or norm in QUERY_WORDS_TO_IGNORE):
                continue
            if norm in QUERY_WORDS_TO_IGNORE:
                continue
            if norm in qf.get("words", set()):
                continue
            if norm in GENERIC_LIST_SET_HEAD_TOKENS:
                continue
            if norm in LIST_ITEM_FRAGMENT_IGNORE_TOKENS:
                continue
            if norm in GENERIC_FACT_TOKENS:
                continue
            cleaned.append(norm)
        if not cleaned:
            return
        if _query_requests_plan_meetup(qf):
            informative = [token for token in cleaned if token not in LIST_ITEM_PLAN_GENERIC_TOKENS]
            if not informative:
                return
            cleaned = informative
        key = " ".join(cleaned[:3]).strip()
        if key and key not in seen:
            keys.append(key)
            seen.add(key)

    if (_query_requests_location(qf) or _query_requests_plan_meetup(qf)) and not GREETING_FACT_RE.search(speaker_stripped):
        for match in LIST_ITEM_ANCHOR_RE.finditer(speaker_stripped):
            candidate = " ".join(match.group(0).split()).strip(" \t\r\n:;,.!?-")
            if not candidate:
                continue
            _add_key(candidate)

    article_match = ARTICLE_ITEM_PHRASE_RE.match(speaker_stripped)
    if article_match:
        _add_key(article_match.group("phrase"))

    for match in LIST_SET_ITEM_VERB_RE.finditer(speaker_stripped):
        tail = speaker_stripped[match.end():].strip(" \t\r\n:;,-")
        if not tail:
            continue
        for fragment in LIST_ITEM_FRAGMENT_SPLIT_RE.split(tail):
            _add_key(fragment)
    return keys


def _fact_has_explicit_list_item_anchor(fact_text: str, qf: dict) -> bool:
    if _fact_matches_list_head(fact_text, qf):
        return True
    return bool(_fact_list_item_keys(fact_text, qf))


def _query_requests_plan_meetup(qf: dict) -> bool:
    words = set(qf.get("words", set()))
    if not (qf.get("operator_plan") or {}).get("list_set", {}).get("enabled"):
        return False
    return bool(words & {"plan", "plann", "future", "upcom", "schedul"})


def _fact_plan_commitment_score(fact_text: str, qf: dict) -> float:
    if not _query_requests_plan_meetup(qf):
        return 0.0
    lower = fact_text.lower().replace("`", "'")
    future_anchor = bool(FUTURE_TIME_ANCHOR_RE.search(lower))
    commitment = bool(FUTURE_COMMITMENT_RE.search(lower))
    tentative = bool(FUTURE_TENTATIVE_RE.search(lower))
    has_specific_item = bool(_fact_list_item_keys(fact_text, qf)) or _fact_location_hint_score(fact_text, qf) > 0
    score = 0.0
    if has_specific_item and future_anchor:
        score += 1.0
    if has_specific_item and commitment:
        score += 0.75
    if future_anchor and commitment:
        score += 0.5
    if tentative and not commitment:
        score -= 0.75
    if not has_specific_item and (future_anchor or commitment):
        score -= 0.5
    return score


def _fact_has_primary_list_signal(fact_text: str, qf: dict) -> bool:
    if _fact_list_item_keys(fact_text, qf):
        return True
    return _fact_plan_commitment_score(fact_text, qf) > 0


def _select_bounded_chain_seed_facts(
    facts: list[dict],
    qf: dict,
    token_freq: dict[str, int],
    query_specificity_bonus: float,
    seed_count: int,
) -> list[dict]:
    if seed_count <= 0 or not facts:
        return []
    ranked = sorted(
        facts,
        key=lambda fact: (
            -_fact_entity_anchor_score(fact.get("fact", ""), qf),
            -_fact_rank_tuple(fact, qf, token_freq, query_specificity_bonus)[1],
            -_fact_rank_tuple(fact, qf, token_freq, query_specificity_bonus)[0],
            _fact_rank_tuple(fact, qf, token_freq, query_specificity_bonus)[6],
            _fact_rank_tuple(fact, qf, token_freq, query_specificity_bonus)[7],
        ),
    )
    return ranked[:seed_count]


def _expand_bounded_chain_support_facts(
    chosen: list[dict],
    episode_fact_sets: dict[str, list[dict]],
    qf: dict,
    token_freq: dict[str, int],
    query_specificity_bonus: float,
    max_total: int,
) -> list[dict]:
    if not chosen or max_total <= 0:
        return chosen
    operator_tuning = get_tuning_section("operators")
    seed_count = int(operator_tuning.get("bounded_chain_support_fact_seed_count", 1))
    max_extra = int(operator_tuning.get("bounded_chain_support_fact_max_extra", 0))
    if max_extra <= 0:
        return chosen

    seeds = _select_bounded_chain_seed_facts(
        chosen,
        qf,
        token_freq,
        query_specificity_bonus,
        seed_count=max(1, seed_count),
    )
    if not seeds:
        return chosen

    frontier: set[str] = set()
    seen_relation_tokens: set[str] = set()
    seen_resolver_tokens: set[str] = set()
    seed_ids = {fact.get("id", "") for fact in seeds}
    for fact in seeds:
        frontier |= _fact_chain_tokens(fact.get("fact", ""), qf)
        seen_relation_tokens |= _fact_relation_tokens(fact.get("fact", ""))
        seen_resolver_tokens |= _fact_resolver_tokens(fact.get("fact", ""), qf)
    unresolved_words = set(_fact_query_words(qf)) - frontier - seen_resolver_tokens

    remaining = [
        fact
        for facts in episode_fact_sets.values()
        for fact in facts
        if fact.get("id", "") not in seed_ids
    ]
    if not remaining:
        return chosen

    extras: list[dict] = []
    used_extra_ids: set[str] = set()
    unresolved_weight = float(operator_tuning.get("bounded_chain_unresolved_overlap_weight", 2.0))
    novelty_weight = float(operator_tuning.get("bounded_chain_novelty_weight", 0.75))
    relation_novelty_weight = float(operator_tuning.get("bounded_chain_relation_novelty_weight", 1.0))
    relation_repeat_penalty = float(operator_tuning.get("bounded_chain_relation_repeat_penalty", 1.0))
    bm25_carry_weight = float(operator_tuning.get("bounded_chain_bm25_carry_weight", 0.01))
    lookahead_weight = float(operator_tuning.get("bounded_chain_lookahead_weight", 1.0))
    location_bonus_weight = float(operator_tuning.get("bounded_chain_location_bonus_weight", 0.0))

    def _candidate_rows(
        active_frontier: set[str],
        active_unresolved: set[str],
        active_relations: set[str],
        used_ids: set[str],
    ) -> list[tuple[float, dict, set[str], set[str], set[str]]]:
        rows: list[tuple[float, dict, set[str], set[str], set[str]]] = []
        for fact in remaining:
            fact_id = fact.get("id", "")
            if fact_id in used_ids:
                continue
            tokens = _fact_chain_tokens(fact.get("fact", ""), qf)
            overlap = len(active_frontier & tokens)
            if overlap <= 0:
                continue
            resolver_tokens = _fact_resolver_tokens(fact.get("fact", ""), qf)
            relation_tokens = _fact_relation_tokens(fact.get("fact", ""))
            unresolved_overlap = len(active_unresolved & resolver_tokens)
            novelty = len(tokens - active_frontier)
            relation_novelty = len(relation_tokens - active_relations)
            relation_repeat = len(relation_tokens & active_relations)
            base_rank = _fact_rank_tuple(fact, qf, token_freq, query_specificity_bonus)[0]
            location_bonus = location_bonus_weight * _fact_location_hint_score(fact.get("fact", ""), qf)
            score = (
                overlap
                + (unresolved_weight * unresolved_overlap)
                + (novelty_weight * novelty)
                + (relation_novelty_weight * relation_novelty)
                - (relation_repeat_penalty * relation_repeat)
                + location_bonus
                + (bm25_carry_weight * base_rank)
            )
            rows.append((score, fact, tokens, resolver_tokens, relation_tokens))
        rows.sort(
            key=lambda row: (
                -row[0],
                -_fact_rank_tuple(row[1], qf, token_freq, query_specificity_bonus)[1],
                row[1].get("id", ""),
            ),
        )
        return rows

    def _best_path(
        active_frontier: set[str],
        active_unresolved: set[str],
        active_relations: set[str],
        used_ids: set[str],
        depth: int,
    ) -> tuple[float, list[dict]]:
        if depth <= 0:
            return 0.0, []
        best_score = 0.0
        best_path: list[dict] = []
        for local_score, fact, tokens, resolver_tokens, relation_tokens in _candidate_rows(
            active_frontier,
            active_unresolved,
            active_relations,
            used_ids,
        ):
            next_used = set(used_ids)
            next_used.add(fact.get("id", ""))
            next_frontier = set(active_frontier) | tokens
            next_unresolved = set(active_unresolved) - tokens - resolver_tokens
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
                best_path = [fact] + future_path
        return best_score, best_path

    _, extras = _best_path(
        frontier,
        unresolved_words,
        seen_relation_tokens,
        used_extra_ids,
        max_extra,
    )

    if not extras:
        return chosen

    front_ids = [fact.get("id", "") for fact in seeds] + [fact.get("id", "") for fact in extras]
    ordered = []
    seen_ids = set()
    fact_by_id = {fact.get("id", ""): fact for fact in chosen}
    for fact in extras:
        fact_by_id[fact.get("id", "")] = fact
    for fact_id in front_ids:
        fact = fact_by_id.get(fact_id)
        if not fact or fact_id in seen_ids:
            continue
        ordered.append(fact)
        seen_ids.add(fact_id)
    for fact in chosen:
        fact_id = fact.get("id", "")
        if fact_id in seen_ids:
            continue
        ordered.append(fact)
        seen_ids.add(fact_id)
    return ordered[:max_total]


def build_bounded_chain_candidate_bundle(
    question: str,
    seed_facts: list[dict],
    candidate_facts: list[dict],
    *,
    max_candidates: int = 12,
    query_specificity_bonus: float = 0.0,
) -> dict:
    """Collect a small ambiguity bundle around a bounded-chain seed.

    Unlike `_expand_bounded_chain_support_facts`, this does not choose one best path.
    It keeps the seed facts, then appends the top same-source candidate facts that
    continue the frontier. This is intended for packet construction where the model,
    not the retriever, should resolve the final competing branch.
    """
    if not seed_facts or not candidate_facts or max_candidates <= 0:
        return {"facts": list(seed_facts[:max_candidates]), "trace": {"mode": "empty"}}

    qf = extract_query_features(question)
    token_freq: Counter[str] = Counter()
    for fact in candidate_facts:
        token_freq.update(set(_fact_content_tokens(fact.get("fact", ""), qf)))

    frontier: set[str] = set()
    seen_relation_tokens: set[str] = set()
    seen_resolver_tokens: set[str] = set()
    seed_ids = {fact.get("id", "") for fact in seed_facts}
    for fact in seed_facts:
        frontier |= _fact_chain_tokens(fact.get("fact", ""), qf)
        seen_relation_tokens |= _fact_relation_tokens(fact.get("fact", ""))
        seen_resolver_tokens |= _fact_resolver_tokens(fact.get("fact", ""), qf)
    unresolved_words = set(_fact_query_words(qf)) - frontier - seen_resolver_tokens

    operator_tuning = get_tuning_section("operators")
    unresolved_weight = float(operator_tuning.get("bounded_chain_unresolved_overlap_weight", 2.0))
    novelty_weight = float(operator_tuning.get("bounded_chain_novelty_weight", 0.75))
    location_bonus_weight = float(operator_tuning.get("bounded_chain_location_bonus_weight", 0.0))
    frontier_overlap_weight = float(
        operator_tuning.get("bounded_chain_bundle_frontier_overlap_weight", 1.0)
    )
    relation_novelty_weight = float(operator_tuning.get("bounded_chain_relation_novelty_weight", 1.0))
    relation_repeat_penalty = float(operator_tuning.get("bounded_chain_relation_repeat_penalty", 1.0))
    bm25_carry_weight = float(operator_tuning.get("bounded_chain_bm25_carry_weight", 0.01))
    relation_signature_cap = max(
        0,
        int(operator_tuning.get("bounded_chain_bundle_relation_signature_cap", 0)),
    )
    same_entity_conflict_cap = max(
        0,
        int(operator_tuning.get("bounded_chain_bundle_same_entity_conflict_cap", 0)),
    )

    rows = []
    for fact in candidate_facts:
        fact_id = fact.get("id", "")
        if not fact_id or fact_id in seed_ids:
            continue
        tokens = _fact_chain_tokens(fact.get("fact", ""), qf)
        overlap = len(frontier & tokens)
        if overlap <= 0:
            continue
        resolver_tokens = _fact_resolver_tokens(fact.get("fact", ""), qf)
        relation_tokens = _fact_relation_tokens(fact.get("fact", ""))
        unresolved_overlap = len(unresolved_words & resolver_tokens)
        novelty = len(tokens - frontier)
        relation_novelty = len(relation_tokens - seen_relation_tokens)
        relation_repeat = len(relation_tokens & seen_relation_tokens)
        base_rank = _fact_rank_tuple(fact, qf, token_freq, query_specificity_bonus)[0]
        location_bonus = location_bonus_weight * _fact_location_hint_score(fact.get("fact", ""), qf)
        score = (
            (frontier_overlap_weight * overlap)
            + (unresolved_weight * unresolved_overlap)
            + (novelty_weight * novelty)
            + (relation_novelty_weight * relation_novelty)
            - (relation_repeat_penalty * relation_repeat)
            + location_bonus
            + (bm25_carry_weight * base_rank)
        )
        rows.append((score, fact))

    rows.sort(
        key=lambda row: (
            -row[0],
            -_fact_rank_tuple(row[1], qf, token_freq, query_specificity_bonus)[1],
            row[1].get("id", ""),
        ),
    )

    def _bundle_dedup_key(fact: dict) -> str:
        return re.sub(r"\s+", " ", (fact.get("fact", "") or "").strip().lower())

    def _relation_signature(fact: dict) -> tuple[str, ...]:
        return tuple(sorted(_fact_relation_tokens(fact.get("fact", ""))))

    extras = []
    seen_texts = {
        _bundle_dedup_key(fact)
        for fact in seed_facts
        if _bundle_dedup_key(fact)
    }
    relation_signature_counts: dict[tuple[str, ...], int] = {}
    for fact in seed_facts:
        signature = _relation_signature(fact)
        if signature:
            relation_signature_counts[signature] = relation_signature_counts.get(signature, 0) + 1
    skipped_relation_signature = 0
    for _score, fact in rows:
        dedup_key = _bundle_dedup_key(fact)
        if dedup_key and dedup_key in seen_texts:
            continue
        signature = _relation_signature(fact)
        if (
            relation_signature_cap > 0
            and signature
            and relation_signature_counts.get(signature, 0) >= relation_signature_cap
        ):
            skipped_relation_signature += 1
            continue
        extras.append(fact)
        if dedup_key:
            seen_texts.add(dedup_key)
        if signature:
            relation_signature_counts[signature] = relation_signature_counts.get(signature, 0) + 1
        if len(extras) >= max(0, max_candidates - len(seed_facts)):
            break
    selected_relation_signatures = {
        _relation_signature(fact)
        for fact in list(seed_facts) + extras
        if _relation_signature(fact)
    }
    same_entity_conflicts = []
    if same_entity_conflict_cap > 0:
        for _score, fact in rows:
            fact_id = fact.get("id", "")
            if not fact_id or fact in extras or fact_id in seed_ids:
                continue
            if _fact_entity_anchor_score(fact.get("fact", ""), qf) <= 0:
                continue
            signature = _relation_signature(fact)
            if not signature or signature not in selected_relation_signatures:
                continue
            dedup_key = _bundle_dedup_key(fact)
            if dedup_key and dedup_key in seen_texts:
                continue
            same_entity_conflicts.append(fact)
            if dedup_key:
                seen_texts.add(dedup_key)
            if len(same_entity_conflicts) >= same_entity_conflict_cap:
                break
    bundled = []
    seen_ids = set()
    for fact in list(seed_facts) + same_entity_conflicts + extras:
        fact_id = fact.get("id", "")
        if not fact_id or fact_id in seen_ids:
            continue
        bundled.append(fact)
        seen_ids.add(fact_id)
        if len(bundled) >= max_candidates:
            break
    return {
        "facts": bundled,
        "trace": {
            "mode": "bounded_chain_candidate_bundle",
            "seed_fact_ids": [fact.get("id", "") for fact in seed_facts],
            "frontier": sorted(frontier),
            "unresolved_words": sorted(unresolved_words),
            "candidate_count": len(candidate_facts),
            "selected_fact_ids": [fact.get("id", "") for fact in bundled],
            "frontier_overlap_weight": frontier_overlap_weight,
            "relation_signature_cap": relation_signature_cap,
            "skipped_relation_signature": skipped_relation_signature,
            "same_entity_conflict_cap": same_entity_conflict_cap,
            "same_entity_conflict_ids": [fact.get("id", "") for fact in same_entity_conflicts],
        },
    }


def extract_local_episode_snippet(
    raw_text: str,
    supporting_fact_texts: list[str],
    query: str = "",
    snippet_chars: int = 1200,
) -> str:
    if not raw_text:
        return ""
    raw_lower = raw_text.lower()
    spans = []

    anchor_texts = list(supporting_fact_texts)
    if query:
        anchor_texts.append(query)

    for fact_text in anchor_texts:
        fact_lower = fact_text.lower().strip()
        if not fact_lower:
            continue
        positions = []
        pos = raw_lower.find(fact_lower)
        if pos >= 0:
            positions.append((pos, len(fact_text)))
        else:
            anchors = []
            anchors.extend(PERMIT_CODE_RE.findall(fact_text))
            anchors.extend(CHAINAGE_RE.findall(fact_text.lower()))
            anchors.extend(NUMBER_RE.findall(fact_text))
            anchors.extend(
                [
                    word
                    for word in WORD_RE.findall(fact_text.lower())
                    if len(word) >= 5 and word not in QUERY_WORDS_TO_IGNORE
                ]
            )
            seen = set()
            for anchor in anchors[:12]:
                anchor_lower = anchor.lower()
                search_pos = 0
                while anchor_lower and len(positions) < 4:
                    pos = raw_lower.find(anchor_lower, search_pos)
                    if pos < 0:
                        break
                    if pos not in seen:
                        positions.append((pos, len(anchor)))
                        seen.add(pos)
                    search_pos = pos + len(anchor_lower)
        for pos, anchor_len in positions[:4]:
            start = max(0, pos - snippet_chars // 2)
            end = min(len(raw_text), pos + anchor_len + snippet_chars // 2)
            spans.append((start, end))

    if not spans:
        return raw_text[:snippet_chars]

    spans.sort()
    merged = []
    cur_start, cur_end = spans[0]
    for start, end in spans[1:]:
        if start <= cur_end + 80:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))

    pieces = []
    for start, end in merged[:3]:
        pieces.append(raw_text[start:end].strip())
    return "\n...\n".join(piece for piece in pieces if piece)


def _pseudo_facts_from_episode(ep_id: str, episode: dict, qf: dict | None = None) -> list[dict]:
    raw = episode.get("raw_text", "")
    sentence_rows = []
    seen_texts = set()
    for idx, text in enumerate(episode_sentences(raw), start=1):
        text = text.strip()
        if not (15 <= len(text) <= 320):
            continue
        if text in seen_texts:
            continue
        score = _fact_overlap_score(text, qf) if qf else float(-idx)
        sentence_rows.append((score, idx, text))
        seen_texts.add(text)
    candidates = sentence_rows
    if qf:
        by_index = {idx: (score, text) for score, idx, text in sentence_rows}
        candidate_indices = set()
        for score, idx, _text in sentence_rows:
            if score <= 0:
                continue
            candidate_indices.add(idx)
            if idx - 1 in by_index:
                candidate_indices.add(idx - 1)
            if idx + 1 in by_index:
                candidate_indices.add(idx + 1)
        candidates = [
            (by_index[idx][0], idx, by_index[idx][1])
            for idx in candidate_indices
        ]
        candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    pseudo = []
    for score, idx, text in candidates:
        pseudo.append(
            {
                "id": f"raw_{ep_id}_{idx:02d}",
                "session": 0,
                "fact": text,
                "metadata": {"episode_id": ep_id},
            }
        )
        if len(pseudo) >= 24:
            break
    return pseudo


def pseudo_facts_from_episode(ep_id: str, episode: dict) -> list[dict]:
    """Public wrapper for deterministic pseudo-fact generation."""
    return _pseudo_facts_from_episode(ep_id, episode)


def pick_supporting_facts(
    question: str,
    selected_episode_ids: list[str],
    facts_by_episode: dict[str, list[dict]],
    episode_lookup: dict[str, dict] | None = None,
    fact_episode_ids: list[str] | None = None,
    max_total: int = 10,
    max_per_episode: int = 3,
    allow_pseudo_facts: bool = True,
    query_features: dict | None = None,
    local_anchor_fact_radius: int = 12,
    list_set_dedup_overlap: float = 1.0,
    bounded_chain_fact_bonus: float = 0.0,
    query_specificity_bonus: float = 0.0,
) -> list[dict]:
    qf = query_features or extract_query_features(question)
    operator_plan = qf.get("operator_plan", {})
    operator_tuning = get_tuning_section("operators")
    local_anchor_enabled = operator_plan.get("local_anchor", {}).get("enabled", False)
    step_range = qf.get("step_range")
    list_set_enabled = operator_plan.get("list_set", {}).get("enabled", False)
    bounded_chain_enabled = operator_plan.get("bounded_chain", {}).get("enabled", False)
    compositional_enabled = operator_plan.get("compositional", {}).get("enabled", False)
    generic_list_head = _list_set_has_generic_head(qf)
    bounded_chain_entity_anchor_bonus_weight = float(
        operator_tuning.get("bounded_chain_entity_anchor_bonus_weight", 0.0)
    )
    list_set_item_min_freq = int(operator_tuning.get("list_set_item_recurrence_min_freq", 2))
    list_set_item_bonus_weight = float(operator_tuning.get("list_set_item_recurrence_bonus", 0.0))
    list_set_item_max_tokens = int(operator_tuning.get("list_set_item_max_content_tokens", 4))
    list_set_compact_item_bonus_weight = float(
        operator_tuning.get("list_set_compact_item_bonus", 0.0)
    )
    list_set_enumeration_bonus_weight = float(
        operator_tuning.get("list_set_enumeration_bonus", 0.0)
    )
    fact_episode_ids = list(dict.fromkeys(fact_episode_ids or selected_episode_ids))
    selected_episode_set = set(selected_episode_ids)
    chosen = []
    ranked_by_episode: dict[str, list[dict]] = {}
    token_freq: Counter[str] = Counter()
    episode_fact_sets: dict[str, list[dict]] = {}
    for ep_id in fact_episode_ids:
        facts = list(facts_by_episode.get(ep_id, []))
        if allow_pseudo_facts and episode_lookup and ep_id in episode_lookup:
            pseudo = _pseudo_facts_from_episode(ep_id, episode_lookup[ep_id], qf=qf)
            if bounded_chain_enabled and not pseudo:
                pseudo = _pseudo_facts_from_episode(ep_id, episode_lookup[ep_id])
            if facts:
                seen_pairs = {
                    (fact.get("fact", ""), (fact.get("metadata") or {}).get("episode_id", ""))
                    for fact in facts
                }
                for pf in pseudo:
                    key = (pf.get("fact", ""), (pf.get("metadata") or {}).get("episode_id", ""))
                    if key not in seen_pairs:
                        facts.append(pf)
                        seen_pairs.add(key)
            else:
                facts = pseudo
        episode_fact_sets[ep_id] = facts
        for fact in facts:
            token_freq.update(set(_fact_content_tokens(fact.get("fact", ""), qf)))

    chain_frontier: set[str] = set()
    for ep_id in fact_episode_ids:
        facts = episode_fact_sets.get(ep_id, [])

        def _rank_key(fact: dict) -> tuple[float, float, float, float, float, float, float, int, str]:
            rank_tuple = _fact_rank_tuple(fact, qf, token_freq, query_specificity_bonus)
            chain_bonus = 0.0
            if bounded_chain_enabled and bounded_chain_fact_bonus > 0 and chain_frontier:
                chain_bonus = bounded_chain_fact_bonus * len(
                    chain_frontier & _fact_chain_tokens(fact.get("fact", ""), qf)
                )
            entity_chain_bonus = 0.0
            if bounded_chain_enabled and bounded_chain_entity_anchor_bonus_weight > 0:
                entity_chain_bonus = bounded_chain_entity_anchor_bonus_weight * _fact_entity_anchor_score(
                    fact.get("fact", ""),
                    qf,
                )
            list_bonus = 0.0
            if list_set_enabled and not generic_list_head:
                list_bonus = _list_set_item_bonus(
                    fact.get("fact", ""),
                    qf,
                    token_freq,
                    min_freq=list_set_item_min_freq,
                    bonus_weight=list_set_item_bonus_weight,
                    max_item_tokens=list_set_item_max_tokens,
                )
                list_bonus += _list_set_compact_item_bonus(
                    fact.get("fact", ""),
                    qf,
                    compact_bonus=list_set_compact_item_bonus_weight,
                    enumeration_bonus=list_set_enumeration_bonus_weight,
                    max_item_tokens=list_set_item_max_tokens,
                )
            plan_bonus = _fact_plan_commitment_score(fact.get("fact", ""), qf) if list_set_enabled else 0.0
            location_priority = _fact_location_hint_score(fact.get("fact", ""), qf) if list_set_enabled else 0.0
            explicit_item_anchor = 1.0 if list_set_enabled and _fact_has_explicit_list_item_anchor(
                fact.get("fact", ""),
                qf,
            ) else 0.0
            predicate_support = 1.0 if list_set_enabled and plan_bonus > 0 else 0.0
            total = rank_tuple[0] + chain_bonus + entity_chain_bonus + list_bonus
            return (
                -(explicit_item_anchor + predicate_support),
                -list_bonus,
                -location_priority,
                -plan_bonus,
                -total,
                -entity_chain_bonus,
                -rank_tuple[1],
                -chain_bonus,
                -rank_tuple[2],
                -rank_tuple[3],
                -rank_tuple[6],
                rank_tuple[7],
            )

        ranked = sorted(
            facts,
            key=_rank_key,
        )
        if local_anchor_enabled and selected_episode_ids and ep_id == selected_episode_ids[0]:
            if step_range:
                ranked = _rerank_step_range_facts(ranked, step_range)
            else:
                ranked = _rerank_local_anchor_facts(ranked, local_anchor_fact_radius)
        ranked_by_episode[ep_id] = ranked
        if bounded_chain_enabled and ranked:
            for fact in ranked[: min(max_per_episode, 2)]:
                chain_frontier |= _fact_chain_tokens(fact.get("fact", ""), qf)
        if not list_set_enabled and not compositional_enabled:
            chosen.extend(ranked[:max_per_episode])

    if list_set_enabled or compositional_enabled:
        primary_episode_ids = [
            ep_id
            for ep_id in selected_episode_ids
            if ep_id in ranked_by_episode
        ]
        overflow_episode_ids = [
            ep_id
            for ep_id in fact_episode_ids
            if ep_id in ranked_by_episode and ep_id not in set(primary_episode_ids)
        ]
        if list_set_enabled:
            selected_signal_episode_ids = [
                ep_id
                for ep_id in primary_episode_ids
                if any(
                    _fact_has_primary_list_signal(fact.get("fact", ""), qf)
                    for fact in ranked_by_episode.get(ep_id, [])
                )
            ]
            overflow_signal_episode_ids = [
                ep_id
                for ep_id in overflow_episode_ids
                if any(
                    _fact_has_primary_list_signal(fact.get("fact", ""), qf)
                    for fact in ranked_by_episode.get(ep_id, [])
                )
            ]
            selected_weak_episode_ids = [
                ep_id for ep_id in primary_episode_ids if ep_id not in set(selected_signal_episode_ids)
            ]
            overflow_weak_episode_ids = [
                ep_id for ep_id in overflow_episode_ids if ep_id not in set(overflow_signal_episode_ids)
            ]

            covered_item_keys: set[str] = set()
            chosen_fact_ids: set[str] = set()

            def _append_fact(fact: dict) -> bool:
                fact_id = fact.get("id", "")
                if not fact_id or fact_id in chosen_fact_ids:
                    return False
                chosen.append(fact)
                chosen_fact_ids.add(fact_id)
                covered_item_keys.update(_fact_list_item_keys(fact.get("fact", ""), qf))
                return True

            def _pick_episode_fact(ep_id: str, *, require_new_item: bool) -> bool:
                for fact in ranked_by_episode.get(ep_id, []):
                    item_keys = set(_fact_list_item_keys(fact.get("fact", ""), qf))
                    if require_new_item:
                        if not item_keys:
                            continue
                        if not (item_keys - covered_item_keys):
                            continue
                    if _append_fact(fact):
                        return True
                return False

            for episode_group in (selected_signal_episode_ids, overflow_signal_episode_ids):
                for ep_id in episode_group:
                    _pick_episode_fact(ep_id, require_new_item=True)
                    if len(chosen) >= max_total:
                        break
                if len(chosen) >= max_total:
                    break

            if len(chosen) < max_total:
                for episode_group in (selected_signal_episode_ids, overflow_signal_episode_ids):
                    if not episode_group:
                        continue
                    for round_idx in range(max_per_episode):
                        for ep_id in episode_group:
                            ranked = ranked_by_episode.get(ep_id, [])
                            if round_idx < len(ranked):
                                _append_fact(ranked[round_idx])
                                if len(chosen) >= max_total:
                                    break
                        if len(chosen) >= max_total:
                            break
                    if len(chosen) >= max_total:
                        break

            if len(chosen) < max_total:
                for episode_group in (selected_weak_episode_ids, overflow_weak_episode_ids):
                    if not episode_group:
                        continue
                    for round_idx in range(max_per_episode):
                        for ep_id in episode_group:
                            ranked = ranked_by_episode.get(ep_id, [])
                            if round_idx < len(ranked):
                                _append_fact(ranked[round_idx])
                                if len(chosen) >= max_total:
                                    break
                        if len(chosen) >= max_total:
                            break
                    if len(chosen) >= max_total:
                        break
        else:
            for episode_group in (primary_episode_ids, overflow_episode_ids):
                if not episode_group:
                    continue
                for round_idx in range(max_per_episode):
                    for ep_id in episode_group:
                        ranked = ranked_by_episode.get(ep_id, [])
                        if round_idx < len(ranked):
                            chosen.append(ranked[round_idx])
                            if len(chosen) >= max_total:
                                break
                    if len(chosen) >= max_total:
                        break
                if len(chosen) >= max_total:
                    break
    else:
        chosen.sort(
            key=lambda fact: (
                -_fact_rank_tuple(fact, qf, token_freq, query_specificity_bonus)[0],
                -_fact_rank_tuple(fact, qf, token_freq, query_specificity_bonus)[1],
                -_fact_rank_tuple(fact, qf, token_freq, query_specificity_bonus)[2],
                -_fact_rank_tuple(fact, qf, token_freq, query_specificity_bonus)[3],
                -_fact_rank_tuple(fact, qf, token_freq, query_specificity_bonus)[6],
                _fact_rank_tuple(fact, qf, token_freq, query_specificity_bonus)[7],
            ),
        )
        if bounded_chain_enabled:
            chosen = _expand_bounded_chain_support_facts(
                chosen,
                episode_fact_sets,
                qf,
                token_freq,
                query_specificity_bonus,
                max_total=max_total,
            )

    limited = chosen[:max_total]
    if step_range and limited:
        limited = sorted(
            limited,
            key=lambda fact: (
                0 if step_range_overlap(fact.get("fact", ""), step_range) else 1,
                _fact_step_anchor_value(fact, step_range),
                fact.get("id", ""),
            ),
        )[:max_total]
    if not list_set_enabled:
        return limited

    deduped = []
    seen_token_sets: list[set[str]] = []
    threshold = float(list_set_dedup_overlap)
    for fact in limited:
        tokens = {
            token.lower()
            for token in WORD_RE.findall(fact.get("fact", ""))
            if len(token) >= 3 and token.lower() not in QUERY_WORDS_TO_IGNORE
        }
        duplicate = False
        for prior in seen_token_sets:
            if not tokens or not prior:
                continue
            overlap = len(tokens & prior) / max(1, min(len(tokens), len(prior)))
            if overlap >= threshold:
                duplicate = True
                break
        if duplicate:
            continue
        deduped.append(fact)
        seen_token_sets.append(tokens)
    return deduped[:max_total]


def _fact_local_index(fact: dict) -> int | None:
    fact_id = fact.get("id", "")
    match = re.search(r"(?:_f_|_)(\d+)$", fact_id)
    if match:
        return int(match.group(1))
    return None


def _fact_step_anchor_value(fact: dict, step_range: tuple[int, int] | None = None) -> int:
    text = fact.get("fact", "")
    if step_range:
        overlap = sorted(step_range_overlap(text, step_range))
        if overlap:
            return overlap[0]
    local_idx = _fact_local_index(fact)
    if local_idx is not None:
        return local_idx
    mentions = step_mentions(text)
    if mentions:
        return mentions[0]
    return 10**9


def _rerank_local_anchor_facts(facts: list[dict], radius: int) -> list[dict]:
    if not facts or radius <= 0:
        return facts
    anchor_index = _fact_local_index(facts[0])
    if anchor_index is None:
        return facts
    original_rank = {
        fact.get("id", f"fact_{idx}"): idx
        for idx, fact in enumerate(facts)
    }

    def _proximity(fact: dict) -> float:
        local_idx = _fact_local_index(fact)
        if local_idx is None:
            return 0.0
        distance = abs(local_idx - anchor_index)
        if distance > radius:
            return 0.0
        return float(radius - distance + 1)

    return sorted(
        facts,
        key=lambda fact: (
            -_proximity(fact),
            original_rank.get(fact.get("id", ""), 10**6),
            fact.get("id", ""),
        ),
    )


def _rerank_step_range_facts(
    facts: list[dict],
    step_range: tuple[int, int],
) -> list[dict]:
    if not facts or not step_range:
        return facts
    original_rank = {
        fact.get("id", f"fact_{idx}"): idx
        for idx, fact in enumerate(facts)
    }
    return sorted(
        facts,
        key=lambda fact: (
            0 if step_range_overlap(fact.get("fact", ""), step_range) else 1,
            _fact_step_anchor_value(fact, step_range),
            original_rank.get(fact.get("id", ""), 10**6),
            fact.get("id", ""),
        ),
    )


_STEP_BLOCK_RE = re.compile(
    r"(?ims)(^\[step\s+(\d+)\]\s*$.*?)(?=^\[step\s+\d+\]\s*$|\Z)"
)
_STEP_ACTION_LINE_RE = re.compile(r"(?im)^Action:\s*([A-Za-z0-9_.-]+)(:[ \t]*([^\r\n]*))?$")
_STEP_REL_COORD_RE = re.compile(r"(\d+)\s+steps?\s+(?:to\s+the\s+)?(left|right|up|down)", re.I)


def _iter_step_blocks(raw_text: str) -> list[tuple[int, str]]:
    blocks: list[tuple[int, str]] = []
    for match in _STEP_BLOCK_RE.finditer(raw_text or ""):
        try:
            step_num = int(match.group(2))
        except Exception:
            continue
        blocks.append((step_num, match.group(1).strip()))
    return blocks


def _adjacent_episode_id(ep_id: str, delta: int = 1) -> str | None:
    match = re.search(r"^(.*_e)(\d+)\b", ep_id or "")
    if not match:
        return None
    prefix, raw_num = match.groups()
    next_num = int(raw_num) + int(delta)
    if next_num <= 0:
        return None
    return f"{prefix}{next_num:0{len(raw_num)}d}"


def _extract_companion_step_block(raw_text: str, step_num: int) -> str:
    raw = (raw_text or "").strip()
    if not raw:
        return ""
    first_marker = re.search(r"(?im)^\[step\s+\d+\]\s*$", raw)
    if first_marker and first_marker.start() > 0:
        raw = raw[: first_marker.start()].strip()
    if not raw:
        return ""
    if not (
        _STEP_ACTION_LINE_RE.search(raw)
        or "Observation:" in raw
        or "Objects on the map:" in raw
    ):
        return ""
    return f"[Step {int(step_num)}]\n{raw}"


def _resolve_target_step_blocks(
    injection_episode_ids: list[str],
    episode_lookup: dict[str, dict],
    target_steps: set[int],
) -> list[tuple[int, str, str]]:
    if not injection_episode_ids or not target_steps:
        return []

    resolved: dict[int, tuple[str, str]] = {}
    marker_episode_by_step: dict[int, str] = {}

    for ep_id in injection_episode_ids:
        ep = episode_lookup.get(ep_id) or {}
        raw = ep.get("raw_text", "")
        if not raw:
            continue
        for step_num, block in _iter_step_blocks(raw):
            if step_num not in target_steps:
                continue
            marker_episode_by_step.setdefault(step_num, ep_id)
            if (
                _STEP_ACTION_LINE_RE.search(block)
                or "Observation:" in block
                or "Objects on the map:" in block
            ):
                current = resolved.get(step_num)
                candidate = block.strip()
                if current is None or len(candidate) > len(current[1]):
                    resolved[step_num] = (ep_id, candidate)

    for step_num in sorted(target_steps):
        if step_num in resolved:
            continue
        marker_ep_id = marker_episode_by_step.get(step_num)
        if not marker_ep_id:
            continue
        marker_ep = episode_lookup.get(marker_ep_id) or {}
        next_ep_id = _adjacent_episode_id(marker_ep_id, 1)
        if not next_ep_id:
            continue
        next_ep = episode_lookup.get(next_ep_id) or {}
        if next_ep.get("source_id", "") != marker_ep.get("source_id", ""):
            continue
        if next_ep.get("source_type", "") != marker_ep.get("source_type", ""):
            continue
        block = _extract_companion_step_block(next_ep.get("raw_text", ""), step_num)
        if block:
            resolved[step_num] = (next_ep_id, block)

    return [
        (step_num, ep_id, block)
        for step_num, (ep_id, block) in sorted(resolved.items())
    ]


def _extract_step_action_name(block: str) -> str:
    action_match = _STEP_ACTION_LINE_RE.search(block or "")
    if not action_match:
        return ""
    return (action_match.group(1) or "").strip().lower()


def _parse_step_object_positions(block: str) -> dict[str, tuple[int, int]]:
    if not block or "Objects on the map:" not in block:
        return {}
    _, object_section = block.split("Objects on the map:", 1)
    positions: dict[str, tuple[int, int]] = {}
    for raw_line in object_section.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        first = _STEP_REL_COORD_RE.search(line)
        if not first:
            continue
        name = line[: first.start()].strip()
        if not name:
            continue
        dx = 0
        dy = 0
        for steps_raw, direction in _STEP_REL_COORD_RE.findall(line):
            steps = int(steps_raw)
            direction = direction.lower()
            if direction == "left":
                dx -= steps
            elif direction == "right":
                dx += steps
            elif direction == "up":
                dy -= steps
            elif direction == "down":
                dy += steps
        positions[name] = (dx, dy)
    return positions


def _extract_step_anchor_snippet(raw_text: str, target_steps: set[int], radius: int = 1) -> str:
    if not raw_text or not target_steps:
        return ""
    blocks = _iter_step_blocks(raw_text)
    if not blocks:
        return ""

    expanded_steps = {
        candidate
        for step_num in target_steps
        for candidate in range(max(0, int(step_num) - radius), int(step_num) + radius + 1)
    }
    selected = [block for step_num, block in blocks if step_num in expanded_steps]
    if not selected:
        return ""
    return "\n".join(selected[: max(2, len(target_steps) * (radius * 2 + 1))]).strip()


def _extract_step_call_argument_notes(raw_text: str, target_steps: set[int]) -> list[str]:
    if not raw_text or not target_steps:
        return []

    notes: list[str] = []
    seen: set[tuple[int, str]] = set()
    for step_num, block in _iter_step_blocks(raw_text):
        if step_num not in target_steps:
            continue
        action_match = _STEP_ACTION_LINE_RE.search(block)
        if not action_match:
            continue
        action_name = (action_match.group(1) or "").strip()
        if not action_name:
            continue
        if action_match.group(2) is None:
            continue
        payload = (action_match.group(3) or "").strip()
        key = (step_num, action_name)
        if key in seen:
            continue
        if not payload:
            notes.append(
                f"Step {step_num}: action `{action_name}` has no argument payload after the colon."
            )
            seen.add(key)
            continue
        if payload in {"{}", "[]"}:
            notes.append(
                f"Step {step_num}: action `{action_name}` has an explicit empty argument payload `{payload}`."
            )
            seen.add(key)
    return notes


def _build_step_trajectory_notes(
    question: str,
    injection_episode_ids: list[str],
    episode_lookup: dict[str, dict],
    qf: dict,
) -> list[str]:
    step_range = qf.get("step_range")
    if not step_range or not injection_episode_ids:
        return []

    start_step, end_step = step_range
    range_steps = set(range(start_step, end_step + 1))
    resolved_blocks = _resolve_target_step_blocks(
        injection_episode_ids,
        episode_lookup,
        range_steps,
    )
    blocks = [(step_num, block) for step_num, _ep_id, block in resolved_blocks]
    if len(blocks) < 2:
        return []

    notes: list[str] = []
    action_vec = {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0),
    }

    actions = [(s, _extract_step_action_name(block)) for s, block in blocks]
    distinct_actions = {name for _s, name in actions if name}
    repeated_action = distinct_actions.pop() if len(distinct_actions) == 1 else ""
    if repeated_action:
        notes.append(f"Steps {start_step}-{end_step} all use action `{repeated_action}`.")

    move = action_vec.get(repeated_action)
    positions_by_step = {s: _parse_step_object_positions(block) for s, block in blocks}
    if not move:
        return notes

    candidate_scores: list[tuple[int, str]] = []
    all_names = set()
    for positions in positions_by_step.values():
        all_names.update(positions.keys())
    min_presence = max(2, len(blocks) - 1)
    for name in all_names:
        present_steps = [s for s, _block in blocks if name in positions_by_step[s]]
        if len(present_steps) < min_presence:
            continue
        coords = [positions_by_step[s][name] for s in present_steps]
        if len(set(coords)) != 1:
            continue
        dx, dy = coords[0]
        if (dx, dy) != move:
            continue
        score = len(present_steps)
        if present_steps == [s for s, _block in blocks[:-1]]:
            score += 1
        candidate_scores.append((score, name))

    if not candidate_scores:
        return notes

    candidate_scores.sort(key=lambda row: (-row[0], row[1]))
    pushed_name = candidate_scores[0][1]
    direction_word = "downward" if repeated_action == "down" else (
        "upward" if repeated_action == "up" else (
            "leftward" if repeated_action == "left" else "rightward"
        )
    )
    notes.append(
        f"Across steps {start_step}-{end_step}, {pushed_name} stays directly ahead of the agent ({repeated_action}), which indicates the agent is pushing that object {direction_word}."
    )

    available_steps = [step for step, _block in blocks if step in positions_by_step]
    first_step = min(available_steps) if available_steps else start_step
    last_step = max(available_steps) if available_steps else end_step
    start_positions = positions_by_step.get(first_step, {})
    end_positions = positions_by_step.get(last_step, {})
    axis = 1 if repeated_action in {"up", "down"} else 0
    best_blocker: tuple[int, int, str, tuple[int, int], tuple[int, int]] | None = None
    for name, start_pos in start_positions.items():
        if name == pushed_name or name not in end_positions:
            continue
        end_pos = end_positions[name]
        start_axis = start_pos[axis]
        end_axis = end_pos[axis]
        moved_away = (
            end_axis < start_axis if repeated_action == "down"
            else end_axis > start_axis if repeated_action == "up"
            else end_axis > start_axis if repeated_action == "left"
            else end_axis < start_axis
        )
        if not moved_away:
            continue
        lateral_axis = 0 if axis == 1 else 1
        row = (abs(start_axis), abs(start_pos[lateral_axis]), name, start_pos, end_pos)
        if best_blocker is None or row < best_blocker:
            best_blocker = row
    if best_blocker is not None:
        _distance, _lateral, blocker_name, start_pos, end_pos = best_blocker
        notes.append(
            f"Over the same span, {blocker_name} moves from {abs(start_pos[axis])} step(s) {'down' if start_pos[axis] > 0 else 'up' if start_pos[axis] < 0 else 'on the same row as'} the agent to {abs(end_pos[axis])} step(s) {'down' if end_pos[axis] > 0 else 'up' if end_pos[axis] < 0 else 'on the same row as'} the agent, so the maneuver is clearing a path past that blocker."
        )

    return notes


def _build_raw_anchor_snippets(
    question: str,
    injection_episode_ids: list[str],
    episode_lookup: dict[str, dict],
    qf: dict,
    local_anchor_window_chars: int | None,
) -> list[str]:
    snippets: list[str] = []
    step_numbers = set(qf.get("step_numbers", set()) or set())
    step_range = qf.get("step_range")
    target_steps = set(step_numbers)
    if step_range:
        start_step, end_step = step_range
        target_steps |= set(range(start_step, end_step + 1))
    local_anchor_enabled = bool(qf.get("operator_plan", {}).get("local_anchor", {}).get("enabled"))
    if not injection_episode_ids or not (target_steps or local_anchor_enabled):
        return snippets

    if target_steps:
        resolved_blocks = _resolve_target_step_blocks(
            injection_episode_ids,
            episode_lookup,
            target_steps,
        )
        if resolved_blocks:
            snippet_limit = 1 if len(target_steps) == 1 else min(4, max(2, len(target_steps)))
            for step_num, ep_id, block in resolved_blocks[:snippet_limit]:
                anchor_text = _extract_step_anchor_snippet(block, {step_num}, radius=0) or block.strip()
                if not anchor_text:
                    continue
                snippets.append(f"[Episode: {ep_id}]\n{anchor_text}")
            if snippets:
                return snippets

    snippet_chars = int(local_anchor_window_chars or 1200)
    for ep_id in injection_episode_ids[:2]:
        ep = episode_lookup.get(ep_id) or {}
        raw = ep.get("raw_text", "")
        if not raw:
            continue
        anchor_text = ""
        if target_steps:
            anchor_text = _extract_step_anchor_snippet(raw, target_steps, radius=1)
        if not anchor_text and local_anchor_enabled:
            anchor_text = extract_local_episode_snippet(
                raw,
                [],
                query=question,
                snippet_chars=snippet_chars,
            ).strip()
        if not anchor_text:
            continue
        snippets.append(f"[Episode: {ep_id}]\n{anchor_text}")
    return snippets


def _build_step_call_notes(
    injection_episode_ids: list[str],
    episode_lookup: dict[str, dict],
    qf: dict,
) -> list[str]:
    step_numbers = set(qf.get("step_numbers", set()) or set())
    step_range = qf.get("step_range")
    target_steps = set(step_numbers)
    if step_range:
        start_step, end_step = step_range
        target_steps |= set(range(start_step, end_step + 1))
    if not injection_episode_ids or not target_steps:
        return []

    notes: list[str] = []
    seen: set[str] = set()
    for ep_id in injection_episode_ids[:2]:
        ep = episode_lookup.get(ep_id) or {}
        raw = ep.get("raw_text", "")
        for note in _extract_step_call_argument_notes(raw, target_steps):
            if note in seen:
                continue
            notes.append(note)
            seen.add(note)
    return notes


def build_context_from_selected_episodes(
    question: str,
    selected_episode_ids: list[str],
    episode_lookup: dict[str, dict],
    facts_by_episode: dict[str, list[dict]],
    fact_episode_ids: list[str] | None = None,
    support_episode_ids: list[str] | None = None,
    budget: int = 8000,
    max_total_facts: int = 10,
    max_facts_per_episode: int = 3,
    snippet_mode: bool = False,
    snippet_chars: int = 1200,
    allow_pseudo_facts: bool = True,
    query_features: dict | None = None,
    local_anchor_window_chars: int | None = None,
    local_anchor_fact_radius: int = 12,
    list_set_dedup_overlap: float = 1.0,
    bounded_chain_fact_bonus: float = 0.0,
    query_specificity_bonus: float = 0.0,
    inject_support_fact_episodes: bool = False,
    max_injected_support_fact_episodes: int = 8,
) -> tuple[str, list[str], list[str]]:
    qf = query_features or extract_query_features(question)
    operator_plan = qf.get("operator_plan", {})
    supporting_facts = pick_supporting_facts(
        question,
        selected_episode_ids,
        facts_by_episode,
        episode_lookup=episode_lookup,
        fact_episode_ids=fact_episode_ids,
        max_total=max_total_facts,
        max_per_episode=max_facts_per_episode,
        allow_pseudo_facts=allow_pseudo_facts,
        query_features=qf,
        local_anchor_fact_radius=local_anchor_fact_radius,
        list_set_dedup_overlap=list_set_dedup_overlap,
        bounded_chain_fact_bonus=bounded_chain_fact_bonus,
        query_specificity_bonus=query_specificity_bonus,
    )

    parts = []
    selected_fact_ids = []
    fact_texts_by_episode = defaultdict(list)
    for fact in supporting_facts:
        ep_id = (fact.get("metadata") or {}).get("episode_id", "")
        if ep_id:
            fact_texts_by_episode[ep_id].append(fact.get("fact", ""))

    list_set_self_grounded = False
    if (
        operator_plan.get("list_set", {}).get("enabled", False)
        and _list_set_has_generic_head(qf)
        and selected_episode_ids
    ):
        anchor_coverage = {
            ep_id: False
            for ep_id in selected_episode_ids
            if ep_id in fact_texts_by_episode
        }
        for ep_id, texts in fact_texts_by_episode.items():
            if ep_id not in anchor_coverage:
                continue
            anchor_coverage[ep_id] = any(
                _fact_has_explicit_list_item_anchor(text, qf)
                for text in texts
            )
        list_set_self_grounded = bool(anchor_coverage) and all(anchor_coverage.values())

    injection_episode_ids = list(dict.fromkeys(selected_episode_ids))
    if inject_support_fact_episodes:
        seen_injected = set(injection_episode_ids)
        extra_episode_ids: list[str] = []
        for ep_id in support_episode_ids or []:
            if ep_id in seen_injected:
                continue
            if ep_id not in episode_lookup:
                continue
            extra_episode_ids.append(ep_id)
            seen_injected.add(ep_id)
        for ep_id in fact_episode_ids or []:
            if ep_id not in fact_texts_by_episode or ep_id in seen_injected:
                continue
            extra_episode_ids.append(ep_id)
            seen_injected.add(ep_id)
        if max_injected_support_fact_episodes >= 0:
            extra_episode_ids = extra_episode_ids[:max_injected_support_fact_episodes]
        injection_episode_ids.extend(extra_episode_ids)
    if list_set_self_grounded:
        injection_episode_ids = []

    anchor_snippets = _build_raw_anchor_snippets(
        question,
        injection_episode_ids,
        episode_lookup,
        qf,
        local_anchor_window_chars,
    )
    step_call_notes = _build_step_call_notes(
        injection_episode_ids,
        episode_lookup,
        qf,
    )
    trajectory_notes = _build_step_trajectory_notes(
        question,
        injection_episode_ids,
        episode_lookup,
        qf,
    )
    if anchor_snippets:
        parts.append("LOCAL ANCHOR SNIPPET:")
        for idx, snippet in enumerate(anchor_snippets, 1):
            parts.append(f"[A{idx}] {snippet}")
        parts.append("")
    if step_call_notes:
        parts.append("STEP CALL NOTES:")
        for idx, note in enumerate(step_call_notes, 1):
            parts.append(f"[C{idx}] {note}")
        parts.append("")
    if trajectory_notes:
        parts.append("STEP TRAJECTORY NOTES:")
        for idx, note in enumerate(trajectory_notes, 1):
            parts.append(f"[R{idx}] {note}")
        parts.append("")

    parts.append("RETRIEVED FACTS:")
    for idx, fact in enumerate(supporting_facts, 1):
        ep_id = (fact.get("metadata") or {}).get("episode_id", "")
        parts.append(f"[{idx}] (S{fact.get('session', '?')}) {fact.get('fact', '')} [Episode: {ep_id}]")
        selected_fact_ids.append(fact.get("id", ""))

    temporal_notes = _build_temporal_grounding_notes(question, supporting_facts, episode_lookup)
    if temporal_notes:
        parts.append("")
        parts.append("TEMPORAL NOTES:")
        for idx, note in enumerate(temporal_notes, 1):
            parts.append(f"[T{idx}] {note}")

    raw_slot_candidate_notes = _build_fact_slot_candidate_notes(
        supporting_facts,
        qf,
        question=question,
    )
    raw_slot_candidate_notes.extend(_build_raw_slot_candidate_notes(
        injection_episode_ids,
        episode_lookup,
        qf,
    ))
    if raw_slot_candidate_notes:
        parts.append("")
        parts.append("RAW SLOT CANDIDATES:")
        for idx, note in enumerate(raw_slot_candidate_notes, 1):
            parts.append(f"[Q{idx}] {note}")

    list_evidence_episode_ids = list(dict.fromkeys(
        list(selected_episode_ids)
        + list(injection_episode_ids)
        + list(support_episode_ids or [])
        + list(fact_episode_ids or [])
    ))
    if (
        operator_plan.get("list_set", {}).get("enabled")
        and not operator_plan.get("commonality", {}).get("enabled")
        and list_evidence_episode_ids
    ):
        list_evidence_source_ids = {
            str((episode_lookup.get(ep_id) or {}).get("source_id") or "")
            for ep_id in list_evidence_episode_ids
            if episode_lookup.get(ep_id)
        }
        if list_evidence_source_ids:
            list_evidence_episode_ids = list(dict.fromkeys(
                list_evidence_episode_ids
                + [
                    ep_id
                    for ep_id, ep in episode_lookup.items()
                    if str((ep or {}).get("source_id") or "") in list_evidence_source_ids
                ]
            ))
    raw_list_candidate_notes = []
    if not operator_plan.get("commonality", {}).get("enabled"):
        raw_list_candidate_notes = _build_raw_list_candidate_notes(
            list_evidence_episode_ids,
            episode_lookup,
            qf,
        )
    if raw_list_candidate_notes:
        parts.append("")
        parts.append("RAW LIST EVIDENCE:")
        for idx, note in enumerate(raw_list_candidate_notes, 1):
            parts.append(f"[L{idx}] {note}")

    chars_used = 0
    injected = []
    adaptive_snippet_mode = snippet_mode
    total_raw_chars = sum(len(episode_lookup.get(ep_id, {}).get("raw_text", "")) for ep_id in injection_episode_ids)
    if not adaptive_snippet_mode and len(injection_episode_ids) > 1 and total_raw_chars > budget:
        adaptive_snippet_mode = True
    adaptive_snippet_chars = snippet_chars
    if adaptive_snippet_mode and injection_episode_ids:
        adaptive_snippet_chars = min(
            snippet_chars,
            max(450, budget // max(len(injection_episode_ids), 1) - 250),
        )
        if qf.get("step_numbers"):
            adaptive_snippet_chars = max(adaptive_snippet_chars, 1400)
        if qf.get("operator_plan", {}).get("local_anchor", {}).get("enabled") and local_anchor_window_chars:
            adaptive_snippet_chars = max(adaptive_snippet_chars, int(local_anchor_window_chars))
    if injection_episode_ids:
        parts.append("")
        parts.append("--- SOURCE EPISODE RAW TEXT ---")
    for idx, ep_id in enumerate(injection_episode_ids):
        ep = episode_lookup.get(ep_id)
        if not ep:
            continue
        header = (
            f"\n[Episode: {ep_id}] "
            f"topic={ep.get('topic_key', '')} "
            f"state={ep.get('state_label', '')} "
            f"currentness={ep.get('currentness', '')}"
        )
        raw = ep.get("raw_text", "")
        if adaptive_snippet_mode:
            keep_full = idx == 0 and len(raw) <= int(budget * 0.78) and len(selected_episode_ids) > 1
            if not keep_full:
                raw = extract_local_episode_snippet(
                    raw,
                    fact_texts_by_episode.get(ep_id, []),
                    query=question,
                    snippet_chars=adaptive_snippet_chars,
                )
        block = header + "\n" + raw
        if chars_used + len(block) > budget:
            remaining = budget - chars_used
            if remaining <= 300:
                break
            block = header + "\n" + raw[: max(0, remaining - len(header) - 1)]
        parts.append(block)
        chars_used += len(block)
        injected.append(ep_id)

    return "\n".join(parts), injected, selected_fact_ids


def build_context_from_retrieved_facts(
    retrieved_facts: list[dict],
    episode_lookup: dict[str, dict],
    fact_lookup: dict[str, dict],
    budget: int = 8000,
    snippet_chars: int = 1200,
    question: str | None = None,
    query_features: dict | None = None,
) -> tuple[str, list[str]]:
    """Build whole-episode packet from retrieved facts via episode linkage only."""
    original_rank = {
        (item.get("fact_id", item.get("id", "")) or f"fact_{idx}"): idx
        for idx, item in enumerate(retrieved_facts)
    }

    def _ordered_fact_items() -> list[dict]:
        resolved_items = []
        for idx, item in enumerate(retrieved_facts):
            fid = item.get("fact_id", item.get("id", ""))
            fact_obj = fact_lookup.get(fid, item)
            episode_ids = fact_episode_ids(fact_obj)
            ep_id = episode_ids[0] if episode_ids else ""
            resolved_items.append(
                {
                    "item": item,
                    "fact": fact_obj,
                    "fact_id": fid,
                    "episode_id": ep_id,
                    "episode_ids": episode_ids,
                    "original_rank": idx,
                    "local_idx": _fact_local_index(fact_obj) or 10**9,
                }
            )
        episode_order = {}
        for row in resolved_items:
            ep_id = row["episode_id"]
            if ep_id and ep_id not in episode_order:
                episode_order[ep_id] = row["original_rank"]
        resolved_items.sort(
            key=lambda row: (
                episode_order.get(row["episode_id"], row["original_rank"]),
                row["local_idx"],
                row["original_rank"],
            )
        )
        return resolved_items

    ordered_items = _ordered_fact_items()
    fact_lines = []
    for i, row in enumerate(ordered_items):
        item = row["item"]
        fid = row["fact_id"]
        fact_obj = row["fact"]
        session = fact_obj.get("session", "?")
        fact_text = fact_obj.get("fact", item.get("fact", ""))
        episode_ids = row.get("episode_ids", [])
        if len(episode_ids) == 1:
            label = f" [Episode: {episode_ids[0]}]"
        elif len(episode_ids) > 1:
            label = f" [Episodes: {', '.join(episode_ids)}]"
        else:
            label = ""
        fact_lines.append(f"[{i+1}] (S{session}) {fact_text}{label}")

    ep_anchors = Counter()
    ep_best_rank = {}
    for rank, row in enumerate(ordered_items):
        for ep_id in row.get("episode_ids", []):
            if ep_id and ep_id in episode_lookup:
                ep_anchors[ep_id] += 1
                if ep_id not in ep_best_rank:
                    ep_best_rank[ep_id] = rank

    ranked_eps = sorted(
        ep_anchors.keys(),
        key=lambda ep_id: (-ep_anchors[ep_id], ep_best_rank.get(ep_id, 999)),
    )
    fact_texts_by_episode = defaultdict(list)
    for row in ordered_items:
        fact_text = row["fact"].get("fact", row["item"].get("fact", ""))
        if fact_text:
            for ep_id in row.get("episode_ids", []):
                if ep_id:
                    fact_texts_by_episode[ep_id].append(fact_text)

    parts = ["RETRIEVED FACTS:"]
    parts.extend(fact_lines)
    qf = query_features or (extract_query_features(question) if question else {})
    support_facts = [row["fact"] for row in ordered_items if row.get("fact")]
    raw_slot_candidate_notes = _build_fact_slot_candidate_notes(support_facts, qf, question=question)
    raw_slot_candidate_notes.extend(_build_raw_slot_candidate_notes(ranked_eps, episode_lookup, qf))
    if raw_slot_candidate_notes:
        parts.append("")
        parts.append("RAW SLOT CANDIDATES:")
        for idx, note in enumerate(raw_slot_candidate_notes, 1):
            parts.append(f"[Q{idx}] {note}")

    list_evidence_episode_ids = list(ranked_eps)
    if (
        qf.get("operator_plan", {}).get("list_set", {}).get("enabled")
        and not qf.get("operator_plan", {}).get("commonality", {}).get("enabled")
        and list_evidence_episode_ids
    ):
        list_evidence_source_ids = {
            str((episode_lookup.get(ep_id) or {}).get("source_id") or "")
            for ep_id in list_evidence_episode_ids
            if episode_lookup.get(ep_id)
        }
        if list_evidence_source_ids:
            list_evidence_episode_ids = list(dict.fromkeys(
                list_evidence_episode_ids
                + [
                    ep_id
                    for ep_id, ep in episode_lookup.items()
                    if str((ep or {}).get("source_id") or "") in list_evidence_source_ids
                ]
            ))
    raw_list_candidate_notes = []
    if not qf.get("operator_plan", {}).get("commonality", {}).get("enabled"):
        raw_list_candidate_notes = _build_raw_list_candidate_notes(list_evidence_episode_ids, episode_lookup, qf)
    if raw_list_candidate_notes:
        parts.append("")
        parts.append("RAW LIST EVIDENCE:")
        for idx, note in enumerate(raw_list_candidate_notes, 1):
            parts.append(f"[L{idx}] {note}")
    injected = []
    chars_used = 0
    if ranked_eps:
        parts.append("")
        parts.append("--- SOURCE EPISODE RAW TEXT ---")
    adaptive_snippet_chars = snippet_chars
    if ranked_eps:
        adaptive_snippet_chars = min(
            snippet_chars,
            max(450, budget // max(len(ranked_eps), 1) - 250),
        )
    for ep_id in ranked_eps:
        raw = episode_lookup[ep_id].get("raw_text", "")
        if len(ranked_eps) > 1 or len(raw) > budget * 0.6:
            raw = extract_local_episode_snippet(
                raw,
                fact_texts_by_episode.get(ep_id, []),
                snippet_chars=adaptive_snippet_chars,
            )
        if chars_used + len(raw) > budget:
            remaining = budget - chars_used
            if remaining > 200:
                raw = raw[:remaining]
            else:
                break
        parts.append(f"\n[Episode: {ep_id}]")
        parts.append(raw)
        chars_used += len(raw)
        injected.append(ep_id)
    return "\n".join(parts), injected


def fact_episode_ids(fact_obj: dict) -> list[str]:
    metadata = fact_obj.get("metadata") or {}
    single = metadata.get("episode_id")
    if isinstance(single, str) and single:
        return [single]
    values = []
    for episode_id in (metadata.get("episode_ids") or []):
        if isinstance(episode_id, str) and episode_id and episode_id not in values:
            values.append(episode_id)
    return values
