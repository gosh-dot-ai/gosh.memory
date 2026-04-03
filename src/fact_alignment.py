#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic fact-to-raw alignment for facts-as-selectors MVP."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable

from .block_segmenter import segment_conversation_blocks, segment_document_blocks
from .common import STOP_WORDS, normalize_term_token

_TRUE_VALUES = {"1", "true", "yes", "on"}
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_LINE_RE = re.compile(r"[^\n]+")


def facts_as_selectors_enabled() -> bool:
    return os.getenv("GOSH_FACT_SELECTORS", "").strip().lower() in _TRUE_VALUES


def align_facts_batch(
    facts: list[dict],
    raw_fields: dict[str, str],
    *,
    episode_id: str,
    source_kind: str,
    speakers: dict[str, str] | None = None,
) -> None:
    """Attach conservative raw-span provenance to extractive facts in-place."""
    candidate_segments = _build_candidate_segments(
        raw_fields=raw_fields,
        source_kind=source_kind,
        speakers=speakers,
    )
    for fact in facts:
        _align_fact_in_place(fact, candidate_segments, episode_id=episode_id)


def _build_candidate_segments(
    *,
    raw_fields: dict[str, str],
    source_kind: str,
    speakers: dict[str, str] | None,
) -> list[dict]:
    segments: list[dict] = []
    for source_field, raw_text in raw_fields.items():
        if not isinstance(raw_text, str) or not raw_text.strip():
            continue
        if source_field == "raw_text":
            if source_kind == "document":
                blocks = segment_document_blocks(raw_text)
            else:
                blocks = segment_conversation_blocks(raw_text, speakers=speakers)
            for block in blocks:
                start, end = block.raw_span
                segments.append(
                    {
                        "source_field": source_field,
                        "start": int(start),
                        "end": int(end),
                        "text": raw_text[start:end],
                        "order": block.order,
                    }
                )
        else:
            segments.append(
                {
                    "source_field": source_field,
                    "start": 0,
                    "end": len(raw_text),
                    "text": raw_text,
                    "order": 0,
                }
            )
    return segments


def _align_fact_in_place(fact: dict, candidate_segments: list[dict], *, episode_id: str) -> None:
    fact_text = str(fact.get("fact") or "").strip()
    if not fact_text or not candidate_segments:
        fact["fact_class"] = "unaligned"
        fact["alignment_failed"] = True
        fact.pop("support_spans", None)
        return

    support_span = _find_support_span(
        fact_text=fact_text,
        entities=_entity_tokens(fact.get("entities") or []),
        candidate_segments=candidate_segments,
        episode_id=episode_id,
    )
    if support_span is None:
        fact["fact_class"] = "unaligned"
        fact["alignment_failed"] = True
        fact.pop("support_spans", None)
        return

    fact["fact_class"] = "extractive"
    fact["support_spans"] = [support_span]
    fact.pop("alignment_failed", None)


def _find_support_span(
    *,
    fact_text: str,
    entities: set[str],
    candidate_segments: list[dict],
    episode_id: str,
) -> dict | None:
    exact = _exact_match_span(fact_text, candidate_segments, episode_id=episode_id)
    if exact is not None:
        return exact

    fact_tokens = _content_tokens(fact_text)
    if not fact_tokens and not entities:
        return None

    best_seg: dict | None = None
    best_score = 0.0
    best_overlap = 0
    best_entity_overlap = 0
    fact_token_set = set(fact_tokens)

    for seg in candidate_segments:
        seg_tokens = set(_content_tokens(seg["text"]))
        if not seg_tokens and not entities:
            continue
        token_overlap = len(fact_token_set & seg_tokens)
        entity_overlap = len(entities & seg_tokens) if entities else 0
        coverage = token_overlap / max(len(fact_token_set), 1)
        score = coverage + (0.35 * entity_overlap)
        if score > best_score or (
            score == best_score
            and (token_overlap, entity_overlap, -seg["order"])
            > (best_overlap, best_entity_overlap, -(best_seg or {"order": 10**9})["order"])
        ):
            best_seg = seg
            best_score = score
            best_overlap = token_overlap
            best_entity_overlap = entity_overlap

    if best_seg is None:
        return None
    if best_score < 0.45:
        return None
    if best_overlap < 2 and best_entity_overlap == 0 and best_score < 0.9:
        return None

    parent_span = {
        "episode_id": episode_id,
        "source_field": best_seg["source_field"],
        "start": best_seg["start"],
        "end": best_seg["end"],
        "role": "primary",
        "method": "block_overlap",
        "confidence": round(min(best_score, 1.0), 3),
    }
    refined = _refine_support_span(
        fact_text=fact_text,
        entities=entities,
        candidate_segment=best_seg,
        episode_id=episode_id,
    )
    if refined is not None:
        return refined
    return parent_span


def _exact_match_span(fact_text: str, candidate_segments: list[dict], *, episode_id: str) -> dict | None:
    lowered_fact = fact_text.lower()
    for seg in candidate_segments:
        raw_text = seg["text"]
        idx = raw_text.lower().find(lowered_fact)
        if idx < 0:
            continue
        return {
            "episode_id": episode_id,
            "source_field": seg["source_field"],
            "start": seg["start"] + idx,
            "end": seg["start"] + idx + len(fact_text),
            "role": "primary",
            "method": "exact",
            "confidence": 1.0,
        }
    return None


def _refine_support_span(
    *,
    fact_text: str,
    entities: set[str],
    candidate_segment: dict,
    episode_id: str,
) -> dict | None:
    subsegments = _segment_line_candidates(candidate_segment)
    if len(subsegments) <= 1:
        return None

    exact = _exact_match_span(fact_text, subsegments, episode_id=episode_id)
    if exact is not None:
        exact["method"] = "line_exact"
        return exact

    fact_tokens = _content_tokens(fact_text)
    fact_token_set = set(fact_tokens)
    if not fact_token_set and not entities:
        return None

    best_seg: dict | None = None
    best_score = 0.0
    best_overlap = 0
    best_entity_overlap = 0
    for seg in subsegments:
        seg_tokens = set(_content_tokens(seg["text"]))
        if not seg_tokens and not entities:
            continue
        token_overlap = len(fact_token_set & seg_tokens)
        entity_overlap = len(entities & seg_tokens) if entities else 0
        coverage = token_overlap / max(len(fact_token_set), 1)
        score = coverage + (0.35 * entity_overlap)
        if score > best_score or (
            score == best_score
            and (token_overlap, entity_overlap, -seg["order"])
            > (
                best_overlap,
                best_entity_overlap,
                -(best_seg or {"order": 10**9})["order"],
            )
        ):
            best_seg = seg
            best_score = score
            best_overlap = token_overlap
            best_entity_overlap = entity_overlap

    if best_seg is None:
        return None
    if best_score < 0.45:
        return None
    if best_overlap < 2 and best_entity_overlap == 0 and best_score < 0.9:
        return None
    if (best_seg["end"] - best_seg["start"]) >= (candidate_segment["end"] - candidate_segment["start"]):
        return None

    return {
        "episode_id": episode_id,
        "source_field": candidate_segment["source_field"],
        "start": best_seg["start"],
        "end": best_seg["end"],
        "role": "primary",
        "method": "line_overlap",
        "confidence": round(min(best_score, 1.0), 3),
    }


def _segment_line_candidates(segment: dict) -> list[dict]:
    raw_text = str(segment.get("text") or "")
    if not raw_text.strip():
        return []

    candidates: list[dict] = []
    for order, match in enumerate(_LINE_RE.finditer(raw_text)):
        text = match.group(0).strip()
        if not text:
            continue
        start = int(segment["start"]) + match.start()
        end = int(segment["start"]) + match.end()
        candidates.append(
            {
                "source_field": segment["source_field"],
                "start": start,
                "end": end,
                "text": text,
                "order": order,
            }
        )
    if candidates:
        return candidates
    return [segment]


def _entity_tokens(values: Iterable[str]) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        for token in _content_tokens(str(value or "")):
            tokens.add(token)
    return tokens


def _content_tokens(text: str) -> list[str]:
    out: list[str] = []
    for match in _WORD_RE.finditer((text or "").lower()):
        token = normalize_term_token(match.group(0))
        if not token or len(token) <= 2 or token in STOP_WORDS:
            continue
        out.append(token)
    return out


def selector_surface_text(
    fact: dict,
    episode_lookup: dict[str, dict],
    *,
    fact_lookup: dict[str, dict] | None = None,
    max_spans: int = 2,
    max_chars_per_span: int = 220,
) -> str:
    """Return a compact raw-backed selector surface for retrieval."""
    if not facts_as_selectors_enabled() or not isinstance(fact, dict):
        return ""
    seen_refs: set[tuple[str, str, int, int]] = set()
    parts: list[str] = []
    for span in iter_support_spans(fact, fact_lookup=fact_lookup):
        ep_id = str(span.get("episode_id") or (fact.get("metadata") or {}).get("episode_id") or "")
        if not ep_id:
            continue
        source_field = str(span.get("source_field") or "raw_text")
        try:
            start = int(span.get("start", 0))
            end = int(span.get("end", 0))
        except Exception:
            continue
        ref = (ep_id, source_field, start, end)
        if ref in seen_refs:
            continue
        seen_refs.add(ref)
        raw = _lookup_span_source_text(ep_id, source_field, episode_lookup)
        if not raw:
            continue
        start = max(0, min(start, len(raw)))
        end = max(start, min(end, len(raw)))
        if end <= start:
            continue
        snippet = raw[start:end].strip().replace("\n", " ")
        if not snippet:
            continue
        if len(snippet) > max_chars_per_span:
            snippet = snippet[: max_chars_per_span - 3].rstrip() + "..."
        parts.append(snippet)
        if len(parts) >= max_spans:
            break
    return " ".join(parts)


def iter_support_spans(
    fact: dict,
    *,
    fact_lookup: dict[str, dict] | None = None,
    _seen_fact_ids: set[str] | None = None,
):
    support_spans = fact.get("support_spans") or []
    if isinstance(support_spans, list):
        for span in support_spans:
            if isinstance(span, dict):
                yield span
        if support_spans:
            return

    support_fact_ids = fact.get("support_fact_ids") or []
    if not fact_lookup or not isinstance(support_fact_ids, list):
        return
    seen_fact_ids = set(_seen_fact_ids or set())
    fact_id = str(fact.get("id") or "")
    if fact_id:
        seen_fact_ids.add(fact_id)
    for support_fact_id in support_fact_ids:
        support_fact_id = str(support_fact_id or "")
        if not support_fact_id or support_fact_id in seen_fact_ids:
            continue
        child = fact_lookup.get(support_fact_id)
        if not isinstance(child, dict):
            continue
        child_seen = set(seen_fact_ids)
        child_seen.add(support_fact_id)
        yield from iter_support_spans(child, fact_lookup=fact_lookup, _seen_fact_ids=child_seen)


def _lookup_span_source_text(ep_id: str, source_field: str, episode_lookup: dict[str, dict]) -> str:
    episode = episode_lookup.get(ep_id) or {}
    if source_field == "raw_text":
        return str(episode.get("raw_text") or "")
    value = episode.get(source_field)
    if isinstance(value, str):
        return value
    return ""
