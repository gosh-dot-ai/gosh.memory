#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Source adapters and deterministic atom/block helpers."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from .block_segmenter import segment_document_blocks

_SEPARATOR_LINES = {"---", "___", "***"}
_DOC_ID_ONLY_RE = re.compile(r"^[A-Z0-9][A-Z0-9_-]{3,}$")


def _is_semantic_document_block(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False

    lines = [line.strip() for line in stripped.splitlines()]
    meaningful = [line for line in lines if line and line not in _SEPARATOR_LINES]
    if not meaningful:
        return False

    if len(meaningful) == 1 and _DOC_ID_ONLY_RE.fullmatch(meaningful[0]):
        return False

    return True


def segment_document_text(text: str, source_id: str) -> tuple[list[dict], list]:
    """Segment one document text into deterministic block dicts."""
    blocks = [b for b in segment_document_blocks(text) if _is_semantic_document_block(b.text)]
    block_dicts = []
    for i, b in enumerate(blocks):
        block_dicts.append({
            "block_id": f"{source_id}_b{i:03d}",
            "order": b.order,
            "family": b.family,
            "section_path": b.section_path or "",
            "text": b.text,
            "text_preview": b.text[:240],
            "char_len": len(b.text),
            "raw_span": list(b.raw_span),
        })
    return block_dicts, blocks


def segment_document_path(path: str | Path, source_id: str | None = None) -> tuple[str, list[dict], list]:
    """Load document and segment into deterministic blocks."""
    p = Path(path)
    text = p.read_text()
    sid = source_id or p.stem
    block_dicts, blocks = segment_document_text(text, sid)
    return text, block_dicts, blocks


def build_block_manifest(documents: dict[str, str | Path]) -> dict[str, list[dict]]:
    """Build deterministic block manifest from {source_id: path} mapping."""
    manifest = {}
    for source_id, path in documents.items():
        _text, block_dicts, _blocks = segment_document_path(path, source_id=source_id)
        manifest[source_id] = block_dicts
    return manifest


def audit_episode_partition(corpus: dict, block_manifest: dict[str, list[dict]]) -> dict:
    """Audit exhaustive partition over deterministic blocks."""
    per_doc = {}
    total_missing = 0
    total_duplicates = 0
    mega_episodes = []

    for doc in corpus.get("documents", []):
        doc_id = doc.get("doc_id")
        expected = {b["block_id"] for b in block_manifest.get(doc_id, [])}
        assigned = []
        episode_block_ids = {}
        for ep in doc.get("episodes", []):
            bids = list((ep.get("provenance") or {}).get("block_ids", []))
            episode_block_ids[ep["episode_id"]] = bids
            assigned.extend(bids)
            if len(ep.get("raw_text", "")) > 12000:
                mega_episodes.append({
                    "doc_id": doc_id,
                    "episode_id": ep["episode_id"],
                    "raw_chars": len(ep.get("raw_text", "")),
                    "topic_key": ep.get("topic_key", ""),
                    "state_label": ep.get("state_label", ""),
                })

        assigned_counter = Counter(assigned)
        missing = sorted(expected - set(assigned))
        duplicates = sorted([bid for bid, cnt in assigned_counter.items() if cnt > 1])
        unexpected = sorted(set(assigned) - expected)
        total_missing += len(missing)
        total_duplicates += len(duplicates)
        per_doc[doc_id] = {
            "expected_blocks": len(expected),
            "assigned_blocks": len(assigned),
            "unique_assigned_blocks": len(set(assigned)),
            "episode_count": len(doc.get("episodes", [])),
            "missing_blocks": missing,
            "duplicate_blocks": duplicates,
            "unexpected_blocks": unexpected,
            "episode_block_ids": episode_block_ids,
        }

    return {
        "total_missing_blocks": total_missing,
        "total_duplicate_blocks": total_duplicates,
        "mega_episode_count": len(mega_episodes),
        "mega_episodes": mega_episodes,
        "per_doc": per_doc,
    }


def sessions_to_episodes(
    sessions: list[str],
    conv_id: str,
    session_dates: list[str] | None = None,
    topic_prefix: str = "session",
) -> dict:
    """Explicit conversation adapter: one session maps to one episode."""
    session_dates = session_dates or []
    documents = [{
        "doc_id": conv_id,
        "episodes": [
            {
                "episode_id": f"{conv_id}_e{i+1:02d}",
                "source_type": "conversation",
                "source_id": conv_id,
                "source_date": session_dates[i] if i < len(session_dates) else "",
                "topic_key": f"{topic_prefix}_{i+1}",
                "state_label": "session",
                "currentness": "unknown",
                "raw_text": text,
                "provenance": {"raw_span": [0, len(text)]},
            }
            for i, text in enumerate(sessions)
        ],
    }]
    return {"documents": documents}
