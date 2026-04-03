#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Block segmenter for conversation sessions.

Copyright 2026 (c) Mitja Goroshevsky and GOSH Technology Ltd.
License: MIT.

Splits a conversation session into typed blocks (PROSE, LIST, TABLE)
for per-block extraction. Pure deterministic — no LLM calls.

Speaker detection policy:
  SOURCE-FIRST: if caller provides explicit speakers, use them and
  skip inference entirely. This is the preferred path for structured
  inputs (JSON conversations, dataset metadata, DOM roles).

  PRECISION-FIRST: when inferring speakers from flat text, false
  positive split is much worse than missed split. Ambiguous input
  stays unsplit.
"""

import re
from collections import Counter
from dataclasses import dataclass


@dataclass
class Block:
    family: str  # PROSE | LIST | TABLE | KV | CODE | UNKNOWN
    text: str  # cleaned content for LLM, speaker prefix stripped
    order: int  # position within session
    raw_span: tuple[int, int]  # slice into original session_text
    speaker: str | None  # detected speaker name, e.g. "user", "Caroline"
    speaker_role: str | None  # "user" | "assistant" | "system" | None
    lead_in: str | None  # preceding prose before embedded structure
    section_path: str | None = None  # document: "Chapter 3 > Requirements"


_KNOWN_ROLES = {"user", "assistant", "system", "human", "ai"}

_SPEAKER_RE = re.compile(r"^\s*([^:\n]{1,80}?)\s*:\s+\S", re.MULTILINE)

_LIST_LINE = re.compile(r"^\s*(\d+[.)]\s+|[-*•]\s+)")
_TABLE_LINE = re.compile(r"^\s*\|.+\|")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)", re.MULTILINE)
_KV_LINE = re.compile(r"^[A-Za-z][A-Za-z0-9_ ]{0,40}:\s+\S")
_CODE_FENCE = re.compile(r"^```")


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def _role_for(speaker_norm: str) -> str | None:
    """Map normalized speaker to role."""
    if speaker_norm in ("user", "human"):
        return "user"
    if speaker_norm in ("assistant", "ai"):
        return "assistant"
    if speaker_norm == "system":
        return "system"
    return None


# ── Speaker detection ──


def detect_speakers(session_text: str) -> dict[str, str]:
    """Infer speakers from flat text. Precision-first fallback.

    Returns mapping of normalized (lowercased) -> original case speaker name.

    This is a FALLBACK for flat text without source metadata. When the
    caller has structured speaker information (JSON conversations, dataset
    metadata), pass speakers explicitly to segment_conversation_blocks()
    instead of relying on this inference.

    Rules (precision-first — false positive split > missed split):
      1. Known roles (user/assistant/system/human/ai) always accepted.
      2. If any known roles found, non-role candidates are NEVER promoted.
         Everything between known-role turns is turn content.
      3. If no known roles found, accept non-role candidate ONLY if:
         a. freq >= 2
         b. does not appear self-consecutively in prefix sequence
         No content-length, no language-specific, no rescue heuristics.
      4. If signal insufficient, return empty / only known roles.
    """
    candidates: dict[str, str] = {}
    counts: Counter[str] = Counter()

    for m in _SPEAKER_RE.finditer(session_text):
        raw = m.group(1).strip()
        norm = raw.lower()
        counts[norm] += 1
        if norm not in candidates:
            candidates[norm] = raw

    # Phase 1: known roles — always accepted
    result = {}
    for norm, original in candidates.items():
        if norm in _KNOWN_ROLES:
            result[norm] = original

    # Phase 2: non-role candidates
    # If known roles present → never promote non-role (safe default).
    # If no known roles → freq >= 2 + no self-chaining only.
    if not result:
        lines = session_text.split("\n")
        non_empty_lines = [l for l in lines if l.strip()]
        prefix_sequence = []
        for li, line in enumerate(lines):
            m = _SPEAKER_RE.match(line)
            if m:
                raw = m.group(1).strip()
                norm = raw.lower()
                if norm in candidates:
                    prefix_sequence.append((li, norm))

        # Structural guard: compute median turn size (in lines) between
        # consecutive prefix lines. If median turn size <= 1 line (all
        # single-line turns), it's ambiguous (KV or terse chat) → don't split.
        turn_sizes = []
        for i, (li, _) in enumerate(prefix_sequence):
            if i + 1 < len(prefix_sequence):
                next_li = prefix_sequence[i + 1][0]
                turn_sizes.append(next_li - li)
            else:
                turn_sizes.append(len(lines) - li)
        median_turn = sorted(turn_sizes)[len(turn_sizes) // 2] if turn_sizes else 1

        if median_turn <= 1:
            pass  # all single-line turns — ambiguous (KV or terse chat). Don't split.
        else:
            for norm, original in candidates.items():
                if counts[norm] < 2:
                    continue
                positions = [i for i, (_, n) in enumerate(prefix_sequence) if n == norm]
                self_consecutive = any(
                    positions[k + 1] == positions[k] + 1
                    for k in range(len(positions) - 1)
                )
                if not self_consecutive:
                    result[norm] = original

    return result


# ── Turn splitting ──


def _find_turns(session_text: str, speakers: dict[str, str]) -> list[tuple[str, str | None, int, int]]:
    """Split session into turns by speaker prefix.

    Returns list of (text_after_prefix, speaker_original, start_offset, end_offset).
    Offsets are into session_text.
    """
    if not speakers:
        return [(session_text, None, 0, len(session_text))]

    escaped = [re.escape(orig) for orig in speakers.values()]
    for norm in speakers:
        escaped.append(re.escape(norm))
    escaped = sorted(set(escaped), key=len, reverse=True)
    prefix_re = re.compile(
        r"^[ \t]*(" + "|".join(escaped) + r")[ \t]*:[ \t]*",
        re.MULTILINE | re.IGNORECASE,
    )

    splits = list(prefix_re.finditer(session_text))
    if not splits:
        return [(session_text, None, 0, len(session_text))]

    turns = []
    for i, m in enumerate(splits):
        speaker_raw = m.group(1).strip()
        speaker_norm = speaker_raw.lower()
        speaker_orig = speakers.get(speaker_norm, speaker_raw)
        content_start = m.end()
        if i + 1 < len(splits):
            content_end = splits[i + 1].start()
        else:
            content_end = len(session_text)
        text = session_text[content_start:content_end].rstrip()
        turns.append((text, speaker_orig, m.start(), content_end))

    if splits and splits[0].start() > 0:
        pre = session_text[: splits[0].start()].rstrip()
        if pre.strip():
            turns.insert(0, (pre, None, 0, splits[0].start()))

    return turns


# ── Block classification within a turn ──


def _classify_lines(lines: list[str]) -> list[tuple[str, int, int]]:
    """Classify consecutive lines into (family, start_idx, end_idx) spans."""
    n = len(lines)
    if n == 0:
        return []

    tags = []
    for line in lines:
        if _TABLE_LINE.match(line):
            tags.append("TABLE")
        elif _LIST_LINE.match(line):
            tags.append("LIST")
        else:
            tags.append("PROSE")

    spans = []
    i = 0
    while i < n:
        family = tags[i]
        j = i + 1
        while j < n and tags[j] == family:
            j += 1
        if family == "LIST" and (j - i) < 2:
            family = "PROSE"
        if family == "TABLE" and (j - i) < 2:
            family = "PROSE"
        spans.append((family, i, j))
        i = j

    merged = []
    for family, start, end in spans:
        if merged and merged[-1][0] == "PROSE" and family == "PROSE":
            merged[-1] = ("PROSE", merged[-1][1], end)
        else:
            merged.append((family, start, end))

    return merged


def _sub_segment_prose(text: str) -> list[str]:
    """Split long prose into sub-blocks at paragraph or sentence boundaries."""
    if _estimate_tokens(text) <= 300:
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)
    if len(paragraphs) > 1:
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for p in paragraphs:
            p_tokens = _estimate_tokens(p)
            if current and current_tokens + p_tokens > 300:
                chunks.append("\n\n".join(current))
                current = [p]
                current_tokens = p_tokens
            else:
                current.append(p)
                current_tokens += p_tokens
        if current:
            chunks.append("\n\n".join(current))
        if len(chunks) > 1:
            return chunks

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) <= 1:
        return [text]

    chunks = []
    current = []
    current_tokens = 0
    for s in sentences:
        s_tokens = _estimate_tokens(s)
        if current and current_tokens + s_tokens > 300:
            chunks.append(" ".join(current))
            current = [s]
            current_tokens = s_tokens
        else:
            current.append(s)
            current_tokens += s_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks if len(chunks) > 1 else [text]


# ── Main segmenter ──


def segment_conversation_blocks(
    session_text: str,
    speakers: dict[str, str] | None = None,
) -> list[Block]:
    """Segment a conversation session into typed blocks.

    Args:
        session_text: raw session text.
        speakers: optional explicit speaker mapping (normalized -> original).
            SOURCE-FIRST: if provided, these speakers are used directly
            and detect_speakers() is NOT called. This is the preferred
            path for structured inputs.
            If None, falls back to detect_speakers() inference.

    Returns list of Block with exclusive line ownership and
    raw_span provenance into session_text.
    """
    if speakers is None:
        speakers = detect_speakers(session_text)
    turns = _find_turns(session_text, speakers)

    blocks: list[Block] = []
    order = 0

    for turn_text, speaker_orig, turn_start, turn_end in turns:
        speaker_norm = speaker_orig.lower() if speaker_orig else None
        speaker_role = _role_for(speaker_norm) if speaker_norm else None

        lines = turn_text.split("\n")
        line_spans = _classify_lines(lines)

        line_offsets = []
        pos = 0
        for line in lines:
            line_offsets.append(pos)
            pos += len(line) + 1

        prev_prose_text = None

        for family, line_start, line_end in line_spans:
            block_text_start = line_offsets[line_start]
            if line_end < len(lines):
                block_text_end = line_offsets[line_end]
            else:
                block_text_end = len(turn_text)

            raw_content = turn_text[block_text_start:block_text_end].rstrip()

            if turn_text:
                content_abs_start = session_text.find(turn_text[:min(50, len(turn_text))], turn_start)
                if content_abs_start < 0:
                    content_abs_start = turn_start
            else:
                content_abs_start = turn_start

            if block_text_start == 0:
                abs_start = turn_start
            else:
                abs_start = content_abs_start + block_text_start
            abs_end = content_abs_start + block_text_end

            lead_in = None
            if family in ("LIST", "TABLE") and prev_prose_text is not None:
                lead_in = prev_prose_text
                prev_prose_text = None

            content_abs_offset = content_abs_start + block_text_start

            if family == "PROSE":
                sub_blocks = _sub_segment_prose(raw_content)
                if len(sub_blocks) > 1:
                    sub_offset = 0
                    for sb_text in sub_blocks:
                        sb_pos = raw_content.find(sb_text, sub_offset)
                        if sb_pos < 0:
                            sb_pos = sub_offset
                        sb_abs_start = content_abs_offset + sb_pos
                        sb_abs_end = sb_abs_start + len(sb_text)

                        blocks.append(Block(
                            family="PROSE",
                            text=sb_text.strip(),
                            order=order,
                            raw_span=(sb_abs_start, sb_abs_end),
                            speaker=speaker_orig,
                            speaker_role=speaker_role,
                            lead_in=None,
                        ))
                        order += 1
                        sub_offset = sb_pos + len(sb_text)
                    prev_prose_text = sub_blocks[-1].strip()
                else:
                    blocks.append(Block(
                        family="PROSE",
                        text=raw_content.strip(),
                        order=order,
                        raw_span=(abs_start, abs_end),
                        speaker=speaker_orig,
                        speaker_role=speaker_role,
                        lead_in=None,
                    ))
                    order += 1
                    prev_prose_text = raw_content.strip()
            else:
                blocks.append(Block(
                    family=family,
                    text=raw_content.strip(),
                    order=order,
                    raw_span=(abs_start, abs_end),
                    speaker=speaker_orig,
                    speaker_role=speaker_role,
                    lead_in=lead_in,
                ))
                order += 1
                prev_prose_text = None

    return blocks


# ── Document segmenter ──


def _classify_doc_lines(lines: list[str]) -> list[tuple[str, int, int]]:
    """Classify document lines into block family spans.

    Handles CODE (fenced blocks), KV, LIST, TABLE, PROSE.
    Returns (family, start_line, end_line) spans.
    """
    n = len(lines)
    if n == 0:
        return []

    tags = []
    in_code = False
    for line in lines:
        if _CODE_FENCE.match(line):
            if in_code:
                tags.append("CODE")  # closing fence
                in_code = False
            else:
                tags.append("CODE")  # opening fence
                in_code = True
        elif in_code:
            tags.append("CODE")
        elif _TABLE_LINE.match(line):
            tags.append("TABLE")
        elif _LIST_LINE.match(line):
            tags.append("LIST")
        elif _KV_LINE.match(line) and ":" in line:
            tags.append("KV")
        else:
            tags.append("PROSE")

    # Group consecutive same-family runs
    spans = []
    i = 0
    while i < n:
        family = tags[i]
        j = i + 1
        while j < n and tags[j] == family:
            j += 1
        # Single list/table/kv item → PROSE
        if family == "LIST" and (j - i) < 2:
            family = "PROSE"
        if family == "TABLE" and (j - i) < 2:
            family = "PROSE"
        if family == "KV" and (j - i) < 2:
            family = "PROSE"
        spans.append((family, i, j))
        i = j

    # Merge adjacent PROSE
    merged = []
    for family, start, end in spans:
        if merged and merged[-1][0] == "PROSE" and family == "PROSE":
            merged[-1] = ("PROSE", merged[-1][1], end)
        else:
            merged.append((family, start, end))

    return merged


def segment_document_blocks(document_text: str) -> list[Block]:
    """Segment a markdown document into typed blocks with section hierarchy.

    Splits by heading boundaries, then classifies content within each section
    into PROSE, LIST, TABLE, KV, CODE blocks. Heading lines are structural
    markers — they are NOT included in block text.

    Document blocks: speaker=None, speaker_role=None, section_path set.
    """
    lines = document_text.split("\n")

    # Build line offsets for raw_span computation
    line_offsets = []
    pos = 0
    for line in lines:
        line_offsets.append(pos)
        pos += len(line) + 1  # +1 for \n

    # Parse headings → build section spans
    heading_stack: list[tuple[int, str]] = []  # (level, title)
    sections: list[tuple[str, int, int]] = []  # (section_path, start_line, end_line)
    section_starts: list[tuple[int, int, str]] = []  # (line_idx, level, title)

    for li, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            section_starts.append((li, level, title))

    # Build sections from heading positions
    for i, (li, level, title) in enumerate(section_starts):
        # Pop stack to current level
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, title))
        path = " > ".join(t for _, t in heading_stack)

        # Section content: from line after heading to next heading (or end)
        content_start = li + 1
        if i + 1 < len(section_starts):
            content_end = section_starts[i + 1][0]
        else:
            content_end = len(lines)
        sections.append((path, content_start, content_end))

    # If no headings, entire document is one section
    if not sections:
        sections.append(("", 0, len(lines)))

    # Also handle content before first heading
    if section_starts and section_starts[0][0] > 0:
        sections.insert(0, ("", 0, section_starts[0][0]))

    blocks: list[Block] = []
    order = 0

    for section_path, sec_start, sec_end in sections:
        sec_lines = lines[sec_start:sec_end]
        if not any(l.strip() for l in sec_lines):
            continue

        spans = _classify_doc_lines(sec_lines)
        prev_prose_text = None

        for family, span_start, span_end in spans:
            # Compute absolute line indices
            abs_line_start = sec_start + span_start
            abs_line_end = sec_start + span_end

            # raw_span in document_text
            raw_start = line_offsets[abs_line_start]
            if abs_line_end < len(lines):
                raw_end = line_offsets[abs_line_end]
            else:
                raw_end = len(document_text)

            raw_content = document_text[raw_start:raw_end].rstrip()

            lead_in = None
            if family in ("LIST", "TABLE", "KV", "CODE") and prev_prose_text is not None:
                lead_in = prev_prose_text
                prev_prose_text = None

            if family == "PROSE":
                sub_blocks = _sub_segment_prose(raw_content)
                if len(sub_blocks) > 1:
                    sub_offset = 0
                    for sb_text in sub_blocks:
                        sb_pos = raw_content.find(sb_text, sub_offset)
                        if sb_pos < 0:
                            sb_pos = sub_offset
                        sb_abs_start = raw_start + sb_pos
                        sb_abs_end = sb_abs_start + len(sb_text)
                        blocks.append(Block(
                            family="PROSE",
                            text=sb_text.strip(),
                            order=order,
                            raw_span=(sb_abs_start, sb_abs_end),
                            speaker=None,
                            speaker_role=None,
                            lead_in=None,
                            section_path=section_path or None,
                        ))
                        order += 1
                        sub_offset = sb_pos + len(sb_text)
                    prev_prose_text = sub_blocks[-1].strip()
                else:
                    blocks.append(Block(
                        family="PROSE",
                        text=raw_content.strip(),
                        order=order,
                        raw_span=(raw_start, raw_end),
                        speaker=None,
                        speaker_role=None,
                        lead_in=None,
                        section_path=section_path or None,
                    ))
                    order += 1
                    prev_prose_text = raw_content.strip()
            else:
                blocks.append(Block(
                    family=family,
                    text=raw_content.strip(),
                    order=order,
                    raw_span=(raw_start, raw_end),
                    speaker=None,
                    speaker_role=None,
                    lead_in=lead_in,
                    section_path=section_path or None,
                ))
                order += 1
                prev_prose_text = None

    return blocks
