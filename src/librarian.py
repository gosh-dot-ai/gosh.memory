#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GOSH Memory — Librarian extraction pipeline.

Three-tier fact indexing:
  1. Atomic extraction (conversation sessions -> granular facts + temporal links)
  2. Consolidation (per-session granular facts -> merged statements)
  3. Cross-session (per-entity facts across sessions -> comprehensive statements)

Extracted from sprint-17c/s17c_full_scale.py.
Format-selective extraction from sprint-23d/run_23d.py.
"""

import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path as _Path

# ── Prompt loader ──

_PROMPT_DIR = _Path(__file__).parent / "prompts" / "extraction"


def _load_extraction_prompt(name: str) -> str:
    """Load extraction prompt from .md file."""
    path = _PROMPT_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"Extraction prompt '{name}' not found at {path}. "
            f"Expected src/prompts/extraction/{name}.md"
        )
    return path.read_text(encoding="utf-8")


# ── Prompts (loaded from src/prompts/extraction/*.md) ──

EXTRACTION_PROMPT = _load_extraction_prompt("legacy")
EXTRACTION_PROMPT_CONVERSATION = _load_extraction_prompt("conversation")
EXTRACTION_PROMPT_AGENT_TRACE = _load_extraction_prompt("agent_trace")
EXTRACTION_PROMPT_DOCUMENT = _load_extraction_prompt("document")
EXTRACTION_PROMPT_FACT_LIST = _load_extraction_prompt("fact_list")
EXTRACTION_PROMPT_NARRATIVE = _load_extraction_prompt("narrative")

CONSOLIDATION_PROMPT = """You are consolidating atomic facts into fewer, more comprehensive statements.

Rules:
1. Merge facts about the same topic, person, event into ONE complete statement.
2. Preserve ALL exact values: names, dates, numbers, places. Never lose information.
3. Each consolidated fact must be SELF-CONTAINED.
4. Preserve distinct named targets, places, events, and items. Do NOT collapse unique targets into a vague summary.
5. If a fact is unique and doesn't relate to others, keep it as-is.
6. Target: ~40-60% fewer facts than input. But NEVER lose information.
7. {temporal_rule}

Input: granular facts from {source_desc}.
Output JSON:
{{
  "facts": [
    {{
      "id": "cf_01",
      "fact": "Consolidated statement with ALL details merged.",
      "entities": ["Entity1", "Entity2"],
      "tags": ["tag1"],
      "source_ids": ["f_01", "f_03"],
      "depends_on": []
    }}
  ]
}}"""

CROSS_SESSION_PROMPT = """You are consolidating facts about the same entity across different time periods.

Input: atomic facts about "{entity}" from multiple sessions of a conversation.

Your job: merge related facts into comprehensive statements that connect
information from different sessions. Each merged fact should be self-contained.

Rules:
1. CONNECT facts that refer to the same event, relationship, or topic across sessions.
2. PRESERVE all exact values: names, dates, numbers, places.
3. PRESERVE temporal information: when things happened, in what order.
4. If facts contradict, keep BOTH with timestamps.
5. Preserve distinct targets, places, events, and items as separate supported facts. Do NOT merge multiple targets into one vague summary.
6. KEEP exact named venues, named events, establishments, and destinations from source facts. Do NOT replace them with generic activity labels.
7. If facts are unrelated despite sharing an entity, keep them SEPARATE.
8. Each output fact = self-contained.

Output JSON:
{{
  "facts": [
    {{
      "id": "xf_01",
      "fact": "Comprehensive statement connecting information from multiple sessions.",
      "entities": ["Entity1"],
      "source_ids": ["af_conv-42_s03_015"],
      "sessions": [3, 15],
      "tags": ["tag1"]
    }}
  ]
}}"""


# ── Helpers ──


def _normalize_lineage_refs(fact: dict, source_facts: list[dict], *, derive_sessions: bool = False) -> None:
    """Canonicalize source_ids/sessions against the caller-provided source facts.

    LLMs often emit shorthand source_ids like ``f_01`` even when upstream code has
    already scoped them to session-prefixed IDs like ``s3_f_01``. Normalize any
    unambiguous suffix matches back to the canonical source IDs so downstream
    visibility and lineage logic operate on stable references.
    """
    if not isinstance(fact, dict) or not source_facts:
        return

    source_by_id = {}
    for sf in source_facts:
        sid = sf.get("id")
        if isinstance(sid, str) and sid:
            source_by_id[sid] = sf

    raw_source_ids = fact.get("source_ids")
    if isinstance(raw_source_ids, list):
        canonical = []
        seen = set()
        for raw_sid in raw_source_ids:
            if not isinstance(raw_sid, str) or not raw_sid:
                continue
            resolved = raw_sid if raw_sid in source_by_id else None
            if resolved is None:
                matches = [sid for sid in source_by_id if sid.endswith(raw_sid)]
                if len(matches) == 1:
                    resolved = matches[0]
            resolved = resolved or raw_sid
            if resolved not in seen:
                canonical.append(resolved)
                seen.add(resolved)
        fact["source_ids"] = canonical

    if derive_sessions and isinstance(fact.get("source_ids"), list):
        sessions = sorted({
            sf.get("session")
            for sid in fact["source_ids"]
            for sf in [source_by_id.get(sid)]
            if isinstance(sf, dict) and isinstance(sf.get("session"), int)
        })
        if sessions:
            fact["sessions"] = sessions


def _preserve_uncovered_session_facts(consolidated_facts: list[dict], session_facts: list[dict]) -> list[dict]:
    """Append source facts that consolidation failed to cover.

    Consolidation is allowed to merge, but not to drop unique source facts.
    Any session fact not referenced by at least one consolidated fact is
    preserved as a self-contained fallback consolidated fact.
    """
    covered = set()
    has_lineage = False
    for fact in consolidated_facts:
        source_ids = fact.get("source_ids", []) or []
        if source_ids:
            has_lineage = True
        for source_id in source_ids:
            if isinstance(source_id, str) and source_id:
                covered.add(source_id)

    if not has_lineage:
        return list(consolidated_facts)

    preserved = list(consolidated_facts)
    next_idx = 1
    existing_ids = {
        fact.get("id")
        for fact in consolidated_facts
        if isinstance(fact, dict) and isinstance(fact.get("id"), str)
    }

    for sf in session_facts:
        sid = sf.get("id")
        if not isinstance(sid, str) or not sid or sid in covered:
            continue
        fallback_id = f"cf_keep_{next_idx:02d}"
        while fallback_id in existing_ids:
            next_idx += 1
            fallback_id = f"cf_keep_{next_idx:02d}"
        existing_ids.add(fallback_id)
        next_idx += 1
        preserved.append({
            "id": fallback_id,
            "fact": sf.get("fact", ""),
            "entities": list(sf.get("entities") or []),
            "tags": list(sf.get("tags") or []),
            "source_ids": [sid],
            "depends_on": [],
        })
    return preserved


def _preserve_uncovered_cross_facts(cross_facts: list[dict], entity_facts: list[dict]) -> list[dict]:
    """Append entity facts that cross-session synthesis failed to cover.

    Cross-session synthesis may merge related items, but it must not silently
    drop a distinct source fact that carries its own target/place/event/object.
    Any source fact not referenced by at least one synthesized cross fact is
    preserved as a fallback cross fact.
    """
    covered = set()
    has_lineage = False
    for fact in cross_facts:
        source_ids = fact.get("source_ids", []) or []
        if source_ids:
            has_lineage = True
        for source_id in source_ids:
            if isinstance(source_id, str) and source_id:
                covered.add(source_id)

    if not has_lineage:
        return list(cross_facts)

    preserved = list(cross_facts)
    next_idx = 1
    existing_ids = {
        fact.get("id")
        for fact in cross_facts
        if isinstance(fact, dict) and isinstance(fact.get("id"), str)
    }

    for sf in entity_facts:
        sid = sf.get("id")
        if not isinstance(sid, str) or not sid or sid in covered:
            continue
        fallback_id = f"xf_keep_{next_idx:02d}"
        while fallback_id in existing_ids:
            next_idx += 1
            fallback_id = f"xf_keep_{next_idx:02d}"
        existing_ids.add(fallback_id)
        next_idx += 1
        sessions = [sf["session"]] if isinstance(sf.get("session"), int) else []
        fallback = {
            "id": fallback_id,
            "fact": sf.get("fact", ""),
            "entities": list(sf.get("entities") or []),
            "tags": list(sf.get("tags") or []),
            "source_ids": [sid],
            "sessions": sessions,
        }
        preserved.append(fallback)
    return preserved

def format_session(session_turns, session_num):
    """Format conversation turns into readable text."""
    lines = []
    for turn in session_turns:
        speaker = turn.get("speaker", "Unknown")
        text = turn.get("text", "")
        dia_id = turn.get("dia_id", f"D{session_num}:?")
        lines.append(f"[{dia_id}] {speaker}: {text}")
    return "\n".join(lines)


# ── Extraction ──

async def extract_session(session_text, session_num, session_date, conv_id,
                          speakers, model, call_extract_fn, fmt=None,
                          block_prompt_overrides=None):
    """Extract atomic facts + temporal links from a single session.

    Format-aware prompt selection.  When *fmt* is ``None`` the format is
    auto-detected via :func:`detect_format`.

    Args:
        call_extract_fn: async fn(model, system, user_msg, max_tokens) -> dict
        fmt: explicit format override (None = auto-detect)
    Returns:
        (conv_id, session_num, session_date, facts, temporal_links)
    """
    if fmt is None:
        fmt = detect_format(session_text)

    # ── JSON_CONV — preprocess then extract ──
    if fmt == "JSON_CONV":
        parsed = _preprocess_json_conv(session_text)
        if isinstance(parsed, list):
            # Multiple chunks — extract each, renumber per-chunk to avoid
            # duplicate IDs and ensure temporal links remap correctly.
            all_facts, all_tlinks = [], []
            fact_offset = 0
            for chunk in parsed:
                _, _, _, cf, ct = await extract_session(
                    chunk, session_num, session_date, conv_id,
                    speakers, model, call_extract_fn, fmt="CONVERSATION")
                # Remap this chunk's IDs with running offset
                chunk_map = {}
                for i, f in enumerate(cf):
                    new_id = f"f_{fact_offset + i + 1:02d}"
                    old_id = f.get("id", "")
                    if old_id:
                        chunk_map[old_id] = new_id
                    f["id"] = new_id
                for tl in ct:
                    tl["before"] = chunk_map.get(tl.get("before", ""), tl.get("before", ""))
                    tl["after"] = chunk_map.get(tl.get("after", ""), tl.get("after", ""))
                fact_offset += len(cf)
                all_facts.extend(cf)
                all_tlinks.extend(ct)
            return conv_id, session_num, session_date, all_facts, all_tlinks
        else:
            session_text = parsed
            fmt = "CONVERSATION"

    # ── LLM extraction — select prompt based on format ──
    try:
        dt = datetime.fromisoformat(session_date.replace("Z", "+00:00"))
        date_str = dt.strftime("%d %B %Y")
        year_minus_1 = str(dt.year - 1)
    except Exception:
        date_str = "2023"
        year_minus_1 = "2022"

    if fmt in ("AGENT_TRACE", "WEB_DOM", "GAME_BOARD", "CODE_TRACE"):
        system_prompt = EXTRACTION_PROMPT_AGENT_TRACE.format(
            episode_id=conv_id, domain=fmt, chunk_num=session_num, total_chunks="?")
        user_msg = f"Chunk {session_num}:\n\n{session_text}\n\nExtract memory-relevant facts."
    elif fmt == "DOCUMENT":
        # ── Block pipeline for documents ──
        from .block_extractor import extract_block
        from .block_merger import merge_block_results
        from .block_segmenter import segment_document_blocks

        blocks = segment_document_blocks(session_text)
        session_metadata = {
            "container_kind": "document",
            "session_date": date_str,
            "session_num": session_num,
            "speakers": speakers,
        }

        block_results = []
        for block in blocks:
            result = await extract_block(
                block, session_metadata, model=model,
                call_extract_fn=call_extract_fn,
                prompt_overrides=block_prompt_overrides)
            block_results.append((block, result))

        merged = merge_block_results(block_results, session_num)
        facts = merged["facts"]
        temporal_links = merged["temporal_links"]

        for f in facts:
            f.setdefault("speaker", None)
            f.setdefault("speaker_role", None)
            f.setdefault("kind", "fact")
        print(f"  [{conv_id}] S{session_num}: {len(facts)} facts, "
              f"{len(temporal_links)} temporal links ({fmt}, {len(blocks)} blocks)")
        return conv_id, session_num, session_date, facts, temporal_links
    elif fmt == "FACT_LIST":
        system_prompt = EXTRACTION_PROMPT_FACT_LIST.format(session_num=session_num)
        user_msg = f"Fact list:\n\n{session_text}\n\nExtract each item as separate atomic facts."
    elif fmt == "NARRATIVE":
        system_prompt = EXTRACTION_PROMPT_NARRATIVE.format(
            session_date=date_str, session_num=session_num, speakers=speakers)
        user_msg = (f"Narrative chunk {session_num} (Date: {date_str}):\n\n{session_text}\n\n"
                    f"Extract atomic facts and explicit temporal links from this narrative.")
    elif fmt == "CONVERSATION":
        # ── Block pipeline for conversations ──
        from .block_extractor import extract_block
        from .block_merger import merge_block_results
        from .block_segmenter import segment_conversation_blocks

        explicit_speakers = None
        if speakers:
            parts = re.split(r"\s+and\s+", speakers, flags=re.IGNORECASE)
            if len(parts) >= 2:
                explicit_speakers = {}
                for p in parts:
                    p = p.strip()
                    if p:
                        explicit_speakers[p.lower()] = p

        blocks = segment_conversation_blocks(session_text, speakers=explicit_speakers)
        session_metadata = {
            "container_kind": "conversation",
            "session_date": date_str,
            "session_num": session_num,
            "speakers": speakers,
        }

        block_results = []
        for block in blocks:
            result = await extract_block(
                block, session_metadata, model=model,
                call_extract_fn=call_extract_fn,
                prompt_overrides=block_prompt_overrides)
            block_results.append((block, result))

        merged = merge_block_results(block_results, session_num)
        facts = merged["facts"]
        temporal_links = merged["temporal_links"]

        for f in facts:
            f.setdefault("speaker", None)
            f.setdefault("speaker_role", None)
            f.setdefault("kind", "fact")
        print(f"  [{conv_id}] S{session_num}: {len(facts)} facts, "
              f"{len(temporal_links)} temporal links ({fmt}, {len(blocks)} blocks)")
        return conv_id, session_num, session_date, facts, temporal_links

    else:
        # Unknown format — legacy fallback.
        system_prompt = EXTRACTION_PROMPT.format(
            session_date=date_str, year_minus_1=year_minus_1, session_num=session_num)
        user_msg = (f"Conversation between {speakers}, Session {session_num} "
                    f"(Date: {date_str}):\n\n{session_text}\n\n"
                    f"Extract all atomic facts and temporal links from this session.")

    result = await call_extract_fn(model, system_prompt, user_msg, max_tokens=8192)
    if isinstance(result, list):
        facts = [f for f in result if isinstance(f, dict)]
        temporal_links = []
    elif isinstance(result, dict):
        facts = [f for f in result.get("facts", []) if isinstance(f, dict)]
        temporal_links = result.get("temporal_links", [])
    else:
        facts, temporal_links = [], []
    for f in facts:
        f["session"] = session_num
        f.setdefault("speaker", None)
        f.setdefault("speaker_role", None)
        f.setdefault("kind", "fact")
    print(f"  [{conv_id}] S{session_num}: {len(facts)} facts, "
          f"{len(temporal_links)} temporal links ({fmt})")
    return conv_id, session_num, session_date, facts, temporal_links


async def consolidate_session(conv_id, sn, session_facts, speakers, model,
                              call_extract_fn):
    """Consolidate granular facts from a single session.

    Returns:
        (conv_id, sn, session_date, consolidated_facts)
    """
    date = session_facts[0].get("session_date", "2023")[:10]
    sys_p = CONSOLIDATION_PROMPT.format(
        source_desc=f"a conversation between {speakers}, session {sn}",
        temporal_rule=f"Session date: {date}. Convert relative dates to absolute.")
    user_msg = f"Session {sn} ({len(session_facts)} facts):\n\n"
    for f in session_facts:
        user_msg += f"- [{f['id']}] {f['fact']}\n"
    user_msg += "\nConsolidate."
    result = await call_extract_fn(model, sys_p, user_msg, max_tokens=8192)
    if isinstance(result, list):
        facts = [f for f in result if isinstance(f, dict)]
    elif isinstance(result, dict):
        facts = [f for f in result.get("facts", []) if isinstance(f, dict)]
    else:
        facts = []
    for f in facts:
        f["session"] = sn
        _normalize_lineage_refs(f, session_facts)
    facts = _preserve_uncovered_session_facts(facts, session_facts)
    for f in facts:
        f["session"] = sn
    return conv_id, sn, session_facts[0].get("session_date", ""), facts


async def cross_session_entity(conv_id, ename, efacts, sessions, model,
                               call_extract_fn):
    """Consolidate facts about one entity across sessions.

    Returns:
        (conv_id, ename, cross_session_facts)
    """
    CHUNK = 15
    chunks = [efacts[i:i + CHUNK] for i in range(0, len(efacts), CHUNK)]
    all_cross = []
    for chunk in chunks:
        sys_p = CROSS_SESSION_PROMPT.format(entity=ename)
        user_msg = f"Entity: {ename}\nSessions: {sessions}\n{len(chunk)} facts:\n\n"
        for fact in chunk:
            user_msg += f"- [{fact['id']}] (session {fact.get('session', '?')}) {fact['fact']}\n"
        user_msg += "\nConsolidate across sessions."
        result = await call_extract_fn(model, sys_p, user_msg, max_tokens=8192)
        if isinstance(result, list):
            all_cross.extend(f for f in result if isinstance(f, dict))
        elif isinstance(result, dict):
            all_cross.extend(f for f in result.get("facts", []) if isinstance(f, dict))
        # else: skip malformed result
    for f in all_cross:
        _normalize_lineage_refs(f, efacts, derive_sessions=True)
    all_cross = _preserve_uncovered_cross_facts(all_cross, efacts)
    return conv_id, ename, all_cross


# ── Supersession Resolution ──

def resolve_supersession(all_facts, fact_lookup):
    """Link facts with supersedes_topic to existing facts via text overlap.

    For each new fact with supersedes_topic set, find the best matching
    existing fact and create bidirectional links:
      new_fact["supersedes"] = old_fact_id
      old_fact["status"] = "superseded"
      old_fact["superseded_by"] = new_fact_id

    Uses simple token overlap (no external deps). Called after all
    extraction tiers are built and fact_lookup is populated.
    """
    # Index facts by entity for fast lookup
    from collections import defaultdict
    entity_index = defaultdict(list)
    for f in all_facts:
        for e in f.get("entities", []):
            if isinstance(e, str):
                entity_index[e.lower()].append(f)

    def _tokenize(text):
        return set(text.lower().split())

    linked = 0
    for f in all_facts:
        topic = f.get("supersedes_topic")
        if not topic:
            continue
        # Guard: LLM sometimes returns list instead of string
        if isinstance(topic, list):
            topic = " ".join(str(t) for t in topic)
        if not isinstance(topic, str):
            continue

        f_id = f.get("id")
        if not f_id:
            continue

        topic_tokens = _tokenize(topic)
        # Candidates: facts sharing at least one entity, from earlier sessions
        f_session = f.get("session", 999)
        candidates = []
        for e in f.get("entities", []):
            for c in entity_index.get(e.lower(), []):
                c_id = c.get("id")
                if c_id and c_id != f_id and c.get("session", 999) < f_session:
                    candidates.append(c)

        if not candidates:
            # Broaden: any fact with topic token overlap
            for c in all_facts:
                c_id = c.get("id")
                if not c_id or c_id == f_id:
                    continue
                if c.get("session", 999) >= f_session:
                    continue
                fact_tokens = _tokenize(c.get("fact", ""))
                if topic_tokens & fact_tokens:
                    candidates.append(c)

        if not candidates:
            continue

        # Score by token overlap with supersedes_topic
        best, best_score = None, 0
        for c in candidates:
            fact_tokens = _tokenize(c.get("fact", ""))
            score = len(topic_tokens & fact_tokens)
            if score > best_score:
                best, best_score = c, score

        if best and best_score > 0:
            f["supersedes"] = best.get("id", "")
            best["status"] = "superseded"
            best["superseded_by"] = f_id
            # Phase 1B: mark superseded fact as outdated
            meta = best.setdefault("metadata", {})
            meta["version_status"] = "outdated"
            meta["version_superseded_by"] = f.get("supersedes_topic", "")
            linked += 1

    return linked


# ═══════════════════════════════════════════════════════════════════════════
# Format Detection (deterministic)
# Moved from multibench/sprint23d/run_23d.py and extended with prompt-aware
# FACT_LIST / NARRATIVE routing.
# ═══════════════════════════════════════════════════════════════════════════

_CONVERSATION_MARKERS = (
    re.compile(r'^\s*(user|assistant|system|tool)\s*:', re.I | re.M),
    re.compile(r'^\[D\d+:[^\]]*\]', re.I | re.M),
)
_DOCUMENT_MARKERS = (
    re.compile(r'^#{1,3}\s', re.M),
    re.compile(r'^---$', re.M),
    re.compile(r'^\*\*[A-Z][^*]+\*\*:', re.M),
)
_FACT_LIST_LINE = re.compile(r'^\s*(\d+[\.\)]\s+|[-*•]\s+)')
_NARRATIVE_SEQUENCE = re.compile(
    r'\b(then|after|before|later|when|eventually|meanwhile)\b', re.I)
_NARRATIVE_THIRD_PERSON = re.compile(
    r'\b(he|she|they|his|her|their)\b', re.I)


def _has_conversation_markers(text: str) -> bool:
    return any(p.search(text) for p in _CONVERSATION_MARKERS)


def _has_document_markers(text: str) -> bool:
    return any(p.search(text) for p in _DOCUMENT_MARKERS)


def _is_fact_list(text: str) -> bool:
    non_empty = [line for line in text.splitlines() if line.strip()]
    if len(non_empty) < 4:
        return False
    if _has_conversation_markers(text) or _has_document_markers(text):
        return False
    numbered = sum(1 for line in non_empty if _FACT_LIST_LINE.match(line))
    return (numbered / len(non_empty)) > 0.6


def _is_narrative(text: str) -> bool:
    if _has_conversation_markers(text) or _has_document_markers(text):
        return False
    if _is_fact_list(text):
        return False
    paragraphs = [p for p in re.split(r'\n\s*\n', text) if p.strip()]
    if len(paragraphs) < 2 and len(text) <= 800:
        return False
    narrative_signals = sum([
        bool(re.search(r'["“”]', text)),
        bool(_NARRATIVE_SEQUENCE.search(text)),
        bool(_NARRATIVE_THIRD_PERSON.search(text)),
    ])
    return narrative_signals >= 2

def detect_format(text: str) -> str:
    """Deterministic format detection for session text.

    Returns one of:
        CONVERSATION, DOCUMENT, AGENT_TRACE, JSON_CONV,
        WEB_DOM, GAME_BOARD, CODE_TRACE, FACT_LIST, NARRATIVE
    """
    if "RootWebArea" in text and "focused:" in text:
        return "WEB_DOM"
    if re.search(r'\[Step \d+\]\nAction:', text):
        return "AGENT_TRACE"
    if re.search(r'^[A-Z]\|(\s[A-Z]){3,}', text, re.M):
        return "GAME_BOARD"
    if re.search(r'execute_bash|EXECUTION RESULT|^\$\s', text, re.M):
        return "CODE_TRACE"
    # JSON_CONV: raw JSON conversation arrays
    if text.strip().startswith('[') and (
        '"role"' in text[:500] or "'Chat Time:" in text[:500]):
        return "JSON_CONV"
    if _is_fact_list(text):
        return "FACT_LIST"
    if _has_document_markers(text):
        return "DOCUMENT"
    if _is_narrative(text):
        return "NARRATIVE"
    return "CONVERSATION"


# ═══════════════════════════════════════════════════════════════════════════
# Preprocessing helpers
# ═══════════════════════════════════════════════════════════════════════════

def _preprocess_json_conv(text: str):
    """Convert raw JSON conversation string to readable text.

    Handles both Python literal (ast.literal_eval) and standard JSON formats.
    Supports flat dicts, nested lists of dicts, and mixed string/list items.
    """
    import ast
    import json as _json

    try:
        data = ast.literal_eval(text)
    except Exception:
        try:
            data = _json.loads(text)
        except Exception:
            return text

    lines = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                lines.append(item)  # "Chat Time: ..."
            elif isinstance(item, list):
                for msg in item:
                    if isinstance(msg, dict):
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")
                        if content:
                            lines.append(f"{role}: {content}")
            elif isinstance(item, dict):
                role = item.get("role", "unknown")
                content = item.get("content", "")
                if content:
                    lines.append(f"{role}: {content}")
    readable = "\n".join(lines) if lines else text

    # If result is large — split into chunks for extraction
    if len(readable) > MAX_DOC_CHUNK_CHARS:
        return _chunk_document(readable, chunk_size=MAX_DOC_CHUNK_CHARS)
    return readable


MAX_DOC_CHUNK_CHARS = 8000  # ~2K tokens per chunk


def _chunk_document(text: str, chunk_size: int = MAX_DOC_CHUNK_CHARS) -> list:
    """Split long document into overlapping chunks.

    Uses a 500-character overlap to avoid losing context at chunk boundaries.
    Returns a list of non-empty chunk strings.
    """
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    step = chunk_size - 500  # 500 char overlap
    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


EXTRACTION_PROMPT_LEGACY = EXTRACTION_PROMPT  # alias for backward compat


# ═══════════════════════════════════════════════════════════════════════════
# 3-tier decision helper
# ═══════════════════════════════════════════════════════════════════════════

def _needs_3tier(sessions: list) -> bool:
    """Determine if 3-tier indexing helps or hurts.

    3-tier helps: haystack data (many sessions, multi-conversation)
    3-tier hurts: single conversation data (few sessions, same speakers)

    Threshold from EV-14/EV-16: 3-tier helps when N >= 10 sessions.
    Below 10 sessions, granular-only retrieval performs better.
    """
    return len(sessions) >= 10


# ═══════════════════════════════════════════════════════════════════════════
# L1 Classification (lightweight metadata enrichment for pre-extracted facts)
# ═══════════════════════════════════════════════════════════════════════════

async def classify_fact(text: str, model: str, call_extract_fn) -> dict:
    """Classify a single fact text into kind/entities/tags via LLM.

    Uses the classify.md prompt. Returns dict with kind, entities, tags,
    event_date, supersedes_topic.  Returns {} on any error.
    """
    prompt = _load_extraction_prompt("classify")
    system = prompt.format(chunk_text=text)
    try:
        result = await call_extract_fn(model, system, "", max_tokens=256)
        valid_kinds = {"fact", "rule", "constraint", "decision", "lesson_learned",
                       "preference", "count_item", "action_item", "observation"}
        if result.get("kind") not in valid_kinds:
            result["kind"] = "fact"
        if not isinstance(result.get("entities"), list):
            result["entities"] = []
        if not isinstance(result.get("tags"), list):
            result["tags"] = []
        return result
    except Exception:
        return {}


def merge_l1_metadata(fact: dict, metadata: dict) -> None:
    """Merge L1 classification metadata into an existing fact dict.

    Only overwrites fields that are missing or at default values in the fact.
    """
    if not metadata:
        return
    if metadata.get("kind", "fact") != "fact" and fact.get("kind", "fact") == "fact":
        fact["kind"] = metadata["kind"]
    if metadata.get("entities") and not fact.get("entities"):
        fact["entities"] = metadata["entities"]
    if metadata.get("tags") and not fact.get("tags"):
        fact["tags"] = metadata["tags"]
    if metadata.get("event_date") and not fact.get("event_date"):
        fact["event_date"] = metadata["event_date"]
    if metadata.get("supersedes_topic") and not fact.get("supersedes_topic"):
        fact["supersedes_topic"] = metadata["supersedes_topic"]
