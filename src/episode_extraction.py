#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reusable episode extraction primitives built on deterministic blocks."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .common import call_oai
from .episodes import save_episode_corpus, validate_episode


def extract_doc_metadata(text: str, doc_id: str) -> dict:
    title = doc_id
    date = "2026-01-01"
    m = re.search(r"^#\s+DOC-\d+:\s*(.+)$", text, re.MULTILINE)
    if m:
        title = m.group(1).strip()
    m = re.search(r"\|\s*Date\s*\|\s*(.+?)\s*\|", text)
    if m:
        date = _parse_date(m.group(1))
    else:
        m = re.search(r"\*\*Date:\*\*\s*(.+)", text)
        if m:
            date = _parse_date(m.group(1))
    return {"title": title, "date": date}


def _parse_date(raw: str) -> str:
    raw = raw.strip()
    if re.match(r"\d{4}-\d{2}-\d{2}", raw):
        return raw[:10]
    months = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
    }
    m = re.match(r"(\d{1,2})\s+(\w+)\s+(\d{4})", raw)
    if m:
        return f"{m.group(3)}-{months.get(m.group(2).lower(), '01')}-{m.group(1).zfill(2)}"
    return raw


def build_grouping_prompt(
    doc_id: str,
    doc_title: str,
    doc_date: str,
    blocks_text: str,
    config: dict,
) -> str:
    shared = f"""You are a document episode segmenter.

Your job is to partition the document into semantic EPISODES using ONLY the block IDs provided.

HARD RULES:
- Every block must belong to exactly one episode.
- No block may be omitted.
- No block may appear in more than one episode.
- Do NOT rewrite the document text.
- Do NOT summarize the final raw text.
- Your output is only block grouping + metadata.
- Split when topic changes OR state changes.
- Do NOT merge fail + pass.
- Do NOT merge draft + canonical.
- Do NOT merge requirement + actual status.
- Keep episodes operationally useful and readable.

Each episode must have:
- topic_key
- state_label
- currentness
- block_ids

Document ID: {doc_id}
Document Title: {doc_title}
Document Date: {doc_date}

Blocks:
{blocks_text}

Return ONLY valid JSON array:
[
  {{
    "topic_key": "...",
    "state_label": "...",
    "currentness": "current|outdated|historical|unknown",
    "block_ids": ["{doc_id}_b000", "{doc_id}_b001"]
  }}
]
"""
    mode = config.get("prompt_mode", "baseline")
    if mode == "strict_partition":
        return shared + "\nAdditional guidance: if unsure, prefer smaller episodes over giant mixed episodes.\n/no_think"
    if mode == "strict_small":
        return shared + "\nAdditional guidance: avoid giant overview episodes; when in doubt split by local event, measurement, approval, or state transition.\n/no_think"
    if mode == "metadata_first":
        return shared + "\nAdditional guidance: be minimal in labels, but precise in block_ids; exact partition is more important than elegant topic names.\n/no_think"
    return shared + "\n/no_think"


def parse_grouping_json_array(text: str) -> list | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text).strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "episodes" in result:
            return result["episodes"]
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            return None
    return None


def blocks_text(block_dicts: list[dict], preview_only: bool = False) -> str:
    parts = []
    for bd in block_dicts:
        preview = bd["text_preview"] if preview_only else bd["text"]
        parts.append(
            f"[{bd['block_id']}] section={bd['section_path'] or '(top)'} "
            f"type={bd['family']} chars={bd['char_len']}\n{preview}"
        )
    return "\n\n".join(parts)


async def group_block_batch(
    model: str,
    doc_id: str,
    doc_title: str,
    doc_date: str,
    block_batch: list[dict],
    config: dict,
    sem,
) -> tuple[list | None, str]:
    attempts = [
        ("full", blocks_text(block_batch, preview_only=False)),
        ("preview", blocks_text(block_batch, preview_only=True)),
    ]
    last_raw = ""
    for _mode_name, batch_text in attempts:
        prompt = build_grouping_prompt(doc_id, doc_title, doc_date, batch_text, config)
        last_raw = await call_oai(model, prompt, max_tokens=8192, temperature=0, semaphore=sem)
        parsed = parse_grouping_json_array(last_raw)
        if parsed:
            return parsed, last_raw
    return None, last_raw


def split_large_episode(valid_bids: list[str], block_map: dict[str, dict], size_cap: int) -> list[list[str]]:
    if not valid_bids:
        return []
    chunks = []
    current = []
    current_len = 0
    for bid in valid_bids:
        block_len = len(block_map[bid]["text"])
        if current and current_len + block_len > size_cap:
            chunks.append(current)
            current = [bid]
            current_len = block_len
        else:
            current.append(bid)
            current_len += block_len
    if current:
        chunks.append(current)
    return chunks


def reconstruct_episodes(
    doc_id: str,
    doc_date: str,
    block_dicts: list[dict],
    llm_episodes: list[dict],
    config: dict,
) -> list[dict]:
    block_map = {bd["block_id"]: bd for bd in block_dicts}
    used = set()
    episodes = []

    def _episode_from_bids(ep_idx: int, topic_key: str, state_label: str, currentness: str, valid_bids: list[str]):
        if not valid_bids:
            return None
        section_paths = []
        for bid in valid_bids:
            sp = block_map[bid]["section_path"]
            if sp and sp not in section_paths:
                section_paths.append(sp)
        raw_text = "\n\n".join(block_map[bid]["text"] for bid in valid_bids)
        raw_span = [block_map[valid_bids[0]]["raw_span"][0], block_map[valid_bids[-1]]["raw_span"][1]]
        episode = {
            "episode_id": f"{doc_id}_e{ep_idx:02d}",
            "source_type": "document",
            "source_id": doc_id,
            "source_date": doc_date,
            "topic_key": topic_key,
            "state_label": state_label,
            "currentness": currentness if currentness in ("current", "outdated", "historical", "unknown") else "unknown",
            "raw_text": raw_text,
            "provenance": {
                "block_ids": valid_bids,
                "source_section_path": " + ".join(section_paths) if section_paths else None,
                "raw_span": raw_span,
            },
        }
        if validate_episode(episode):
            return None
        return episode

    ep_counter = 1
    for llm_ep in llm_episodes:
        raw_bids = list(llm_ep.get("block_ids", []))
        valid = []
        for bid in raw_bids:
            if bid not in block_map:
                continue
            if config.get("strip_duplicates", True) and bid in used:
                continue
            valid.append(bid)
            used.add(bid)
        if not valid:
            continue
        chunks = split_large_episode(valid, block_map, config.get("size_cap_chars", 12000))
        for chunk_idx, chunk in enumerate(chunks):
            topic = llm_ep.get("topic_key", f"topic_{ep_counter}")
            state = llm_ep.get("state_label", "unknown")
            if len(chunks) > 1:
                topic = f"{topic} part {chunk_idx + 1}"
            ep = _episode_from_bids(
                ep_counter,
                topic,
                state,
                llm_ep.get("currentness", "unknown"),
                chunk,
            )
            if ep is not None:
                episodes.append(ep)
                ep_counter += 1

    expected = [bd["block_id"] for bd in block_dicts]
    missing = [bid for bid in expected if bid not in used]
    if missing and config.get("attach_missing"):
        for bid in missing:
            attached = False
            for ep in reversed(episodes):
                last_bid = ep["provenance"]["block_ids"][-1]
                last_num = int(last_bid.rsplit("b", 1)[1])
                cur_num = int(bid.rsplit("b", 1)[1])
                if abs(last_num - cur_num) <= 1:
                    ep["provenance"]["block_ids"].append(bid)
                    ep["raw_text"] += "\n\n" + block_map[bid]["text"]
                    ep["provenance"]["raw_span"][1] = block_map[bid]["raw_span"][1]
                    used.add(bid)
                    attached = True
                    break
            if not attached:
                continue
        missing = [bid for bid in expected if bid not in used]

    if missing and config.get("singleton_missing"):
        for bid in missing:
            ep = _episode_from_bids(
                ep_counter,
                "structural_context",
                "structural_context",
                "unknown",
                [bid],
            )
            if ep is not None:
                episodes.append(ep)
                ep_counter += 1
                used.add(bid)

    return episodes


def build_singleton_episodes(
    doc_id: str,
    doc_date: str,
    block_dicts: list[dict],
) -> list[dict]:
    """Deterministic fallback: one block becomes one episode."""
    llm_episodes = [
        {
            "topic_key": "structural_context",
            "state_label": "structural_context",
            "currentness": "unknown",
            "block_ids": [bd["block_id"]],
        }
        for bd in block_dicts
    ]
    return reconstruct_episodes(
        doc_id,
        doc_date,
        block_dicts,
        llm_episodes,
        {
            "strip_duplicates": True,
            "size_cap_chars": 10**9,
            "attach_missing": False,
            "singleton_missing": False,
        },
    )


async def group_document(
    model: str,
    doc_id: str,
    doc_title: str,
    doc_date: str,
    block_dicts: list[dict],
    config: dict,
    sem,
) -> tuple[list[dict], str, str]:
    attempts = [
        ("full", blocks_text(block_dicts, preview_only=False)),
        ("preview", blocks_text(block_dicts, preview_only=True)),
    ]
    raw_response = ""
    llm_episodes = None
    mode_used = None
    for mode_name, block_text in attempts:
        prompt = build_grouping_prompt(doc_id, doc_title, doc_date, block_text, config)
        raw_response = await call_oai(model, prompt, max_tokens=8192, temperature=0, semaphore=sem)
        llm_episodes = parse_grouping_json_array(raw_response)
        if llm_episodes:
            mode_used = mode_name
            break
    if not llm_episodes and len(block_dicts) > 80:
        batched = []
        batch_raw_outputs = []
        batch_size = 40
        for start in range(0, len(block_dicts), batch_size):
            batch = block_dicts[start:start + batch_size]
            parsed, batch_raw = await group_block_batch(
                model, doc_id, doc_title, doc_date, batch, config, sem
            )
            batch_raw_outputs.append({"start": start, "end": start + len(batch) - 1, "raw_output": batch_raw})
            if not parsed:
                raise RuntimeError(f"{doc_id}: failed to parse batched grouping response")
            batched.extend(parsed)
        llm_episodes = batched
        raw_response = json.dumps(batch_raw_outputs, ensure_ascii=False)
        mode_used = "batched"
    if not llm_episodes:
        raise RuntimeError(f"{doc_id}: failed to parse episode grouping response")
    episodes = reconstruct_episodes(doc_id, doc_date, block_dicts, llm_episodes, config)
    return episodes, raw_response, mode_used or "unknown"
