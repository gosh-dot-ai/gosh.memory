#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic temporal index helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .temporal_values import temporal_sort_day_from_repr


def empty_temporal_index() -> dict:
    return {
        "timelines": {},
        "events": {},
        "anchors": {},
        "calendar_sorted_event_ids": [],
    }


def load_temporal_index(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return empty_temporal_index()
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return empty_temporal_index()
    index = empty_temporal_index()
    index["timelines"] = dict(data.get("timelines") or {})
    index["events"] = dict(data.get("events") or {})
    index["anchors"] = dict(data.get("anchors") or {})
    index["calendar_sorted_event_ids"] = list(data.get("calendar_sorted_event_ids") or [])
    return index


def add_event(index: dict, event: dict) -> None:
    event_id = str(event.get("event_id") or "").strip()
    if not event_id:
        return
    timeline_id = str(event.get("timeline_id") or "").strip()
    index["events"][event_id] = event
    if timeline_id:
        timeline = index["timelines"].setdefault(
            timeline_id,
            {
                "ordinal_kind": event.get("ordinal_kind"),
                "ordered_event_ids": [],
            },
        )
        ordered = timeline.setdefault("ordered_event_ids", [])
        if event_id not in ordered:
            ordered.append(event_id)
    ordinal_kind = str(event.get("ordinal_kind") or "").strip().lower()
    ordinal_index = event.get("ordinal_index")
    if ordinal_kind and isinstance(ordinal_index, int):
        anchor_key = f"{ordinal_kind}:{ordinal_index}"
        anchor_list = index["anchors"].setdefault(anchor_key, [])
        if event_id not in anchor_list:
            anchor_list.append(event_id)


def _event_time_sort_start(event: dict) -> int | None:
    value = event.get("time_sort_start")
    if isinstance(value, (int, float)):
        return int(value)
    return temporal_sort_day_from_repr(str(event.get("time_start") or "").strip())


def _event_time_sort_end(event: dict) -> int | None:
    value = event.get("time_sort_end")
    if isinstance(value, (int, float)):
        return int(value)
    fallback = str(event.get("time_end") or event.get("time_start") or "").strip()
    return temporal_sort_day_from_repr(fallback)


def finalize_temporal_index(index: dict) -> dict:
    def _ordinal_sort_value(event_id: str) -> int:
        value = index["events"].get(event_id, {}).get("ordinal_start")
        return value if isinstance(value, int) else 10**9

    for timeline in index.get("timelines", {}).values():
        ordered = list(timeline.get("ordered_event_ids") or [])
        ordered.sort(
            key=lambda event_id: (
                _ordinal_sort_value(event_id),
                index["events"].get(event_id, {}).get("_source_order", 10**9),
                event_id,
            )
        )
        timeline["ordered_event_ids"] = ordered
    calendar_event_ids = [
        event_id
        for event_id, event in index.get("events", {}).items()
        if event.get("time_start")
    ]
    calendar_event_ids.sort(
        key=lambda event_id: (
            _event_time_sort_start(index["events"].get(event_id, {}) or {}) if _event_time_sort_start(index["events"].get(event_id, {}) or {}) is not None else 10**18,
            index["events"].get(event_id, {}).get("_source_order", 10**9),
            event_id,
        )
    )
    index["calendar_sorted_event_ids"] = calendar_event_ids
    for event in index.get("events", {}).values():
        event.pop("_source_order", None)
    return index


def _event_matches_scope(event: dict, source_ids: set[str] | None, timeline_ids: set[str] | None) -> bool:
    if source_ids is not None and event.get("source_id", "") not in source_ids:
        return False
    if timeline_ids is not None and event.get("timeline_id", "") not in timeline_ids:
        return False
    return True


def _ordinal_event_quality(event: dict) -> tuple[int, int]:
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
    if payload.get("episode_id"):
        score += 1
    support_fact_ids = [str(fid).strip() for fid in (event.get("support_fact_ids") or []) if str(fid).strip()]
    return score, len(support_fact_ids)


def _best_event_per_timeline_ordinal(events: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for event in events:
        value = event.get("ordinal_index")
        if not isinstance(value, int):
            continue
        timeline_id = str(event.get("timeline_id") or "")
        grouped.setdefault((timeline_id, value), []).append(event)
    selected: list[dict] = []
    for (timeline_id, value) in sorted(grouped):
        ranked = sorted(
            grouped[(timeline_id, value)],
            key=lambda event: (
                -_ordinal_event_quality(event)[0],
                -_ordinal_event_quality(event)[1],
                int(event["ordinal_start"]) if isinstance(event.get("ordinal_start"), int) else 10**9,
                str(event.get("event_id") or ""),
            ),
        )
        if ranked:
            selected.append(ranked[0])
    return selected


def lookup_ordinal_anchor(
    index: dict,
    *,
    kind: str,
    value: int,
    source_ids: set[str] | None = None,
    timeline_ids: set[str] | None = None,
) -> list[dict]:
    anchor_key = f"{str(kind).lower()}:{int(value)}"
    events = []
    for event_id in index.get("anchors", {}).get(anchor_key, []):
        event = index.get("events", {}).get(event_id)
        if not event:
            continue
        if _event_matches_scope(event, source_ids, timeline_ids):
            events.append(event)
    events.sort(
        key=lambda event: (
            event.get("ordinal_start", 10**9),
            event.get("timeline_id", ""),
            event.get("event_id", ""),
        )
    )
    return events


def lookup_ordinal_range(
    index: dict,
    *,
    kind: str,
    start: int,
    end: int,
    source_ids: set[str] | None = None,
    timeline_ids: set[str] | None = None,
) -> list[dict]:
    lo = min(int(start), int(end))
    hi = max(int(start), int(end))
    hits = []
    for event in index.get("events", {}).values():
        if str(event.get("ordinal_kind") or "").lower() != str(kind).lower():
            continue
        ordinal_index = event.get("ordinal_index")
        if not isinstance(ordinal_index, int) or not (lo <= ordinal_index <= hi):
            continue
        if _event_matches_scope(event, source_ids, timeline_ids):
            hits.append(event)
    hits.sort(
        key=lambda event: (
            event.get("timeline_id", ""),
            event.get("ordinal_start", 10**9),
            event.get("event_id", ""),
        )
    )
    return _best_event_per_timeline_ordinal(hits)


def lookup_calendar_overlap(
    index: dict,
    *,
    time_start: str,
    time_end: str,
    time_start_sort: int | None = None,
    time_end_sort: int | None = None,
    source_ids: set[str] | None = None,
    timeline_ids: set[str] | None = None,
) -> list[dict]:
    if not time_start or not time_end:
        return []
    query_start_sort = int(time_start_sort) if isinstance(time_start_sort, (int, float)) else temporal_sort_day_from_repr(time_start)
    query_end_sort = int(time_end_sort) if isinstance(time_end_sort, (int, float)) else temporal_sort_day_from_repr(time_end)
    hits = []
    for event_id in index.get("calendar_sorted_event_ids", []):
        event = index.get("events", {}).get(event_id)
        if not event:
            continue
        if not _event_matches_scope(event, source_ids, timeline_ids):
            continue
        event_start = event.get("time_start")
        event_end = event.get("time_end") or event_start
        if not event_start:
            continue
        event_start_sort = _event_time_sort_start(event)
        event_end_sort = _event_time_sort_end(event)
        if (
            query_start_sort is not None
            and query_end_sort is not None
            and event_start_sort is not None
            and event_end_sort is not None
        ):
            if event_end_sort < query_start_sort or event_start_sort > query_end_sort:
                continue
        elif event_end < time_start or event_start > time_end:
            continue
        hits.append(event)
    return hits


def latest_calendar_anchor(
    index: dict,
    *,
    source_ids: set[str] | None = None,
    timeline_ids: set[str] | None = None,
) -> str | None:
    latest_value: str | None = None
    latest_sort: int | None = None
    for event_id in index.get("calendar_sorted_event_ids", []):
        event = index.get("events", {}).get(event_id)
        if not event:
            continue
        if not _event_matches_scope(event, source_ids, timeline_ids):
            continue
        candidate = str(event.get("time_end") or event.get("time_start") or "").strip()
        if not candidate:
            continue
        candidate_sort = _event_time_sort_end(event)
        if latest_sort is None or (candidate_sort is not None and candidate_sort > latest_sort):
            latest_sort = candidate_sort
            latest_value = candidate
        elif latest_sort is None and latest_value is not None and candidate > latest_value:
            latest_value = candidate
    return latest_value


def lookup_events_for_fact(
    index: dict,
    *,
    fact_id: str,
    source_ids: set[str] | None = None,
    timeline_ids: set[str] | None = None,
) -> list[dict]:
    if not fact_id:
        return []
    hits = []
    for event in index.get("events", {}).values():
        support_fact_ids = event.get("support_fact_ids") or []
        if fact_id not in support_fact_ids:
            continue
        if _event_matches_scope(event, source_ids, timeline_ids):
            hits.append(event)
    hits.sort(
        key=lambda event: (
            _event_time_sort_start(event) if _event_time_sort_start(event) is not None else 10**18,
            event.get("ordinal_start", 10**9),
            event.get("timeline_id", ""),
            event.get("event_id", ""),
        )
    )
    return hits


def lookup_adjacent_timeline_event(
    index: dict,
    *,
    event: dict,
    delta: int = 1,
) -> dict | None:
    timeline_id = str(event.get("timeline_id") or "").strip()
    event_id = str(event.get("event_id") or "").strip()
    if not timeline_id or not event_id:
        return None
    timeline = index.get("timelines", {}).get(timeline_id) or {}
    ordered = list(timeline.get("ordered_event_ids") or [])
    if event_id not in ordered:
        return None
    next_pos = ordered.index(event_id) + int(delta)
    if next_pos < 0 or next_pos >= len(ordered):
        return None
    next_id = ordered[next_pos]
    return index.get("events", {}).get(next_id)


def iso_now_floor(value: str | None) -> str | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None
    return dt.isoformat()
