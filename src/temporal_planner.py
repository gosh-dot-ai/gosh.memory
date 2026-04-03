#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal temporal planner and executors."""

from __future__ import annotations

import re
from datetime import datetime

from .temporal import (
    lookup_adjacent_timeline_event,
    lookup_calendar_overlap,
    lookup_ordinal_anchor,
    lookup_ordinal_range,
)
from .temporal_values import (
    format_temporal_date_repr,
    shift_anchor_by_days,
    temporal_sort_day_from_datetime,
    temporal_sort_day_from_repr,
)

ORDINAL_FROM_TO_RE = re.compile(
    r"\bfrom\s+(step|turn|message)\s+(\d+)\s+(?:to|through)\s+(?:(?:step|turn|message)\s+)?(\d+)\b",
    re.I,
)
ORDINAL_BETWEEN_RE = re.compile(
    r"\bbetween\s+(step|turn|message)s?\s+(\d+)\s+and\s+(?:(?:step|turn|message)s?\s+)?(\d+)\b",
    re.I,
)
ORDINAL_SPAN_RE = re.compile(
    r"\b(step|turn|message)s?\s+(\d+)\s*(?:-|to|through)\s*(\d+)\b",
    re.I,
)
ORDINAL_EXACT_RE = re.compile(r"\b(step|turn|message)\s+(\d+)\b", re.I)
ORDINAL_RANGE_CUE_RE = re.compile(r"\bbetween\b|\bfrom\b", re.I)
OSCILLATION_LOOP_RE = re.compile(r"\b(?:loop|oscillat(?:e|ed|ing|ion))\b", re.I)
OSCILLATION_MODULE_PAIR_RE = re.compile(r"between\s+the\s+'([^']+)'\s+and\s+'([^']+)'\s+modules", re.I)
FIRST_OCCURRENCE_RE = re.compile(r"^\s*(?:which|what)\s+step\s+first\s+(.+?)\??\s*$", re.I)
CONSECUTIVE_BEFORE_SCROLL_RE = re.compile(
    r"^\s*how\s+many\s+consecutive\s+'([^']+)'\s+link\s+clicks\s+occurred\s+before\s+the\s+first\s+scroll\s+action\??\s*$",
    re.I,
)
FAILED_PARSE_TRACE_RE = re.compile(
    r"\bfailed to parse actions\b.*\bfirst made the formatting choice\b",
    re.I,
)
STEP_NUMBER_RE = re.compile(r"\bstep\s+(\d+)\b", re.I)
ID_LABEL_STEP_RE = re.compile(r"'([^']+)'\s+link\s+clicked\s+in\s+Step\s+(\d+)", re.I)
CALENDAR_EXACT_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
CALENDAR_TEXT_DATE_RE = re.compile(
    r"\b(?:on\s+)?(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+\d{1,2},?\s+\d{4}\b",
    re.I,
)
CALENDAR_YEAR_RE = re.compile(r"\bin\s+\d{4}\b", re.I)
CALENDAR_RELATIVE_RE = re.compile(
    r"\b(?:\d+\s+(?:years?|months?|weeks?)\s+ago|last\s+(?:week|month|year)|for\s+\d+\s+(?:years?|months?|weeks?))\b",
    re.I,
)
AGO_RE = re.compile(r"\b(\d+)\s+(years?|months?|weeks?)\s+ago\b", re.I)
LAST_RE = re.compile(r"\blast\s+(week|month|year)\b", re.I)
FOR_RE = re.compile(r"\bfor\s+(\d+)\s+(years?|months?|weeks?)\b", re.I)
CALENDAR_SEEKING_PREFIX_RULES = (
    (re.compile(r"^\s*when did\b", re.I), "date"),
    (re.compile(r"^\s*when was\b", re.I), "date"),
    (re.compile(r"^\s*when is\b", re.I), "date"),
    (re.compile(r"^\s*what year\b", re.I), "year"),
    (re.compile(r"^\s*which year\b", re.I), "year"),
    (re.compile(r"^\s*what month\b", re.I), "month"),
    (re.compile(r"^\s*which month\b", re.I), "month"),
    (re.compile(r"^\s*what date\b", re.I), "date"),
    (re.compile(r"^\s*what was the date\b", re.I), "date"),
    (re.compile(r"^\s*since when\b", re.I), "date"),
)
EXACT_TOOL_RE = re.compile(r"\b(?:which|what)\s+tool\b", re.I)
EXACT_SQL_RE = re.compile(r"\b(?:what|which)\s+(?:exact\s+)?(?:sql|query)\b", re.I)
EXACT_SQL_RESULT_RE = re.compile(
    r"\b(?:what|which)\s+(?:exact\s+)?(?:sql|query)\b.*\b(?:numeric result|result did .* observe|what numeric result)\b",
    re.I,
)
EXACT_STATUS_RE = re.compile(
    r"\b(?:specific\s+)?(?:result|execution)\s+status\b|\bstatus of the action\b",
    re.I,
)
TOOL_MISSING_INPUT_RE = re.compile(
    r"\b(?:what|which)\s+tool\b.*(?:\bmissing\b.*\b(?:input|argument|parameter)\b|\b(?:input|argument|parameter)\b.*\bmissing\b)",
    re.I,
)
EXACT_COMMAND_RE = re.compile(
    r"\b(?:what|which)\s+(?:(?:was|is)\s+the\s+|(?:was|is)\s+)?(?:exact\s+)?(?:[a-z_/-]+\s+){0,2}command\b",
    re.I,
)
EXACT_ACTION_RE = re.compile(
    r"\b(?:what|which)\s+(?:(?:exact|specific)\s+)?action\b|\bwhat did (?:the )?(?:agent|assistant|system) do\b",
    re.I,
)
EXACT_PATH_RE = re.compile(r"\b(?:what|which)\s+(?:exact\s+)?(?:file|filepath|path)\b", re.I)
EXACT_ID_RE = re.compile(r"\b(?:what|which)\s+(?:exact\s+)?id\b", re.I)
AFTER_STEP_CONTEXT_RE = re.compile(r"^\s*after\b.*\b(step|turn|message)\s+\d+\b", re.I)
AT_ANCHOR_FOLLOW_RE = re.compile(
    r"\bat\s+(step|turn|message)\s+\d+\b.*\b(?:immediately followed|happened immediately after|came immediately after)\b",
    re.I,
)
ACTION_LIST_RANGE_RE = re.compile(
    r"\b(?:what actions were performed|(?:what|which)(?:\s+\w+){0,2}\s+actions?\s+occurred)\b",
    re.I,
)
MOVEMENT_ACTION_LIST_RE = re.compile(r"\bwhat are these .*movement actions\b", re.I)
BETWEEN_BOUNDARY_STEPS_RE = re.compile(
    r"\bbetween\b.+\(\s*(?:step|turn|message)\s+\d+\s*\).+\(\s*(?:step|turn|message)\s+\d+\s*\)",
    re.I,
)
ENVIRONMENT_CHANGE_RANGE_RE = re.compile(
    r"\bwhat actions made the environment changes\b|\bwhat were the environment changes\b",
    re.I,
)
CHECKBOX_ORDER_RANGE_RE = re.compile(r"\bcheckbox(?:es)?\b.*\bin what order\b", re.I)
INVALID_ACTION_RANGE_RE = re.compile(r"\binvalid action\b.*\bformatting error\b", re.I)
RELATIVE_DISTANCE_RE = re.compile(r"\bhow many (steps|turns|messages)\s+(?:later|after)\b", re.I)
SEMANTIC_DISTANCE_QUERY_RE = re.compile(
    r"^\s*how many\s+(steps|turns|messages)\s+after\s+(.+?)\s+did\s+(.+?)\??\s*$",
    re.I,
)
CLICK_ACTION_CANON_RE = re.compile(
    r"^click\s+\[(?P<id>\d+)\]\s+where\s+\[\d+\]\s+is\s+(?P<role>[A-Za-z_]+)\s+'(?P<label>[^']+)'",
    re.I,
)
TYPE_ACTION_CANON_RE = re.compile(
    r"^type\s+\[(?P<id>\d+)\]\s+\[(?P<value>.*?)\]\s+where\s+\[\d+\]\s+is\s+(?P<role>[A-Za-z_]+)\s+'(?P<label>[^']+)'",
    re.I,
)
DISTANCE_TARGET_RE = re.compile(
    r"\bdid\s+(?:the\s+first\s+)?(.+?)\s+(?:occur|happen|appear|follow)\b",
    re.I,
)
STATE_CONSTRAINT_RE = re.compile(r"\b([a-z_][a-z0-9_]*)\s*:\s*(true|false)\b", re.I)
GRID_SWAP_ACTION_RE = re.compile(
    r"^\(\((\d+)\s*,\s*(\d+)\)\s*,\s*\((\d+)\s*,\s*(\d+)\)\)$"
)
GRID_BOARD_ROW_RE = re.compile(r"^\s*(\d+)\|\s*([A-Za-z0-9](?:\s+[A-Za-z0-9])*)\s*$", re.M)
GENERIC_OBJECTIVE_STOP_WORDS = {
    "at", "step", "turn", "message", "what", "which", "exact", "action", "actions",
    "command", "commands", "file", "filepath", "path", "tool", "id", "agent",
    "assistant", "system", "did", "do", "execute", "executed", "taken", "take",
    "used", "use", "performed", "perform", "run", "ran", "was", "were", "the",
    "to", "for", "of", "in", "on", "and", "or", "that", "this", "it", "its",
    "identify", "identified", "find", "found", "locate", "located", "location",
    "immediately", "after", "before", "first", "precise", "specific", "furthermore",
    "cite", "prove", "interaction", "indices", "index",
    "method", "class", "function", "code", "implementation",
}
EDIT_ACTION_INTENT_RE = re.compile(
    r"\b(?:modif(?:y|ied|ication)|edit|change|fix|implement|insert|add|update|rewrite)\b",
    re.I,
)
VIEW_ACTION_INTENT_RE = re.compile(
    r"\b(?:examin(?:e|ed|ing)|inspect|view|read|look(?:ing)?|open|explore)\b",
    re.I,
)
SEARCH_ACTION_INTENT_RE = re.compile(
    r"\b(?:search|grep|find)\b",
    re.I,
)
CLIENT_ERROR_RE = re.compile(r"\b(\d{3})\s+Client Error:\s*([A-Za-z][A-Za-z ]+?)(?=\s+for\s+url\b|[.)\n]|$)", re.I)
SERVER_ERROR_RE = re.compile(r"\b(\d{3})\s+Server Error:\s*([A-Za-z][A-Za-z ]+?)(?=\s+for\s+url\b|[.)\n]|$)", re.I)
RESULT_ROW_RE = re.compile(r"^([A-Za-z0-9_-]+)\s*,\s*([0-9][0-9,]*)\s*$", re.M)
MALFORMED_ACTION_PREFIX_RE = re.compile(
    r"^\s*(?:The previous prediction you issued was|Let's think step-?by-?step|Let'?s think step by step)",
    re.I,
)
REQUIRED_TOOL_ARGS = {
    "execute_snowflake_sql": "sql",
    "execute_bash": "command",
}
TAKE_FROM_RE = re.compile(r"^take\s+(.+?)\s+from\s+(.+)$", re.I)
MOVE_TO_RE = re.compile(r"^move\s+(.+?)\s+to\s+(.+)$", re.I)
PUT_INTO_RE = re.compile(r"^put\s+(.+?)\s+(?:in|into|on)\s+(.+)$", re.I)
DROP_RE = re.compile(r"^drop\s+(.+)$", re.I)


def _iso_date(dt: datetime) -> str:
    return dt.date().isoformat()


def classify_temporal_query(question: str) -> str:
    if extract_ordinal_query(question) is not None:
        return "ordinal"
    if extract_calendar_query(question) is not None:
        return "calendar"
    if (
        CALENDAR_EXACT_DATE_RE.search(question)
        or CALENDAR_TEXT_DATE_RE.search(question)
        or CALENDAR_YEAR_RE.search(question)
        or CALENDAR_RELATIVE_RE.search(question)
    ):
        return "calendar"
    return "semantic"


def extract_ordinal_query(question: str) -> dict | None:
    text = str(question or "")
    if FAILED_PARSE_TRACE_RE.search(text):
        step_values = [int(value) for value in STEP_NUMBER_RE.findall(text)]
        upper_bound = max(step_values) if step_values else None
        if isinstance(upper_bound, int):
            return {
                "mode": "first_malformed_before_anchor",
                "kind": "step",
                "upper_bound": upper_bound,
            }
    first_occurrence = FIRST_OCCURRENCE_RE.search(text.strip())
    if first_occurrence:
        return {
            "mode": "first_occurrence",
            "kind": "step",
            "objective_phrase": first_occurrence.group(1).strip(" \t\r\n?.!"),
        }
    consecutive = CONSECUTIVE_BEFORE_SCROLL_RE.search(text.strip())
    if consecutive:
        return {
            "mode": "consecutive_before_scroll",
            "kind": "step",
            "label": consecutive.group(1).strip(),
        }
    if OSCILLATION_LOOP_RE.search(text):
        module_pair = OSCILLATION_MODULE_PAIR_RE.search(text)
        step_values = [int(value) for value in STEP_NUMBER_RE.findall(text)]
        if module_pair and len(step_values) >= 2:
            return {
                "mode": "loop_window",
                "kind": "step",
                "module_terms": [module_pair.group(1).strip(), module_pair.group(2).strip()],
                "step_values": sorted(set(step_values)),
                "id_refs": [
                    {"label": label.strip(), "step": int(step)}
                    for label, step in ID_LABEL_STEP_RE.findall(text)
                ],
            }
    for pattern in (ORDINAL_FROM_TO_RE, ORDINAL_BETWEEN_RE, ORDINAL_SPAN_RE):
        match = pattern.search(question)
        if match:
            kind = match.group(1).lower()
            start = int(match.group(2))
            end = int(match.group(3))
            return {
                "mode": "range",
                "kind": kind,
                "start": min(start, end),
                "end": max(start, end),
            }
    ordinal_refs = [(kind.lower(), int(value)) for kind, value in ORDINAL_EXACT_RE.findall(text)]
    if ORDINAL_RANGE_CUE_RE.search(text) and len(ordinal_refs) >= 2:
        kinds = {kind for kind, _value in ordinal_refs}
        if len(kinds) == 1:
            kind = ordinal_refs[0][0]
            start = ordinal_refs[0][1]
            end = ordinal_refs[1][1]
            return {
                "mode": "range",
                "kind": kind,
                "start": min(start, end),
                "end": max(start, end),
            }
    semantic_distance = SEMANTIC_DISTANCE_QUERY_RE.search(str(question or "").strip())
    if semantic_distance:
        return {
            "mode": "semantic_distance",
            "kind": semantic_distance.group(1).rstrip("s").lower(),
            "anchor_phrase": semantic_distance.group(2).strip(" \t\r\n,?.!"),
            "target_phrase": semantic_distance.group(3).strip(" \t\r\n,?.!"),
        }
    match = ORDINAL_EXACT_RE.search(question)
    if not match:
        return None
    return {
        "mode": "exact",
        "kind": match.group(1).lower(),
        "value": int(match.group(2)),
    }


def extract_calendar_query(question: str) -> dict | None:
    text = str(question or "").strip()
    for pattern, granularity in CALENDAR_SEEKING_PREFIX_RULES:
        match = pattern.search(text)
        if not match:
            continue
        content_query = text[match.end():].strip(" \t:-,?.!")
        content_query = re.sub(r"^(?:did|was|is|were|are|has|have|had)\b\s*", "", content_query, flags=re.I)
        if not content_query:
            return None
        return {
            "mode": "seeking",
            "granularity": granularity,
            "content_query": content_query,
            "prefix": match.group(0).strip().lower(),
        }
    if (
        CALENDAR_EXACT_DATE_RE.search(text)
        or CALENDAR_TEXT_DATE_RE.search(text)
        or CALENDAR_YEAR_RE.search(text)
        or CALENDAR_RELATIVE_RE.search(text)
    ):
        return {"mode": "answer"}
    return None


def resolve_calendar_query_interval(
    question: str,
    *,
    anchor_timestamp: str | None = None,
) -> dict | None:
    text = str(question or "").strip()
    if not text:
        return None
    match = CALENDAR_EXACT_DATE_RE.search(text)
    if match:
        date_str = match.group(0)
        sort_day = temporal_sort_day_from_repr(date_str)
        if sort_day is None:
            return None
        return {
            "time_raw": date_str,
            "time_kind": "point",
            "time_start": date_str,
            "time_end": date_str,
            "time_granularity": "day",
            "time_sort_start": sort_day,
            "time_sort_end": sort_day,
        }
    match = CALENDAR_TEXT_DATE_RE.search(text)
    if match:
        raw = match.group(0)
        normalized = raw.lower().removeprefix("on ").title().replace("  ", " ")
        dt = None
        for pattern in ("%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y"):
            try:
                dt = datetime.strptime(normalized, pattern)
                break
            except Exception:
                continue
        if dt is None:
            return None
        date_str = _iso_date(dt)
        sort_day = temporal_sort_day_from_datetime(dt)
        return {
            "time_raw": raw,
            "time_kind": "point",
            "time_start": date_str,
            "time_end": date_str,
            "time_granularity": "day",
            "time_sort_start": sort_day,
            "time_sort_end": sort_day,
        }
    match = CALENDAR_YEAR_RE.search(text)
    if match:
        year = int(match.group(0).split()[-1])
        start = format_temporal_date_repr(year, 1, 1)
        end = format_temporal_date_repr(year, 12, 31)
        start_sort = temporal_sort_day_from_repr(start)
        end_sort = temporal_sort_day_from_repr(end)
        if start_sort is None or end_sort is None:
            return None
        return {
            "time_raw": match.group(0),
            "time_kind": "interval",
            "time_start": start,
            "time_end": end,
            "time_granularity": "year",
            "time_sort_start": start_sort,
            "time_sort_end": end_sort,
        }
    match = CALENDAR_RELATIVE_RE.search(text)
    if not match:
        return None
    phrase = match.group(0)
    ago_match = AGO_RE.search(phrase)
    if ago_match:
        amount = int(ago_match.group(1))
        unit = ago_match.group(2)
        day_delta = -7 * amount if unit.lower().startswith("week") else -30 * amount if unit.lower().startswith("month") else -365 * amount
        resolved = shift_anchor_by_days(anchor_timestamp, day_delta)
        if resolved is None:
            return None
        date_str, sort_day = resolved
        granularity = "year" if unit.lower().startswith("year") else "month" if unit.lower().startswith("month") else "day"
        return {
            "time_raw": phrase,
            "time_kind": "point",
            "time_start": date_str,
            "time_end": date_str,
            "time_granularity": granularity,
            "time_sort_start": sort_day,
            "time_sort_end": sort_day,
        }
    last_match = LAST_RE.search(phrase)
    if last_match:
        unit = last_match.group(1).lower()
        if unit == "week":
            end = shift_anchor_by_days(anchor_timestamp, -1)
            start = shift_anchor_by_days(anchor_timestamp, -7)
            granularity = "day"
        elif unit == "month":
            end = shift_anchor_by_days(anchor_timestamp, -1)
            start = shift_anchor_by_days(anchor_timestamp, -30)
            granularity = "month"
        else:
            end = shift_anchor_by_days(anchor_timestamp, -1)
            start = shift_anchor_by_days(anchor_timestamp, -365)
            granularity = "year"
        if start is None or end is None:
            return None
        start_repr, start_sort = start
        end_repr, end_sort = end
        return {
            "time_raw": phrase,
            "time_kind": "interval",
            "time_start": start_repr,
            "time_end": end_repr,
            "time_granularity": granularity,
            "time_sort_start": start_sort,
            "time_sort_end": end_sort,
        }
    for_match = FOR_RE.search(phrase)
    if for_match:
        amount = int(for_match.group(1))
        unit = for_match.group(2)
        day_delta = -7 * amount if unit.lower().startswith("week") else -30 * amount if unit.lower().startswith("month") else -365 * amount
        start = shift_anchor_by_days(anchor_timestamp, day_delta)
        end = shift_anchor_by_days(anchor_timestamp, 0)
        if start is None or end is None:
            return None
        start_repr, start_sort = start
        end_repr, end_sort = end
        granularity = "year" if unit.lower().startswith("year") else "month" if unit.lower().startswith("month") else "day"
        return {
            "time_raw": phrase,
            "time_kind": "duration",
            "time_start": start_repr,
            "time_end": end_repr,
            "time_granularity": granularity,
            "time_sort_start": start_sort,
            "time_sort_end": end_sort,
        }
    return None


def execute_ordinal_query(
    question: str,
    index: dict,
    *,
    source_ids: set[str] | None = None,
    timeline_ids: set[str] | None = None,
    limit: int | None = None,
) -> dict:
    plan = extract_ordinal_query(question)
    if plan is None:
        return {"matched": False, "resolved": False, "mode": None, "kind": None, "events": []}
    if plan["mode"] == "loop_window":
        reduction = _reduce_loop_window(index, plan, source_ids=source_ids, timeline_ids=timeline_ids)
        return {
            "matched": bool(reduction.get("answer")),
            "resolved": bool(reduction.get("answer") or reduction.get("event") or reduction.get("events")),
            "mode": plan["mode"],
            "kind": plan["kind"],
            "events": list(reduction.get("events") or []),
            "query": plan,
            "primary_event": reduction.get("event"),
            "deterministic_answer": reduction.get("answer"),
            "reducer": reduction.get("reducer"),
        }
    if plan["mode"] == "first_occurrence":
        reduction = _reduce_first_occurrence(index, plan, source_ids=source_ids, timeline_ids=timeline_ids)
        return {
            "matched": bool(reduction.get("answer")),
            "resolved": bool(reduction.get("answer") or reduction.get("event") or reduction.get("events")),
            "mode": plan["mode"],
            "kind": plan["kind"],
            "events": list(reduction.get("events") or []),
            "query": plan,
            "primary_event": reduction.get("event"),
            "deterministic_answer": reduction.get("answer"),
            "reducer": reduction.get("reducer"),
        }
    if plan["mode"] == "first_malformed_before_anchor":
        reduction = _reduce_first_malformed_before_anchor(index, plan, source_ids=source_ids, timeline_ids=timeline_ids)
        return {
            "matched": bool(reduction.get("answer")),
            "resolved": bool(reduction.get("answer") or reduction.get("event") or reduction.get("events")),
            "mode": plan["mode"],
            "kind": plan["kind"],
            "events": list(reduction.get("events") or []),
            "query": plan,
            "primary_event": reduction.get("event"),
            "deterministic_answer": reduction.get("answer"),
            "reducer": reduction.get("reducer"),
        }
    if plan["mode"] == "consecutive_before_scroll":
        reduction = _reduce_consecutive_before_scroll(index, plan, source_ids=source_ids, timeline_ids=timeline_ids)
        return {
            "matched": bool(reduction.get("answer")),
            "resolved": bool(reduction.get("answer") or reduction.get("event") or reduction.get("events")),
            "mode": plan["mode"],
            "kind": plan["kind"],
            "events": list(reduction.get("events") or []),
            "query": plan,
            "primary_event": reduction.get("event"),
            "deterministic_answer": reduction.get("answer"),
            "reducer": reduction.get("reducer"),
        }
    if plan["mode"] == "semantic_distance":
        reduction = _reduce_semantic_relative_distance(index, plan, source_ids=source_ids, timeline_ids=timeline_ids)
        return {
            "matched": bool(reduction.get("answer")),
            "resolved": bool(reduction.get("answer") or reduction.get("event") or reduction.get("events")),
            "mode": plan["mode"],
            "kind": plan["kind"],
            "events": list(reduction.get("events") or []),
            "query": plan,
            "primary_event": reduction.get("event"),
            "deterministic_answer": reduction.get("answer"),
            "reducer": reduction.get("reducer"),
        }
    resolved_kind = plan["kind"]
    if plan["mode"] == "range":
        events = lookup_ordinal_range(
            index,
            kind=resolved_kind,
            start=plan["start"],
            end=plan["end"],
            source_ids=source_ids,
            timeline_ids=timeline_ids,
        )
    else:
        events = lookup_ordinal_anchor(
            index,
            kind=resolved_kind,
            value=plan["value"],
            source_ids=source_ids,
            timeline_ids=timeline_ids,
        )
    if not events and resolved_kind in {"step", "turn"}:
        alias_kind = "turn" if resolved_kind == "step" else "step"
        if plan["mode"] == "range":
            events = lookup_ordinal_range(
                index,
                kind=alias_kind,
                start=plan["start"],
                end=plan["end"],
                source_ids=source_ids,
                timeline_ids=timeline_ids,
            )
        else:
            events = lookup_ordinal_anchor(
                index,
                kind=alias_kind,
                value=plan["value"],
                source_ids=source_ids,
                timeline_ids=timeline_ids,
            )
        if events:
            resolved_kind = alias_kind
    reduction = (
        _reduce_ordinal_range(question, events, plan=plan)
        if plan["mode"] == "range"
        else _reduce_ordinal_events(question, index, events, plan=plan)
    )
    reduction_events = list(reduction.get("events") or [])
    selected_events = reduction_events if reduction_events else list(events)
    if plan["mode"] != "range" and limit is not None:
        selected_events = selected_events[: max(0, int(limit))]
    resolved = bool(
        reduction.get("answer")
        or reduction.get("event")
        or (plan["mode"] == "range" and reduction_events)
    )
    return {
        "matched": bool(events),
        "resolved": resolved,
        "mode": plan["mode"],
        "kind": plan["kind"],
        "resolved_kind": resolved_kind,
        "events": selected_events,
        "query": plan,
        "primary_event": reduction.get("event"),
        "deterministic_answer": reduction.get("answer"),
        "reducer": reduction.get("reducer"),
    }


def execute_calendar_query(
    question: str,
    index: dict,
    *,
    anchor_timestamp: str | None = None,
    source_ids: set[str] | None = None,
    timeline_ids: set[str] | None = None,
    limit: int | None = None,
) -> dict:
    interval = resolve_calendar_query_interval(question, anchor_timestamp=anchor_timestamp)
    if not interval:
        return {"matched": False, "events": [], "query": None}
    events = lookup_calendar_overlap(
        index,
        time_start=interval["time_start"],
        time_end=interval["time_end"],
        time_start_sort=interval.get("time_sort_start"),
        time_end_sort=interval.get("time_sort_end"),
        source_ids=source_ids,
        timeline_ids=timeline_ids,
    )
    if limit is not None:
        events = events[: max(0, int(limit))]
    return {"matched": bool(events), "events": events, "query": interval}


def _ordinal_event_payload_quality(event: dict) -> tuple[int, int]:
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


def _best_ordinal_event(events: list[dict]) -> dict | None:
    ranked = sorted(
        events,
        key=lambda event: (
            -_ordinal_event_payload_quality(event)[0],
            -_ordinal_event_payload_quality(event)[1],
            int(event["ordinal_start"]) if isinstance(event.get("ordinal_start"), int) else 10**9,
            str(event.get("event_id") or ""),
        ),
    )
    return ranked[0] if ranked else None


def _question_objective_tokens(question: str) -> list[str]:
    raw = str(question or "")
    lowered = raw.lower()
    tokens = re.findall(r"[a-z0-9_]+", lowered)
    camel_parts: list[str] = []
    for word in re.findall(r"\b[A-Z][A-Za-z0-9_]*\b", raw):
        camel_parts.append(word.lower())
        camel_parts.extend(
            part.lower()
            for part in re.findall(r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|\d+", word)
            if part
        )
    tokens.extend(camel_parts)
    seen: set[str] = set()
    ordered: list[str] = []
    for token in tokens:
        if token in GENERIC_OBJECTIVE_STOP_WORDS:
            continue
        if token.isdigit():
            continue
        if len(token) <= 2 and "_" not in token:
            continue
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def _question_action_intent(question: str, reducer: str | None) -> str | None:
    if reducer != "exact_action":
        return None
    text = str(question or "")
    if VIEW_ACTION_INTENT_RE.search(text):
        return "view"
    if SEARCH_ACTION_INTENT_RE.search(text):
        return "search"
    if EDIT_ACTION_INTENT_RE.search(text):
        return "edit"
    return None


def _event_objective_text(event: dict) -> str:
    payload = event.get("payload") or {}
    tool_args = payload.get("tool_args") or {}
    parts = [
        str(payload.get("action_raw") or "").strip(),
        str(payload.get("tool_name") or "").strip(),
        str(payload.get("tool_args_raw") or "").strip(),
        str(payload.get("observation_raw") or "").strip(),
        str(payload.get("step_body") or "").strip(),
    ]
    if isinstance(tool_args, dict):
        parts.extend(str(value).strip() for value in tool_args.values() if str(value).strip())
    parts.extend(str(path).strip() for path in (payload.get("paths") or []) if str(path).strip())
    parts.extend(str(text).strip() for text in (payload.get("support_texts") or []) if str(text).strip())
    return "\n".join(part for part in parts if part).lower()


def _event_action_family(event: dict) -> str:
    payload = event.get("payload") or {}
    tool_name = str(payload.get("tool_name") or payload.get("action_name") or "").strip().lower()
    tool_args = payload.get("tool_args") or {}
    tool_args_raw = str(payload.get("tool_args_raw") or "").strip().lower()
    action_raw = str(payload.get("action_raw") or "").strip().lower()

    if tool_name == "think" or action_raw.startswith("think:"):
        return "think"
    if tool_name == "str_replace_editor":
        command = str(tool_args.get("command") or "").strip().lower() if isinstance(tool_args, dict) else ""
        if command == "view":
            return "view"
        if command in {"str_replace", "insert", "create", "append", "delete"}:
            return "edit"
        return "view"
    if tool_name == "execute_bash":
        command = str(tool_args.get("command") or tool_args_raw or action_raw).strip().lower()
        if re.search(r"\b(?:grep|rg|find|ls)\b", command):
            return "search"
        if re.search(r"\b(?:cat|head|tail|sed|awk|view)\b", command):
            return "view"
        return "run"
    if tool_name == "execute_snowflake_sql":
        return "run"
    if action_raw.startswith("click ") or action_raw.startswith("type "):
        return "ui"
    return "other"


def _event_action_text(event: dict) -> str:
    payload = event.get("payload") or {}
    tool_args = payload.get("tool_args") or {}
    parts = [
        str(payload.get("action_raw") or "").strip(),
        str(payload.get("tool_name") or "").strip(),
        str(payload.get("tool_args_raw") or "").strip(),
    ]
    if isinstance(tool_args, dict):
        parts.extend(str(value).strip() for value in tool_args.values() if str(value).strip())
    parts.extend(str(path).strip() for path in (payload.get("paths") or []) if str(path).strip())
    return "\n".join(part for part in parts if part).lower()


def _event_objective_score(event: dict, tokens: list[str], *, action_only: bool = False) -> int:
    if not tokens:
        return 0
    text = _event_action_text(event) if action_only else _event_objective_text(event)
    score = 0
    for token in tokens:
        if token and token in text:
            score += 2 if ("_" in token or len(token) >= 6) else 1
    return score


def _event_in_scope(event: dict, *, source_ids: set[str] | None = None, timeline_ids: set[str] | None = None) -> bool:
    event_source_id = str(event.get("source_id") or "").strip()
    event_timeline_id = str(event.get("timeline_id") or "").strip()
    if source_ids is not None and event_source_id not in source_ids:
        return False
    if timeline_ids is not None and event_timeline_id not in timeline_ids:
        return False
    return True


def _timeline_best_events(
    index: dict,
    timeline_id: str,
    *,
    kind: str,
    source_ids: set[str] | None = None,
    timeline_ids: set[str] | None = None,
) -> list[dict]:
    if timeline_ids is not None and timeline_id not in timeline_ids:
        return []
    timeline = index.get("timelines", {}).get(timeline_id) or {}
    ordered: list[dict] = []
    by_step: dict[int, list[dict]] = {}
    for event_id in timeline.get("ordered_event_ids") or []:
        event = index.get("events", {}).get(event_id) or {}
        if not _event_in_scope(event, source_ids=source_ids, timeline_ids=timeline_ids):
            continue
        if str(event.get("ordinal_kind") or "").strip().lower() != kind:
            continue
        value = event.get("ordinal_index")
        if not isinstance(value, int):
            continue
        by_step.setdefault(value, []).append(event)
    for value in sorted(by_step):
        best = _best_ordinal_event(by_step[value])
        if best:
            ordered.append(best)
    return ordered


def _action_step_sequence(
    index: dict,
    *,
    kind: str = "step",
    source_ids: set[str] | None = None,
    timeline_ids: set[str] | None = None,
) -> list[dict]:
    sequences: list[dict] = []
    for timeline_id in sorted(index.get("timelines", {})):
        events = _timeline_best_events(
            index,
            timeline_id,
            kind=kind,
            source_ids=source_ids,
            timeline_ids=timeline_ids,
        )
        if not events:
            continue
        sequences.append({"timeline_id": timeline_id, "events": events})
    return sequences


def _objective_candidate_metrics(
    event: dict,
    tokens: list[str],
    *,
    action_intent: str | None,
    anchor_value: int | None,
) -> dict:
    action_score = _event_objective_score(event, tokens, action_only=True)
    total_score = _event_objective_score(event, tokens)
    family = _event_action_family(event)
    intent_bonus = 4 if action_intent and family == action_intent else 0
    intent_mismatch_penalty = 2 if action_intent and family in {"edit", "view", "search"} and family != action_intent else 0
    think_penalty = 3 if family == "think" else 0
    ordinal_value = event.get("ordinal_index")
    distance = (
        abs(int(ordinal_value) - int(anchor_value))
        if isinstance(ordinal_value, int) and isinstance(anchor_value, int)
        else 10**6
    )
    quality_major, quality_minor = _ordinal_event_payload_quality(event)
    composite = (
        (action_score * 2)
        + total_score
        + intent_bonus
        - intent_mismatch_penalty
        - think_penalty
    )
    return {
        "event": event,
        "family": family,
        "action_score": action_score,
        "total_score": total_score,
        "intent_bonus": intent_bonus,
        "intent_mismatch_penalty": intent_mismatch_penalty,
        "think_penalty": think_penalty,
        "distance": distance,
        "quality_major": quality_major,
        "quality_minor": quality_minor,
        "composite": composite,
    }


def _rank_objective_candidates(
    events: list[dict],
    tokens: list[str],
    *,
    action_intent: str | None,
    anchor_value: int | None,
) -> list[dict]:
    ranked = [
        _objective_candidate_metrics(
            event,
            tokens,
            action_intent=action_intent,
            anchor_value=anchor_value,
        )
        for event in events
    ]
    ranked.sort(
        key=lambda row: (
            -row["composite"],
            -row["action_score"],
            -row["total_score"],
            row["distance"],
            -row["quality_major"],
            -row["quality_minor"],
            str((row.get("event") or {}).get("event_id") or ""),
        ),
    )
    return ranked


def _objective_ranked_event(
    events: list[dict],
    tokens: list[str],
    *,
    action_only: bool = False,
) -> tuple[dict | None, int]:
    if not events:
        return None, 0
    ranked = sorted(
        events,
        key=lambda event: (
            -_event_objective_score(event, tokens, action_only=action_only),
            -_ordinal_event_payload_quality(event)[0],
            -_ordinal_event_payload_quality(event)[1],
            str(event.get("event_id") or ""),
        ),
    )
    best = ranked[0]
    return best, _event_objective_score(best, tokens, action_only=action_only)


def _select_objective_event(
    question: str,
    index: dict,
    events: list[dict],
    *,
    plan: dict | None,
    reducer: str | None,
) -> dict | None:
    primary = _best_ordinal_event(events)
    if not primary or reducer != "exact_action":
        return primary

    tokens = _question_objective_tokens(question)
    if not tokens:
        return primary
    action_intent = _question_action_intent(question, reducer)
    if not isinstance(plan, dict) or plan.get("mode") != "exact":
        ranked_current = _rank_objective_candidates(
            events,
            tokens,
            action_intent=action_intent,
            anchor_value=plan.get("value") if isinstance(plan, dict) else None,
        )
        return ((ranked_current[0] or {}).get("event") if ranked_current else None) or primary

    kind = str(plan.get("kind") or primary.get("ordinal_kind") or "").strip().lower()
    value = plan.get("value")
    timeline_id = str(primary.get("timeline_id") or "").strip()
    if not kind or not isinstance(value, int) or not timeline_id:
        ranked_current = _rank_objective_candidates(
            events,
            tokens,
            action_intent=action_intent,
            anchor_value=value if isinstance(value, int) else None,
        )
        return ((ranked_current[0] or {}).get("event") if ranked_current else None) or primary

    ranked_current = _rank_objective_candidates(
        events,
        tokens,
        action_intent=action_intent,
        anchor_value=value,
    )
    best_current_row = ranked_current[0] if ranked_current else None
    best_current = (best_current_row or {}).get("event") or primary
    best_current_match = max(
        int((best_current_row or {}).get("action_score") or 0),
        int((best_current_row or {}).get("total_score") or 0),
    )
    best_current_family = str((best_current_row or {}).get("family") or "")

    # Exact-step queries should stay pinned to the selected step by default.
    # Only if the current step has no meaningful objective match and the trace
    # is a code/tool timeline do we allow a small local repair window.
    allow_repair = best_current_match <= 0 or (
        action_intent is not None
        and best_current_family in {"edit", "view", "search"}
        and best_current_family != action_intent
    )
    if not allow_repair:
        return best_current or primary

    current_families = {_event_action_family(event) for event in events}
    if current_families & {"ui", "think"}:
        return best_current or primary

    local_events = [
        candidate
        for candidate in (index.get("events") or {}).values()
        if str(candidate.get("timeline_id") or "").strip() == timeline_id
        and str(candidate.get("ordinal_kind") or "").strip().lower() == kind
        and isinstance(candidate.get("ordinal_index"), int)
        and abs(int(candidate.get("ordinal_index")) - int(value)) <= 6
        and candidate.get("event_id") not in {event.get("event_id") for event in events}
        and _event_action_family(candidate) not in {"ui", "think"}
    ]
    if not local_events:
        return best_current or primary

    ranked_local = _rank_objective_candidates(
        local_events,
        tokens,
        action_intent=action_intent,
        anchor_value=value,
    )
    best_local_row = ranked_local[0] if ranked_local else None
    if not best_local_row:
        best_local_row = None

    best_current_composite = int((best_current_row or {}).get("composite") or 0)
    if best_local_row:
        best_local_match = max(
            int(best_local_row.get("action_score") or 0),
            int(best_local_row.get("total_score") or 0),
        )
        if best_local_match > 0 and int(best_local_row.get("composite") or 0) > best_current_composite:
            return best_local_row.get("event") or best_current or primary
    return best_current or primary


def _select_reducer(question: str) -> str | None:
    if TOOL_MISSING_INPUT_RE.search(question):
        return "tool_missing_input"
    if EXACT_STATUS_RE.search(question):
        return "exact_status"
    if EXACT_SQL_RESULT_RE.search(question):
        return "exact_sql_result"
    if EXACT_TOOL_RE.search(question):
        return "exact_tool"
    if EXACT_SQL_RE.search(question):
        return "exact_sql"
    if EXACT_COMMAND_RE.search(question):
        return "exact_command"
    if EXACT_PATH_RE.search(question):
        return "exact_path"
    if EXACT_ID_RE.search(question):
        return "exact_id"
    if EXACT_ACTION_RE.search(question):
        return "exact_action"
    return None


def _event_answer_target(question: str, index: dict, event: dict, *, plan: dict | None = None) -> dict:
    if AFTER_STEP_CONTEXT_RE.search(question) or AT_ANCHOR_FOLLOW_RE.search(question):
        kind = str((plan or {}).get("kind") or event.get("ordinal_kind") or "").strip().lower()
        value = (plan or {}).get("value")
        if kind and isinstance(value, int):
            next_step_events = lookup_ordinal_anchor(
                index,
                kind=kind,
                value=value + 1,
                source_ids={str(event.get("source_id") or "")} if str(event.get("source_id") or "").strip() else None,
                timeline_ids={str(event.get("timeline_id") or "")} if str(event.get("timeline_id") or "").strip() else None,
            )
            best_next = _best_ordinal_event(next_step_events)
            if best_next:
                return best_next
        adjacent = lookup_adjacent_timeline_event(index, event=event, delta=1)
        if adjacent:
            return adjacent
    return event


def _canonicalize_action_text(action_raw: str) -> str:
    text = str(action_raw or "").strip()
    if not text:
        return ""
    type_match = TYPE_ACTION_CANON_RE.match(text)
    if type_match:
        value = str(type_match.group("value") or "").strip()
        role = str(type_match.group("role") or "").strip().lower()
        label = str(type_match.group("label") or "").strip()
        if value and label:
            return f"type '{value}' into {role} '{label}'"
    click_match = CLICK_ACTION_CANON_RE.match(text)
    if click_match:
        role = str(click_match.group("role") or "").strip().lower()
        label = str(click_match.group("label") or "").strip()
        if label:
            return f"click {role} '{label}'"
    return text


def _canonicalize_shell_command_text(command: str) -> str:
    text = str(command or "").strip()
    if not text:
        return ""
    if re.search(r"\b(?:grep|rg|find)\b", text):
        text = re.sub(r"\s*\|\s*head(?:\s+-\d+)?\s*$", "", text, flags=re.I).strip()
    return text


def _extract_distance_target_phrase(question: str) -> str:
    text = str(question or "").strip()
    match = DISTANCE_TARGET_RE.search(text)
    if not match:
        return ""
    target = str(match.group(1) or "").strip(" \t\r\n?.!,")
    return re.sub(r"\s+", " ", target)


def _tokenize_distance_target(target: str) -> list[str]:
    lowered = str(target or "").lower()
    tokens = re.findall(r"[a-z0-9_]+", lowered)
    stop = {"the", "a", "an", "first"}
    return [tok for tok in tokens if tok not in stop]


def _event_text_for_distance_match(event: dict) -> str:
    payload = event.get("payload") or {}
    parts = [
        str(payload.get("action_raw") or "").strip(),
        str(payload.get("observation_raw") or "").strip(),
        str(payload.get("step_body") or "").strip(),
        str(payload.get("tool_name") or "").strip(),
        str(payload.get("tool_args_raw") or "").strip(),
    ]
    return "\n".join(part for part in parts if part).lower()


def _format_relative_distance(
    distance: int,
    kind: str,
    step_value: int,
    *,
    include_step_label: bool = True,
) -> str:
    unit = {
        "step": "step",
        "turn": "turn",
        "message": "message",
    }.get(str(kind or "").lower(), "step")
    plural = unit if distance == 1 else f"{unit}s"
    if not include_step_label:
        return f"{distance} {plural} later."
    return f"{distance} {plural} later (Step {step_value})."


def _phrase_tokens(phrase: str) -> list[str]:
    stop = GENERIC_OBJECTIVE_STOP_WORDS | {
        "applying", "applied", "apply", "typing", "typed", "type",
        "leaving", "leave", "left", "clicking", "clicked", "click",
        "occur", "happen", "appear", "follow", "followed",
    }
    tokens = re.findall(r"[a-z0-9_]+", str(phrase or "").lower())
    seen: set[str] = set()
    ordered: list[str] = []
    for token in tokens:
        if token in stop:
            continue
        if len(token) == 1 and not token.isdigit():
            continue
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def _normalize_target_phrase(phrase: str) -> str:
    text = str(phrase or "").strip(" \t\r\n?.!")
    return re.sub(r"\b(?:occur|happen|appear|follow|followed)\b\s*$", "", text, flags=re.I).strip()


def _query_state_constraints(text: str) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    ordered: list[tuple[str, str]] = []
    for attr, value in STATE_CONSTRAINT_RE.findall(str(text or "")):
        pair = (str(attr).strip().lower(), str(value).strip().lower())
        if pair in seen:
            continue
        seen.add(pair)
        ordered.append(pair)
    return ordered


def _event_satisfies_state_constraints(event: dict, constraints: list[tuple[str, str]]) -> bool:
    if not constraints:
        return True
    payload = event.get("payload") or {}
    haystacks = [
        str(payload.get("action_raw") or "").lower(),
        str(payload.get("observation_raw") or "").lower(),
        str(payload.get("step_body") or "").lower(),
        str(payload.get("raw_step_block") or "").lower(),
    ]
    for attr, value in constraints:
        target = f"{attr}: {value}"
        if not any(target in hay for hay in haystacks if hay):
            return False
    return True


def _group_timeline_events_by_step(
    index: dict,
    *,
    kind: str,
    source_ids: set[str] | None = None,
    timeline_ids: set[str] | None = None,
) -> dict[str, dict[int, list[dict]]]:
    grouped: dict[str, dict[int, list[dict]]] = {}
    for timeline_id, timeline in (index.get("timelines") or {}).items():
        if timeline_ids is not None and str(timeline_id) not in timeline_ids:
            continue
        by_step: dict[int, list[dict]] = {}
        for event_id in timeline.get("ordered_event_ids") or []:
            event = index.get("events", {}).get(event_id) or {}
            if not _event_in_scope(event, source_ids=source_ids, timeline_ids=timeline_ids):
                continue
            if str(event.get("ordinal_kind") or "").strip().lower() != kind:
                continue
            value = event.get("ordinal_index")
            if not isinstance(value, int):
                continue
            by_step.setdefault(value, []).append(event)
        if by_step:
            grouped[str(timeline_id)] = by_step
    return grouped


def _term_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9_]+", str(text or "").lower())
    ordered: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if len(token) <= 1:
            continue
        for candidate in (token, token[:-1] if token.endswith("s") and len(token) > 3 else None):
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def _step_summary(events: list[dict]) -> dict:
    episode_ids: set[str] = set()
    texts: list[str] = []
    has_action = False
    ids: list[str] = []
    for event in events:
        payload = event.get("payload") or {}
        episode_id = str(payload.get("episode_id") or "").strip()
        if episode_id:
            episode_ids.add(episode_id)
        text = _event_objective_text(event)
        if text:
            texts.append(text)
        if str(payload.get("action_raw") or "").strip():
            has_action = True
        ids.extend(str(value).strip() for value in (payload.get("ids") or []) if str(value).strip())
    return {
        "text": "\n".join(texts),
        "episode_ids": frozenset(episode_ids),
        "has_action": has_action,
        "ids": ids,
    }


def _step_matches_module_terms(summary: dict, tokens: list[str]) -> bool:
    text = str(summary.get("text") or "")
    if not text:
        return False
    return any(token in text for token in tokens)


def _best_event_for_label(events: list[dict], label: str) -> dict | None:
    label_tokens = _term_tokens(label)
    ranked = sorted(
        events,
        key=lambda event: (
            -_event_objective_score(event, label_tokens, action_only=True),
            -_event_objective_score(event, label_tokens),
            -_ordinal_event_payload_quality(event)[0],
            str(event.get("event_id") or ""),
        ),
    )
    return ranked[0] if ranked else None


def _click_label(event: dict) -> str:
    payload = event.get("payload") or {}
    action_raw = str(payload.get("action_raw") or "").strip()
    click_match = CLICK_ACTION_CANON_RE.match(action_raw)
    if click_match:
        return str(click_match.group("label") or "").strip()
    return ""


def _is_checkbox_click(event: dict) -> bool:
    payload = event.get("payload") or {}
    action_raw = str(payload.get("action_raw") or "").strip().lower()
    return action_raw.startswith("click [") and " is checkbox " in action_raw


def _is_scroll_action(event: dict) -> bool:
    payload = event.get("payload") or {}
    action_raw = str(payload.get("action_raw") or "").strip().lower()
    return action_raw.startswith("scroll ")


def _is_malformed_action(event: dict) -> bool:
    payload = event.get("payload") or {}
    action_raw = str(payload.get("action_raw") or "").strip()
    if not action_raw:
        return False
    if MALFORMED_ACTION_PREFIX_RE.search(action_raw):
        return True
    if len(action_raw) > 180 and not (
        action_raw.startswith("click [")
        or action_raw.startswith("type [")
        or action_raw.startswith("scroll [")
        or action_raw.startswith("goto [")
        or ":" in action_raw
    ):
        return True
    return False


def _malformed_action_explanation(event: dict) -> str:
    payload = event.get("payload") or {}
    action_raw = str(payload.get("action_raw") or "").strip()
    if MALFORMED_ACTION_PREFIX_RE.search(action_raw):
        return "it used the wrong format by placing long reasoning text in the action field instead of a valid command/action"
    return "it used the wrong format instead of a valid command/action"


def _extract_click_id(event: dict) -> str | None:
    payload = event.get("payload") or {}
    ids = [str(value).strip() for value in (payload.get("ids") or []) if str(value).strip()]
    if ids:
        return ids[0]
    action_raw = str(payload.get("action_raw") or "").strip()
    click_match = CLICK_ACTION_CANON_RE.match(action_raw)
    if click_match:
        return str(click_match.group("id") or "").strip() or None
    return None


def _parse_grid_board(event: dict | None) -> list[list[str]] | None:
    if not isinstance(event, dict):
        return None
    payload = event.get("payload") or {}
    combined = "\n".join(
        part
        for part in (
            str(payload.get("observation_raw") or "").strip(),
            str(payload.get("step_body") or "").strip(),
            str(payload.get("raw_step_block") or "").strip(),
        )
        if part
    )
    rows: dict[int, list[str]] = {}
    for row_idx, row_text in GRID_BOARD_ROW_RE.findall(combined):
        try:
            idx = int(row_idx)
        except Exception:
            continue
        tokens = [token.strip() for token in str(row_text).split() if token.strip()]
        if tokens:
            rows[idx] = tokens
    if not rows:
        return None
    ordered = []
    expected = list(range(min(rows), max(rows) + 1))
    for idx in expected:
        if idx not in rows:
            return None
        ordered.append(rows[idx])
    width = len(ordered[0])
    if width <= 0 or any(len(row) != width for row in ordered):
        return None
    return ordered


def _find_previous_grid_board(index: dict, event: dict, *, max_delta: int = 4) -> tuple[dict | None, list[list[str]] | None]:
    for delta in range(1, max(1, int(max_delta)) + 1):
        candidate = lookup_adjacent_timeline_event(index, event=event, delta=-delta)
        board = _parse_grid_board(candidate)
        if board:
            return candidate, board
    return None, None


def _grid_run_coords(grid: list[list[str]], row: int, col: int, token: str) -> list[tuple[int, int]]:
    height = len(grid)
    width = len(grid[0]) if grid else 0
    if not (0 <= row < height and 0 <= col < width):
        return []
    best: list[tuple[int, int]] = []
    for dr, dc in ((1, 0), (0, 1)):
        start_r, start_c = row, col
        while 0 <= start_r - dr < height and 0 <= start_c - dc < width and grid[start_r - dr][start_c - dc] == token:
            start_r -= dr
            start_c -= dc
        coords: list[tuple[int, int]] = []
        cur_r, cur_c = start_r, start_c
        while 0 <= cur_r < height and 0 <= cur_c < width and grid[cur_r][cur_c] == token:
            coords.append((cur_r, cur_c))
            cur_r += dr
            cur_c += dc
        if len(coords) >= 3 and len(coords) > len(best):
            best = coords
    return best


def _click_label_with_id(event: dict) -> str | None:
    payload = event.get("payload") or {}
    action_raw = str(payload.get("action_raw") or "").strip()
    click_match = CLICK_ACTION_CANON_RE.match(action_raw)
    if not click_match:
        return None
    label = str(click_match.group("label") or "").strip()
    click_id = str(click_match.group("id") or "").strip()
    if label and click_id:
        return f"clicking '{label}' [{click_id}]"
    if label:
        return f"clicking '{label}'"
    return None


def _is_substantive_step(events: list[dict]) -> bool:
    return any(_ordinal_event_payload_quality(event)[0] > 0 for event in events)


def _substantive_distance(by_step: dict[int, list[dict]], *, anchor_step: int, target_step: int) -> int:
    substantive = [
        step_value
        for step_value in sorted(by_step)
        if step_value > anchor_step
        and step_value <= target_step
        and _is_substantive_step(by_step[step_value])
    ]
    if substantive:
        return len(substantive)
    return max(0, int(target_step) - int(anchor_step))


def _reduce_semantic_relative_distance(
    index: dict,
    plan: dict,
    *,
    source_ids: set[str] | None = None,
    timeline_ids: set[str] | None = None,
) -> dict:
    kind = str(plan.get("kind") or "").strip().lower()
    anchor_tokens = _phrase_tokens(str(plan.get("anchor_phrase") or ""))
    target_tokens = _phrase_tokens(_normalize_target_phrase(str(plan.get("target_phrase") or "")))
    if kind not in {"step", "turn", "message"} or not anchor_tokens or not target_tokens:
        return {"events": [], "event": None, "answer": None, "reducer": None}

    best: dict | None = None
    for timeline_id, by_step in _group_timeline_events_by_step(
        index,
        kind=kind,
        source_ids=source_ids,
        timeline_ids=timeline_ids,
    ).items():
        ordered_steps = sorted(by_step)
        anchor_candidates: list[tuple[int, int, int, dict]] = []
        for step_value in ordered_steps:
            action_event, action_score = _objective_ranked_event(by_step[step_value], anchor_tokens, action_only=True)
            if action_score > 0:
                event, score, action_matched = action_event, action_score, 1
            else:
                event, score = _objective_ranked_event(by_step[step_value], anchor_tokens)
                action_matched = 0
            if event and score > 0:
                anchor_candidates.append((action_matched, score, step_value, event))
        anchor_candidates.sort(key=lambda row: (-row[0], -row[1], row[2], str(row[3].get("event_id") or "")))
        for anchor_action_matched, anchor_score, anchor_step, anchor_event in anchor_candidates:
            target_event = None
            target_score = 0
            target_step = None
            target_action_matched = 0
            for step_value in ordered_steps:
                if step_value <= anchor_step:
                    continue
                action_event, action_score = _objective_ranked_event(by_step[step_value], target_tokens, action_only=True)
                if action_score > 0:
                    event, score, action_matched = action_event, action_score, 1
                else:
                    event, score = _objective_ranked_event(by_step[step_value], target_tokens)
                    action_matched = 0
                if event and score > 0:
                    target_event = event
                    target_score = score
                    target_step = step_value
                    target_action_matched = action_matched
                    break
            if target_event is None or target_step is None:
                continue
            candidate = {
                "timeline_id": timeline_id,
                "anchor_event": anchor_event,
                "target_event": target_event,
                "anchor_action_matched": anchor_action_matched,
                "target_action_matched": target_action_matched,
                "anchor_score": anchor_score,
                "target_score": target_score,
                "anchor_step": anchor_step,
                "target_step": target_step,
                "distance": _substantive_distance(
                    by_step,
                    anchor_step=int(anchor_step),
                    target_step=int(target_step),
                ),
            }
            if best is None:
                best = candidate
                continue
            candidate_key = (
                candidate["target_action_matched"],
                candidate["anchor_score"] + candidate["target_score"],
                -candidate["distance"],
                candidate["anchor_action_matched"],
                candidate["anchor_score"],
                candidate["target_score"],
                -candidate["target_step"],
            )
            best_key = (
                best["target_action_matched"],
                best["anchor_score"] + best["target_score"],
                -best["distance"],
                best["anchor_action_matched"],
                best["anchor_score"],
                best["target_score"],
                -best["target_step"],
            )
            if candidate_key > best_key:
                best = candidate
    if best is None:
        return {"events": [], "event": None, "answer": None, "reducer": None}
    return {
        "events": [best["anchor_event"], best["target_event"]],
        "event": best["target_event"],
        "answer": _format_relative_distance(
            int(best["distance"]),
            kind,
            int(best["target_step"]),
            include_step_label=False,
        ),
        "reducer": "relative_distance",
    }


def _reduce_first_occurrence(
    index: dict,
    plan: dict,
    *,
    source_ids: set[str] | None = None,
    timeline_ids: set[str] | None = None,
) -> dict:
    objective = str(plan.get("objective_phrase") or "").strip()
    tokens = _phrase_tokens(objective)
    constraints = _query_state_constraints(objective)
    if not tokens:
        return {"events": [], "event": None, "answer": None, "reducer": None}
    best: dict | None = None
    for sequence in _action_step_sequence(
        index,
        kind=str(plan.get("kind") or "step"),
        source_ids=source_ids,
        timeline_ids=timeline_ids,
    ):
        for event in sequence["events"]:
            if not _event_satisfies_state_constraints(event, constraints):
                continue
            action_score = _event_objective_score(event, tokens, action_only=True)
            total_score = _event_objective_score(event, tokens)
            if max(action_score, total_score) <= 0:
                continue
            value = event.get("ordinal_index")
            if not isinstance(value, int):
                continue
            candidate = {
                "event": event,
                "value": value,
                "action_score": action_score,
                "total_score": total_score,
                "action_matched": 1 if action_score > 0 else 0,
            }
            if best is None:
                best = candidate
                continue
            candidate_key = (
                candidate["action_matched"],
                -candidate["value"],
                candidate["action_score"],
                candidate["total_score"],
            )
            best_key = (
                best["action_matched"],
                -best["value"],
                best["action_score"],
                best["total_score"],
            )
            if candidate_key > best_key:
                best = candidate
    if best is None:
        return {"events": [], "event": None, "answer": None, "reducer": None}
    action = _extract_exact_answer_from_event("exact_action", best["event"])
    answer = f"Step {best['value']}"
    if action:
        answer += f", {action}."
    else:
        answer += "."
    return {
        "events": [best["event"]],
        "event": best["event"],
        "answer": answer,
        "reducer": "first_occurrence",
    }


def _reduce_consecutive_before_scroll(
    index: dict,
    plan: dict,
    *,
    source_ids: set[str] | None = None,
    timeline_ids: set[str] | None = None,
) -> dict:
    label = str(plan.get("label") or "").strip().lower()
    if not label:
        return {"events": [], "event": None, "answer": None, "reducer": None}
    best: dict | None = None
    for sequence in _action_step_sequence(
        index,
        kind=str(plan.get("kind") or "step"),
        source_ids=source_ids,
        timeline_ids=timeline_ids,
    ):
        events = sequence["events"]
        for idx, event in enumerate(events):
            action = _extract_exact_answer_from_event("exact_action", event) or ""
            if "click " not in action.lower() or label not in action.lower():
                continue
            start = idx
            end = idx
            while end + 1 < len(events):
                next_event = events[end + 1]
                next_action = _extract_exact_answer_from_event("exact_action", next_event) or ""
                if _is_scroll_action(next_event):
                    break
                if "click " not in next_action.lower() or label not in next_action.lower():
                    break
                end += 1
            if end + 1 >= len(events) or not _is_scroll_action(events[end + 1]):
                continue
            count = (end - start) + 1
            if count <= 0:
                continue
            candidate = {
                "events": events[start:end + 2],
                "event": events[start],
                "count": count,
                "start_value": events[start].get("ordinal_index"),
                "end_value": events[end].get("ordinal_index"),
            }
            if best is None or candidate["count"] > best["count"] or (
                candidate["count"] == best["count"]
                and int(candidate["start_value"] or 10**9) < int(best["start_value"] or 10**9)
            ):
                best = candidate
    if best is None:
        return {"events": [], "event": None, "answer": None, "reducer": None}
    return {
        "events": best["events"],
        "event": best["event"],
        "answer": f"{best['count']} consecutive clicks (Steps {best['start_value']}-{best['end_value']}).",
        "reducer": "consecutive_before_scroll",
    }


def _reduce_first_malformed_before_anchor(
    index: dict,
    plan: dict,
    *,
    source_ids: set[str] | None = None,
    timeline_ids: set[str] | None = None,
) -> dict:
    upper_bound = plan.get("upper_bound")
    if not isinstance(upper_bound, int):
        return {"events": [], "event": None, "answer": None, "reducer": None}
    best: dict | None = None
    for sequence in _action_step_sequence(
        index,
        kind=str(plan.get("kind") or "step"),
        source_ids=source_ids,
        timeline_ids=timeline_ids,
    ):
        for event in sequence["events"]:
            value = event.get("ordinal_index")
            if not isinstance(value, int) or value > upper_bound:
                continue
            if not _is_malformed_action(event):
                continue
            candidate = {"event": event, "value": value}
            if best is None or value < best["value"]:
                best = candidate
            break
    if best is None:
        return {"events": [], "event": None, "answer": None, "reducer": None}
    return {
        "events": [best["event"]],
        "event": best["event"],
        "answer": (
            f"Step {best['value']}. "
            f"The agent put long reasoning text into the action field instead of a valid command/action."
        ),
        "reducer": "first_malformed_before_anchor",
    }


def _reduce_loop_window(
    index: dict,
    plan: dict,
    *,
    source_ids: set[str] | None = None,
    timeline_ids: set[str] | None = None,
) -> dict:
    kind = str(plan.get("kind") or "").strip().lower()
    if kind != "step":
        return {"events": [], "event": None, "answer": None, "reducer": None}
    step_values = [int(value) for value in (plan.get("step_values") or []) if isinstance(value, int)]
    if len(step_values) < 2:
        return {"events": [], "event": None, "answer": None, "reducer": None}
    module_tokens: list[str] = []
    for term in (plan.get("module_terms") or []):
        module_tokens.extend(_term_tokens(term))
    module_tokens = list(dict.fromkeys(module_tokens))
    if not module_tokens:
        return {"events": [], "event": None, "answer": None, "reducer": None}

    best: dict | None = None
    grouped = _group_timeline_events_by_step(
        index,
        kind=kind,
        source_ids=source_ids,
        timeline_ids=timeline_ids,
    )
    for timeline_id, by_step in grouped.items():
        if not all(value in by_step for value in step_values):
            continue
        summaries = {step: _step_summary(events) for step, events in by_step.items()}
        matching_steps = {
            step for step, summary in summaries.items()
            if _step_matches_module_terms(summary, module_tokens)
        }
        if not matching_steps:
            continue

        lower_ref = min(step_values)
        upper_ref = max(step_values)
        prev_cut = max(
            (
                step
                for step, summary in summaries.items()
                if step < lower_ref and step not in matching_steps and summary["has_action"]
            ),
            default=None,
        )
        next_cut = min(
            (
                step
                for step, summary in summaries.items()
                if step > upper_ref and step not in matching_steps and summary["has_action"]
            ),
            default=None,
        )
        start_candidates = [
            step
            for step in matching_steps
            if step <= lower_ref and (prev_cut is None or step > prev_cut)
        ]
        end_candidates = [
            step
            for step in matching_steps
            if step >= upper_ref and (next_cut is None or step < next_cut)
        ]
        start = min(start_candidates) if start_candidates else lower_ref
        end = max(end_candidates) if end_candidates else upper_ref

        while (
            (start + 1) <= end
            and start in summaries
            and (start + 1) in summaries
            and not summaries[start]["has_action"]
            and not summaries[start + 1]["has_action"]
            and summaries[start]["episode_ids"]
            and summaries[start]["episode_ids"] == summaries[start + 1]["episode_ids"]
        ):
            start += 1

        ref_events: list[dict] = []
        id_answers: list[dict] = []
        id_refs = list(plan.get("id_refs") or [])
        for ref in id_refs:
            label = str(ref.get("label") or "").strip()
            step = ref.get("step")
            if not label or not isinstance(step, int):
                continue
            events = by_step.get(step) or []
            if not events:
                continue
            best_event = _best_event_for_label(events, label)
            if not best_event:
                continue
            ref_events.append(best_event)
            click_id = _extract_click_id(best_event)
            if click_id:
                id_answers.append({"label": label, "step": step, "id": click_id})

        if len(id_answers) < len(id_refs):
            continue

        candidate = {
            "timeline_id": timeline_id,
            "start": start,
            "end": end,
            "events": ref_events,
            "id_answers": id_answers,
        }
        if best is None:
            best = candidate
            continue
        candidate_span = candidate["end"] - candidate["start"]
        best_span = best["end"] - best["start"]
        candidate_key = (len(candidate["id_answers"]), -candidate_span, -candidate["start"])
        best_key = (len(best["id_answers"]), -best_span, -best["start"])
        if candidate_key > best_key:
            best = candidate

    if best is None:
        return {"events": [], "event": None, "answer": None, "reducer": None}

    parts = [f"The loop occurs between Step {best['start']} and Step {best['end']}."]
    for ref in best["id_answers"]:
        parts.append(
            f"The '{ref['label']}' link clicked in Step {ref['step']} has the element ID [{ref['id']}]."
        )
    return {
        "events": best["events"],
        "event": best["events"][0] if best["events"] else None,
        "answer": " ".join(parts),
        "reducer": "loop_window",
    }


def _reduce_relative_distance(question: str, index: dict, event: dict, *, plan: dict | None = None) -> dict:
    if not RELATIVE_DISTANCE_RE.search(question):
        return {"event": None, "answer": None, "reducer": None}
    if not isinstance(plan, dict) or plan.get("mode") != "exact":
        return {"event": None, "answer": None, "reducer": None}
    anchor_value = plan.get("value")
    kind = str(plan.get("kind") or event.get("ordinal_kind") or "").strip().lower()
    if not isinstance(anchor_value, int) or not kind:
        return {"event": None, "answer": None, "reducer": None}
    timeline_id = str(event.get("timeline_id") or "").strip()
    if not timeline_id:
        return {"event": None, "answer": None, "reducer": None}
    target_phrase = _extract_distance_target_phrase(question)
    target_tokens = _tokenize_distance_target(target_phrase)
    if not target_tokens:
        return {"event": None, "answer": None, "reducer": None}

    timeline = index.get("timelines", {}).get(timeline_id) or {}
    ordered = list(timeline.get("ordered_event_ids") or [])
    by_step: dict[int, list[dict]] = {}
    for event_id in ordered:
        candidate = index.get("events", {}).get(event_id) or {}
        if str(candidate.get("ordinal_kind") or "").strip().lower() != kind:
            continue
        value = candidate.get("ordinal_index")
        if not isinstance(value, int) or value <= anchor_value:
            continue
        by_step.setdefault(value, []).append(candidate)

    for step_value in sorted(by_step):
        ranked = sorted(
            by_step[step_value],
            key=lambda candidate: (
                -_ordinal_event_payload_quality(candidate)[0],
                -_ordinal_event_payload_quality(candidate)[1],
                str(candidate.get("event_id") or ""),
            ),
        )
        for candidate in ranked:
            text = _event_text_for_distance_match(candidate)
            if all(token in text for token in target_tokens):
                distance = step_value - anchor_value
                return {
                    "event": candidate,
                    "answer": _format_relative_distance(distance, kind, step_value),
                    "reducer": "relative_distance",
                }
    return {"event": None, "answer": None, "reducer": None}


def _reduce_grid_swap_match(question: str, index: dict, event: dict, *, plan: dict | None = None) -> dict:
    if not isinstance(plan, dict) or plan.get("mode") != "exact":
        return {"event": None, "answer": None, "reducer": None}
    text = str(question or "")
    if not (re.search(r"\bboard state\b", text, re.I) and re.search(r"\bcoordinates?\b", text, re.I)):
        return {"event": None, "answer": None, "reducer": None}
    payload = event.get("payload") or {}
    action_raw = str(payload.get("action_raw") or "").strip()
    action_match = GRID_SWAP_ACTION_RE.match(action_raw)
    if not action_match:
        return {"event": None, "answer": None, "reducer": None}
    previous_event, board = _find_previous_grid_board(index, event)
    if not board:
        return {"event": None, "answer": None, "reducer": None}

    r1, c1, r2, c2 = [int(value) for value in action_match.groups()]
    height = len(board)
    width = len(board[0]) if board else 0
    if not all(0 <= value < limit for value, limit in ((r1, height), (r2, height), (c1, width), (c2, width))):
        return {"event": None, "answer": None, "reducer": None}

    src_a = board[r1][c1]
    src_b = board[r2][c2]
    swapped = [list(row) for row in board]
    swapped[r1][c1], swapped[r2][c2] = swapped[r2][c2], swapped[r1][c1]

    candidates = []
    for token, from_coord, to_coord in (
        (src_b, (r2, c2), (r1, c1)),
        (src_a, (r1, c1), (r2, c2)),
    ):
        run = _grid_run_coords(swapped, to_coord[0], to_coord[1], token)
        if run:
            candidates.append((len(run), token, from_coord, to_coord, run))
    if not candidates:
        return {"event": None, "answer": None, "reducer": None}

    _len, moved_token, from_coord, to_coord, run = max(candidates, key=lambda row: row[0])
    stationary_token = src_a if moved_token == src_b else src_b
    run_text = ", ".join(f"({row},{col})" for row, col in run)
    answer = (
        f"The '{moved_token}' candy from ({from_coord[0]},{from_coord[1]}) was moved into "
        f"({to_coord[0]},{to_coord[1]}), swapping with the '{stationary_token}' candy that had been there. "
        f"This created a match at coordinates {run_text}."
    )
    return {
        "event": event,
        "answer": answer,
        "reducer": "grid_swap_match",
    }


def _format_editor_view_action(tool_args: dict) -> str | None:
    path = str(tool_args.get("path") or "").strip()
    if not path:
        return None
    view_range = tool_args.get("view_range")
    if isinstance(view_range, (list, tuple)) and len(view_range) == 2:
        try:
            start = int(view_range[0])
            end = int(view_range[1])
        except Exception:
            start = end = None
        if start is not None and end is not None:
            return f"Used str_replace_editor to view lines {start}-{end} of {path}"
    return f"Used str_replace_editor to view {path}"


def _summarize_str_replace_delta(tool_args: dict) -> str | None:
    path = str(tool_args.get("path") or "").strip()
    old_str = str(tool_args.get("old_str") or "")
    new_str = str(tool_args.get("new_str") or "")
    if not path:
        return None
    old_literals = set(re.findall(r"'([^']+)'|\"([^\"]+)\"", old_str))
    new_literals = set(re.findall(r"'([^']+)'|\"([^\"]+)\"", new_str))
    old_tokens = {a or b for a, b in old_literals if (a or b)}
    new_tokens = {a or b for a, b in new_literals if (a or b)}
    added_tokens = [token for token in sorted(new_tokens) if token and token not in old_tokens]
    if added_tokens:
        return f"Added '{added_tokens[0]}' to {path}"
    old_lines = {line.strip() for line in old_str.splitlines() if line.strip()}
    new_lines = [line.strip() for line in new_str.splitlines() if line.strip() and line.strip() not in old_lines]
    if new_lines:
        return f"Updated {path} with: {new_lines[0][:120]}"
    return f"Used str_replace_editor to modify {path}"


def _format_editor_action(tool_args: dict) -> str | None:
    command = str(tool_args.get("command") or "").strip().lower()
    path = str(tool_args.get("path") or "").strip()
    if command == "view":
        return _format_editor_view_action(tool_args)
    if command == "create" and path:
        return f"Created {path}"
    if command == "insert" and path:
        return f"Inserted code into {path}"
    if command == "str_replace":
        return _summarize_str_replace_delta(tool_args)
    if command and path:
        return f"Used str_replace_editor to {command} {path}"
    return None


def _extract_numeric_result_text(event: dict | None) -> str | None:
    if not isinstance(event, dict):
        return None
    payload = event.get("payload") or {}
    combined = "\n".join(
        part
        for part in (
            str(payload.get("observation_raw") or "").strip(),
            str(payload.get("step_body") or "").strip(),
            str(payload.get("raw_step_block") or "").strip(),
        )
        if part
    )
    row_match = RESULT_ROW_RE.search(combined)
    if row_match:
        label = str(row_match.group(1) or "").strip()
        number = str(row_match.group(2) or "").strip()
        if label and number:
            return f"{label}, {number}"
    ids = [str(value).strip() for value in (payload.get("ids") or []) if str(value).strip()]
    numeric_ids = [value for value in ids if re.fullmatch(r"[0-9][0-9,]*", value)]
    if numeric_ids:
        return numeric_ids[0]
    return None


def _extract_status_text(event: dict | None) -> str | None:
    if not isinstance(event, dict):
        return None
    payload = event.get("payload") or {}
    combined = "\n".join(
        part
        for part in (
            str(payload.get("observation_raw") or "").strip(),
            str(payload.get("step_body") or "").strip(),
            str(payload.get("raw_step_block") or "").strip(),
        )
        if part
    )
    client_match = CLIENT_ERROR_RE.search(combined)
    if client_match:
        return f"Execution failed with a {client_match.group(1)} Client Error ({client_match.group(2).strip()})."
    server_match = SERVER_ERROR_RE.search(combined)
    if server_match:
        return f"Execution failed with a {server_match.group(1)} Server Error ({server_match.group(2).strip()})."
    if re.search(r"\bquery executed successfully\b", combined, re.I):
        return "Query executed successfully."
    status_match = re.search(r"\bstep_status\s+is\s+([A-Za-z_ -]+)", combined, re.I)
    if status_match:
        return status_match.group(1).strip().rstrip("].,")
    post_match = re.search(r"\bstatus set to\s+([A-Za-z_ -]+)", combined, re.I)
    if post_match:
        return post_match.group(1).strip().rstrip("].,")
    return None


def _infer_missing_required_input(event: dict | None) -> str | None:
    if not isinstance(event, dict):
        return None
    payload = event.get("payload") or {}
    tool_name = str(payload.get("tool_name") or payload.get("action_name") or "").strip().lower()
    if not tool_name:
        return None
    tool_args = payload.get("tool_args")
    tool_args_raw = str(payload.get("tool_args_raw") or "").strip()
    if isinstance(tool_args, dict) and tool_args:
        return None
    if tool_args_raw:
        return None
    return REQUIRED_TOOL_ARGS.get(tool_name)


def _select_result_event(events: list[dict], primary: dict | None) -> dict | None:
    candidates = []
    for event in events:
        if primary and event.get("event_id") == primary.get("event_id"):
            continue
        payload = event.get("payload") or {}
        if str(payload.get("action_raw") or "").strip():
            continue
        numeric = _extract_numeric_result_text(event)
        if not numeric:
            continue
        candidates.append(event)
    return _best_ordinal_event(candidates) if candidates else None


def _extract_exact_answer_from_event(reducer: str, event: dict) -> str | None:
    payload = event.get("payload") or {}
    tool_args = payload.get("tool_args") or {}
    tool_args_raw = str(payload.get("tool_args_raw") or "").strip()
    action_raw = str(payload.get("action_raw") or "").strip()
    tool_name = str(payload.get("tool_name") or payload.get("action_name") or "").strip()
    paths = [str(path).strip() for path in (payload.get("paths") or []) if str(path).strip()]
    ids = [str(value).strip() for value in (payload.get("ids") or []) if str(value).strip()]

    if reducer == "exact_tool":
        return tool_name or None
    if reducer == "exact_status":
        return _extract_status_text(event)
    if reducer in {"exact_sql", "exact_command"}:
        preferred_keys = ("sql", "command", "query", "value")
        if isinstance(tool_args, dict):
            for key in preferred_keys:
                value = str(tool_args.get(key) or "").strip()
                if value:
                    if reducer == "exact_command":
                        return _canonicalize_shell_command_text(value) or value
                    return value
        if tool_args_raw:
            if reducer == "exact_command":
                return _canonicalize_shell_command_text(tool_args_raw) or tool_args_raw
            return tool_args_raw
        return action_raw or None
    if reducer == "exact_action":
        if str(tool_name).strip().lower() in {"execute_bash", "execute_snowflake_sql"} and isinstance(tool_args, dict):
            for key in ("command", "sql"):
                value = str(tool_args.get(key) or "").strip()
                if value:
                    return value
        if str(tool_name).strip().lower() == "str_replace_editor" and isinstance(tool_args, dict):
            formatted = _format_editor_action(tool_args)
            if formatted:
                return formatted
        canonical = _canonicalize_action_text(action_raw)
        return canonical or tool_name or None
    if reducer == "exact_path":
        return paths[0] if paths else None
    if reducer == "exact_id":
        if len(ids) == 1:
            return ids[0]
        if ids:
            return ", ".join(ids)
        return None
    return None


def _best_event_per_ordinal(events: list[dict]) -> list[dict]:
    by_step: dict[int, list[dict]] = {}
    for event in events:
        value = event.get("ordinal_index")
        if not isinstance(value, int):
            continue
        by_step.setdefault(value, []).append(event)
    selected: list[dict] = []
    for value in sorted(by_step):
        best = _best_ordinal_event(by_step[value])
        if best:
            selected.append(best)
    return selected


def _environment_delta_lines(action: str) -> list[str]:
    text = str(action or "").strip()
    if not text:
        return []
    take_match = TAKE_FROM_RE.match(text)
    if take_match:
        item = str(take_match.group(1) or "").strip()
        source = str(take_match.group(2) or "").strip()
        if item and source:
            return [
                f"{item} added to inventory",
                f"{item} moved from {source} to inventory",
            ]
    move_match = MOVE_TO_RE.match(text)
    if move_match:
        item = str(move_match.group(1) or "").strip()
        dest = str(move_match.group(2) or "").strip()
        if item and dest:
            return [
                f"{item} removed from inventory",
                f"{item} moved from inventory to {dest}",
            ]
    put_match = PUT_INTO_RE.match(text)
    if put_match:
        item = str(put_match.group(1) or "").strip()
        dest = str(put_match.group(2) or "").strip()
        if item and dest:
            return [
                f"{item} removed from inventory",
                f"{item} moved from inventory to {dest}",
            ]
    drop_match = DROP_RE.match(text)
    if drop_match:
        item = str(drop_match.group(1) or "").strip()
        if item:
            return [
                f"{item} removed from inventory",
                f"{item} dropped into the environment",
            ]
    return []


def _reduce_ordinal_range(question: str, events: list[dict], *, plan: dict | None = None) -> dict:
    if not isinstance(plan, dict) or plan.get("mode") != "range":
        return {"events": [], "event": None, "answer": None, "reducer": None}
    kind = str(plan.get("kind") or "").strip().lower()
    if kind not in {"step", "turn", "message"}:
        return {"events": [], "event": None, "answer": None, "reducer": None}
    ordered = _best_event_per_ordinal(events)
    if not ordered:
        return {"events": [], "event": None, "answer": None, "reducer": None}
    if CHECKBOX_ORDER_RANGE_RE.search(question):
        checkbox_events = [event for event in ordered if _is_checkbox_click(event)]
        if len(checkbox_events) >= 2:
            first, second = checkbox_events[0], checkbox_events[1]
            first_id = _extract_click_id(first)
            second_id = _extract_click_id(second)
            if first_id and second_id:
                return {
                    "events": checkbox_events[:2],
                    "event": checkbox_events[0],
                    "answer": f"Checkbox [{first_id}] was clicked first, then checkbox [{second_id}].",
                    "reducer": "checkbox_order",
                }
    if INVALID_ACTION_RANGE_RE.search(question):
        malformed = [event for event in ordered if _is_malformed_action(event)]
        if malformed:
            bad = malformed[0]
            restored = next(
                (
                    event
                    for event in ordered
                    if isinstance(event.get("ordinal_index"), int)
                    and int(event["ordinal_index"]) > int(bad.get("ordinal_index") or -1)
                    and not _is_malformed_action(event)
                    and str(_extract_exact_answer_from_event("exact_action", event) or "").strip()
                ),
                None,
            )
            if restored and isinstance(bad.get("ordinal_index"), int) and isinstance(restored.get("ordinal_index"), int):
                restored_action = _click_label_with_id(restored) or _extract_exact_answer_from_event("exact_action", restored) or "restored valid navigation"
                return {
                    "events": [bad, restored],
                    "event": bad,
                    "answer": (
                        f"Step {bad['ordinal_index']} was the invalid action: {_malformed_action_explanation(bad)}, so no state change occurred. "
                        f"Step {restored['ordinal_index']} restored valid navigation by {restored_action}."
                    ),
                    "reducer": "invalid_action_window",
                }
    if ENVIRONMENT_CHANGE_RANGE_RE.search(question):
        parts: list[str] = []
        changed_events: list[dict] = []
        for event in ordered:
            action = _extract_exact_answer_from_event("exact_action", event)
            step_value = event.get("ordinal_index")
            if not action or not isinstance(step_value, int):
                continue
            delta_lines = _environment_delta_lines(action)
            if not delta_lines:
                continue
            changed_events.append(event)
            parts.append(
                f"{kind} {step_value}: '{action}' caused [{'; '.join(delta_lines)}]"
            )
        if parts:
            return {
                "events": changed_events,
                "event": changed_events[0] if changed_events else None,
                "answer": "Action-environment changes: " + " | ".join(parts),
                "reducer": "environment_changes",
            }
    if not (ACTION_LIST_RANGE_RE.search(question) or MOVEMENT_ACTION_LIST_RE.search(question)):
        return {"events": [], "event": None, "answer": None, "reducer": None}

    action_events = list(ordered)
    if BETWEEN_BOUNDARY_STEPS_RE.search(question):
        interior = [
            event
            for event in ordered
            if isinstance(event.get("ordinal_index"), int)
            and int(event["ordinal_index"]) > int(plan.get("start") or -10**9)
            and int(event["ordinal_index"]) < int(plan.get("end") or 10**9)
        ]
        if interior:
            action_events = interior

    parts: list[str] = []
    for event in action_events:
        action = _extract_exact_answer_from_event("exact_action", event)
        step_value = event.get("ordinal_index")
        if not action or not isinstance(step_value, int):
            continue
        parts.append(f"at {kind} {step_value}, {action}")
    if not parts:
        return {"events": [], "event": None, "answer": None, "reducer": None}

    start = plan.get("start")
    end = plan.get("end")
    prefix = f"Between {kind} {start} and {kind} {end}, the agent performed the following actions: "
    return {
        "events": action_events,
        "event": action_events[0],
        "answer": prefix + "; ".join(parts) + ".",
        "reducer": "range_actions",
    }


def _reduce_ordinal_events(
    question: str,
    index: dict,
    events: list[dict],
    *,
    plan: dict | None = None,
) -> dict:
    if not events or not isinstance(plan, dict) or plan.get("mode") != "exact":
        return {"event": None, "answer": None, "reducer": None}
    reducer = _select_reducer(question)
    primary = _select_objective_event(question, index, events, plan=plan, reducer=reducer)
    if not primary:
        return {"event": None, "answer": None, "reducer": None}
    distance_reduction = _reduce_relative_distance(question, index, primary, plan=plan)
    if distance_reduction.get("answer"):
        return distance_reduction
    grid_reduction = _reduce_grid_swap_match(question, index, primary, plan=plan)
    if grid_reduction.get("answer"):
        return grid_reduction
    if reducer is None:
        return {"event": None, "answer": None, "reducer": None}
    if reducer == "exact_sql_result":
        answer_event = _event_answer_target(question, index, primary, plan=plan)
        sql = _extract_exact_answer_from_event("exact_sql", answer_event)
        result_event = _select_result_event(events, answer_event)
        numeric_result = _extract_numeric_result_text(result_event)
        if sql and numeric_result:
            return {
                "event": answer_event,
                "answer": f"{sql}\n\nObserved result: {numeric_result}.",
                "reducer": reducer,
            }
        if sql:
            return {"event": answer_event, "answer": sql, "reducer": "exact_sql"}
        return {"event": None, "answer": None, "reducer": None}
    if reducer == "tool_missing_input":
        answer_event = _event_answer_target(question, index, primary, plan=plan)
        tool = _extract_exact_answer_from_event("exact_tool", answer_event)
        missing = _infer_missing_required_input(answer_event)
        if tool and missing:
            return {
                "event": answer_event,
                "answer": f"The agent switched to {tool}; the required {missing} input was missing.",
                "reducer": reducer,
            }
        if tool:
            return {"event": answer_event, "answer": tool, "reducer": "exact_tool"}
        return {"event": None, "answer": None, "reducer": None}
    answer_event = _event_answer_target(question, index, primary, plan=plan)
    answer = _extract_exact_answer_from_event(reducer, answer_event)
    return {
        "event": answer_event if answer else None,
        "answer": answer,
        "reducer": reducer if answer else None,
    }
