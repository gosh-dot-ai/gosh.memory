"""Unified extraction substrate validation and persistence helpers."""

from __future__ import annotations

import copy
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

EPISODE_EXTRACTION_BUDGET = 6
SOURCE_AGGREGATION_BUDGET = 3
LAYER_ORDER = [
    "atomic_fact_layer",
    "locality_layer",
    "revision_currentness_layer",
    "event_layer",
    "record_layer",
    "edge_layer",
]
SEMANTIC_EDGE_TYPES = {
    "causes",
    "leads_to",
    "supports",
    "conflicts_with",
    "bridge_to",
    "resolver_for",
}


class SubstrateValidationError(ValueError):
    """Structured validation error used internally and by strict validators."""

    def __init__(self, code: str, layer: str, kind: str, message: str):
        super().__init__(message)
        self.code = code
        self.layer = layer
        self.kind = kind


def _raise_validation(code: str, layer: str, kind: str, message: str) -> None:
    raise SubstrateValidationError(code, layer, kind, message)


def require_dict(value: Any, field: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{field}: expected object")
    return value


def require_list(value: Any, field: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{field}: expected list")
    return value


def require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}: expected non-empty string")
    return value


def require_optional_str(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return require_str(value, field)


def require_number(value: Any, field: str) -> int | float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field}: expected int or float")
    return value


def require_optional_number(value: Any, field: str) -> int | float | None:
    if value is None:
        return None
    return require_number(value, field)


def require_float_01(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field}: expected float in [0, 1]")
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{field}: expected float in [0, 1]")
    return value


def require_enum(value: Any, field: str, allowed: set[str]) -> str:
    value = require_str(value, field)
    if value not in allowed:
        raise ValueError(f"{field}: expected one of {sorted(allowed)}, got {value!r}")
    return value


def require_list_of_str(value: Any, field: str) -> list[str]:
    value = require_list(value, field)
    out: list[str] = []
    for i, item in enumerate(value):
        out.append(require_str(item, f"{field}[{i}]"))
    return out


def require_optional_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field}: expected int")
    return value


def require_optional_date(value: Any, field: str) -> str | None:
    if value is None:
        return None
    value = require_str(value, field)
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field}: expected ISO date YYYY-MM-DD") from exc
    return value


def normalize_ws(text: str) -> str:
    return " ".join(str(text).split())


def text_contains_value(text: str, value: str) -> bool:
    return normalize_ws(value).lower() in normalize_ws(text).lower()


def require_span_grounded(source_text: str, span: str, field: str) -> str:
    span = require_str(span, field)
    if normalize_ws(span) not in normalize_ws(source_text):
        raise ValueError(f"{field}: source_span not found in source_text")
    return span


def require_span_offsets(source_text: str, span: str, start: int | None, end: int | None, field: str) -> None:
    if start is None or end is None:
        return
    if start < 0 or end < start or end > len(source_text):
        raise ValueError(f"{field}: invalid source span offsets")
    extracted = source_text[start:end]
    if normalize_ws(extracted) != normalize_ws(span):
        raise ValueError(f"{field}: source_span does not match source_span_start/source_span_end")


def extract_date_candidates(text: str) -> set[str]:
    text = normalize_ws(text)
    candidates: set[str] = set()
    month_map = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }
    for match in re.finditer(r"\b(\d{4}-\d{2}-\d{2})\b", text):
        candidates.add(match.group(1))
    for match in re.finditer(r"\b([A-Za-z]+)\.?\s+(\d{1,2}),\s+(\d{4})\b", text):
        month_name, day_s, year_s = match.groups()
        month = month_map.get(month_name.lower())
        if month is None:
            continue
        candidates.add(f"{int(year_s):04d}-{month:02d}-{int(day_s):02d}")
    return candidates


def require_grounded_date(value: str | None, support_text: str, field: str) -> str | None:
    value = require_optional_date(value, field)
    if value is None:
        return None
    if value not in extract_date_candidates(support_text):
        raise ValueError(f"{field}: date {value!r} not grounded in support text")
    return value


def extract_number_unit_candidates(text: str) -> set[tuple[float, str | None]]:
    text = normalize_ws(text)
    out: set[tuple[float, str | None]] = set()
    for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*([A-Za-z%]+)?\b", text):
        num_s, unit = match.groups()
        out.add((float(num_s), unit.lower() if unit else None))
    return out


def require_grounded_number_unit(
    number: int | float | None,
    unit: str | None,
    support_text: str,
    number_field: str,
    unit_field: str,
) -> None:
    if number is None:
        return
    candidates = extract_number_unit_candidates(support_text)
    normalized_unit = unit.lower() if unit else None
    if (float(number), normalized_unit) not in candidates:
        raise ValueError(
            f"{number_field}/{unit_field}: numeric value {(float(number), normalized_unit)!r} not grounded in support text"
        )


def require_grounded_string(value: str | None, support_text: str, field: str) -> str | None:
    value = require_optional_str(value, field)
    if value is None:
        return None
    if not text_contains_value(support_text, value):
        raise ValueError(f"{field}: value {value!r} not grounded in support text")
    return value


def _stem_token(token: str) -> str:
    token = token.lower()
    irregular = {
        "lost": "lose",
        "losing": "lose",
        "lose": "lose",
        "jobs": "job",
        "businesses": "business",
    }
    if token in irregular:
        return irregular[token]
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    return token


def require_grounded_anchor_key(value: str | None, support_text: str, field: str) -> str:
    value = require_str(value, field)
    support_tokens = {_stem_token(tok) for tok in re.findall(r"[A-Za-z0-9]+", support_text)}
    key_tokens = [_stem_token(tok) for tok in re.findall(r"[A-Za-z0-9]+", value)]
    if not key_tokens:
        raise ValueError(f"{field}: expected non-empty string")
    for token in key_tokens:
        if not any(token == st or token[:3] == st[:3] for st in support_tokens if st):
            raise ValueError(f"{field}: value {value!r} not grounded in support text")
    return value


def require_id_ref(value: Any, field: str, allowed_ids: set[str]) -> str:
    value = require_str(value, field)
    if value not in allowed_ids:
        raise ValueError(f"{field}: unknown id reference {value!r}")
    return value


def require_id_ref_list(value: Any, field: str, allowed_ids: set[str]) -> list[str]:
    ids = require_list_of_str(value, field)
    for i, item in enumerate(ids):
        if item not in allowed_ids:
            raise ValueError(f"{field}[{i}]: unknown id reference {item!r}")
    return ids


def require_unique_ids(items: list[dict], id_field: str, field: str) -> None:
    seen: set[str] = set()
    for i, item in enumerate(items):
        value = require_str(item.get(id_field), f"{field}[{i}].{id_field}")
        if value in seen:
            raise ValueError(f"{field}[{i}].{id_field}: duplicate id {value!r}")
        seen.add(value)


def require_disjoint_id_sets(*named_sets: tuple[str, set[str]]) -> None:
    merged: dict[str, str] = {}
    for label, ids in named_sets:
        for value in ids:
            if value in merged:
                raise ValueError(f"id collision: {value!r} appears in both {merged[value]!r} and {label!r}")
            merged[value] = label


def validate_uniqueness(payload: dict) -> None:
    try:
        require_unique_ids(payload["atomic_facts"], "fact_id", "atomic_facts")
        require_unique_ids(payload["revision_currentness"], "revision_id", "revision_currentness")
        require_unique_ids(payload["events"], "event_id", "events")
        require_unique_ids(payload["records"], "record_id", "records")
        require_unique_ids(payload["edges"], "edge_id", "edges")
        fact_ids = {item["fact_id"] for item in payload["atomic_facts"]}
        event_ids = {item["event_id"] for item in payload["events"]}
        record_ids = {item["record_id"] for item in payload["records"]}
        require_disjoint_id_sets(
            ("atomic_facts", fact_ids),
            ("events", event_ids),
            ("records", record_ids),
        )
    except ValueError as exc:
        code = "cross_layer_id_collision" if "id collision" in str(exc) else "duplicate_ids"
        _raise_validation(code, "top_level", "schema", str(exc))


def validate_envelope(payload: dict) -> dict:
    try:
        payload = require_dict(payload, "payload")
        require_str(payload.get("schema"), "schema")
        if payload["schema"] != "extraction_substrate":
            raise ValueError("schema: unsupported schema")
        require_str(payload.get("source_id"), "source_id")
        require_str(payload.get("episode_id"), "episode_id")
        require_enum(payload.get("source_kind"), "source_kind", {"document_episode", "conversation_episode"})
        require_list(payload.get("atomic_facts"), "atomic_facts")
        require_dict(payload.get("locality"), "locality")
        require_list(payload.get("revision_currentness"), "revision_currentness")
        require_list(payload.get("events"), "events")
        require_list(payload.get("records"), "records")
        require_list(payload.get("edges"), "edges")
        return payload
    except ValueError as exc:
        _raise_validation("invalid_top_level_schema", "top_level", "schema", str(exc))


def validate_source_aggregation_envelope(payload: dict) -> dict:
    try:
        payload = require_dict(payload, "payload")
        require_str(payload.get("schema"), "schema")
        if payload["schema"] != "extraction_substrate":
            raise ValueError("schema: unsupported schema")
        require_enum(payload.get("payload_scope"), "payload_scope", {"source_aggregation"})
        require_str(payload.get("source_id"), "source_id")
        require_enum(payload.get("source_kind"), "source_kind", {"document", "conversation"})
        episode_ids = require_list_of_str(payload.get("episode_ids"), "episode_ids")
        locality_by_episode = require_dict(payload.get("locality_by_episode"), "locality_by_episode")
        for episode_id in episode_ids:
            if episode_id not in locality_by_episode:
                raise ValueError(f"locality_by_episode: missing locality object for {episode_id!r}")
            require_dict(locality_by_episode[episode_id], f"locality_by_episode[{episode_id!r}]")
        require_list(payload.get("atomic_facts"), "atomic_facts")
        require_list(payload.get("revision_currentness"), "revision_currentness")
        require_list(payload.get("events"), "events")
        require_list(payload.get("records"), "records")
        require_list(payload.get("edges"), "edges")
        return payload
    except ValueError as exc:
        _raise_validation("invalid_source_aggregation_schema", "top_level", "schema", str(exc))


def validate_source_aggregation_non_empty(payload: dict) -> dict:
    higher_order_count = (
        len(payload.get("revision_currentness", []) or [])
        + len(payload.get("events", []) or [])
        + len(payload.get("records", []) or [])
        + len(payload.get("edges", []) or [])
    )
    if higher_order_count <= 0:
        _raise_validation(
            "empty_source_aggregation",
            "top_level",
            "schema",
            "source_aggregation: expected at least one revision_currentness/event/record/edge object",
        )
    return payload


def validate_atomic_fact(obj: dict) -> dict:
    try:
        obj = require_dict(obj, "atomic_fact")
        require_str(obj.get("fact_id"), "fact_id")
        require_str(obj.get("subject"), "subject")
        require_str(obj.get("relation"), "relation")
        require_str(obj.get("object"), "object")
        require_optional_str(obj.get("value_text"), "value_text")
        require_optional_number(obj.get("value_number"), "value_number")
        require_optional_str(obj.get("value_unit"), "value_unit")
        require_enum(obj.get("polarity"), "polarity", {"positive", "negative", "uncertain"})
        require_float_01(obj.get("confidence"), "confidence")
        require_str(obj.get("source_span"), "source_span")
        require_optional_int(obj.get("source_span_start"), "source_span_start")
        require_optional_int(obj.get("source_span_end"), "source_span_end")
        require_optional_date(obj.get("asserted_at"), "asserted_at")
        require_list_of_str(obj.get("entity_ids"), "entity_ids")
        return obj
    except ValueError as exc:
        _raise_validation("invalid_atomic_fact_schema", "atomic_fact_layer", "schema", str(exc))


def validate_locality(obj: dict, source_id: str, episode_id: str) -> dict:
    try:
        obj = require_dict(obj, "locality")
        if require_str(obj.get("source_id"), "locality.source_id") != source_id:
            raise ValueError("locality.source_id: must match envelope source_id")
        if require_str(obj.get("episode_id"), "locality.episode_id") != episode_id:
            raise ValueError("locality.episode_id: must match envelope episode_id")
        require_optional_str(obj.get("section_id"), "locality.section_id")
        require_optional_str(obj.get("heading"), "locality.heading")
        require_optional_str(obj.get("table_id"), "locality.table_id")
        require_optional_str(obj.get("list_id"), "locality.list_id")
        require_optional_str(obj.get("paragraph_cluster_id"), "locality.paragraph_cluster_id")
        require_list_of_str(obj.get("neighbor_episode_ids"), "locality.neighbor_episode_ids")
        return obj
    except ValueError as exc:
        _raise_validation("invalid_locality_schema", "locality_layer", "schema", str(exc))


def validate_revision(obj: dict, fact_ids: set[str]) -> dict:
    try:
        obj = require_dict(obj, "revision_currentness")
        require_str(obj.get("revision_id"), "revision_id")
        require_str(obj.get("topic_key"), "topic_key")
        require_id_ref(obj.get("old_fact_id"), "old_fact_id", fact_ids)
        require_id_ref(obj.get("new_fact_id"), "new_fact_id", fact_ids)
        require_enum(obj.get("link_type"), "link_type", {"supersedes", "superseded_by", "current_value_for"})
        require_id_ref(obj.get("current_fact_id"), "current_fact_id", fact_ids)
        require_optional_date(obj.get("effective_date"), "effective_date")
        require_id_ref_list(obj.get("revision_source_fact_ids"), "revision_source_fact_ids", fact_ids)
        return obj
    except ValueError as exc:
        _raise_validation("invalid_revision_schema", "revision_currentness_layer", "schema", str(exc))


def validate_event(obj: dict, fact_ids: set[str]) -> dict:
    try:
        obj = require_dict(obj, "event")
        require_str(obj.get("event_id"), "event_id")
        require_str(obj.get("event_type"), "event_type")
        require_list_of_str(obj.get("participants"), "participants")
        require_optional_str(obj.get("object"), "object")
        require_optional_date(obj.get("time"), "time")
        require_optional_str(obj.get("location"), "location")
        params = require_list(obj.get("parameters"), "parameters")
        for i, param in enumerate(params):
            param = require_dict(param, f"parameters[{i}]")
            require_str(param.get("name"), f"parameters[{i}].name")
            require_optional_number(param.get("value_number"), f"parameters[{i}].value_number")
            require_optional_str(param.get("value_unit"), f"parameters[{i}].value_unit")
            require_optional_str(param.get("value_text"), f"parameters[{i}].value_text")
        require_optional_str(obj.get("outcome"), "outcome")
        require_optional_str(obj.get("status"), "status")
        require_id_ref_list(obj.get("support_fact_ids"), "support_fact_ids", fact_ids)
        return obj
    except ValueError as exc:
        _raise_validation("invalid_event_schema", "event_layer", "schema", str(exc))


def validate_record(obj: dict, fact_ids: set[str]) -> dict:
    try:
        obj = require_dict(obj, "record")
        require_str(obj.get("record_id"), "record_id")
        require_str(obj.get("record_type"), "record_type")
        require_str(obj.get("item_id"), "item_id")
        require_optional_str(obj.get("status"), "status")
        require_optional_date(obj.get("date"), "date")
        require_optional_str(obj.get("qualifier"), "qualifier")
        require_optional_str(obj.get("owner"), "owner")
        require_optional_str(obj.get("source_section"), "source_section")
        require_id_ref_list(obj.get("support_fact_ids"), "support_fact_ids", fact_ids)
        return obj
    except ValueError as exc:
        _raise_validation("invalid_record_schema", "record_layer", "schema", str(exc))


def validate_edge(obj: dict, all_node_ids: set[str], fact_ids: set[str]) -> dict:
    try:
        obj = require_dict(obj, "edge")
        require_str(obj.get("edge_id"), "edge_id")
        require_enum(
            obj.get("edge_type"),
            "edge_type",
            {
                "causes",
                "leads_to",
                "supports",
                "same_anchor",
                "conflicts_with",
                "bridge_to",
                "resolver_for",
                "belongs_to_event",
                "belongs_to_record",
            },
        )
        require_id_ref(obj.get("from_id"), "from_id", all_node_ids)
        require_id_ref(obj.get("to_id"), "to_id", all_node_ids)
        require_optional_str(obj.get("edge_evidence_text"), "edge_evidence_text")
        require_optional_str(obj.get("anchor_key"), "anchor_key")
        require_list(obj.get("anchor_basis_fact_ids", []), "anchor_basis_fact_ids")
        require_id_ref_list(obj.get("support_fact_ids"), "support_fact_ids", fact_ids)
        return obj
    except ValueError as exc:
        _raise_validation("invalid_edge_schema", "edge_layer", "schema", str(exc))


def validate_structural_edge_direction(obj: dict, event_ids: set[str], record_ids: set[str]) -> dict:
    try:
        edge_type = obj["edge_type"]
        from_id = obj["from_id"]
        to_id = obj["to_id"]
        if edge_type == "belongs_to_event":
            if to_id not in event_ids:
                raise ValueError("belongs_to_event: to_id must reference an event_id")
            if from_id in event_ids:
                raise ValueError("belongs_to_event: from_id must not reference an event_id")
        if edge_type == "belongs_to_record":
            if to_id not in record_ids:
                raise ValueError("belongs_to_record: to_id must reference a record_id")
            if from_id in record_ids:
                raise ValueError("belongs_to_record: from_id must not reference a record_id")
        return obj
    except ValueError as exc:
        _raise_validation("invalid_edge_schema", "edge_layer", "schema", str(exc))


def validate_atomic_fact_grounding(obj: dict, source_text: str, *, require_offsets: bool = True) -> dict:
    try:
        span = require_span_grounded(source_text, obj.get("source_span"), "source_span")
        if require_offsets:
            require_span_offsets(
                source_text,
                span,
                obj.get("source_span_start"),
                obj.get("source_span_end"),
                "source_span",
            )
        require_grounded_date(obj.get("asserted_at"), span, "asserted_at")
        require_grounded_number_unit(
            obj.get("value_number"),
            obj.get("value_unit"),
            span,
            "value_number",
            "value_unit",
        )
        require_grounded_string(obj.get("object"), span, "object")
        if obj.get("value_text") is not None:
            require_grounded_string(obj.get("value_text"), span, "value_text")
        return obj
    except ValueError as exc:
        _raise_validation("ungrounded_atomic_fact", "atomic_fact_layer", "grounding", str(exc))


def validate_atomic_fact_grounding_source_level(obj: dict) -> dict:
    return validate_atomic_fact_grounding(obj, obj["source_span"], require_offsets=False)


def build_fact_lookup(atomic_facts: list[dict]) -> dict[str, dict]:
    return {fact["fact_id"]: fact for fact in atomic_facts}


def support_text_from_fact_ids(fact_lookup: dict[str, dict], fact_ids: list[str]) -> str:
    return "\n".join(fact_lookup[fact_id]["source_span"] for fact_id in fact_ids)


def _episode_id_from_fact_id(fact_id: str) -> str | None:
    match = re.match(r"^ep_(.+?)_f(?:_|$)", fact_id or "")
    if not match:
        return None
    return match.group(1)


def validate_locality_grounding(obj: dict, locality_metadata: dict) -> dict:
    try:
        metadata = require_dict(locality_metadata, "locality_metadata")
        expected_source_id = require_str(metadata.get("source_id"), "locality_metadata.source_id")
        expected_episode_id = require_str(metadata.get("episode_id"), "locality_metadata.episode_id")
        if obj["source_id"] != expected_source_id:
            raise ValueError("locality.source_id: not grounded in deterministic locality metadata")
        if obj["episode_id"] != expected_episode_id:
            raise ValueError("locality.episode_id: not grounded in deterministic locality metadata")
        for field in ("section_id", "heading", "table_id", "list_id", "paragraph_cluster_id"):
            expected = metadata.get(field)
            actual = obj.get(field)
            if expected is None:
                if actual is not None:
                    raise ValueError(f"locality.{field}: metadata missing, field must be null")
            elif actual != expected:
                raise ValueError(f"locality.{field}: not grounded in deterministic locality metadata")
        expected_neighbors = metadata.get("neighbor_episode_ids", [])
        if obj.get("neighbor_episode_ids") != expected_neighbors:
            raise ValueError("locality.neighbor_episode_ids: not grounded in deterministic locality metadata")
        return obj
    except ValueError as exc:
        _raise_validation("ungrounded_locality", "locality_layer", "grounding", str(exc))


def validate_source_aggregation_locality(
    locality_by_episode: dict,
    source_id: str,
    episode_ids: list[str],
    locality_metadata_by_episode: dict[str, dict],
) -> dict:
    try:
        locality_by_episode = require_dict(locality_by_episode, "locality_by_episode")
        for episode_id in episode_ids:
            if episode_id not in locality_by_episode:
                raise ValueError(f"locality_by_episode: missing locality for {episode_id!r}")
            locality = validate_locality(locality_by_episode[episode_id], source_id, episode_id)
            if episode_id not in locality_metadata_by_episode:
                raise ValueError(f"locality_metadata_by_episode: missing metadata for {episode_id!r}")
            validate_locality_grounding(locality, locality_metadata_by_episode[episode_id])
        return locality_by_episode
    except SubstrateValidationError:
        raise
    except ValueError as exc:
        _raise_validation("ungrounded_locality", "locality_layer", "grounding", str(exc))


def validate_revision_grounding(obj: dict, fact_lookup: dict[str, dict]) -> dict:
    try:
        support_text = support_text_from_fact_ids(fact_lookup, obj["revision_source_fact_ids"])
        require_grounded_date(obj.get("effective_date"), support_text, "effective_date")
        return obj
    except ValueError as exc:
        _raise_validation("ungrounded_revision", "revision_currentness_layer", "grounding", str(exc))


def validate_event_grounding(obj: dict, fact_lookup: dict[str, dict]) -> dict:
    try:
        support_text = support_text_from_fact_ids(fact_lookup, obj["support_fact_ids"])
        for i, participant in enumerate(obj["participants"]):
            require_grounded_string(participant, support_text, f"participants[{i}]")
        require_grounded_string(obj.get("object"), support_text, "object")
        require_grounded_date(obj.get("time"), support_text, "time")
        require_grounded_string(obj.get("location"), support_text, "location")
        require_grounded_string(obj.get("outcome"), support_text, "outcome")
        require_grounded_string(obj.get("status"), support_text, "status")
        for i, param in enumerate(obj["parameters"]):
            require_grounded_number_unit(
                param.get("value_number"),
                param.get("value_unit"),
                support_text,
                f"parameters[{i}].value_number",
                f"parameters[{i}].value_unit",
            )
            require_grounded_string(param.get("value_text"), support_text, f"parameters[{i}].value_text")
        return obj
    except ValueError as exc:
        _raise_validation("ungrounded_event", "event_layer", "grounding", str(exc))


def validate_record_grounding(obj: dict, fact_lookup: dict[str, dict], locality: dict) -> dict:
    try:
        support_text = support_text_from_fact_ids(fact_lookup, obj["support_fact_ids"])
        require_grounded_string(obj.get("item_id"), support_text, "item_id")
        require_grounded_string(obj.get("status"), support_text, "status")
        require_grounded_date(obj.get("date"), support_text, "date")
        require_grounded_string(obj.get("qualifier"), support_text, "qualifier")
        require_grounded_string(obj.get("owner"), support_text, "owner")
        source_section = require_optional_str(obj.get("source_section"), "source_section")
        if source_section is not None:
            allowed_sections = {value for value in [locality.get("heading"), locality.get("section_id")] if value is not None}
            if source_section not in allowed_sections:
                raise ValueError("source_section: not grounded in locality metadata")
        return obj
    except ValueError as exc:
        _raise_validation("ungrounded_record", "record_layer", "grounding", str(exc))


def validate_record_grounding_source_level(
    obj: dict,
    fact_lookup: dict[str, dict],
    locality_by_episode: dict[str, dict],
) -> dict:
    try:
        support_text = support_text_from_fact_ids(fact_lookup, obj["support_fact_ids"])
        require_grounded_string(obj.get("item_id"), support_text, "item_id")
        require_grounded_string(obj.get("status"), support_text, "status")
        require_grounded_date(obj.get("date"), support_text, "date")
        require_grounded_string(obj.get("qualifier"), support_text, "qualifier")
        require_grounded_string(obj.get("owner"), support_text, "owner")
        source_section = require_optional_str(obj.get("source_section"), "source_section")
        if source_section is not None:
            allowed_sections: set[str] = set()
            for locality in locality_by_episode.values():
                for value in (locality.get("heading"), locality.get("section_id")):
                    if value is not None:
                        allowed_sections.add(value)
            if source_section not in allowed_sections:
                raise ValueError("source_section: not grounded in source-level locality metadata")
        return obj
    except ValueError as exc:
        _raise_validation("ungrounded_record", "record_layer", "grounding", str(exc))


def validate_edge_grounding(obj: dict, fact_lookup: dict[str, dict]) -> dict:
    try:
        support_text = support_text_from_fact_ids(fact_lookup, obj["support_fact_ids"])
        edge_type = obj["edge_type"]
        if edge_type in SEMANTIC_EDGE_TYPES:
            require_grounded_string(obj.get("edge_evidence_text"), support_text, "edge_evidence_text")
        elif edge_type == "same_anchor":
            basis = require_id_ref_list(obj.get("anchor_basis_fact_ids"), "anchor_basis_fact_ids", set(fact_lookup))
            if not basis:
                raise ValueError("anchor_basis_fact_ids: same_anchor requires non-empty basis facts")
            basis_text = support_text_from_fact_ids(fact_lookup, basis)
            require_grounded_anchor_key(obj.get("anchor_key"), basis_text, "anchor_key")
        elif edge_type in {"belongs_to_event", "belongs_to_record"}:
            support_ids = require_id_ref_list(obj.get("support_fact_ids"), "support_fact_ids", set(fact_lookup))
            if not support_ids:
                raise ValueError("support_fact_ids: structural membership edge requires non-empty support facts")
        return obj
    except ValueError as exc:
        _raise_validation("ungrounded_edge", "edge_layer", "grounding", str(exc))


def _validate_higher_order_layers_episode(payload: dict, locality: dict) -> dict:
    fact_lookup = build_fact_lookup(payload["atomic_facts"])
    fact_ids = set(fact_lookup)
    for obj in payload["revision_currentness"]:
        validate_revision(obj, fact_ids)
        validate_revision_grounding(obj, fact_lookup)
    for obj in payload["events"]:
        validate_event(obj, fact_ids)
        validate_event_grounding(obj, fact_lookup)
    for obj in payload["records"]:
        validate_record(obj, fact_ids)
        validate_record_grounding(obj, fact_lookup, locality)
    all_node_ids = fact_ids | {obj["event_id"] for obj in payload["events"]} | {obj["record_id"] for obj in payload["records"]}
    event_ids = {obj["event_id"] for obj in payload["events"]}
    record_ids = {obj["record_id"] for obj in payload["records"]}
    for obj in payload["edges"]:
        validate_edge(obj, all_node_ids, fact_ids)
        validate_structural_edge_direction(obj, event_ids, record_ids)
        validate_edge_grounding(obj, fact_lookup)
    return payload


def _validate_higher_order_layers_source(payload: dict, locality_by_episode: dict[str, dict]) -> dict:
    fact_lookup = build_fact_lookup(payload["atomic_facts"])
    fact_ids = set(fact_lookup)
    for obj in payload["revision_currentness"]:
        validate_revision(obj, fact_ids)
        validate_revision_grounding(obj, fact_lookup)
    for obj in payload["events"]:
        validate_event(obj, fact_ids)
        validate_event_grounding(obj, fact_lookup)
    for obj in payload["records"]:
        validate_record(obj, fact_ids)
        validate_record_grounding_source_level(obj, fact_lookup, locality_by_episode)
    all_node_ids = fact_ids | {obj["event_id"] for obj in payload["events"]} | {obj["record_id"] for obj in payload["records"]}
    event_ids = {obj["event_id"] for obj in payload["events"]}
    record_ids = {obj["record_id"] for obj in payload["records"]}
    for obj in payload["edges"]:
        validate_edge(obj, all_node_ids, fact_ids)
        validate_structural_edge_direction(obj, event_ids, record_ids)
        validate_edge_grounding(obj, fact_lookup)
    return payload


def validate_episode_payload(payload: dict, *, source_text: str, locality_metadata: dict) -> dict:
    payload = copy.deepcopy(payload)
    validate_envelope(payload)
    validate_uniqueness(payload)
    for obj in payload["atomic_facts"]:
        validate_atomic_fact(obj)
        validate_atomic_fact_grounding(obj, source_text)
    locality = validate_locality(payload["locality"], payload["source_id"], payload["episode_id"])
    validate_locality_grounding(locality, locality_metadata)
    _validate_higher_order_layers_episode(payload, locality)
    return payload


def validate_source_aggregation_payload(
    payload: dict,
    *,
    locality_metadata_by_episode: dict[str, dict],
    source_text_by_episode: dict[str, str] | None = None,
) -> dict:
    payload = copy.deepcopy(payload)
    validate_source_aggregation_envelope(payload)
    validate_uniqueness(payload)
    for obj in payload["atomic_facts"]:
        validate_atomic_fact(obj)
        source_text = None
        if source_text_by_episode:
            episode_id = _episode_id_from_fact_id(obj.get("fact_id", ""))
            source_text = source_text_by_episode.get(episode_id or "")
        if source_text:
            validate_atomic_fact_grounding(obj, source_text, require_offsets=True)
        else:
            validate_atomic_fact_grounding_source_level(obj)
    validate_source_aggregation_locality(
        payload["locality_by_episode"],
        payload["source_id"],
        payload["episode_ids"],
        locality_metadata_by_episode,
    )
    _validate_higher_order_layers_source(payload, payload["locality_by_episode"])
    validate_source_aggregation_non_empty(payload)
    return payload


def _drop_layer(payload: dict, layer_name: str) -> dict:
    payload = copy.deepcopy(payload)
    if layer_name == "event_layer":
        removed_ids = {obj["event_id"] for obj in payload.get("events", [])}
        payload["events"] = []
        if removed_ids:
            payload["edges"] = [
                edge for edge in payload.get("edges", [])
                if edge.get("from_id") not in removed_ids and edge.get("to_id") not in removed_ids
            ]
    elif layer_name == "record_layer":
        removed_ids = {obj["record_id"] for obj in payload.get("records", [])}
        payload["records"] = []
        if removed_ids:
            payload["edges"] = [
                edge for edge in payload.get("edges", [])
                if edge.get("from_id") not in removed_ids and edge.get("to_id") not in removed_ids
            ]
    elif layer_name == "edge_layer":
        payload["edges"] = []
    elif layer_name == "revision_currentness_layer":
        payload["revision_currentness"] = []
    return payload


def _coerce_id_ref(item: Any) -> str | None:
    if isinstance(item, str):
        value = item.strip()
        return value or None
    if isinstance(item, dict):
        for key in ("fact_id", "id"):
            value = item.get(key)
            if isinstance(value, str):
                value = value.strip()
                if value:
                    return value
    return None


def _coerce_id_ref_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        coerced = _coerce_id_ref(item)
        if coerced is not None:
            normalized.append(coerced)
    return normalized


def _sanitize_episode_payload(payload: dict) -> dict:
    payload = copy.deepcopy(payload)
    fact_lookup = build_fact_lookup(payload.get("atomic_facts", []))
    for revision in payload.get("revision_currentness", []):
        revision["revision_source_fact_ids"] = _coerce_id_ref_list(revision.get("revision_source_fact_ids"))
    for event in payload.get("events", []):
        support_ids = _coerce_id_ref_list(event.get("support_fact_ids"))
        event["support_fact_ids"] = support_ids
        if all(fid in fact_lookup for fid in support_ids):
            support_text = support_text_from_fact_ids(fact_lookup, support_ids)
            try:
                require_grounded_string(event.get("location"), support_text, "location")
            except ValueError:
                event["location"] = None
    for record in payload.get("records", []):
        support_ids = _coerce_id_ref_list(record.get("support_fact_ids"))
        record["support_fact_ids"] = support_ids
        if all(fid in fact_lookup for fid in support_ids):
            support_text = support_text_from_fact_ids(fact_lookup, support_ids)
            try:
                require_grounded_string(record.get("owner"), support_text, "owner")
            except ValueError:
                record["owner"] = None
    for edge in payload.get("edges", []):
        edge["support_fact_ids"] = _coerce_id_ref_list(edge.get("support_fact_ids"))
        edge["anchor_basis_fact_ids"] = _coerce_id_ref_list(edge.get("anchor_basis_fact_ids"))
    return payload


def _sanitize_source_payload(payload: dict) -> dict:
    return _sanitize_episode_payload(payload)


def _accepted_layers(dropped_layers: list[str]) -> list[str]:
    return [layer for layer in LAYER_ORDER if layer not in dropped_layers]


def compute_episode_extraction_budget() -> int:
    return EPISODE_EXTRACTION_BUDGET


def compute_source_aggregation_budget() -> int:
    return SOURCE_AGGREGATION_BUDGET


def compute_source_pipeline_budget(episode_count: int) -> int:
    if not isinstance(episode_count, int) or episode_count <= 0:
        raise ValueError("episode_count: expected positive integer")
    return (EPISODE_EXTRACTION_BUDGET * episode_count) + SOURCE_AGGREGATION_BUDGET


def run_episode_validation_pipeline(
    attempt_payloads: list[dict],
    *,
    source_text: str,
    locality_metadata: dict,
) -> dict:
    max_attempts = min(len(attempt_payloads), 3)
    last_err: SubstrateValidationError | None = None
    last_payload: dict | None = None
    for idx in range(max_attempts):
        candidate = _sanitize_episode_payload(attempt_payloads[idx])
        last_payload = candidate
        try:
            validated = validate_episode_payload(
                candidate,
                source_text=source_text,
                locality_metadata=locality_metadata,
            )
            return {
                "payload": validated,
                "extraction_status": "accepted",
                "schema_error_count": 0,
                "grounding_error_count": 0,
                "repair_attempt_count": max(0, idx),
                "accepted_layers": list(LAYER_ORDER),
                "dropped_layers": [],
                "failure_reasons": [],
                "source_pipeline_attempt_count": idx + 1,
            }
        except SubstrateValidationError as err:
            last_err = err
            continue

    if last_err is None or last_payload is None:
        raise ValueError("attempt_payloads: expected at least one payload")

    if last_err.layer in {"atomic_fact_layer", "locality_layer", "top_level"}:
        return {
            "payload": None,
            "extraction_status": "failed",
            "schema_error_count": 1 if last_err.kind == "schema" else 0,
            "grounding_error_count": 1 if last_err.kind == "grounding" else 0,
            "repair_attempt_count": max_attempts - 1,
            "accepted_layers": [],
            "dropped_layers": [],
            "failure_reasons": [last_err.code, "extraction_retry_exhausted"],
            "source_pipeline_attempt_count": max_attempts,
        }

    dropped_layer = last_err.layer
    dropped_payload = _drop_layer(last_payload, dropped_layer)
    try:
        validated = validate_episode_payload(
            dropped_payload,
            source_text=source_text,
            locality_metadata=locality_metadata,
        )
    except SubstrateValidationError as err:
        failure_reasons = [last_err.code]
        if err.code not in failure_reasons:
            failure_reasons.append(err.code)
        failure_reasons.append("extraction_retry_exhausted")
        return {
            "payload": None,
            "extraction_status": "failed",
            "schema_error_count": int(last_err.kind == "schema") + int(err.kind == "schema"),
            "grounding_error_count": int(last_err.kind == "grounding") + int(err.kind == "grounding"),
            "repair_attempt_count": max_attempts - 1,
            "accepted_layers": [],
            "dropped_layers": [],
            "failure_reasons": failure_reasons,
            "source_pipeline_attempt_count": max_attempts,
        }
    return {
        "payload": validated,
        "extraction_status": "partial",
        "schema_error_count": 1 if last_err.kind == "schema" else 0,
        "grounding_error_count": 1 if last_err.kind == "grounding" else 0,
        "repair_attempt_count": max_attempts - 1,
        "accepted_layers": _accepted_layers([dropped_layer]),
        "dropped_layers": [dropped_layer],
        "failure_reasons": [last_err.code, "extraction_retry_exhausted"],
        "source_pipeline_attempt_count": max_attempts,
    }


def run_source_aggregation_validation_pipeline(
    attempt_payloads: list[dict],
    *,
    locality_metadata_by_episode: dict[str, dict],
    episode_count: int,
    source_text_by_episode: dict[str, str] | None = None,
) -> dict:
    max_attempts = min(len(attempt_payloads), 3)
    last_err: SubstrateValidationError | None = None
    last_payload: dict | None = None
    for idx in range(max_attempts):
        candidate = _sanitize_source_payload(attempt_payloads[idx])
        last_payload = candidate
        try:
            validated = validate_source_aggregation_payload(
                candidate,
                locality_metadata_by_episode=locality_metadata_by_episode,
                source_text_by_episode=source_text_by_episode,
            )
            return {
                "payload": validated,
                "aggregation_status": "accepted",
                "schema_error_count": 0,
                "grounding_error_count": 0,
                "aggregation_repair_attempt_count": max(0, idx),
                "accepted_layers": list(LAYER_ORDER),
                "dropped_layers": [],
                "failure_reasons": [],
                "source_pipeline_attempt_count": compute_source_pipeline_budget(episode_count),
            }
        except SubstrateValidationError as err:
            last_err = err
            continue

    if last_err is None or last_payload is None:
        raise ValueError("attempt_payloads: expected at least one payload")

    if last_err.layer in {"atomic_fact_layer", "locality_layer", "top_level"}:
        return {
            "payload": None,
            "aggregation_status": "failed",
            "schema_error_count": 1 if last_err.kind == "schema" else 0,
            "grounding_error_count": 1 if last_err.kind == "grounding" else 0,
            "aggregation_repair_attempt_count": max_attempts - 1,
            "accepted_layers": [],
            "dropped_layers": [],
            "failure_reasons": [last_err.code, "source_aggregation_retry_exhausted"],
            "source_pipeline_attempt_count": compute_source_pipeline_budget(episode_count),
        }

    dropped_layer = last_err.layer
    dropped_payload = _drop_layer(last_payload, dropped_layer)
    try:
        validated = validate_source_aggregation_payload(
            dropped_payload,
            locality_metadata_by_episode=locality_metadata_by_episode,
            source_text_by_episode=source_text_by_episode,
        )
    except SubstrateValidationError as err:
        failure_reasons = [last_err.code]
        if err.code not in failure_reasons:
            failure_reasons.append(err.code)
        failure_reasons.append("source_aggregation_retry_exhausted")
        return {
            "payload": None,
            "aggregation_status": "failed",
            "schema_error_count": int(last_err.kind == "schema") + int(err.kind == "schema"),
            "grounding_error_count": int(last_err.kind == "grounding") + int(err.kind == "grounding"),
            "aggregation_repair_attempt_count": max_attempts - 1,
            "accepted_layers": [],
            "dropped_layers": [],
            "failure_reasons": failure_reasons,
            "source_pipeline_attempt_count": compute_source_pipeline_budget(episode_count),
        }
    return {
        "payload": validated,
        "aggregation_status": "partial",
        "schema_error_count": 1 if last_err.kind == "schema" else 0,
        "grounding_error_count": 1 if last_err.kind == "grounding" else 0,
        "aggregation_repair_attempt_count": max_attempts - 1,
        "accepted_layers": _accepted_layers([dropped_layer]),
        "dropped_layers": [dropped_layer],
        "failure_reasons": [last_err.code, "source_aggregation_retry_exhausted"],
        "source_pipeline_attempt_count": compute_source_pipeline_budget(episode_count),
    }


def persist_episode_payload(root_dir: str | Path, payload: dict) -> Path:
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "episode_extraction.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return path


def persist_source_aggregation_payload(root_dir: str | Path, payload: dict) -> Path:
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "source_aggregation.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return path
