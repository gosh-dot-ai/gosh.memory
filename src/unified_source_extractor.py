"""Runtime source-level extraction using the unified extraction substrate."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .extraction_substrate import (
    SubstrateValidationError,
    run_source_aggregation_validation_pipeline,
    validate_source_aggregation_payload,
)

_PROMPT_DIR = Path(__file__).parent / "prompts" / "extraction"


def _load_prompt(name: str) -> str:
    path = _PROMPT_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8")


def _parse_json(raw: Any) -> dict | None:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(1))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _deterministic_locality(source_id: str, episode: dict, episode_ids: list[str]) -> dict:
    episode_id = episode["episode_id"]
    idx = episode_ids.index(episode_id)
    neighbors = []
    if idx > 0:
        neighbors.append(episode_ids[idx - 1])
    if idx + 1 < len(episode_ids):
        neighbors.append(episode_ids[idx + 1])
    section_path = ((episode.get("metadata") or {}).get("source_section_path") or "").strip() or None
    return {
        "source_id": source_id,
        "episode_id": episode_id,
        "section_id": section_path,
        "heading": section_path,
        "table_id": None,
        "list_id": None,
        "paragraph_cluster_id": f"{episode_id}_p01",
        "neighbor_episode_ids": neighbors,
    }


def _episode_descriptor_block(episodes: list[dict]) -> str:
    lines: list[str] = []
    for ep in episodes:
        lines.append(f"[EPISODE {ep['episode_id']}]")
        source_date = (ep.get("source_date") or "").strip() or "unknown"
        lines.append(f"DATE: {source_date}")
        lines.append("TEXT:")
        lines.append(ep.get("raw_text", ""))
        lines.append("")
    return "\n".join(lines).strip()


def _grounded_fact_catalog(source_facts: list[dict]) -> list[dict]:
    catalog: list[dict] = []
    seen_ids: set[str] = set()
    for fact in source_facts:
        if not isinstance(fact, dict):
            continue
        fact_id = str(fact.get("id") or "").strip()
        fact_text = str(fact.get("fact") or "").strip()
        if not fact_id or not fact_text or fact_id in seen_ids:
            continue
        metadata = fact.get("metadata") or {}
        episode_id = str(
            metadata.get("episode_id")
            or metadata.get("episode_source_id")
            or fact.get("episode_id")
            or ""
        ).strip()
        entities = [
            entity
            for entity in (fact.get("entities") or [])
            if isinstance(entity, str) and entity.strip()
        ]
        catalog.append(
            {
                "fact_id": fact_id,
                "episode_id": episode_id or None,
                "fact_text": fact_text,
                "entity_ids": entities,
            }
        )
        seen_ids.add(fact_id)
    return catalog


def _base_atomic_facts(source_facts: list[dict]) -> list[dict]:
    atomic_facts: list[dict] = []
    for row in _grounded_fact_catalog(source_facts):
        fact_text = row["fact_text"]
        entity_ids = list(row.get("entity_ids") or [])
        subject = entity_ids[0] if entity_ids else (row.get("episode_id") or row["fact_id"])
        atomic_facts.append(
            {
                "fact_id": row["fact_id"],
                "subject": subject,
                "relation": "grounded_support",
                "object": fact_text,
                "value_text": fact_text,
                "value_number": None,
                "value_unit": None,
                "polarity": "positive",
                "confidence": 1.0,
                "source_span": fact_text,
                "source_span_start": None,
                "source_span_end": None,
                "asserted_at": None,
                "entity_ids": entity_ids,
                "episode_id": row.get("episode_id"),
            }
        )
    return atomic_facts


def _grounded_fact_payload(source_facts: list[dict]) -> str:
    return json.dumps(_grounded_fact_catalog(source_facts), ensure_ascii=False, indent=2)


def _compact_previous_json_for_repair(prev_json: dict | None) -> dict:
    if not isinstance(prev_json, dict):
        return {}

    def _trim_value(value, *, depth: int = 0):
        if isinstance(value, str):
            if len(value) <= 400:
                return value
            return value[:400] + "... [truncated]"
        if isinstance(value, list):
            limit = 8 if depth < 2 else 4
            trimmed = [_trim_value(item, depth=depth + 1) for item in value[:limit]]
            if len(value) > limit:
                trimmed.append({"_truncated_items": len(value) - limit})
            return trimmed
        if isinstance(value, dict):
            compact: dict = {}
            for key, inner in value.items():
                if key == "atomic_facts":
                    facts = inner if isinstance(inner, list) else []
                    compact["atomic_fact_count"] = len(facts)
                    compact["atomic_fact_preview"] = [
                        {
                            "fact_id": fact.get("fact_id"),
                            "fact": _trim_value(fact.get("fact", ""), depth=depth + 1),
                            "entity_ids": _trim_value(fact.get("entity_ids", []), depth=depth + 1),
                            "episode_id": fact.get("episode_id"),
                        }
                        for fact in facts[:4]
                        if isinstance(fact, dict)
                    ]
                    continue
                if key == "locality_by_episode":
                    locality = inner if isinstance(inner, dict) else {}
                    compact["locality_episode_count"] = len(locality)
                    compact["locality_preview"] = {
                        episode_id: _trim_value(locality[episode_id], depth=depth + 1)
                        for episode_id in list(locality.keys())[:4]
                    }
                    continue
                compact[key] = _trim_value(inner, depth=depth + 1)
            return compact
        return value

    return _trim_value(prev_json)


def _repair_prompt(
    source_id: str,
    source_kind: str,
    episodes: list[dict],
    source_facts: list[dict],
    prev_json: dict | None,
    error_text: str,
    prompt_overrides: dict[str, str] | None = None,
) -> str:
    if prompt_overrides and "unified_source_aggregation_repair" in prompt_overrides:
        prompt = prompt_overrides["unified_source_aggregation_repair"]
    else:
        prompt = _load_prompt("unified_source_aggregation_repair")
    return prompt.format(
        source_id=source_id,
        source_kind=source_kind,
        episode_payload=_episode_descriptor_block(episodes),
        grounded_fact_payload=_grounded_fact_payload(source_facts),
        previous_json=json.dumps(
            _compact_previous_json_for_repair(prev_json),
            ensure_ascii=False,
            indent=2,
        ),
        validation_error=error_text,
    )


def _base_prompt(source_id: str, source_kind: str, episodes: list[dict], source_facts: list[dict],
                 prompt_overrides: dict[str, str] | None = None) -> str:
    if prompt_overrides and "unified_source_aggregation" in prompt_overrides:
        prompt = prompt_overrides["unified_source_aggregation"]
    else:
        prompt = _load_prompt("unified_source_aggregation")
    return prompt.format(
        source_id=source_id,
        source_kind=source_kind,
        episode_payload=_episode_descriptor_block(episodes),
        grounded_fact_payload=_grounded_fact_payload(source_facts),
    )


def _build_payload_envelope(
    source_id: str,
    source_kind: str,
    episodes: list[dict],
    source_facts: list[dict],
    body: dict,
) -> tuple[dict, dict[str, dict], dict[str, str]]:
    episode_ids = [ep["episode_id"] for ep in episodes]
    locality_by_episode = {
        ep["episode_id"]: _deterministic_locality(source_id, ep, episode_ids)
        for ep in episodes
    }
    source_text_by_episode = {
        ep["episode_id"]: ep.get("raw_text", "")
        for ep in episodes
    }
    payload = {
        "schema": "extraction_substrate",
        "payload_scope": "source_aggregation",
        "source_id": source_id,
        "source_kind": source_kind,
        "episode_ids": episode_ids,
        "locality_by_episode": locality_by_episode,
        "atomic_facts": _base_atomic_facts(source_facts),
        "revision_currentness": body.get("revision_currentness", []),
        "events": body.get("events", []),
        "records": body.get("records", []),
        "edges": body.get("edges", []),
    }
    return payload, locality_by_episode, source_text_by_episode


def _atomic_lookup(payload: dict) -> dict[str, dict]:
    return {fact["fact_id"]: fact for fact in payload.get("atomic_facts", []) if isinstance(fact, dict) and fact.get("fact_id")}


def _event_lookup(payload: dict) -> dict[str, dict]:
    return {event["event_id"]: event for event in payload.get("events", []) if isinstance(event, dict) and event.get("event_id")}


def _record_lookup(payload: dict) -> dict[str, dict]:
    return {record["record_id"]: record for record in payload.get("records", []) if isinstance(record, dict) and record.get("record_id")}


def _support_entity_ids(payload: dict, support_fact_ids: list[str]) -> list[str]:
    fact_lookup = _atomic_lookup(payload)
    out: list[str] = []
    seen: set[str] = set()
    for fact_id in support_fact_ids or []:
        fact = fact_lookup.get(fact_id) or {}
        for entity in fact.get("entity_ids", []) or []:
            if isinstance(entity, str) and entity and entity not in seen:
                seen.add(entity)
                out.append(entity)
    return out


def _event_summary(event: dict) -> str:
    parts: list[str] = []
    participants = event.get("participants") or []
    if participants:
        parts.append(", ".join(participants))
    event_type = (event.get("event_type") or "event").replace("_", " ")
    if event_type:
        parts.append(event_type)
    if event.get("object"):
        parts.append(f"for {event['object']}")
    if event.get("time"):
        parts.append(f"on {event['time']}")
    if event.get("location"):
        parts.append(f"at {event['location']}")
    params = []
    for param in event.get("parameters", []) or []:
        value_text = param.get("value_text")
        if value_text:
            params.append(f"{param.get('name')}: {value_text}")
    if params:
        parts.append(f"with {', '.join(params)}")
    if event.get("outcome"):
        parts.append(f"outcome {event['outcome']}")
    if event.get("status"):
        parts.append(f"status {event['status']}")
    return " ".join(parts).strip().rstrip(".") + "."


def _record_summary(record: dict) -> str:
    parts = [f"{record.get('record_type', 'record').replace('_', ' ')} {record.get('item_id', '')}".strip()]
    if record.get("status"):
        parts.append(f"status {record['status']}")
    if record.get("date"):
        parts.append(f"date {record['date']}")
    if record.get("qualifier"):
        parts.append(f"qualifier {record['qualifier']}")
    if record.get("owner"):
        parts.append(f"owner {record['owner']}")
    if record.get("source_section"):
        parts.append(f"section {record['source_section']}")
    return " ".join(parts).strip().rstrip(".") + "."


def _fact_summary(fact: dict) -> str:
    obj = fact.get("value_text") or fact.get("object") or ""
    return f"{fact.get('subject', '').strip()} {fact.get('relation', '').strip().replace('_', ' ')} {obj}".strip()


def _episode_id_from_fact(payload: dict, fact_id: str) -> str | None:
    fact = _atomic_lookup(payload).get(fact_id) or {}
    episode_id = str(fact.get("episode_id") or "").strip()
    if episode_id:
        return episode_id
    return _episode_id_from_fact_id(fact_id)


def _episode_metadata(payload: dict, fact_ids: list[str]) -> dict[str, Any]:
    episode_ids: set[str] = set()
    for fact_id in fact_ids:
        episode_id = _episode_id_from_fact(payload, fact_id)
        if episode_id:
            episode_ids.add(episode_id)
    metadata: dict[str, Any] = {
        "source_aggregation": True,
        "episode_ids": sorted(episode_ids),
    }
    if len(metadata["episode_ids"]) == 1:
        metadata["episode_id"] = metadata["episode_ids"][0]
    return metadata


def _node_summary(payload: dict, node_id: str) -> str:
    event = _event_lookup(payload).get(node_id)
    if event:
        return _event_summary(event).rstrip(".")
    record = _record_lookup(payload).get(node_id)
    if record:
        return _record_summary(record).rstrip(".")
    fact = _atomic_lookup(payload).get(node_id)
    if fact:
        return _fact_summary(fact)
    return node_id


def flatten_source_aggregation_payload(payload: dict) -> list[dict]:
    facts: list[dict] = []
    fact_lookup = _atomic_lookup(payload)

    for revision in payload.get("revision_currentness", []) or []:
        old_fact = fact_lookup.get(revision["old_fact_id"], {})
        new_fact = fact_lookup.get(revision["new_fact_id"], {})
        text = (
            f"Current value for {revision['topic_key'].replace('_', ' ')} is "
            f"{new_fact.get('value_text') or new_fact.get('object') or _fact_summary(new_fact)}; "
            f"this supersedes {old_fact.get('value_text') or old_fact.get('object') or _fact_summary(old_fact)}"
        )
        if revision.get("effective_date"):
            text += f" effective {revision['effective_date']}"
        facts.append(
            {
                "id": revision["revision_id"],
                "fact": text.rstrip(".") + ".",
                "kind": "fact",
                "entities": _support_entity_ids(payload, revision.get("revision_source_fact_ids", [])),
                "tags": ["substrate", "revision_currentness"],
                "source_ids": list(revision.get("revision_source_fact_ids", [])),
                "metadata": {
                    "substrate_layer": "revision_currentness_layer",
                    **_episode_metadata(payload, list(revision.get("revision_source_fact_ids", []))),
                },
            }
        )

    for event in payload.get("events", []) or []:
        facts.append(
            {
                "id": event["event_id"],
                "fact": _event_summary(event),
                "kind": "fact",
                "entities": list(dict.fromkeys((event.get("participants") or []) + _support_entity_ids(payload, event.get("support_fact_ids", [])))),
                "tags": ["substrate", "event", event.get("event_type", "event")],
                "source_ids": list(event.get("support_fact_ids", [])),
                "metadata": {
                    "substrate_layer": "event_layer",
                    **_episode_metadata(payload, list(event.get("support_fact_ids", []))),
                },
            }
        )

    for record in payload.get("records", []) or []:
        facts.append(
            {
                "id": record["record_id"],
                "fact": _record_summary(record),
                "kind": "fact",
                "entities": _support_entity_ids(payload, record.get("support_fact_ids", [])),
                "tags": ["substrate", "record", record.get("record_type", "record")],
                "source_ids": list(record.get("support_fact_ids", [])),
                "metadata": {
                    "substrate_layer": "record_layer",
                    **_episode_metadata(payload, list(record.get("support_fact_ids", []))),
                },
            }
        )

    for edge in payload.get("edges", []) or []:
        edge_type = edge.get("edge_type")
        if edge_type in {"belongs_to_event", "belongs_to_record"}:
            continue
        if edge_type == "same_anchor":
            text = (
                f"{_node_summary(payload, edge['from_id'])} and {_node_summary(payload, edge['to_id'])} "
                f"share the same anchor: {edge.get('anchor_key')}."
            )
        else:
            text = (
                f"{_node_summary(payload, edge['from_id'])} {edge_type.replace('_', ' ')} "
                f"{_node_summary(payload, edge['to_id'])}."
            )
            if edge.get("edge_evidence_text"):
                text = text.rstrip(".") + f" Evidence: {edge['edge_evidence_text']}."
        facts.append(
            {
                "id": edge["edge_id"],
                "fact": text,
                "kind": "fact",
                "entities": _support_entity_ids(payload, edge.get("support_fact_ids", [])),
                "tags": ["substrate", "edge", edge_type],
                "source_ids": list(edge.get("support_fact_ids", [])),
                "metadata": {
                    "substrate_layer": "edge_layer",
                    **_episode_metadata(payload, list(edge.get("support_fact_ids", []))),
                },
            }
        )

    return facts


def _episode_id_from_fact_id(fact_id: str) -> str | None:
    match = re.match(r"^ep_(.+?)_f(?:_|$)", fact_id or "")
    if not match:
        return None
    return match.group(1)


async def extract_source_aggregation(
    *,
    source_id: str,
    source_kind: str,
    episodes: list[dict],
    source_facts: list[dict],
    model: str,
    call_extract_fn,
    prompt_overrides: dict[str, str] | None = None,
) -> dict | None:
    if not episodes or not source_facts:
        return None

    attempts: list[dict] = []
    system_prompt = _base_prompt(source_id, source_kind, episodes, source_facts, prompt_overrides=prompt_overrides)
    raw = await call_extract_fn(model, system_prompt, "", max_tokens=8192)
    parsed = _parse_json(raw)
    if parsed is not None:
        payload, locality_by_episode, source_text_by_episode = _build_payload_envelope(
            source_id,
            source_kind,
            episodes,
            source_facts,
            parsed,
        )
        attempts.append(payload)
        try:
            validate_source_aggregation_payload(
                payload,
                locality_metadata_by_episode=locality_by_episode,
                source_text_by_episode=source_text_by_episode,
            )
            return {
                "validation": {
                    "payload": payload,
                    "aggregation_status": "accepted",
                    "accepted_layers": ["atomic_fact_layer", "locality_layer", "revision_currentness_layer", "event_layer", "record_layer", "edge_layer"],
                    "dropped_layers": [],
                    "failure_reasons": [],
                },
                "derived_facts": flatten_source_aggregation_payload(payload),
            }
        except Exception as exc:
            repair_prompt = _repair_prompt(source_id, source_kind, episodes, source_facts, payload, str(exc), prompt_overrides=prompt_overrides)
    else:
        repair_prompt = _repair_prompt(source_id, source_kind, episodes, source_facts, None, "invalid_json", prompt_overrides=prompt_overrides)

    for _ in range(2):
        raw = await call_extract_fn(model, repair_prompt, "", max_tokens=8192)
        parsed = _parse_json(raw)
        if parsed is None:
            repair_prompt = _repair_prompt(source_id, source_kind, episodes, source_facts, None, "invalid_json", prompt_overrides=prompt_overrides)
            continue
        payload, locality_by_episode, source_text_by_episode = _build_payload_envelope(
            source_id,
            source_kind,
            episodes,
            source_facts,
            parsed,
        )
        attempts.append(payload)
        try:
            validate_source_aggregation_payload(
                payload,
                locality_metadata_by_episode=locality_by_episode,
                source_text_by_episode=source_text_by_episode,
            )
            return {
                "validation": {
                    "payload": payload,
                    "aggregation_status": "accepted",
                    "accepted_layers": ["atomic_fact_layer", "locality_layer", "revision_currentness_layer", "event_layer", "record_layer", "edge_layer"],
                    "dropped_layers": [],
                    "failure_reasons": [],
                },
                "derived_facts": flatten_source_aggregation_payload(payload),
            }
        except Exception as exc:
            repair_prompt = _repair_prompt(source_id, source_kind, episodes, source_facts, payload, str(exc), prompt_overrides=prompt_overrides)

    if not attempts:
        return None

    payload0, locality_by_episode, source_text_by_episode = _build_payload_envelope(
        source_id,
        source_kind,
        episodes,
        source_facts,
        {},
    )
    result = run_source_aggregation_validation_pipeline(
        attempts,
        locality_metadata_by_episode=locality_by_episode,
        episode_count=len(episodes),
        source_text_by_episode=source_text_by_episode,
    )
    validated_payload = result.get("payload")
    if not validated_payload:
        return {
            "validation": result,
            "derived_facts": [],
        }
    try:
        validate_source_aggregation_payload(
            validated_payload,
            locality_metadata_by_episode=locality_by_episode,
            source_text_by_episode=source_text_by_episode,
        )
    except SubstrateValidationError as exc:
        failure_reasons = list(result.get("failure_reasons", []))
        if exc.code not in failure_reasons:
            failure_reasons.append(exc.code)
        if "source_aggregation_retry_exhausted" not in failure_reasons:
            failure_reasons.append("source_aggregation_retry_exhausted")
        return {
            "validation": {
                **result,
                "payload": None,
                "aggregation_status": "failed",
                "accepted_layers": [],
                "dropped_layers": [],
                "failure_reasons": failure_reasons,
                "schema_error_count": int(result.get("schema_error_count", 0)) + (1 if exc.kind == "schema" else 0),
                "grounding_error_count": int(result.get("grounding_error_count", 0)) + (1 if exc.kind == "grounding" else 0),
            },
            "derived_facts": [],
        }
    return {
        "validation": result,
        "derived_facts": flatten_source_aggregation_payload(validated_payload),
    }
