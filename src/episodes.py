#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Episode contract, validation, and lookup helpers.

This module owns the reusable episode-native data contract. Benchmark- or
report-specific heuristics must stay outside `src/`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

EPISODE_REQUIRED_FIELDS = {
    "episode_id",
    "source_type",
    "source_id",
    "topic_key",
    "state_label",
    "currentness",
    "raw_text",
    "provenance",
}

EPISODE_OPTIONAL_FIELDS = {
    "source_date",
    "scope_label",
    "authority_label",
    "notes",
}

RESULT_REQUIRED_FIELDS = {
    "query_id",
    "retrieved_fact_ids",
    "retrieved_episode_ids",
    "actual_injected_episode_ids",
    "packet_has_gold_evidence",
    "human_sufficiency",
    "answer",
    "score",
    "assembled_context",
}

RESULTS_REQUIRED_TOP_LEVEL = {"experiment", "results"}

PROVENANCE_KEYS = {"source_section_path", "block_ids", "raw_span"}

FACT_REQUIRED_METADATA = {
    "episode_id",
    "episode_source_id",
}


def validate_episode(ep: dict) -> list[str]:
    """Return list of validation errors (empty = valid)."""
    errors = []
    for field in EPISODE_REQUIRED_FIELDS:
        if field not in ep:
            errors.append(f"missing required field: {field}")
    if ep.get("source_type") not in ("document", "conversation"):
        errors.append(
            f"source_type must be 'document' or 'conversation', got {ep.get('source_type')!r}"
        )
    if ep.get("currentness") not in ("current", "outdated", "historical", "unknown"):
        errors.append(
            f"currentness must be current/outdated/historical/unknown, got {ep.get('currentness')!r}"
        )
    prov = ep.get("provenance")
    if not isinstance(prov, dict):
        errors.append("provenance must be a dict")
    elif not (PROVENANCE_KEYS & set(prov.keys())):
        errors.append(f"provenance must contain at least one of {sorted(PROVENANCE_KEYS)}")
    if not isinstance(ep.get("raw_text"), str) or not ep["raw_text"].strip():
        errors.append("raw_text must be a non-empty string")
    return errors


def validate_result(r: dict) -> list[str]:
    errors = []
    for field in RESULT_REQUIRED_FIELDS:
        if field not in r:
            errors.append(f"missing required field: {field}")
    if r.get("human_sufficiency") not in ("yes", "borderline", "no"):
        errors.append(
            f"human_sufficiency must be yes/borderline/no, got {r.get('human_sufficiency')!r}"
        )
    return errors


def validate_episode_fact(fact: dict) -> list[str]:
    errors = []
    meta = fact.get("metadata")
    if not isinstance(meta, dict):
        errors.append("fact must have metadata dict")
        return errors
    for field in FACT_REQUIRED_METADATA:
        if field not in meta:
            errors.append(f"fact metadata missing required field: {field}")
    return errors


def _validate_corpus_envelope(corpus: dict) -> list[str]:
    errors = []
    if not isinstance(corpus, dict):
        return ["corpus must be a dict"]
    if "documents" not in corpus:
        errors.append("corpus missing required 'documents' key")
    elif not isinstance(corpus["documents"], list):
        errors.append("corpus['documents'] must be a list")
    else:
        for i, doc in enumerate(corpus["documents"]):
            if not isinstance(doc, dict):
                errors.append(f"documents[{i}] must be a dict")
            elif "doc_id" not in doc:
                errors.append(f"documents[{i}] missing required 'doc_id'")
            elif "episodes" not in doc:
                errors.append(f"documents[{i}] (doc_id={doc.get('doc_id')}) missing 'episodes'")
            elif not isinstance(doc["episodes"], list):
                errors.append(
                    f"documents[{i}] (doc_id={doc.get('doc_id')}) 'episodes' must be a list"
                )
    return errors


def load_episode_corpus(path: str | Path, strict: bool = True) -> dict:
    with open(path) as f:
        corpus = json.load(f)
    errors = _validate_corpus_envelope(corpus)
    if not errors:
        for doc in corpus["documents"]:
            for ep in doc.get("episodes", []):
                ep_errors = validate_episode(ep)
                if ep_errors:
                    errors.append(f"{ep.get('episode_id', '?')}: {ep_errors}")
    if errors and strict:
        msg = f"{len(errors)} corpus validation errors:\n"
        msg += "\n".join(f"  {e}" for e in errors[:10])
        raise ValueError(msg)
    return corpus


def save_episode_corpus(corpus: dict, path: str | Path):
    errors = _validate_corpus_envelope(corpus)
    if errors:
        msg = "Refusing to save corpus with envelope errors:\n"
        msg += "\n".join(f"  {e}" for e in errors)
        raise ValueError(msg)
    for doc in corpus["documents"]:
        for ep in doc.get("episodes", []):
            ep_errors = validate_episode(ep)
            if ep_errors:
                errors.append(f"{ep.get('episode_id', '?')}: {ep_errors}")
    if errors:
        msg = f"Refusing to save corpus with {len(errors)} validation errors:\n"
        msg += "\n".join(f"  {e}" for e in errors[:10])
        raise ValueError(msg)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)


def load_results(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_results(results: dict, path: str | Path):
    if not isinstance(results, dict):
        raise ValueError("results must be a dict")
    missing_top = RESULTS_REQUIRED_TOP_LEVEL - set(results.keys())
    if missing_top:
        raise ValueError(f"results missing required top-level keys: {sorted(missing_top)}")
    if not isinstance(results["results"], list):
        raise ValueError("results['results'] must be a list")
    errors = []
    for r in results["results"]:
        r_errors = validate_result(r)
        if r_errors:
            errors.append(f"{r.get('query_id', '?')}: {r_errors}")
    if errors:
        msg = f"Refusing to save results with {len(errors)} validation errors:\n"
        msg += "\n".join(f"  {e}" for e in errors[:10])
        raise ValueError(msg)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def corpus_to_flat_episodes(corpus: dict) -> list[dict]:
    episodes = []
    for doc in corpus.get("documents", []):
        episodes.extend(doc.get("episodes", []))
    return episodes


def build_episode_lookup(corpus: dict) -> dict[str, dict]:
    return {ep["episode_id"]: ep for ep in corpus_to_flat_episodes(corpus)}


def build_episode_raw_index(corpus: dict) -> dict[str, str]:
    return {ep["episode_id"]: ep["raw_text"] for ep in corpus_to_flat_episodes(corpus)}


def build_facts_by_episode(ep_facts: list[dict]) -> dict[str, list[dict]]:
    by_episode = defaultdict(list)
    for fact in ep_facts:
        ep_id = episode_id_from_fact(fact)
        if ep_id:
            by_episode[ep_id].append(fact)
    return dict(by_episode)


def episode_id_from_fact(fact: dict) -> str:
    return (fact.get("metadata") or {}).get("episode_id", "")


def episode_source_id(ep_or_fact: dict) -> str:
    if "source_id" in ep_or_fact:
        return ep_or_fact.get("source_id", "")
    return (ep_or_fact.get("metadata") or {}).get("episode_source_id", "")


def episode_text(ep: dict) -> str:
    return ep.get("raw_text", "")

