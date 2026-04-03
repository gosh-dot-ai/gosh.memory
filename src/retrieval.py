#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Production retrieval utilities.

This module intentionally contains only generic building blocks that are
owned by the final production runtime:
- base query-type detection
- a dependency-free BM25 index

Legacy adaptive fact-tier retrieval lives outside `src/` in
`multibench.legacy.retrieval`.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .common import STOP_WORDS, normalize_term_token
from .temporal_planner import classify_temporal_query

TEMPORAL_OVERRIDE_RE = re.compile(
    r"\bhow many (days|weeks|months|years)\b", re.I
)

TEMPORAL_RE = re.compile(
    r"\b(when did|first time|last time|how long ago|what date|in what year|"
    r"how many days|sequence of|chronolog|earlier than|later than|"
    r"prior to|following the|before or after|before|after|"
    r"how many weeks|how many months)\b",
    re.I,
)

COUNTING_RE = re.compile(
    r"\b(how many|how often|count|number of times|total number|"
    r"total weight|total cost|sum of|how much did)\b",
    re.I,
)

CURRENT_RE = re.compile(
    r"\b(current|currently|now|latest|most recent|still|anymore|"
    r"updated to|changed to|moved to|now lives|works now|"
    r"where does .* live|where did .* move)\b",
    re.I,
)

RULE_RE = re.compile(
    r"\b(rule|policy|guideline|procedure|protocol|standard|"
    r"requirement|must|shall|always|never|whenever)\b",
    re.I,
)

SYNTHESIS_RE = re.compile(
    r"\b(summary|summarize|overall|generally|typically|tend to|"
    r"prefer|like|enjoy|favorite|recommend|suggest|what kind|"
    r"tell me about|how do I usually|what do I)\b",
    re.I,
)

PROSPECTIVE_RE = re.compile(
    r"\b(assigned to|upcoming|pending|scheduled|todo|next step)\b", re.I
)

CAUSAL_RE = re.compile(
    r"\b(why|because|reason|cause|led to|result of)\b", re.I
)

WORD_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)?")
COMPOSITIONAL_SPLIT_RE = re.compile(r"\b(?:while\s+also|while|so\s+that)\b", re.I)
COMPOSITIONAL_COMPONENT_IGNORE_TOKENS = STOP_WORDS | {
    "what",
    "which",
    "while",
    "also",
    "make",
    "making",
    "help",
    "helping",
    "keep",
    "keeping",
    "let",
    "letting",
    "would",
    "could",
    "should",
    "doing",
    "thing",
    "things",
    "activity",
    "activities",
    "indoor",
    "outdoor",
}


def detect_query_type(query: str) -> str:
    """Classify query into coarse retrieval types.

    This is the base query-type classifier used by the production runtime.
    Query operators such as ordinal/commonality/list_set are detected
    separately in the episode query layer.
    """
    if len(re.findall(r"\blabel:\s*\d+", query[:3000], re.I)) >= 2:
        return "icl"

    if re.search(
        r"\b(write a summary|write a .{0,20}summary|summarize (?:everything|all|the)"
        r"|summary of about \d|summary of .{0,30} words"
        r"|\d+ (?:to \d+ )?words? summary"
        r"|words?[. ] only write)\b",
        query,
        re.I,
    ):
        return "summarize"

    if COUNTING_RE.search(query):
        if TEMPORAL_OVERRIDE_RE.search(query):
            return "temporal"
        return "counting"
    if re.match(r"^\s*when\b", query, re.I):
        return "temporal"
    if classify_temporal_query(query) == "calendar":
        return "temporal"
    if TEMPORAL_OVERRIDE_RE.search(query):
        return "temporal"
    if TEMPORAL_RE.search(query):
        return "temporal"
    if CURRENT_RE.search(query):
        return "current"
    if RULE_RE.search(query):
        return "rule"
    if SYNTHESIS_RE.search(query):
        return "synthesis"
    if PROSPECTIVE_RE.search(query):
        return "prospective"
    if CAUSAL_RE.search(query):
        return "causal"
    return "default"


class BM25Index:
    """Simple BM25 index with no external dependency."""

    def __init__(self, documents, ids, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = len(documents)
        self.ids = ids
        self.doc_freqs = {}
        self.doc_lens = []
        self.doc_terms = []

        for doc in documents:
            terms = self._tokenize(doc)
            self.doc_terms.append(Counter(terms))
            self.doc_lens.append(len(terms))
            for term in set(terms):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.avgdl = sum(self.doc_lens) / max(self.N, 1)

    def _tokenize(self, text: str) -> list[str]:
        tokens = []
        for word in re.findall(r"\w+", text):
            lower = word.lower()
            if lower in STOP_WORDS or len(lower) <= 1:
                continue
            norm = normalize_term_token(lower)
            if norm and norm not in STOP_WORDS and len(norm) > 1:
                tokens.append(norm)
        return tokens

    def score(self, query_terms: list[str], doc_idx: int) -> float:
        doc = self.doc_terms[doc_idx]
        dl = self.doc_lens[doc_idx]
        score = 0.0
        for term in query_terms:
            if term not in doc:
                continue
            tf = doc[term]
            df = self.doc_freqs.get(term, 0)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * numerator / denominator
        return score

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        query_terms = self._tokenize(query)
        if not query_terms:
            return []
        scored = []
        for idx in range(self.N):
            score = self.score(query_terms, idx)
            if score > 0:
                scored.append({"id": self.ids[idx], "s": score})
        scored.sort(key=lambda row: -row["s"])
        return scored[:top_k]


def _token_set(text: str) -> set[str]:
    return {
        normalize_term_token(token)
        for token in WORD_RE.findall(text)
        if normalize_term_token(token)
    }


def _tokens_match(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    if min(len(left), len(right)) < 4:
        return False
    return left.startswith(right) or right.startswith(left)


def _token_overlap_count(fact_tokens: set[str], target_tokens: set[str]) -> int:
    return sum(
        1
        for fact_token in fact_tokens
        for target_token in target_tokens
        if _tokens_match(fact_token, target_token)
    )



def _compositional_component_groups(question: str, qf: dict) -> list[set[str]]:
    operator_plan = qf.get("operator_plan") or {}
    if not operator_plan.get("compositional", {}).get("enabled"):
        return []
    entity_tokens = {
        token
        for tokens in (qf.get("entity_phrase_tokens") or {}).values()
        for token in tokens
    }
    parts = [segment.strip() for segment in COMPOSITIONAL_SPLIT_RE.split(question) if segment.strip()]
    if len(parts) <= 1 and " both " in question.lower() and " and " in question.lower():
        tail = question.split("both", 1)[-1]
        parts = [segment.strip() for segment in re.split(r"\band\b", tail, maxsplit=1, flags=re.I) if segment.strip()]
    groups: list[set[str]] = []
    for part in parts:
        tokens = {
            normalize_term_token(token)
            for token in WORD_RE.findall(part)
            if normalize_term_token(token)
        }
        tokens -= COMPOSITIONAL_COMPONENT_IGNORE_TOKENS
        tokens -= entity_tokens
        if not tokens:
            continue
        if any(tokens <= existing for existing in groups):
            continue
        groups = [existing for existing in groups if not existing < tokens]
        groups.append(tokens)
    return groups


def get_entity_matched(question: str, facts: list[dict]) -> list[int]:
    """Return indices of facts whose entities/text overlap with question entities."""
    if not facts:
        return []
    q_lower = question.lower()
    known_ents = set()
    for fact in facts:
        for entity in fact.get("entities", []):
            name = entity if isinstance(entity, str) else str(entity)
            if len(name) > 1:
                known_ents.add(name.lower())
    matched_ents = {ent for ent in known_ents if ent in q_lower}
    for match in re.finditer(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", question):
        matched_ents.add(match.group().lower())
    if not matched_ents:
        return []
    matched = []
    for idx, fact in enumerate(facts):
        fact_ents = [
            entity.lower() if isinstance(entity, str) else str(entity).lower()
            for entity in fact.get("entities", [])
        ]
        fact_text = fact.get("fact", "").lower()
        for query_ent in matched_ents:
            if any(query_ent in fact_ent or fact_ent in query_ent for fact_ent in fact_ents) or query_ent in fact_text:
                matched.append(idx)
                break
    return matched


def reciprocal_rank_fusion(*ranked_lists: list[dict], k: int = 60, top_k: int = 15) -> list[dict]:
    """Merge ranked lists via reciprocal-rank fusion."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            item_id = str(item.get("id", ""))
            if not item_id:
                continue
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return [
        {"id": item_id, "s": score}
        for item_id, score in sorted(scores.items(), key=lambda row: -row[1])[:top_k]
    ]


def _fact_index_text(fact: dict) -> str:
    text = fact.get("fact", "")
    speaker = fact.get("speaker")
    if speaker:
        text += f" {speaker}"
    speaker_role = fact.get("speaker_role")
    if speaker_role:
        text += f" {speaker_role}"
    return text


def source_local_fact_sweep(
    question: str,
    facts: list[dict],
    embeddings: np.ndarray,
    *,
    query_embedding: np.ndarray | None,
    top_k: int = 12,
    bm25_pool: int = 24,
    vector_pool: int = 24,
    entity_pool: int = 16,
    rrf_k: int = 60,
) -> dict:
    """Rank facts inside one already-chosen source using BM25 + vector + entity fusion."""
    if not facts:
        return {"retrieved": [], "trace": {"mode": "empty", "candidate_count": 0}}

    from .episode_features import extract_query_features
    from .episode_packet import _fact_list_item_keys, _fact_plan_commitment_score, _query_requests_plan_meetup

    bm25 = BM25Index([_fact_index_text(fact) for fact in facts], [str(i) for i in range(len(facts))])
    bm25_hits = bm25.search(question, top_k=max(top_k, bm25_pool))

    entity_hits = [
        {"id": str(idx), "s": 1.0 / (rank + 1)}
        for rank, idx in enumerate(get_entity_matched(question, facts)[: max(top_k, entity_pool)])
    ]
    qf = extract_query_features(question)
    list_plan = (qf.get("operator_plan") or {}).get("list_set") or {}
    compositional_groups = _compositional_component_groups(question, qf)
    plan_query = _query_requests_plan_meetup(qf)
    item_pool = max(top_k * 3, entity_pool, 24) if list_plan.get("enabled") else max(top_k, entity_pool)
    component_pool = max(top_k * 2, entity_pool, 16)
    list_item_hits: list[dict] = []
    component_hits: list[dict] = []
    if list_plan.get("enabled"):
        item_rows = []
        for idx, fact in enumerate(facts):
            text = fact.get("fact", "")
            item_keys = tuple(dict.fromkeys(_fact_list_item_keys(text, qf)))
            if not item_keys:
                continue
            plan_score = float(_fact_plan_commitment_score(text, qf))
            if plan_query and plan_score <= 0:
                continue
            lexical_bonus = 0.0
            fact_tokens = {normalize_term_token(token) for token in re.findall(r"[A-Za-z0-9']+", text)}
            lexical_bonus += 0.2 * len(fact_tokens & set(qf.get("words", set())))
            item_rows.append((idx, float(len(item_keys)) + max(plan_score, 0.0) + lexical_bonus))
        item_rows.sort(key=lambda row: (-row[1], row[0]))
        list_item_hits = [
            {"id": str(idx), "s": score}
            for idx, score in item_rows[:item_pool]
        ]
    if compositional_groups:
        component_rows = []
        for idx, fact in enumerate(facts):
            fact_tokens = _token_set(fact.get("fact", ""))
            group_overlap = {
                group_idx: _token_overlap_count(fact_tokens, component)
                for group_idx, component in enumerate(compositional_groups)
            }
            matched_groups = tuple(
                group_idx
                for group_idx, overlap in group_overlap.items()
                if overlap > 0
            )
            if not matched_groups:
                continue
            match_score = float(len(matched_groups))
            match_score += 0.2 * sum(group_overlap[group_idx] for group_idx in matched_groups)
            component_rows.append(
                {
                    "idx": idx,
                    "score": match_score,
                    "matched_groups": matched_groups,
                    "group_overlap": group_overlap,
                }
            )
        ordered_component_rows = []
        remaining_component_rows = list(component_rows)
        covered_components: set[int] = set()
        while remaining_component_rows:
            ranked = []
            for row in remaining_component_rows:
                new_components = tuple(
                    group_idx for group_idx in row["matched_groups"] if group_idx not in covered_components
                )
                if not new_components:
                    continue
                ranked.append(
                    (
                        -len(new_components),
                        -sum(row["group_overlap"].get(group_idx, 0) for group_idx in new_components),
                        -float(row["score"]),
                        row["idx"],
                        row,
                        new_components,
                    )
                )
            if not ranked:
                break
            ranked.sort(key=lambda entry: entry[:-2])
            row = ranked[0][-2]
            new_components = ranked[0][-1]
            ordered_component_rows.append(row)
            covered_components.update(new_components)
            remaining_component_rows = [candidate for candidate in remaining_component_rows if candidate is not row]
        remaining_component_rows.sort(
            key=lambda row: (
                -sum(row["group_overlap"].values()),
                -len(row["matched_groups"]),
                -float(row["score"]),
                row["idx"],
            )
        )
        component_hits = [
            {"id": str(row["idx"]), "s": float(row["score"])}
            for row in (ordered_component_rows + remaining_component_rows)[:component_pool]
        ]

    vector_hits: list[dict] = []
    if (
        query_embedding is not None
        and isinstance(embeddings, np.ndarray)
        and len(embeddings) == len(facts)
        and len(facts) > 0
        and embeddings.ndim == 2
        and query_embedding.ndim == 1
        and embeddings.shape[1] == query_embedding.shape[0]
    ):
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        f_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        sims = q_norm @ f_norms.T
        order = np.argsort(-sims)[: max(top_k, vector_pool)]
        vector_hits = [{"id": str(int(idx)), "s": float(sims[int(idx)])} for idx in order]

    fused = reciprocal_rank_fusion(
        vector_hits,
        bm25_hits,
        entity_hits,
        list_item_hits,
        component_hits,
        k=rrf_k,
        top_k=max(top_k, bm25_pool, vector_pool, entity_pool, item_pool, component_pool),
    )
    if list_plan.get("enabled"):
        fused_rows = []
        for order, item in enumerate(fused):
            idx = int(item["id"])
            fact = facts[idx]
            text = fact.get("fact", "")
            item_keys = tuple(dict.fromkeys(_fact_list_item_keys(text, qf)))
            plan_score = float(_fact_plan_commitment_score(text, qf))
            fused_rows.append(
                {
                    "item": item,
                    "order": order,
                    "fact": fact,
                    "item_keys": item_keys,
                    "plan_score": plan_score,
                }
            )

        ordered_rows = []
        remaining = list(fused_rows)
        covered_item_keys: set[str] = set()

        while remaining:
            ranked = []
            for row in remaining:
                new_keys = tuple(key for key in row["item_keys"] if key not in covered_item_keys)
                if not new_keys:
                    continue
                if plan_query and row["plan_score"] <= 0:
                    continue
                ranked.append(
                    (
                        -(1 if row["plan_score"] > 0 else 0),
                        -len(new_keys),
                        -len(row["item_keys"]),
                        -float(row["item"]["s"]),
                        row["order"],
                        row,
                        new_keys,
                    )
                )
            if not ranked:
                break
            ranked.sort(key=lambda entry: entry[:-2])
            _key = ranked[0]
            row = _key[-2]
            new_keys = _key[-1]
            ordered_rows.append(row)
            covered_item_keys.update(new_keys)
            remaining = [candidate for candidate in remaining if candidate is not row]

        remaining.sort(
            key=lambda row: (
                -(1 if row["plan_score"] > 0 else 0),
                1 if row["plan_score"] < 0 else 0,
                -(1 if row["item_keys"] else 0),
                -len(row["item_keys"]),
                -float(row["item"]["s"]),
                row["order"],
            )
        )
        fused = [row["item"] for row in ordered_rows + remaining]
    elif compositional_groups:
        fused_rows = []
        for order, item in enumerate(fused):
            idx = int(item["id"])
            fact_tokens = _token_set(facts[idx].get("fact", ""))
            group_overlap = {
                group_idx: _token_overlap_count(fact_tokens, component)
                for group_idx, component in enumerate(compositional_groups)
            }
            component_ids = tuple(
                group_idx
                for group_idx, overlap in group_overlap.items()
                if overlap > 0
            )
            fused_rows.append(
                {
                    "item": item,
                    "order": order,
                    "component_ids": component_ids,
                    "component_overlap": group_overlap,
                }
            )
        ordered_rows = []
        remaining = list(fused_rows)
        covered_components: set[int] = set()
        while remaining:
            ranked = []
            for row in remaining:
                new_components = tuple(component for component in row["component_ids"] if component not in covered_components)
                if not new_components:
                    continue
                ranked.append(
                    (
                        -len(new_components),
                        -sum(row["component_overlap"].get(component, 0) for component in new_components),
                        -len(row["component_ids"]),
                        -float(row["item"]["s"]),
                        row["order"],
                        row,
                        new_components,
                    )
                )
            if not ranked:
                break
            ranked.sort(key=lambda entry: entry[:-2])
            row = ranked[0][-2]
            new_components = ranked[0][-1]
            ordered_rows.append(row)
            covered_components.update(new_components)
            remaining = [candidate for candidate in remaining if candidate is not row]
        remaining.sort(
            key=lambda row: (
                -len(row["component_ids"]),
                -sum(row["component_overlap"].values()),
                -float(row["item"]["s"]),
                row["order"],
            )
        )
        fused = [row["item"] for row in ordered_rows + remaining]

    retrieved = []
    for item in fused[:top_k]:
        idx = int(item["id"])
        retrieved.append(
            {
                "fact": facts[idx],
                "score": float(item["s"]),
                "bm25": next((float(row["s"]) for row in bm25_hits if row["id"] == item["id"]), 0.0),
                "vector": next((float(row["s"]) for row in vector_hits if row["id"] == item["id"]), 0.0),
                "entity": next((float(row["s"]) for row in entity_hits if row["id"] == item["id"]), 0.0),
            }
        )

    return {
        "retrieved": retrieved,
        "trace": {
            "mode": "hybrid_fact_sweep",
            "candidate_count": len(facts),
            "bm25_pool": bm25_pool,
            "vector_pool": vector_pool,
            "entity_pool": entity_pool,
            "list_item_pool": item_pool,
            "component_pool": component_pool,
            "rrf_k": rrf_k,
            "top_k": top_k,
            "bm25_hits": bm25_hits[: min(len(bm25_hits), 8)],
            "entity_hits": entity_hits[: min(len(entity_hits), 8)],
            "list_item_hits": list_item_hits[: min(len(list_item_hits), 8)],
            "component_hits": component_hits[: min(len(component_hits), 8)],
            "vector_hits": vector_hits[: min(len(vector_hits), 8)],
            "selected_fact_ids": [
                row["fact"].get("id", "")
                for row in retrieved[: min(len(retrieved), 12)]
            ],
        },
    }
