#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio

import numpy as np

import src.memory as memory_mod
from src.coverage_recovery import (
    classify_coverage_query,
    compute_coverage_stats,
    merge_coverage_recovery_facts,
    needs_coverage_recovery,
)
from src.memory import MemoryServer


def _fact(
    fact_id: str,
    *,
    episode_id: str,
    source_id: str,
    fact_text: str,
    entities: list[str] | None = None,
    span_start: int = 0,
    span_end: int = 10,
) -> dict:
    return {
        "id": fact_id,
        "fact": fact_text,
        "entities": list(entities or []),
        "source_id": source_id,
        "support_spans": [
            {
                "episode_id": episode_id,
                "source_field": "raw_text",
                "start": span_start,
                "end": span_end,
            }
        ],
        "metadata": {
            "episode_id": episode_id,
            "episode_source_id": source_id,
        },
        "session": 1,
    }


def test_classify_coverage_query_detects_multi_item_shapes():
    assert classify_coverage_query("What activities have John and James done together?") == "list"
    assert classify_coverage_query("What do Jon and Gina have in common?") == "commonality"
    assert classify_coverage_query("Compare what changed before and after the move.") == "compare"
    assert classify_coverage_query("What car did John own?") == "none"


def test_needs_coverage_recovery_triggers_on_low_diversity_list():
    facts = [
        _fact(
            "f1",
            episode_id="conv_e01",
            source_id="conv",
            fact_text="They talked about a Prius purchase.",
            entities=["Prius"],
            span_start=0,
            span_end=20,
        )
    ]
    stats = compute_coverage_stats("list", facts, selected_source_count=1)
    assert needs_coverage_recovery("list", stats) is True


def test_needs_coverage_recovery_ignores_single_answer_queries():
    stats = compute_coverage_stats("none", [], selected_source_count=1)
    assert needs_coverage_recovery("none", stats) is False


def test_merge_coverage_recovery_facts_increases_diversity():
    existing = [
        _fact(
            "f1",
            episode_id="conv_e01",
            source_id="conv",
            fact_text="They bought a Prius.",
            entities=["Prius"],
            span_start=0,
            span_end=18,
        )
    ]
    recovered = [
        _fact(
            "f2",
            episode_id="conv_e02",
            source_id="conv",
            fact_text="They also bought a bicycle.",
            entities=["bicycle"],
            span_start=19,
            span_end=40,
        ),
        _fact(
            "f3",
            episode_id="conv_e01",
            source_id="conv",
            fact_text="They bought a Prius.",
            entities=["Prius"],
            span_start=0,
            span_end=18,
        ),
    ]

    merged = merge_coverage_recovery_facts("list", existing, recovered, max_facts=4)

    assert [fact["id"] for fact in merged] == ["f1", "f2"]
    stats = compute_coverage_stats("list", merged, selected_source_count=1)
    assert stats["distinct_episodes"] == 2
    assert stats["distinct_entities"] == 2
    assert stats["distinct_support_spans"] == 2


def test_merge_coverage_recovery_facts_ignores_noisy_duplicates():
    existing = [
        _fact(
            "f1",
            episode_id="conv_e01",
            source_id="conv",
            fact_text="They bought a Prius.",
            entities=["Prius"],
            span_start=0,
            span_end=18,
        ),
        _fact(
            "f2",
            episode_id="conv_e02",
            source_id="conv",
            fact_text="They joined a bowling league.",
            entities=["bowling"],
            span_start=20,
            span_end=44,
        ),
    ]
    recovered = [
        _fact(
            "f3",
            episode_id="conv_e02",
            source_id="conv",
            fact_text="They joined a bowling league.",
            entities=["bowling"],
            span_start=20,
            span_end=44,
        ),
        _fact(
            "f4",
            episode_id="conv_e01",
            source_id="conv",
            fact_text="They bought a Prius.",
            entities=["Prius"],
            span_start=0,
            span_end=18,
        ),
    ]

    merged = merge_coverage_recovery_facts("list", existing, recovered, max_facts=4)

    assert [fact["id"] for fact in merged] == ["f1", "f2"]


def test_recover_multi_item_coverage_packet_expands_within_selected_source(tmp_path, monkeypatch):
    ms = MemoryServer(str(tmp_path), "coverage_local")
    f1 = _fact(
        "f1",
        episode_id="conv_a_e01",
        source_id="conv_a",
        fact_text="They bought a Prius.",
        entities=["Prius"],
        span_start=0,
        span_end=18,
    )
    f2 = _fact(
        "f2",
        episode_id="conv_a_e02",
        source_id="conv_a",
        fact_text="They bought a bicycle.",
        entities=["bicycle"],
        span_start=19,
        span_end=38,
    )
    g1 = _fact(
        "g1",
        episode_id="conv_b_e01",
        source_id="conv_b",
        fact_text="They booked a trip to Boston.",
        entities=["Boston"],
        span_start=0,
        span_end=22,
    )
    ms._all_granular = [f1, f2, g1]
    ms._all_cross = []
    ms._data_dict = {
        "atomic_embs": np.asarray(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "cross_embs": np.zeros((0, 2), dtype=np.float32),
    }

    async def _fake_embed_query(_query):
        return np.asarray([1.0, 0.0], dtype=np.float32)

    seen_candidate_ids = {}

    def _fake_source_local_fact_sweep(_question, facts, _embeddings, **_kwargs):
        seen_candidate_ids["ids"] = [fact["id"] for fact in facts]
        return {
            "retrieved": [
                {"fact": facts[1], "score": 1.0},
                {"fact": facts[0], "score": 0.8},
            ],
            "trace": {"mode": "hybrid_fact_sweep"},
        }

    monkeypatch.setattr(memory_mod, "embed_query", _fake_embed_query)
    monkeypatch.setattr(memory_mod, "source_local_fact_sweep", _fake_source_local_fact_sweep)

    packet = {
        "retrieved_episode_ids": ["conv_a_e01"],
        "retrieved_fact_ids": ["f1"],
        "selector_config": {"budget": 4000, "supporting_facts_total": 4},
        "tuning_snapshot": {"packet": {"snippet_chars": 600}},
        "retrieval_families": ["conversation"],
        "search_family": "conversation",
        "temporal_trace": {"query_class": "semantic"},
    }
    episode_lookup = {
        "conv_a_e01": {
            "episode_id": "conv_a_e01",
            "source_id": "conv_a",
            "source_type": "conversation",
            "raw_text": "They bought a Prius.",
        },
        "conv_a_e02": {
            "episode_id": "conv_a_e02",
            "source_id": "conv_a",
            "source_type": "conversation",
            "raw_text": "They bought a bicycle.",
        },
        "conv_b_e01": {
            "episode_id": "conv_b_e01",
            "source_id": "conv_b",
            "source_type": "conversation",
            "raw_text": "They booked a trip to Boston.",
        },
    }

    updated_packet, recovered = asyncio.run(
        ms._recover_multi_item_coverage_packet(
            query="What purchases did we talk about?",
            packet=packet,
            episode_lookup=episode_lookup,
            fact_filter=lambda _fact: True,
        )
    )

    assert seen_candidate_ids["ids"] == ["f1", "f2"]
    assert recovered is not None
    assert [fact["id"] for fact in recovered] == ["f1", "f2"]
    assert updated_packet["retrieved_fact_ids"] == ["f1", "f2"]
    assert updated_packet["fact_episode_ids"] == ["conv_a_e01", "conv_a_e02"]
    assert "Prius" in updated_packet["context"]
    assert "bicycle" in updated_packet["context"]
