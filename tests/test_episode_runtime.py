"""Tests for the reusable src episode runtime."""

import asyncio
from collections import Counter

import pytest

import src.episode_packet as episode_packet
import src.episode_retrieval as episode_retrieval
from karnali.eval.episode_common import build_episode_context
from src.episode_features import extract_query_features
from src.episode_packet import (
    _fact_content_tokens,
    _pseudo_facts_from_episode,
    _select_bounded_chain_seed_facts,
    build_bounded_chain_candidate_bundle,
    build_context_from_retrieved_facts,
    build_context_from_selected_episodes,
)
from src.episode_retrieval import DEFAULT_SELECTION_CONFIG
from src.episodes import build_episode_lookup
from src.memory import MemoryServer, build_episode_hybrid_context
from src.retrieval import BM25Index


def _make_corpus():
    return {
        "documents": [
            {
                "doc_id": "DOC-1",
                "episodes": [
                    {
                        "episode_id": "DOC-1_e01",
                        "source_type": "document",
                        "source_id": "DOC-1",
                        "source_date": "2026-01-01",
                        "topic_key": "route 1 canonical length",
                        "state_label": "canonical",
                        "currentness": "current",
                        "raw_text": "Route 1 final approved length is 14.3 km.",
                        "provenance": {"block_ids": ["DOC-1_b000"]},
                    },
                    {
                        "episode_id": "DOC-1_e02",
                        "source_type": "document",
                        "source_id": "DOC-1",
                        "source_date": "2026-01-01",
                        "topic_key": "route 2 draft length",
                        "state_label": "draft",
                        "currentness": "outdated",
                        "raw_text": "Draft route 2 option measured 14.1 km before final approval.",
                        "provenance": {"block_ids": ["DOC-1_b001"]},
                    },
                ],
            }
        ]
    }


def _make_facts():
    return [
        {
            "id": "f_current",
            "session": 1,
            "fact": "Route 1 final approved length is 14.3 km.",
            "metadata": {"episode_id": "DOC-1_e01", "episode_source_id": "DOC-1"},
        },
        {
            "id": "f_old",
            "session": 2,
            "fact": "Draft route 2 option measured 14.1 km before final approval.",
            "metadata": {"episode_id": "DOC-1_e02", "episode_source_id": "DOC-1"},
        },
    ]


def test_build_context_from_selected_episodes_surfaces_list_evidence_from_same_source_for_indoor_queries():
    query = "What indoor activities has Alex pursued with his girlfriend?"
    qf = extract_query_features(query)
    episode_lookup = {
        "SRC_e01": {
            "episode_id": "SRC_e01",
            "source_id": "SRC",
            "source_type": "conversation",
            "topic_key": "one",
            "state_label": "session",
            "currentness": "current",
            "raw_text": "Alex: We went on a hike outside last weekend.",
        },
        "SRC_e02": {
            "episode_id": "SRC_e02",
            "source_id": "SRC",
            "source_type": "conversation",
            "topic_key": "two",
            "state_label": "session",
            "currentness": "current",
            "raw_text": "Alex: My GF and I had a great experience volunteering at a pet shelter on Monday. It was really rewarding.",
        },
    }
    context, injected, _fact_ids = build_context_from_selected_episodes(
        query,
        ["SRC_e01"],
        episode_lookup,
        facts_by_episode={},
        query_features=qf,
    )
    assert injected == ["SRC_e01"]
    assert "RAW LIST EVIDENCE:" in context
    assert "SRC_e02" in context
    assert "volunteering at a pet shelter" in context


def test_build_context_from_selected_episodes_skips_raw_list_evidence_for_commonality_queries():
    query = "What kind of interests do Joanna and Nate share?"
    qf = extract_query_features(query)
    episode_lookup = {
        "SRC_e01": {
            "episode_id": "SRC_e01",
            "source_id": "SRC",
            "source_type": "conversation",
            "topic_key": "one",
            "state_label": "session",
            "currentness": "current",
            "raw_text": "Joanna: Nate and I have similar interests, especially watching movies together.",
        },
        "SRC_e02": {
            "episode_id": "SRC_e02",
            "source_id": "SRC",
            "source_type": "conversation",
            "topic_key": "two",
            "state_label": "session",
            "currentness": "current",
            "raw_text": "Joanna: We also made desserts together after movie night.",
        },
    }
    context, injected, _fact_ids = build_context_from_selected_episodes(
        query,
        ["SRC_e01"],
        episode_lookup,
        facts_by_episode={},
        query_features=qf,
    )
    assert injected == ["SRC_e01"]
    assert "RAW LIST EVIDENCE:" not in context


def test_build_context_from_retrieved_facts_surfaces_list_evidence_from_same_source_for_indoor_queries():
    query = "What indoor activities has Alex pursued with his girlfriend?"
    qf = extract_query_features(query)
    episode_lookup = {
        "SRC_e01": {
            "episode_id": "SRC_e01",
            "source_id": "SRC",
            "source_type": "conversation",
            "topic_key": "one",
            "state_label": "session",
            "currentness": "current",
            "raw_text": "Alex: We went on a hike outside last weekend.",
        },
        "SRC_e02": {
            "episode_id": "SRC_e02",
            "source_id": "SRC",
            "source_type": "conversation",
            "topic_key": "two",
            "state_label": "session",
            "currentness": "current",
            "raw_text": "Alex: My GF and I had a great experience volunteering at a pet shelter on Monday. It was really rewarding.",
        },
    }
    retrieved_fact = {
        "id": "f1",
        "session": 1,
        "fact": "Alex went on a hike outside last weekend.",
        "metadata": {"episode_id": "SRC_e01", "episode_source_id": "SRC"},
    }
    context, injected = build_context_from_retrieved_facts(
        [retrieved_fact],
        episode_lookup,
        {"f1": retrieved_fact},
        question=query,
        query_features=qf,
    )
    assert injected == ["SRC_e01"]
    assert "RAW LIST EVIDENCE:" in context
    assert "SRC_e02" in context
    assert "volunteering at a pet shelter" in context


def test_default_selection_config_is_generic():
    assert DEFAULT_SELECTION_CONFIG["budget"] == 8000
    assert DEFAULT_SELECTION_CONFIG["max_sources_per_family"] == 2
    assert DEFAULT_SELECTION_CONFIG["late_fusion_per_family"] == 3
    assert DEFAULT_SELECTION_CONFIG["chainage_overlap_bonus"] == 0.0
    assert "permit_diversity" not in DEFAULT_SELECTION_CONFIG
    assert "commercial_pair_preserve" not in DEFAULT_SELECTION_CONFIG


def test_bounded_chain_same_source_scan_knob_is_consumed(monkeypatch):
    corpus = {
        "documents": [
            {
                "doc_id": "DOC-CHAIN",
                "episodes": [
                    {
                        "episode_id": "DOC-CHAIN_e01",
                        "source_type": "document",
                        "source_id": "DOC-CHAIN",
                        "source_date": "2026-01-01",
                        "topic_key": "citizenship",
                        "state_label": "fact",
                        "currentness": "unknown",
                        "raw_text": 'Nate "Tiny" Archibald is a citizen of United States of America.',
                        "provenance": {"block_ids": ["DOC-CHAIN_b001"]},
                    },
                    {
                        "episode_id": "DOC-CHAIN_e02",
                        "source_type": "document",
                        "source_id": "DOC-CHAIN",
                        "source_date": "2026-01-02",
                        "topic_key": "distractor",
                        "state_label": "fact",
                        "currentness": "unknown",
                        "raw_text": "West Coast hip hop was created in the country of United States of America.",
                        "provenance": {"block_ids": ["DOC-CHAIN_b002"]},
                    },
                    {
                        "episode_id": "DOC-CHAIN_e03",
                        "source_type": "document",
                        "source_id": "DOC-CHAIN",
                        "source_date": "2026-01-03",
                        "topic_key": "resolver",
                        "state_label": "fact",
                        "currentness": "unknown",
                        "raw_text": "The official language of United States of America is German.",
                        "provenance": {"block_ids": ["DOC-CHAIN_b003"]},
                    },
                ],
            }
        ]
    }
    episode_lookup = build_episode_lookup(corpus)
    texts = [episode_lookup[ep_id]["raw_text"] for ep_id in sorted(episode_lookup)]
    ids = sorted(episode_lookup)
    bm25 = BM25Index(texts, ids)
    query = 'In which language are the official documents written in the country of citizenship of Nate "Tiny" Archibald?'
    config = dict(DEFAULT_SELECTION_CONFIG)
    config["max_candidates"] = 1
    config["max_episodes_default"] = 3
    real_get = episode_retrieval.get_tuning_section

    def _fake_get_tuning_section(*path):
        if path == ("operators",):
            section = dict(real_get(*path))
            section["bounded_chain_expand_same_source_episodes"] = enabled
            section["bounded_chain_same_source_scan_limit"] = 16
            return section
        return real_get(*path)

    enabled = False
    monkeypatch.setattr(episode_retrieval, "get_tuning_section", _fake_get_tuning_section)
    disabled = episode_retrieval.choose_episode_ids_with_trace(query, bm25, episode_lookup, config)

    enabled = True
    monkeypatch.setattr(episode_retrieval, "get_tuning_section", _fake_get_tuning_section)
    expanded = episode_retrieval.choose_episode_ids_with_trace(query, bm25, episode_lookup, config)

    assert disabled["selected_ids"] == ["DOC-CHAIN_e01"]
    assert "DOC-CHAIN_e03" in expanded["selected_ids"]
    post_gate = expanded["trace"]["post_source_gate"]
    assert any(row["episode_id"] == "DOC-CHAIN_e03" for row in post_gate)


def test_bounded_chain_candidate_bundle_relation_cap_is_consumed(monkeypatch):
    question = 'In which language are the official documents written in the country of citizenship of Nate "Tiny" Archibald?'
    seed = {
        "id": "seed",
        "fact": 'Nate "Tiny" Archibald is a citizen of United States of America.',
        "metadata": {"episode_id": "DOC-CHAIN_e01", "episode_source_id": "DOC-CHAIN"},
    }
    distractor_a = {
        "id": "d1",
        "fact": "West Coast hip hop was created in the country of United States of America.",
        "metadata": {"episode_id": "DOC-CHAIN_e02", "episode_source_id": "DOC-CHAIN"},
    }
    distractor_b = {
        "id": "d2",
        "fact": "Little Miss Sunshine was created in the country of United States of America.",
        "metadata": {"episode_id": "DOC-CHAIN_e03", "episode_source_id": "DOC-CHAIN"},
    }
    resolver = {
        "id": "r1",
        "fact": "The official language of United States of America is German.",
        "metadata": {"episode_id": "DOC-CHAIN_e04", "episode_source_id": "DOC-CHAIN"},
    }
    facts = [seed, distractor_a, distractor_b, resolver]
    real_get = episode_packet.get_tuning_section

    def _fake_get_tuning_section(*path):
        section = dict(real_get(*path))
        if path == ("operators",):
            section["bounded_chain_bundle_relation_signature_cap"] = relation_cap
        return section

    relation_cap = 0
    monkeypatch.setattr(episode_packet, "get_tuning_section", _fake_get_tuning_section)
    unbounded = build_bounded_chain_candidate_bundle(
        question,
        [seed],
        facts,
        max_candidates=3,
        query_specificity_bonus=0.35,
    )

    relation_cap = 1
    monkeypatch.setattr(episode_packet, "get_tuning_section", _fake_get_tuning_section)
    capped = build_bounded_chain_candidate_bundle(
        question,
        [seed],
        facts,
        max_candidates=3,
        query_specificity_bonus=0.35,
    )

    assert [fact["id"] for fact in unbounded["facts"]] == ["seed", "d1", "d2"]
    assert [fact["id"] for fact in capped["facts"]] == ["seed", "d1", "r1"]
    assert capped["trace"]["relation_signature_cap"] == 1
    assert capped["trace"]["skipped_relation_signature"] >= 1


def test_bounded_chain_candidate_bundle_frontier_overlap_weight_is_consumed(monkeypatch):
    question = 'In which language are the official documents written in the country of citizenship of Nate "Tiny" Archibald?'
    seed = {
        "id": "seed",
        "fact": 'Nate "Tiny" Archibald is a citizen of United States of America.',
        "metadata": {"episode_id": "DOC-CHAIN_e01", "episode_source_id": "DOC-CHAIN"},
    }
    usa = {
        "id": "usa",
        "fact": "The official language of United States of America is German.",
        "metadata": {"episode_id": "DOC-CHAIN_e02", "episode_source_id": "DOC-CHAIN"},
    }
    uk = {
        "id": "uk",
        "fact": "The official language of United Kingdom of Great Britain and Ireland is Hungarian.",
        "metadata": {"episode_id": "DOC-CHAIN_e03", "episode_source_id": "DOC-CHAIN"},
    }
    facts = [seed, usa, uk]
    real_get = episode_packet.get_tuning_section

    def _fake_get_tuning_section(*path):
        section = dict(real_get(*path))
        if path == ("operators",):
            section["bounded_chain_bundle_relation_signature_cap"] = 1
            section["bounded_chain_bundle_frontier_overlap_weight"] = overlap_weight
        return section

    overlap_weight = 0.0
    monkeypatch.setattr(episode_packet, "get_tuning_section", _fake_get_tuning_section)
    novelty_first = build_bounded_chain_candidate_bundle(
        question,
        [seed],
        facts,
        max_candidates=2,
        query_specificity_bonus=0.35,
    )

    overlap_weight = 2.0
    monkeypatch.setattr(episode_packet, "get_tuning_section", _fake_get_tuning_section)
    frontier_first = build_bounded_chain_candidate_bundle(
        question,
        [seed],
        facts,
        max_candidates=2,
        query_specificity_bonus=0.35,
    )

    assert [fact["id"] for fact in novelty_first["facts"]] == ["seed", "uk"]
    assert [fact["id"] for fact in frontier_first["facts"]] == ["seed", "usa"]
    assert frontier_first["trace"]["frontier_overlap_weight"] == 2.0


def test_bounded_chain_seed_selection_prefers_exact_entity_anchor():
    question = "What is the country of origin of the sport played by Christian Abbiati?"
    qf = extract_query_features(question)
    facts = [
        {
            "id": "distractor",
            "fact": "Asante Samuel plays the position of cornerback.",
            "metadata": {"episode_id": "DOC-CHAIN_e01", "episode_source_id": "DOC-CHAIN"},
        },
        {
            "id": "anchor",
            "fact": "Christian Abbiati plays the position of goaltender.",
            "metadata": {"episode_id": "DOC-CHAIN_e02", "episode_source_id": "DOC-CHAIN"},
        },
    ]
    token_freq = Counter(
        token
        for fact in facts
        for token in set(_fact_content_tokens(fact["fact"], qf))
    )

    selected = _select_bounded_chain_seed_facts(
        facts,
        qf,
        token_freq,
        query_specificity_bonus=0.35,
        seed_count=1,
    )

    assert [fact["id"] for fact in selected] == ["anchor"]


def test_qf_aware_pseudo_fact_generation_surfaces_late_entity_line():
    episode = {
        "episode_id": "DOC-CHAIN_e01",
        "raw_text": "\n".join(
            [
                "One distractor line about basketball.",
                "Another distractor line about baseball.",
                "More noise here about unrelated topics.",
                "Christian Abbiati plays the position of goalkeeper.",
                "The official language of Philippines is Tagalog.",
            ]
        ),
    }
    qf = extract_query_features("What is the country of origin of the sport played by Christian Abbiati?")

    pseudo = _pseudo_facts_from_episode("DOC-CHAIN_e01", episode, qf=qf)

    assert any(fact["fact"] == "Christian Abbiati plays the position of goalkeeper." for fact in pseudo)
    assert all(isinstance(fact["session"], int) for fact in pseudo)


def test_bounded_chain_bundle_same_entity_conflict_cap_is_consumed(monkeypatch):
    question = "What is the country of origin of the sport played by Christian Abbiati?"
    seed = {
        "id": "seed",
        "fact": "Christian Abbiati plays the position of cornerback.",
        "metadata": {"episode_id": "DOC-CHAIN_e01", "episode_source_id": "DOC-CHAIN"},
    }
    conflict = {
        "id": "conflict",
        "fact": "Christian Abbiati plays the position of goalkeeper.",
        "metadata": {"episode_id": "DOC-CHAIN_e02", "episode_source_id": "DOC-CHAIN"},
    }
    distractor = {
        "id": "distractor",
        "fact": "cornerback is associated with the sport of field hockey.",
        "metadata": {"episode_id": "DOC-CHAIN_e03", "episode_source_id": "DOC-CHAIN"},
    }
    distractor_2 = {
        "id": "distractor_2",
        "fact": "cornerback is associated with the sport of cricket.",
        "metadata": {"episode_id": "DOC-CHAIN_e04", "episode_source_id": "DOC-CHAIN"},
    }
    facts = [seed, conflict, distractor, distractor_2]
    real_get = episode_packet.get_tuning_section

    def _fake_get_tuning_section(*path):
        section = dict(real_get(*path))
        if path == ("operators",):
            section["bounded_chain_bundle_same_entity_conflict_cap"] = conflict_cap
            section["bounded_chain_bundle_relation_signature_cap"] = 0
            section["bounded_chain_bundle_frontier_overlap_weight"] = 2.0
        return section

    conflict_cap = 0
    monkeypatch.setattr(episode_packet, "get_tuning_section", _fake_get_tuning_section)
    uncapped = build_bounded_chain_candidate_bundle(
        question,
        [seed],
        facts,
        max_candidates=3,
        query_specificity_bonus=0.35,
    )

    conflict_cap = 1
    monkeypatch.setattr(episode_packet, "get_tuning_section", _fake_get_tuning_section)
    capped = build_bounded_chain_candidate_bundle(
        question,
        [seed],
        facts,
        max_candidates=3,
        query_specificity_bonus=0.35,
    )

    assert [fact["id"] for fact in uncapped["facts"]] == ["seed", "distractor", "distractor_2"]
    assert [fact["id"] for fact in capped["facts"]] == ["seed", "conflict", "distractor"]
    assert capped["trace"]["same_entity_conflict_cap"] == 1
    assert capped["trace"]["same_entity_conflict_ids"] == ["conflict"]


def test_episode_packet_keeps_multiple_episodes_from_same_source():
    corpus = _make_corpus()
    episode_lookup = build_episode_lookup(corpus)
    facts = _make_facts()
    fact_lookup = {fact["id"]: fact for fact in facts}
    retrieved = [{"fact_id": "f_current"}, {"fact_id": "f_old"}]

    context, injected = build_context_from_retrieved_facts(
        retrieved,
        episode_lookup,
        fact_lookup,
        budget=4000,
    )

    assert injected == ["DOC-1_e01", "DOC-1_e02"]
    assert "[Episode: DOC-1_e01]" in context
    assert "[Episode: DOC-1_e02]" in context
    assert "14.3 km" in context
    assert "14.1 km" in context


def test_selected_episode_context_can_draw_supporting_facts_from_same_source_pool():
    corpus = {
        "documents": [
            {
                "doc_id": "DOC-PIPE",
                "episodes": [
                    {
                        "episode_id": "DOC-PIPE_e01",
                        "source_type": "document",
                        "source_id": "DOC-PIPE",
                        "source_date": "2026-01-01",
                        "topic_key": "inspection fail",
                        "state_label": "inspection_fail",
                        "currentness": "historical",
                        "raw_text": "At km 2.3 the pipe cover depth was 780mm before rework.",
                        "provenance": {"block_ids": ["DOC-PIPE_b001"]},
                    },
                    {
                        "episode_id": "DOC-PIPE_e02",
                        "source_type": "document",
                        "source_id": "DOC-PIPE",
                        "source_date": "2026-01-02",
                        "topic_key": "rework order",
                        "state_label": "rework_required",
                        "currentness": "historical",
                        "raw_text": "The section at km 2.3 must be reworked to at least 900mm depth.",
                        "provenance": {"block_ids": ["DOC-PIPE_b002"]},
                    },
                    {
                        "episode_id": "DOC-PIPE_e03",
                        "source_type": "document",
                        "source_id": "DOC-PIPE",
                        "source_date": "2026-01-03",
                        "topic_key": "reinspection",
                        "state_label": "reinspection_passed",
                        "currentness": "historical",
                        "raw_text": "After re-inspection, the pipe cover depth at km 2.3 was 980mm.",
                        "provenance": {"block_ids": ["DOC-PIPE_b003"]},
                    },
                ],
            }
        ]
    }
    episode_lookup = build_episode_lookup(corpus)
    facts_by_episode = {
        "DOC-PIPE_e01": [{"id": "f1", "session": 1, "fact": "At km 2.3 the pipe cover depth was 780mm before rework.", "metadata": {"episode_id": "DOC-PIPE_e01", "episode_source_id": "DOC-PIPE"}}],
        "DOC-PIPE_e02": [{"id": "f2", "session": 2, "fact": "The section at km 2.3 must be reworked to at least 900mm depth.", "metadata": {"episode_id": "DOC-PIPE_e02", "episode_source_id": "DOC-PIPE"}}],
        "DOC-PIPE_e03": [{"id": "f3", "session": 3, "fact": "After re-inspection, the pipe cover depth at km 2.3 was 980mm.", "metadata": {"episode_id": "DOC-PIPE_e03", "episode_source_id": "DOC-PIPE"}}],
    }

    context, injected, fact_ids = build_context_from_selected_episodes(
        "What is the current pipe cover depth at km 2.3?",
        ["DOC-PIPE_e01", "DOC-PIPE_e02"],
        episode_lookup,
        facts_by_episode,
        fact_episode_ids=["DOC-PIPE_e01", "DOC-PIPE_e02", "DOC-PIPE_e03"],
        budget=4000,
        max_total_facts=6,
        max_facts_per_episode=2,
        allow_pseudo_facts=False,
        query_features=extract_query_features("What is the current pipe cover depth at km 2.3?"),
    )

    assert injected == ["DOC-PIPE_e01", "DOC-PIPE_e02"]
    assert "f3" in fact_ids
    assert "980mm" in context
    assert "[Episode: DOC-PIPE_e03]" in context


def test_bounded_chain_fact_rerank_surfaces_bridge_pseudo_fact():
    corpus = {
        "documents": [
            {
                "doc_id": "DOC-CHAIN",
                "episodes": [
                    {
                        "episode_id": "DOC-CHAIN_e01",
                        "source_type": "document",
                        "source_id": "DOC-CHAIN",
                        "source_date": "2026-01-01",
                        "topic_key": "bagratuni religion",
                        "state_label": "fact",
                        "currentness": "unknown",
                        "raw_text": "Bagratuni Dynasty is affiliated with the religion of Christianity.",
                        "provenance": {"block_ids": ["DOC-CHAIN_b001"]},
                    },
                    {
                        "episode_id": "DOC-CHAIN_e02",
                        "source_type": "document",
                        "source_id": "DOC-CHAIN",
                        "source_date": "2026-01-02",
                        "topic_key": "christianity origin",
                        "state_label": "fact",
                        "currentness": "unknown",
                        "raw_text": "Christianity was founded in the city of Jerusalem.\nChristianity was founded in the city of Taipei.",
                        "provenance": {"block_ids": ["DOC-CHAIN_b002"]},
                    },
                ],
            }
        ]
    }
    episode_lookup = build_episode_lookup(corpus)
    facts_by_episode = {
        "DOC-CHAIN_e01": [
            {
                "id": "chain_f1",
                "session": 1,
                "fact": "Bagratuni Dynasty is affiliated with the religion of Christianity.",
                "metadata": {"episode_id": "DOC-CHAIN_e01", "episode_source_id": "DOC-CHAIN"},
            }
        ],
        "DOC-CHAIN_e02": [],
    }

    context, _injected, fact_ids = build_context_from_selected_episodes(
        "Where did the religion associated with the Bagratuni Dynasty come into existence?",
        ["DOC-CHAIN_e01", "DOC-CHAIN_e02"],
        episode_lookup,
        facts_by_episode,
        budget=4000,
        max_total_facts=6,
        max_facts_per_episode=3,
        allow_pseudo_facts=True,
        query_features=extract_query_features("Where did the religion associated with the Bagratuni Dynasty come into existence?"),
        bounded_chain_fact_bonus=2.0,
    )

    assert any(fid.startswith("raw_DOC-CHAIN_e02_") for fid in fact_ids)
    assert "Taipei" in context


def test_build_episode_hybrid_context_uses_episode_runtime_defaults():
    corpus = _make_corpus()
    packet = build_episode_hybrid_context(
        "What is the final approved route length?",
        corpus,
        _make_facts(),
    )

    assert packet["retrieved_episode_ids"]
    assert packet["actual_injected_episode_ids"]
    assert packet["selector_config"]["budget"] == 8000
    assert packet["selector_config"]["max_sources_per_family"] == 2
    assert "commercial_pair_preserve" not in packet["selector_config"]
    assert "[Episode: DOC-1_e01]" in packet["context"]
    assert packet["query_operator_plan"]["operators"] == []


def test_query_operator_plan_is_explicit_and_structured():
    qf = extract_query_features(
        "What do Alice and Bob have in common and which store did they both visit first?"
    )

    plan = qf["operator_plan"]
    assert plan["commonality"]["enabled"] is True
    assert plan["ordinal"]["enabled"] is True
    assert plan["local_anchor"]["enabled"] is True
    assert "commonality" in plan["operators"]
    assert "ordinal" in plan["operators"]


def test_plural_head_query_enables_list_set_operator():
    qf = extract_query_features("What martial arts has John done?")

    plan = qf["operator_plan"]
    assert plan["list_set"]["enabled"] is True
    assert "list_set" in plan["operators"]


def test_possessive_group_query_enables_list_set_operator():
    qf = extract_query_features("Which of Deborah`s family and friends have passed away?")

    plan = qf["operator_plan"]
    assert plan["list_set"]["enabled"] is True
    assert "list_set" in plan["operators"]
    assert plan["list_set"]["head_phrase"] == "family and friends"


def test_where_relation_query_enables_bounded_chain_operator():
    qf = extract_query_features(
        "Where did the religion associated with the Bagratuni Dynasty come into existence?"
    )

    plan = qf["operator_plan"]
    assert plan["local_anchor"]["enabled"] is True
    assert plan["bounded_chain"]["enabled"] is True
    assert "bounded_chain" in plan["operators"]


def test_first_person_local_anchor_query_does_not_enable_spurious_bounded_chain():
    qf = extract_query_features("Where did I redeem a $5 coupon on coffee creamer?")

    plan = qf["operator_plan"]
    assert plan["local_anchor"]["enabled"] is True
    assert plan["bounded_chain"]["enabled"] is False
    assert "bounded_chain" not in plan["operators"]


def test_structural_query_dedupes_multiword_entity_tokens():
    qf = extract_query_features(
        'In which language are the official documents written in the country of citizenship of Nate "Tiny" Archibald?'
    )

    assert qf["entity_phrases"] == ['Nate "Tiny" Archibald']
    plan = qf["operator_plan"]
    assert plan["bounded_chain"]["enabled"] is True
    assert "bounded_chain" in plan["operators"]


def test_retrieval_target_strips_output_constraints():
    qf = extract_query_features(
        "Prepend lPiXbALSeQ to the 1st song about frames. Do not include any other text."
    )

    assert qf["output_constraints"]["prepend_prefix"] == "lPiXbALSeQ"
    assert qf["output_constraints"]["return_only"] is True
    assert "prepend" not in qf["retrieval_target"].lower()
    assert "do not include any other text" not in qf["retrieval_target"].lower()
    assert "song about frames" in qf["retrieval_target"].lower()


def test_operator_plan_changes_packet_knobs_for_structural_queries():
    corpus = _make_corpus()
    packet = build_episode_hybrid_context(
        "Which section changed after the draft and where is the current value?",
        corpus,
        _make_facts(),
    )

    assert packet["query_operator_plan"]["compare_diff"]["enabled"] is True
    assert packet["query_operator_plan"]["local_anchor"]["enabled"] is True
    assert packet["selector_config"]["snippet_mode"] is True
    assert packet["selector_config"]["supporting_facts_total"] >= 12


def test_structural_operator_tuning_knobs_are_consumed(monkeypatch):
    corpus = _make_corpus()

    def _tuning():
        return {
            "routing": {"max_plausible_families": 2, "ambiguous_family_fanout": 2},
            "episodes": {"document_grouping": {"prompt_mode": "strict_small", "size_cap_chars": 12000, "strip_duplicates": True, "attach_missing": False, "singleton_missing": True}},
            "retrieval": {"selector": DEFAULT_SELECTION_CONFIG},
            "operators": {
                "ordinal_candidate_budget": 12,
                "compare_alignment_budget": 12,
                "list_set_dedup_overlap": 0.9,
                "enable_snippet_for_ordinal": True,
                "enable_snippet_for_local_anchor": True,
                "local_anchor_window_chars": 1200,
                "structural_query_supporting_facts_total": 21,
                "structural_query_supporting_facts_per_episode": 6,
                "bounded_chain_max_hops": 4,
            },
            "packet": {
                "budget": 8000,
                "max_facts": 10,
                "max_facts_per_episode": 3,
                "max_episodes": 3,
                "per_source_cap": 2,
                "per_family_cap": 3,
                "snippet_mode": False,
                "snippet_chars": 1200,
            },
            "telemetry": {"include_runtime_trace": True, "max_selection_scores": 12, "max_scope_source_ids": 32, "max_family_candidates": 8, "max_rejected_candidates": 8, "max_packet_fact_ids": 24},
        }

    monkeypatch.setattr("src.memory.get_runtime_tuning", _tuning)
    packet = build_episode_hybrid_context(
        "Which section changed after the draft and where is the current value?",
        corpus,
        _make_facts(),
    )

    assert packet["selector_config"]["snippet_mode"] is True
    assert packet["selector_config"]["supporting_facts_total"] == 21
    assert packet["selector_config"]["supporting_facts_per_episode"] == 6

    chain_corpus = {
        "documents": [{
            "doc_id": "document:DOC-CHAIN",
            "episodes": [
                {
                    "episode_id": "chain_e1",
                    "source_type": "document",
                    "source_id": "DOC-CHAIN",
                    "source_date": "2026-01-01",
                    "topic_key": "bagratuni religion",
                    "state_label": "fact",
                    "currentness": "unknown",
                    "raw_text": "Bagratuni follows Christianity.",
                    "provenance": {"block_ids": ["DOC-CHAIN_b001"]},
                },
                {
                    "episode_id": "chain_e2",
                    "source_type": "document",
                    "source_id": "DOC-CHAIN",
                    "source_date": "2026-01-01",
                    "topic_key": "christianity origin",
                    "state_label": "fact",
                    "currentness": "unknown",
                    "raw_text": "Christianity was founded in Taipei.",
                    "provenance": {"block_ids": ["DOC-CHAIN_b002"]},
                },
            ],
        }]
    }
    chain_facts = [
        {"id": "chain_f1", "session": 1, "fact": "Bagratuni follows Christianity.", "metadata": {"episode_id": "chain_e1", "episode_source_id": "DOC-CHAIN"}},
        {"id": "chain_f2", "session": 2, "fact": "Christianity was founded in Taipei.", "metadata": {"episode_id": "chain_e2", "episode_source_id": "DOC-CHAIN"}},
    ]
    chain_packet = build_episode_hybrid_context(
        "Which city was the religion of Bagratuni founded in?",
        chain_corpus,
        chain_facts,
    )
    assert chain_packet["query_operator_plan"]["bounded_chain"]["max_hops"] == 4


def test_selector_overrides_survive_packet_defaults():
    corpus = _make_corpus()
    packet = build_episode_hybrid_context(
        "What is the final approved route length?",
        corpus,
        _make_facts(),
        selector_config={
            "max_episodes_default": 1,
            "supporting_facts_total": 1,
            "supporting_facts_per_episode": 1,
            "budget": 1234,
            "max_sources_per_family": 1,
        },
    )

    assert packet["selector_config"]["max_episodes_default"] == 1
    assert packet["selector_config"]["supporting_facts_total"] == 1
    assert packet["selector_config"]["supporting_facts_per_episode"] == 1
    assert packet["selector_config"]["budget"] == 1234
    assert packet["selector_config"]["max_sources_per_family"] == 1
    assert len(packet["retrieved_episode_ids"]) == 1


def test_selector_overrides_take_precedence_over_packet_tuning(monkeypatch):
    corpus = {
        "documents": [{
            "doc_id": "document:DOC-PRECEDENCE",
            "episodes": [
                {
                    "episode_id": "DOC-PRECEDENCE_e01",
                    "source_type": "document",
                    "source_id": "DOC-PRECEDENCE",
                    "source_date": "2026-01-01",
                    "topic_key": "route 1 current",
                    "state_label": "approved",
                    "currentness": "current",
                    "raw_text": "Route 1 final approved length is 14.3 km.",
                    "provenance": {"block_ids": ["DOC-PRECEDENCE_b001"]},
                },
                {
                    "episode_id": "DOC-PRECEDENCE_e02",
                    "source_type": "document",
                    "source_id": "DOC-PRECEDENCE",
                    "source_date": "2026-01-01",
                    "topic_key": "route 2 current",
                    "state_label": "approved",
                    "currentness": "current",
                    "raw_text": "Route 2 final approved length is 16.1 km.",
                    "provenance": {"block_ids": ["DOC-PRECEDENCE_b002"]},
                },
            ],
        }]
    }
    facts = [
        {
            "id": "f1",
            "session": 1,
            "fact": "Route 1 final approved length is 14.3 km.",
            "metadata": {"episode_id": "DOC-PRECEDENCE_e01", "episode_source_id": "DOC-PRECEDENCE"},
        },
        {
            "id": "f2",
            "session": 2,
            "fact": "Route 2 final approved length is 16.1 km.",
            "metadata": {"episode_id": "DOC-PRECEDENCE_e02", "episode_source_id": "DOC-PRECEDENCE"},
        },
    ]

    def _tuning():
        return {
            "routing": {"max_plausible_families": 2, "ambiguous_family_fanout": 2},
            "episodes": {"document_grouping": {"prompt_mode": "strict_small", "size_cap_chars": 12000, "strip_duplicates": True, "attach_missing": False, "singleton_missing": True}},
            "retrieval": {"selector": DEFAULT_SELECTION_CONFIG},
            "operators": {
                "ordinal_candidate_budget": 12,
                "compare_alignment_budget": 12,
                "list_set_dedup_overlap": 0.9,
                "list_set_max_episodes": 12,
                "list_set_supporting_facts_total": 12,
                "list_set_supporting_facts_per_episode": 2,
                "enable_snippet_for_ordinal": True,
                "enable_snippet_for_local_anchor": True,
                "local_anchor_max_episodes": 1,
                "local_anchor_window_chars": 1200,
                "local_anchor_fact_radius": 12,
                "local_anchor_supporting_facts_total": 12,
                "local_anchor_supporting_facts_per_episode": 6,
                "structural_query_supporting_facts_total": 12,
                "structural_query_supporting_facts_per_episode": 4,
                "bounded_chain_max_hops": 2,
                "bounded_chain_fact_bonus": 0.0,
            },
            "packet": {
                "budget": 9999,
                "max_facts": 99,
                "max_facts_per_episode": 99,
                "max_episodes": 99,
                "support_episode_pool_size": 6,
                "per_source_cap": 99,
                "per_family_cap": 99,
                "snippet_mode": False,
                "snippet_chars": 1200,
            },
            "telemetry": {"include_runtime_trace": True, "max_selection_scores": 12, "max_scope_source_ids": 32, "max_family_candidates": 8, "max_rejected_candidates": 8, "max_packet_fact_ids": 24},
        }

    monkeypatch.setattr("src.memory.get_runtime_tuning", _tuning)
    packet = build_episode_hybrid_context(
        "What is the final approved route length?",
        corpus,
        facts,
        selector_config={
            "max_episodes_default": 1,
            "supporting_facts_total": 1,
            "supporting_facts_per_episode": 1,
            "budget": 1234,
            "max_sources_per_family": 1,
        },
    )

    assert packet["selector_config"]["max_episodes_default"] == 1
    assert packet["selector_config"]["supporting_facts_total"] == 1
    assert packet["selector_config"]["supporting_facts_per_episode"] == 1
    assert packet["selector_config"]["budget"] == 1234
    assert packet["selector_config"]["max_sources_per_family"] == 1
    assert len(packet["retrieved_episode_ids"]) == 1
    assert len(packet["retrieved_fact_ids"]) == 1


def test_operator_packet_knobs_are_consumed(monkeypatch):
    corpus = {
        "documents": [{
            "doc_id": "conversation:conv_knob_packet",
            "episodes": [{
                "episode_id": "conv_knob_packet_e0001",
                "source_type": "conversation",
                "source_id": "conv_knob_packet",
                "source_date": "2026-01-01",
                "topic_key": "session_1",
                "state_label": "session",
                "currentness": "unknown",
                "raw_text": ("A" * 1200) + " Target store " + ("B" * 1200),
                "provenance": {"raw_span": [0, 2413]},
            }],
        }]
    }
    facts = [{
        "id": "f_anchor",
        "session": 1,
        "fact": "Coupon redemption happened at Target store.",
        "metadata": {
            "episode_id": "conv_knob_packet_e0001",
            "episode_source_id": "conv_knob_packet",
        },
    }]

    def _tuning_with_window(window_chars: int):
        return {
            "routing": {"max_plausible_families": 2, "ambiguous_family_fanout": 2},
            "episodes": {"document_grouping": {"prompt_mode": "strict_small", "size_cap_chars": 12000, "strip_duplicates": True, "attach_missing": False, "singleton_missing": True}},
            "retrieval": {"selector": DEFAULT_SELECTION_CONFIG},
            "operators": {
                "ordinal_candidate_budget": 12,
                "compare_alignment_budget": 12,
                "list_set_dedup_overlap": 0.9,
                "enable_snippet_for_ordinal": False,
                "enable_snippet_for_local_anchor": True,
                "local_anchor_window_chars": window_chars,
                "structural_query_supporting_facts_total": 21,
                "structural_query_supporting_facts_per_episode": 6,
                "bounded_chain_max_hops": 2,
            },
            "packet": {
                "budget": 8000,
                "max_facts": 10,
                "max_facts_per_episode": 3,
                "max_episodes": 3,
                "per_source_cap": 2,
                "per_family_cap": 3,
                "snippet_mode": False,
                "snippet_chars": 600,
            },
            "telemetry": {"include_runtime_trace": True, "max_selection_scores": 12, "max_scope_source_ids": 32, "max_family_candidates": 8, "max_rejected_candidates": 8, "max_packet_fact_ids": 24},
        }

    monkeypatch.setattr("src.memory.get_runtime_tuning", lambda: _tuning_with_window(600))
    packet_small = build_episode_hybrid_context(
        "Which store did the coupon redemption happen at?",
        corpus,
        facts,
    )
    monkeypatch.setattr("src.memory.get_runtime_tuning", lambda: _tuning_with_window(2200))
    packet_large = build_episode_hybrid_context(
        "Which store did the coupon redemption happen at?",
        corpus,
        facts,
    )

    assert packet_small["selector_config"]["snippet_mode"] is True
    assert packet_large["selector_config"]["snippet_mode"] is True
    assert "Target store" in packet_small["context"]
    assert "Target store" in packet_large["context"]
    assert len(packet_large["context"]) > len(packet_small["context"])


def test_enable_snippet_for_ordinal_knob_is_consumed(monkeypatch):
    corpus = _make_corpus()
    facts = _make_facts()

    def _tuning(enabled: bool):
        return {
            "routing": {"max_plausible_families": 2, "ambiguous_family_fanout": 2},
            "episodes": {"document_grouping": {"prompt_mode": "strict_small", "size_cap_chars": 12000, "strip_duplicates": True, "attach_missing": False, "singleton_missing": True}},
            "retrieval": {"selector": DEFAULT_SELECTION_CONFIG},
            "operators": {
                "ordinal_candidate_budget": 12,
                "compare_alignment_budget": 12,
                "list_set_dedup_overlap": 0.9,
                "enable_snippet_for_ordinal": enabled,
                "enable_snippet_for_local_anchor": True,
                "local_anchor_window_chars": 1200,
                "structural_query_supporting_facts_total": 12,
                "structural_query_supporting_facts_per_episode": 4,
                "bounded_chain_max_hops": 2,
            },
            "packet": {
                "budget": 8000,
                "max_facts": 10,
                "max_facts_per_episode": 3,
                "max_episodes": 3,
                "per_source_cap": 2,
                "per_family_cap": 3,
                "snippet_mode": False,
                "snippet_chars": 1200,
            },
            "telemetry": {"include_runtime_trace": True, "max_selection_scores": 12, "max_scope_source_ids": 32, "max_family_candidates": 8, "max_rejected_candidates": 8, "max_packet_fact_ids": 24},
        }

    monkeypatch.setattr("src.memory.get_runtime_tuning", lambda: _tuning(False))
    packet_off = build_episode_hybrid_context(
        "What is the 1st approved route length?",
        corpus,
        facts,
    )
    monkeypatch.setattr("src.memory.get_runtime_tuning", lambda: _tuning(True))
    packet_on = build_episode_hybrid_context(
        "What is the 1st approved route length?",
        corpus,
        facts,
    )

    assert packet_off["selector_config"]["snippet_mode"] is False
    assert packet_on["selector_config"]["snippet_mode"] is True


def test_list_set_packet_knobs_are_consumed(monkeypatch):
    corpus = {
        "documents": [{
            "doc_id": "conversation:conv_list_knobs",
            "episodes": [
                {
                    "episode_id": "list_knob_e1",
                    "source_type": "conversation",
                    "source_id": "conv_list_knobs",
                    "source_date": "2026-01-01",
                    "topic_key": "martial arts one",
                    "state_label": "session",
                    "currentness": "unknown",
                    "raw_text": "John practiced kickboxing.",
                    "provenance": {"raw_span": [0, 26]},
                },
                {
                    "episode_id": "list_knob_e2",
                    "source_type": "conversation",
                    "source_id": "conv_list_knobs",
                    "source_date": "2026-01-02",
                    "topic_key": "martial arts two",
                    "state_label": "session",
                    "currentness": "unknown",
                    "raw_text": "John practiced taekwondo.",
                    "provenance": {"raw_span": [0, 25]},
                },
                {
                    "episode_id": "list_knob_e3",
                    "source_type": "conversation",
                    "source_id": "conv_list_knobs",
                    "source_date": "2026-01-03",
                    "topic_key": "martial arts three",
                    "state_label": "session",
                    "currentness": "unknown",
                    "raw_text": "John practiced judo.",
                    "provenance": {"raw_span": [0, 21]},
                },
            ],
        }]
    }
    facts = [
        {"id": "f_kick", "session": 1, "fact": "John practiced kickboxing.", "metadata": {"episode_id": "list_knob_e1", "episode_source_id": "conv_list_knobs"}},
        {"id": "f_tkd", "session": 2, "fact": "John practiced taekwondo.", "metadata": {"episode_id": "list_knob_e2", "episode_source_id": "conv_list_knobs"}},
        {"id": "f_judo", "session": 3, "fact": "John practiced judo.", "metadata": {"episode_id": "list_knob_e3", "episode_source_id": "conv_list_knobs"}},
    ]

    def _tuning(max_episodes: int, total_facts: int, per_episode: int):
        return {
            "routing": {"max_plausible_families": 2, "ambiguous_family_fanout": 2},
            "episodes": {"document_grouping": {"prompt_mode": "strict_small", "size_cap_chars": 12000, "strip_duplicates": True, "attach_missing": False, "singleton_missing": True}},
            "retrieval": {"selector": DEFAULT_SELECTION_CONFIG},
            "operators": {
                "ordinal_candidate_budget": 12,
                "compare_alignment_budget": 12,
                "list_set_dedup_overlap": 1.1,
                "list_set_max_episodes": max_episodes,
                "list_set_supporting_facts_total": total_facts,
                "list_set_supporting_facts_per_episode": per_episode,
                "enable_snippet_for_ordinal": True,
                "enable_snippet_for_local_anchor": True,
                "local_anchor_max_episodes": 1,
                "local_anchor_window_chars": 1200,
                "local_anchor_fact_radius": 12,
                "local_anchor_supporting_facts_total": 12,
                "local_anchor_supporting_facts_per_episode": 6,
                "structural_query_supporting_facts_total": 12,
                "structural_query_supporting_facts_per_episode": 4,
                "bounded_chain_max_hops": 2,
                "bounded_chain_fact_bonus": 0.0,
            },
            "packet": {
                "budget": 8000,
                "max_facts": 10,
                "max_facts_per_episode": 3,
                "max_episodes": 2,
                "support_episode_pool_size": 3,
                "per_source_cap": 2,
                "per_family_cap": 3,
                "snippet_mode": False,
                "snippet_chars": 1200,
            },
            "telemetry": {"include_runtime_trace": True, "max_selection_scores": 12, "max_scope_source_ids": 32, "max_family_candidates": 8, "max_rejected_candidates": 8, "max_packet_fact_ids": 24},
        }

    monkeypatch.setattr("src.memory.get_runtime_tuning", lambda: _tuning(2, 2, 1))
    packet_small = build_episode_hybrid_context(
        "What martial arts has John done?",
        corpus,
        facts,
    )
    monkeypatch.setattr("src.memory.get_runtime_tuning", lambda: _tuning(3, 6, 3))
    packet_large = build_episode_hybrid_context(
        "What martial arts has John done?",
        corpus,
        facts,
    )

    assert len(packet_small["retrieved_episode_ids"]) == 2
    assert len(packet_large["retrieved_episode_ids"]) == 3
    assert packet_small["selector_config"]["supporting_facts_total"] == 12
    assert packet_small["selector_config"]["supporting_facts_per_episode"] == 4
    assert packet_large["selector_config"]["supporting_facts_total"] == 12
    assert packet_large["selector_config"]["supporting_facts_per_episode"] == 4


def test_local_anchor_packet_knobs_are_consumed(monkeypatch):
    corpus = {
        "documents": [{
            "doc_id": "conversation:conv_anchor_knobs",
            "episodes": [
                {
                    "episode_id": "anchor_knob_e1",
                    "source_type": "conversation",
                    "source_id": "conv_anchor_knobs",
                    "source_date": "2026-01-01",
                    "topic_key": "coupon mention",
                    "state_label": "session",
                    "currentness": "unknown",
                    "raw_text": "I redeemed a coupon on coffee creamer yesterday.",
                    "provenance": {"raw_span": [0, 47]},
                },
                {
                    "episode_id": "anchor_knob_e2",
                    "source_type": "conversation",
                    "source_id": "conv_anchor_knobs",
                    "source_date": "2026-01-02",
                    "topic_key": "store mention",
                    "state_label": "session",
                    "currentness": "unknown",
                    "raw_text": "I redeemed a coupon on coffee creamer at Target.",
                    "provenance": {"raw_span": [0, 50]},
                },
            ],
        }]
    }
    facts = [
        {"id": "f_coupon", "session": 1, "fact": "I redeemed a coupon on coffee creamer yesterday.", "metadata": {"episode_id": "anchor_knob_e1", "episode_source_id": "conv_anchor_knobs"}},
        {"id": "f_target", "session": 2, "fact": "I redeemed a coupon on coffee creamer at Target.", "metadata": {"episode_id": "anchor_knob_e2", "episode_source_id": "conv_anchor_knobs"}},
    ]

    def _tuning(max_episodes: int, total_facts: int, per_episode: int):
        return {
            "routing": {"max_plausible_families": 2, "ambiguous_family_fanout": 2},
            "episodes": {"document_grouping": {"prompt_mode": "strict_small", "size_cap_chars": 12000, "strip_duplicates": True, "attach_missing": False, "singleton_missing": True}},
            "retrieval": {"selector": DEFAULT_SELECTION_CONFIG},
            "operators": {
                "ordinal_candidate_budget": 12,
                "compare_alignment_budget": 12,
                "list_set_dedup_overlap": 0.9,
                "list_set_max_episodes": 12,
                "list_set_supporting_facts_total": 12,
                "list_set_supporting_facts_per_episode": 2,
                "enable_snippet_for_ordinal": True,
                "enable_snippet_for_local_anchor": True,
                "local_anchor_max_episodes": max_episodes,
                "local_anchor_window_chars": 1200,
                "local_anchor_fact_radius": 12,
                "local_anchor_supporting_facts_total": total_facts,
                "local_anchor_supporting_facts_per_episode": per_episode,
                "structural_query_supporting_facts_total": 12,
                "structural_query_supporting_facts_per_episode": 4,
                "bounded_chain_max_hops": 2,
                "bounded_chain_fact_bonus": 0.0,
            },
            "packet": {
                "budget": 8000,
                "max_facts": 10,
                "max_facts_per_episode": 3,
                "max_episodes": 3,
                "support_episode_pool_size": 2,
                "per_source_cap": 2,
                "per_family_cap": 3,
                "snippet_mode": False,
                "snippet_chars": 1200,
            },
            "telemetry": {"include_runtime_trace": True, "max_selection_scores": 12, "max_scope_source_ids": 32, "max_family_candidates": 8, "max_rejected_candidates": 8, "max_packet_fact_ids": 24},
        }

    monkeypatch.setattr("src.memory.get_runtime_tuning", lambda: _tuning(1, 13, 5))
    packet_small = build_episode_hybrid_context(
        "Which store mentions the coupon redemption?",
        corpus,
        facts,
    )
    monkeypatch.setattr("src.memory.get_runtime_tuning", lambda: _tuning(2, 15, 7))
    packet_large = build_episode_hybrid_context(
        "Which store mentions the coupon redemption?",
        corpus,
        facts,
    )

    assert len(packet_small["retrieved_episode_ids"]) == 1
    assert len(packet_large["retrieved_episode_ids"]) == 2
    assert packet_small["selector_config"]["supporting_facts_total"] == 13
    assert packet_small["selector_config"]["supporting_facts_per_episode"] == 5
    assert packet_large["selector_config"]["supporting_facts_total"] == 15
    assert packet_large["selector_config"]["supporting_facts_per_episode"] == 7


def test_local_anchor_step_range_queries_can_expand_episode_budget(monkeypatch):
    corpus = {
        "documents": [{
            "doc_id": "document:AMA-RANGE",
            "episodes": [
                {
                    "episode_id": f"AMA-RANGE_e0{i}",
                    "source_type": "document",
                    "source_id": "AMA-RANGE",
                    "source_date": f"2026-01-{i:02d}",
                    "topic_key": f"step {i}",
                    "state_label": "step",
                    "currentness": "historical",
                    "raw_text": f"[Step {i}] Action: move thing {i}. Observation: environment changed at step {i}.",
                    "provenance": {"block_ids": [f"AMA-RANGE_b0{i}"]},
                }
                for i in range(1, 9)
            ],
        }]
    }
    facts = [
        {
            "id": f"f_step_{i}",
            "session": i,
            "fact": f"At step {i}, move thing {i} changed the environment.",
            "metadata": {"episode_id": f"AMA-RANGE_e0{i}", "episode_source_id": "AMA-RANGE"},
        }
        for i in range(1, 9)
    ]

    def _tuning():
        return {
            "routing": {"max_plausible_families": 2, "ambiguous_family_fanout": 2},
            "episodes": {"document_grouping": {"prompt_mode": "strict_small", "size_cap_chars": 12000, "strip_duplicates": True, "attach_missing": False, "singleton_missing": True}},
            "retrieval": {"selector": DEFAULT_SELECTION_CONFIG},
            "operators": {
                "ordinal_candidate_budget": 12,
                "compare_alignment_budget": 12,
                "list_set_dedup_overlap": 0.9,
                "list_set_max_episodes": 12,
                "list_set_supporting_facts_total": 12,
                "list_set_supporting_facts_per_episode": 2,
                "enable_snippet_for_ordinal": True,
                "enable_snippet_for_local_anchor": True,
                "local_anchor_max_episodes": 1,
                "local_anchor_window_chars": 1200,
                "local_anchor_fact_radius": 12,
                "local_anchor_supporting_facts_total": 12,
                "local_anchor_supporting_facts_per_episode": 6,
                "structural_query_supporting_facts_total": 12,
                "structural_query_supporting_facts_per_episode": 4,
                "bounded_chain_max_hops": 2,
                "bounded_chain_fact_bonus": 0.0,
            },
            "packet": {
                "budget": 8000,
                "max_facts": 12,
                "max_facts_per_episode": 3,
                "max_episodes": 1,
                "support_episode_pool_size": 8,
                "per_source_cap": 8,
                "per_family_cap": 8,
                "snippet_mode": False,
                "snippet_chars": 1200,
            },
            "telemetry": {"include_runtime_trace": True, "max_selection_scores": 12, "max_scope_source_ids": 32, "max_family_candidates": 8, "max_rejected_candidates": 8, "max_packet_fact_ids": 24},
        }

    monkeypatch.setattr("src.memory.get_runtime_tuning", _tuning)
    packet = build_episode_hybrid_context(
        "From step 1 to step 7, what actions made the environment changes and what were the environment changes?",
        corpus,
        facts,
    )

    assert packet["selector_config"]["max_episodes_default"] == 7
    assert packet["retrieved_episode_ids"] == [
        "AMA-RANGE_e01",
        "AMA-RANGE_e02",
        "AMA-RANGE_e03",
        "AMA-RANGE_e04",
        "AMA-RANGE_e05",
        "AMA-RANGE_e06",
        "AMA-RANGE_e07",
    ]
    assert "step 1" in packet["context"].lower()
    assert "step 7" in packet["context"].lower()


def test_step_range_queries_include_adjacent_action_observation_companions(monkeypatch):
    corpus = {
        "documents": [{
            "doc_id": "document:AMA-COMPANION",
            "episodes": [
                {
                    "episode_id": "AMA-COMPANION_e01",
                    "source_type": "document",
                    "source_id": "AMA-COMPANION",
                    "source_date": "2026-01-01",
                    "topic_key": "step 22",
                    "state_label": "structural_context",
                    "currentness": "historical",
                    "raw_text": "The current available actions are: take cd 3 from drawer 4, close drawer 4\n[Step 22]",
                    "provenance": {"block_ids": ["AMA-COMPANION_b001"]},
                },
                {
                    "episode_id": "AMA-COMPANION_e02",
                    "source_type": "document",
                    "source_id": "AMA-COMPANION",
                    "source_date": "2026-01-01",
                    "topic_key": "step 22 action",
                    "state_label": "structural_context",
                    "currentness": "historical",
                    "raw_text": "Action: take cd 3 from drawer 4\nObservation: You pick up the cd 3 from the drawer 4.",
                    "provenance": {"block_ids": ["AMA-COMPANION_b002"]},
                },
                {
                    "episode_id": "AMA-COMPANION_e03",
                    "source_type": "document",
                    "source_id": "AMA-COMPANION",
                    "source_date": "2026-01-01",
                    "topic_key": "step 26",
                    "state_label": "structural_context",
                    "currentness": "historical",
                    "raw_text": "The current available actions are: move cd 3 to safe 1, open safe 1\n[Step 26]",
                    "provenance": {"block_ids": ["AMA-COMPANION_b003"]},
                },
                {
                    "episode_id": "AMA-COMPANION_e04",
                    "source_type": "document",
                    "source_id": "AMA-COMPANION",
                    "source_date": "2026-01-01",
                    "topic_key": "step 26 action",
                    "state_label": "structural_context",
                    "currentness": "historical",
                    "raw_text": "Action: move cd 3 to safe 1\nObservation: You move the cd 3 to the safe 1.",
                    "provenance": {"block_ids": ["AMA-COMPANION_b004"]},
                },
            ],
        }]
    }
    facts = [
        {
            "id": "f_step_22_snapshot",
            "session": 1,
            "fact": "At step 22, the available action list includes take cd 3 from drawer 4.",
            "metadata": {"episode_id": "AMA-COMPANION_e01", "episode_source_id": "AMA-COMPANION"},
        },
        {
            "id": "f_step_22_action",
            "session": 2,
            "fact": "Take cd 3 from drawer 4. You pick up the cd 3 from the drawer 4.",
            "metadata": {"episode_id": "AMA-COMPANION_e02", "episode_source_id": "AMA-COMPANION"},
        },
        {
            "id": "f_step_26_snapshot",
            "session": 3,
            "fact": "At step 26, the available action list includes move cd 3 to safe 1.",
            "metadata": {"episode_id": "AMA-COMPANION_e03", "episode_source_id": "AMA-COMPANION"},
        },
        {
            "id": "f_step_26_action",
            "session": 4,
            "fact": "Move cd 3 to safe 1. You move the cd 3 to the safe 1.",
            "metadata": {"episode_id": "AMA-COMPANION_e04", "episode_source_id": "AMA-COMPANION"},
        },
    ]

    def _tuning():
        return {
            "routing": {"max_plausible_families": 2, "ambiguous_family_fanout": 2},
            "episodes": {"document_grouping": {"prompt_mode": "strict_small", "size_cap_chars": 12000, "strip_duplicates": True, "attach_missing": False, "singleton_missing": True}},
            "retrieval": {"selector": DEFAULT_SELECTION_CONFIG},
            "operators": {
                "ordinal_candidate_budget": 12,
                "compare_alignment_budget": 12,
                "list_set_dedup_overlap": 0.9,
                "list_set_max_episodes": 12,
                "list_set_supporting_facts_total": 12,
                "list_set_supporting_facts_per_episode": 2,
                "enable_snippet_for_ordinal": True,
                "enable_snippet_for_local_anchor": True,
                "local_anchor_max_episodes": 1,
                "local_anchor_window_chars": 1200,
                "local_anchor_fact_radius": 12,
                "local_anchor_supporting_facts_total": 12,
                "local_anchor_supporting_facts_per_episode": 6,
                "structural_query_supporting_facts_total": 12,
                "structural_query_supporting_facts_per_episode": 4,
                "bounded_chain_max_hops": 2,
                "bounded_chain_fact_bonus": 0.0,
            },
            "packet": {
                "budget": 8000,
                "max_facts": 12,
                "max_facts_per_episode": 3,
                "max_episodes": 1,
                "support_episode_pool_size": 8,
                "per_source_cap": 8,
                "per_family_cap": 8,
                "snippet_mode": False,
                "snippet_chars": 1200,
            },
            "telemetry": {"include_runtime_trace": True, "max_selection_scores": 12, "max_scope_source_ids": 32, "max_family_candidates": 8, "max_rejected_candidates": 8, "max_packet_fact_ids": 24},
        }

    monkeypatch.setattr("src.memory.get_runtime_tuning", _tuning)
    packet = build_episode_hybrid_context(
        "From step 19 to step 26, what actions made the environment changes and what were the environment changes?",
        corpus,
        facts,
    )

    assert "AMA-COMPANION_e01" in packet["retrieved_episode_ids"]
    assert "AMA-COMPANION_e03" in packet["retrieved_episode_ids"]
    assert "AMA-COMPANION_e02" in packet["fact_episode_ids"]
    assert "AMA-COMPANION_e04" in packet["fact_episode_ids"]
    assert "take cd 3 from drawer 4. you pick up the cd 3 from the drawer 4." in packet["context"].lower()
    assert "move cd 3 to safe 1. you move the cd 3 to the safe 1." in packet["context"].lower()


def test_step_range_queries_can_pull_support_episodes_from_fact_grounded_steps(monkeypatch):
    corpus = {
        "documents": [{
            "doc_id": "document:AMA-FACT-SUPPORT",
            "episodes": [
                {
                    "episode_id": "AMA-FACT-SUPPORT_e01",
                    "source_type": "document",
                    "source_id": "AMA-FACT-SUPPORT",
                    "source_date": "2026-01-01",
                    "topic_key": "step 13 marker",
                    "state_label": "structural_context",
                    "currentness": "historical",
                    "raw_text": "[Step 13]",
                    "provenance": {"block_ids": ["AMA-FACT-SUPPORT_b001"]},
                },
                {
                    "episode_id": "AMA-FACT-SUPPORT_e02",
                    "source_type": "document",
                    "source_id": "AMA-FACT-SUPPORT",
                    "source_date": "2026-01-01",
                    "topic_key": "step 13 action",
                    "state_label": "structural_context",
                    "currentness": "historical",
                    "raw_text": "Action: ((4,3),(5,3))\nObservation: Board:",
                    "provenance": {"block_ids": ["AMA-FACT-SUPPORT_b002"]},
                },
                {
                    "episode_id": "AMA-FACT-SUPPORT_e03",
                    "source_type": "document",
                    "source_id": "AMA-FACT-SUPPORT",
                    "source_date": "2026-01-01",
                    "topic_key": "step 13 summary",
                    "state_label": "structural_context",
                    "currentness": "historical",
                    "raw_text": "Score: 28\nMoves Left: 29\nLast Action: ((4,3),(5,3))",
                    "provenance": {"block_ids": ["AMA-FACT-SUPPORT_b003"]},
                },
            ],
        }]
    }
    facts = [
        {
            "id": "f_step_13_match",
            "session": 2,
            "fact": "At step 13, swapping green at (4,3) with red at (5,3) moved the red candy from (5,3) to (4,3), creating a match at (2,3), (3,3), (4,3).",
            "tags": ["step_trace", "match_creation"],
            "metadata": {
                "episode_id": "AMA-FACT-SUPPORT_e02",
                "episode_source_id": "AMA-FACT-SUPPORT",
                "structured_trace": True,
            },
        },
        {
            "id": "f_step_13_summary",
            "session": 3,
            "fact": "At step 13, score was 28 and moves left was 29.",
            "tags": ["step_trace", "board_summary"],
            "metadata": {
                "episode_id": "AMA-FACT-SUPPORT_e03",
                "episode_source_id": "AMA-FACT-SUPPORT",
                "structured_trace": True,
            },
        },
    ]

    def _tuning():
        return {
            "routing": {"max_plausible_families": 2, "ambiguous_family_fanout": 2},
            "episodes": {"document_grouping": {"prompt_mode": "strict_small", "size_cap_chars": 12000, "strip_duplicates": True, "attach_missing": False, "singleton_missing": True}},
            "retrieval": {"selector": DEFAULT_SELECTION_CONFIG},
            "operators": {
                "ordinal_candidate_budget": 12,
                "compare_alignment_budget": 12,
                "list_set_dedup_overlap": 0.9,
                "list_set_max_episodes": 12,
                "list_set_supporting_facts_total": 12,
                "list_set_supporting_facts_per_episode": 2,
                "enable_snippet_for_ordinal": True,
                "enable_snippet_for_local_anchor": True,
                "local_anchor_max_episodes": 1,
                "local_anchor_window_chars": 1200,
                "local_anchor_fact_radius": 12,
                "local_anchor_supporting_facts_total": 12,
                "local_anchor_supporting_facts_per_episode": 6,
                "structural_query_supporting_facts_total": 12,
                "structural_query_supporting_facts_per_episode": 4,
                "bounded_chain_max_hops": 2,
                "bounded_chain_fact_bonus": 0.0,
            },
            "packet": {
                "budget": 8000,
                "max_facts": 12,
                "max_facts_per_episode": 3,
                "max_episodes": 1,
                "support_episode_pool_size": 6,
                "per_source_cap": 8,
                "per_family_cap": 8,
                "snippet_mode": False,
                "snippet_chars": 1200,
            },
            "telemetry": {"include_runtime_trace": True, "max_selection_scores": 12, "max_scope_source_ids": 32, "max_family_candidates": 8, "max_rejected_candidates": 8, "max_packet_fact_ids": 24},
        }

    monkeypatch.setattr("src.memory.get_runtime_tuning", _tuning)
    packet = build_episode_hybrid_context(
        "In step 13, explain which candy moved to complete the match.",
        corpus,
        facts,
    )

    assert "AMA-FACT-SUPPORT_e02" in packet["fact_episode_ids"]
    assert "AMA-FACT-SUPPORT_e03" in packet["fact_episode_ids"]
    assert "creating a match at (2,3), (3,3), (4,3)" in packet["context"]


def test_exact_step_queries_include_next_action_episode_when_marker_is_in_previous_episode(monkeypatch):
    corpus = {
        "documents": [{
            "doc_id": "document:AMA-STEP-EXACT",
            "episodes": [
                {
                    "episode_id": "AMA-STEP-EXACT_e01",
                    "source_type": "document",
                    "source_id": "AMA-STEP-EXACT",
                    "source_date": "2026-01-01",
                    "topic_key": "step 8 marker",
                    "state_label": "schema_preview",
                    "currentness": "historical",
                    "raw_text": (
                        "The output has been truncated.\n"
                        "[Step 8]"
                    ),
                    "provenance": {"block_ids": ["AMA-STEP-EXACT_b001"]},
                },
                {
                    "episode_id": "AMA-STEP-EXACT_e02",
                    "source_type": "document",
                    "source_id": "AMA-STEP-EXACT",
                    "source_date": "2026-01-01",
                    "topic_key": "operational action",
                    "state_label": "operational",
                    "currentness": "historical",
                    "raw_text": (
                        "Action: execute_bash\n"
                        "Observation: EXECUTION RESULT of [execute_bash]"
                    ),
                    "provenance": {"block_ids": ["AMA-STEP-EXACT_b002"]},
                },
            ],
        }]
    }
    facts = [
        {
            "id": "f_step_8_bash",
            "session": 2,
            "fact": 'At step 8, the agent executed bash command `grep -E "collection_id|instance_size" DICOM_ALL.json | head -30`.',
            "metadata": {"episode_id": "AMA-STEP-EXACT_e02", "episode_source_id": "AMA-STEP-EXACT"},
        },
    ]

    def _tuning():
        return {
            "routing": {"max_plausible_families": 2, "ambiguous_family_fanout": 2},
            "episodes": {"document_grouping": {"prompt_mode": "strict_small", "size_cap_chars": 12000, "strip_duplicates": True, "attach_missing": False, "singleton_missing": True}},
            "retrieval": {"selector": DEFAULT_SELECTION_CONFIG},
            "operators": {
                "ordinal_candidate_budget": 12,
                "compare_alignment_budget": 12,
                "list_set_dedup_overlap": 0.9,
                "list_set_max_episodes": 12,
                "list_set_supporting_facts_total": 12,
                "list_set_supporting_facts_per_episode": 2,
                "enable_snippet_for_ordinal": True,
                "enable_snippet_for_local_anchor": True,
                "local_anchor_max_episodes": 1,
                "local_anchor_window_chars": 1200,
                "local_anchor_fact_radius": 12,
                "local_anchor_supporting_facts_total": 12,
                "local_anchor_supporting_facts_per_episode": 6,
                "structural_query_supporting_facts_total": 12,
                "structural_query_supporting_facts_per_episode": 4,
                "bounded_chain_max_hops": 2,
                "bounded_chain_fact_bonus": 0.0,
            },
            "packet": {
                "budget": 8000,
                "max_facts": 12,
                "max_facts_per_episode": 3,
                "max_episodes": 1,
                "support_episode_pool_size": 4,
                "per_source_cap": 8,
                "per_family_cap": 8,
                "snippet_mode": False,
                "snippet_chars": 1200,
            },
            "telemetry": {"include_runtime_trace": True, "max_selection_scores": 12, "max_scope_source_ids": 32, "max_family_candidates": 8, "max_rejected_candidates": 8, "max_packet_fact_ids": 24},
        }

    monkeypatch.setattr("src.memory.get_runtime_tuning", _tuning)
    packet = build_episode_hybrid_context(
        "At step 8, what exact bash command did the agent run?",
        corpus,
        facts,
    )

    assert "AMA-STEP-EXACT_e01" in packet["retrieved_episode_ids"]
    assert "AMA-STEP-EXACT_e02" in packet["fact_episode_ids"]
    anchor_section = packet["context"].split("RETRIEVED FACTS:", 1)[0]
    assert "[Episode: AMA-STEP-EXACT_e02]" in anchor_section
    assert "Action: execute_bash" in anchor_section
    assert 'grep -E "collection_id|instance_size" DICOM_ALL.json | head -30' in packet["context"]


def test_step_range_queries_include_next_action_episode_when_end_marker_is_in_previous_action_episode(monkeypatch):
    corpus = {
        "documents": [{
            "doc_id": "document:AMA-STEP-RANGE-END",
            "episodes": [
                {
                    "episode_id": "AMA-STEP-RANGE-END_e01",
                    "source_type": "document",
                    "source_id": "AMA-STEP-RANGE-END",
                    "source_date": "2026-01-01",
                    "topic_key": "step 4 action",
                    "state_label": "manipulation",
                    "currentness": "historical",
                    "raw_text": (
                        "Action: open drawer 1\n"
                        "Observation: You open the drawer 1.\n"
                        "[Step 5]"
                    ),
                    "provenance": {"block_ids": ["AMA-STEP-RANGE-END_b001"]},
                },
                {
                    "episode_id": "AMA-STEP-RANGE-END_e02",
                    "source_type": "document",
                    "source_id": "AMA-STEP-RANGE-END",
                    "source_date": "2026-01-01",
                    "topic_key": "step 5 action",
                    "state_label": "manipulation",
                    "currentness": "historical",
                    "raw_text": (
                        "Action: move peppershaker 3 to drawer 1\n"
                        "Observation: You move the peppershaker 3 to the drawer 1.\n"
                        "[Step 6]"
                    ),
                    "provenance": {"block_ids": ["AMA-STEP-RANGE-END_b002"]},
                },
            ],
        }]
    }
    facts = [
        {
            "id": "f_step_5_move",
            "session": 2,
            "fact": "At step 5, peppershaker 3 was moved to drawer 1.",
            "metadata": {"episode_id": "AMA-STEP-RANGE-END_e02", "episode_source_id": "AMA-STEP-RANGE-END"},
        },
    ]

    def _tuning():
        return {
            "routing": {"max_plausible_families": 2, "ambiguous_family_fanout": 2},
            "episodes": {"document_grouping": {"prompt_mode": "strict_small", "size_cap_chars": 12000, "strip_duplicates": True, "attach_missing": False, "singleton_missing": True}},
            "retrieval": {"selector": DEFAULT_SELECTION_CONFIG},
            "operators": {
                "ordinal_candidate_budget": 12,
                "compare_alignment_budget": 12,
                "list_set_dedup_overlap": 0.9,
                "list_set_max_episodes": 12,
                "list_set_supporting_facts_total": 12,
                "list_set_supporting_facts_per_episode": 2,
                "enable_snippet_for_ordinal": True,
                "enable_snippet_for_local_anchor": True,
                "local_anchor_max_episodes": 1,
                "local_anchor_window_chars": 1200,
                "local_anchor_fact_radius": 12,
                "local_anchor_supporting_facts_total": 12,
                "local_anchor_supporting_facts_per_episode": 6,
                "structural_query_supporting_facts_total": 12,
                "structural_query_supporting_facts_per_episode": 4,
                "bounded_chain_max_hops": 2,
                "bounded_chain_fact_bonus": 0.0,
            },
            "packet": {
                "budget": 8000,
                "max_facts": 12,
                "max_facts_per_episode": 3,
                "max_episodes": 1,
                "support_episode_pool_size": 4,
                "per_source_cap": 8,
                "per_family_cap": 8,
                "snippet_mode": False,
                "snippet_chars": 1200,
            },
            "telemetry": {"include_runtime_trace": True, "max_selection_scores": 12, "max_scope_source_ids": 32, "max_family_candidates": 8, "max_rejected_candidates": 8, "max_packet_fact_ids": 24},
        }

    monkeypatch.setattr("src.memory.get_runtime_tuning", _tuning)
    packet = build_episode_hybrid_context(
        "From step 1 to step 5, what actions made the environment changes and what were the environment changes?",
        corpus,
        facts,
    )

    assert "AMA-STEP-RANGE-END_e01" in packet["retrieved_episode_ids"]
    assert "AMA-STEP-RANGE-END_e02" in packet["fact_episode_ids"]
    anchor_section = packet["context"].split("RETRIEVED FACTS:", 1)[0]
    assert "[Episode: AMA-STEP-RANGE-END_e02]" in anchor_section
    assert "move peppershaker 3 to drawer 1" in anchor_section.lower()
    assert "peppershaker 3 was moved to drawer 1" in packet["context"].lower()


def test_support_episode_ids_can_inject_companions_without_selected_facts():
    corpus = {
        "documents": [{
            "doc_id": "document:STEP-COMPANION",
            "episodes": [
                {
                    "episode_id": "STEP-COMPANION_e01",
                    "source_type": "document",
                    "source_id": "STEP-COMPANION",
                    "source_date": "2026-01-01",
                    "topic_key": "step 1 marker",
                    "state_label": "step_marker",
                    "currentness": "historical",
                    "raw_text": "Observation: marker only\n[Step 1]",
                    "provenance": {"block_ids": ["STEP-COMPANION_b001"]},
                },
                {
                    "episode_id": "STEP-COMPANION_e02",
                    "source_type": "document",
                    "source_id": "STEP-COMPANION",
                    "source_date": "2026-01-01",
                    "topic_key": "step 1 action",
                    "state_label": "action_observation",
                    "currentness": "historical",
                    "raw_text": "Action: execute_bash: echo ok\nObservation: ok\n[Step 2]",
                    "provenance": {"block_ids": ["STEP-COMPANION_b002"]},
                },
            ],
        }]
    }
    episode_lookup = build_episode_lookup(corpus)
    facts_by_episode = {
        "STEP-COMPANION_e01": [
            {
                "id": "f_marker",
                "session": 1,
                "fact": "At step 1, a marker was recorded.",
                "metadata": {"episode_id": "STEP-COMPANION_e01"},
            },
        ],
    }

    context, injected, _selected_fact_ids = build_context_from_selected_episodes(
        "At step 1, what command did the agent run?",
        ["STEP-COMPANION_e01"],
        episode_lookup,
        facts_by_episode,
        fact_episode_ids=["STEP-COMPANION_e01"],
        support_episode_ids=["STEP-COMPANION_e02"],
        inject_support_fact_episodes=True,
    )

    assert injected == ["STEP-COMPANION_e01", "STEP-COMPANION_e02"]
    assert "[Episode: STEP-COMPANION_e02]" in context
    assert "execute_bash: echo ok" in context


def test_exact_step_queries_prepend_same_source_step_episode_when_late_fusion_misses(monkeypatch):
    corpus = {
        "documents": [{
            "doc_id": "document:STEP-PRIORITY",
            "episodes": [
                {
                    "episode_id": "STEP-PRIORITY_e01",
                    "source_type": "document",
                    "source_id": "STEP-PRIORITY",
                    "source_date": "2026-01-01",
                    "topic_key": "schema noise",
                    "state_label": "schema",
                    "currentness": "historical",
                    "raw_text": "Action: inspect schema\nObservation: generic schema output",
                    "provenance": {"block_ids": ["STEP-PRIORITY_b001"]},
                },
                {
                    "episode_id": "STEP-PRIORITY_e02",
                    "source_type": "document",
                    "source_id": "STEP-PRIORITY",
                    "source_date": "2026-01-01",
                    "topic_key": "step 8 action",
                    "state_label": "action",
                    "currentness": "historical",
                    "raw_text": (
                        "[Step 8]\n"
                        "Action: execute_snowflake_sql: SELECT * FROM wholesale WHERE year BETWEEN 2020 AND 2023\n"
                        "Observation: Query executed successfully"
                    ),
                    "provenance": {"block_ids": ["STEP-PRIORITY_b002"]},
                },
            ],
        }]
    }
    facts = [
        {
            "id": "f_schema",
            "session": 1,
            "fact": "Agent inspected a generic schema output.",
            "metadata": {"episode_id": "STEP-PRIORITY_e01", "episode_source_id": "STEP-PRIORITY"},
        },
        {
            "id": "f_step_8",
            "session": 2,
            "fact": "At step 8, the agent ran SQL over wholesale for years 2020 through 2023.",
            "metadata": {"episode_id": "STEP-PRIORITY_e02", "episode_source_id": "STEP-PRIORITY"},
        },
    ]

    def _wrong_late_fusion(_question, _family_results, _episode_lookup, _selector):
        return {
            "selected_ids": ["STEP-PRIORITY_e01"],
            "scored": [("STEP-PRIORITY_e01", 0.0)],
            "trace": {},
        }

    monkeypatch.setattr("src.memory.select_episode_ids_late_fusion_with_trace", _wrong_late_fusion)
    packet = build_episode_hybrid_context(
        "At step 8, what SQL did the agent run?",
        corpus,
        facts,
    )

    assert packet["retrieved_episode_ids"][0] == "STEP-PRIORITY_e02"
    assert "[Episode: STEP-PRIORITY_e02]" in packet["context"]
    assert "BETWEEN 2020 AND 2023" in packet["context"]


def test_recall_keeps_zero_fact_document_step_episode_visible(monkeypatch, tmp_path):
    corpus = {
        "documents": [{
            "doc_id": "document:STEP-VISIBLE",
            "episodes": [
                {
                    "episode_id": "STEP-VISIBLE_e01",
                    "source_type": "document",
                    "source_id": "STEP-VISIBLE",
                    "source_date": "2026-01-01",
                    "topic_key": "schema noise",
                    "state_label": "schema",
                    "currentness": "historical",
                    "raw_text": "Action: inspect schema\nObservation: generic schema output",
                    "provenance": {"block_ids": ["STEP-VISIBLE_b001"]},
                },
                {
                    "episode_id": "STEP-VISIBLE_e02",
                    "source_type": "document",
                    "source_id": "STEP-VISIBLE",
                    "source_date": "2026-01-01",
                    "topic_key": "step 8 action",
                    "state_label": "action",
                    "currentness": "historical",
                    "raw_text": (
                        "[Step 8]\n"
                        "Action: execute_snowflake_sql: SELECT * FROM wholesale WHERE year BETWEEN 2020 AND 2023\n"
                        "Observation: Query executed successfully"
                    ),
                    "provenance": {"block_ids": ["STEP-VISIBLE_b002"]},
                },
            ],
        }]
    }
    ms = MemoryServer(str(tmp_path), "step_visible_runtime")
    ms._data_dict = {}
    ms._episode_corpus = corpus
    ms._raw_sessions = [{"session_num": 1}]
    ms._all_granular = [
        {
            "id": "f_schema",
            "session": 1,
            "fact": "Agent inspected a generic schema output.",
            "kind": "fact",
            "metadata": {"episode_id": "STEP-VISIBLE_e01", "episode_source_id": "STEP-VISIBLE"},
        },
    ]
    ms._all_cons = []
    ms._all_cross = []
    ms._fact_lookup = {fact["id"]: fact for fact in ms._all_granular}
    ms._source_records = {"STEP-VISIBLE": {"family": "document"}}

    def _wrong_late_fusion(_question, _family_results, _episode_lookup, _selector):
        return {
            "selected_ids": ["STEP-VISIBLE_e01"],
            "scored": [("STEP-VISIBLE_e01", 0.0)],
            "trace": {},
        }

    monkeypatch.setattr("src.memory.select_episode_ids_late_fusion_with_trace", _wrong_late_fusion)
    result = asyncio.run(ms.recall("At step 8, what SQL did the agent run?"))

    assert result["retrieved_episode_ids"][0] == "STEP-VISIBLE_e02"
    assert "[Episode: STEP-VISIBLE_e02]" in result["context"]
    assert "BETWEEN 2020 AND 2023" in result["context"]


def test_temporal_step_queries_preserve_granular_step_facts_in_runtime(tmp_path):
    corpus = {
        "documents": [{
            "doc_id": "document:SQLSTEP",
            "episodes": [{
                "episode_id": "SQLSTEP_e01",
                "source_type": "document",
                "source_id": "SQLSTEP",
                "source_date": "2026-01-01",
                "topic_key": "step 6 sql",
                "state_label": "action",
                "currentness": "historical",
                "raw_text": (
                    "[Step 6]\n"
                    "Action: execute_snowflake_sql: "
                    "SELECT previous_volume FROM t WHERE ticker = 'BTC'\n"
                    "Observation: ok"
                ),
                "provenance": {"block_ids": ["SQLSTEP_b001"]},
            }],
        }]
    }
    ms = MemoryServer(str(tmp_path), "temporal_step_runtime")
    ms._data_dict = {}
    ms._episode_corpus = corpus
    ms._raw_sessions = [{"session_num": 1}]
    granular_fact = {
        "id": "f_sql",
        "session": 1,
        "fact": "At step 6, the agent executed SQL to inspect previous_volume for ticker BTC.",
        "kind": "fact",
        "metadata": {"episode_id": "SQLSTEP_e01", "episode_source_id": "SQLSTEP"},
    }
    temporal_semantic = {
        "id": "x_temp",
        "session": 1,
        "fact": "Temporal summary: the second execution happened at step 6.",
        "kind": "fact",
        "metadata": {
            "episode_id": "SQLSTEP_e01",
            "episode_source_id": "SQLSTEP",
            "source_aggregation": True,
            "semantic_class": "temporal_semantics",
        },
    }
    ms._all_granular = [granular_fact]
    ms._all_cons = []
    ms._all_cross = [temporal_semantic]
    ms._fact_lookup = {fact["id"]: fact for fact in [granular_fact, temporal_semantic]}
    ms._source_records = {"SQLSTEP": {"family": "document"}}

    result = asyncio.run(
        ms.recall(
            "At step 6, after the second execution, what exact SQL action did the agent execute?"
        )
    )

    assert result["query_type"] == "temporal"
    assert "previous_volume for ticker BTC" in result["context"]
    assert "Temporal summary: the second execution happened at step 6." not in result["context"]


def test_step_trajectory_notes_can_use_companion_episodes_for_range_queries():
    corpus = {
        "documents": [{
            "doc_id": "document:STEP-TRAJECTORY",
            "episodes": [
                {
                    "episode_id": "STEP-TRAJECTORY_e01",
                    "source_type": "document",
                    "source_id": "STEP-TRAJECTORY",
                    "source_date": "2026-01-01",
                    "topic_key": "step 1 marker",
                    "state_label": "step_marker",
                    "currentness": "historical",
                    "raw_text": "Observation: marker only\n[Step 1]",
                    "provenance": {"block_ids": ["STEP-TRAJECTORY_b001"]},
                },
                {
                    "episode_id": "STEP-TRAJECTORY_e02",
                    "source_type": "document",
                    "source_id": "STEP-TRAJECTORY",
                    "source_date": "2026-01-01",
                    "topic_key": "step 1 action",
                    "state_label": "action_observation",
                    "currentness": "historical",
                    "raw_text": (
                        "Action: down\n"
                        "Observation: stable\n\n"
                        "Objects on the map:\n"
                        "crate 1 step down\n"
                        "goal 2 steps down\n"
                        "[Step 2]"
                    ),
                    "provenance": {"block_ids": ["STEP-TRAJECTORY_b002"]},
                },
                {
                    "episode_id": "STEP-TRAJECTORY_e03",
                    "source_type": "document",
                    "source_id": "STEP-TRAJECTORY",
                    "source_date": "2026-01-01",
                    "topic_key": "step 2 action",
                    "state_label": "action_observation",
                    "currentness": "historical",
                    "raw_text": (
                        "Action: down\n"
                        "Observation: stable\n\n"
                        "Objects on the map:\n"
                        "crate 1 step down\n"
                        "goal 1 step down\n"
                        "[Step 3]"
                    ),
                    "provenance": {"block_ids": ["STEP-TRAJECTORY_b003"]},
                },
                {
                    "episode_id": "STEP-TRAJECTORY_e04",
                    "source_type": "document",
                    "source_id": "STEP-TRAJECTORY",
                    "source_date": "2026-01-01",
                    "topic_key": "step 3 action",
                    "state_label": "action_observation",
                    "currentness": "historical",
                    "raw_text": (
                        "Action: down\n"
                        "Observation: stable\n\n"
                        "Objects on the map:\n"
                        "crate 1 step down\n"
                        "goal 0 steps down\n"
                    ),
                    "provenance": {"block_ids": ["STEP-TRAJECTORY_b004"]},
                },
            ],
        }]
    }
    episode_lookup = build_episode_lookup(corpus)
    context, _injected, _selected_fact_ids = build_context_from_selected_episodes(
        "From step 1 to step 3, what is being pushed and why?",
        ["STEP-TRAJECTORY_e01", "STEP-TRAJECTORY_e02", "STEP-TRAJECTORY_e03"],
        episode_lookup,
        {},
        support_episode_ids=["STEP-TRAJECTORY_e04"],
        inject_support_fact_episodes=True,
    )

    assert "STEP TRAJECTORY NOTES:" in context
    assert "Steps 1-3 all use action `down`." in context
    assert "crate stays directly ahead of the agent" in context


def test_support_episode_pool_size_knob_is_consumed(monkeypatch):
    corpus = {
        "documents": [{
            "doc_id": "document:DOC-POOL",
            "episodes": [
                {
                    "episode_id": "DOC-POOL_e01",
                    "source_type": "document",
                    "source_id": "DOC-POOL",
                    "source_date": "2026-01-01",
                    "topic_key": "inspection fail",
                    "state_label": "inspection_fail",
                    "currentness": "historical",
                    "raw_text": "At km 2.3 the pipe cover depth was 780mm before rework.",
                    "provenance": {"block_ids": ["DOC-POOL_b001"]},
                },
                {
                    "episode_id": "DOC-POOL_e02",
                    "source_type": "document",
                    "source_id": "DOC-POOL",
                    "source_date": "2026-01-02",
                    "topic_key": "rework order",
                    "state_label": "rework_required",
                    "currentness": "historical",
                    "raw_text": "The section at km 2.3 must be reworked to at least 900mm depth.",
                    "provenance": {"block_ids": ["DOC-POOL_b002"]},
                },
                {
                    "episode_id": "DOC-POOL_e03",
                    "source_type": "document",
                    "source_id": "DOC-POOL",
                    "source_date": "2026-01-03",
                    "topic_key": "reinspection",
                    "state_label": "reinspection_passed",
                    "currentness": "current",
                    "raw_text": "After re-inspection, the pipe cover depth at km 2.3 was 980mm.",
                    "provenance": {"block_ids": ["DOC-POOL_b003"]},
                },
            ],
        }]
    }
    facts = [
        {"id": "f1", "session": 1, "fact": "At km 2.3 the pipe cover depth was 780mm before rework.", "metadata": {"episode_id": "DOC-POOL_e01", "episode_source_id": "DOC-POOL"}},
        {"id": "f2", "session": 2, "fact": "The section at km 2.3 must be reworked to at least 900mm depth.", "metadata": {"episode_id": "DOC-POOL_e02", "episode_source_id": "DOC-POOL"}},
        {"id": "f3", "session": 3, "fact": "After re-inspection, the pipe cover depth at km 2.3 was 980mm.", "metadata": {"episode_id": "DOC-POOL_e03", "episode_source_id": "DOC-POOL"}},
    ]

    def _tuning(pool_size: int):
        return {
            "routing": {"max_plausible_families": 2, "ambiguous_family_fanout": 2},
            "episodes": {"document_grouping": {"prompt_mode": "strict_small", "size_cap_chars": 12000, "strip_duplicates": True, "attach_missing": False, "singleton_missing": True}},
            "retrieval": {"selector": DEFAULT_SELECTION_CONFIG},
            "operators": {
                "ordinal_candidate_budget": 12,
                "compare_alignment_budget": 12,
                "list_set_dedup_overlap": 0.9,
                "list_set_max_episodes": 12,
                "list_set_supporting_facts_total": 12,
                "list_set_supporting_facts_per_episode": 2,
                "enable_snippet_for_ordinal": True,
                "enable_snippet_for_local_anchor": True,
                "local_anchor_max_episodes": 1,
                "local_anchor_window_chars": 1200,
                "local_anchor_fact_radius": 12,
                "local_anchor_supporting_facts_total": 12,
                "local_anchor_supporting_facts_per_episode": 6,
                "structural_query_supporting_facts_total": 12,
                "structural_query_supporting_facts_per_episode": 4,
                "bounded_chain_max_hops": 2,
                "bounded_chain_fact_bonus": 0.0,
            },
            "packet": {
                "budget": 8000,
                "max_facts": 10,
                "max_facts_per_episode": 3,
                "max_episodes": 2,
                "support_episode_pool_size": pool_size,
                "per_source_cap": 2,
                "per_family_cap": 3,
                "snippet_mode": False,
                "snippet_chars": 1200,
            },
            "telemetry": {"include_runtime_trace": True, "max_selection_scores": 12, "max_scope_source_ids": 32, "max_family_candidates": 8, "max_rejected_candidates": 8, "max_packet_fact_ids": 24},
        }

    monkeypatch.setattr("src.memory.get_runtime_tuning", lambda: _tuning(2))
    packet_small = build_episode_hybrid_context(
        "What is the current pipe cover depth at km 2.3?",
        corpus,
        facts,
    )
    monkeypatch.setattr("src.memory.get_runtime_tuning", lambda: _tuning(3))
    packet_large = build_episode_hybrid_context(
        "What is the current pipe cover depth at km 2.3?",
        corpus,
        facts,
    )

    assert len(packet_small["fact_episode_ids"]) == 2
    assert len(packet_large["fact_episode_ids"]) == 3
    assert "DOC-POOL_e02" not in packet_small["fact_episode_ids"]
    assert "DOC-POOL_e02" in packet_large["fact_episode_ids"]


def test_packet_builder_can_disable_pseudo_fact_fallback():
    corpus = _make_corpus()
    episode_lookup = build_episode_lookup(corpus)

    context, injected, selected_fact_ids = build_context_from_selected_episodes(
        "What is the final approved route length?",
        ["DOC-1_e01"],
        episode_lookup,
        {},
        allow_pseudo_facts=False,
    )

    assert injected == ["DOC-1_e01"]
    assert selected_fact_ids == []
    assert context.startswith("RETRIEVED FACTS:")
    assert "[1]" not in context
    assert "[Episode: DOC-1_e01]" in context


def test_karnali_wrapper_requires_episode_lookup_not_raw_index():
    facts = _make_facts()
    fact_lookup = {fact["id"]: fact for fact in facts}

    with pytest.raises(TypeError):
        build_episode_context(
            [{"fact_id": "f_current"}],
            {"DOC-1_e01": "raw episode text"},
            fact_lookup,
        )
