"""Regression tests for generic episode retrieval behavior."""

from __future__ import annotations

from collections import Counter
import numpy as np
import pytest

from src.episode_packet import (
    _fact_list_item_keys,
    _fact_slot_fill_candidates,
    build_bounded_chain_candidate_bundle,
    build_context_from_retrieved_facts,
    build_context_from_selected_episodes,
    pick_supporting_facts,
)
from src.episode_packet import _fact_content_tokens, _fact_rank_tuple
from src.episode_retrieval import (
    _apply_bounded_chain_expansion,
    _chain_tokens_with_frontier,
    build_episode_bm25,
    choose_episode_ids,
    choose_episode_ids_with_trace,
    route_retrieval_families,
    select_episode_ids_late_fusion,
)
from src.episode_features import extract_query_features
from src.common import normalize_term_token
from src.retrieval import BM25Index, source_local_fact_sweep
from src.memory import MemoryServer, build_episode_hybrid_context
from src.tuning import get_runtime_tuning


def _episode(
    ep_id: str,
    raw_text: str,
    *,
    source_id: str = "conv_a",
    source_type: str = "conversation",
    topic: str = "session",
) -> dict:
    return {
        "episode_id": ep_id,
        "source_type": source_type,
        "source_id": source_id,
        "source_date": "2024-06-01",
        "topic_key": topic,
        "state_label": "session",
        "currentness": "unknown",
        "raw_text": raw_text,
        "provenance": {"raw_span": [0, len(raw_text)]},
    }


def test_route_retrieval_families_prefers_explicit_family():
    routed = route_retrieval_families(
        "How many cats does Alice have?",
        ["conversation", "document"],
        explicit_family="document",
    )
    assert routed == ["document"]


def test_extract_query_features_trims_possessive_slot_head_prepositional_tail():
    qf = extract_query_features("What is John's main focus in local politics?")

    assert qf["operator_plan"]["slot_query"]["enabled"] is True
    assert qf["operator_plan"]["slot_query"]["head_phrase"] == "main focus"
    assert qf["operator_plan"]["slot_query"]["head_tokens"] == ["main", "focus"]


def test_extract_query_features_marks_slot_query_for_type_request():
    qf = extract_query_features("What kind of project is Gina doing?")

    assert qf["operator_plan"]["slot_query"]["enabled"] is True
    assert qf["operator_plan"]["slot_query"]["head_phrase"] == "project"
    assert qf["operator_plan"]["list_set"]["enabled"] is False


def test_fact_slot_fill_candidates_ignore_meta_answer_scaffolding():
    qf = extract_query_features("What temporary job did Jon take to cover expenses?")

    candidates = _fact_slot_fill_candidates(
        "Answer: Jon took a temporary job to help cover expenses while working on his dance studio and seeking investors. This is explicitly stated in the RAW CONTEXT.",
        qf,
    )

    assert candidates == []


def test_extract_query_features_treats_explicit_step_reference_as_anchor_not_ordinal():
    qf = extract_query_features("At step 7, which tool did the agent switch to?")

    assert qf["step_numbers"] == {7}
    assert qf["operator_plan"]["local_anchor"]["enabled"] is True
    assert qf["operator_plan"]["ordinal"]["enabled"] is False


def test_extract_query_features_detects_explicit_step_range():
    qf = extract_query_features(
        "From step 25 to step 28, what actions made the environment changes?"
    )

    assert qf["step_numbers"] == {25, 28}
    assert qf["step_range"] == (25, 28)
    assert qf["operator_plan"]["local_anchor"]["enabled"] is True
    assert qf["operator_plan"]["ordinal"]["enabled"] is False


def test_choose_episode_ids_does_not_treat_step_prefixes_as_exact_matches():
    corpus = {
        "documents": [{
            "doc_id": "DOC-STEPS",
            "episodes": [
                _episode("step_2", "[Step 2]\nAction: execute_bash:\nObservation: API error 500", source_id="DOC-STEPS", source_type="document"),
                _episode("step_20", "[Step 20]\nAction: execute_snowflake_sql:\nObservation: API error 500", source_id="DOC-STEPS", source_type="document"),
                _episode("step_21", "[Step 21]\nAction: execute_snowflake_sql:\nObservation: API error 500", source_id="DOC-STEPS", source_type="document"),
            ],
        }]
    }
    lookup = {ep["episode_id"]: ep for doc in corpus["documents"] for ep in doc["episodes"]}
    bm25 = build_episode_bm25(corpus)

    result = choose_episode_ids_with_trace(
        "At step 2, which tool action was used?",
        bm25,
        lookup,
        {"max_candidates": 10, "max_episodes_default": 3, "max_sources_per_family": 1},
    )

    rows = {row["episode_id"]: row for row in result["trace"]["pre_source_gate"]}
    assert rows["step_2"]["score_breakdown"]["step"] > 0
    assert rows["step_20"]["score_breakdown"]["step"] == 0
    assert rows["step_21"]["score_breakdown"]["step"] == 0
    assert result["selected_ids"][0] == "step_2"


def test_choose_episode_ids_rewards_interior_step_range_coverage():
    corpus = {
        "documents": [{
            "doc_id": "DOC-RANGE",
            "episodes": [
                _episode(
                    "step_26",
                    "[Step 26]\nAction: move creditcard 2 to dresser 1\nObservation: changed",
                    source_id="DOC-RANGE",
                    source_type="document",
                ),
                _episode(
                    "step_27",
                    "[Step 27]\nAction: take book 2 from bed 1\nObservation: changed",
                    source_id="DOC-RANGE",
                    source_type="document",
                ),
                _episode(
                    "step_41",
                    "[Step 41]\nAction: move book 1 to desk 1\nObservation: changed",
                    source_id="DOC-RANGE",
                    source_type="document",
                ),
            ],
        }]
    }
    lookup = {ep["episode_id"]: ep for doc in corpus["documents"] for ep in doc["episodes"]}
    bm25 = build_episode_bm25(corpus)

    result = choose_episode_ids_with_trace(
        "From step 25 to step 28, what actions made the environment changes?",
        bm25,
        lookup,
        {"max_candidates": 10, "max_episodes_default": 3, "max_sources_per_family": 1},
    )

    rows = {row["episode_id"]: row for row in result["trace"]["pre_source_gate"]}
    assert rows["step_26"]["score_breakdown"]["step"] > 0
    assert rows["step_27"]["score_breakdown"]["step"] > 0
    assert rows["step_41"]["score_breakdown"]["step"] == 0
    assert result["selected_ids"][:2] == ["step_26", "step_27"]


def test_pick_supporting_facts_prefers_exact_step_over_prefix_steps():
    question = "At step 2, which tool action was used?"
    qf = extract_query_features(question)
    selected = pick_supporting_facts(
        question,
        ["ep_trace"],
        {
            "ep_trace": [
                {
                    "id": "trace_20",
                    "fact": "At step 20, the agent executed execute_snowflake_sql.",
                    "session": 1,
                    "metadata": {"episode_id": "ep_trace"},
                },
                {
                    "id": "trace_2",
                    "fact": "At step 2, the agent executed execute_bash.",
                    "session": 1,
                    "metadata": {"episode_id": "ep_trace"},
                },
                {
                    "id": "trace_21",
                    "fact": "At step 21, the agent executed execute_snowflake_sql.",
                    "session": 1,
                    "metadata": {"episode_id": "ep_trace"},
                },
            ]
        },
        max_total=3,
        max_per_episode=3,
        allow_pseudo_facts=False,
        query_features=qf,
    )

    assert [fact["id"] for fact in selected][:1] == ["trace_2"]


def test_pick_supporting_facts_prefers_interior_range_steps_over_unrelated_steps():
    question = "From step 25 to step 28, what actions made the environment changes?"
    qf = extract_query_features(question)
    selected = pick_supporting_facts(
        question,
        ["ep_trace"],
        {
            "ep_trace": [
                {
                    "id": "trace_41",
                    "fact": "At step 41, the agent moved book 1 to desk 1.",
                    "session": 1,
                    "metadata": {"episode_id": "ep_trace"},
                },
                {
                    "id": "trace_26",
                    "fact": "At step 26, the agent moved creditcard 2 to dresser 1.",
                    "session": 1,
                    "metadata": {"episode_id": "ep_trace"},
                },
                {
                    "id": "trace_27",
                    "fact": "At step 27, the agent took book 2 from bed 1.",
                    "session": 1,
                    "metadata": {"episode_id": "ep_trace"},
                },
            ]
        },
        max_total=3,
        max_per_episode=3,
        allow_pseudo_facts=False,
        query_features=qf,
    )

    assert [fact["id"] for fact in selected][:2] == ["trace_26", "trace_27"]


def test_extract_query_features_ignores_boolean_glue_in_list_set_heads():
    qf = extract_query_features("Which places or events have Alice and Ben planned for the future?")

    assert qf["operator_plan"]["list_set"]["enabled"] is True
    assert qf["operator_plan"]["list_set"]["head_tokens"] == ["place", "event"]


def test_extract_query_features_supports_slash_separated_list_heads():
    qf = extract_query_features("Which places/events have Alice and Ben planned for the future?")

    assert qf["operator_plan"]["list_set"]["enabled"] is True
    assert qf["operator_plan"]["list_set"]["head_tokens"] == ["place", "event"]


def test_extract_query_features_marks_compositional_for_multi_constraint_query():
    qf = extract_query_features(
        "What hobby would make Toby happy while also helping Maria learn new recipes?"
    )

    assert qf["operator_plan"]["compositional"]["enabled"] is True


def test_extract_query_features_marks_constraint_while_query_as_compositional():
    qf = extract_query_features(
        "What is an indoor activity that Andrew would enjoy doing while make his dog happy?"
    )

    assert qf["operator_plan"]["compositional"]["enabled"] is True


def test_extract_query_features_leaves_bare_while_question_as_bounded_chain():
    qf = extract_query_features("What was Maria doing while John was cooking?")

    assert qf["operator_plan"]["bounded_chain"]["enabled"] is True
    assert qf["operator_plan"]["slot_query"]["enabled"] is False
    assert qf["operator_plan"]["compositional"]["enabled"] is False


def test_extract_query_features_marks_temporal_grounding_for_absolute_date_reference():
    qf = extract_query_features("Who did Maria make dinner with on May 3, 2023?")

    assert qf["operator_plan"]["temporal_grounding"]["enabled"] is True


def test_route_retrieval_families_fans_out_when_ambiguous():
    routed = route_retrieval_families(
        "How many cats does Alice have?",
        ["conversation", "document"],
        explicit_family="auto",
    )
    assert routed == ["conversation", "document"]


def test_slot_fill_candidates_reject_broader_topic_without_requested_field_fill():
    qf = extract_query_features("What temporary job did Jon take to help cover expenses?")

    assert _fact_slot_fill_candidates(
        "Jon took a temporary job to help cover expenses.",
        qf,
    ) == []


def test_slot_fill_candidates_reject_temp_job_paraphrase_without_actual_type():
    qf = extract_query_features("What temporary job did Jon take to help cover expenses?")

    assert _fact_slot_fill_candidates(
        "I got a temp job to help cover expenses while I look for investors.",
        qf,
    ) == []


def test_slot_fill_candidates_accept_capitalized_value_even_when_query_mentions_old_instance():
    qf = extract_query_features("What type of car did Evan get after his old Prius broke down?")

    assert _fact_slot_fill_candidates(
        "Evan owns a new Prius.",
        qf,
    ) == ["Prius"]


def test_slot_fill_candidates_accept_plural_head_with_copular_suffix_value():
    qf = extract_query_features("What is John's main focus in local politics?")

    assert _fact_slot_fill_candidates(
        "John's main focuses are improving education and infrastructure in his community.",
        qf,
    ) == ["improving education and infrastructure"]


def test_slot_fill_candidates_preserve_coordination_for_plural_focus_fact():
    qf = extract_query_features("What is John's main focus in local politics?")

    assert _fact_slot_fill_candidates(
        "Improving education and infrastructure are John's main focuses.",
        qf,
    ) == ["Improving education and infrastructure"]


def test_slot_fill_candidates_accept_specific_type_value():
    qf = extract_query_features("What kind of project is Gina doing?")

    assert _fact_slot_fill_candidates(
        "Gina is doing an electrical engineering project for school.",
        qf,
    ) == ["electrical engineering"]


def test_slot_fill_candidates_do_not_treat_preposition_plus_article_as_prefix_value():
    qf = extract_query_features("What kind of project was Jolene working on?")

    assert _fact_slot_fill_candidates(
        "Jolene had a big breakthrough with the project.",
        qf,
        allow_loose_fallback=False,
    ) == []


def test_slot_fill_candidates_strict_mode_ignores_loose_unrelated_titleish_fallback():
    qf = extract_query_features("What is John's main focus in local politics?")

    assert _fact_slot_fill_candidates(
        "John is doing kickboxing and it gives him a lot of energy.",
        qf,
        allow_loose_fallback=False,
    ) == []


def test_choose_episode_ids_prefers_coherent_source_cluster():
    corpus = {
        "documents": [
            {
                "doc_id": "conversation:conv_a",
                "episodes": [
                    _episode("a_e1", "Alice bought apples at North Market.", source_id="conv_a"),
                    _episode("a_e2", "Alice bought milk at North Market.", source_id="conv_a"),
                ],
            },
            {
                "doc_id": "conversation:conv_b",
                "episodes": [
                    _episode(
                        "b_e1",
                        "Alice likes apples and milk and talks about markets often.",
                        source_id="conv_b",
                    ),
                ],
            },
        ]
    }
    lookup = {
        ep["episode_id"]: ep
        for doc in corpus["documents"]
        for ep in doc["episodes"]
    }
    bm25 = build_episode_bm25(corpus)
    selected, _scored = choose_episode_ids(
        "Where did Alice buy apples and milk?",
        bm25,
        lookup,
        {"max_candidates": 10, "max_episodes_default": 3, "max_sources_per_family": 2},
    )
    assert selected[:2] == ["a_e1", "a_e2"]
    assert "b_e1" not in selected[:2]


def test_choose_episode_ids_uses_retrieval_target_not_raw_output_instruction():
    class _DummyBM25:
        def __init__(self):
            self.queries = []

        def search(self, query, top_k):
            self.queries.append((query, top_k))
            return []

    bm25 = _DummyBM25()
    choose_episode_ids_with_trace(
        "Prepend lPiXbALSeQ to the 1st song about frames. Do not include any other text.",
        bm25,
        {},
        {"max_candidates": 7},
    )
    assert bm25.queries[0][0] == "the 1st song about frames"
    assert bm25.queries[0][1] == 7


def test_generic_output_directive_does_not_create_fake_bounded_chain_anchor():
    qf = extract_query_features("What benchmark is listed? Return the exact token.")

    assert qf["retrieval_target"] == "What benchmark is listed"
    assert qf["entity_phrases"] == []
    assert qf["operator_plan"]["bounded_chain"]["enabled"] is False


def test_literal_return_text_is_preserved_in_retrieval_target():
    qf = extract_query_features("What did the note say? Return to sender or destroy?")

    assert qf["retrieval_target"] == "What did the note say? Return to sender or destroy?"


def test_literal_provide_text_is_preserved_in_retrieval_target():
    qf = extract_query_features("What did the note say? Provide the keys and lockbox code?")

    assert qf["retrieval_target"] == "What did the note say? Provide the keys and lockbox code?"


def test_literal_include_text_is_preserved_in_retrieval_target():
    qf = extract_query_features("What did the note say? Include Evelyn's pager number and floor?")

    assert qf["retrieval_target"] == "What did the note say? Include Evelyn's pager number and floor?"


def test_provide_only_directive_is_stripped_from_retrieval_target():
    qf = extract_query_features("What permit is active? Provide only the code.")

    assert qf["retrieval_target"] == "What permit is active"


def test_bm25_normalizes_singular_plural_terms():
    bm25 = BM25Index(
        [
            "The project needs a permit approval.",
            "These permits are required for site entry.",
        ],
        ["permit_singular", "permit_plural"],
    )

    hits = bm25.search("What permits does the project require?", top_k=5)
    assert [row["id"] for row in hits][:2] == ["permit_singular", "permit_plural"]


def test_source_local_fact_sweep_prefers_semantic_answer_facts():
    facts = [
        {"id": "noise_1", "fact": "John and others had fun making pizza together", "entities": ["John"]},
        {"id": "noise_2", "fact": "John helped renovate schools in the community", "entities": ["John"]},
        {"id": "kick", "fact": "John is doing kickboxing", "entities": ["John"]},
        {"id": "tae", "fact": "John is going to do taekwondo", "entities": ["John"]},
    ]
    embeddings = np.array(
        [
            [0.0, 1.0],
            [0.1, 0.9],
            [1.0, 0.0],
            [0.9, 0.1],
        ],
        dtype=float,
    )
    result = source_local_fact_sweep(
        "What martial arts has John done?",
        facts,
        embeddings,
        query_embedding=np.array([1.0, 0.0], dtype=float),
        top_k=2,
        bm25_pool=4,
        vector_pool=4,
        entity_pool=4,
        rrf_k=60,
    )

    assert [row["fact"]["id"] for row in result["retrieved"][:2]] == ["kick", "tae"]
    assert result["trace"]["mode"] == "hybrid_fact_sweep"


def test_source_local_fact_sweep_preserves_distinct_plan_list_items():
    facts = [
        {"id": "noise", "fact": "Alice and Ben discussed hobbies and weekend weather", "entities": ["Alice", "Ben"]},
        {"id": "cafe", "fact": "Alice will meet at Harbor Cafe tomorrow", "entities": ["Alice"]},
        {"id": "workshop", "fact": "We will attend the robotics workshop next Saturday", "entities": ["Alice", "Ben"]},
        {"id": "concert", "fact": "Ben and Alice are going to the jazz concert on 2024-06-01", "entities": ["Ben", "Alice"]},
    ]
    embeddings = np.array(
        [
            [0.2, 0.8],
            [1.0, 0.0],
            [0.95, 0.05],
            [0.9, 0.1],
        ],
        dtype=float,
    )
    result = source_local_fact_sweep(
        "Which places or events have Alice and Ben planned for the future?",
        facts,
        embeddings,
        query_embedding=np.array([1.0, 0.0], dtype=float),
        top_k=3,
        bm25_pool=4,
        vector_pool=4,
        entity_pool=4,
        rrf_k=60,
    )

    assert {row["fact"]["id"] for row in result["retrieved"][:3]} == {"cafe", "workshop", "concert"}


def test_source_local_fact_sweep_covers_compositional_components():
    facts = [
        {"id": "noise", "fact": "Andrew loves hiking with his dog outdoors", "entities": ["Andrew"]},
        {"id": "cook", "fact": "Cooking has been enjoyable for Andrew", "entities": ["Andrew"]},
        {"id": "dog_happy", "fact": "The dogs are doing great and look happy", "entities": ["Andrew"]},
    ]
    embeddings = np.array(
        [
            [0.4, 0.6],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    result = source_local_fact_sweep(
        "What is an indoor activity that Andrew would enjoy doing while make his dog happy?",
        facts,
        embeddings,
        query_embedding=np.array([0.7, 0.7], dtype=float),
        top_k=2,
        bm25_pool=3,
        vector_pool=3,
        entity_pool=3,
        rrf_k=60,
    )

    assert {row["fact"]["id"] for row in result["retrieved"][:2]} == {"cook", "dog_happy"}


def test_source_local_fact_sweep_status_queries_use_generic_query_overlap():
    facts = [
        {"id": "noise_1", "fact": "Riley had a long day", "entities": ["Riley"]},
        {"id": "noise_2", "fact": "Riley talked with Sam about errands", "entities": ["Riley"]},
        {"id": "signal", "fact": "Riley's current situation is unstable because rent payments are overdue.", "entities": ["Riley"]},
    ]
    embeddings = np.array(
        [
            [0.5, 0.5],
            [0.4, 0.6],
            [1.0, 0.0],
        ],
        dtype=float,
    )
    result = source_local_fact_sweep(
        "What is Riley's current situation?",
        facts,
        embeddings,
        query_embedding=np.array([0.8, 0.2], dtype=float),
        top_k=2,
        bm25_pool=3,
        vector_pool=3,
        entity_pool=3,
        rrf_k=60,
    )

    assert result["retrieved"][0]["fact"]["id"] == "signal"


def test_build_context_from_retrieved_facts_preserves_local_fact_order_within_episode():
    episode_lookup = {
        "doc_e1": _episode(
            "doc_e1",
            "40. Christianity was founded in the city of Jerusalem.\n43. Christianity was founded in the city of Taipei.",
            source_id="doc_shared",
            source_type="document",
            topic="section_1",
        )
    }
    fact_lookup = {
        "s477_f_44": {
            "id": "s477_f_44",
            "session": 477,
            "fact": "Christianity was founded in the city of Taipei.",
            "metadata": {"episode_id": "doc_e1"},
        },
        "s477_f_41": {
            "id": "s477_f_41",
            "session": 477,
            "fact": "Christianity was founded in the city of Jerusalem.",
            "metadata": {"episode_id": "doc_e1"},
        },
    }
    context, _injected = build_context_from_retrieved_facts(
        [fact_lookup["s477_f_44"], fact_lookup["s477_f_41"]],
        episode_lookup,
        fact_lookup,
        budget=4000,
    )
    assert context.index("Jerusalem") < context.index("Taipei")


def test_build_context_from_retrieved_facts_snippets_multiple_episodes_under_budget():
    episode_lookup = {
        "doc_e1": _episode(
            "doc_e1",
            ("noise " * 1000) + "\nBagratuni Dynasty is affiliated with the religion of Christianity.\n" + ("tail " * 1000),
            source_id="doc_shared",
            source_type="document",
            topic="section_1",
        ),
        "doc_e2": _episode(
            "doc_e2",
            ("prefix " * 1000) + "\nChristianity was founded in the city of Taipei.\n" + ("suffix " * 1000),
            source_id="doc_shared",
            source_type="document",
            topic="section_2",
        ),
    }
    fact_lookup = {
        "bagratuni": {
            "id": "bagratuni",
            "session": 422,
            "fact": "Bagratuni Dynasty is affiliated with the religion of Christianity.",
            "metadata": {"episode_id": "doc_e1"},
        },
        "taipei": {
            "id": "taipei",
            "session": 462,
            "fact": "Christianity was founded in the city of Taipei.",
            "metadata": {"episode_id": "doc_e2"},
        },
    }
    context, injected = build_context_from_retrieved_facts(
        [fact_lookup["bagratuni"], fact_lookup["taipei"]],
        episode_lookup,
        fact_lookup,
        budget=1800,
        snippet_chars=500,
    )
    assert injected == ["doc_e1", "doc_e2"]
    assert "Bagratuni Dynasty is affiliated with the religion of Christianity." in context
    assert "Christianity was founded in the city of Taipei." in context
    assert context.count("--- SOURCE EPISODE RAW TEXT ---") == 1


def test_build_context_from_retrieved_facts_uses_episode_ids_list_from_cross_fact_metadata():
    episode_lookup = {
        "doc_e1": _episode(
            "doc_e1",
            "Initial measurement at km 2.3 recorded 780 mm cover depth.",
            source_id="doc_shared",
            source_type="document",
            topic="section_1",
        ),
        "doc_e2": _episode(
            "doc_e2",
            "On remeasurement, the pipe cover depth at km 2.3 was 980 mm.",
            source_id="doc_shared",
            source_type="document",
            topic="section_2",
        ),
    }
    cross_fact = {
        "id": "substrate_rev_001",
        "session": 10,
        "fact": "Current value for pipe cover depth at km 2.3 is 980 mm; this supersedes 780 mm.",
        "metadata": {
            "episode_ids": ["doc_e1", "doc_e2"],
        },
    }
    context, injected = build_context_from_retrieved_facts(
        [cross_fact],
        episode_lookup,
        {"substrate_rev_001": cross_fact},
        budget=4000,
    )
    assert injected == ["doc_e1", "doc_e2"]
    assert "[Episodes: doc_e1, doc_e2]" in context
    assert "[Episode: doc_e1]" in context
    assert "[Episode: doc_e2]" in context


def test_bounded_chain_candidate_bundle_prefers_location_candidates_for_where_query():
    seed = {
        "id": "bagratuni",
        "session": 1,
        "fact": "Bagratuni Dynasty is affiliated with the religion of Christianity.",
        "metadata": {"episode_id": "doc_e1"},
    }
    candidate_facts = [
        seed,
        {
            "id": "founder",
            "session": 2,
            "fact": "Christianity was founded by Noel Pemberton Billing.",
            "metadata": {"episode_id": "doc_e2"},
        },
        {
            "id": "jerusalem",
            "session": 3,
            "fact": "Christianity was founded in the city of Jerusalem.",
            "metadata": {"episode_id": "doc_e3"},
        },
        {
            "id": "taipei",
            "session": 4,
            "fact": "Christianity was founded in the city of Taipei.",
            "metadata": {"episode_id": "doc_e4"},
        },
    ]
    bundle = build_bounded_chain_candidate_bundle(
        "Where did the religion associated with the Bagratuni Dynasty come into existence?",
        [seed],
        candidate_facts,
        max_candidates=4,
    )
    selected_ids = [fact["id"] for fact in bundle["facts"]]
    assert selected_ids[0] == "bagratuni"
    assert selected_ids[1] in {"jerusalem", "taipei"}


def test_bounded_chain_candidate_bundle_dedups_identical_fact_text():
    seed = {
        "id": "bagratuni",
        "session": 1,
        "fact": "Bagratuni Dynasty is affiliated with the religion of Christianity.",
        "metadata": {"episode_id": "doc_e1"},
    }
    candidate_facts = [
        seed,
        {
            "id": "taipei_a",
            "session": 2,
            "fact": "Christianity was founded in the city of Taipei.",
            "metadata": {"episode_id": "doc_e2"},
        },
        {
            "id": "taipei_b",
            "session": 3,
            "fact": "Christianity was founded in the city of Taipei.",
            "metadata": {"episode_id": "doc_e3"},
        },
    ]
    bundle = build_bounded_chain_candidate_bundle(
        "Where did the religion associated with the Bagratuni Dynasty come into existence?",
        [seed],
        candidate_facts,
        max_candidates=4,
    )
    selected_ids = [fact["id"] for fact in bundle["facts"]]
    assert selected_ids == ["bagratuni", "taipei_a"]


@pytest.mark.asyncio
async def test_recall_uses_conversation_source_local_fact_sweep(tmp_path, monkeypatch):
    server = MemoryServer(str(tmp_path), "conv_structural")
    source_id = "conv_john"
    corpus = {
        "documents": [{
            "doc_id": f"conversation:{source_id}",
            "episodes": [
                _episode("conv_e1", "John and others had fun making pizza together.", source_id=source_id),
                _episode("conv_e2", "John helped renovate schools in the community.", source_id=source_id),
                _episode("conv_e3", "John is doing kickboxing.", source_id=source_id),
                _episode("conv_e4", "John is going to do taekwondo.", source_id=source_id),
            ],
        }]
    }
    facts = [
        {
            "id": "noise_1",
            "session": 1,
            "fact": "John and others had fun making pizza together",
            "entities": ["John"],
            "source_id": source_id,
            "metadata": {"episode_id": "conv_e1", "episode_source_id": source_id},
        },
        {
            "id": "noise_2",
            "session": 2,
            "fact": "John helped renovate schools in the community",
            "entities": ["John"],
            "source_id": source_id,
            "metadata": {"episode_id": "conv_e2", "episode_source_id": source_id},
        },
        {
            "id": "kick",
            "session": 3,
            "fact": "John is doing kickboxing",
            "entities": ["John"],
            "source_id": source_id,
            "metadata": {"episode_id": "conv_e3", "episode_source_id": source_id},
        },
        {
            "id": "tae",
            "session": 4,
            "fact": "John is going to do taekwondo",
            "entities": ["John"],
            "source_id": source_id,
            "metadata": {"episode_id": "conv_e4", "episode_source_id": source_id},
        },
    ]
    server._episode_corpus = corpus
    server._all_granular = facts
    server._all_cons = []
    server._all_cross = []
    server._raw_sessions = [
        {"session_num": idx + 1, "format": "conversation", "source_id": source_id}
        for idx in range(4)
    ]
    server._data_dict = {
        "atomic_embs": np.array(
            [
                [0.0, 1.0],
                [0.1, 0.9],
                [1.0, 0.0],
                [0.9, 0.1],
            ],
            dtype=float,
        ),
        "cons_embs": np.zeros((0, 2)),
        "cross_embs": np.zeros((0, 2)),
        "fact_lookup": {fact["id"]: fact for fact in facts},
    }
    server._fact_lookup = {fact["id"]: fact for fact in facts}

    async def _fake_embed_query(_text, model=None, provider=None):
        return np.array([1.0, 0.0], dtype=float)

    monkeypatch.setattr("src.memory.embed_query", _fake_embed_query)

    result = await server.recall("What martial arts has John done?")

    fact_ids = {fact["id"] for fact in result["retrieved"]}
    assert {"kick", "tae"} <= fact_ids
    sweep_trace = result["runtime_trace"]["packet"]["source_local_fact_sweep"]
    assert sweep_trace["mode"] == "hybrid_fact_sweep"
    assert "kick" in sweep_trace["selected_fact_ids"]
    assert "tae" in sweep_trace["selected_fact_ids"]


@pytest.mark.asyncio
async def test_recall_uses_document_source_local_fact_sweep_with_seed_expansion(tmp_path, monkeypatch):
    server = MemoryServer(str(tmp_path), "doc_structural")
    source_id = "doc_shared"
    corpus = {
        "documents": [{
            "doc_id": f"document:{source_id}",
            "episodes": [
                _episode(
                    "doc_e1",
                    "Bagratuni Dynasty is affiliated with the religion of Christianity.",
                    source_id=source_id,
                    source_type="document",
                    topic="section_1",
                ),
                _episode(
                    "doc_e2",
                    "Christianity was founded in the city of Jerusalem.",
                    source_id=source_id,
                    source_type="document",
                    topic="section_2",
                ),
                _episode(
                    "doc_e3",
                    "Christianity was founded in the city of Taipei.",
                    source_id=source_id,
                    source_type="document",
                    topic="section_3",
                ),
                _episode(
                    "doc_e4",
                    "quarterback is associated with the sport of American football.",
                    source_id=source_id,
                    source_type="document",
                    topic="section_4",
                ),
            ],
        }]
    }
    facts = [
        {
            "id": "bagratuni",
            "session": 1,
            "fact": "Bagratuni Dynasty is affiliated with the religion of Christianity.",
            "entities": ["Bagratuni Dynasty", "Christianity"],
            "source_id": source_id,
            "metadata": {"episode_id": "doc_e1", "episode_source_id": source_id},
        },
        {
            "id": "jerusalem",
            "session": 2,
            "fact": "Christianity was founded in the city of Jerusalem.",
            "entities": ["Christianity", "Jerusalem"],
            "source_id": source_id,
            "metadata": {"episode_id": "doc_e2", "episode_source_id": source_id},
        },
        {
            "id": "taipei",
            "session": 3,
            "fact": "Christianity was founded in the city of Taipei.",
            "entities": ["Christianity", "Taipei"],
            "source_id": source_id,
            "metadata": {"episode_id": "doc_e3", "episode_source_id": source_id},
        },
        {
            "id": "noise",
            "session": 4,
            "fact": "quarterback is associated with the sport of American football.",
            "entities": ["quarterback", "American football"],
            "source_id": source_id,
            "metadata": {"episode_id": "doc_e4", "episode_source_id": source_id},
        },
    ]
    server._episode_corpus = corpus
    server._all_granular = facts
    server._all_cons = []
    server._all_cross = []
    server._raw_sessions = []
    server._data_dict = {
        "atomic_embs": np.array(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.8, 0.2],
                [0.0, 1.0],
            ],
            dtype=float,
        ),
        "cons_embs": np.zeros((0, 2)),
        "cross_embs": np.zeros((0, 2)),
        "fact_lookup": {fact["id"]: fact for fact in facts},
    }
    server._fact_lookup = {fact["id"]: fact for fact in facts}

    async def _fake_embed_query(text, model=None, provider=None):
        if "Bagratuni Dynasty is affiliated with the religion of Christianity." in text:
            return np.array([1.0, 0.0], dtype=float)
        return np.array([0.2, 0.8], dtype=float)

    monkeypatch.setattr("src.memory.embed_query", _fake_embed_query)

    result = await server.recall(
        "Where did the religion associated with the Bagratuni Dynasty come into existence?"
    )

    fact_ids = {fact["id"] for fact in result["retrieved"]}
    assert "bagratuni" in fact_ids
    assert "taipei" in fact_ids
    sweep_trace = result["runtime_trace"]["packet"]["source_local_fact_sweep"]
    assert sweep_trace["family"] == "document"
    assert sweep_trace["seed_fact_ids"] == ["bagratuni"]
    assert sweep_trace["mode"] == "bounded_chain_candidate_bundle"
    assert "christianity" in sweep_trace["frontier"]


@pytest.mark.asyncio
async def test_document_source_local_fact_sweep_accepts_pseudo_seed(tmp_path, monkeypatch):
    server = MemoryServer(str(tmp_path), "doc_pseudo_seed")
    source_id = "doc_shared"
    episode_lookup = {
        "doc_e1": _episode(
            "doc_e1",
            "Bagratuni Dynasty is affiliated with the religion of Christianity.",
            source_id=source_id,
            source_type="document",
            topic="section_1",
        ),
        "doc_e2": _episode(
            "doc_e2",
            "Christianity was founded in the city of Jerusalem.",
            source_id=source_id,
            source_type="document",
            topic="section_2",
        ),
        "doc_e3": _episode(
            "doc_e3",
            "Christianity was founded in the city of Taipei.",
            source_id=source_id,
            source_type="document",
            topic="section_3",
        ),
    }
    facts = [
        {
            "id": "jerusalem",
            "session": 2,
            "fact": "Christianity was founded in the city of Jerusalem.",
            "entities": ["Christianity", "Jerusalem"],
            "source_id": source_id,
            "metadata": {"episode_id": "doc_e2", "episode_source_id": source_id},
        },
        {
            "id": "taipei",
            "session": 3,
            "fact": "Christianity was founded in the city of Taipei.",
            "entities": ["Christianity", "Taipei"],
            "source_id": source_id,
            "metadata": {"episode_id": "doc_e3", "episode_source_id": source_id},
        },
    ]
    server._all_granular = facts
    server._all_cons = []
    server._all_cross = []
    server._data_dict = {
        "atomic_embs": np.array(
            [
                [0.8, 0.2],
                [0.8, 0.2],
            ],
            dtype=float,
        ),
    }

    async def _fake_embed_query(_text, model=None, provider=None):
        return None

    monkeypatch.setattr("src.memory.embed_query", _fake_embed_query)

    packet = {
        "query_operator_plan": {"bounded_chain": {"enabled": True}},
        "retrieved_episode_ids": ["doc_e1", "doc_e2", "doc_e3"],
        "retrieved_fact_ids": ["raw_doc_e1_01"],
        "selector_config": {"budget": 8000},
    }

    augmented, retrieved = await server._augment_document_structural_packet(
        query="Where did the religion associated with the Bagratuni Dynasty come into existence?",
        packet=packet,
        episode_lookup=episode_lookup,
        fact_filter=lambda _fact: True,
    )

    fact_ids = {fact["id"] for fact in retrieved}
    assert "taipei" in fact_ids
    sweep_trace = augmented["source_local_fact_sweep_trace"]
    assert sweep_trace["seed_fact_ids"] == ["raw_doc_e1_01"]
    assert sweep_trace["mode"] == "bounded_chain_candidate_bundle"
    assert "christianity" in sweep_trace["frontier"]


def test_ordinal_operator_changes_selected_episode():
    corpus = {
        "documents": [{
            "doc_id": "conversation:conv_ord",
            "episodes": [
                _episode("conv_ord_e0001", "Song about frames alpha.", source_id="conv_ord"),
                _episode("conv_ord_e0002", "Song about frames alpha with extra chorus frames frames.", source_id="conv_ord"),
            ],
        }]
    }
    lookup = {ep["episode_id"]: ep for doc in corpus["documents"] for ep in doc["episodes"]}
    bm25 = build_episode_bm25(corpus)
    result = choose_episode_ids_with_trace(
        "What is the 1st song about frames?",
        bm25,
        lookup,
        {"max_candidates": 10, "max_episodes_default": 1, "max_sources_per_family": 1},
    )
    assert result["selected_ids"] == ["conv_ord_e0001"]


def test_commonality_operator_prefers_shared_anchor_episode():
    corpus = {
        "documents": [{
            "doc_id": "conversation:conv_common",
            "episodes": [
                _episode("common_e1", "Alice joined pottery class.", source_id="conv_common"),
                _episode("common_e2", "Bob joined pottery class.", source_id="conv_common"),
                _episode("common_e3", "Alice and Bob both joined pottery class together.", source_id="conv_common"),
            ],
        }]
    }
    lookup = {ep["episode_id"]: ep for doc in corpus["documents"] for ep in doc["episodes"]}
    bm25 = build_episode_bm25(corpus)
    result = choose_episode_ids_with_trace(
        "What do Alice and Bob have in common?",
        bm25,
        lookup,
        {"max_candidates": 10, "max_episodes_default": 1, "max_sources_per_family": 1},
    )
    assert result["selected_ids"] == ["common_e3"]


def test_compare_diff_operator_surfaces_contrasting_pair():
    corpus = {
        "documents": [{
            "doc_id": "document:DOC-COMPARE",
            "episodes": [
                _episode(
                    "cmp_e1",
                    "Route 1 draft length was 14.1 km before approval.",
                    source_id="DOC-COMPARE",
                    source_type="document",
                    topic="route draft length",
                ),
                _episode(
                    "cmp_e2",
                    "Route 1 final approved length is 14.3 km.",
                    source_id="DOC-COMPARE",
                    source_type="document",
                    topic="route final length",
                ),
                _episode(
                    "cmp_e3",
                    "Route 9 signage checklist is complete.",
                    source_id="DOC-COMPARE",
                    source_type="document",
                    topic="signage checklist",
                ),
            ],
        }]
    }
    corpus["documents"][0]["episodes"][0]["currentness"] = "outdated"
    corpus["documents"][0]["episodes"][0]["state_label"] = "draft"
    corpus["documents"][0]["episodes"][1]["currentness"] = "current"
    corpus["documents"][0]["episodes"][1]["state_label"] = "approved"
    lookup = {ep["episode_id"]: ep for doc in corpus["documents"] for ep in doc["episodes"]}
    bm25 = build_episode_bm25(corpus)
    result = choose_episode_ids_with_trace(
        "What changed before and after Route 1 approval?",
        bm25,
        lookup,
        {"max_candidates": 10, "max_episodes_default": 2, "max_sources_per_family": 1},
    )
    assert result["selected_ids"][:2] == ["cmp_e1", "cmp_e2"]


def test_current_query_prefers_later_anchor_match():
    corpus = {
        "documents": [{
            "doc_id": "document:DOC-CURRENT",
            "episodes": [
                _episode(
                    "curr_e1",
                    "At km 2.3 the pipe cover depth was 780mm before rework.",
                    source_id="DOC-CURRENT",
                    source_type="document",
                    topic="initial measurement",
                ),
                _episode(
                    "curr_e2",
                    "At km 2.3 the pipe cover depth was 980mm after re-inspection.",
                    source_id="DOC-CURRENT",
                    source_type="document",
                    topic="reinspection",
                ),
            ],
        }]
    }
    lookup = {ep["episode_id"]: ep for doc in corpus["documents"] for ep in doc["episodes"]}
    bm25 = build_episode_bm25(corpus)
    result = choose_episode_ids_with_trace(
        "What is the current pipe cover depth at km 2.3?",
        bm25,
        lookup,
        {"max_candidates": 10, "max_episodes_default": 1, "max_sources_per_family": 1},
    )
    assert result["selected_ids"] == ["curr_e2"]


def test_local_anchor_operator_prefers_exact_anchor_episode():
    corpus = {
        "documents": [{
            "doc_id": "conversation:conv_anchor",
            "episodes": [
                _episode("anchor_e1", "I redeemed a coupon on coffee creamer yesterday.", source_id="conv_anchor"),
                _episode("anchor_e2", "I redeemed a coupon on coffee creamer at Target.", source_id="conv_anchor"),
            ],
        }]
    }
    lookup = {ep["episode_id"]: ep for doc in corpus["documents"] for ep in doc["episodes"]}
    bm25 = build_episode_bm25(corpus)
    result = choose_episode_ids_with_trace(
        "Which store did I redeem a coupon on coffee creamer at?",
        bm25,
        lookup,
        {"max_candidates": 10, "max_episodes_default": 1, "max_sources_per_family": 1},
    )
    assert result["selected_ids"] == ["anchor_e2"]


def test_list_set_operator_deduplicates_near_duplicate_episodes():
    corpus = {
        "documents": [{
            "doc_id": "conversation:conv_list",
            "episodes": [
                _episode("list_e1", "John played kickboxing and taekwondo.", source_id="conv_list"),
                _episode("list_e2", "John played kickboxing and taekwondo competitively.", source_id="conv_list"),
                _episode("list_e3", "John also trained in judo.", source_id="conv_list"),
            ],
        }]
    }
    lookup = {ep["episode_id"]: ep for doc in corpus["documents"] for ep in doc["episodes"]}
    bm25 = build_episode_bm25(corpus)
    result = choose_episode_ids_with_trace(
        "List all sports John trained in.",
        bm25,
        lookup,
        {"max_candidates": 10, "max_episodes_default": 2, "max_sources_per_family": 1},
    )
    assert result["selected_ids"] == ["list_e1", "list_e3"]


def test_list_set_operator_avoids_substring_false_positives():
    corpus = {
        "documents": [{
            "doc_id": "conversation:conv_substrings",
            "episodes": [
                _episode("sub_e1", "John shared a heartwarming story and said he started baking.", source_id="conv_substrings"),
                _episode("sub_e2", "John is doing martial arts kickboxing this weekend.", source_id="conv_substrings"),
                _episode("sub_e3", "John is off to do some martial arts taekwondo tonight.", source_id="conv_substrings"),
            ],
        }]
    }
    lookup = {ep["episode_id"]: ep for doc in corpus["documents"] for ep in doc["episodes"]}
    bm25 = build_episode_bm25(corpus)
    result = choose_episode_ids_with_trace(
        "What martial arts has John done?",
        bm25,
        lookup,
        {"max_candidates": 10, "max_episodes_default": 2, "max_sources_per_family": 1},
    )
    assert result["selected_ids"] == ["sub_e2", "sub_e3"]


def test_list_set_future_event_queries_prefer_committed_event_episodes_over_noise():
    corpus = {
        "documents": [{
            "doc_id": "conversation:conv_future_events",
            "episodes": [
                _episode(
                    "event_noise",
                    "Alice and Ben discussed hobbies and weekend weather.",
                    source_id="conv_future_events",
                ),
                _episode(
                    "event_workshop",
                    "Alice: The robotics workshop looks good.\n\n"
                    "Ben: Let's do it next Saturday.",
                    source_id="conv_future_events",
                ),
                _episode(
                    "event_cafe",
                    "Ben: The Harbor Cafe is open late.\n\n"
                    "Alice: We will meet at Harbor Cafe tomorrow.",
                    source_id="conv_future_events",
                ),
                _episode(
                    "event_concert",
                    "Alice and Ben are going to the jazz concert on 2024-06-01.",
                    source_id="conv_future_events",
                ),
            ],
        }]
    }
    lookup = {ep["episode_id"]: ep for doc in corpus["documents"] for ep in doc["episodes"]}
    bm25 = build_episode_bm25(corpus)
    result = choose_episode_ids_with_trace(
        "Which places or events have Alice and Ben planned for the future?",
        bm25,
        lookup,
        {"max_candidates": 10, "max_episodes_default": 3, "max_sources_per_family": 1},
    )
    assert set(result["selected_ids"]) == {"event_cafe", "event_workshop", "event_concert"}


def test_list_set_supporting_facts_prefer_recurring_item_facts():
    episode_lookup = {
        "ep_main": _episode(
            "ep_main",
            "Maria asked if it was a martial arts place.\n\n"
            "They offer yoga, kickboxing, and circuit training.\n\n"
            "John has done weight training so far too.",
            source_id="conv_list_support",
        ),
        "ep_support": _episode(
            "ep_support",
            "John thought the family road trip was fun.\n\n"
            "John is doing kickboxing and it gives him energy.",
            source_id="conv_list_support",
        ),
        "ep_extra": _episode(
            "ep_extra",
            "John is going to do taekwondo tonight.",
            source_id="conv_list_support",
        ),
    }
    facts_by_episode = {
        "ep_main": [
            {"id": "f_main_generic", "fact": "John goes to a yoga studio often.", "session": 1, "metadata": {"episode_id": "ep_main"}},
            {"id": "f_main_item", "fact": "They offer yoga, kickboxing, and circuit training.", "session": 1, "metadata": {"episode_id": "ep_main"}},
            {"id": "f_main_other", "fact": "John has done weight training so far too.", "session": 1, "metadata": {"episode_id": "ep_main"}},
        ],
        "ep_support": [
            {"id": "f_support_generic", "fact": "John thought the family road trip was fun.", "session": 1, "metadata": {"episode_id": "ep_support"}},
            {"id": "f_support_item", "fact": "John is doing kickboxing.", "session": 1, "metadata": {"episode_id": "ep_support"}},
        ],
        "ep_extra": [
            {"id": "f_extra_item", "fact": "John is going to do taekwondo.", "session": 1, "metadata": {"episode_id": "ep_extra"}},
        ],
    }
    facts = pick_supporting_facts(
        "What martial arts has John done?",
        ["ep_main"],
        facts_by_episode,
        episode_lookup=episode_lookup,
        fact_episode_ids=["ep_main", "ep_support", "ep_extra"],
        max_total=6,
        max_per_episode=1,
        allow_pseudo_facts=False,
    )
    rendered = [fact.get("fact", "") for fact in facts]
    assert "They offer yoga, kickboxing, and circuit training." in rendered
    assert "John is doing kickboxing." in rendered
    assert "John is going to do taekwondo." in rendered
    assert "John thought the family road trip was fun." not in rendered


def test_bounded_chain_operator_expands_same_source_chain():
    corpus = {
        "documents": [{
            "doc_id": "document:DOC-CHAIN",
            "episodes": [
                _episode(
                    "chain_e1",
                    "Bagratuni follows Christianity.",
                    source_id="DOC-CHAIN",
                    source_type="document",
                    topic="bagratuni religion",
                ),
                _episode(
                    "chain_e2",
                    "Christianity was founded in Taipei.",
                    source_id="DOC-CHAIN",
                    source_type="document",
                    topic="christianity origin",
                ),
                _episode(
                    "chain_e3",
                    "Jerusalem appears in other religious notes.",
                    source_id="DOC-CHAIN",
                    source_type="document",
                    topic="other notes",
                ),
            ],
        }]
    }
    lookup = {ep["episode_id"]: ep for doc in corpus["documents"] for ep in doc["episodes"]}
    bm25 = build_episode_bm25(corpus)
    result = choose_episode_ids_with_trace(
        "Where was the religion of Bagratuni founded?",
        bm25,
        lookup,
        {"max_candidates": 10, "max_episodes_default": 2, "max_sources_per_family": 1},
    )
    assert result["selected_ids"][:2] == ["chain_e1", "chain_e2"]


def test_bounded_chain_uses_bridge_terms_from_matched_support_line():
    corpus = {
        "documents": [{
            "doc_id": "document:DOC-BRIDGE",
            "episodes": [
                _episode(
                    "bridge_e1",
                    'Nate "Tiny" Archibald is a citizen of United States of America. William Waynflete speaks English.',
                    source_id="DOC-BRIDGE",
                    source_type="document",
                    topic="citizenship note",
                ),
                _episode(
                    "bridge_e2",
                    "The official language of United States of America is German.",
                    source_id="DOC-BRIDGE",
                    source_type="document",
                    topic="language note",
                ),
                _episode(
                    "bridge_e3",
                    "The official language of Uruguay is Spanish.",
                    source_id="DOC-BRIDGE",
                    source_type="document",
                    topic="other language note",
                ),
            ],
        }]
    }
    lookup = {ep["episode_id"]: ep for doc in corpus["documents"] for ep in doc["episodes"]}
    bm25 = build_episode_bm25(corpus)
    result = choose_episode_ids_with_trace(
        'In which language are the official documents written in the country of citizenship of Nate "Tiny" Archibald?',
        bm25,
        lookup,
        {"max_candidates": 10, "max_episodes_default": 2, "max_sources_per_family": 1},
    )
    assert result["selected_ids"][:2] == ["bridge_e1", "bridge_e2"]


def test_query_words_do_not_duplicate_entity_anchor_tokens():
    qf = extract_query_features("What martial arts has John done?")
    assert "John" in qf["entity_phrases"]
    assert "john" not in qf["words"]


def test_query_features_expand_origin_paraphrases():
    qf_city = extract_query_features(
        "Where did the religion associated with the Bagratuni Dynasty come into existence?"
    )
    assert normalize_term_token("founded") in qf_city["words"]

    qf_country = extract_query_features(
        "What is the country of origin of the sport played by Christian Abbiati?"
    )
    assert normalize_term_token("created") in qf_country["words"]


def test_list_set_head_tokens_do_not_overreward_category_echo():
    qf = extract_query_features("What martial arts has John done?")
    taekwondo = {
        "id": "f_taekwondo",
        "fact": "John has done the martial art taekwondo.",
        "session": 1,
        "metadata": {"episode_id": "ep1"},
    }
    arts_and_crafts = {
        "id": "f_arts",
        "fact": "John tried arts and crafts activities with his family.",
        "session": 2,
        "metadata": {"episode_id": "ep2"},
    }
    token_freq = Counter()
    for fact in (taekwondo, arts_and_crafts):
        token_freq.update(set(_fact_content_tokens(fact["fact"], qf)))
    taekwondo_score = _fact_rank_tuple(taekwondo, qf, token_freq, 4.0)[0]
    arts_score = _fact_rank_tuple(arts_and_crafts, qf, token_freq, 4.0)[0]
    assert taekwondo_score > arts_score


def test_bounded_chain_ignores_generic_relation_bridge_tokens():
    corpus = {
        "documents": [{
            "doc_id": "document:DOC-GENERIC-BRIDGE",
            "episodes": [
                _episode(
                    "generic_root",
                    "Bagratuni Dynasty is affiliated with the religion of Christianity.",
                    source_id="DOC-GENERIC-BRIDGE",
                    source_type="document",
                    topic="bagratuni religion",
                ),
                _episode(
                    "generic_noise",
                    "Another dynasty is affiliated with the religion of Catholic Church.",
                    source_id="DOC-GENERIC-BRIDGE",
                    source_type="document",
                    topic="other religion note",
                ),
                _episode(
                    "generic_target",
                    "The religion associated with Christianity came into existence in Taipei.",
                    source_id="DOC-GENERIC-BRIDGE",
                    source_type="document",
                    topic="christianity origin",
                ),
            ],
        }]
    }
    lookup = {ep["episode_id"]: ep for doc in corpus["documents"] for ep in doc["episodes"]}
    bm25 = build_episode_bm25(corpus)
    result = choose_episode_ids_with_trace(
        "Where did the religion associated with the Bagratuni Dynasty come into existence?",
        bm25,
        lookup,
        {"max_candidates": 10, "max_episodes_default": 2, "max_sources_per_family": 1},
    )
    assert result["selected_ids"][:2] == ["generic_root", "generic_target"]


def test_bounded_chain_prefers_new_relation_step_over_repeated_relation_noise():
    lookup = {
        "chain_root": _episode(
            "chain_root",
            "Bagratuni Dynasty is affiliated with the religion of Christianity.",
            source_id="DOC-CHAIN-STEP",
            source_type="document",
            topic="bagratuni religion",
        ),
        "chain_noise": _episode(
            "chain_noise",
            "Another ruler is affiliated with the religion of Christianity.",
            source_id="DOC-CHAIN-STEP",
            source_type="document",
            topic="other religion note",
        ),
        "chain_answer": _episode(
            "chain_answer",
            "Christianity was founded in the city of Taipei.",
            source_id="DOC-CHAIN-STEP",
            source_type="document",
            topic="christianity origin",
        ),
    }
    rows = [
        {"episode_id": "chain_root", "score": 8.6, "bm25_rank": 1, "source_id": "DOC-CHAIN-STEP", "source_type": "document", "topic_key": "bagratuni religion", "state_label": "session", "score_breakdown": {}},
        {"episode_id": "chain_noise", "score": 5.1, "bm25_rank": 2, "source_id": "DOC-CHAIN-STEP", "source_type": "document", "topic_key": "other religion note", "state_label": "session", "score_breakdown": {}},
        {"episode_id": "chain_answer", "score": 1.2, "bm25_rank": 3, "source_id": "DOC-CHAIN-STEP", "source_type": "document", "topic_key": "christianity origin", "state_label": "session", "score_breakdown": {}},
    ]
    qf = extract_query_features(
        "Where did the religion associated with the Bagratuni Dynasty come into existence?"
    )
    ordered = _apply_bounded_chain_expansion(
        rows,
        lookup,
        qf,
        get_runtime_tuning()["operators"],
        max_hops=2,
    )
    assert [row["episode_id"] for row in ordered[:2]] == ["chain_root", "chain_answer"]


def test_bounded_chain_ignores_stopword_frontier_noise():
    lookup = {
        "chain_root": _episode(
            "chain_root",
            "Bagratuni Dynasty is affiliated with the religion of Christianity.",
            source_id="DOC-CHAIN-STOPWORDS",
            source_type="document",
            topic="bagratuni religion",
        ),
        "chain_noise": _episode(
            "chain_noise",
            "Angelo Sodano is affiliated with the religion of Catholic Church.",
            source_id="DOC-CHAIN-STOPWORDS",
            source_type="document",
            topic="other religion note",
        ),
        "chain_answer": _episode(
            "chain_answer",
            "Christianity was founded in the city of Taipei.",
            source_id="DOC-CHAIN-STOPWORDS",
            source_type="document",
            topic="christianity origin",
        ),
    }
    rows = [
        {"episode_id": "chain_root", "score": 8.6, "bm25_rank": 1, "source_id": "DOC-CHAIN-STOPWORDS", "source_type": "document", "topic_key": "bagratuni religion", "state_label": "session", "score_breakdown": {}},
        {"episode_id": "chain_noise", "score": 5.1, "bm25_rank": 2, "source_id": "DOC-CHAIN-STOPWORDS", "source_type": "document", "topic_key": "other religion note", "state_label": "session", "score_breakdown": {}},
        {"episode_id": "chain_answer", "score": 1.2, "bm25_rank": 3, "source_id": "DOC-CHAIN-STOPWORDS", "source_type": "document", "topic_key": "christianity origin", "state_label": "session", "score_breakdown": {}},
    ]
    qf = extract_query_features(
        "Where did the religion associated with the Bagratuni Dynasty come into existence?"
    )
    ordered = _apply_bounded_chain_expansion(
        rows,
        lookup,
        qf,
        get_runtime_tuning()["operators"],
        max_hops=2,
    )
    assert [row["episode_id"] for row in ordered[:2]] == ["chain_root", "chain_answer"]


def test_bounded_chain_lookahead_prefers_path_with_downstream_resolver():
    lookup = {
        "path_root": _episode(
            "path_root",
            "Christian Abbiati plays the position of cornerback.",
            source_id="DOC-CHAIN-PATH",
            source_type="document",
            topic="player position",
        ),
        "path_noise": _episode(
            "path_noise",
            "cornerback is associated with the sport of American football.",
            source_id="DOC-CHAIN-PATH",
            source_type="document",
            topic="other sport note",
        ),
        "path_bridge": _episode(
            "path_bridge",
            "cornerback is associated with the sport of field hockey.",
            source_id="DOC-CHAIN-PATH",
            source_type="document",
            topic="bridge sport note",
        ),
        "path_answer": _episode(
            "path_answer",
            "field hockey was created in the country of Philippines.",
            source_id="DOC-CHAIN-PATH",
            source_type="document",
            topic="origin note",
        ),
    }
    rows = [
        {"episode_id": "path_root", "score": 8.5, "bm25_rank": 1, "source_id": "DOC-CHAIN-PATH", "source_type": "document", "topic_key": "player position", "state_label": "session", "score_breakdown": {}},
        {"episode_id": "path_noise", "score": 5.4, "bm25_rank": 2, "source_id": "DOC-CHAIN-PATH", "source_type": "document", "topic_key": "other sport note", "state_label": "session", "score_breakdown": {}},
        {"episode_id": "path_bridge", "score": 5.1, "bm25_rank": 3, "source_id": "DOC-CHAIN-PATH", "source_type": "document", "topic_key": "bridge sport note", "state_label": "session", "score_breakdown": {}},
        {"episode_id": "path_answer", "score": 4.8, "bm25_rank": 4, "source_id": "DOC-CHAIN-PATH", "source_type": "document", "topic_key": "origin note", "state_label": "session", "score_breakdown": {}},
    ]
    qf = extract_query_features(
        "What is the country of origin of the sport played by Christian Abbiati?"
    )
    ordered = _apply_bounded_chain_expansion(
        rows,
        lookup,
        qf,
        get_runtime_tuning()["operators"],
        max_hops=2,
    )
    assert [row["episode_id"] for row in ordered[:3]] == ["path_root", "path_bridge", "path_answer"]


def test_bounded_chain_support_sentence_score_knob_is_consumed():
    lookup = {
        "generic_root": _episode(
            "generic_root",
            "Bagratuni Dynasty is affiliated with the religion of Christianity.",
            source_id="DOC-GENERIC-BRIDGE",
            source_type="document",
            topic="bagratuni religion",
        ),
        "generic_noise": _episode(
            "generic_noise",
            "Another dynasty is affiliated with the religion of Catholic Church.",
            source_id="DOC-GENERIC-BRIDGE",
            source_type="document",
            topic="other religion note",
        ),
        "generic_target": _episode(
            "generic_target",
            "The religion associated with Christianity came into existence in Taipei.",
            source_id="DOC-GENERIC-BRIDGE",
            source_type="document",
            topic="christianity origin",
        ),
    }
    rows = [
        {"episode_id": "generic_root", "score": 8.6, "bm25_rank": 2, "source_id": "DOC-GENERIC-BRIDGE", "source_type": "document", "topic_key": "bagratuni religion", "state_label": "session", "score_breakdown": {}},
        {"episode_id": "generic_noise", "score": 6.0, "bm25_rank": 3, "source_id": "DOC-GENERIC-BRIDGE", "source_type": "document", "topic_key": "other religion note", "state_label": "session", "score_breakdown": {}},
        {"episode_id": "generic_target", "score": 1.0, "bm25_rank": 1, "source_id": "DOC-GENERIC-BRIDGE", "source_type": "document", "topic_key": "christianity origin", "state_label": "session", "score_breakdown": {}},
    ]
    qf = extract_query_features(
        "Where did the religion associated with the Bagratuni Dynasty come into existence?"
    )
    base = _apply_bounded_chain_expansion(
        rows,
        lookup,
        qf,
        {
            "bounded_chain_root_support_sentence_budget": 1,
            "bounded_chain_support_sentence_budget": 2,
            "bounded_chain_same_source_bonus": 0.5,
            "bounded_chain_unresolved_overlap_weight": 0.0,
            "bounded_chain_novelty_weight": 0.0,
            "bounded_chain_bm25_carry_weight": 0.0,
            "bounded_chain_relation_novelty_weight": 0.0,
            "bounded_chain_relation_repeat_penalty": 0.0,
            "bounded_chain_lookahead_weight": 0.0,
            "bounded_chain_support_sentence_score_weight": 0.0,
            "bounded_chain_sentence_entity_match_weight": 6.0,
            "bounded_chain_sentence_word_match_weight": 1.0,
            "bounded_chain_sentence_number_match_weight": 1.5,
            "bounded_chain_sentence_frontier_overlap_weight": 2.0,
        },
        max_hops=2,
    )
    boosted = _apply_bounded_chain_expansion(
        rows,
        lookup,
        qf,
        {
            "bounded_chain_root_support_sentence_budget": 1,
            "bounded_chain_support_sentence_budget": 2,
            "bounded_chain_same_source_bonus": 0.5,
            "bounded_chain_unresolved_overlap_weight": 0.0,
            "bounded_chain_novelty_weight": 0.0,
            "bounded_chain_bm25_carry_weight": 0.0,
            "bounded_chain_relation_novelty_weight": 0.0,
            "bounded_chain_relation_repeat_penalty": 0.0,
            "bounded_chain_lookahead_weight": 0.0,
            "bounded_chain_support_sentence_score_weight": 1.0,
            "bounded_chain_sentence_entity_match_weight": 6.0,
            "bounded_chain_sentence_word_match_weight": 1.0,
            "bounded_chain_sentence_number_match_weight": 1.5,
            "bounded_chain_sentence_frontier_overlap_weight": 2.0,
        },
        max_hops=2,
    )
    assert [row["episode_id"] for row in base][:2] == ["generic_root", "generic_noise"]
    assert [row["episode_id"] for row in boosted][:2] == ["generic_root", "generic_target"]


def test_bounded_chain_same_source_bonus_knob_is_consumed():
    lookup = {
        "chain_root": _episode(
            "chain_root",
            'Nate "Tiny" Archibald is a citizen of United States of America.',
            source_id="DOC-CHAIN-BONUS",
            source_type="document",
            topic="citizenship note",
        ),
        "chain_same": _episode(
            "chain_same",
            "United States official language is English.",
            source_id="DOC-CHAIN-BONUS",
            source_type="document",
            topic="language note",
        ),
        "chain_cross": _episode(
            "chain_cross",
            "Official documents in the United States of America are written in English.",
            source_id="DOC-OTHER",
            source_type="document",
            topic="documents note",
        ),
    }
    rows = [
        {"episode_id": "chain_root", "score": 12.0, "bm25_rank": 1},
        {"episode_id": "chain_same", "score": 1.0, "bm25_rank": 3},
        {"episode_id": "chain_cross", "score": 6.0, "bm25_rank": 2},
    ]
    qf = extract_query_features(
        'In which language are the official documents written in the country of citizenship of Nate "Tiny" Archibald?'
    )

    def _operator_tuning(bonus: float):
        base = get_runtime_tuning()["operators"]
        merged = dict(base)
        merged.update({
            "bounded_chain_root_support_sentence_budget": 1,
            "bounded_chain_support_sentence_budget": 2,
            "bounded_chain_same_source_bonus": bonus,
            "bounded_chain_unresolved_overlap_weight": 1.5,
            "bounded_chain_novelty_weight": 0.75,
            "bounded_chain_bm25_carry_weight": 0.01,
            "bounded_chain_relation_novelty_weight": 0.0,
            "bounded_chain_relation_repeat_penalty": 0.0,
            "bounded_chain_lookahead_weight": 0.0,
            "bounded_chain_sentence_entity_match_weight": 6.0,
            "bounded_chain_sentence_word_match_weight": 1.0,
            "bounded_chain_sentence_number_match_weight": 1.5,
            "bounded_chain_sentence_frontier_overlap_weight": 2.0,
        })
        return merged

    without_bonus = _apply_bounded_chain_expansion(
        rows,
        lookup,
        qf,
        _operator_tuning(0.0),
        max_hops=1,
    )
    assert [row["episode_id"] for row in without_bonus[:2]] == ["chain_root", "chain_cross"]

    with_bonus = _apply_bounded_chain_expansion(
        rows,
        lookup,
        qf,
        _operator_tuning(8.0),
        max_hops=1,
    )
    assert [row["episode_id"] for row in with_bonus[:2]] == ["chain_root", "chain_same"]


def test_bounded_chain_sentence_weight_knob_is_consumed():
    episode = _episode(
        "chain_sent",
        "Christianity was founded in Taipei. Unrelated appendix language note.",
        source_id="DOC-CHAIN",
        source_type="document",
        topic="christianity origin",
    )
    qf = {
        "entity_phrases": ["Christianity"],
        "entity_phrase_tokens": {"christianity": {"christianity"}},
        "words": {"founded", "city"},
        "numbers": set(),
    }

    low_tokens = _chain_tokens_with_frontier(
        episode,
        qf,
        {
            "bounded_chain_sentence_entity_match_weight": 0.0,
            "bounded_chain_sentence_word_match_weight": 0.0,
            "bounded_chain_sentence_number_match_weight": 0.0,
            "bounded_chain_sentence_frontier_overlap_weight": 0.0,
        },
        frontier=None,
        sentence_budget=1,
    )
    high_tokens = _chain_tokens_with_frontier(
        episode,
        qf,
        {
            "bounded_chain_sentence_entity_match_weight": 6.0,
            "bounded_chain_sentence_word_match_weight": 1.0,
            "bounded_chain_sentence_number_match_weight": 1.5,
            "bounded_chain_sentence_frontier_overlap_weight": 2.0,
        },
        frontier=None,
        sentence_budget=1,
    )

    assert "taipei" not in low_tokens
    assert "taipei" in high_tokens


def test_bounded_chain_sentence_unresolved_weight_knob_is_consumed():
    episode = _episode(
        "chain_unresolved",
        (
            "Christianity is affiliated with the religion of Catholic Church. "
            "Christianity was founded in the city of Taipei."
        ),
        source_id="DOC-CHAIN-UNRESOLVED",
        source_type="document",
        topic="christianity note",
    )
    qf = {
        "entity_phrases": ["Christianity"],
        "entity_phrase_tokens": {"christianity": {"christianity"}},
        "words": {"religion", "found"},
        "numbers": set(),
    }

    low_tokens = _chain_tokens_with_frontier(
        episode,
        qf,
        {
            "bounded_chain_sentence_entity_match_weight": 6.0,
            "bounded_chain_sentence_word_match_weight": 1.0,
            "bounded_chain_sentence_unresolved_weight": 0.0,
            "bounded_chain_sentence_number_match_weight": 1.5,
            "bounded_chain_sentence_frontier_overlap_weight": 2.0,
        },
        frontier={"christianity"},
        unresolved_words={"found"},
        sentence_budget=1,
    )
    high_tokens = _chain_tokens_with_frontier(
        episode,
        qf,
        {
            "bounded_chain_sentence_entity_match_weight": 6.0,
            "bounded_chain_sentence_word_match_weight": 1.0,
            "bounded_chain_sentence_unresolved_weight": 2.0,
            "bounded_chain_sentence_number_match_weight": 1.5,
            "bounded_chain_sentence_frontier_overlap_weight": 2.0,
        },
        frontier={"christianity"},
        unresolved_words={"found"},
        sentence_budget=1,
    )

    assert "taipei" not in low_tokens
    assert "taipei" in high_tokens


def test_pick_supporting_facts_bounded_chain_adds_downstream_fact():
    question = "Where did the religion associated with the Bagratuni Dynasty come into existence?"
    facts_by_episode = {
        "root": [
            {
                "id": "root_f1",
                "fact": "Bagratuni Dynasty is affiliated with the religion of Christianity.",
                "session": 1,
                "metadata": {"episode_id": "root"},
            }
        ],
        "target": [
            {
                "id": "target_f1",
                "fact": "Christianity was founded in the city of Taipei.",
                "session": 2,
                "metadata": {"episode_id": "target"},
            }
        ],
    }
    supporting = pick_supporting_facts(
        question,
        ["root"],
        facts_by_episode,
        fact_episode_ids=["root", "target"],
        max_total=3,
        max_per_episode=1,
        allow_pseudo_facts=False,
        bounded_chain_fact_bonus=2.0,
        query_specificity_bonus=4.0,
    )
    texts = [fact["fact"] for fact in supporting]
    assert any("Bagratuni Dynasty" in text for text in texts)
    assert any("Taipei" in text for text in texts)


def test_pick_supporting_facts_bounded_chain_multihop_across_episodes():
    question = "What is the country of origin of the sport played by Christian Abbiati?"
    facts_by_episode = {
        "root": [
            {
                "id": "root_f1",
                "fact": "Christian Abbiati plays the position of cornerback.",
                "session": 1,
                "metadata": {"episode_id": "root"},
            }
        ],
        "bridge": [
            {
                "id": "bridge_f1",
                "fact": "cornerback is associated with the sport of field hockey.",
                "session": 2,
                "metadata": {"episode_id": "bridge"},
            }
        ],
        "target": [
            {
                "id": "target_f1",
                "fact": "field hockey was created in the country of Philippines.",
                "session": 3,
                "metadata": {"episode_id": "target"},
            }
        ],
    }
    supporting = pick_supporting_facts(
        question,
        ["root"],
        facts_by_episode,
        fact_episode_ids=["root", "bridge", "target"],
        max_total=4,
        max_per_episode=1,
        allow_pseudo_facts=False,
        bounded_chain_fact_bonus=2.0,
        query_specificity_bonus=4.0,
    )
    texts = [fact["fact"] for fact in supporting]
    assert any("Christian Abbiati" in text for text in texts)
    assert any("field hockey" in text and "cornerback" in text for text in texts)
    assert any("Philippines" in text for text in texts)


def test_bounded_chain_support_fact_max_extra_knob_is_consumed(monkeypatch):
    question = "Where did the religion associated with the Bagratuni Dynasty come into existence?"
    facts_by_episode = {
        "root": [
            {
                "id": "root_f1",
                "fact": "Bagratuni Dynasty is affiliated with the religion of Christianity.",
                "session": 1,
                "metadata": {"episode_id": "root"},
            }
        ],
        "target": [
            {
                "id": "target_f1",
                "fact": "Bagratuni Dynasty is affiliated with the religion of Christianity.",
                "session": 2,
                "metadata": {"episode_id": "target"},
            },
            {
                "id": "target_f2",
                "fact": "Christianity was founded in the city of Taipei.",
                "session": 2,
                "metadata": {"episode_id": "target"},
            }
        ],
    }

    base_tuning = get_runtime_tuning()["operators"]

    def _patched_get_tuning_section(name, *keys):
        if name != "operators":
            raise AssertionError((name, keys))
        merged = dict(base_tuning)
        merged["bounded_chain_support_fact_seed_count"] = 1
        merged["bounded_chain_support_fact_max_extra"] = 0
        return merged

    monkeypatch.setattr("src.episode_packet.get_tuning_section", _patched_get_tuning_section)
    without_extra = pick_supporting_facts(
        question,
        ["root"],
        facts_by_episode,
        fact_episode_ids=["root", "target"],
        max_total=4,
        max_per_episode=1,
        allow_pseudo_facts=False,
        bounded_chain_fact_bonus=2.0,
        query_specificity_bonus=4.0,
    )
    assert not any("Taipei" in fact["fact"] for fact in without_extra)

    def _patched_get_tuning_section_enabled(name, *keys):
        if name != "operators":
            raise AssertionError((name, keys))
        merged = dict(base_tuning)
        merged["bounded_chain_support_fact_seed_count"] = 1
        merged["bounded_chain_support_fact_max_extra"] = 2
        return merged

    monkeypatch.setattr("src.episode_packet.get_tuning_section", _patched_get_tuning_section_enabled)
    with_extra = pick_supporting_facts(
        question,
        ["root"],
        facts_by_episode,
        fact_episode_ids=["root", "target"],
        max_total=4,
        max_per_episode=1,
        allow_pseudo_facts=False,
        bounded_chain_fact_bonus=2.0,
        query_specificity_bonus=4.0,
    )
    assert any("Taipei" in fact["fact"] for fact in with_extra)


def test_bounded_chain_prefers_resolver_fact_with_unresolved_query_words():
    facts_by_episode = {
        "seed": [
            {
                "id": "seed_f1",
                "fact": "Bagratuni Dynasty is affiliated with the religion of Christianity.",
                "session": 1,
                "metadata": {"episode_id": "seed"},
            }
        ],
        "noise": [
            {
                "id": "noise_f1",
                "fact": "Another ruler is affiliated with the religion of Christianity.",
                "session": 2,
                "metadata": {"episode_id": "noise"},
            }
        ],
        "answer": [
            {
                "id": "answer_f1",
                "fact": "Christianity was founded in the city of Taipei.",
                "session": 3,
                "metadata": {"episode_id": "answer"},
            }
        ],
    }
    supporting = pick_supporting_facts(
        "Where did the religion associated with the Bagratuni Dynasty come into existence?",
        ["seed"],
        facts_by_episode,
        fact_episode_ids=["seed", "noise", "answer"],
        max_total=3,
        max_per_episode=1,
        allow_pseudo_facts=False,
        bounded_chain_fact_bonus=2.0,
        query_specificity_bonus=4.0,
    )
    texts = [fact["fact"] for fact in supporting[:2]]
    assert texts == [
        "Bagratuni Dynasty is affiliated with the religion of Christianity.",
        "Christianity was founded in the city of Taipei.",
    ]


def test_bounded_chain_prefers_country_of_origin_resolver_fact():
    facts_by_episode = {
        "seed": [
            {
                "id": "seed_f1",
                "fact": "Christian Abbiati plays the position of cornerback.",
                "session": 1,
                "metadata": {"episode_id": "seed"},
            }
        ],
        "bridge": [
            {
                "id": "bridge_f1",
                "fact": "cornerback is associated with the sport of field hockey.",
                "session": 2,
                "metadata": {"episode_id": "bridge"},
            }
        ],
        "noise": [
            {
                "id": "noise_f1",
                "fact": "cornerback is associated with the sport of American football.",
                "session": 3,
                "metadata": {"episode_id": "noise"},
            }
        ],
        "answer": [
            {
                "id": "answer_f1",
                "fact": "field hockey was created in the country of Philippines.",
                "session": 4,
                "metadata": {"episode_id": "answer"},
            }
        ],
    }
    supporting = pick_supporting_facts(
        "What is the country of origin of the sport played by Christian Abbiati?",
        ["seed"],
        facts_by_episode,
        fact_episode_ids=["seed", "bridge", "noise", "answer"],
        max_total=4,
        max_per_episode=1,
        allow_pseudo_facts=False,
        bounded_chain_fact_bonus=2.0,
        query_specificity_bonus=4.0,
    )
    texts = [fact["fact"] for fact in supporting[:3]]
    assert texts == [
        "Christian Abbiati plays the position of cornerback.",
        "cornerback is associated with the sport of field hockey.",
        "field hockey was created in the country of Philippines.",
    ]


def test_operator_selection_knobs_are_consumed(monkeypatch):
    corpus = {
        "documents": [{
            "doc_id": "conversation:conv_knobs",
            "episodes": [
                _episode("ord_e0001", "Song about frames alpha.", source_id="conv_knobs"),
                _episode("ord_e0002", "Song about frames alpha with extra chorus frames frames.", source_id="conv_knobs"),
                _episode("cmp_e1", "Route 1 draft length was 14.1 km before approval.", source_id="conv_knobs", topic="route draft"),
                _episode("cmp_e2", "Route 1 final approved length is 14.3 km.", source_id="conv_knobs", topic="route final"),
                _episode("list_e1", "John played kickboxing and taekwondo.", source_id="conv_knobs"),
                _episode("list_e2", "John played kickboxing and taekwondo competitively.", source_id="conv_knobs"),
                _episode("list_e3", "John also trained in judo.", source_id="conv_knobs"),
                _episode("chain_e1", "Bagratuni follows Christianity.", source_id="conv_knobs", source_type="document"),
                _episode("chain_e2", "Christianity was founded in Taipei.", source_id="conv_knobs", source_type="document"),
                _episode("chain_e3", "Christianity appears again in a separate Taipei historical note.", source_id="conv_knobs", source_type="document"),
            ],
        }]
    }
    corpus["documents"][0]["episodes"][2]["currentness"] = "outdated"
    corpus["documents"][0]["episodes"][2]["state_label"] = "draft"
    corpus["documents"][0]["episodes"][3]["currentness"] = "current"
    corpus["documents"][0]["episodes"][3]["state_label"] = "approved"
    lookup = {ep["episode_id"]: ep for doc in corpus["documents"] for ep in doc["episodes"]}
    bm25 = build_episode_bm25(corpus)
    compare_corpus = {
        "documents": [{
            "doc_id": "document:DOC-CMP-KNOB",
            "episodes": [
                _episode("cmp_noise1", "Route 1 before and after approval summary memo.", source_id="DOC-CMP-KNOB", source_type="document"),
                _episode("cmp_noise2", "Route 1 before and after approval checklist.", source_id="DOC-CMP-KNOB", source_type="document"),
                _episode("cmp_chain1", "Route 1 draft length was 14.1 km before approval.", source_id="DOC-CMP-KNOB", source_type="document"),
                _episode("cmp_chain2", "Route 1 final approved length is 14.3 km.", source_id="DOC-CMP-KNOB", source_type="document"),
            ],
        }]
    }
    compare_corpus["documents"][0]["episodes"][0]["state_label"] = "session"
    compare_corpus["documents"][0]["episodes"][0]["currentness"] = "unknown"
    compare_corpus["documents"][0]["episodes"][1]["state_label"] = "session"
    compare_corpus["documents"][0]["episodes"][1]["currentness"] = "unknown"
    compare_corpus["documents"][0]["episodes"][2]["state_label"] = "draft"
    compare_corpus["documents"][0]["episodes"][2]["currentness"] = "outdated"
    compare_corpus["documents"][0]["episodes"][3]["state_label"] = "approved"
    compare_corpus["documents"][0]["episodes"][3]["currentness"] = "current"
    compare_lookup = {
        ep["episode_id"]: ep
        for doc in compare_corpus["documents"]
        for ep in doc["episodes"]
    }
    compare_bm25 = build_episode_bm25(compare_corpus)

    def _patched_get_tuning_section(name, *keys):
        if name == "retrieval":
            return {
                "selector": {
                    "max_candidates": 80,
                    "max_episodes_default": 3,
                    "max_episodes_per_family": 3,
                    "max_sources_per_family": 2,
                    "late_fusion_per_family": 3,
                    "rrf_k": 60,
                    "word_overlap_bonus": 0.45,
                    "number_overlap_bonus": 1.4,
                    "chainage_overlap_bonus": 0.0,
                    "identifier_overlap_bonus": 8.0,
                    "entity_phrase_bonus": 2.0,
                    "step_bonus": 3.5,
                    "currentness_bonus": 0.8,
                    "generic_penalty": 1.2,
                    "mega_penalty": 8.0,
                    "supporting_facts_per_episode": 3,
                    "supporting_facts_total": 10,
                    "budget": 8000,
                    "snippet_mode": False,
                }
            }["selector"]
        if name == "operators":
            return {
                "ordinal_candidate_budget": 1,
                "compare_alignment_budget": 1,
                "list_set_dedup_overlap": 1.1,
                "enable_snippet_for_ordinal": True,
                "enable_snippet_for_local_anchor": True,
                "local_anchor_window_chars": 1200,
                "structural_query_supporting_facts_total": 12,
                "structural_query_supporting_facts_per_episode": 4,
                "bounded_chain_max_hops": 1,
            }
        if name == "telemetry":
            return {"max_family_candidates": 8}
        raise AssertionError((name, keys))

    monkeypatch.setattr("src.episode_retrieval.get_tuning_section", _patched_get_tuning_section)

    ordinal_tight = choose_episode_ids_with_trace(
        "What is the 1st song about frames?",
        bm25,
        lookup,
        {"max_candidates": 10, "max_episodes_default": 1, "max_sources_per_family": 1},
    )
    compare_tight = choose_episode_ids_with_trace(
        "What changed before and after Route 1 approval?",
        compare_bm25,
        compare_lookup,
        {"max_candidates": 10, "max_episodes_default": 2, "max_sources_per_family": 1},
    )
    list_loose = choose_episode_ids_with_trace(
        "List all sports John trained in.",
        bm25,
        lookup,
        {"max_candidates": 10, "max_episodes_default": 2, "max_sources_per_family": 1},
    )

    def _patched_get_tuning_section_wide(name, *keys):
        if name == "retrieval":
            return {
                "selector": {
                    "max_candidates": 80,
                    "max_episodes_default": 3,
                    "max_episodes_per_family": 3,
                    "max_sources_per_family": 2,
                    "late_fusion_per_family": 3,
                    "rrf_k": 60,
                    "word_overlap_bonus": 0.45,
                    "number_overlap_bonus": 1.4,
                    "chainage_overlap_bonus": 0.0,
                    "identifier_overlap_bonus": 8.0,
                    "entity_phrase_bonus": 2.0,
                    "step_bonus": 3.5,
                    "currentness_bonus": 0.8,
                    "generic_penalty": 1.2,
                    "mega_penalty": 8.0,
                    "supporting_facts_per_episode": 3,
                    "supporting_facts_total": 10,
                    "budget": 8000,
                    "snippet_mode": False,
                }
            }["selector"]
        if name == "operators":
            return {
                "ordinal_candidate_budget": 4,
                "compare_alignment_budget": 4,
                "list_set_dedup_overlap": 0.8,
                "enable_snippet_for_ordinal": True,
                "enable_snippet_for_local_anchor": True,
                "local_anchor_window_chars": 1200,
                "structural_query_supporting_facts_total": 12,
                "structural_query_supporting_facts_per_episode": 4,
                "bounded_chain_max_hops": 2,
            }
        if name == "telemetry":
            return {"max_family_candidates": 8}
        raise AssertionError((name, keys))

    monkeypatch.setattr("src.episode_retrieval.get_tuning_section", _patched_get_tuning_section_wide)

    ordinal_wide = choose_episode_ids_with_trace(
        "What is the 1st song about frames?",
        bm25,
        lookup,
        {"max_candidates": 10, "max_episodes_default": 1, "max_sources_per_family": 1},
    )
    compare_wide = choose_episode_ids_with_trace(
        "What changed before and after Route 1 approval?",
        compare_bm25,
        compare_lookup,
        {"max_candidates": 10, "max_episodes_default": 2, "max_sources_per_family": 1},
    )
    list_deduped = choose_episode_ids_with_trace(
        "List all sports John trained in.",
        bm25,
        lookup,
        {"max_candidates": 10, "max_episodes_default": 2, "max_sources_per_family": 1},
    )

    assert ordinal_tight["selected_ids"] != ordinal_wide["selected_ids"]
    assert compare_tight["selected_ids"] != compare_wide["selected_ids"]
    assert list_loose["selected_ids"] != list_deduped["selected_ids"]


def test_late_fusion_uses_per_family_rank_not_one_giant_pool():
    episode_lookup = {
        "conv_e1": _episode("conv_e1", "Alice has 5 cats.", source_id="conv_1"),
        "doc_e1": _episode(
            "doc_e1",
            "Contract section 5 sets the delivery deadline.",
            source_id="doc_1",
            source_type="document",
        ),
    }
    selected, scored = select_episode_ids_late_fusion(
        "How many cats does Alice have?",
        [
            {"family": "conversation", "selected_ids": ["conv_e1"], "scored": [("conv_e1", 9.0)]},
            {"family": "document", "selected_ids": ["doc_e1"], "scored": [("doc_e1", 2.0)]},
        ],
        episode_lookup,
        {"max_episodes_default": 2, "late_fusion_per_family": 2},
    )
    assert selected[0] == "conv_e1"
    assert scored[0][0] == "conv_e1"


def test_supporting_facts_include_pseudo_fact_from_raw_text():
    episode_lookup = {
        "ep1": _episode(
            "ep1",
            "User redeemed a $5 coupon on coffee creamer.\n\nThe store was Target.",
        )
    }
    facts_by_episode = {
        "ep1": [{
            "id": "f1",
            "fact": "User redeemed a $5 coupon on coffee creamer.",
            "session": 1,
            "metadata": {"episode_id": "ep1"},
        }]
    }
    facts = pick_supporting_facts(
        "Where did I redeem a $5 coupon on coffee creamer?",
        ["ep1"],
        facts_by_episode,
        episode_lookup=episode_lookup,
        max_total=5,
        max_per_episode=5,
        allow_pseudo_facts=True,
    )
    assert any("Target" in fact.get("fact", "") for fact in facts)


def test_supporting_facts_include_query_ranked_pseudo_fact_from_support_episode():
    episode_lookup = {
        "ep_selected": _episode("ep_selected", "The related position is goaltender."),
        "ep_support": _episode(
            "ep_support",
            "Arsenal F.C. Academy is associated with the sport of baseball.\n\n"
            "goaltender is associated with the sport of pesäpallo.\n\n"
            "Another unrelated sports note.",
        ),
    }
    facts_by_episode = {
        "ep_selected": [
            {"id": "f1", "fact": "The related position is goaltender.", "session": 1, "metadata": {"episode_id": "ep_selected"}},
        ],
        "ep_support": [
            {"id": "f2", "fact": "Arsenal F.C. Academy is associated with the sport of baseball.", "session": 1, "metadata": {"episode_id": "ep_support"}},
        ],
    }
    facts = pick_supporting_facts(
        "Which sport is goaltender associated with?",
        ["ep_selected"],
        facts_by_episode,
        episode_lookup=episode_lookup,
        fact_episode_ids=["ep_selected", "ep_support"],
        max_total=5,
        max_per_episode=3,
        allow_pseudo_facts=True,
        query_specificity_bonus=4.0,
    )
    assert any("pesäpallo" in fact.get("fact", "") for fact in facts)


def test_local_anchor_supporting_facts_expand_around_anchor_fact():
    episode_lookup = {
        "ep1": _episode(
            "ep1",
            "Cartwheel from Target was helpful.\n\nI redeemed a $5 coupon on coffee creamer.\n\nMany retailers like Target send exclusive coupon emails.\n\nI shop at Target frequently.",
        )
    }
    facts_by_episode = {
        "ep1": [
            {"id": "s88_f_19", "fact": "User has been using the Cartwheel app from Target.", "session": 88, "metadata": {"episode_id": "ep1"}},
            {"id": "s88_f_30", "fact": "User redeemed a $5 coupon on coffee creamer last Sunday.", "session": 88, "metadata": {"episode_id": "ep1"}},
            {"id": "s88_f_36", "fact": "Many retailers, like Target, send exclusive coupons and promotions to their email subscribers.", "session": 88, "metadata": {"episode_id": "ep1"}},
            {"id": "s88_f_41", "fact": "User shops at Target pretty frequently, maybe every other week.", "session": 88, "metadata": {"episode_id": "ep1"}},
        ]
    }
    facts = pick_supporting_facts(
        "Where did I redeem a $5 coupon on coffee creamer?",
        ["ep1"],
        facts_by_episode,
        episode_lookup=episode_lookup,
        max_total=4,
        max_per_episode=4,
        allow_pseudo_facts=False,
        local_anchor_fact_radius=12,
    )
    rendered = [fact.get("fact", "") for fact in facts]
    assert any("Target" in fact for fact in rendered)


def test_supporting_fact_specificity_prefers_anchor_fact_over_generic_relation_scaffold():
    facts_by_episode = {
        "ep_anchor_1": [
            {"id": "f1", "fact": "goaltender is associated with the sport of ice hockey.", "session": 1, "metadata": {"episode_id": "ep_anchor_1"}},
        ],
        "ep_noise": [
            {"id": "f2", "fact": "S.C. Salgueiros is associated with the sport of association football.", "session": 1, "metadata": {"episode_id": "ep_noise"}},
            {"id": "f3", "fact": "Arsenal F.C. Academy is associated with the sport of baseball.", "session": 1, "metadata": {"episode_id": "ep_noise"}},
        ],
        "ep_anchor_2": [
            {"id": "f4", "fact": "goaltender is associated with the sport of pesäpallo.", "session": 1, "metadata": {"episode_id": "ep_anchor_2"}},
        ],
    }
    facts = pick_supporting_facts(
        "Which sport is goaltender associated with?",
        ["ep_anchor_1"],
        facts_by_episode,
        fact_episode_ids=["ep_anchor_1", "ep_noise", "ep_anchor_2"],
        max_total=3,
        max_per_episode=2,
        query_specificity_bonus=4.0,
    )
    rendered = [fact.get("fact", "") for fact in facts[:2]]
    assert rendered == [
        "goaltender is associated with the sport of ice hockey.",
        "goaltender is associated with the sport of pesäpallo.",
    ]


def test_list_set_prefers_compact_item_facts_over_generic_same_entity_facts():
    facts_by_episode = {
        "ep_a": [
            {"id": "f_road", "fact": "John got back from a family road trip yesterday.", "session": 1, "metadata": {"episode_id": "ep_a"}},
            {"id": "f_kick", "fact": "John is doing kickboxing.", "session": 1, "metadata": {"episode_id": "ep_a"}},
            {"id": "f_leaders", "fact": "John will chat with local leaders and organizations, get support, and gather ideas for his next move.", "session": 1, "metadata": {"episode_id": "ep_a"}},
        ],
        "ep_b": [
            {"id": "f_tkd", "fact": "John is going to do taekwondo.", "session": 2, "metadata": {"episode_id": "ep_b"}},
        ],
    }
    facts = pick_supporting_facts(
        "What martial arts has John done?",
        ["ep_a"],
        facts_by_episode,
        fact_episode_ids=["ep_a", "ep_b"],
        max_total=4,
        max_per_episode=2,
        allow_pseudo_facts=False,
        query_specificity_bonus=4.0,
    )
    rendered = [fact.get("fact", "") for fact in facts[:2]]
    assert rendered == [
        "John is doing kickboxing.",
        "John is going to do taekwondo.",
    ]


def test_list_set_compact_item_bonus_knob_is_consumed(monkeypatch):
    facts_by_episode = {
        "ep_a": [
            {
                "id": "f_verbose",
                "fact": "John has been going to a martial arts gym for kickboxing practice.",
                "session": 1,
                "metadata": {"episode_id": "ep_a"},
            },
            {"id": "f_kick", "fact": "John is doing kickboxing.", "session": 1, "metadata": {"episode_id": "ep_a"}},
        ],
    }

    def _with_bonus(weight: float):
        ops = dict(get_runtime_tuning()["operators"])
        ops["list_set_compact_item_bonus"] = weight
        ops["list_set_item_recurrence_bonus"] = 0.0
        ops["list_set_enumeration_bonus"] = 0.0
        return ops

    monkeypatch.setattr(
        "src.episode_packet.get_tuning_section",
        lambda section: _with_bonus(0.0) if section == "operators" else get_runtime_tuning()[section],
    )
    without_bonus = pick_supporting_facts(
        "What martial arts has John done?",
        ["ep_a"],
        facts_by_episode,
        fact_episode_ids=["ep_a"],
        max_total=2,
        max_per_episode=1,
        allow_pseudo_facts=False,
        query_specificity_bonus=4.0,
    )

    monkeypatch.setattr(
        "src.episode_packet.get_tuning_section",
        lambda section: _with_bonus(3.0) if section == "operators" else get_runtime_tuning()[section],
    )
    with_bonus = pick_supporting_facts(
        "What martial arts has John done?",
        ["ep_a"],
        facts_by_episode,
        fact_episode_ids=["ep_a"],
        max_total=2,
        max_per_episode=1,
        allow_pseudo_facts=False,
        query_specificity_bonus=4.0,
    )

    assert without_bonus[0]["fact"] == "John has been going to a martial arts gym for kickboxing practice."
    assert with_bonus[0]["fact"] == "John is doing kickboxing."


def test_list_set_future_event_queries_prefer_committed_items_over_generic_noise():
    facts_by_episode = {
        "ep_workshop": [
            {"id": "f_workshop_1", "fact": "Alice: The robotics workshop looks good.", "session": 1, "metadata": {"episode_id": "ep_workshop"}},
            {"id": "f_workshop_2", "fact": "Ben: Let's do it next Saturday.", "session": 1, "metadata": {"episode_id": "ep_workshop"}},
        ],
        "ep_cafe": [
            {"id": "f_cafe_1", "fact": "Ben: Harbor Cafe stays open late.", "session": 2, "metadata": {"episode_id": "ep_cafe"}},
            {"id": "f_cafe_2", "fact": "Alice: We will meet at Harbor Cafe tomorrow.", "session": 2, "metadata": {"episode_id": "ep_cafe"}},
        ],
        "ep_concert": [
            {"id": "f_concert_1", "fact": "Alice and Ben are going to the jazz concert on 2024-06-01.", "session": 3, "metadata": {"episode_id": "ep_concert"}},
        ],
        "ep_noise": [
            {"id": "f_noise", "fact": "Alice and Ben talked about hobbies and weather.", "session": 4, "metadata": {"episode_id": "ep_noise"}},
        ],
    }

    facts = pick_supporting_facts(
        "Which places or events have Alice and Ben planned for the future?",
        ["ep_workshop", "ep_cafe", "ep_concert"],
        facts_by_episode,
        fact_episode_ids=["ep_workshop", "ep_cafe", "ep_concert", "ep_noise"],
        max_total=6,
        max_per_episode=2,
        allow_pseudo_facts=False,
        query_specificity_bonus=4.0,
    )
    rendered = [fact.get("fact", "") for fact in facts]

    assert any("robotics workshop" in fact for fact in rendered[:4])
    assert any("Harbor Cafe" in fact for fact in rendered[:4])
    assert any("jazz concert on 2024-06-01" in fact for fact in rendered[:4])
    assert all("hobbies and weather" not in fact for fact in rendered[:4])


def test_list_set_plan_queries_treat_backtick_lets_as_commitment():
    facts_by_episode = {
        "ep_workshop": [
            {
                "id": "f_workshop_1",
                "fact": "Alice: The robotics workshop looks useful.",
                "session": 1,
                "metadata": {"episode_id": "ep_workshop"},
            },
            {
                "id": "f_workshop_2",
                "fact": "Ben: Let`s do it next Saturday.",
                "session": 1,
                "metadata": {"episode_id": "ep_workshop"},
            },
            {
                "id": "f_workshop_3",
                "fact": "Alice: It will be great to work with you, Ben.",
                "session": 1,
                "metadata": {"episode_id": "ep_workshop"},
            },
        ],
    }

    facts = pick_supporting_facts(
        "Which places or events have Alice and Ben planned for the future?",
        ["ep_workshop"],
        facts_by_episode,
        fact_episode_ids=["ep_workshop"],
        max_total=2,
        max_per_episode=2,
        allow_pseudo_facts=False,
        query_specificity_bonus=4.0,
    )

    rendered = [fact.get("fact", "") for fact in facts]
    assert "Ben: Let`s do it next Saturday." in rendered[:2]


@pytest.mark.asyncio
async def test_episode_runtime_uses_hybrid_prompt_when_raw_text_present(tmp_path, monkeypatch):
    async def mock_embed_texts(texts, **kwargs):
        return np.ones((len(texts), 3072), dtype=np.float32)

    async def mock_embed_query(text, **kwargs):
        return np.ones(3072, dtype=np.float32)

    async def mock_extract_session(**kwargs):
        sn = kwargs.get("session_num", 1)
        facts = [{
            "id": "f0",
            "fact": "John's goal is to improve his shooting percentage.",
            "kind": "fact",
            "entities": ["John"],
            "tags": ["test"],
            "session": sn,
        }]
        return ("conv", sn, "2024-06-01", facts, [])

    async def mock_consolidate_session(**kwargs):
        return ("conv", 1, "2024-06-01", [])

    async def mock_cross_session_entity(**kwargs):
        return ("conv", "john", [])

    monkeypatch.setattr("src.memory.embed_texts", mock_embed_texts)
    monkeypatch.setattr("src.memory.embed_query", mock_embed_query)
    monkeypatch.setattr("src.memory.extract_session", mock_extract_session)
    monkeypatch.setattr("src.memory.consolidate_session", mock_consolidate_session)
    monkeypatch.setattr("src.memory.cross_session_entity", mock_cross_session_entity)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda facts, lookup: None)

    server = MemoryServer(str(tmp_path), "hybrid_prompt_server")
    await server.store(
        "John: My goal is to improve my shooting percentage and win a championship.",
        session_num=1,
        session_date="2024-06-01",
    )
    recall = await server.recall("What are John's goals with regards to his basketball career?")
    assert recall["recommended_prompt_type"] == "hybrid"
    assert "--- SOURCE EPISODE RAW TEXT ---" in recall["context"]


def test_packet_adaptive_snippet_keeps_multiple_episodes():
    episode_lookup = {
        "ep1": _episode("ep1", "A" * 7000 + "\n\nfoo"),
        "ep2": _episode("ep2", "B" * 7000 + "\n\nbar"),
    }
    facts_by_episode = {
        "ep1": [{"id": "f1", "fact": "foo", "session": 1, "metadata": {"episode_id": "ep1"}}],
        "ep2": [{"id": "f2", "fact": "bar", "session": 2, "metadata": {"episode_id": "ep2"}}],
    }
    context, injected, _fact_ids = build_context_from_selected_episodes(
        "foo bar",
        ["ep1", "ep2"],
        episode_lookup,
        facts_by_episode,
        budget=4000,
        max_total_facts=4,
        max_facts_per_episode=2,
        snippet_mode=False,
        allow_pseudo_facts=True,
    )
    assert injected == ["ep1", "ep2"]
    assert "--- SOURCE EPISODE RAW TEXT ---" in context


def test_packet_can_inject_support_fact_episodes_from_same_source():
    episode_lookup = {
        "ep_kickboxing": _episode(
            "ep_kickboxing",
            "John says he is doing kickboxing.",
            source_id="conv_martial",
        ),
        "ep_taekwondo": _episode(
            "ep_taekwondo",
            "John says he is going to do taekwondo.",
            source_id="conv_martial",
        ),
    }
    facts_by_episode = {
        "ep_kickboxing": [
            {
                "id": "f_kickboxing",
                "fact": "John has done the martial art kickboxing.",
                "session": 1,
                "metadata": {"episode_id": "ep_kickboxing"},
            }
        ],
        "ep_taekwondo": [
            {
                "id": "f_taekwondo",
                "fact": "John has done the martial art taekwondo.",
                "session": 2,
                "metadata": {"episode_id": "ep_taekwondo"},
            }
        ],
    }
    context, injected, selected_fact_ids = build_context_from_selected_episodes(
        "What martial arts has John done?",
        ["ep_kickboxing"],
        episode_lookup,
        facts_by_episode,
        fact_episode_ids=["ep_kickboxing", "ep_taekwondo"],
        budget=4000,
        max_total_facts=6,
        max_facts_per_episode=2,
        allow_pseudo_facts=False,
        inject_support_fact_episodes=True,
        max_injected_support_fact_episodes=4,
    )
    assert "f_taekwondo" in selected_fact_ids
    assert injected == ["ep_kickboxing", "ep_taekwondo"]
    assert "taekwondo" in context.lower()


def test_pick_supporting_facts_prefers_explicit_slot_fill_over_generic_placeholder():
    qf = extract_query_features("What kind of project is Gina doing?")
    supporting = pick_supporting_facts(
        "What kind of project is Gina doing?",
        ["ep_project"],
        {
            "ep_project": [
                {
                    "id": "generic",
                    "fact": "Gina is doing a big project for school.",
                    "session": 1,
                    "metadata": {"episode_id": "ep_project"},
                },
                {
                    "id": "specific",
                    "fact": "Gina is doing an electrical engineering project for school.",
                    "session": 1,
                    "metadata": {"episode_id": "ep_project"},
                },
            ]
        },
        max_total=1,
        max_per_episode=1,
        allow_pseudo_facts=False,
        query_features=qf,
    )

    assert [fact["id"] for fact in supporting] == ["specific"]


def test_pick_supporting_facts_round_robins_for_compositional_queries():
    question = "What hobby would make Toby happy while also helping Maria learn new recipes?"
    supporting = pick_supporting_facts(
        question,
        ["ep_recipes", "ep_dog"],
        {
            "ep_recipes": [
                {
                    "id": "recipes_1",
                    "fact": "Maria has been learning new recipes in cooking class.",
                    "session": 1,
                    "metadata": {"episode_id": "ep_recipes"},
                },
                {
                    "id": "recipes_2",
                    "fact": "Maria wants hobbies that help her practice cooking.",
                    "session": 2,
                    "metadata": {"episode_id": "ep_recipes"},
                },
            ],
            "ep_dog": [
                {
                    "id": "dog_1",
                    "fact": "Toby is happiest when he goes on long runs.",
                    "session": 3,
                    "metadata": {"episode_id": "ep_dog"},
                },
            ],
        },
        max_total=2,
        max_per_episode=2,
        allow_pseudo_facts=False,
        query_features=extract_query_features(question),
    )

    assert {fact["metadata"]["episode_id"] for fact in supporting} == {"ep_recipes", "ep_dog"}


def test_build_context_from_selected_episodes_surfaces_step_anchor_snippet_before_aggregate_facts():
    raw = "\n".join(
        [
            "[Step 0]",
            "Action: None",
            "Observation: None",
            "[Step 6]",
            "Action: execute_bash:",
            "Observation: {'error': 'API error 500'}",
            "[Step 7]",
            "Action: execute_snowflake_sql:",
            "Observation: {'error': 'API error 500'}",
            "[Step 8]",
            "Action: execute_snowflake_sql:",
            "Observation: {'error': 'API error 500'}",
        ]
    )
    episode_lookup = {
        "ep_trace": _episode(
            "ep_trace",
            raw,
            source_id="conv_trace",
        )
    }
    facts_by_episode = {
        "ep_trace": [
            {
                "id": "trace_1",
                "fact": "20 attempts to execute Snowflake SQL commands resulted in API error 500.",
                "session": 1,
                "metadata": {"episode_id": "ep_trace"},
            },
            {
                "id": "trace_2",
                "fact": "5 attempts to execute bash commands resulted in API error 500.",
                "session": 1,
                "metadata": {"episode_id": "ep_trace"},
            },
        ]
    }

    question = (
        "At step 7, which tool did the agent switch to, and what key issue is visible in the call "
        "arguments at that step?"
    )
    context, injected, selected_fact_ids = build_context_from_selected_episodes(
        question,
        ["ep_trace"],
        episode_lookup,
        facts_by_episode,
        budget=5000,
        max_total_facts=3,
        max_facts_per_episode=2,
        snippet_mode=True,
        allow_pseudo_facts=False,
        query_features=extract_query_features(question),
    )

    assert injected == ["ep_trace"]
    assert selected_fact_ids == ["trace_1", "trace_2"]
    assert "LOCAL ANCHOR SNIPPET:" in context
    assert "STEP CALL NOTES:" in context
    assert context.index("LOCAL ANCHOR SNIPPET:") < context.index("RETRIEVED FACTS:")
    assert context.index("STEP CALL NOTES:") < context.index("RETRIEVED FACTS:")
    assert "[Step 6]" in context
    assert "[Step 7]" in context
    assert "Action: execute_snowflake_sql:" in context
    assert "Step 7: action `execute_snowflake_sql` has no argument payload after the colon." in context


def test_build_context_from_selected_episodes_adds_temporal_notes_for_absolute_date_query():
    episode_lookup = {
        "ep_temporal": {
            **_episode(
                "ep_temporal",
                "Maria and her mother made dinner together last night.",
                source_id="conv_temporal",
            ),
            "source_date": "2023-05-04",
        }
    }
    facts_by_episode = {
        "ep_temporal": [
            {
                "id": "temporal_fact",
                "fact": "Maria and her mother made dinner together last night.",
                "session": 1,
                "metadata": {"episode_id": "ep_temporal"},
            }
        ]
    }

    context, injected, selected_fact_ids = build_context_from_selected_episodes(
        "Who did Maria make dinner with on May 3, 2023?",
        ["ep_temporal"],
        episode_lookup,
        facts_by_episode,
        budget=3000,
        max_total_facts=3,
        max_facts_per_episode=2,
        allow_pseudo_facts=False,
    )

    assert injected == ["ep_temporal"]
    assert selected_fact_ids == ["temporal_fact"]
    assert "TEMPORAL NOTES:" in context
    assert "2023-05-03" in context


def test_build_context_from_selected_episodes_surfaces_step_trajectory_notes():
    raw = "\n".join(
        [
            "[Step 2]",
            "Action: left",
            "Observation: Active rules:",
            "key is win",
            "door is win",
            "baba is you",
            "",
            "Objects on the map:",
            "rule `wall` 1 step down",
            "rule `stop` 2 steps to the right and 1 step down",
            "[Step 3]",
            "Action: down",
            "Observation: Active rules:",
            "key is win",
            "door is win",
            "baba is you",
            "",
            "Objects on the map:",
            "rule `stop` 2 steps to the right",
            "rule `wall` 1 step down",
            "[Step 4]",
            "Action: down",
            "Observation: Active rules:",
            "key is win",
            "door is win",
            "baba is you",
            "",
            "Objects on the map:",
            "rule `stop` 2 steps to the right and 1 step up",
            "rule `wall` 1 step down",
            "[Step 5]",
            "Action: down",
            "Observation: Active rules:",
            "key is win",
            "door is win",
            "baba is you",
            "",
            "Objects on the map:",
            "rule `stop` 2 steps to the right and 2 steps up",
            "rule `wall` 1 step down",
            "[Step 6]",
            "Action: down",
            "Observation: Active rules:",
            "key is win",
            "door is win",
            "baba is you",
            "",
            "Objects on the map:",
            "rule `stop` 2 steps to the right and 3 steps up",
        ]
    )
    episode_lookup = {
        "ep_trace": _episode(
            "ep_trace",
            raw,
            source_id="conv_trace",
        )
    }
    facts_by_episode = {
        "ep_trace": [
            {
                "id": "trace_1",
                "fact": "rule `wall` is 1 step down",
                "session": 1,
                "metadata": {"episode_id": "ep_trace"},
            },
        ]
    }

    question = (
        "In the trajectory from step 3 to step 6, the agent performs a sequence of four "
        "consecutive 'down' actions. What specific object is the agent methodically "
        "pushing, and what strategic goal does this multi-step maneuver accomplish?"
    )
    context, injected, selected_fact_ids = build_context_from_selected_episodes(
        question,
        ["ep_trace"],
        episode_lookup,
        facts_by_episode,
        budget=5000,
        max_total_facts=3,
        max_facts_per_episode=2,
        snippet_mode=True,
        allow_pseudo_facts=False,
        query_features=extract_query_features(question),
    )

    assert injected == ["ep_trace"]
    assert selected_fact_ids == ["trace_1"]
    assert "STEP TRAJECTORY NOTES:" in context
    assert "Steps 3-6 all use action `down`." in context
    assert "rule `wall` stays directly ahead of the agent" in context
    assert "clearing a path past that blocker" in context


def test_build_episode_hybrid_context_respects_explicit_search_family():
    corpus = {
        "documents": [
            {
                "doc_id": "conversation:conv_1",
                "episodes": [
                    _episode("conv_e1", "Alice has 5 cats.", source_id="conv_1"),
                ],
            },
            {
                "doc_id": "document:doc_1",
                "episodes": [
                    _episode(
                        "doc_e1",
                        "Contract section 5 sets the delivery deadline.",
                        source_id="doc_1",
                        source_type="document",
                    ),
                ],
            },
        ]
    }
    facts = [
        {
            "id": "f_conv",
            "session": 1,
            "fact": "Alice has 5 cats.",
            "metadata": {"episode_id": "conv_e1", "episode_source_id": "conv_1"},
        },
        {
            "id": "f_doc",
            "session": 2,
            "fact": "Contract section 5 sets the delivery deadline.",
            "metadata": {"episode_id": "doc_e1", "episode_source_id": "doc_1"},
        },
    ]
    packet = build_episode_hybrid_context(
        "What does contract section 5 say?",
        corpus,
        facts,
        search_family="document",
    )
    assert packet["retrieval_families"] == ["document"]
    assert packet["retrieved_episode_ids"] == ["doc_e1"]



def test_build_context_from_retrieved_facts_surfaces_plural_goal_slot_candidate():
    question = "What are John's goals with regards to his basketball career?"
    qf = extract_query_features(question)
    episode_lookup = {
        "ep_goal": _episode("ep_goal", "John: My goal is to improve my shooting percentage and win a championship."),
    }
    fact_lookup = {
        "goal": {
            "id": "goal",
            "session": 1,
            "fact": "John's goal is to improve his shooting percentage.",
            "metadata": {"episode_id": "ep_goal"},
        },
    }
    context, _injected = build_context_from_retrieved_facts(
        [fact_lookup["goal"]],
        episode_lookup,
        fact_lookup,
        question=question,
        query_features=qf,
    )
    assert "RAW SLOT CANDIDATES:" in context
    assert "improve his shooting percentage" in context.lower()


def test_build_context_from_retrieved_facts_surfaces_headless_primary_subject_slot_candidate():
    question = "What kind of car does Evan drive?"
    qf = extract_query_features(question)
    episode_lookup = {
        "ep_car": _episode("ep_car", "Evan: I just got back from a trip with my family in my new Prius."),
    }
    fact_lookup = {
        "car": {
            "id": "car",
            "session": 1,
            "fact": "Evan owns a new Prius.",
            "metadata": {"episode_id": "ep_car"},
        },
    }
    context, _injected = build_context_from_retrieved_facts(
        [fact_lookup["car"]],
        episode_lookup,
        fact_lookup,
        question=question,
        query_features=qf,
    )
    assert "RAW SLOT CANDIDATES:" in context
    assert "prius" in context.lower()


def test_build_context_from_retrieved_facts_surfaces_headless_country_candidate_for_primary_subject():
    question = "Which country was Evan visiting in May 2023?"
    qf = extract_query_features(question)
    episode_lookup = {
        "ep_trip": _episode("ep_trip", "Evan: We all hiked the trails in the Canadian Rockies last week."),
    }
    fact_lookup = {
        "trip": {
            "id": "trip",
            "session": 1,
            "fact": "Evan will travel to Canada next month for honeymoon.",
            "metadata": {"episode_id": "ep_trip"},
        },
    }
    context, _injected = build_context_from_retrieved_facts(
        [fact_lookup["trip"]],
        episode_lookup,
        fact_lookup,
        question=question,
        query_features=qf,
    )
    assert "RAW SLOT CANDIDATES:" in context
    assert "canada" in context.lower()


def test_build_context_from_retrieved_facts_does_not_surface_headless_slot_candidate_for_other_person():
    question = "What type of car did Sam get after his old Prius broke down?"
    qf = extract_query_features(question)
    episode_lookup = {
        "ep_car": _episode("ep_car", "Evan: I just got back from a trip with my family in my new Prius."),
    }
    fact_lookup = {
        "car": {
            "id": "car",
            "session": 1,
            "fact": "Evan owns a new Prius.",
            "metadata": {"episode_id": "ep_car"},
        },
    }
    context, _injected = build_context_from_retrieved_facts(
        [fact_lookup["car"]],
        episode_lookup,
        fact_lookup,
        question=question,
        query_features=qf,
    )
    assert "RAW SLOT CANDIDATES:" not in context



def test_build_context_from_retrieved_facts_keeps_specific_country_candidate_when_earlier_noise_exists():
    question = "Which country was Evan visiting in May 2023?"
    qf = extract_query_features(question)
    episode_lookup = {
        "ep_1": _episode("ep_1", "Evan: I just got back from a trip in my new Prius."),
        "ep_8": _episode("ep_8", "Evan: We should visit the place together soon."),
        "ep_16": _episode("ep_16", "Evan: We will travel to Canada next month for our honeymoon."),
    }
    fact_lookup = {
        "f1": {"id": "f1", "session": 1, "fact": "Evan proposes to visit the place together soon", "metadata": {"episode_id": "ep_8"}},
        "f2": {"id": "f2", "session": 2, "fact": "Evan has a new Prius", "metadata": {"episode_id": "ep_1"}},
        "f3": {"id": "f3", "session": 3, "fact": "Evan and companions went to the Rockies.", "metadata": {"episode_id": "ep_1"}},
        "f4": {"id": "f4", "session": 4, "fact": "Evan loves the idea", "metadata": {"episode_id": "ep_1"}},
        "f5": {"id": "f5", "session": 5, "fact": "Evan will travel to Canada next month for honeymoon.", "metadata": {"episode_id": "ep_16"}},
    }
    context, _injected = build_context_from_retrieved_facts(
        [fact_lookup["f1"], fact_lookup["f2"], fact_lookup["f3"], fact_lookup["f4"], fact_lookup["f5"]],
        episode_lookup,
        fact_lookup,
        question=question,
        query_features=qf,
    )
    raw_idx = context.index("RAW SLOT CANDIDATES:")
    section = context[raw_idx: context.index("--- SOURCE EPISODE RAW TEXT ---")]
    assert "canada" in section.lower()
