"""Unit 8: Complexity hint v2 — thorough verification tests."""

import pytest

from src.memory import (
    _compute_complexity_hint,
    _compute_content_complexity,
    _compute_query_shape_complexity,
)
from src.retrieval import detect_query_type

# ── action_item → 0.70 ──

def test_action_item_score_070():
    facts = [{"kind": "action_item", "entities": [], "fact": "Do X"}]
    score = _compute_content_complexity(facts)
    assert score == 0.70


# ── preference → 0.10 ──

def test_preference_score_010():
    facts = [{"kind": "preference", "entities": [], "fact": "Likes Y"}]
    score = _compute_content_complexity(facts)
    assert score == 0.10


# ── 25 entities → ≥ 0.60 ──

def test_25_entities_gte_060():
    entities = [f"entity_{i}" for i in range(25)]
    facts = [{"kind": "fact", "entities": entities, "fact": "many entities"}]
    score = _compute_content_complexity(facts)
    assert score >= 0.60


# ── multi_hop + cross_scope → 0.60 ──

def test_multi_hop_plus_cross_scope():
    fact_lookup = {
        "f1": {"agent_id": "agent_a", "kind": "fact", "_session_content_complexity": 0.0},
        "f2": {"agent_id": "agent_b", "kind": "fact", "_session_content_complexity": 0.0},
    }
    hint = _compute_complexity_hint(
        [{"fact_id": "f1", "sim": 0.8}, {"fact_id": "f2", "sim": 0.7}],
        "default", True, fact_lookup
    )
    # multi_hop(0.35) + cross_scope(0.25) = 0.60
    assert hint["retrieval_complexity"] == 0.60
    assert "multi_hop" in hint["signals"]
    assert "cross_scope" in hint["signals"]


# ── score = max(retrieval, content) ──

def test_score_is_max_of_axes():
    fact_lookup = {
        "f1": {"agent_id": "a", "kind": "action_item", "entities": [], "tags": []},
    }
    hint = _compute_complexity_hint(
        [{"fact_id": "f1", "sim": 0.9}],
        "default", True, fact_lookup
    )
    # retrieval = 0.35 (multi_hop), content = 0.70
    assert hint["score"] == max(hint["retrieval_complexity"], hint["content_complexity"])
    assert hint["score"] == 0.70
    assert hint["dominant"] == "content"


def test_score_max_retrieval_wins():
    fact_lookup = {
        "f1": {"agent_id": "a", "_session_content_complexity": 0.05},
        "f2": {"agent_id": "b", "_session_content_complexity": 0.05},
    }
    hint = _compute_complexity_hint(
        [{"fact_id": f"f{i}", "sim": 0.5} for i in range(1, 3)],
        "supersession", True, fact_lookup
    )
    # multi_hop(0.35) + cross_scope(0.25) + conflict(0.20) = 0.80
    assert hint["retrieval_complexity"] == 0.80
    assert hint["score"] == 0.80
    assert hint["dominant"] == "retrieval"


# ── all response fields present ──

def test_all_response_fields_present():
    hint = _compute_complexity_hint([], "default", False, {})
    required = {"score", "level", "signals", "retrieval_complexity",
                "content_complexity", "query_complexity", "dominant"}
    assert required.issubset(set(hint.keys()))


# ── level boundaries ──

def test_level_boundaries():
    """Verify level mapping for all boundary values."""
    assert _compute_complexity_hint([], "default", False, {})["level"] == 1

    content_cases = [
        ({"fact_id": "f1", "sim": 0.9}, {"f1": {"agent_id": "a", "kind": "fact"}}, 1),
        ({"fact_id": "f1", "sim": 0.9}, {"f1": {"agent_id": "a", "kind": "fact", "event_date": "2024-01-01"}}, 2),
        ({"fact_id": "f1", "sim": 0.9}, {"f1": {"agent_id": "a", "kind": "decision", "entities": [], "tags": []}}, 3),
        ({"fact_id": "f1", "sim": 0.9}, {"f1": {"agent_id": "a", "kind": "action_item", "entities": [], "tags": []}}, 4),
    ]
    for retrieved, fl, expected_level in content_cases:
        hint = _compute_complexity_hint([retrieved], "default", False, fl)
        assert hint["level"] == expected_level

    level5 = _compute_complexity_hint(
        [{"fact_id": f"f{i}", "sim": 0.5} for i in range(60)],
        "supersession",
        True,
        {f"f{i}": {"agent_id": f"agent_{i % 2}", "kind": "fact"} for i in range(60)},
    )
    assert level5["level"] == 5


# ── edge cases ──

def test_empty_facts_returns_zero():
    assert _compute_content_complexity([]) == 0.0


def test_content_complexity_capped():
    score = _compute_content_complexity([
        {"kind": "action_item", "entities": [f"e{i}" for i in range(30)],
         "fact": f"F{i}", "event_date": "2024-01-01"} for i in range(60)
    ])
    assert score <= 1.0


def test_decision_kind():
    facts = [{"kind": "decision", "entities": [], "fact": "Decided X"}]
    assert _compute_content_complexity(facts) == 0.50


def test_constraint_kind():
    facts = [{"kind": "constraint", "entities": [], "fact": "Cannot do Y"}]
    assert _compute_content_complexity(facts) == 0.45


def test_query_shape_complexity_simple_lookup_stays_zero():
    score = _compute_query_shape_complexity(
        "What is the new router serial number for the branch edge replacement?",
        "default",
    )
    assert score == 0.0


def test_query_shape_complexity_security_comparison_stays_balanced():
    score = _compute_query_shape_complexity(
        "Which rollout option should be chosen after the security review, and why is the other risky?",
        "default",
    )
    assert score == 0.50


def test_query_shape_complexity_multi_constraint_tradeoff_escalates_to_strong():
    score = _compute_query_shape_complexity(
        "For the analytics migration, which option is recommended given latency, ownership, and rollback constraints?",
        "default",
    )
    assert score == 0.70


def test_query_shape_complexity_canonical_synthesis_type_uses_constraint_fallback():
    query = "What do I prefer for deployment when cost and compliance both matter?"
    assert detect_query_type(query) == "synthesis"
    score = _compute_query_shape_complexity(query, "synthesis")
    assert score == 0.55


@pytest.mark.parametrize(
    "resolved_type",
    ["aggregate", "counting", "synthesize", "synthesis", "procedural", "rule"],
)
def test_query_shape_complexity_aliases_and_canonical_types_share_constraint_fallback(
    resolved_type,
):
    score = _compute_query_shape_complexity(
        "What approach fits cost and compliance constraints?",
        resolved_type,
    )
    assert score == 0.55


def test_query_complexity_can_dominate_over_content_and_retrieval():
    fact_lookup = {
        "f1": {"agent_id": "a", "kind": "constraint", "entities": [], "tags": []},
    }
    hint = _compute_complexity_hint(
        [{"fact_id": "f1", "sim": 0.9}],
        "default",
        False,
        fact_lookup,
        query="For the analytics migration, which option is recommended given latency, ownership, and rollback constraints?",
    )
    assert hint["query_complexity"] == 0.70
    assert hint["score"] == 0.70
    assert hint["level"] == 4
    assert hint["dominant"] == "query"
