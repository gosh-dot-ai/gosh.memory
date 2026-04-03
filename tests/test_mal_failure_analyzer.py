"""MAL failure_analyzer tests — SPEC Section 3.2, 13 + Section 20.

Covers:
- deterministic failure_family = {stage, operator_class_or_shape, signature}
- derived failure_family_key
- no scope/answer to optimizer
- conversation-only reprocessing restricted
"""

import pytest


@pytest.fixture
def analyzer():
    from src.mal.failure_analyzer import FailureAnalyzer
    return FailureAnalyzer()


def _make_trace(stage="first_pass", operator="lookup", **overrides):
    base = {
        "stages": {
            "first_pass": {"status": "ok"},
            "late_fusion": {"status": "ok"},
            "query_operators": {"status": "ok"},
            "packet": {"status": "ok"},
            "episodes": {"status": "ok"},
            "facts": {"status": "ok"},
        },
        "query_type": operator,
        "source_families": ["document"],
    }
    base["stages"][stage] = {"status": "failed", "detail": overrides.get("detail", "score_below_threshold")}
    base.update({k: v for k, v in overrides.items() if k != "detail"})
    return base


# ── deterministic family ──


def test_diagnose_returns_structured_failure_family(analyzer):
    trace = _make_trace(stage="first_pass", operator="lookup", detail="low_entity_phrase_match")
    family = analyzer.diagnose(trace)
    assert "stage" in family
    assert "operator_class_or_shape" in family
    assert "signature" in family


def test_failure_family_key_format(analyzer):
    trace = _make_trace(stage="facts", operator="local_anchor", detail="missing_fact_support")
    family = analyzer.diagnose(trace)
    key = analyzer.derive_family_key(family)
    assert key == f"{family['stage']}|{family['operator_class_or_shape']}|{family['signature']}"
    assert "|" in key
    parts = key.split("|")
    assert len(parts) == 3


def test_earliest_broken_stage_detected(analyzer):
    trace = _make_trace(stage="first_pass")
    family = analyzer.diagnose(trace)
    assert family["stage"] == "first_pass"


def test_later_broken_stage_detected(analyzer):
    trace = _make_trace(stage="packet")
    family = analyzer.diagnose(trace)
    assert family["stage"] == "packet"


# ── mode selection ──


def test_first_pass_failure_selects_retrieval_only(analyzer):
    trace = _make_trace(stage="first_pass")
    mode = analyzer.select_mode(analyzer.diagnose(trace), trace)
    assert mode == "retrieval-only"


def test_episodes_failure_selects_reprocessing(analyzer):
    trace = _make_trace(stage="episodes")
    mode = analyzer.select_mode(analyzer.diagnose(trace), trace)
    assert mode == "reprocessing"


def test_facts_failure_selects_extraction(analyzer):
    trace = _make_trace(stage="facts")
    mode = analyzer.select_mode(analyzer.diagnose(trace), trace)
    assert mode == "extraction"


# ── conversation-only reprocessing restriction ──


def test_reprocessing_on_conversation_only_returns_mode_restricted(analyzer):
    trace = _make_trace(stage="episodes", source_families=["conversation"])
    family = analyzer.diagnose(trace)
    mode = analyzer.select_mode(family, trace)
    assert mode == "MODE_RESTRICTED"


def test_reprocessing_on_mixed_corpus_is_allowed(analyzer):
    trace = _make_trace(stage="episodes", source_families=["conversation", "document"])
    family = analyzer.diagnose(trace)
    mode = analyzer.select_mode(family, trace)
    assert mode == "reprocessing"


# ── no answer/scope leak ──


def test_diagnose_does_not_include_answer_text(analyzer):
    trace = _make_trace(stage="first_pass")
    trace["answer_text"] = "The depth is 780 mm."
    family = analyzer.diagnose(trace)
    assert "answer_text" not in family
    assert "780" not in str(family)


def test_diagnose_does_not_include_query_text(analyzer):
    trace = _make_trace(stage="first_pass")
    trace["query"] = "What is the pipe depth at km 2.3?"
    family = analyzer.diagnose(trace)
    assert "query" not in family
    assert "pipe depth" not in str(family)
