"""MAL feedback_store tests — SPEC Section 1.2, 3.1, 3.1a + Section 20.

Covers:
- normalized feedback event persistence
- disabled bindings reject feedback capture
- explicit/manual and auto-captured events use same schema
- queued -> reserved -> consumed lifecycle
- reserved events requeue on cancel/internal failure
- feedback lacking runtime_trace_ref is not trigger-eligible
- exact retry_chain_id duplicate handling
- no fuzzy retry detection
"""

import json
from pathlib import Path

import pytest


@pytest.fixture
def data_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def control(data_dir):
    from src.mal.control_store import ControlStore
    return ControlStore(data_dir)


@pytest.fixture
def store(data_dir, control):
    from src.mal.feedback_store import FeedbackStore
    return FeedbackStore(data_dir, control)


def _make_event(**overrides):
    base = {
        "key": "proj",
        "agent_id": "default",
        "signal_source": "user_negative_feedback",
        "verdict": "bad_answer",
        "query": "What is the depth at km 2.3?",
        "runtime_trace_ref": "ask_20260327_001",
    }
    base.update(overrides)
    return base


# ── persistence ──


def test_submit_creates_event_file(store, control, data_dir):
    control.set("proj", "default", enabled=True)
    event_id = store.submit("proj", "default", _make_event())
    event_path = Path(data_dir) / "mal" / "proj" / "default" / "feedback" / f"{event_id}.json"
    assert event_path.exists()
    data = json.loads(event_path.read_text())
    assert data["feedback_event_id"] == event_id
    assert data["status"] == "queued"
    assert data["verdict"] == "bad_answer"


def test_submit_assigns_unique_ids(store, control):
    control.set("proj", "default", enabled=True)
    id1 = store.submit("proj", "default", _make_event())
    id2 = store.submit("proj", "default", _make_event(query="different question"))
    assert id1 != id2


# ── disabled binding rejection ──


def test_submit_raises_when_binding_disabled(store):
    with pytest.raises(ValueError, match="MAL_DISABLED"):
        store.submit("proj", "default", _make_event())


def test_submit_raises_when_binding_does_not_exist(store):
    with pytest.raises(ValueError, match="MAL_DISABLED"):
        store.submit("nonexistent", "default", _make_event())


# ── schema consistency ──


def test_event_has_required_fields(store, control):
    control.set("proj", "default", enabled=True)
    event_id = store.submit("proj", "default", _make_event())
    event = store.get_event("proj", "default", event_id)
    assert "feedback_event_id" in event
    assert "key" in event
    assert "agent_id" in event
    assert "signal_source" in event
    assert "verdict" in event
    assert "query" in event
    assert "status" in event
    assert "runtime_trace_ref" in event
    assert "created_at" in event


def test_optional_fields_are_preserved(store, control):
    control.set("proj", "default", enabled=True)
    event_id = store.submit("proj", "default", _make_event(
        response_excerpt="The depth is 780 mm.",
        corrected_answer="980 mm",
        retry_chain_id="retry_001",
        source_ids_hint=["DOC-022"],
    ))
    event = store.get_event("proj", "default", event_id)
    assert event["response_excerpt"] == "The depth is 780 mm."
    assert event["corrected_answer"] == "980 mm"
    assert event["retry_chain_id"] == "retry_001"
    assert event["source_ids_hint"] == ["DOC-022"]


# ── queued -> reserved -> consumed lifecycle (SPEC 3.1a) ──


def test_list_queued_returns_only_queued_events(store, control):
    control.set("proj", "default", enabled=True)
    store.submit("proj", "default", _make_event())
    store.submit("proj", "default", _make_event(query="q2", runtime_trace_ref="ask_002"))
    queued = store.list_queued("proj", "default")
    assert len(queued) == 2
    assert all(e["status"] == "queued" for e in queued)


def test_reserve_transitions_events_from_queued_to_reserved(store, control):
    control.set("proj", "default", enabled=True)
    id1 = store.submit("proj", "default", _make_event())
    id2 = store.submit("proj", "default", _make_event(query="q2", runtime_trace_ref="ask_002"))
    store.reserve("proj", "default", [id1, id2], run_id="run_001")
    queued = store.list_queued("proj", "default")
    assert len(queued) == 0
    e1 = store.get_event("proj", "default", id1)
    assert e1["status"] == "reserved"
    assert e1["reserved_by_run_id"] == "run_001"


def test_reserved_events_are_not_eligible_for_concurrent_selection(store, control):
    control.set("proj", "default", enabled=True)
    id1 = store.submit("proj", "default", _make_event())
    store.reserve("proj", "default", [id1], run_id="run_001")
    eligible = store.list_trigger_eligible("proj", "default")
    assert len(eligible) == 0


def test_consume_transitions_reserved_to_consumed(store, control):
    control.set("proj", "default", enabled=True)
    id1 = store.submit("proj", "default", _make_event())
    store.reserve("proj", "default", [id1], run_id="run_001")
    store.consume("proj", "default", [id1], run_id="run_001")
    e1 = store.get_event("proj", "default", id1)
    assert e1["status"] == "consumed"


def test_consume_from_queued_raises(store, control):
    """Cannot skip reservation — must go queued -> reserved -> consumed."""
    control.set("proj", "default", enabled=True)
    id1 = store.submit("proj", "default", _make_event())
    with pytest.raises(ValueError, match="not reserved"):
        store.consume("proj", "default", [id1], run_id="run_001")


def test_consume_by_wrong_run_id_raises(store, control):
    """Only the reserving run may consume its events."""
    control.set("proj", "default", enabled=True)
    id1 = store.submit("proj", "default", _make_event())
    store.reserve("proj", "default", [id1], run_id="run_001")
    with pytest.raises(ValueError, match="reserved.*run_001"):
        store.consume("proj", "default", [id1], run_id="run_002")


def test_release_transitions_reserved_back_to_queued(store, control):
    """Cancel/internal failure requeues reserved events."""
    control.set("proj", "default", enabled=True)
    id1 = store.submit("proj", "default", _make_event())
    store.reserve("proj", "default", [id1], run_id="run_001")
    assert len(store.list_queued("proj", "default")) == 0
    store.release("proj", "default", [id1], run_id="run_001")
    queued = store.list_queued("proj", "default")
    assert len(queued) == 1
    e1 = store.get_event("proj", "default", id1)
    assert e1["status"] == "queued"
    assert "reserved_by_run_id" not in e1 or e1.get("reserved_by_run_id") is None


def test_release_by_wrong_run_id_raises(store, control):
    """Only the reserving run may release its events."""
    control.set("proj", "default", enabled=True)
    id1 = store.submit("proj", "default", _make_event())
    store.reserve("proj", "default", [id1], run_id="run_001")
    with pytest.raises(ValueError, match="reserved.*run_001"):
        store.release("proj", "default", [id1], run_id="run_002")


def test_release_from_consumed_raises(store, control):
    """Terminal consumed state cannot be released."""
    control.set("proj", "default", enabled=True)
    id1 = store.submit("proj", "default", _make_event())
    store.reserve("proj", "default", [id1], run_id="run_001")
    store.consume("proj", "default", [id1], run_id="run_001")
    with pytest.raises(ValueError, match="not reserved"):
        store.release("proj", "default", [id1], run_id="run_001")


def test_consumed_events_do_not_reappear_in_queued(store, control):
    control.set("proj", "default", enabled=True)
    id1 = store.submit("proj", "default", _make_event())
    store.reserve("proj", "default", [id1], run_id="run_001")
    store.consume("proj", "default", [id1])
    assert len(store.list_queued("proj", "default")) == 0


# ── trigger-eligibility ──


def test_event_without_runtime_trace_ref_is_not_trigger_eligible(store, control):
    control.set("proj", "default", enabled=True)
    store.submit("proj", "default", _make_event(runtime_trace_ref=None))
    eligible = store.list_trigger_eligible("proj", "default")
    assert len(eligible) == 0


def test_event_with_runtime_trace_ref_is_trigger_eligible(store, control):
    control.set("proj", "default", enabled=True)
    store.submit("proj", "default", _make_event())
    eligible = store.list_trigger_eligible("proj", "default")
    assert len(eligible) == 1


def test_event_with_non_adaptation_verdict_is_not_trigger_eligible(store, control):
    control.set("proj", "default", enabled=True)
    store.submit("proj", "default", _make_event(verdict="good_answer"))
    eligible = store.list_trigger_eligible("proj", "default")
    assert len(eligible) == 0


ADAPTATION_VERDICTS = ["bad_answer", "incomplete_answer", "user_correction"]


@pytest.mark.parametrize("verdict", ADAPTATION_VERDICTS)
def test_adaptation_relevant_verdicts_are_eligible(store, control, verdict):
    control.set("proj", "default", enabled=True)
    store.submit("proj", "default", _make_event(verdict=verdict))
    eligible = store.list_trigger_eligible("proj", "default")
    assert len(eligible) == 1


# ── retry-chain deduplication ──


def test_same_retry_chain_id_counts_as_one_independent_failure(store, control):
    control.set("proj", "default", enabled=True)
    store.submit("proj", "default", _make_event(
        retry_chain_id="retry_007",
        runtime_trace_ref="ask_001",
    ))
    store.submit("proj", "default", _make_event(
        retry_chain_id="retry_007",
        runtime_trace_ref="ask_002",
        query="same but retried",
    ))
    eligible = store.list_trigger_eligible("proj", "default")
    independent = store.count_independent_failures(eligible)
    assert independent == 1


def test_different_retry_chain_ids_count_as_independent(store, control):
    control.set("proj", "default", enabled=True)
    store.submit("proj", "default", _make_event(
        retry_chain_id="retry_007",
        runtime_trace_ref="ask_001",
    ))
    store.submit("proj", "default", _make_event(
        retry_chain_id="retry_008",
        runtime_trace_ref="ask_002",
        query="different chain",
    ))
    eligible = store.list_trigger_eligible("proj", "default")
    independent = store.count_independent_failures(eligible)
    assert independent == 2


def test_null_retry_chain_id_falls_back_to_trace_ref_equality(store, control):
    control.set("proj", "default", enabled=True)
    store.submit("proj", "default", _make_event(runtime_trace_ref="ask_001"))
    store.submit("proj", "default", _make_event(runtime_trace_ref="ask_001", query="same trace"))
    eligible = store.list_trigger_eligible("proj", "default")
    independent = store.count_independent_failures(eligible)
    assert independent == 1


def test_null_retry_chain_different_traces_are_independent(store, control):
    control.set("proj", "default", enabled=True)
    store.submit("proj", "default", _make_event(runtime_trace_ref="ask_001"))
    store.submit("proj", "default", _make_event(runtime_trace_ref="ask_002", query="different"))
    eligible = store.list_trigger_eligible("proj", "default")
    independent = store.count_independent_failures(eligible)
    assert independent == 2


def test_no_fuzzy_duplicate_detection(store, control):
    """Events with similar queries but different trace refs are independent."""
    control.set("proj", "default", enabled=True)
    store.submit("proj", "default", _make_event(
        query="What is the pipe depth at km 2.3?",
        runtime_trace_ref="ask_001",
    ))
    store.submit("proj", "default", _make_event(
        query="What is the pipe depth at km 2.3?",
        runtime_trace_ref="ask_002",
    ))
    eligible = store.list_trigger_eligible("proj", "default")
    independent = store.count_independent_failures(eligible)
    assert independent == 2


# ── binding isolation for feedback ──


def test_feedback_isolated_between_bindings(store, control):
    control.set("proj", "agent-A", enabled=True)
    control.set("proj", "agent-B", enabled=True)
    store.submit("proj", "agent-A", _make_event(key="proj", agent_id="agent-A"))
    queued_a = store.list_queued("proj", "agent-A")
    queued_b = store.list_queued("proj", "agent-B")
    assert len(queued_a) == 1
    assert len(queued_b) == 0
