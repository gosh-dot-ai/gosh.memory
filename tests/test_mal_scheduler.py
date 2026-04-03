"""MAL scheduler tests — SPEC Section 3, 13 + Section 20.

Covers:
- disabled binding -> MAL_DISABLED
- no eligible feedback -> NO_FEEDBACK_SIGNAL
- explicit feedback_event_ids override queue
- selected feedback ids recorded into run state
- adaptive required_family_support = max(2, ceil(sqrt(N_family)))
- insufficient family support -> INSUFFICIENT_FAMILY_SUPPORT
- EVAL_BUDGET_EXCEEDED
- convergence reset rules per binding
- Phase 0 gate
"""

import math
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
def feedback(data_dir, control):
    from src.mal.feedback_store import FeedbackStore
    return FeedbackStore(data_dir, control)


@pytest.fixture
def scheduler(data_dir, control, feedback):
    from src.mal.scheduler import Scheduler
    return Scheduler(data_dir, control, feedback)


def _submit_feedback(feedback, control, key="proj", agent_id="default", n=1, **kw):
    control.set(key, agent_id, enabled=True)
    ids = []
    for i in range(n):
        event_id = feedback.submit(key, agent_id, {
            "key": key,
            "agent_id": agent_id,
            "signal_source": "user_negative_feedback",
            "verdict": "bad_answer",
            "query": f"Question {i}",
            "runtime_trace_ref": f"ask_{i:04d}",
            **kw,
        })
        ids.append(event_id)
    return ids


# ── disabled binding ──


def test_trigger_disabled_binding_returns_mal_disabled(scheduler):
    result = scheduler.trigger("proj", "default")
    assert result["error"] == "MAL_DISABLED"


# ── no feedback ──


def test_trigger_enabled_but_no_feedback_returns_no_signal(scheduler, control):
    control.set("proj", "default", enabled=True)
    result = scheduler.trigger("proj", "default")
    assert result["error"] == "NO_FEEDBACK_SIGNAL"


# ── explicit feedback_event_ids override ──


def test_trigger_with_explicit_ids_uses_only_those(scheduler, control, feedback):
    ids = _submit_feedback(feedback, control, n=3)
    result = scheduler.trigger("proj", "default", feedback_event_ids=[ids[0]])
    assert "error" not in result, f"unexpected error: {result.get('error')}"
    assert "selected_feedback_event_ids" in result, "run state must include selected_feedback_event_ids"
    assert result["selected_feedback_event_ids"] == [ids[0]]


# ── missing trace on explicit ids ──


def test_trigger_with_traceless_explicit_id_returns_trace_missing(scheduler, control, feedback):
    control.set("proj", "default", enabled=True)
    event_id = feedback.submit("proj", "default", {
        "key": "proj",
        "agent_id": "default",
        "signal_source": "user_negative_feedback",
        "verdict": "bad_answer",
        "query": "no trace question",
        "runtime_trace_ref": None,
    })
    result = scheduler.trigger("proj", "default", feedback_event_ids=[event_id])
    assert result["error"] == "FEEDBACK_TRACE_MISSING"


# ── adaptive family support ──


@pytest.mark.parametrize("n_family,expected_support", [
    (1, 2),
    (2, 2),
    (3, 2),
    (4, 2),
    (5, 3),
    (9, 3),
    (10, 4),
    (25, 5),
    (100, 10),
])
def test_required_family_support_formula(n_family, expected_support):
    from src.mal.scheduler import required_family_support
    assert required_family_support(n_family) == expected_support
    assert required_family_support(n_family) == max(2, math.ceil(math.sqrt(n_family)))


# ── convergence ──


def test_convergence_initial_state_is_active(scheduler, control, feedback):
    control.set("proj", "default", enabled=True)
    state = scheduler.get_convergence_state("proj", "default")
    assert state["convergence_state"] == "active"
    assert state["rejected_streak"] == 0


def test_convergence_rejected_streak_increments(scheduler, control):
    control.set("proj", "default", enabled=True)
    scheduler.record_outcome("proj", "default", "rejected")
    state = scheduler.get_convergence_state("proj", "default")
    assert state["rejected_streak"] == 1


def test_convergence_accepted_resets_streak(scheduler, control):
    control.set("proj", "default", enabled=True)
    scheduler.record_outcome("proj", "default", "rejected")
    scheduler.record_outcome("proj", "default", "rejected")
    scheduler.record_outcome("proj", "default", "accepted")
    state = scheduler.get_convergence_state("proj", "default")
    assert state["rejected_streak"] == 0
    assert state["convergence_state"] == "active"


def test_convergence_five_rejected_enters_converged(scheduler, control):
    control.set("proj", "default", enabled=True)
    for _ in range(5):
        scheduler.record_outcome("proj", "default", "rejected")
    state = scheduler.get_convergence_state("proj", "default")
    assert state["convergence_state"] == "converged"


def test_convergence_mode_restricted_does_not_count(scheduler, control):
    control.set("proj", "default", enabled=True)
    for _ in range(4):
        scheduler.record_outcome("proj", "default", "rejected")
    scheduler.record_outcome("proj", "default", "mode_restricted")
    state = scheduler.get_convergence_state("proj", "default")
    assert state["rejected_streak"] == 4
    assert state["convergence_state"] == "active"


def test_convergence_invalid_atom_does_not_count(scheduler, control):
    control.set("proj", "default", enabled=True)
    for _ in range(4):
        scheduler.record_outcome("proj", "default", "rejected")
    scheduler.record_outcome("proj", "default", "invalid_atom")
    state = scheduler.get_convergence_state("proj", "default")
    assert state["rejected_streak"] == 4


def test_convergence_exit_after_accepted_atom(scheduler, control):
    control.set("proj", "default", enabled=True)
    for _ in range(5):
        scheduler.record_outcome("proj", "default", "rejected")
    assert scheduler.get_convergence_state("proj", "default")["convergence_state"] == "converged"
    scheduler.record_outcome("proj", "default", "accepted")
    assert scheduler.get_convergence_state("proj", "default")["convergence_state"] == "active"


# ── estimate_only (SPEC 20 scheduler: estimate_only=True) ──


def test_estimate_only_returns_cost_preview_without_snapshot(scheduler, control, feedback):
    _submit_feedback(feedback, control, n=3)
    result = scheduler.trigger("proj", "default", estimate_only=True)
    assert "error" not in result
    assert "estimated_eval_cost_usd" in result
    assert "estimated_apply_cost_usd" in result
    assert result["status"] == "estimated"


def test_estimate_only_does_not_reserve_feedback_events(scheduler, control, feedback):
    _submit_feedback(feedback, control, n=3)
    scheduler.trigger("proj", "default", estimate_only=True)
    queued = feedback.list_queued("proj", "default")
    assert len(queued) == 3
    assert all(e["status"] == "queued" for e in queued)


def test_estimate_only_does_not_create_snapshot_run_or_workspace(scheduler, control, feedback, data_dir):
    _submit_feedback(feedback, control, n=3)
    scheduler.trigger("proj", "default", estimate_only=True)
    mal_dir = Path(data_dir) / "mal" / "proj" / "default"
    runs_dir = mal_dir / "runs"
    snapshots_dir = mal_dir / "snapshots"
    workspace_dir = mal_dir / "apply_workspace"
    if runs_dir.exists():
        assert len(list(runs_dir.iterdir())) == 0, "estimate_only must not create run artifacts"
    if snapshots_dir.exists():
        assert len(list(snapshots_dir.iterdir())) == 0, "estimate_only must not create snapshots"
    assert not workspace_dir.exists(), "estimate_only must not create apply workspaces"


# ── eval budget gate (SPEC 3.6 + 20) ──


def test_trigger_returns_eval_budget_exceeded_when_cost_too_high(scheduler, control, feedback, monkeypatch):
    _submit_feedback(feedback, control, n=3)
    monkeypatch.setattr("src.mal.scheduler.MAL_EVAL_BUDGET_USD", 0.01)
    result = scheduler.trigger("proj", "default")
    assert result["error"] == "EVAL_BUDGET_EXCEEDED"
    assert "estimated_eval_cost_usd" in result


def test_trigger_force_overrides_eval_budget(scheduler, control, feedback, monkeypatch):
    _submit_feedback(feedback, control, n=3)
    monkeypatch.setattr("src.mal.scheduler.MAL_EVAL_BUDGET_USD", 0.01)
    result = scheduler.trigger("proj", "default", force=True)
    assert result.get("error") != "EVAL_BUDGET_EXCEEDED"


# ── phase 0 gate ──


def test_trigger_returns_phase_0_incomplete_when_not_satisfied(scheduler, control, feedback, monkeypatch):
    _submit_feedback(feedback, control, n=3)
    monkeypatch.setattr("src.mal.scheduler.is_episode_phase_0_satisfied", lambda: False)
    result = scheduler.trigger("proj", "default")
    assert result["error"] == "PHASE_0_INCOMPLETE"
    assert result["phase_0_satisfied"] is False
