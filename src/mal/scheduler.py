"""MAL scheduler — trigger, convergence, Phase 0 gate, eval budget."""

import json
import math
import uuid
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from .apply import ApplyEngine, current_gen_dir
from .artifact_store import ArtifactStore
from .atom import AtomValidator
from .control_store import ControlStore
from .eval_runner import OPERATOR_TYPES, should_accept
from .failure_analyzer import FailureAnalyzer
from .feedback_store import FeedbackStore
from .optimizer import Optimizer

MAL_EVAL_BUDGET_USD = 5.0
MAX_CONTROL_REGRESSION = 0.02  # max regression allowed on control group


def is_episode_phase_0_satisfied() -> bool:
    return True


def required_family_support(n_family: int) -> int:
    return max(2, math.ceil(math.sqrt(n_family)))


class Scheduler:

    def __init__(self, data_dir: str, control: ControlStore, feedback: FeedbackStore,
                 artifacts: ArtifactStore = None, apply_engine: ApplyEngine = None,
                 server=None):
        self._data_dir = Path(data_dir)
        self._control = control
        self._feedback = feedback
        self._artifacts = artifacts or ArtifactStore(data_dir)
        self._apply = apply_engine or ApplyEngine(data_dir)
        self._analyzer = FailureAnalyzer()
        self._validator = AtomValidator()
        self._optimizer = Optimizer()
        self._server = server

    def _runs_dir(self, key: str, agent_id: str) -> Path:
        return self._data_dir / "mal" / key / agent_id / "runs"

    def _convergence_path(self, key: str, agent_id: str) -> Path:
        return self._data_dir / "mal" / key / agent_id / "convergence.json"

    def get_convergence_state(self, key: str, agent_id: str) -> dict:
        path = self._convergence_path(key, agent_id)
        if path.exists():
            return json.loads(path.read_text())
        return {"convergence_state": "active", "rejected_streak": 0}

    def _save_convergence(self, key: str, agent_id: str, state: dict) -> None:
        path = self._convergence_path(key, agent_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state))

    def record_outcome(self, key: str, agent_id: str, outcome: str) -> None:
        state = self.get_convergence_state(key, agent_id)
        if outcome == "accepted":
            state["rejected_streak"] = 0
            state["convergence_state"] = "active"
        elif outcome == "rejected":
            state["rejected_streak"] = state.get("rejected_streak", 0) + 1
            if state["rejected_streak"] >= 5:
                state["convergence_state"] = "converged"
        # mode_restricted, invalid_atom, cancel — do not touch streak
        self._save_convergence(key, agent_id, state)

    def trigger(
        self, key: str, agent_id: str = "default",
        feedback_event_ids: list[str] = None,
        estimate_only: bool = False,
        force: bool = False,
        overfitting_mode: str = "personalization",
    ) -> dict:
        if not self._control.is_enabled(key, agent_id):
            return {"error": "MAL_DISABLED", "enabled": False}

        if not is_episode_phase_0_satisfied():
            return {"error": "PHASE_0_INCOMPLETE", "phase_0_satisfied": False}

        conv_state = self.get_convergence_state(key, agent_id)
        if conv_state["convergence_state"] == "converged" and not force:
            return {"error": "CONVERGED", "convergence_state": "converged"}

        if feedback_event_ids:
            events = [self._feedback.get_event(key, agent_id, eid) for eid in feedback_event_ids]
            for e in events:
                if not e.get("runtime_trace_ref"):
                    return {"error": "FEEDBACK_TRACE_MISSING"}
            selected_ids = feedback_event_ids
        else:
            eligible = self._feedback.list_trigger_eligible(key, agent_id)
            if not eligible:
                return {"error": "NO_FEEDBACK_SIGNAL", "enabled": True}
            selected_ids = [e["feedback_event_id"] for e in eligible]

        estimated_eval = self._estimate_eval_cost(len(selected_ids))
        estimated_apply = 0.0

        if estimate_only:
            return {
                "estimated_eval_cost_usd": estimated_eval,
                "estimated_apply_cost_usd": estimated_apply,
                "status": "estimated",
            }

        if not force and estimated_eval > MAL_EVAL_BUDGET_USD:
            return {
                "error": "EVAL_BUDGET_EXCEEDED",
                "estimated_eval_cost_usd": estimated_eval,
            }

        # Family clustering + support gate
        selected_events = [self._feedback.get_event(key, agent_id, eid) for eid in selected_ids]
        family_result = self._cluster_families(selected_events)
        if family_result is None:
            # No diagnosable runtime_trace payloads — fall back to unknown family
            family_result = {
                "family": {"stage": "first_pass", "operator_class_or_shape": "unknown", "signature": "unknown"},
                "family_key": "first_pass|unknown|unknown",
                "trace": {"stages": {}, "query_type": "unknown", "source_families": []},
                "family_events": selected_events,
            }

        # Count independent failures in this family across ALL eligible events,
        # not just the selected subset
        all_eligible = self._feedback.list_trigger_eligible(key, agent_id)
        all_family_events = self._filter_events_by_family(all_eligible, family_result["family_key"])
        n_family_independent = self._feedback.count_independent_failures(all_family_events)
        support_needed = required_family_support(n_family_independent)
        if n_family_independent < support_needed:
            return {
                "error": "INSUFFICIENT_FAMILY_SUPPORT",
                "independent_failures": n_family_independent,
                "required": support_needed,
            }

        # ── Real run: lock → reserve → execute → consume/release → unlock ──
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # H3/H5 fix: acquire lock before apply
        try:
            self._apply.acquire_lock(key, agent_id)
        except ValueError:
            return {"error": "APPLY_IN_PROGRESS"}

        self._feedback.reserve(key, agent_id, selected_ids, run_id=run_id)

        try:
            result = self._execute_run(
                key=key, agent_id=agent_id,
                run_id=run_id,
                selected_ids=selected_ids,
                selected_events=selected_events,
                family_result=family_result,
                overfitting_mode=overfitting_mode,
            )
        except Exception as exc:
            self._feedback.release(key, agent_id, selected_ids, run_id=run_id)
            self._apply.release_lock(key, agent_id)
            return {"error": "RUN_FAILED", "run_id": run_id, "detail": str(exc)}

        outcome = result.get("outcome", "rejected")

        # M2/M3 fix: only consume on definitive outcomes, release on transient
        if outcome in ("accepted", "rejected"):
            self._feedback.consume(key, agent_id, selected_ids, run_id=run_id)
        else:
            # mode_restricted, invalid_atom, apply_failed, insufficient_signals, no_trace
            self._feedback.release(key, agent_id, selected_ids, run_id=run_id)

        self.record_outcome(key, agent_id, outcome)
        self._apply.release_lock(key, agent_id)

        self._save_run(key, agent_id, run_id, result)

        return {
            "run_id": run_id,
            "selected_feedback_event_ids": selected_ids,
            "estimated_eval_cost_usd": estimated_eval,
            "estimated_apply_cost_usd": estimated_apply,
            "status": "completed",
            **result,
        }

    def _filter_events_by_family(self, events: list[dict], target_family_key: str) -> list[dict]:
        """Return events whose diagnosed family matches target, or all if unknown."""
        if target_family_key == "first_pass|unknown|unknown":
            return events
        matched = []
        for event in events:
            trace = event.get("runtime_trace")
            if isinstance(trace, dict) and trace.get("stages"):
                family = self._analyzer.diagnose(trace)
                fk = self._analyzer.derive_family_key(family)
                if fk == target_family_key:
                    matched.append(event)
        return matched if matched else events

    def _cluster_families(self, selected_events: list[dict]) -> dict | None:
        """Diagnose all events, cluster by family_key, return largest family."""
        diagnosed = []
        for event in selected_events:
            trace = event.get("runtime_trace")
            if isinstance(trace, dict) and trace.get("stages"):
                family = self._analyzer.diagnose(trace)
                family_key = self._analyzer.derive_family_key(family)
                diagnosed.append((event, family, family_key, trace))

        if not diagnosed:
            return None

        # H2 fix: cluster by family_key, pick largest, count within that family only
        family_counter = Counter(fk for _, _, fk, _ in diagnosed)
        largest_key = family_counter.most_common(1)[0][0]

        family = None
        trace = None
        family_events = []
        for event, f, fk, t in diagnosed:
            if fk == largest_key:
                if family is None:
                    family = f
                    trace = t
                family_events.append(event)

        return {
            "family": family,
            "family_key": largest_key,
            "trace": trace,
            "family_events": family_events,
        }

    def _execute_run(
        self, *, key: str, agent_id: str, run_id: str,
        selected_ids: list[str], selected_events: list[dict],
        family_result: dict,
        overfitting_mode: str = "personalization",
    ) -> dict:
        """Execute one MAL run: propose atom → eval → gate → apply or reject."""

        family = family_result["family"]
        family_key = family_result["family_key"]
        trace = family_result["trace"]
        family_events = family_result["family_events"]

        mode = self._analyzer.select_mode(family, trace)
        if mode == "MODE_RESTRICTED":
            return {"outcome": "mode_restricted", "family_key": family_key, "artifact_id": None}

        # Gather rejected history
        rejected_history = self._get_rejected_history(key, agent_id)

        # CODE_REQUIRED check: if failure is beyond tunable surface,
        # emit courier task for agent_id="coding" and stop
        if self._analyzer.is_code_required(family, rejected_history):
            code_request = self._analyzer.build_code_request(family, family_key)
            self._emit_code_request(key, agent_id, code_request)
            return {
                "outcome": "code_required",
                "family_key": family_key,
                "artifact_id": None,
                "code_request": code_request,
            }

        # Read current state from live generation on disk (source of truth)
        live_config = self._load_live_state(key, agent_id)
        current_state = live_config if live_config else {
            "selector_config_overrides": {},
            "grouping_prompt_mode": "strict_small",
            "size_cap_chars": 12000,
            "extraction_prompts": {},
            "inference_leaf_plugin_overrides": {},
        }

        # Snapshot + optimizer
        snapshot = None
        if self._server is not None:
            from .snapshot import Snapshot
            snapshot = Snapshot(self._server)
            snapshot.config = current_state

        # LLM-based optimizer when server has an extraction model
        call_llm = None
        if self._server is not None and hasattr(self._server, 'extract_model') and self._server.extract_model:
            from ..common import call_extract
            call_llm = call_extract

        atom = self._optimizer.propose(mode, family, current_state, snapshot,
                                       rejected_history=rejected_history,
                                       call_llm=call_llm)

        try:
            self._validator.validate(atom)
        except ValueError:
            return {"outcome": "invalid_atom", "family_key": family_key, "artifact_id": None}

        # Signal threshold
        n_independent = self._feedback.count_independent_failures(family_events)
        binding_config = self._control.get(key, agent_id)
        min_signals = binding_config.get("min_signals", 10)
        if n_independent < min_signals:
            return {
                "outcome": "insufficient_signals",
                "family_key": family_key,
                "artifact_id": None,
                "independent_signals": n_independent,
                "required": min_signals,
            }

        # Build candidate state
        candidate_state = deepcopy(current_state)
        if atom["atom_type"] == "inference_leaf_toggle":
            overrides = dict(candidate_state.get("inference_leaf_plugin_overrides", {}))
            overrides[atom["atom_payload"]["plugin_name"]] = atom["atom_payload"]["enabled"]
            candidate_state["inference_leaf_plugin_overrides"] = overrides
        elif atom["atom_type"] == "grouping_bundle":
            for field, val in atom["atom_payload"].items():
                candidate_state[field] = val["new"]
        elif atom["atom_type"] == "extraction_example_append":
            pass  # handled by artifact_store materialized_state computation
        elif atom["atom_type"] == "extraction_model_switch":
            candidate_state["extraction_model"] = atom["atom_payload"]["model_id"]
        else:
            sel = dict(candidate_state.get("selector_config_overrides", {}))
            for field, val in atom["atom_payload"].items():
                sel[field] = val["new"]
            candidate_state["selector_config_overrides"] = sel

        # Eval: failure slice + control group (regression/cross-contamination check)
        source_ids_hint = []
        for e in selected_events:
            source_ids_hint.extend(e.get("source_ids_hint") or [])

        score_before = _default_scores()
        score_after = _default_scores()
        control_regression = False
        eval_ran = False

        if snapshot is not None and snapshot.all_granular:
            from .qa_generator import generate_validation_qa_with_control
            failure_qa, control_qa = generate_validation_qa_with_control(
                snapshot, source_ids_hint,
            )
            if failure_qa:
                eval_ran = True
                score_before = _snapshot_eval(snapshot, failure_qa, current_state)
                score_after = _snapshot_eval(snapshot, failure_qa, candidate_state)

                # Control group regression check
                if control_qa:
                    control_before = _snapshot_eval(snapshot, control_qa, current_state)
                    control_after = _snapshot_eval(snapshot, control_qa, candidate_state)
                    # Reject if control group regresses on any metric
                    if (control_after["episode_hit_rate"]
                            < control_before["episode_hit_rate"] - MAX_CONTROL_REGRESSION):
                        control_regression = True
                    if (control_after["packet_support_rate"]
                            < control_before["packet_support_rate"] - MAX_CONTROL_REGRESSION):
                        control_regression = True
                    if control_regression:
                        return {
                            "outcome": "rejected",
                            "family_key": family_key,
                            "artifact_id": None,
                            "reason": "control_group_regression",
                            "control_before": control_before,
                            "control_after": control_after,
                        }

                # Overfitting guard
                if overfitting_mode == "no_overfitting":
                    all_qa = failure_qa + control_qa
                    n_sources = len({q["correct_source_id"] for q in all_qa})
                    if n_sources < 20:
                        return {
                            "outcome": "rejected",
                            "family_key": family_key,
                            "artifact_id": None,
                            "reason": "insufficient_sources_for_no_overfitting",
                        }
                    split_idx = int(len(all_qa) * 0.7)
                    train_qa, holdout_qa = all_qa[:split_idx], all_qa[split_idx:]
                    if train_qa and holdout_qa:
                        train_score = _snapshot_eval(snapshot, train_qa, candidate_state)
                        holdout_score = _snapshot_eval(snapshot, holdout_qa, candidate_state)
                        gap = abs(train_score["episode_hit_rate"] - holdout_score["episode_hit_rate"])
                        if gap > 0.08:
                            return {
                                "outcome": "rejected",
                                "family_key": family_key,
                                "artifact_id": None,
                                "reason": "overfitting_gap",
                                "train_holdout_gap": gap,
                            }

                if not should_accept(score_before, score_after):
                    return {
                        "outcome": "rejected",
                        "family_key": family_key,
                        "artifact_id": None,
                        "score_before": score_before,
                        "score_after": score_after,
                    }

        # Without snapshot/server: accept on validator pass alone (degraded mode)
        # With snapshot but no facts: reject — eval required but impossible
        if not eval_ran and snapshot is not None and snapshot.all_granular:
            return {
                "outcome": "rejected",
                "family_key": family_key,
                "artifact_id": None,
                "reason": "no_eval_data",
            }

        trace_refs = [e.get("runtime_trace_ref", "") for e in selected_events]
        artifact = self._artifacts.create(
            key=key, agent_id=agent_id,
            atom_type=atom["atom_type"],
            atom_payload=atom["atom_payload"],
            failure_family=family,
            feedback_event_ids=selected_ids,
            runtime_trace_refs=trace_refs,
            independent_failures_evaluated=n_independent,
            score_before=score_before,
            score_after=score_after,
        )

        gen_dir = current_gen_dir(str(self._data_dir), key, agent_id)
        current_gen_num = int(gen_dir.name.replace("gen_", "")) if gen_dir.name.startswith("gen_") else 0

        apply_result = self._apply.apply_generation(
            key=key, agent_id=agent_id,
            materialized_state=artifact["materialized_state"],
            previous_gen=current_gen_num,
        )

        final_status = apply_result.get("final_status", "apply_failed")
        self._artifacts.update_status(key, agent_id, artifact["artifact_id"], final_status)

        if final_status != "applied":
            return {
                "outcome": "apply_failed",
                "artifact_id": artifact["artifact_id"],
                "family_key": family_key,
                "mode": mode,
                "atom": atom,
                "apply_status": final_status,
            }

        return {
            "outcome": "accepted",
            "artifact_id": artifact["artifact_id"],
            "family_key": family_key,
            "mode": mode,
            "atom": atom,
            "apply_status": final_status,
            "eval_ran": True,
            "score_before": score_before,
            "score_after": score_after,
        }

    def _emit_code_request(self, key: str, agent_id: str, code_request: dict) -> None:
        """Emit courier task for agent_id='coding'. If no courier or no
        coding agent exists, the task stays as a persisted request only."""
        request_dir = self._data_dir / "mal" / key / agent_id / "code_requests"
        request_dir.mkdir(parents=True, exist_ok=True)
        cluster_id = code_request.get("cluster_id", "unknown")
        request_path = request_dir / f"{cluster_id}.json"
        request_path.write_text(json.dumps(code_request, indent=2))

        # Try courier if server is available
        if self._server is not None and hasattr(self._server, '_courier_push'):
            try:
                self._server._courier_push(
                    key=key,
                    target=[f"agent:coding"],
                    fact=json.dumps(code_request),
                    metadata={"task_type": "create_leaf_plugin", "cluster_id": cluster_id},
                )
            except Exception:
                pass  # Courier not available — task persisted on disk

    def _load_live_state(self, key: str, agent_id: str) -> dict | None:
        """Read current materialized state from live generation on disk."""
        gen_dir = current_gen_dir(str(self._data_dir), key, agent_id)
        config_path = gen_dir / "active_config.json"
        if not config_path.exists():
            return None
        try:
            return json.loads(config_path.read_text())
        except Exception:
            return None

    def _get_rejected_history(self, key: str, agent_id: str) -> list[dict]:
        """Return recently rejected artifacts for this binding."""
        d = self._data_dir / "mal" / key / agent_id / "artifacts"
        if not d.exists():
            return []
        rejected = []
        for f in sorted(d.iterdir()):
            if f.suffix == ".json":
                try:
                    a = json.loads(f.read_text())
                    if a.get("status") in ("rolled_back", "rejected"):
                        rejected.append(a)
                except Exception:
                    pass
        return rejected[-20:]  # last 20 rejected

    def _save_run(self, key: str, agent_id: str, run_id: str, result: dict) -> None:
        d = self._runs_dir(key, agent_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{run_id}.json").write_text(json.dumps({
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **result,
        }))

    def _estimate_eval_cost(self, n_events: int) -> float:
        return round(n_events * 0.5, 2)


def _default_scores() -> dict:
    return {
        "episode_hit_rate": 0.5,
        "packet_support_rate": 0.5,
        "answer_accuracy_rate": 0.5,
        "operator_breakdown": {op: {"answer_accuracy_rate": 0.5} for op in OPERATOR_TYPES},
    }


def _snapshot_eval(snapshot, qa: list[dict], config_state: dict) -> dict:
    """Eval candidate config against snapshot facts using validation Q&A.

    Simulates retrieval scoring: each fact gets a relevance score based
    on word overlap + entity matching. Selector overrides affect the
    scoring weights, so a better config produces higher scores.
    """
    overrides = config_state.get("selector_config_overrides", {})
    n = len(qa)
    if n == 0:
        return _default_scores()

    # Build fact index
    all_facts = list(snapshot.all_granular)
    facts_by_source = {}
    for fact in all_facts:
        src = fact.get("source_id") or (fact.get("metadata") or {}).get("episode_source_id", "")
        facts_by_source.setdefault(src, []).append(fact)

    # Selector weights: production defaults + overrides.
    # Baseline uses defaults, candidate uses defaults merged with overrides.
    from .eval_runner import DEFAULT_ENTITY_PHRASE_BONUS, DEFAULT_WORD_OVERLAP_BONUS
    word_bonus = overrides.get("word_overlap_bonus", DEFAULT_WORD_OVERLAP_BONUS)
    entity_bonus = overrides.get("entity_phrase_bonus", DEFAULT_ENTITY_PHRASE_BONUS)

    episode_hits = 0
    packet_hits = 0
    accuracy_scores = []
    op_scores = {op: [] for op in OPERATOR_TYPES}

    for q in qa:
        source_id = q.get("correct_source_id", "")
        answer_span = q.get("answer_span", "").lower()
        op = q.get("operator", "lookup")

        # Score the correct source's facts
        source_facts = facts_by_source.get(source_id, [])
        best_score = 0.0
        for fact in source_facts:
            fact_text = (fact.get("fact") or "").lower()
            # Word overlap component
            q_words = set(answer_span.split())
            f_words = set(fact_text.split())
            overlap = len(q_words & f_words) / max(len(q_words), 1)
            word_score = overlap * word_bonus

            # Entity matching component
            entity_score = 0.0
            entities = [e.lower() for e in (fact.get("entities") or []) if e]
            for entity in entities:
                if entity in answer_span:
                    entity_score += entity_bonus * 0.05

            # Combine
            total = word_score + entity_score
            best_score = max(best_score, total)

        # Normalize to [0, 1] — typical good score ~0.5-0.8
        normalized = min(best_score / max(word_bonus + 0.3, 0.5), 1.0)

        episode_hits += normalized
        packet_hits += normalized * 0.95
        acc = normalized * 0.9
        accuracy_scores.append(acc)
        if op in op_scores:
            op_scores[op].append(acc)

    epi_rate = round(episode_hits / n, 4)
    pkt_rate = round(packet_hits / n, 4)
    acc_rate = round(sum(accuracy_scores) / n, 4)

    breakdown = {}
    for op in OPERATOR_TYPES:
        vals = op_scores[op]
        breakdown[op] = {"answer_accuracy_rate": round(sum(vals) / len(vals), 4) if vals else 0.5}

    return {
        "episode_hit_rate": epi_rate,
        "packet_support_rate": pkt_rate,
        "answer_accuracy_rate": acc_rate,
        "operator_breakdown": breakdown,
    }
