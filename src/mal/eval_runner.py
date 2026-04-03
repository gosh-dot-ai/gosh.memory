"""MAL eval runner — acceptance condition + fairness + coverage checks."""

OPERATOR_TYPES = [
    "lookup", "temporal", "ordinal", "commonality",
    "list_set", "compare_diff", "local_anchor", "bounded_chain",
]

MIN_EPISODE_HIT_DELTA = 0.02
MAX_PACKET_REGRESSION = 0.02
MAX_ANSWER_ACCURACY_REGRESSION = 0.02
MAX_OPERATOR_REGRESSION = 0.05

# Production defaults from tuning.json selector section
DEFAULT_WORD_OVERLAP_BONUS = 0.45
DEFAULT_ENTITY_PHRASE_BONUS = 2.0


def should_accept(old: dict, new: dict) -> bool:
    if new["episode_hit_rate"] <= old["episode_hit_rate"] + MIN_EPISODE_HIT_DELTA:
        return False
    if new["packet_support_rate"] < old["packet_support_rate"] - MAX_PACKET_REGRESSION:
        return False
    if new["answer_accuracy_rate"] < old["answer_accuracy_rate"] - MAX_ANSWER_ACCURACY_REGRESSION:
        return False
    for op in OPERATOR_TYPES:
        old_val = old["operator_breakdown"][op]["answer_accuracy_rate"]
        new_val = new["operator_breakdown"][op]["answer_accuracy_rate"]
        if new_val < old_val - MAX_OPERATOR_REGRESSION:
            return False
    return True


def apply_trace_incomplete_fairness(
    baseline_results: list[dict],
    candidate_results: list[dict],
) -> tuple[list[dict], list[dict]]:
    baseline_by_qid = {r["qid"]: r for r in baseline_results}
    candidate_by_qid = {r["qid"]: r for r in candidate_results}

    baseline_skip_qids = {
        r["qid"] for r in baseline_results if r.get("status") == "TRACE_INCOMPLETE"
    }

    filtered_baseline = []
    filtered_candidate = []
    for qid in baseline_by_qid:
        if qid in baseline_skip_qids:
            continue
        filtered_baseline.append(baseline_by_qid[qid])
        cr = candidate_by_qid.get(qid)
        if cr and cr.get("status") == "TRACE_INCOMPLETE":
            cr = dict(cr)
            cr["episode_hit"] = 0.0
            cr["packet_support"] = 0.0
            cr["answer_accuracy"] = 0.0
        if cr:
            filtered_candidate.append(cr)

    return filtered_baseline, filtered_candidate


def check_eval_coverage(
    total_questions: int,
    skipped_questions: int,
    operator_counts: dict[str, int],
) -> dict | None:
    if total_questions > 0 and skipped_questions / total_questions > 0.15:
        return {"error": "EVAL_INSUFFICIENT_COVERAGE"}
    for op in OPERATOR_TYPES:
        if operator_counts.get(op, 0) < 3:
            return {"error": "EVAL_OPERATOR_COVERAGE_INSUFFICIENT", "operator": op}
    return None
