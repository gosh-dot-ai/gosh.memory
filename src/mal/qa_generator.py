"""MAL validation Q&A generator — failure slice + control group."""

from .eval_runner import OPERATOR_TYPES


def _collect_facts(snapshot, source_filter=None):
    """Collect facts from all tiers, optionally filtered by source."""
    results = []
    for fact in snapshot.all_granular + snapshot.all_cons + snapshot.all_cross:
        source_id = (
            (fact.get("metadata") or {}).get("episode_source_id")
            or fact.get("source_id")
            or fact.get("conv_id", "")
        )
        text = fact.get("fact", "")
        if not text:
            continue
        if source_filter is not None and not source_filter(source_id):
            continue
        episode_id = (fact.get("metadata") or {}).get("episode_id") or fact.get("id", "")
        results.append({
            "source_id": source_id,
            "episode_id": episode_id,
            "text": text,
        })
    return results


def _build_questions(candidates, n_questions, group_tag="failure"):
    questions = []
    for i in range(min(n_questions, len(candidates))):
        c = candidates[i % len(candidates)]
        op = OPERATOR_TYPES[i % len(OPERATOR_TYPES)]
        questions.append({
            "question": f"What do you know about: {c['text'][:80]}?",
            "correct_source_id": c["source_id"],
            "correct_episode_id": c["episode_id"],
            "answer_span": c["text"],
            "operator": op,
            "group": group_tag,
        })
    return questions


def generate_validation_qa(snapshot, source_ids_hint: list[str],
                           n_questions: int = 10) -> list[dict]:
    """Generate validation Q&A from failure slice only (backward compat)."""
    if source_ids_hint:
        candidates = _collect_facts(snapshot, lambda sid: sid in source_ids_hint)
    else:
        candidates = _collect_facts(snapshot)
    if not candidates:
        candidates = _collect_facts(snapshot)
    return _build_questions(candidates, n_questions, group_tag="failure")


def generate_validation_qa_with_control(
    snapshot,
    source_ids_hint: list[str],
    n_failure_questions: int = 10,
    n_control_questions: int = 10,
) -> tuple[list[dict], list[dict]]:
    """Generate both failure slice and control group Q&A.

    Returns (failure_qa, control_qa).

    failure_qa: questions from sources implicated by feedback.
    control_qa: questions from sources NOT implicated — the regression
        control group. If the atom improves failure_qa but regresses
        control_qa, it's cross-contamination and must be rejected.
    """
    hint_set = set(source_ids_hint) if source_ids_hint else set()

    # Failure slice: facts from implicated sources
    if hint_set:
        failure_facts = _collect_facts(snapshot, lambda sid: sid in hint_set)
    else:
        failure_facts = _collect_facts(snapshot)

    if not failure_facts:
        failure_facts = _collect_facts(snapshot)

    # Control group: facts from NON-implicated sources
    if hint_set:
        control_facts = _collect_facts(snapshot, lambda sid: sid not in hint_set)
    else:
        # No source hints — split facts by position as rough proxy
        all_facts = _collect_facts(snapshot)
        mid = len(all_facts) // 2
        failure_facts = all_facts[:mid] if all_facts else []
        control_facts = all_facts[mid:] if all_facts else []

    failure_qa = _build_questions(failure_facts, n_failure_questions, group_tag="failure")
    control_qa = _build_questions(control_facts, n_control_questions, group_tag="control")

    return failure_qa, control_qa
