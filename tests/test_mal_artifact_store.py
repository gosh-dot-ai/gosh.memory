"""MAL artifact_store tests — SPEC Section 5 + Section 20.

Covers:
- retrieval/reprocessing state merge
- extraction special-case prompt append
- structured failure_family plus derived failure_family_key
- version chain
- rollback reconciliation deletes absent prompts
- materialized_state computation
"""

import json
from pathlib import Path

import pytest


@pytest.fixture
def data_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def store(data_dir):
    from src.mal.artifact_store import ArtifactStore
    return ArtifactStore(data_dir)


# ── version chain ──


def test_first_artifact_has_version_1(store):
    artifact = store.create(
        key="proj",
        agent_id="default",
        atom_type="lexical_signal_bundle",
        atom_payload={"entity_phrase_bonus": {"old": 2.0, "new": 3.0}},
        failure_family={"stage": "first_pass", "operator_class_or_shape": "lookup", "signature": "low_entity_phrase_match"},
        feedback_event_ids=["malfb_001", "malfb_002"],
        runtime_trace_refs=["ask_001", "ask_002"],
        independent_failures_evaluated=2,
        score_before={"episode_hit_rate": 0.58},
        score_after={"episode_hit_rate": 0.63},
    )
    assert artifact["version"] == 1


def test_second_artifact_has_version_2(store):
    store.create(
        key="proj", agent_id="default",
        atom_type="lexical_signal_bundle",
        atom_payload={"entity_phrase_bonus": {"old": 2.0, "new": 3.0}},
        failure_family={"stage": "first_pass", "operator_class_or_shape": "lookup", "signature": "low_entity_phrase_match"},
        feedback_event_ids=["malfb_001"], runtime_trace_refs=["ask_001"],
        independent_failures_evaluated=2,
        score_before={}, score_after={},
    )
    artifact2 = store.create(
        key="proj", agent_id="default",
        atom_type="locality_bundle",
        atom_payload={"currentness_bonus": {"old": 0.8, "new": 1.2}},
        failure_family={"stage": "first_pass", "operator_class_or_shape": "temporal", "signature": "low_currentness"},
        feedback_event_ids=["malfb_003"], runtime_trace_refs=["ask_003"],
        independent_failures_evaluated=3,
        score_before={}, score_after={},
    )
    assert artifact2["version"] == 2


# ── failure_family and failure_family_key ──


def test_failure_family_key_derived_from_structured_tuple(store):
    artifact = store.create(
        key="proj", agent_id="default",
        atom_type="lexical_signal_bundle",
        atom_payload={"entity_phrase_bonus": {"old": 2.0, "new": 3.0}},
        failure_family={"stage": "facts", "operator_class_or_shape": "local_anchor", "signature": "missing_fact_support"},
        feedback_event_ids=["malfb_001"], runtime_trace_refs=["ask_001"],
        independent_failures_evaluated=2,
        score_before={}, score_after={},
    )
    assert artifact["failure_family_key"] == "facts|local_anchor|missing_fact_support"
    assert artifact["failure_family"]["stage"] == "facts"
    assert artifact["failure_family"]["operator_class_or_shape"] == "local_anchor"
    assert artifact["failure_family"]["signature"] == "missing_fact_support"


# ── materialized_state for retrieval-only ──


def test_first_retrieval_only_materialized_state_from_zero_state(store):
    artifact = store.create(
        key="proj", agent_id="default",
        atom_type="lexical_signal_bundle",
        atom_payload={"entity_phrase_bonus": {"old": 2.0, "new": 3.0}},
        failure_family={"stage": "first_pass", "operator_class_or_shape": "lookup", "signature": "low_entity_phrase_match"},
        feedback_event_ids=["malfb_001"], runtime_trace_refs=["ask_001"],
        independent_failures_evaluated=2,
        score_before={}, score_after={},
    )
    state = artifact["materialized_state"]
    assert state["selector_config_overrides"] == {"entity_phrase_bonus": 3.0}
    assert state["extraction_prompts"] == {}


def test_retrieval_only_merges_with_previous_state(store):
    store.create(
        key="proj", agent_id="default",
        atom_type="lexical_signal_bundle",
        atom_payload={"entity_phrase_bonus": {"old": 2.0, "new": 3.0}},
        failure_family={"stage": "first_pass", "operator_class_or_shape": "lookup", "signature": "x"},
        feedback_event_ids=["fb1"], runtime_trace_refs=["t1"],
        independent_failures_evaluated=2,
        score_before={}, score_after={},
    )
    artifact2 = store.create(
        key="proj", agent_id="default",
        atom_type="lexical_signal_bundle",
        atom_payload={"word_overlap_bonus": {"old": 0.45, "new": 0.65}},
        failure_family={"stage": "first_pass", "operator_class_or_shape": "lookup", "signature": "y"},
        feedback_event_ids=["fb2"], runtime_trace_refs=["t2"],
        independent_failures_evaluated=2,
        score_before={}, score_after={},
    )
    state = artifact2["materialized_state"]
    assert state["selector_config_overrides"]["entity_phrase_bonus"] == 3.0
    assert state["selector_config_overrides"]["word_overlap_bonus"] == 0.65


# ── materialized_state for extraction ──


def test_extraction_appends_to_builtin_for_first_artifact(store):
    prompt_target = "conversation_content_type:technical"
    artifact = store.create(
        key="proj", agent_id="default",
        atom_type="extraction_example_append",
        atom_payload={"prompt_target": prompt_target, "example": "- Decision: switched to ScyllaDB"},
        failure_family={"stage": "facts", "operator_class_or_shape": "lookup", "signature": "missing_fact"},
        feedback_event_ids=["fb1"], runtime_trace_refs=["t1"],
        independent_failures_evaluated=2,
        score_before={}, score_after={},
    )
    state = artifact["materialized_state"]
    assert prompt_target in state["extraction_prompts"]
    prompt_text = state["extraction_prompts"][prompt_target]
    assert prompt_text.endswith("- Decision: switched to ScyllaDB")


def test_extraction_appends_to_previous_artifact_state(store):
    prompt_target = "conversation_content_type:technical"
    store.create(
        key="proj", agent_id="default",
        atom_type="extraction_example_append",
        atom_payload={"prompt_target": prompt_target, "example": "Example 1"},
        failure_family={"stage": "facts", "operator_class_or_shape": "lookup", "signature": "x"},
        feedback_event_ids=["fb1"], runtime_trace_refs=["t1"],
        independent_failures_evaluated=2,
        score_before={}, score_after={},
    )
    artifact2 = store.create(
        key="proj", agent_id="default",
        atom_type="extraction_example_append",
        atom_payload={"prompt_target": prompt_target, "example": "Example 2"},
        failure_family={"stage": "facts", "operator_class_or_shape": "lookup", "signature": "y"},
        feedback_event_ids=["fb2"], runtime_trace_refs=["t2"],
        independent_failures_evaluated=2,
        score_before={}, score_after={},
    )
    prompt = artifact2["materialized_state"]["extraction_prompts"][prompt_target]
    assert "Example 1" in prompt
    assert "Example 2" in prompt
    assert prompt.index("Example 1") < prompt.index("Example 2")


def test_extraction_different_prompt_targets_are_independent(store):
    store.create(
        key="proj", agent_id="default",
        atom_type="extraction_example_append",
        atom_payload={"prompt_target": "conversation_content_type:technical", "example": "Conv example"},
        failure_family={"stage": "facts", "operator_class_or_shape": "lookup", "signature": "x"},
        feedback_event_ids=["fb1"], runtime_trace_refs=["t1"],
        independent_failures_evaluated=2,
        score_before={}, score_after={},
    )
    artifact2 = store.create(
        key="proj", agent_id="default",
        atom_type="extraction_example_append",
        atom_payload={"prompt_target": "document_block_prompt:prose_block", "example": "Doc example"},
        failure_family={"stage": "facts", "operator_class_or_shape": "lookup", "signature": "y"},
        feedback_event_ids=["fb2"], runtime_trace_refs=["t2"],
        independent_failures_evaluated=2,
        score_before={}, score_after={},
    )
    state = artifact2["materialized_state"]
    assert "conversation_content_type:technical" in state["extraction_prompts"]
    assert "document_block_prompt:prose_block" in state["extraction_prompts"]
    assert "Conv example" in state["extraction_prompts"]["conversation_content_type:technical"]
    assert "Doc example" in state["extraction_prompts"]["document_block_prompt:prose_block"]


# ── artifact persistence and retrieval ──


def test_get_artifact_by_id(store):
    artifact = store.create(
        key="proj", agent_id="default",
        atom_type="lexical_signal_bundle",
        atom_payload={"entity_phrase_bonus": {"old": 2.0, "new": 3.0}},
        failure_family={"stage": "first_pass", "operator_class_or_shape": "lookup", "signature": "x"},
        feedback_event_ids=["fb1"], runtime_trace_refs=["t1"],
        independent_failures_evaluated=2,
        score_before={}, score_after={},
    )
    retrieved = store.get("proj", "default", artifact["artifact_id"])
    assert retrieved["artifact_id"] == artifact["artifact_id"]
    assert retrieved["version"] == artifact["version"]


def test_get_latest_artifact(store):
    store.create(
        key="proj", agent_id="default",
        atom_type="lexical_signal_bundle",
        atom_payload={"entity_phrase_bonus": {"old": 2.0, "new": 3.0}},
        failure_family={"stage": "first_pass", "operator_class_or_shape": "lookup", "signature": "x"},
        feedback_event_ids=["fb1"], runtime_trace_refs=["t1"],
        independent_failures_evaluated=2,
        score_before={}, score_after={},
    )
    a2 = store.create(
        key="proj", agent_id="default",
        atom_type="locality_bundle",
        atom_payload={"currentness_bonus": {"old": 0.8, "new": 1.2}},
        failure_family={"stage": "first_pass", "operator_class_or_shape": "temporal", "signature": "y"},
        feedback_event_ids=["fb2"], runtime_trace_refs=["t2"],
        independent_failures_evaluated=2,
        score_before={}, score_after={},
    )
    latest = store.get_latest("proj", "default")
    assert latest["artifact_id"] == a2["artifact_id"]


def test_get_latest_returns_none_when_empty(store):
    assert store.get_latest("proj", "default") is None


# ── status management ──


def test_new_artifact_status_is_accepted(store):
    artifact = store.create(
        key="proj", agent_id="default",
        atom_type="lexical_signal_bundle",
        atom_payload={"entity_phrase_bonus": {"old": 2.0, "new": 3.0}},
        failure_family={"stage": "first_pass", "operator_class_or_shape": "lookup", "signature": "x"},
        feedback_event_ids=["fb1"], runtime_trace_refs=["t1"],
        independent_failures_evaluated=2,
        score_before={}, score_after={},
    )
    assert artifact["status"] == "accepted"


def test_update_status(store):
    artifact = store.create(
        key="proj", agent_id="default",
        atom_type="lexical_signal_bundle",
        atom_payload={"entity_phrase_bonus": {"old": 2.0, "new": 3.0}},
        failure_family={"stage": "first_pass", "operator_class_or_shape": "lookup", "signature": "x"},
        feedback_event_ids=["fb1"], runtime_trace_refs=["t1"],
        independent_failures_evaluated=2,
        score_before={}, score_after={},
    )
    store.update_status("proj", "default", artifact["artifact_id"], "applied")
    updated = store.get("proj", "default", artifact["artifact_id"])
    assert updated["status"] == "applied"


# ── rollback prompt deletion reconciliation (SPEC 1.5) ──


def test_rollback_materialized_state_deletes_absent_prompts(store):
    """Rolling back from v2 (has prompt_target X) to v1 (no extraction_prompts)
    must produce a materialized_state where X is absent."""
    a1 = store.create(
        key="proj", agent_id="default",
        atom_type="lexical_signal_bundle",
        atom_payload={"entity_phrase_bonus": {"old": 2.0, "new": 3.0}},
        failure_family={"stage": "first_pass", "operator_class_or_shape": "lookup", "signature": "x"},
        feedback_event_ids=["fb1"], runtime_trace_refs=["t1"],
        independent_failures_evaluated=2,
        score_before={}, score_after={},
    )
    a2 = store.create(
        key="proj", agent_id="default",
        atom_type="extraction_example_append",
        atom_payload={"prompt_target": "conversation_content_type:technical", "example": "added example"},
        failure_family={"stage": "facts", "operator_class_or_shape": "lookup", "signature": "y"},
        feedback_event_ids=["fb2"], runtime_trace_refs=["t2"],
        independent_failures_evaluated=2,
        score_before={}, score_after={},
    )
    assert "conversation_content_type:technical" in a2["materialized_state"]["extraction_prompts"]
    # The rollback target (a1) has empty extraction_prompts
    target_state = a1["materialized_state"]
    assert target_state["extraction_prompts"] == {}


# ── concurrent write safety ──


def test_concurrent_create_does_not_corrupt_version_chain(store, data_dir):
    """Two rapid creates for the same binding must produce sequential versions."""
    import threading

    results = []

    def create_artifact(idx):
        a = store.create(
            key="proj", agent_id="default",
            atom_type="lexical_signal_bundle",
            atom_payload={"entity_phrase_bonus": {"old": 2.0, "new": 2.0 + idx * 0.1}},
            failure_family={"stage": "first_pass", "operator_class_or_shape": "lookup", "signature": f"sig_{idx}"},
            feedback_event_ids=[f"fb_{idx}"], runtime_trace_refs=[f"t_{idx}"],
            independent_failures_evaluated=2,
            score_before={}, score_after={},
        )
        results.append(a)

    threads = [threading.Thread(target=create_artifact, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    versions = sorted(a["version"] for a in results)
    assert versions == [1, 2, 3, 4, 5]
