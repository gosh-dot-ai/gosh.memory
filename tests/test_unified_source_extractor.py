from __future__ import annotations

import pytest

from src.unified_source_extractor import extract_source_aggregation


def _single_episode() -> list[dict]:
    return [
        {
            "episode_id": "conv-30_cat1_e01",
            "source_type": "conversation",
            "source_id": "conv-30_cat1",
            "source_date": "2024-06-01",
            "topic_key": "session_1",
            "state_label": "session",
            "currentness": "unknown",
            "raw_text": "\n".join(
                [
                    "Jon lost his job as a banker.",
                    "Gina lost her Door Dash job.",
                    "Jon started his own business.",
                    "Gina started her own business.",
                ]
            ),
            "metadata": {},
        }
    ]


def _multi_episode() -> list[dict]:
    return [
        {
            "episode_id": "DOC-022_e02",
            "source_type": "document",
            "source_id": "DOC-022",
            "source_date": "2026-01-18",
            "topic_key": "measurement",
            "state_label": "episode",
            "currentness": "unknown",
            "raw_text": "Initial measurement at km 2.3 recorded 780 mm cover depth on 2026-01-18.",
            "metadata": {"source_section_path": "Initial Measurement"},
        },
        {
            "episode_id": "DOC-022_e10",
            "source_type": "document",
            "source_id": "DOC-022",
            "source_date": "2026-02-04",
            "topic_key": "measurement_update",
            "state_label": "episode",
            "currentness": "unknown",
            "raw_text": "On 2026-02-04, the pipe cover depth at km 2.3 was remeasured at 980 mm, superseding the earlier reading.",
            "metadata": {"source_section_path": "Operations Update"},
        },
    ]


def _conversation_source_facts() -> list[dict]:
    return [
        {
            "id": "fact_jon_job",
            "fact": "Jon lost his job as a banker.",
            "entities": ["Jon"],
            "metadata": {"episode_id": "conv-30_cat1_e01", "episode_source_id": "conv-30_cat1"},
        },
        {
            "id": "fact_gina_job",
            "fact": "Gina lost her Door Dash job.",
            "entities": ["Gina"],
            "metadata": {"episode_id": "conv-30_cat1_e01", "episode_source_id": "conv-30_cat1"},
        },
        {
            "id": "fact_jon_business",
            "fact": "Jon started his own business.",
            "entities": ["Jon"],
            "metadata": {"episode_id": "conv-30_cat1_e01", "episode_source_id": "conv-30_cat1"},
        },
        {
            "id": "fact_gina_business",
            "fact": "Gina started her own business.",
            "entities": ["Gina"],
            "metadata": {"episode_id": "conv-30_cat1_e01", "episode_source_id": "conv-30_cat1"},
        },
    ]


def _document_source_facts_with_noncanonical_ids() -> list[dict]:
    return [
        {
            "id": "s2_f_01",
            "fact": "Initial measurement at km 2.3 recorded 780 mm cover depth on 2026-01-18.",
            "entities": ["pipe_cover_depth_km_2_3"],
            "metadata": {"episode_id": "DOC-022_e02", "episode_source_id": "DOC-022"},
        },
        {
            "id": "s10_f_02",
            "fact": "On 2026-02-04, the pipe cover depth at km 2.3 was remeasured at 980 mm, superseding the earlier reading.",
            "entities": ["pipe_cover_depth_km_2_3"],
            "metadata": {"episode_id": "DOC-022_e10", "episode_source_id": "DOC-022"},
        },
    ]


@pytest.mark.asyncio
async def test_extract_source_aggregation_reuses_provided_grounded_facts_and_ignores_model_atomic_facts():
    async def mock_call_extract_fn(model, system, user_msg, max_tokens=8192):
        return {
            "atomic_facts": [
                {
                    "fact_id": "bogus_fact",
                    "subject": "bogus",
                    "relation": "bogus",
                    "object": "bogus",
                    "value_text": "bogus",
                    "value_number": None,
                    "value_unit": None,
                    "polarity": "positive",
                    "confidence": 1.0,
                    "source_span": "bogus",
                    "source_span_start": 0,
                    "source_span_end": 5,
                    "asserted_at": None,
                    "entity_ids": [],
                }
            ],
            "revision_currentness": [],
            "events": [
                {
                    "event_id": "event_shared_business_start",
                    "event_type": "business_start",
                    "participants": ["Jon", "Gina"],
                    "object": "own business",
                    "time": None,
                    "location": None,
                    "parameters": [],
                    "outcome": None,
                    "status": None,
                    "support_fact_ids": ["fact_jon_business", "fact_gina_business"],
                }
            ],
            "records": [],
            "edges": [
                {
                    "edge_id": "edge_shared_root",
                    "edge_type": "same_anchor",
                    "from_id": "fact_jon_job",
                    "to_id": "fact_gina_job",
                    "edge_evidence_text": None,
                    "anchor_key": "lost job",
                    "anchor_basis_fact_ids": ["fact_jon_job", "fact_gina_job"],
                    "support_fact_ids": ["fact_jon_job", "fact_gina_job"],
                }
            ],
        }

    result = await extract_source_aggregation(
        source_id="conv-30_cat1",
        source_kind="conversation",
        episodes=_single_episode(),
        source_facts=_conversation_source_facts(),
        model="qwen/qwen3-32b",
        call_extract_fn=mock_call_extract_fn,
    )

    assert result is not None
    assert result["validation"]["aggregation_status"] == "accepted"
    payload = result["validation"]["payload"]
    assert {fact["fact_id"] for fact in payload["atomic_facts"]} == {
        "fact_jon_job",
        "fact_gina_job",
        "fact_jon_business",
        "fact_gina_business",
    }
    assert "bogus_fact" not in {fact["fact_id"] for fact in payload["atomic_facts"]}
    assert {fact["id"] for fact in result["derived_facts"]} == {
        "event_shared_business_start",
        "edge_shared_root",
    }


@pytest.mark.asyncio
async def test_extract_source_aggregation_rejects_empty_higher_order_success():
    async def mock_call_extract_fn(model, system, user_msg, max_tokens=8192):
        return {
            "revision_currentness": [],
            "events": [],
            "records": [],
            "edges": [],
        }

    result = await extract_source_aggregation(
        source_id="conv-30_cat1",
        source_kind="conversation",
        episodes=_single_episode(),
        source_facts=_conversation_source_facts(),
        model="qwen/qwen3-32b",
        call_extract_fn=mock_call_extract_fn,
    )

    assert result is not None
    assert result["derived_facts"] == []
    validation = result["validation"]
    assert validation["aggregation_status"] == "failed"
    assert validation["payload"] is None
    assert "empty_source_aggregation" in validation["failure_reasons"]
    assert "source_aggregation_retry_exhausted" in validation["failure_reasons"]


@pytest.mark.asyncio
async def test_extract_source_aggregation_preserves_episode_provenance_for_noncanonical_fact_ids():
    async def mock_call_extract_fn(model, system, user_msg, max_tokens=8192):
        return {
            "revision_currentness": [
                {
                    "revision_id": "rev_cover_depth",
                    "topic_key": "pipe_cover_depth_km_2_3",
                    "old_fact_id": "s2_f_01",
                    "new_fact_id": "s10_f_02",
                    "link_type": "supersedes",
                    "current_fact_id": "s10_f_02",
                    "effective_date": "2026-02-04",
                    "revision_source_fact_ids": ["s2_f_01", "s10_f_02"],
                }
            ],
            "events": [],
            "records": [],
            "edges": [],
        }

    result = await extract_source_aggregation(
        source_id="DOC-022",
        source_kind="document",
        episodes=_multi_episode(),
        source_facts=_document_source_facts_with_noncanonical_ids(),
        model="qwen/qwen3-32b",
        call_extract_fn=mock_call_extract_fn,
    )

    assert result is not None
    assert result["validation"]["aggregation_status"] == "accepted"
    assert result["derived_facts"][0]["metadata"]["episode_ids"] == ["DOC-022_e02", "DOC-022_e10"]


@pytest.mark.asyncio
async def test_extract_source_aggregation_returns_failed_validation_instead_of_raising_on_invalid_higher_order_refs():
    async def mock_call_extract_fn(model, system, user_msg, max_tokens=8192):
        return {
            "revision_currentness": [],
            "events": [
                {
                    "event_id": "event_bad_support",
                    "event_type": "business_start",
                    "participants": ["Jon"],
                    "object": "own business",
                    "time": None,
                    "location": None,
                    "parameters": [],
                    "outcome": None,
                    "status": None,
                    "support_fact_ids": ["missing_fact"],
                }
            ],
            "records": [],
            "edges": [],
        }

    result = await extract_source_aggregation(
        source_id="conv-30_cat1",
        source_kind="conversation",
        episodes=_single_episode(),
        source_facts=_conversation_source_facts(),
        model="qwen/qwen3-32b",
        call_extract_fn=mock_call_extract_fn,
    )

    assert result is not None
    assert result["derived_facts"] == []
    validation = result["validation"]
    assert validation["aggregation_status"] == "failed"
    assert validation["payload"] is None
    assert "invalid_event_schema" in validation["failure_reasons"]
    assert "empty_source_aggregation" in validation["failure_reasons"]
    assert "source_aggregation_retry_exhausted" in validation["failure_reasons"]
