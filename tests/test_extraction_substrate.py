"""TDD contract tests for the unified extraction substrate.

These tests define the production API and behavioral contract before
implementation. They deliberately target ``src/`` only and do not rely on any
multibench harness.
"""

from __future__ import annotations

import copy
import json

import pytest

from src.extraction_substrate import (
    compute_episode_extraction_budget,
    compute_source_aggregation_budget,
    compute_source_pipeline_budget,
    persist_episode_payload,
    persist_source_aggregation_payload,
    run_episode_validation_pipeline,
    run_source_aggregation_validation_pipeline,
    validate_episode_payload,
    validate_source_aggregation_payload,
)

DOC_SOURCE_TEXT = "\n".join(
    [
        "Initial measurement at km 2.3 recorded 780 mm cover depth on 2026-01-18.",
        "On 2026-02-04, the pipe cover depth at km 2.3 was remeasured at 980 mm, superseding the earlier reading.",
        "The pressure test for pipeline segment IS-3 ran at 24 bar for 4 hours and passed with zero leaks.",
        "Permit T-17 was approved on 2026-02-12.",
    ]
)

CONV30_E01_TEXT = "\n".join(
    [
        "After losing his job as a banker, Jon decided to start his own business.",
        "After losing her Door Dash job, Gina decided to start her own business.",
    ]
)

CONV30_E19_TEXT = "\n".join(
    [
        "Jon later opened a dance studio.",
        "Gina later kept building her business.",
    ]
)


def _span_bounds(text: str, span: str) -> tuple[int, int]:
    start = text.index(span)
    end = start + len(span)
    return start, end


def _fact(
    *,
    fact_id: str,
    subject: str,
    relation: str,
    object_text: str,
    source_text: str,
    source_span: str,
    value_text: str | None = None,
    value_number: float | int | None = None,
    value_unit: str | None = None,
    asserted_at: str | None = None,
    entity_ids: list[str] | None = None,
) -> dict:
    start, end = _span_bounds(source_text, source_span)
    return {
        "fact_id": fact_id,
        "subject": subject,
        "relation": relation,
        "object": object_text,
        "value_text": value_text,
        "value_number": value_number,
        "value_unit": value_unit,
        "polarity": "positive",
        "confidence": 0.93,
        "source_span": source_span,
        "source_span_start": start,
        "source_span_end": end,
        "asserted_at": asserted_at,
        "entity_ids": entity_ids or [],
    }


def _document_locality_metadata() -> dict:
    return {
        "source_id": "DOC-022",
        "episode_id": "DOC-022_e10",
        "section_id": "DOC-022_s03",
        "heading": "Operations Update",
        "table_id": None,
        "list_id": None,
        "paragraph_cluster_id": "DOC-022_p07",
        "neighbor_episode_ids": ["DOC-022_e09", "DOC-022_e11"],
    }


def _valid_document_episode_payload() -> dict:
    old_span = "Initial measurement at km 2.3 recorded 780 mm cover depth on 2026-01-18."
    new_span = (
        "On 2026-02-04, the pipe cover depth at km 2.3 was remeasured at 980 mm, "
        "superseding the earlier reading."
    )
    pressure_span = (
        "The pressure test for pipeline segment IS-3 ran at 24 bar for 4 hours "
        "and passed with zero leaks."
    )
    permit_span = "Permit T-17 was approved on 2026-02-12."

    facts = [
        _fact(
            fact_id="ep_DOC-022_e10_f_01",
            subject="pipe cover depth at km 2.3",
            relation="measured_as",
            object_text="780 mm",
            source_text=DOC_SOURCE_TEXT,
            source_span=old_span,
            value_text="780 mm",
            value_number=780.0,
            value_unit="mm",
            asserted_at="2026-01-18",
            entity_ids=["pipe_cover_depth_km_2_3"],
        ),
        _fact(
            fact_id="ep_DOC-022_e10_f_02",
            subject="pipe cover depth at km 2.3",
            relation="measured_as",
            object_text="980 mm",
            source_text=DOC_SOURCE_TEXT,
            source_span=new_span,
            value_text="980 mm",
            value_number=980.0,
            value_unit="mm",
            asserted_at="2026-02-04",
            entity_ids=["pipe_cover_depth_km_2_3"],
        ),
        _fact(
            fact_id="ep_DOC-022_e10_f_03",
            subject="pressure test for pipeline segment IS-3",
            relation="pressure",
            object_text="24 bar",
            source_text=DOC_SOURCE_TEXT,
            source_span=pressure_span,
            value_text="24 bar",
            value_number=24.0,
            value_unit="bar",
            entity_ids=["pressure_test_is_3"],
        ),
        _fact(
            fact_id="ep_DOC-022_e10_f_04",
            subject="pressure test for pipeline segment IS-3",
            relation="duration",
            object_text="4 hours",
            source_text=DOC_SOURCE_TEXT,
            source_span=pressure_span,
            value_text="4 hours",
            value_number=4.0,
            value_unit="hours",
            entity_ids=["pressure_test_is_3"],
        ),
        _fact(
            fact_id="ep_DOC-022_e10_f_05",
            subject="pressure test for pipeline segment IS-3",
            relation="outcome",
            object_text="zero leaks",
            source_text=DOC_SOURCE_TEXT,
            source_span=pressure_span,
            value_text="zero leaks",
            entity_ids=["pressure_test_is_3"],
        ),
        _fact(
            fact_id="ep_DOC-022_e10_f_06",
            subject="Permit T-17",
            relation="status",
            object_text="approved",
            source_text=DOC_SOURCE_TEXT,
            source_span=permit_span,
            value_text="approved",
            asserted_at="2026-02-12",
            entity_ids=["permit_T_17"],
        ),
    ]

    return {
        "schema": "extraction_substrate",
        "source_id": "DOC-022",
        "episode_id": "DOC-022_e10",
        "source_kind": "document_episode",
        "atomic_facts": facts,
        "locality": _document_locality_metadata(),
        "revision_currentness": [
            {
                "revision_id": "rev_DOC-022_km2_3_001",
                "topic_key": "pipe_cover_depth_km_2_3",
                "old_fact_id": "ep_DOC-022_e10_f_01",
                "new_fact_id": "ep_DOC-022_e10_f_02",
                "link_type": "supersedes",
                "current_fact_id": "ep_DOC-022_e10_f_02",
                "effective_date": "2026-02-04",
                "revision_source_fact_ids": [
                    "ep_DOC-022_e10_f_01",
                    "ep_DOC-022_e10_f_02",
                ],
            }
        ],
        "events": [
            {
                "event_id": "event_DOC-022_pressure_test_001",
                "event_type": "pressure_test",
                "participants": ["pipeline segment IS-3"],
                "object": "pipeline segment IS-3",
                "time": None,
                "location": "IS-3",
                "parameters": [
                    {
                        "name": "pressure",
                        "value_number": 24.0,
                        "value_unit": "bar",
                        "value_text": "24 bar",
                    },
                    {
                        "name": "duration",
                        "value_number": 4.0,
                        "value_unit": "hours",
                        "value_text": "4 hours",
                    },
                ],
                "outcome": "zero leaks",
                "status": "passed",
                "support_fact_ids": [
                    "ep_DOC-022_e10_f_03",
                    "ep_DOC-022_e10_f_04",
                    "ep_DOC-022_e10_f_05",
                ],
            }
        ],
        "records": [
            {
                "record_id": "record_DOC-022_permit_T17",
                "record_type": "permit",
                "item_id": "T-17",
                "status": "approved",
                "date": "2026-02-12",
                "qualifier": None,
                "owner": None,
                "source_section": "Operations Update",
                "support_fact_ids": ["ep_DOC-022_e10_f_06"],
            }
        ],
        "edges": [
            {
                "edge_id": "edge_DOC-022_same_anchor_001",
                "edge_type": "same_anchor",
                "from_id": "ep_DOC-022_e10_f_01",
                "to_id": "ep_DOC-022_e10_f_02",
                "edge_evidence_text": None,
                "anchor_key": "pipe cover depth at km 2.3",
                "anchor_basis_fact_ids": [
                    "ep_DOC-022_e10_f_01",
                    "ep_DOC-022_e10_f_02",
                ],
                "support_fact_ids": [
                    "ep_DOC-022_e10_f_01",
                    "ep_DOC-022_e10_f_02",
                ],
            },
            {
                "edge_id": "edge_DOC-022_resolver_001",
                "edge_type": "resolver_for",
                "from_id": "ep_DOC-022_e10_f_02",
                "to_id": "ep_DOC-022_e10_f_01",
                "edge_evidence_text": "superseding the earlier reading",
                "anchor_key": None,
                "anchor_basis_fact_ids": [],
                "support_fact_ids": ["ep_DOC-022_e10_f_02"],
            },
            {
                "edge_id": "edge_DOC-022_fact_to_event_001",
                "edge_type": "belongs_to_event",
                "from_id": "ep_DOC-022_e10_f_03",
                "to_id": "event_DOC-022_pressure_test_001",
                "edge_evidence_text": None,
                "anchor_key": None,
                "anchor_basis_fact_ids": [],
                "support_fact_ids": ["ep_DOC-022_e10_f_03"],
            },
            {
                "edge_id": "edge_DOC-022_fact_to_record_001",
                "edge_type": "belongs_to_record",
                "from_id": "ep_DOC-022_e10_f_06",
                "to_id": "record_DOC-022_permit_T17",
                "edge_evidence_text": None,
                "anchor_key": None,
                "anchor_basis_fact_ids": [],
                "support_fact_ids": ["ep_DOC-022_e10_f_06"],
            },
        ],
    }


def _conversation_locality(episode_id: str, heading: str, neighbors: list[str]) -> dict:
    return {
        "source_id": "conv-30_cat1",
        "episode_id": episode_id,
        "section_id": None,
        "heading": heading,
        "table_id": None,
        "list_id": None,
        "paragraph_cluster_id": f"{episode_id}_p01",
        "neighbor_episode_ids": neighbors,
    }


def _conversation_fact(
    *,
    fact_id: str,
    subject: str,
    relation: str,
    object_text: str,
    source_text: str,
    source_span: str,
) -> dict:
    return _fact(
        fact_id=fact_id,
        subject=subject,
        relation=relation,
        object_text=object_text,
        source_text=source_text,
        source_span=source_span,
        value_text=None,
        value_number=None,
        value_unit=None,
        asserted_at=None,
        entity_ids=[],
    )


def _valid_conv30_source_aggregation_payload() -> tuple[dict, dict[str, dict]]:
    jon_span = "After losing his job as a banker, Jon decided to start his own business."
    gina_span = "After losing her Door Dash job, Gina decided to start her own business."
    dance_span = "Jon later opened a dance studio."

    facts = [
        _conversation_fact(
            fact_id="ep_conv-30_cat1_e01_f_16",
            subject="Jon",
            relation="lost_job",
            object_text="job as a banker",
            source_text=CONV30_E01_TEXT,
            source_span=jon_span,
        ),
        _conversation_fact(
            fact_id="ep_conv-30_cat1_e01_f_17",
            subject="Jon",
            relation="started_business",
            object_text="own business",
            source_text=CONV30_E01_TEXT,
            source_span=jon_span,
        ),
        _conversation_fact(
            fact_id="ep_conv-30_cat1_e01_f_18",
            subject="Gina",
            relation="lost_job",
            object_text="Door Dash job",
            source_text=CONV30_E01_TEXT,
            source_span=gina_span,
        ),
        _conversation_fact(
            fact_id="ep_conv-30_cat1_e01_f_19",
            subject="Gina",
            relation="started_business",
            object_text="own business",
            source_text=CONV30_E01_TEXT,
            source_span=gina_span,
        ),
        _conversation_fact(
            fact_id="ep_conv-30_cat1_e19_f_01",
            subject="Jon",
            relation="opened_business",
            object_text="dance studio",
            source_text=CONV30_E19_TEXT,
            source_span=dance_span,
        ),
    ]

    payload = {
        "schema": "extraction_substrate",
        "payload_scope": "source_aggregation",
        "source_id": "conv-30_cat1",
        "source_kind": "conversation",
        "episode_ids": ["conv-30_cat1_e01", "conv-30_cat1_e19"],
        "locality_by_episode": {
            "conv-30_cat1_e01": _conversation_locality(
                "conv-30_cat1_e01",
                "Job Loss",
                ["conv-30_cat1_e02"],
            ),
            "conv-30_cat1_e19": _conversation_locality(
                "conv-30_cat1_e19",
                "Later Business Update",
                ["conv-30_cat1_e18"],
            ),
        },
        "atomic_facts": facts,
        "revision_currentness": [],
        "events": [
            {
                "event_id": "event_job_loss_jon",
                "event_type": "job_loss",
                "participants": ["Jon"],
                "object": "job as a banker",
                "time": None,
                "location": None,
                "parameters": [],
                "outcome": None,
                "status": None,
                "support_fact_ids": ["ep_conv-30_cat1_e01_f_16"],
            },
            {
                "event_id": "event_business_start_jon",
                "event_type": "business_start",
                "participants": ["Jon"],
                "object": "own business",
                "time": None,
                "location": None,
                "parameters": [],
                "outcome": None,
                "status": None,
                "support_fact_ids": ["ep_conv-30_cat1_e01_f_17"],
            },
            {
                "event_id": "event_job_loss_gina",
                "event_type": "job_loss",
                "participants": ["Gina"],
                "object": "Door Dash job",
                "time": None,
                "location": None,
                "parameters": [],
                "outcome": None,
                "status": None,
                "support_fact_ids": ["ep_conv-30_cat1_e01_f_18"],
            },
            {
                "event_id": "event_business_start_gina",
                "event_type": "business_start",
                "participants": ["Gina"],
                "object": "own business",
                "time": None,
                "location": None,
                "parameters": [],
                "outcome": None,
                "status": None,
                "support_fact_ids": ["ep_conv-30_cat1_e01_f_19"],
            },
            {
                "event_id": "event_business_journey_jon",
                "event_type": "business_journey",
                "participants": ["Jon"],
                "object": "dance studio",
                "time": None,
                "location": None,
                "parameters": [],
                "outcome": None,
                "status": None,
                "support_fact_ids": [
                    "ep_conv-30_cat1_e01_f_17",
                    "ep_conv-30_cat1_e19_f_01",
                ],
            },
        ],
        "records": [],
        "edges": [
            {
                "edge_id": "edge_conv30_jon_root_cause",
                "edge_type": "causes",
                "from_id": "event_job_loss_jon",
                "to_id": "event_business_start_jon",
                "edge_evidence_text": jon_span,
                "anchor_key": None,
                "anchor_basis_fact_ids": [],
                "support_fact_ids": [
                    "ep_conv-30_cat1_e01_f_16",
                    "ep_conv-30_cat1_e01_f_17",
                ],
            },
            {
                "edge_id": "edge_conv30_gina_root_cause",
                "edge_type": "causes",
                "from_id": "event_job_loss_gina",
                "to_id": "event_business_start_gina",
                "edge_evidence_text": gina_span,
                "anchor_key": None,
                "anchor_basis_fact_ids": [],
                "support_fact_ids": [
                    "ep_conv-30_cat1_e01_f_18",
                    "ep_conv-30_cat1_e01_f_19",
                ],
            },
            {
                "edge_id": "edge_conv30_shared_root",
                "edge_type": "same_anchor",
                "from_id": "event_job_loss_jon",
                "to_id": "event_job_loss_gina",
                "edge_evidence_text": None,
                "anchor_key": "lost job",
                "anchor_basis_fact_ids": [
                    "ep_conv-30_cat1_e01_f_16",
                    "ep_conv-30_cat1_e01_f_18",
                ],
                "support_fact_ids": [
                    "ep_conv-30_cat1_e01_f_16",
                    "ep_conv-30_cat1_e01_f_18",
                ],
            },
        ],
    }

    locality_metadata_by_episode = {
        "conv-30_cat1_e01": _conversation_locality(
            "conv-30_cat1_e01",
            "Job Loss",
            ["conv-30_cat1_e02"],
        ),
        "conv-30_cat1_e19": _conversation_locality(
            "conv-30_cat1_e19",
            "Later Business Update",
            ["conv-30_cat1_e18"],
        ),
    }
    return payload, locality_metadata_by_episode


def _valid_document_source_aggregation_payload() -> tuple[dict, dict[str, dict]]:
    e02_text = "Initial measurement at km 2.3 recorded 780 mm cover depth on 2026-01-18."
    e10_text = "\n".join(
        [
            "On 2026-02-04, the pipe cover depth at km 2.3 was remeasured at 980 mm, superseding the earlier reading.",
            "The pressure test for pipeline segment IS-3 ran at 24 bar for 4 hours and passed with zero leaks.",
            "Permit T-17 was approved on 2026-02-12.",
        ]
    )

    facts = [
        _fact(
            fact_id="ep_DOC-022_e02_f_01",
            subject="pipe cover depth at km 2.3",
            relation="measured_as",
            object_text="780 mm",
            source_text=e02_text,
            source_span=e02_text,
            value_text="780 mm",
            value_number=780.0,
            value_unit="mm",
            asserted_at="2026-01-18",
            entity_ids=["pipe_cover_depth_km_2_3"],
        ),
        _fact(
            fact_id="ep_DOC-022_e10_f_02",
            subject="pipe cover depth at km 2.3",
            relation="measured_as",
            object_text="980 mm",
            source_text=e10_text,
            source_span="On 2026-02-04, the pipe cover depth at km 2.3 was remeasured at 980 mm, superseding the earlier reading.",
            value_text="980 mm",
            value_number=980.0,
            value_unit="mm",
            asserted_at="2026-02-04",
            entity_ids=["pipe_cover_depth_km_2_3"],
        ),
        _fact(
            fact_id="ep_DOC-022_e10_f_06",
            subject="Permit T-17",
            relation="status",
            object_text="approved",
            source_text=e10_text,
            source_span="Permit T-17 was approved on 2026-02-12.",
            value_text="approved",
            asserted_at="2026-02-12",
            entity_ids=["permit_T_17"],
        ),
        _fact(
            fact_id="ep_DOC-022_e10_f_03",
            subject="pressure test for pipeline segment IS-3",
            relation="pressure",
            object_text="24 bar",
            source_text=e10_text,
            source_span="The pressure test for pipeline segment IS-3 ran at 24 bar for 4 hours and passed with zero leaks.",
            value_text="24 bar",
            value_number=24.0,
            value_unit="bar",
            entity_ids=["pressure_test_is_3"],
        ),
        _fact(
            fact_id="ep_DOC-022_e10_f_04",
            subject="pressure test for pipeline segment IS-3",
            relation="duration",
            object_text="4 hours",
            source_text=e10_text,
            source_span="The pressure test for pipeline segment IS-3 ran at 24 bar for 4 hours and passed with zero leaks.",
            value_text="4 hours",
            value_number=4.0,
            value_unit="hours",
            entity_ids=["pressure_test_is_3"],
        ),
        _fact(
            fact_id="ep_DOC-022_e10_f_05",
            subject="pressure test for pipeline segment IS-3",
            relation="outcome",
            object_text="zero leaks",
            source_text=e10_text,
            source_span="The pressure test for pipeline segment IS-3 ran at 24 bar for 4 hours and passed with zero leaks.",
            value_text="zero leaks",
            entity_ids=["pressure_test_is_3"],
        ),
    ]

    payload = {
        "schema": "extraction_substrate",
        "payload_scope": "source_aggregation",
        "source_id": "DOC-022",
        "source_kind": "document",
        "episode_ids": ["DOC-022_e02", "DOC-022_e10"],
        "locality_by_episode": {
            "DOC-022_e02": {
                "source_id": "DOC-022",
                "episode_id": "DOC-022_e02",
                "section_id": "DOC-022_s01",
                "heading": "Initial Measurement",
                "table_id": None,
                "list_id": None,
                "paragraph_cluster_id": "DOC-022_p02",
                "neighbor_episode_ids": ["DOC-022_e03"],
            },
            "DOC-022_e10": {
                "source_id": "DOC-022",
                "episode_id": "DOC-022_e10",
                "section_id": "DOC-022_s03",
                "heading": "Operations Update",
                "table_id": None,
                "list_id": None,
                "paragraph_cluster_id": "DOC-022_p07",
                "neighbor_episode_ids": ["DOC-022_e09", "DOC-022_e11"],
            },
        },
        "atomic_facts": facts,
        "revision_currentness": [
            {
                "revision_id": "rev_DOC-022_cross_episode_001",
                "topic_key": "pipe_cover_depth_km_2_3",
                "old_fact_id": "ep_DOC-022_e02_f_01",
                "new_fact_id": "ep_DOC-022_e10_f_02",
                "link_type": "supersedes",
                "current_fact_id": "ep_DOC-022_e10_f_02",
                "effective_date": "2026-02-04",
                "revision_source_fact_ids": [
                    "ep_DOC-022_e02_f_01",
                    "ep_DOC-022_e10_f_02",
                ],
            }
        ],
        "events": [
            {
                "event_id": "event_DOC-022_source_pressure_test",
                "event_type": "pressure_test",
                "participants": ["pipeline segment IS-3"],
                "object": "pipeline segment IS-3",
                "time": None,
                "location": "IS-3",
                "parameters": [
                    {
                        "name": "pressure",
                        "value_number": 24.0,
                        "value_unit": "bar",
                        "value_text": "24 bar",
                    },
                    {
                        "name": "duration",
                        "value_number": 4.0,
                        "value_unit": "hours",
                        "value_text": "4 hours",
                    },
                ],
                "outcome": "zero leaks",
                "status": "passed",
                "support_fact_ids": [
                    "ep_DOC-022_e10_f_03",
                    "ep_DOC-022_e10_f_04",
                    "ep_DOC-022_e10_f_05",
                ],
            }
        ],
        "records": [
            {
                "record_id": "record_DOC-022_source_permit_T17",
                "record_type": "permit",
                "item_id": "T-17",
                "status": "approved",
                "date": "2026-02-12",
                "qualifier": None,
                "owner": None,
                "source_section": "Operations Update",
                "support_fact_ids": ["ep_DOC-022_e10_f_06"],
            }
        ],
        "edges": [
            {
                "edge_id": "edge_DOC-022_cross_same_anchor",
                "edge_type": "same_anchor",
                "from_id": "ep_DOC-022_e02_f_01",
                "to_id": "ep_DOC-022_e10_f_02",
                "edge_evidence_text": None,
                "anchor_key": "pipe cover depth at km 2.3",
                "anchor_basis_fact_ids": [
                    "ep_DOC-022_e02_f_01",
                    "ep_DOC-022_e10_f_02",
                ],
                "support_fact_ids": [
                    "ep_DOC-022_e02_f_01",
                    "ep_DOC-022_e10_f_02",
                ],
            },
            {
                "edge_id": "edge_DOC-022_cross_record_membership",
                "edge_type": "belongs_to_record",
                "from_id": "ep_DOC-022_e10_f_06",
                "to_id": "record_DOC-022_source_permit_T17",
                "edge_evidence_text": None,
                "anchor_key": None,
                "anchor_basis_fact_ids": [],
                "support_fact_ids": ["ep_DOC-022_e10_f_06"],
            },
        ],
    }

    locality_metadata_by_episode = copy.deepcopy(payload["locality_by_episode"])
    return payload, locality_metadata_by_episode


def _episode_error(payload: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_episode_payload(
            payload,
            source_text=DOC_SOURCE_TEXT,
            locality_metadata=_document_locality_metadata(),
        )


def _source_error(payload: dict, locality_metadata_by_episode: dict[str, dict], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_source_aggregation_payload(
            payload,
            locality_metadata_by_episode=locality_metadata_by_episode,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema", "wrong_schema", "schema"),
        ("source_kind", "document", "source_kind"),
    ],
)
def test_validate_episode_payload_rejects_bad_top_level_values(field, value, match):
    payload = _valid_document_episode_payload()
    payload[field] = value
    _episode_error(payload, match)


@pytest.mark.parametrize("field", ["atomic_facts", "locality", "revision_currentness", "events", "records", "edges"])
def test_validate_episode_payload_requires_all_top_level_fields(field):
    payload = _valid_document_episode_payload()
    del payload[field]
    _episode_error(payload, field)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema", "wrong_schema", "schema"),
        ("payload_scope", "episode", "payload_scope"),
        ("source_kind", "document_episode", "source_kind"),
        ("episode_ids", None, "episode_ids"),
        ("locality_by_episode", None, "locality_by_episode"),
    ],
)
def test_validate_source_aggregation_payload_rejects_bad_top_level_values(field, value, match):
    payload, locality_metadata_by_episode = _valid_conv30_source_aggregation_payload()
    payload[field] = value
    _source_error(payload, locality_metadata_by_episode, match)


def test_validate_source_aggregation_payload_requires_locality_entry_for_each_episode():
    payload, locality_metadata_by_episode = _valid_conv30_source_aggregation_payload()
    del payload["locality_by_episode"]["conv-30_cat1_e19"]
    _source_error(payload, locality_metadata_by_episode, "missing locality object")


def test_validate_episode_payload_accepts_grounded_document_layers():
    payload = _valid_document_episode_payload()
    validated = validate_episode_payload(
        payload,
        source_text=DOC_SOURCE_TEXT,
        locality_metadata=_document_locality_metadata(),
    )
    assert validated["schema"] == "extraction_substrate"
    assert validated["events"][0]["parameters"][0]["value_unit"] == "bar"
    assert validated["records"][0]["source_section"] == "Operations Update"


def test_validate_episode_payload_rejects_duplicate_fact_ids():
    payload = _valid_document_episode_payload()
    payload["atomic_facts"][1]["fact_id"] = payload["atomic_facts"][0]["fact_id"]
    with pytest.raises(ValueError, match="duplicate id"):
        validate_episode_payload(
            payload,
            source_text=DOC_SOURCE_TEXT,
            locality_metadata=_document_locality_metadata(),
        )


def test_validate_episode_payload_rejects_cross_layer_id_collision():
    payload = _valid_document_episode_payload()
    payload["events"][0]["event_id"] = payload["atomic_facts"][0]["fact_id"]
    with pytest.raises(ValueError, match="id collision"):
        validate_episode_payload(
            payload,
            source_text=DOC_SOURCE_TEXT,
            locality_metadata=_document_locality_metadata(),
        )


def test_validate_episode_payload_rejects_revision_reference_to_unknown_fact():
    payload = _valid_document_episode_payload()
    payload["revision_currentness"][0]["new_fact_id"] = "missing_fact"
    with pytest.raises(ValueError, match="unknown id reference"):
        validate_episode_payload(
            payload,
            source_text=DOC_SOURCE_TEXT,
            locality_metadata=_document_locality_metadata(),
        )


def test_validate_episode_payload_rejects_invented_atomic_object():
    payload = _valid_document_episode_payload()
    payload["atomic_facts"][0]["object"] = "920 mm"
    with pytest.raises(ValueError, match="object"):
        validate_episode_payload(
            payload,
            source_text=DOC_SOURCE_TEXT,
            locality_metadata=_document_locality_metadata(),
        )


def test_validate_episode_payload_rejects_invented_locality_heading():
    payload = _valid_document_episode_payload()
    payload["locality"]["heading"] = "Invented Heading"
    with pytest.raises(ValueError, match="locality.heading"):
        validate_episode_payload(
            payload,
            source_text=DOC_SOURCE_TEXT,
            locality_metadata=_document_locality_metadata(),
        )


def test_validate_episode_payload_accepts_numeric_event_parameters_with_units():
    payload = _valid_document_episode_payload()
    validated = validate_episode_payload(
        payload,
        source_text=DOC_SOURCE_TEXT,
        locality_metadata=_document_locality_metadata(),
    )
    assert validated["events"][0]["parameters"] == [
        {
            "name": "pressure",
            "value_number": 24.0,
            "value_unit": "bar",
            "value_text": "24 bar",
        },
        {
            "name": "duration",
            "value_number": 4.0,
            "value_unit": "hours",
            "value_text": "4 hours",
        },
    ]


def test_validate_episode_payload_rejects_event_parameter_unit_mismatch():
    payload = _valid_document_episode_payload()
    payload["events"][0]["parameters"][1]["value_unit"] = "days"
    with pytest.raises(ValueError, match="parameters\\[1\\].value_number/parameters\\[1\\].value_unit"):
        validate_episode_payload(
            payload,
            source_text=DOC_SOURCE_TEXT,
            locality_metadata=_document_locality_metadata(),
        )


def test_validate_episode_payload_rejects_ungrounded_event_participant():
    payload = _valid_document_episode_payload()
    payload["events"][0]["participants"] = ["pipeline segment IS-9"]
    with pytest.raises(ValueError, match="participants\\[0\\]"):
        validate_episode_payload(
            payload,
            source_text=DOC_SOURCE_TEXT,
            locality_metadata=_document_locality_metadata(),
        )


def test_validate_episode_payload_accepts_record_source_section_from_locality_metadata():
    payload = _valid_document_episode_payload()
    validated = validate_episode_payload(
        payload,
        source_text=DOC_SOURCE_TEXT,
        locality_metadata=_document_locality_metadata(),
    )
    assert validated["records"][0]["source_section"] == _document_locality_metadata()["heading"]


def test_validate_episode_payload_rejects_record_source_section_not_in_locality():
    payload = _valid_document_episode_payload()
    payload["records"][0]["source_section"] = "Approvals"
    with pytest.raises(ValueError, match="source_section"):
        validate_episode_payload(
            payload,
            source_text=DOC_SOURCE_TEXT,
            locality_metadata=_document_locality_metadata(),
        )


def test_validate_episode_payload_accepts_grounded_semantic_and_same_anchor_edges():
    payload = _valid_document_episode_payload()
    validated = validate_episode_payload(
        payload,
        source_text=DOC_SOURCE_TEXT,
        locality_metadata=_document_locality_metadata(),
    )
    edge_types = {edge["edge_type"] for edge in validated["edges"]}
    assert {"same_anchor", "resolver_for", "belongs_to_event", "belongs_to_record"} <= edge_types


def test_validate_episode_payload_rejects_semantic_edge_without_grounded_evidence():
    payload = _valid_document_episode_payload()
    payload["edges"][1]["edge_evidence_text"] = "invented causal bridge"
    with pytest.raises(ValueError, match="edge_evidence_text"):
        validate_episode_payload(
            payload,
            source_text=DOC_SOURCE_TEXT,
            locality_metadata=_document_locality_metadata(),
        )


def test_validate_episode_payload_rejects_wrong_direction_for_belongs_to_event():
    payload = _valid_document_episode_payload()
    payload["edges"][2] = {
        "edge_id": "edge_DOC-022_bad_direction",
        "edge_type": "belongs_to_event",
        "from_id": "event_DOC-022_pressure_test_001",
        "to_id": "ep_DOC-022_e10_f_03",
        "edge_evidence_text": None,
        "anchor_key": None,
        "anchor_basis_fact_ids": [],
        "support_fact_ids": ["ep_DOC-022_e10_f_03"],
    }
    with pytest.raises(ValueError, match="belongs_to_event"):
        validate_episode_payload(
            payload,
            source_text=DOC_SOURCE_TEXT,
            locality_metadata=_document_locality_metadata(),
        )


def test_validate_source_aggregation_payload_accepts_conv30_style_root_event_structure():
    payload, locality_metadata_by_episode = _valid_conv30_source_aggregation_payload()
    validated = validate_source_aggregation_payload(
        payload,
        locality_metadata_by_episode=locality_metadata_by_episode,
    )
    assert validated["payload_scope"] == "source_aggregation"
    assert any(edge["edge_type"] == "causes" for edge in validated["edges"])
    assert any(event["event_id"] == "event_business_journey_jon" for event in validated["events"])


def test_validate_source_aggregation_payload_rejects_empty_higher_order_layers():
    payload, locality_metadata_by_episode = _valid_conv30_source_aggregation_payload()
    payload["revision_currentness"] = []
    payload["events"] = []
    payload["records"] = []
    payload["edges"] = []
    with pytest.raises(ValueError, match="expected at least one revision_currentness/event/record/edge object"):
        validate_source_aggregation_payload(
            payload,
            locality_metadata_by_episode=locality_metadata_by_episode,
        )


def test_validate_source_aggregation_payload_rejects_malformed_locality_by_episode():
    payload, locality_metadata_by_episode = _valid_conv30_source_aggregation_payload()
    payload["locality_by_episode"]["conv-30_cat1_e19"] = {"source_id": "conv-30_cat1"}
    with pytest.raises(ValueError, match="locality"):
        validate_source_aggregation_payload(
            payload,
            locality_metadata_by_episode=locality_metadata_by_episode,
        )


def test_validate_source_aggregation_payload_rejects_duplicate_event_ids():
    payload, locality_metadata_by_episode = _valid_conv30_source_aggregation_payload()
    payload["events"][1]["event_id"] = payload["events"][0]["event_id"]
    with pytest.raises(ValueError, match="duplicate id"):
        validate_source_aggregation_payload(
            payload,
            locality_metadata_by_episode=locality_metadata_by_episode,
        )


def test_validate_source_aggregation_payload_rejects_unknown_edge_reference():
    payload, locality_metadata_by_episode = _valid_conv30_source_aggregation_payload()
    payload["edges"][0]["to_id"] = "missing_event"
    with pytest.raises(ValueError, match="unknown id reference"):
        validate_source_aggregation_payload(
            payload,
            locality_metadata_by_episode=locality_metadata_by_episode,
        )


def test_validate_source_aggregation_payload_rejects_ungrounded_semantic_edge_evidence():
    payload, locality_metadata_by_episode = _valid_conv30_source_aggregation_payload()
    payload["edges"][0]["edge_evidence_text"] = "invented causal bridge"
    with pytest.raises(ValueError, match="edge_evidence_text"):
        validate_source_aggregation_payload(
            payload,
            locality_metadata_by_episode=locality_metadata_by_episode,
        )


def test_validate_source_aggregation_payload_rejects_ungrounded_same_anchor_key():
    payload, locality_metadata_by_episode = _valid_conv30_source_aggregation_payload()
    payload["edges"][2]["anchor_key"] = "shared dancing career"
    with pytest.raises(ValueError, match="anchor_key"):
        validate_source_aggregation_payload(
            payload,
            locality_metadata_by_episode=locality_metadata_by_episode,
        )


def test_validate_source_aggregation_payload_rejects_wrong_direction_for_belongs_to_event():
    payload, locality_metadata_by_episode = _valid_conv30_source_aggregation_payload()
    payload["edges"].append(
        {
            "edge_id": "edge_conv30_bad_direction",
            "edge_type": "belongs_to_event",
            "from_id": "event_job_loss_jon",
            "to_id": "ep_conv-30_cat1_e01_f_16",
            "edge_evidence_text": None,
            "anchor_key": None,
            "anchor_basis_fact_ids": [],
            "support_fact_ids": ["ep_conv-30_cat1_e01_f_16"],
        }
    )
    with pytest.raises(ValueError, match="belongs_to_event"):
        validate_source_aggregation_payload(
            payload,
            locality_metadata_by_episode=locality_metadata_by_episode,
        )


def test_compute_source_pipeline_budget_scales_with_episode_count():
    assert compute_source_pipeline_budget(1) == 9
    assert compute_source_pipeline_budget(2) == 15
    assert compute_source_pipeline_budget(3) == 21


def test_compute_source_pipeline_budget_rejects_non_positive_episode_count():
    with pytest.raises(ValueError, match="episode_count"):
        compute_source_pipeline_budget(0)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("source_id", "DOC-999", "locality.source_id"),
        ("episode_id", "DOC-999_e10", "locality.episode_id"),
        ("section_id", "DOC-022_s99", "locality.section_id"),
        ("table_id", "DOC-022_t01", "locality.table_id"),
        ("list_id", "DOC-022_l01", "locality.list_id"),
        ("paragraph_cluster_id", "DOC-022_p99", "locality.paragraph_cluster_id"),
        ("neighbor_episode_ids", ["DOC-022_e01"], "neighbor_episode_ids"),
    ],
)
def test_validate_episode_payload_rejects_ungrounded_locality_fields(field, value, match):
    payload = _valid_document_episode_payload()
    payload["locality"][field] = value
    _episode_error(payload, match)


def test_validate_source_aggregation_payload_rejects_missing_locality_metadata_for_episode():
    payload, locality_metadata_by_episode = _valid_conv30_source_aggregation_payload()
    del locality_metadata_by_episode["conv-30_cat1_e19"]
    _source_error(payload, locality_metadata_by_episode, "locality_metadata_by_episode")


def test_validate_episode_payload_rejects_ungrounded_atomic_source_span():
    payload = _valid_document_episode_payload()
    payload["atomic_facts"][0]["source_span"] = "Invented source span"
    _episode_error(payload, "source_span")


def test_validate_episode_payload_rejects_mismatched_atomic_span_offsets():
    payload = _valid_document_episode_payload()
    payload["atomic_facts"][0]["source_span_start"] = 0
    payload["atomic_facts"][0]["source_span_end"] = 5
    _episode_error(payload, "source_span")


def test_validate_episode_payload_rejects_ungrounded_atomic_asserted_at():
    payload = _valid_document_episode_payload()
    payload["atomic_facts"][0]["asserted_at"] = "2026-01-19"
    _episode_error(payload, "asserted_at")


def test_validate_episode_payload_rejects_ungrounded_atomic_value_text():
    payload = _valid_document_episode_payload()
    payload["atomic_facts"][0]["value_text"] = "781 mm"
    _episode_error(payload, "value_text")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("polarity", "mixed", "polarity"),
        ("confidence", 1.5, "confidence"),
    ],
)
def test_validate_episode_payload_rejects_atomic_schema_errors(field, value, match):
    payload = _valid_document_episode_payload()
    payload["atomic_facts"][0][field] = value
    _episode_error(payload, match)


def test_validate_episode_payload_rejects_duplicate_revision_id():
    payload = _valid_document_episode_payload()
    payload["revision_currentness"].append(copy.deepcopy(payload["revision_currentness"][0]))
    _episode_error(payload, "duplicate id")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("old_fact_id", "missing_fact", "unknown id reference"),
        ("current_fact_id", "missing_fact", "unknown id reference"),
        ("link_type", "replaces", "link_type"),
    ],
)
def test_validate_episode_payload_rejects_revision_schema_errors(field, value, match):
    payload = _valid_document_episode_payload()
    payload["revision_currentness"][0][field] = value
    _episode_error(payload, match)


def test_validate_episode_payload_rejects_unknown_revision_source_fact_id():
    payload = _valid_document_episode_payload()
    payload["revision_currentness"][0]["revision_source_fact_ids"] = ["ep_DOC-022_e10_f_01", "missing_fact"]
    _episode_error(payload, "unknown id reference")


def test_validate_episode_payload_rejects_ungrounded_revision_effective_date():
    payload = _valid_document_episode_payload()
    payload["revision_currentness"][0]["effective_date"] = "2026-02-05"
    _episode_error(payload, "effective_date")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("object", "pipeline segment IS-9", "object"),
        ("time", "2026-01-15", "time"),
        ("location", "IS-9", "location"),
        ("outcome", "minor leaks", "outcome"),
        ("status", "failed", "status"),
    ],
)
def test_validate_episode_payload_rejects_ungrounded_event_fields(field, value, match):
    payload = _valid_document_episode_payload()
    payload["events"][0][field] = value
    _episode_error(payload, match)


def test_validate_episode_payload_rejects_ungrounded_event_parameter_value_text():
    payload = _valid_document_episode_payload()
    payload["events"][0]["parameters"][0]["value_text"] = "28 bar"
    _episode_error(payload, "parameters\\[0\\].value_text")


def test_validate_episode_payload_rejects_duplicate_record_id():
    payload = _valid_document_episode_payload()
    payload["records"].append(copy.deepcopy(payload["records"][0]))
    _episode_error(payload, "duplicate id")


def test_validate_episode_payload_rejects_unknown_record_support_fact_ids():
    payload = _valid_document_episode_payload()
    payload["records"][0]["support_fact_ids"] = ["missing_fact"]
    _episode_error(payload, "unknown id reference")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("item_id", "T-99", "item_id"),
        ("status", "pending", "status"),
        ("date", "2026-02-13", "date"),
        ("qualifier", "emergency", "qualifier"),
        ("owner", "Permit Office", "owner"),
    ],
)
def test_validate_episode_payload_rejects_ungrounded_record_fields(field, value, match):
    payload = _valid_document_episode_payload()
    payload["records"][0][field] = value
    _episode_error(payload, match)


def test_validate_episode_payload_rejects_duplicate_edge_id():
    payload = _valid_document_episode_payload()
    payload["edges"].append(copy.deepcopy(payload["edges"][0]))
    _episode_error(payload, "duplicate id")


def test_validate_episode_payload_rejects_invalid_edge_type():
    payload = _valid_document_episode_payload()
    payload["edges"][0]["edge_type"] = "related_to"
    _episode_error(payload, "edge_type")


def test_validate_episode_payload_rejects_unknown_edge_from_id():
    payload = _valid_document_episode_payload()
    payload["edges"][0]["from_id"] = "missing_node"
    _episode_error(payload, "unknown id reference")


def test_validate_episode_payload_rejects_missing_same_anchor_basis_fact_ids():
    payload = _valid_document_episode_payload()
    payload["edges"][0]["anchor_basis_fact_ids"] = []
    _episode_error(payload, "anchor_basis_fact_ids")


def test_validate_episode_payload_rejects_unknown_same_anchor_basis_fact_id():
    payload = _valid_document_episode_payload()
    payload["edges"][0]["anchor_basis_fact_ids"] = ["missing_fact"]
    _episode_error(payload, "unknown id reference")


def test_validate_episode_payload_rejects_empty_support_on_belongs_to_record():
    payload = _valid_document_episode_payload()
    payload["edges"][3]["support_fact_ids"] = []
    _episode_error(payload, "support_fact_ids")


def test_validate_episode_payload_rejects_wrong_direction_for_belongs_to_record():
    payload = _valid_document_episode_payload()
    payload["edges"][3] = {
        "edge_id": "edge_DOC-022_bad_record_direction",
        "edge_type": "belongs_to_record",
        "from_id": "record_DOC-022_permit_T17",
        "to_id": "ep_DOC-022_e10_f_06",
        "edge_evidence_text": None,
        "anchor_key": None,
        "anchor_basis_fact_ids": [],
        "support_fact_ids": ["ep_DOC-022_e10_f_06"],
    }
    _episode_error(payload, "belongs_to_record")


def test_validate_episode_payload_rejects_fact_record_id_collision():
    payload = _valid_document_episode_payload()
    payload["records"][0]["record_id"] = payload["atomic_facts"][0]["fact_id"]
    _episode_error(payload, "id collision")


def test_validate_episode_payload_rejects_event_record_id_collision():
    payload = _valid_document_episode_payload()
    payload["records"][0]["record_id"] = payload["events"][0]["event_id"]
    _episode_error(payload, "id collision")


def test_validate_source_aggregation_payload_rejects_ungrounded_aggregated_event_object():
    payload, locality_metadata_by_episode = _valid_conv30_source_aggregation_payload()
    payload["events"][4]["object"] = "software startup"
    _source_error(payload, locality_metadata_by_episode, "object")


def test_validate_source_aggregation_payload_accepts_cross_episode_revision_and_record():
    payload, locality_metadata_by_episode = _valid_document_source_aggregation_payload()
    validated = validate_source_aggregation_payload(
        payload,
        locality_metadata_by_episode=locality_metadata_by_episode,
    )
    assert validated["revision_currentness"][0]["old_fact_id"] == "ep_DOC-022_e02_f_01"
    assert validated["records"][0]["record_id"] == "record_DOC-022_source_permit_T17"


def test_validate_source_aggregation_payload_rejects_duplicate_revision_id():
    payload, locality_metadata_by_episode = _valid_document_source_aggregation_payload()
    payload["revision_currentness"].append(copy.deepcopy(payload["revision_currentness"][0]))
    _source_error(payload, locality_metadata_by_episode, "duplicate id")


def test_validate_source_aggregation_payload_rejects_ungrounded_cross_episode_revision_effective_date():
    payload, locality_metadata_by_episode = _valid_document_source_aggregation_payload()
    payload["revision_currentness"][0]["effective_date"] = "2026-02-05"
    _source_error(payload, locality_metadata_by_episode, "effective_date")


def test_validate_source_aggregation_payload_rejects_duplicate_record_id():
    payload, locality_metadata_by_episode = _valid_document_source_aggregation_payload()
    payload["records"].append(copy.deepcopy(payload["records"][0]))
    _source_error(payload, locality_metadata_by_episode, "duplicate id")


def test_validate_source_aggregation_payload_rejects_ungrounded_source_level_record_status():
    payload, locality_metadata_by_episode = _valid_document_source_aggregation_payload()
    payload["records"][0]["status"] = "pending"
    _source_error(payload, locality_metadata_by_episode, "status")


def test_validate_source_aggregation_payload_rejects_wrong_direction_for_belongs_to_record():
    payload, locality_metadata_by_episode = _valid_document_source_aggregation_payload()
    payload["edges"][1] = {
        "edge_id": "edge_DOC-022_bad_source_record_direction",
        "edge_type": "belongs_to_record",
        "from_id": "record_DOC-022_source_permit_T17",
        "to_id": "ep_DOC-022_e10_f_06",
        "edge_evidence_text": None,
        "anchor_key": None,
        "anchor_basis_fact_ids": [],
        "support_fact_ids": ["ep_DOC-022_e10_f_06"],
    }
    _source_error(payload, locality_metadata_by_episode, "belongs_to_record")


def test_validate_source_aggregation_payload_rejects_fact_record_id_collision():
    payload, locality_metadata_by_episode = _valid_document_source_aggregation_payload()
    payload["records"][0]["record_id"] = payload["atomic_facts"][0]["fact_id"]
    _source_error(payload, locality_metadata_by_episode, "id collision")


def test_validate_source_aggregation_payload_rejects_event_record_id_collision():
    payload, locality_metadata_by_episode = _valid_document_source_aggregation_payload()
    payload["events"] = [
        {
            "event_id": "record_DOC-022_source_permit_T17",
            "event_type": "permit_event",
            "participants": [],
            "object": "T-17",
            "time": "2026-02-12",
            "location": None,
            "parameters": [],
            "outcome": None,
            "status": "approved",
            "support_fact_ids": ["ep_DOC-022_e10_f_06"],
        }
    ]
    _source_error(payload, locality_metadata_by_episode, "id collision")


def test_compute_episode_and_aggregation_subtotals_are_locked():
    assert compute_episode_extraction_budget() == 6
    assert compute_source_aggregation_budget() == 3


def test_run_episode_validation_pipeline_marks_failed_after_base_layer_exhaustion():
    invalid = _valid_document_episode_payload()
    invalid["atomic_facts"][0]["source_span"] = "Invented source span"
    result = run_episode_validation_pipeline(
        [invalid, copy.deepcopy(invalid), copy.deepcopy(invalid)],
        source_text=DOC_SOURCE_TEXT,
        locality_metadata=_document_locality_metadata(),
    )
    assert result["extraction_status"] == "failed"
    assert result["repair_attempt_count"] == 2
    assert result["source_pipeline_attempt_count"] == 3
    assert result["failure_reasons"]


def test_run_episode_validation_pipeline_drops_invalid_higher_order_layer_to_empty():
    invalid = _valid_document_episode_payload()
    invalid["events"][0]["participants"] = ["pipeline segment IS-9"]
    invalid["edges"] = []
    result = run_episode_validation_pipeline(
        [invalid, copy.deepcopy(invalid), copy.deepcopy(invalid)],
        source_text=DOC_SOURCE_TEXT,
        locality_metadata=_document_locality_metadata(),
    )
    assert result["extraction_status"] == "partial"
    assert result["dropped_layers"] == ["event_layer"]
    assert result["accepted_layers"] == [
        "atomic_fact_layer",
        "locality_layer",
        "revision_currentness_layer",
        "record_layer",
        "edge_layer",
    ]
    assert result["payload"]["events"] == []
    assert result["failure_reasons"] == ["ungrounded_event", "extraction_retry_exhausted"]


def test_run_source_aggregation_validation_pipeline_drops_invalid_aggregated_layer_to_empty():
    invalid, locality_metadata_by_episode = _valid_document_source_aggregation_payload()
    invalid["records"][0]["status"] = "pending"
    invalid["edges"] = []
    result = run_source_aggregation_validation_pipeline(
        [invalid, copy.deepcopy(invalid), copy.deepcopy(invalid)],
        locality_metadata_by_episode=locality_metadata_by_episode,
        episode_count=2,
    )
    assert result["aggregation_status"] == "partial"
    assert result["dropped_layers"] == ["record_layer"]
    assert result["accepted_layers"] == [
        "atomic_fact_layer",
        "locality_layer",
        "revision_currentness_layer",
        "event_layer",
        "edge_layer",
    ]
    assert result["payload"]["records"] == []
    assert result["source_pipeline_attempt_count"] == compute_source_pipeline_budget(2)
    assert result["failure_reasons"] == ["ungrounded_record", "source_aggregation_retry_exhausted"]


def test_persist_episode_payload_writes_episode_extraction_artifact(tmp_path):
    payload = _valid_document_episode_payload()
    path = persist_episode_payload(tmp_path, payload)
    assert path.name == "episode_extraction.json"
    assert json.loads(path.read_text())["episode_id"] == "DOC-022_e10"


def test_persist_source_aggregation_payload_writes_source_aggregation_artifact(tmp_path):
    payload, _ = _valid_document_source_aggregation_payload()
    path = persist_source_aggregation_payload(tmp_path, payload)
    assert path.name == "source_aggregation.json"
    assert json.loads(path.read_text())["payload_scope"] == "source_aggregation"


@pytest.mark.parametrize("field", ["atomic_facts", "revision_currentness", "events", "records", "edges"])
def test_validate_source_aggregation_payload_requires_all_top_level_collections(field):
    payload, locality_metadata_by_episode = _valid_conv30_source_aggregation_payload()
    del payload[field]
    _source_error(payload, locality_metadata_by_episode, field)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("source_id", "wrong-source", "locality.source_id"),
        ("episode_id", "wrong-episode", "locality.episode_id"),
        ("heading", "Invented Heading", "locality.heading"),
        ("section_id", "SEC-9", "locality.section_id"),
        ("table_id", "TABLE-1", "locality.table_id"),
        ("list_id", "LIST-1", "locality.list_id"),
        ("paragraph_cluster_id", "p999", "locality.paragraph_cluster_id"),
        ("neighbor_episode_ids", ["conv-30_cat1_e99"], "neighbor_episode_ids"),
    ],
)
def test_validate_source_aggregation_payload_rejects_ungrounded_locality_values(field, value, match):
    payload, locality_metadata_by_episode = _valid_conv30_source_aggregation_payload()
    payload["locality_by_episode"]["conv-30_cat1_e19"][field] = value
    _source_error(payload, locality_metadata_by_episode, match)


def test_validate_episode_payload_rejects_atomic_numeric_grounding_mismatch():
    payload = _valid_document_episode_payload()
    payload["atomic_facts"][0]["value_number"] = 781.0
    _episode_error(payload, "value_number/value_unit")


def test_validate_episode_payload_rejects_duplicate_event_id():
    payload = _valid_document_episode_payload()
    payload["events"].append(copy.deepcopy(payload["events"][0]))
    _episode_error(payload, "duplicate id")


def test_validate_episode_payload_rejects_invalid_event_type():
    payload = _valid_document_episode_payload()
    payload["events"][0]["event_type"] = ""
    _episode_error(payload, "event_type")


def test_validate_episode_payload_rejects_unknown_event_support_fact_id():
    payload = _valid_document_episode_payload()
    payload["events"][0]["support_fact_ids"] = ["missing_fact"]
    _episode_error(payload, "unknown id reference")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("participants", ["Nobody"], "participants\\[0\\]"),
        ("time", "2026-01-15", "time"),
        ("location", "IS-9", "location"),
        ("outcome", "major leaks", "outcome"),
        ("status", "failed", "status"),
    ],
)
def test_validate_source_aggregation_payload_rejects_ungrounded_event_fields(field, value, match):
    payload, locality_metadata_by_episode = _valid_document_source_aggregation_payload()
    payload["events"][0][field] = value
    _source_error(payload, locality_metadata_by_episode, match)


def test_validate_source_aggregation_payload_rejects_ungrounded_event_parameter_value_text():
    payload, locality_metadata_by_episode = _valid_document_source_aggregation_payload()
    payload["events"][0]["parameters"][0]["value_text"] = "28 bar"
    _source_error(payload, locality_metadata_by_episode, "parameters\\[0\\].value_text")


def test_validate_source_aggregation_payload_rejects_ungrounded_event_parameter_number_unit():
    payload, locality_metadata_by_episode = _valid_document_source_aggregation_payload()
    payload["events"][0]["parameters"][0]["value_unit"] = "psi"
    _source_error(payload, locality_metadata_by_episode, "parameters\\[0\\].value_number/parameters\\[0\\].value_unit")


def test_validate_source_aggregation_payload_rejects_unknown_event_support_fact_id():
    payload, locality_metadata_by_episode = _valid_document_source_aggregation_payload()
    payload["events"][0]["support_fact_ids"] = ["missing_fact"]
    _source_error(payload, locality_metadata_by_episode, "unknown id reference")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("old_fact_id", "missing_fact", "unknown id reference"),
        ("new_fact_id", "missing_fact", "unknown id reference"),
        ("current_fact_id", "missing_fact", "unknown id reference"),
        ("link_type", "replaces", "link_type"),
    ],
)
def test_validate_source_aggregation_payload_rejects_revision_schema_errors(field, value, match):
    payload, locality_metadata_by_episode = _valid_document_source_aggregation_payload()
    payload["revision_currentness"][0][field] = value
    _source_error(payload, locality_metadata_by_episode, match)


def test_validate_source_aggregation_payload_rejects_unknown_revision_source_fact_id():
    payload, locality_metadata_by_episode = _valid_document_source_aggregation_payload()
    payload["revision_currentness"][0]["revision_source_fact_ids"] = ["ep_DOC-022_e02_f_01", "missing_fact"]
    _source_error(payload, locality_metadata_by_episode, "unknown id reference")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("record_type", "", "record_type"),
        ("item_id", "T-99", "item_id"),
        ("date", "2026-02-13", "date"),
        ("qualifier", "urgent", "qualifier"),
        ("owner", "Permit Office", "owner"),
        ("source_section", "Approvals", "source_section"),
    ],
)
def test_validate_source_aggregation_payload_rejects_ungrounded_record_fields(field, value, match):
    payload, locality_metadata_by_episode = _valid_document_source_aggregation_payload()
    payload["records"][0][field] = value
    _source_error(payload, locality_metadata_by_episode, match)


def test_validate_source_aggregation_payload_rejects_unknown_record_support_fact_id():
    payload, locality_metadata_by_episode = _valid_document_source_aggregation_payload()
    payload["records"][0]["support_fact_ids"] = ["missing_fact"]
    _source_error(payload, locality_metadata_by_episode, "unknown id reference")


def test_validate_source_aggregation_payload_rejects_duplicate_edge_id():
    payload, locality_metadata_by_episode = _valid_document_source_aggregation_payload()
    payload["edges"].append(copy.deepcopy(payload["edges"][0]))
    _source_error(payload, locality_metadata_by_episode, "duplicate id")


def test_validate_source_aggregation_payload_rejects_invalid_edge_type():
    payload, locality_metadata_by_episode = _valid_document_source_aggregation_payload()
    payload["edges"][0]["edge_type"] = "related_to"
    _source_error(payload, locality_metadata_by_episode, "edge_type")


def test_validate_source_aggregation_payload_rejects_unknown_edge_from_id():
    payload, locality_metadata_by_episode = _valid_document_source_aggregation_payload()
    payload["edges"][0]["from_id"] = "missing_node"
    _source_error(payload, locality_metadata_by_episode, "unknown id reference")


def test_validate_source_aggregation_payload_rejects_unknown_edge_to_id():
    payload, locality_metadata_by_episode = _valid_document_source_aggregation_payload()
    payload["edges"][0]["to_id"] = "missing_node"
    _source_error(payload, locality_metadata_by_episode, "unknown id reference")


def test_validate_source_aggregation_payload_rejects_missing_same_anchor_basis_fact_ids():
    payload, locality_metadata_by_episode = _valid_document_source_aggregation_payload()
    payload["edges"][0]["anchor_basis_fact_ids"] = []
    _source_error(payload, locality_metadata_by_episode, "anchor_basis_fact_ids")


def test_validate_source_aggregation_payload_rejects_unknown_same_anchor_basis_fact_id():
    payload, locality_metadata_by_episode = _valid_document_source_aggregation_payload()
    payload["edges"][0]["anchor_basis_fact_ids"] = ["missing_fact"]
    _source_error(payload, locality_metadata_by_episode, "unknown id reference")


def test_validate_source_aggregation_payload_rejects_empty_support_on_structural_edge():
    payload, locality_metadata_by_episode = _valid_document_source_aggregation_payload()
    payload["edges"][1]["support_fact_ids"] = []
    _source_error(payload, locality_metadata_by_episode, "support_fact_ids")


def test_validate_source_aggregation_payload_rejects_duplicate_fact_id():
    payload, locality_metadata_by_episode = _valid_document_source_aggregation_payload()
    payload["atomic_facts"][1]["fact_id"] = payload["atomic_facts"][0]["fact_id"]
    _source_error(payload, locality_metadata_by_episode, "duplicate id")


def test_validate_source_aggregation_payload_rejects_fact_event_id_collision():
    payload, locality_metadata_by_episode = _valid_document_source_aggregation_payload()
    payload["events"][0]["event_id"] = payload["atomic_facts"][0]["fact_id"]
    _source_error(payload, locality_metadata_by_episode, "id collision")


def test_run_episode_validation_pipeline_emits_exact_base_failure_telemetry():
    invalid = _valid_document_episode_payload()
    invalid["atomic_facts"][0]["source_span"] = "Invented source span"
    result = run_episode_validation_pipeline(
        [invalid, copy.deepcopy(invalid), copy.deepcopy(invalid)],
        source_text=DOC_SOURCE_TEXT,
        locality_metadata=_document_locality_metadata(),
    )
    assert result["extraction_status"] == "failed"
    assert result["schema_error_count"] == 0
    assert result["grounding_error_count"] == 1
    assert result["repair_attempt_count"] == 2
    assert result["accepted_layers"] == []
    assert result["dropped_layers"] == []
    assert result["failure_reasons"] == ["ungrounded_atomic_fact", "extraction_retry_exhausted"]


def test_run_source_aggregation_validation_pipeline_emits_exact_retry_exhaustion_telemetry():
    invalid, locality_metadata_by_episode = _valid_document_source_aggregation_payload()
    invalid["records"][0]["status"] = "pending"
    result = run_source_aggregation_validation_pipeline(
        [invalid, copy.deepcopy(invalid), copy.deepcopy(invalid)],
        locality_metadata_by_episode=locality_metadata_by_episode,
        episode_count=2,
    )
    assert result["aggregation_status"] == "partial"
    assert result["aggregation_repair_attempt_count"] == 2
    assert result["schema_error_count"] == 0
    assert result["grounding_error_count"] == 1
    assert "record_layer" in result["dropped_layers"]
    assert "atomic_fact_layer" in result["accepted_layers"]
    assert "source_aggregation_retry_exhausted" in result["failure_reasons"]
    assert "ungrounded_record" in result["failure_reasons"]


def test_persist_episode_payload_round_trips_exact_payload(tmp_path):
    payload = _valid_document_episode_payload()
    path = persist_episode_payload(tmp_path, payload)
    assert json.loads(path.read_text()) == payload


def test_persist_source_aggregation_payload_round_trips_exact_payload(tmp_path):
    payload, _ = _valid_document_source_aggregation_payload()
    path = persist_source_aggregation_payload(tmp_path, payload)
    assert json.loads(path.read_text()) == payload


def test_persist_episode_and_source_payloads_stay_separate(tmp_path):
    episode = _valid_document_episode_payload()
    source, _ = _valid_document_source_aggregation_payload()
    ep_path = persist_episode_payload(tmp_path, episode)
    src_path = persist_source_aggregation_payload(tmp_path, source)
    assert ep_path != src_path
    assert json.loads(ep_path.read_text())["episode_id"] == "DOC-022_e10"
    assert "payload_scope" not in json.loads(ep_path.read_text())
    assert json.loads(src_path.read_text())["payload_scope"] == "source_aggregation"
    assert "episode_id" not in json.loads(src_path.read_text())


def test_persist_episode_payload_rewrite_preserves_exact_episode_shape(tmp_path):
    payload = _valid_document_episode_payload()
    path1 = persist_episode_payload(tmp_path, payload)
    path2 = persist_episode_payload(tmp_path, payload)
    assert path1 == path2
    data = json.loads(path2.read_text())
    assert data == payload
    assert "payload_scope" not in data


def test_persist_source_aggregation_payload_rewrite_preserves_exact_source_shape(tmp_path):
    payload, _ = _valid_document_source_aggregation_payload()
    path1 = persist_source_aggregation_payload(tmp_path, payload)
    path2 = persist_source_aggregation_payload(tmp_path, payload)
    assert path1 == path2
    data = json.loads(path2.read_text())
    assert data == payload
    assert data["payload_scope"] == "source_aggregation"
    assert "episode_id" not in data


def test_run_episode_validation_pipeline_nulls_non_key_event_field_when_allowed():
    payload = _valid_document_episode_payload()
    payload["events"][0]["location"] = "IS-9"
    payload["edges"] = []
    result = run_episode_validation_pipeline(
        [payload],
        source_text=DOC_SOURCE_TEXT,
        locality_metadata=_document_locality_metadata(),
    )
    assert result["extraction_status"] == "accepted"
    assert result["payload"]["events"][0]["location"] is None


def test_run_episode_validation_pipeline_drops_event_when_key_field_is_ungrounded():
    payload = _valid_document_episode_payload()
    payload["events"][0]["participants"] = ["pipeline segment IS-9"]
    payload["edges"] = []
    result = run_episode_validation_pipeline(
        [payload, copy.deepcopy(payload), copy.deepcopy(payload)],
        source_text=DOC_SOURCE_TEXT,
        locality_metadata=_document_locality_metadata(),
    )
    assert result["extraction_status"] == "partial"
    assert result["payload"]["events"] == []


def test_run_source_aggregation_validation_pipeline_nulls_non_key_record_field_when_allowed():
    payload, locality_metadata_by_episode = _valid_document_source_aggregation_payload()
    payload["records"][0]["owner"] = "Permit Office"
    payload["edges"] = []
    result = run_source_aggregation_validation_pipeline(
        [payload],
        locality_metadata_by_episode=locality_metadata_by_episode,
        episode_count=2,
    )
    assert result["aggregation_status"] == "accepted"
    assert result["payload"]["records"][0]["owner"] is None


def test_run_source_aggregation_validation_pipeline_coerces_malformed_id_ref_lists():
    payload, locality_metadata_by_episode = _valid_document_source_aggregation_payload()
    payload["revision_currentness"][0]["revision_source_fact_ids"] = [
        {"fact_id": "ep_DOC-022_e02_f_01"},
        "ep_DOC-022_e10_f_02",
        {"junk": "ignore-me"},
    ]
    payload["events"][0]["support_fact_ids"] = [
        {"fact_id": "ep_DOC-022_e10_f_03"},
        "ep_DOC-022_e10_f_04",
        {"fact_id": "ep_DOC-022_e10_f_05"},
        {"junk": "ignore-me"},
    ]
    payload["records"][0]["support_fact_ids"] = [
        {"fact_id": "ep_DOC-022_e10_f_06"},
        {"junk": "ignore-me"},
    ]
    payload["edges"][0]["support_fact_ids"] = [
        {"fact_id": "ep_DOC-022_e02_f_01"},
        {"fact_id": "ep_DOC-022_e10_f_02"},
    ]
    payload["edges"][0]["anchor_basis_fact_ids"] = [
        {"fact_id": "ep_DOC-022_e02_f_01"},
        "ep_DOC-022_e10_f_02",
    ]

    result = run_source_aggregation_validation_pipeline(
        [payload],
        locality_metadata_by_episode=locality_metadata_by_episode,
        episode_count=2,
    )

    assert result["aggregation_status"] == "accepted"
    assert result["payload"]["revision_currentness"][0]["revision_source_fact_ids"] == [
        "ep_DOC-022_e02_f_01",
        "ep_DOC-022_e10_f_02",
    ]
    assert result["payload"]["events"][0]["support_fact_ids"] == [
        "ep_DOC-022_e10_f_03",
        "ep_DOC-022_e10_f_04",
        "ep_DOC-022_e10_f_05",
    ]
    assert result["payload"]["records"][0]["support_fact_ids"] == ["ep_DOC-022_e10_f_06"]
    assert result["payload"]["edges"][0]["support_fact_ids"] == [
        "ep_DOC-022_e02_f_01",
        "ep_DOC-022_e10_f_02",
    ]
    assert result["payload"]["edges"][0]["anchor_basis_fact_ids"] == [
        "ep_DOC-022_e02_f_01",
        "ep_DOC-022_e10_f_02",
    ]


def test_run_source_aggregation_validation_pipeline_drops_record_when_key_field_is_ungrounded():
    payload, locality_metadata_by_episode = _valid_document_source_aggregation_payload()
    payload["records"][0]["item_id"] = "T-99"
    payload["edges"] = []
    result = run_source_aggregation_validation_pipeline(
        [payload, copy.deepcopy(payload), copy.deepcopy(payload)],
        locality_metadata_by_episode=locality_metadata_by_episode,
        episode_count=2,
    )
    assert result["aggregation_status"] == "partial"
    assert result["payload"]["records"] == []
