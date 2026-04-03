"""MAL atom validation tests — SPEC Section 2 + Section 20.

Covers:
- all valid atom types validate
- invalid cross-stage atoms rejected
- single-field degenerate atoms accepted
- window_bundle excludes max_episodes_default in v1
"""

import pytest


@pytest.fixture
def validator():
    from src.mal.atom import AtomValidator
    return AtomValidator()


# ── valid retrieval-only atoms ──


def test_lexical_signal_bundle_valid(validator):
    atom = {
        "atom_type": "lexical_signal_bundle",
        "atom_payload": {
            "word_overlap_bonus": {"old": 0.45, "new": 0.65},
            "entity_phrase_bonus": {"old": 2.0, "new": 3.0},
        },
    }
    assert validator.validate(atom) is True


def test_locality_bundle_valid(validator):
    atom = {
        "atom_type": "locality_bundle",
        "atom_payload": {
            "currentness_bonus": {"old": 0.8, "new": 1.2},
            "generic_penalty": {"old": 1.2, "new": 1.5},
        },
    }
    assert validator.validate(atom) is True


def test_window_bundle_valid_v1(validator):
    atom = {
        "atom_type": "window_bundle",
        "atom_payload": {
            "supporting_facts_per_episode": {"old": 3, "new": 5},
            "supporting_facts_total": {"old": 10, "new": 15},
            "budget": {"old": 8000, "new": 10000},
        },
    }
    assert validator.validate(atom) is True


def test_fusion_bundle_valid(validator):
    atom = {
        "atom_type": "fusion_bundle",
        "atom_payload": {
            "late_fusion_per_family": {"old": 3, "new": 4},
            "max_sources_per_family": {"old": 2, "new": 3},
            "rrf_k": {"old": 60, "new": 50},
        },
    }
    assert validator.validate(atom) is True


# ── valid reprocessing atoms ──


def test_grouping_bundle_valid(validator):
    atom = {
        "atom_type": "grouping_bundle",
        "atom_payload": {
            "grouping_prompt_mode": {"old": "strict_small", "new": "strict_partition"},
            "size_cap_chars": {"old": 12000, "new": 8000},
        },
    }
    assert validator.validate(atom) is True


# ── valid extraction atoms ──


def test_extraction_example_append_conversation_prompt_valid(validator):
    atom = {
        "atom_type": "extraction_example_append",
        "atom_payload": {
            "prompt_target": "conversation_content_type:technical",
            "example": "- Decision: switched from PostgreSQL to ScyllaDB",
        },
    }
    assert validator.validate(atom) is True


def test_extraction_example_append_document_block_prompt_valid(validator):
    atom = {
        "atom_type": "extraction_example_append",
        "atom_payload": {
            "prompt_target": "document_block_prompt:prose_block",
            "example": "- Measurement: pipe cover depth 980 mm at km 2.3",
        },
    }
    assert validator.validate(atom) is True


def test_extraction_example_append_source_aggregation_prompt_valid(validator):
    atom = {
        "atom_type": "extraction_example_append",
        "atom_payload": {
            "prompt_target": "document_source_aggregation_prompt:unified_source_aggregation",
            "example": "- Revision: depth changed from 780mm to 980mm",
        },
    }
    assert validator.validate(atom) is True


def test_extraction_example_append_invalid_prompt_target_rejected(validator):
    atom = {
        "atom_type": "extraction_example_append",
        "atom_payload": {
            "prompt_target": "nonexistent_plane:something",
            "example": "example text",
        },
    }
    with pytest.raises(ValueError):
        validator.validate(atom)


# ── single-field degenerate atoms ──


def test_single_field_lexical_atom_valid(validator):
    atom = {
        "atom_type": "lexical_signal_bundle",
        "atom_payload": {
            "entity_phrase_bonus": {"old": 2.0, "new": 3.5},
        },
    }
    assert validator.validate(atom) is True


def test_single_field_window_atom_valid(validator):
    atom = {
        "atom_type": "window_bundle",
        "atom_payload": {
            "budget": {"old": 8000, "new": 10000},
        },
    }
    assert validator.validate(atom) is True


# ── invalid cross-stage atoms ──


def test_cross_stage_lexical_and_grouping_rejected(validator):
    atom = {
        "atom_type": "lexical_signal_bundle",
        "atom_payload": {
            "word_overlap_bonus": {"old": 0.45, "new": 0.65},
            "grouping_prompt_mode": {"old": "strict_small", "new": "baseline"},
        },
    }
    with pytest.raises(ValueError, match="invalid.*field"):
        validator.validate(atom)


def test_cross_stage_grouping_and_extraction_rejected(validator):
    atom = {
        "atom_type": "grouping_bundle",
        "atom_payload": {
            "grouping_prompt_mode": {"old": "strict_small", "new": "baseline"},
            "extraction_prompt": {"old": "x", "new": "y"},
        },
    }
    with pytest.raises(ValueError, match="invalid.*field"):
        validator.validate(atom)


def test_cross_bundle_lexical_and_window_rejected(validator):
    atom = {
        "atom_type": "lexical_signal_bundle",
        "atom_payload": {
            "entity_phrase_bonus": {"old": 2.0, "new": 3.0},
            "supporting_facts_total": {"old": 10, "new": 15},
        },
    }
    with pytest.raises(ValueError, match="invalid.*field"):
        validator.validate(atom)


# ── excluded parameters ──


def test_km_bonus_always_excluded(validator):
    atom = {
        "atom_type": "lexical_signal_bundle",
        "atom_payload": {
            "km_bonus": {"old": 0.0, "new": 1.0},
        },
    }
    with pytest.raises(ValueError, match="excluded"):
        validator.validate(atom)


def test_code_bonus_always_excluded(validator):
    atom = {
        "atom_type": "lexical_signal_bundle",
        "atom_payload": {
            "code_bonus": {"old": 0.0, "new": 1.0},
        },
    }
    with pytest.raises(ValueError, match="excluded"):
        validator.validate(atom)


def test_step_bonus_always_excluded(validator):
    atom = {
        "atom_type": "lexical_signal_bundle",
        "atom_payload": {
            "step_bonus": {"old": 3.5, "new": 5.0},
        },
    }
    with pytest.raises(ValueError, match="excluded"):
        validator.validate(atom)


def test_max_episodes_default_excluded_in_v1(validator):
    atom = {
        "atom_type": "window_bundle",
        "atom_payload": {
            "max_episodes_default": {"old": 3, "new": 5},
        },
    }
    with pytest.raises(ValueError, match="excluded"):
        validator.validate(atom)


# ── bounds enforcement ──


def test_numeric_field_below_lower_bound_rejected(validator):
    atom = {
        "atom_type": "lexical_signal_bundle",
        "atom_payload": {
            "word_overlap_bonus": {"old": 0.45, "new": 0.1},
        },
    }
    with pytest.raises(ValueError, match="bounds"):
        validator.validate(atom)


def test_numeric_field_above_upper_bound_rejected(validator):
    atom = {
        "atom_type": "lexical_signal_bundle",
        "atom_payload": {
            "word_overlap_bonus": {"old": 0.45, "new": 5.0},
        },
    }
    with pytest.raises(ValueError, match="bounds"):
        validator.validate(atom)


# ── grouping constraints ──


def test_invalid_grouping_prompt_mode_rejected(validator):
    atom = {
        "atom_type": "grouping_bundle",
        "atom_payload": {
            "grouping_prompt_mode": {"old": "strict_small", "new": "invalid_mode"},
        },
    }
    with pytest.raises(ValueError):
        validator.validate(atom)


def test_size_cap_below_minimum_rejected(validator):
    atom = {
        "atom_type": "grouping_bundle",
        "atom_payload": {
            "size_cap_chars": {"old": 12000, "new": 1000},
        },
    }
    with pytest.raises(ValueError, match="bounds"):
        validator.validate(atom)


def test_size_cap_above_maximum_rejected(validator):
    atom = {
        "atom_type": "grouping_bundle",
        "atom_payload": {
            "size_cap_chars": {"old": 12000, "new": 100000},
        },
    }
    with pytest.raises(ValueError, match="bounds"):
        validator.validate(atom)


# ── extraction constraints ──


def test_multiple_extraction_examples_in_one_atom_rejected(validator):
    atom = {
        "atom_type": "extraction_example_append",
        "atom_payload": {
            "prompt_target": "conversation_content_type:technical",
            "example": "first example",
            "example2": "second example",
        },
    }
    with pytest.raises(ValueError):
        validator.validate(atom)


def test_unknown_atom_type_rejected(validator):
    atom = {
        "atom_type": "nonexistent_bundle",
        "atom_payload": {"foo": {"old": 1, "new": 2}},
    }
    with pytest.raises(ValueError, match="unknown.*atom_type"):
        validator.validate(atom)


def test_empty_payload_rejected(validator):
    atom = {
        "atom_type": "lexical_signal_bundle",
        "atom_payload": {},
    }
    with pytest.raises(ValueError, match="empty"):
        validator.validate(atom)


# ── inference leaf toggle ──


def test_inference_leaf_toggle_valid(validator):
    atom = {
        "atom_type": "inference_leaf_toggle",
        "atom_payload": {
            "plugin_name": "list_set",
            "enabled": False,
        },
    }
    assert validator.validate(atom) is True


def test_inference_leaf_toggle_enable(validator):
    atom = {
        "atom_type": "inference_leaf_toggle",
        "atom_payload": {
            "plugin_name": "compositional",
            "enabled": True,
        },
    }
    assert validator.validate(atom) is True


def test_inference_leaf_toggle_unknown_plugin_rejected(validator):
    atom = {
        "atom_type": "inference_leaf_toggle",
        "atom_payload": {
            "plugin_name": "nonexistent_plugin",
            "enabled": True,
        },
    }
    with pytest.raises(ValueError, match="unknown inference leaf"):
        validator.validate(atom)


def test_inference_leaf_toggle_non_boolean_rejected(validator):
    atom = {
        "atom_type": "inference_leaf_toggle",
        "atom_payload": {
            "plugin_name": "list_set",
            "enabled": "yes",
        },
    }
    with pytest.raises(ValueError, match="boolean"):
        validator.validate(atom)


def test_inference_leaf_toggle_extra_fields_rejected(validator):
    atom = {
        "atom_type": "inference_leaf_toggle",
        "atom_payload": {
            "plugin_name": "list_set",
            "enabled": False,
            "priority": 100,
        },
    }
    with pytest.raises(ValueError):
        validator.validate(atom)
