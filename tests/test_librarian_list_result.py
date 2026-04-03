"""Regression: LLM returning bare JSON array instead of {"facts": [...]}."""

import asyncio

import pytest

from src.librarian import consolidate_session, cross_session_entity, extract_session


async def _mock_call_extract_list(model, system, user_msg, max_tokens=8192):
    """Simulates LLM returning bare list instead of {"facts": [...]}."""
    return [
        {"id": "f_01", "fact": "Alice likes dogs", "kind": "event",
         "entities": ["Alice"], "tags": ["pets"]},
        {"id": "f_02", "fact": "Bob likes cats", "kind": "event",
         "entities": ["Bob"], "tags": ["pets"]},
    ]


async def _mock_call_extract_dict(model, system, user_msg, max_tokens=8192):
    """Normal LLM response: {"facts": [...]}."""
    return {
        "facts": [
            {"id": "f_01", "fact": "Alice likes dogs", "kind": "event",
             "entities": ["Alice"], "tags": ["pets"]},
        ],
        "temporal_links": [{"before": "f_01", "after": "f_02", "signal": "then"}],
    }


async def _mock_call_extract_empty(model, system, user_msg, max_tokens=8192):
    """LLM returns empty dict."""
    return {}


async def _mock_call_extract_none(model, system, user_msg, max_tokens=8192):
    """LLM returns None (shouldn't happen but defensive)."""
    return None


async def _mock_call_extract_wrong_session(model, system, user_msg, max_tokens=8192):
    """LLM returns bogus session numbers that caller must override."""
    return {
        "facts": [
            {"id": "f_01", "fact": "Alice likes dogs", "session": 1969},
            {"id": "f_02", "fact": "Bob likes cats", "session": 17},
        ]
    }


async def _mock_call_extract_lineage_shorthand(model, system, user_msg, max_tokens=8192):
    """LLM returns shorthand source_ids and bogus cross-session numbers."""
    return {
        "facts": [
            {
                "id": "cf_01",
                "fact": "Merged statement",
                "source_ids": ["f_01", "f_02"],
                "session": 99,
                "sessions": [17, 1969],
            }
        ]
    }


async def _mock_call_extract_partial_consolidation(model, system, user_msg, max_tokens=8192):
    """LLM consolidates some facts but drops one unique source fact."""
    return {
        "facts": [
            {
                "id": "cf_01",
                "fact": "Alice and Ben plan to attend the robotics workshop and meet at Harbor Cafe.",
                "source_ids": ["s16_f_01", "s16_f_02"],
                "entities": ["Alice", "Ben", "robotics workshop", "Harbor Cafe"],
                "tags": ["future_plan"],
            }
        ]
    }


async def _mock_call_extract_partial_cross(model, system, user_msg, max_tokens=8192):
    """LLM synthesizes some entity facts but drops one distinct source fact."""
    return {
        "facts": [
            {
                "id": "xf_01",
                "fact": "Alice and Ben planned to attend the robotics workshop and meet at Harbor Cafe.",
                "source_ids": ["s1_f_01", "s2_f_02"],
                "sessions": [1, 2],
                "entities": ["Alice", "Ben", "robotics workshop", "Harbor Cafe"],
                "tags": ["future_plan"],
            }
        ]
    }


# ── extract_session ──

def test_extract_session_list_result():
    """extract_session handles bare list from LLM without crash."""
    _, sn, _, facts, tlinks = asyncio.run(extract_session(
        session_text="Alice likes dogs. Bob likes cats.",
        session_num=1, session_date="2024-06-01", conv_id="test",
        speakers="User and Assistant", model="test-model",
        call_extract_fn=_mock_call_extract_list,
    ))
    assert len(facts) == 2
    assert facts[0]["fact"] == "Alice likes dogs"
    assert tlinks == []  # no temporal_links from bare list


def test_extract_session_dict_result():
    """extract_session handles normal dict response."""
    _, _, _, facts, tlinks = asyncio.run(extract_session(
        session_text="Alice likes dogs.",
        session_num=1, session_date="2024-06-01", conv_id="test",
        speakers="User and Assistant", model="test-model",
        call_extract_fn=_mock_call_extract_dict,
    ))
    assert len(facts) == 1
    assert len(tlinks) == 1


def test_extract_session_forces_caller_session():
    """LLM-provided session values must never override caller session_num."""
    _, _, _, facts, _ = asyncio.run(extract_session(
        session_text="Alice likes dogs.",
        session_num=1, session_date="2024-06-01", conv_id="test",
        speakers="User and Assistant", model="test-model",
        call_extract_fn=_mock_call_extract_wrong_session,
    ))
    assert [f["session"] for f in facts] == [1, 1]
    assert facts[0]["speaker"] is None
    assert facts[0]["speaker_role"] is None
    assert facts[0]["kind"] == "fact"


def test_extract_session_empty_result():
    """extract_session handles empty dict."""
    _, _, _, facts, tlinks = asyncio.run(extract_session(
        session_text="Nothing here.",
        session_num=1, session_date="2024-06-01", conv_id="test",
        speakers="User and Assistant", model="test-model",
        call_extract_fn=_mock_call_extract_empty,
    ))
    assert facts == []
    assert tlinks == []


def test_extract_session_none_result():
    """extract_session handles None result without crash."""
    _, _, _, facts, tlinks = asyncio.run(extract_session(
        session_text="Nothing.",
        session_num=1, session_date="2024-06-01", conv_id="test",
        speakers="User and Assistant", model="test-model",
        call_extract_fn=_mock_call_extract_none,
    ))
    assert facts == []


# ── consolidate_session ──

def test_consolidate_session_list_result():
    """consolidate_session handles bare list from LLM."""
    session_facts = [
        {"id": "f_01", "fact": "Alice likes dogs", "session_date": "2024-06-01"},
        {"id": "f_02", "fact": "Bob likes cats", "session_date": "2024-06-01"},
    ]
    _, sn, _, cfacts = asyncio.run(consolidate_session(
        conv_id="test", sn=1, session_facts=session_facts,
        speakers="User and Assistant", model="test-model",
        call_extract_fn=_mock_call_extract_list,
    ))
    assert len(cfacts) == 2
    assert cfacts[0]["fact"] == "Alice likes dogs"


def test_consolidate_session_empty_result():
    """consolidate_session handles empty dict."""
    session_facts = [{"id": "f_01", "fact": "X", "session_date": "2024-06-01"}]
    _, _, _, cfacts = asyncio.run(consolidate_session(
        conv_id="test", sn=1, session_facts=session_facts,
        speakers="User and Assistant", model="test-model",
        call_extract_fn=_mock_call_extract_empty,
    ))
    assert cfacts == []


def test_consolidate_session_forces_caller_session():
    """Per-session consolidation must keep the caller session, not model guesses."""
    session_facts = [{"id": "f_01", "fact": "X", "session_date": "2024-06-01"}]
    _, _, _, cfacts = asyncio.run(consolidate_session(
        conv_id="test", sn=3, session_facts=session_facts,
        speakers="User and Assistant", model="test-model",
        call_extract_fn=_mock_call_extract_wrong_session,
    ))
    assert [f["session"] for f in cfacts] == [3, 3]


def test_consolidate_session_normalizes_source_ids_to_available_ids():
    """Consolidation lineage should point at canonical session-scoped source IDs."""
    session_facts = [
        {"id": "s3_f_01", "fact": "Alice likes dogs", "session_date": "2024-06-01"},
        {"id": "s3_f_02", "fact": "Bob likes cats", "session_date": "2024-06-01"},
    ]
    _, _, _, cfacts = asyncio.run(consolidate_session(
        conv_id="test", sn=3, session_facts=session_facts,
        speakers="User and Assistant", model="test-model",
        call_extract_fn=_mock_call_extract_lineage_shorthand,
    ))
    assert cfacts[0]["source_ids"] == ["s3_f_01", "s3_f_02"]
    assert cfacts[0]["session"] == 3


def test_consolidate_session_preserves_uncovered_source_fact_as_fallback():
    """Distinct source facts dropped by the LLM must survive consolidation."""
    session_facts = [
        {
            "id": "s16_f_01",
            "fact": "Alice and Ben planned to attend the robotics workshop.",
            "entities": ["Alice", "Ben", "robotics workshop"],
            "tags": ["future_plan"],
            "session_date": "2024-06-01",
        },
        {
            "id": "s16_f_02",
            "fact": "Alice and Ben will meet at Harbor Cafe tomorrow.",
            "entities": ["Alice", "Ben", "Harbor Cafe"],
            "tags": ["future_plan"],
            "session_date": "2024-06-01",
        },
        {
            "id": "s16_f_11",
            "fact": "Ben and Carla are going to the jazz concert next Sunday.",
            "entities": ["Ben", "Carla", "jazz concert"],
            "tags": ["future_plan"],
            "session_date": "2024-06-01",
        },
    ]
    _, _, _, cfacts = asyncio.run(consolidate_session(
        conv_id="test", sn=16, session_facts=session_facts,
        speakers="Alice and Ben", model="test-model",
        call_extract_fn=_mock_call_extract_partial_consolidation,
    ))
    assert any(f.get("source_ids") == ["s16_f_11"] for f in cfacts)
    baseball = next(f for f in cfacts if f.get("source_ids") == ["s16_f_11"])
    assert "jazz concert" in baseball["fact"]
    assert baseball["session"] == 16


# ── cross_session_entity ──

def test_cross_session_entity_list_result():
    """cross_session_entity handles bare list from LLM."""
    efacts = [
        {"id": "f_01", "fact": "Alice S1", "session": 1},
        {"id": "f_02", "fact": "Alice S2", "session": 2},
    ]
    _, ename, xfacts = asyncio.run(cross_session_entity(
        conv_id="test", ename="alice", efacts=efacts,
        sessions=[1, 2], model="test-model",
        call_extract_fn=_mock_call_extract_list,
    ))
    assert len(xfacts) == 2


def test_cross_session_entity_derives_sessions_from_canonical_sources():
    """Cross-session lineage should canonicalize source_ids and derive sessions."""
    efacts = [
        {"id": "s1_f_01", "fact": "Alice S1", "session": 1},
        {"id": "s2_f_02", "fact": "Alice S2", "session": 2},
    ]
    _, _, xfacts = asyncio.run(cross_session_entity(
        conv_id="test", ename="alice", efacts=efacts,
        sessions=[1, 2], model="test-model",
        call_extract_fn=_mock_call_extract_lineage_shorthand,
    ))
    assert xfacts[0]["source_ids"] == ["s1_f_01", "s2_f_02"]
    assert xfacts[0]["sessions"] == [1, 2]


def test_cross_session_entity_preserves_uncovered_source_fact_as_fallback():
    """Distinct entity facts dropped by cross-session synthesis must survive."""
    efacts = [
        {
            "id": "s1_f_01",
            "fact": "Alice and Ben planned to attend the robotics workshop next Saturday.",
            "session": 1,
            "entities": ["Alice", "Ben", "robotics workshop"],
            "tags": ["future_plan"],
        },
        {
            "id": "s2_f_02",
            "fact": "Alice will meet Ben at Harbor Cafe tomorrow.",
            "session": 2,
            "entities": ["Alice", "Ben", "Harbor Cafe"],
            "tags": ["future_plan"],
        },
        {
            "id": "s16_f_11",
            "fact": "Ben and Carla are going to the jazz concert next Sunday.",
            "session": 16,
            "entities": ["Ben", "Carla", "jazz concert"],
            "tags": ["future_plan"],
        },
    ]
    _, _, xfacts = asyncio.run(cross_session_entity(
        conv_id="test", ename="ben", efacts=efacts,
        sessions=[1, 2, 16], model="test-model",
        call_extract_fn=_mock_call_extract_partial_cross,
    ))
    assert any(f.get("source_ids") == ["s16_f_11"] for f in xfacts)
    baseball = next(f for f in xfacts if f.get("source_ids") == ["s16_f_11"])
    assert "jazz concert" in baseball["fact"]
    assert baseball["sessions"] == [16]


async def _mock_call_extract_mixed(model, system, user_msg, max_tokens=8192):
    """LLM returns mix of dicts and strings (Groq instability)."""
    return [
        {"id": "f_01", "fact": "Valid fact", "kind": "event", "entities": [], "tags": []},
        "This is a string not a dict",
        42,
        {"id": "f_02", "fact": "Another valid", "kind": "event", "entities": [], "tags": []},
    ]


async def _mock_call_extract_dict_with_strings(model, system, user_msg, max_tokens=8192):
    """LLM returns {"facts": [dict, string, dict]}."""
    return {"facts": [
        {"id": "f_01", "fact": "Good fact"},
        "bad string fact",
        {"id": "f_02", "fact": "Good fact 2"},
    ]}


def test_extract_session_filters_non_dict_elements():
    """String/int elements in facts list are silently dropped."""
    _, _, _, facts, _ = asyncio.run(extract_session(
        session_text="Test", session_num=1, session_date="2024-06-01",
        conv_id="test", speakers="User and Assistant", model="test-model",
        call_extract_fn=_mock_call_extract_mixed,
    ))
    assert len(facts) == 2
    assert all(isinstance(f, dict) for f in facts)


def test_consolidate_session_filters_non_dict():
    """consolidate_session drops non-dict elements from facts list."""
    session_facts = [{"id": "f_01", "fact": "X", "session_date": "2024-06-01"}]
    _, _, _, cfacts = asyncio.run(consolidate_session(
        conv_id="test", sn=1, session_facts=session_facts,
        speakers="User and Assistant", model="test-model",
        call_extract_fn=_mock_call_extract_mixed,
    ))
    assert len(cfacts) == 2
    assert all(isinstance(f, dict) for f in cfacts)


def test_cross_session_entity_filters_non_dict():
    """cross_session_entity drops non-dict elements."""
    efacts = [
        {"id": "f_01", "fact": "Alice S1", "session": 1},
        {"id": "f_02", "fact": "Alice S2", "session": 2},
    ]
    _, _, xfacts = asyncio.run(cross_session_entity(
        conv_id="test", ename="alice", efacts=efacts,
        sessions=[1, 2], model="test-model",
        call_extract_fn=_mock_call_extract_mixed,
    ))
    assert len(xfacts) == 2
    assert all(isinstance(f, dict) for f in xfacts)


def test_extract_session_dict_with_string_elements():
    """{"facts": [dict, string, dict]} → only dicts kept."""
    _, _, _, facts, _ = asyncio.run(extract_session(
        session_text="Test", session_num=1, session_date="2024-06-01",
        conv_id="test", speakers="User and Assistant", model="test-model",
        call_extract_fn=_mock_call_extract_dict_with_strings,
    ))
    assert len(facts) == 2
    assert all(isinstance(f, dict) for f in facts)


def test_cross_session_entity_none_result():
    """cross_session_entity handles None result."""
    efacts = [
        {"id": "f_01", "fact": "Alice S1", "session": 1},
        {"id": "f_02", "fact": "Alice S2", "session": 2},
    ]
    _, ename, xfacts = asyncio.run(cross_session_entity(
        conv_id="test", ename="alice", efacts=efacts,
        sessions=[1, 2], model="test-model",
        call_extract_fn=_mock_call_extract_none,
    ))
    assert xfacts == []
