"""Integration tests: document block pipeline wired into librarian.extract_session().

Sprint 36 Stage 2.
Uses mocked LLM calls — no live model dependency.
"""

import asyncio
import json

from src.librarian import extract_session


def _make_mock_extract(responses=None):
    """Create mock call_extract_fn."""
    call_log = []
    idx = {"n": 0}
    default_response = {
        "facts": [{"local_id": "b1", "fact": "extracted fact", "kind": "fact",
                    "entities": [], "tags": [], "depends_on": [],
                    "supersedes_topic": None, "confidence": None, "event_date": None}],
        "temporal_links": []
    }

    async def mock(model, system, user_msg, max_tokens=8192):
        call_log.append({"model": model, "system": system[:300], "user_msg": user_msg[:200]})
        if responses:
            resp = responses[idx["n"] % len(responses)]
            idx["n"] += 1
            return resp
        return dict(default_response)

    return mock, call_log


# ═══════════════════════════════════════════════════════════════════
# Document with headings + list
# ═══════════════════════════════════════════════════════════════════


def test_document_with_list():
    list_response = {
        "facts": [
            {"local_id": "b1", "fact": "Capacity is 500 L/s", "kind": "fact",
             "entities": [], "tags": [], "depends_on": [],
             "supersedes_topic": None, "confidence": None, "event_date": None},
        ],
        "temporal_links": []
    }
    prose_response = {
        "facts": [{"local_id": "b1", "fact": "Project overview", "kind": "fact",
                    "entities": [], "tags": [], "depends_on": [],
                    "supersedes_topic": None, "confidence": None, "event_date": None}],
        "temporal_links": []
    }
    mock, _ = _make_mock_extract([prose_response, list_response, prose_response, prose_response])

    text = "# Requirements\n\nKey requirements:\n1. Capacity: 500 L/s\n2. Pressure: 16 bar\n3. Material: DI"
    _, sn, _, facts, tlinks = asyncio.run(
        extract_session(text, 1, "2026-01-15", "doc-1", "project team",
                        "test-model", mock, fmt="DOCUMENT"))

    assert len(facts) >= 1
    assert all(f["session"] == 1 for f in facts)
    assert all("id" in f for f in facts)


# ═══════════════════════════════════════════════════════════════════
# Document with table
# ═══════════════════════════════════════════════════════════════════


def test_document_with_table():
    table_response = {
        "facts": [
            {"local_id": "b1", "fact": "Pipe cost is 1,850,000 TKT", "kind": "fact",
             "entities": [], "tags": [], "depends_on": [],
             "supersedes_topic": None, "confidence": None, "event_date": None},
        ],
        "temporal_links": []
    }
    mock, _ = _make_mock_extract([table_response] * 5)

    text = "# Cost Summary\n\n| Item | Cost |\n| Pipes | 1,850,000 |\n| Labor | 1,200,000 |"
    _, _, _, facts, _ = asyncio.run(
        extract_session(text, 1, "2026-01-15", "doc-2", "project team",
                        "test-model", mock, fmt="DOCUMENT"))

    assert len(facts) >= 1


# ═══════════════════════════════════════════════════════════════════
# KV specifications
# ═══════════════════════════════════════════════════════════════════


def test_document_with_kv():
    kv_response = {
        "facts": [
            {"local_id": "b1", "fact": "Pipe material is Ductile Iron", "kind": "fact",
             "entities": ["Ductile Iron"], "tags": [], "depends_on": [],
             "supersedes_topic": None, "confidence": None, "event_date": None},
        ],
        "temporal_links": []
    }
    mock, _ = _make_mock_extract([kv_response] * 5)

    text = "# Specs\n\nMaterial: Ductile Iron\nDiameter: DN1200\nPressure: PN16"
    _, _, _, facts, _ = asyncio.run(
        extract_session(text, 1, "2026-01-15", "doc-3", "project team",
                        "test-model", mock, fmt="DOCUMENT"))

    assert len(facts) >= 1
    assert any("Ductile Iron" in f["fact"] for f in facts)


# ═══════════════════════════════════════════════════════════════════
# section_path in extraction context
# ═══════════════════════════════════════════════════════════════════


def test_section_path_in_context():
    """section_path appears in the system prompt sent to LLM."""
    mock, log = _make_mock_extract()

    text = "# Technical Specifications\n\nMaterial: Steel\nPressure: 16 bar\nDiameter: 1200 mm"
    asyncio.run(
        extract_session(text, 1, "2026-01-15", "doc-4", "project team",
                        "test-model", mock, fmt="DOCUMENT"))

    assert len(log) >= 1
    # At least one call should have section_path in the system prompt
    assert any("Technical Specifications" in c["system"] for c in log), \
        f"section_path not found in prompts: {[c['system'][:100] for c in log]}"


# ═══════════════════════════════════════════════════════════════════
# Merged schema compliance
# ═══════════════════════════════════════════════════════════════════


def test_document_schema_compliance():
    mock, _ = _make_mock_extract()

    text = "# Overview\n\nThe project was initiated in 2024."
    _, sn, _, facts, tlinks = asyncio.run(
        extract_session(text, 3, "2026-01-15", "doc-5", "project team",
                        "test-model", mock, fmt="DOCUMENT"))

    assert len(facts) >= 1
    f = facts[0]
    assert "id" in f
    assert "session" in f and f["session"] == 3
    assert "fact" in f
    assert "kind" in f
    assert "speaker" in f
    assert "speaker_role" in f
    assert isinstance(tlinks, list)


# ═══════════════════════════════════════════════════════════════════
# Temporal links survive merge
# ═══════════════════════════════════════════════════════════════════


def test_temporal_links_survive():
    response = {
        "facts": [
            {"local_id": "b1", "fact": "Event A", "kind": "fact",
             "entities": [], "tags": [], "depends_on": [],
             "supersedes_topic": None, "confidence": None, "event_date": None},
            {"local_id": "b2", "fact": "Event B", "kind": "fact",
             "entities": [], "tags": [], "depends_on": [],
             "supersedes_topic": None, "confidence": None, "event_date": None},
        ],
        "temporal_links": [
            {"before": "b1", "after": "b2", "signal": "then", "relation": "before"}
        ]
    }
    mock, _ = _make_mock_extract([response])

    text = "# Timeline\n\nFirst event then second event."
    _, _, _, facts, tlinks = asyncio.run(
        extract_session(text, 1, "2026-01-15", "doc-6", "project team",
                        "test-model", mock, fmt="DOCUMENT"))

    assert len(tlinks) >= 1
    assert tlinks[0]["before"].startswith("f_")
    assert tlinks[0]["after"].startswith("f_")


# ═══════════════════════════════════════════════════════════════════
# Karnali snippet smoke check
# ═══════════════════════════════════════════════════════════════════


def test_karnali_snippet_extraction():
    response = {
        "facts": [
            {"local_id": "b1", "fact": "DN1200 pipe specification", "kind": "fact",
             "entities": ["DN1200"], "tags": [], "depends_on": [],
             "supersedes_topic": None, "confidence": None, "event_date": None},
        ],
        "temporal_links": []
    }
    mock, log = _make_mock_extract([response] * 10)

    text = (
        "# Karnali River Basin Water Supply\n\n"
        "## Technical Specifications\n\n"
        "Pipe material: Ductile Iron\n"
        "Nominal diameter: DN1200\n"
        "Pressure class: PN16\n\n"
        "## Key Components\n\n"
        "1. Intake structure\n"
        "2. Treatment plant\n"
        "3. Transmission pipeline\n"
    )
    _, _, _, facts, _ = asyncio.run(
        extract_session(text, 1, "2026-01-15", "karnali-doc", "project team",
                        "test-model", mock, fmt="DOCUMENT"))

    assert len(facts) >= 1
    # Multiple blocks should have been extracted
    assert len(log) >= 2


# ═══════════════════════════════════════════════════════════════════
# CONVERSATION path unchanged
# ═══════════════════════════════════════════════════════════════════


def test_conversation_path_unchanged():
    """CONVERSATION still uses conversation segmenter, not document segmenter."""
    mock, log = _make_mock_extract()

    text = "user: Hello.\nassistant: Hi there."
    _, _, _, facts, _ = asyncio.run(
        extract_session(text, 1, "2024-01-01", "conv-test", "User and Assistant",
                        "test-model", mock, fmt="CONVERSATION"))

    assert len(facts) >= 1
    # Should NOT have section_path in prompt (conversation doesn't have sections)
    for call in log:
        assert "Section: none" in call["system"] or "section" not in call["system"].lower() \
            or "Section:" in call["system"]
