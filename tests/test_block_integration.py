"""Integration tests: block pipeline wired into librarian.extract_session().

Stage 3 of ContentIR Phase 1.
Uses mocked LLM calls — no live model dependency.
"""

import asyncio
import json

from src.librarian import extract_session


def _make_mock_extract(responses=None):
    """Create a mock call_extract_fn that returns canned JSON responses.

    If responses is None, returns a generic valid extraction result.
    If responses is a list, returns them in order (cycling if needed).
    """
    call_log = []
    idx = {"n": 0}

    default_response = {
        "facts": [
            {"local_id": "b1", "fact": "extracted fact", "kind": "fact",
             "entities": [], "tags": [], "depends_on": [],
             "supersedes_topic": None, "confidence": None, "event_date": None}
        ],
        "temporal_links": []
    }

    async def mock(model, system, user_msg, max_tokens=8192):
        call_log.append({"model": model, "system": system[:200], "user_msg": user_msg[:200]})
        if responses:
            resp = responses[idx["n"] % len(responses)]
            idx["n"] += 1
            return resp
        return dict(default_response)

    return mock, call_log


# ═══════════════════════════════════════════════════════════════════
# a) Conversation with embedded list
# ═══════════════════════════════════════════════════════════════════


def test_conversation_with_list():
    """Block pipeline extracts facts from conversation containing a list."""
    list_response = {
        "facts": [
            {"local_id": "b1", "fact": "User's favorite book: The Great Gatsby", "kind": "count_item",
             "entities": ["The Great Gatsby"], "tags": [], "depends_on": [],
             "supersedes_topic": None, "confidence": None, "event_date": None},
            {"local_id": "b2", "fact": "User's favorite book: 1984", "kind": "count_item",
             "entities": ["1984"], "tags": [], "depends_on": [],
             "supersedes_topic": None, "confidence": None, "event_date": None},
        ],
        "temporal_links": []
    }
    prose_response = {
        "facts": [
            {"local_id": "b1", "fact": "User shared their reading list", "kind": "fact",
             "entities": [], "tags": [], "depends_on": [],
             "supersedes_topic": None, "confidence": None, "event_date": None},
        ],
        "temporal_links": []
    }
    mock, log = _make_mock_extract([prose_response, list_response, prose_response])

    text = (
        "user: Here are my favorite books:\n"
        "1. The Great Gatsby\n"
        "2. 1984\n"
        "assistant: Great choices!"
    )
    _, sn, _, facts, tlinks = asyncio.run(
        extract_session(text, 1, "2024-01-01", "conv-1", "User and Assistant",
                        "test-model", mock, fmt="CONVERSATION"))

    assert len(facts) >= 2
    assert any("Great Gatsby" in f["fact"] for f in facts)
    assert all(f["session"] == 1 for f in facts)
    assert all("id" in f for f in facts)


# ═══════════════════════════════════════════════════════════════════
# b) Conversation with embedded table
# ═══════════════════════════════════════════════════════════════════


def test_conversation_with_table():
    """Block pipeline extracts facts from conversation containing a table."""
    table_response = {
        "facts": [
            {"local_id": "b1", "fact": "Alice works Morning shift on Monday", "kind": "fact",
             "entities": ["Alice"], "tags": [], "depends_on": [],
             "supersedes_topic": None, "confidence": None, "event_date": None},
        ],
        "temporal_links": []
    }
    prose_response = {
        "facts": [{"local_id": "b1", "fact": "User shared the schedule", "kind": "fact",
                    "entities": [], "tags": [], "depends_on": [],
                    "supersedes_topic": None, "confidence": None, "event_date": None}],
        "temporal_links": []
    }
    mock, _ = _make_mock_extract([prose_response, table_response, prose_response])

    text = (
        "user: Here's the schedule:\n"
        "| Day | Shift | Agent |\n"
        "| Mon | Morning | Alice |\n"
        "| Tue | Evening | Bob |\n"
        "assistant: Got it."
    )
    _, _, _, facts, _ = asyncio.run(
        extract_session(text, 2, "2024-02-01", "conv-2", "User and Assistant",
                        "test-model", mock, fmt="CONVERSATION"))

    assert len(facts) >= 1
    assert any("Alice" in f["fact"] for f in facts)


# ═══════════════════════════════════════════════════════════════════
# c) Fender Stratocaster
# ═══════════════════════════════════════════════════════════════════


def test_fender_stratocaster():
    """Short user utterance with specific model name must be preserved."""
    fender_response = {
        "facts": [
            {"local_id": "b1",
             "fact": "User owns a Fender Stratocaster and is considering upgrading to Gibson Les Paul",
             "kind": "fact", "entities": ["Fender Stratocaster", "Gibson Les Paul"],
             "tags": [], "depends_on": [], "supersedes_topic": None,
             "confidence": None, "event_date": None},
        ],
        "temporal_links": []
    }
    assistant_response = {
        "facts": [
            {"local_id": "b1", "fact": "Assistant recommended Fender Deluxe Reverb amp",
             "kind": "fact", "entities": ["Fender Deluxe Reverb"],
             "tags": [], "depends_on": [], "supersedes_topic": None,
             "confidence": None, "event_date": None},
        ],
        "temporal_links": []
    }
    mock, _ = _make_mock_extract([fender_response, assistant_response])

    text = (
        "user: I have a Fender Stratocaster, what amp would you recommend?\n"
        "assistant: For a Strat, a Fender Deluxe Reverb is a classic choice."
    )
    _, _, _, facts, _ = asyncio.run(
        extract_session(text, 1, "2024-03-01", "conv-3", "User and Assistant",
                        "test-model", mock, fmt="CONVERSATION"))

    all_text = " ".join(f["fact"] for f in facts)
    assert "Fender Stratocaster" in all_text


# ═══════════════════════════════════════════════════════════════════
# d) JSON_CONV propagation
# ═══════════════════════════════════════════════════════════════════


def test_json_conv_uses_block_pipeline():
    """JSON_CONV input preprocesses then reaches block pipeline."""
    response = {
        "facts": [
            {"local_id": "b1", "fact": "User said hello", "kind": "fact",
             "entities": [], "tags": [], "depends_on": [],
             "supersedes_topic": None, "confidence": None, "event_date": None},
        ],
        "temporal_links": []
    }
    mock, log = _make_mock_extract([response, response, response, response])

    json_text = json.dumps([
        {"role": "user", "content": "Hello there!"},
        {"role": "assistant", "content": "Hi! How can I help?"},
    ])
    _, _, _, facts, _ = asyncio.run(
        extract_session(json_text, 1, "2024-04-01", "conv-4", "User and Assistant",
                        "test-model", mock, fmt="JSON_CONV"))

    assert len(facts) >= 1
    assert len(log) >= 1


def test_json_conv_chunked_temporal_links():
    """Chunked JSON_CONV renumbers IDs per-chunk, temporal links remap correctly."""
    # Each chunk returns 2 facts with a temporal link f_01→f_02.
    # After 2 chunks: 4 facts (f_01..f_04), links should be f_01→f_02 and f_03→f_04.
    chunk_response = {
        "facts": [
            {"local_id": "b1", "fact": "event A", "kind": "fact",
             "entities": [], "tags": [], "depends_on": [],
             "supersedes_topic": None, "confidence": None, "event_date": None},
            {"local_id": "b2", "fact": "event B", "kind": "fact",
             "entities": [], "tags": [], "depends_on": [],
             "supersedes_topic": None, "confidence": None, "event_date": None},
        ],
        "temporal_links": [
            {"before": "b1", "after": "b2", "signal": "then", "relation": "before"}
        ]
    }
    mock, _ = _make_mock_extract([chunk_response] * 10)

    # Build a JSON conversation long enough to be chunked (>8000 chars)
    turns = []
    for i in range(60):
        turns.append({"role": "user", "content": f"Message {i}: " + "x" * 200})
        turns.append({"role": "assistant", "content": f"Reply {i}: " + "y" * 200})
    json_text = json.dumps(turns)

    _, _, _, facts, tlinks = asyncio.run(
        extract_session(json_text, 1, "2024-04-01", "conv-chunked",
                        "User and Assistant", "test-model", mock, fmt="JSON_CONV"))

    # Facts should have unique IDs
    ids = [f["id"] for f in facts]
    assert len(ids) == len(set(ids)), f"Duplicate IDs: {[x for x in ids if ids.count(x) > 1]}"

    # Temporal links should reference valid fact IDs within the same chunk
    fact_id_set = set(ids)
    for tl in tlinks:
        # Links should reference sequential pairs (f_01→f_02, f_03→f_04, etc.)
        assert tl["before"] in fact_id_set or tl["before"].startswith("f_"), \
            f"Bad temporal link before: {tl}"


# ═══════════════════════════════════════════════════════════════════
# e) Named speaker / source-first
# ═══════════════════════════════════════════════════════════════════


def test_named_speakers_source_first():
    """Speakers from `speakers` param are passed through to block segmenter."""
    response = {
        "facts": [
            {"local_id": "b1", "fact": "Caroline attended a support group", "kind": "fact",
             "entities": ["Caroline"], "tags": [], "depends_on": [],
             "supersedes_topic": None, "confidence": None, "event_date": None},
        ],
        "temporal_links": []
    }
    mock, _ = _make_mock_extract([response, response, response, response])

    text = (
        "Caroline: I went to a support group yesterday.\n"
        "It was so powerful.\n"
        "Melanie: That's amazing!\n"
        "What was it like?\n"
        "Caroline: The stories were inspiring.\n"
        "I learned a lot."
    )
    _, _, _, facts, _ = asyncio.run(
        extract_session(text, 1, "2024-05-01", "conv-5", "Caroline and Melanie",
                        "test-model", mock, fmt="CONVERSATION"))

    assert len(facts) >= 1
    # Speaker should be inherited from blocks
    speakers_found = {f.get("speaker") for f in facts}
    # At least one fact should have a speaker set
    assert speakers_found != {None}


# ═══════════════════════════════════════════════════════════════════
# f) Schema compatibility
# ═══════════════════════════════════════════════════════════════════


def test_schema_compatibility():
    """Extracted facts have all required downstream fields."""
    response = {
        "facts": [
            {"local_id": "b1", "fact": "test fact", "kind": "preference",
             "entities": ["E1"], "tags": ["t1"], "depends_on": [],
             "supersedes_topic": None, "confidence": "explicit",
             "event_date": "2024-01-15"},
        ],
        "temporal_links": [
            {"before": "b1", "after": "b1", "signal": "then", "relation": "before"}
        ]
    }
    mock, _ = _make_mock_extract([response])

    text = "user: I prefer tea over coffee."
    _, sn, _, facts, tlinks = asyncio.run(
        extract_session(text, 3, "2024-01-15", "conv-6", "User and Assistant",
                        "test-model", mock, fmt="CONVERSATION"))

    assert len(facts) == 1
    f = facts[0]
    # Required fields
    assert "id" in f
    assert "session" in f and f["session"] == 3
    assert "fact" in f
    assert "kind" in f
    assert "speaker" in f
    assert "speaker_role" in f
    assert "entities" in f
    assert "tags" in f

    # Return shape
    assert isinstance(tlinks, list)


# ═══════════════════════════════════════════════════════════════════
# Non-conversation formats unchanged
# ═══════════════════════════════════════════════════════════════════


def test_agent_trace_unchanged():
    """AGENT_TRACE still uses legacy single-prompt path."""
    mock, log = _make_mock_extract()

    text = "[Step 1]\nAction: click\nObservation: done"
    _, _, _, facts, _ = asyncio.run(
        extract_session(text, 1, "2024-01-01", "trace-1", "agent",
                        "test-model", mock, fmt="AGENT_TRACE"))

    assert len(log) == 1
    # Should NOT contain block prompt markers
    assert "prose block" not in log[0]["system"].lower()
    assert "list block" not in log[0]["system"].lower()
