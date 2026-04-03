"""Unit 8: Coverage-based prompt routing — tests against real _route_prompt_type helper."""

import asyncio

import numpy as np

from src.memory import MemoryServer, _route_prompt_type

# ── Summarize branch ──

def test_summarize():
    assert _route_prompt_type("summarize", [{"id": "f1"}], 100, 50, "") == (
        "summarize_with_metadata", True
    )


# ── ICL branch ──

def test_icl():
    assert _route_prompt_type("icl", [{"id": "f1"}], 100, 50, "") == (
        "icl", False
    )


# ── Tool mode: low coverage with facts, >20 sessions, <30% coverage ──

def test_low_coverage_tool():
    """50 sessions, 10 in context (20% < 30%) → tool mode."""
    assert _route_prompt_type(
        "lookup", [{"id": "f1"}], 50, 10, "RETRIEVED FACTS only"
    ) == ("tool", True)


# ── Negative: low session count should NOT trigger tool mode ──

def test_low_session_count_no_tool():
    """Only 5 sessions — even with <30% coverage, no tool mode."""
    assert _route_prompt_type(
        "lookup", [{"id": "f1"}], 5, 1, "RETRIEVED FACTS only"
    ) != ("tool", True)


# ── Negative: empty resolved_facts should NOT trigger tool mode ──

def test_empty_facts_no_tool():
    """No resolved_facts — tool mode requires facts to exist."""
    pt, ut = _route_prompt_type("lookup", [], 50, 10, "RETRIEVED FACTS only")
    assert pt != "tool"
    assert ut is False


# ── Hybrid: high coverage + RAW CONTEXT present ──

def test_high_coverage_hybrid():
    """5 sessions, 3 in context (60%) with RAW CONTEXT → hybrid."""
    assert _route_prompt_type(
        "lookup", [{"id": "f1"}], 5, 3, "RAW CONTEXT blah"
    ) == ("hybrid", False)


# ── Fallback: no raw context, no special type → pass through ──

def test_no_raw_context_fallback():
    """No RAW CONTEXT marker → fallback to resolved_type."""
    assert _route_prompt_type(
        "lookup", [{"id": "f1"}], 5, 3, "RETRIEVED FACTS only"
    ) == ("lookup", False)


# ── Summarize takes priority over low-coverage tool mode ──

def test_summarize_overrides_low_coverage():
    """summarize type wins even with low coverage params."""
    assert _route_prompt_type(
        "summarize", [{"id": "f1"}], 50, 5, "RAW CONTEXT blah"
    ) == ("summarize_with_metadata", True)


# ── ICL takes priority over everything below it ──

def test_icl_overrides_low_coverage():
    """icl type wins even with low coverage params."""
    assert _route_prompt_type(
        "icl", [{"id": "f1"}], 50, 5, "RAW CONTEXT blah"
    ) == ("icl", False)


# ── Episode hybrid marker recognized by _route_prompt_type ──

def test_episode_hybrid_marker():
    """Episode raw-text marker triggers hybrid mode like RAW CONTEXT does."""
    assert _route_prompt_type(
        "lookup", [{"id": "f1"}], 5, 3,
        "Some context\n--- SOURCE EPISODE RAW TEXT ---\nraw text here"
    ) == ("hybrid", False)


def test_document_hybrid_marker():
    """Document section marker must still route the fact path to hybrid."""
    assert _route_prompt_type(
        "lookup",
        [{"id": "f1"}],
        5,
        3,
        "Some context\n--- SOURCE DOCUMENT SECTIONS ---\nsection text here",
    ) == ("hybrid", False)


def test_episode_path_does_not_enter_tool_mode():
    """Episode recall should preserve its legacy no-tool behavior."""
    assert _route_prompt_type(
        "lookup",
        [{"id": "f1"}],
        50,
        10,
        "Some context\n--- SOURCE EPISODE RAW TEXT ---\nraw text here",
        allow_tool_mode=False,
    ) == ("hybrid", False)


# ── Real-pipeline integration: store → recall → routing ──

DIM = 3072


def _patch_llm_and_embeddings(monkeypatch):
    """Mock only LLM extraction and embedding API calls — NOT retrieval/routing."""

    async def mock_extract(**kwargs):
        sn = kwargs.get("session_num", 1)
        return ("conv", sn, "2024-06-01", [
            {"id": f"f{sn}_0", "fact": f"The project uses Rust for the agent executor.",
             "kind": "fact", "entities": ["Rust"], "tags": ["tech"], "session": sn},
            {"id": f"f{sn}_1", "fact": f"Memory subsystem is written in Python.",
             "kind": "fact", "entities": ["Python"], "tags": ["tech"], "session": sn},
        ], [])

    async def mock_consolidate(**kwargs):
        return ("conv", 1, "2024-06-01", [
            {"id": "c0", "fact": "Project stack: Rust agent, Python memory.",
             "kind": "summary", "entities": ["Rust", "Python"], "tags": ["tech"]}
        ])

    async def mock_cross(**kwargs):
        return ("conv", "e", [
            {"id": "x0", "fact": "Multi-language architecture: Rust + Python.",
             "kind": "profile", "entities": ["Rust", "Python"], "tags": ["arch"]}
        ])

    async def mock_embed(texts, **kw):
        # Deterministic embeddings seeded by text content for consistent retrieval
        vecs = []
        for t in texts:
            seed = sum(ord(c) for c in t) % (2**31)
            rng = np.random.RandomState(seed)
            vecs.append(rng.randn(DIM).astype(np.float32))
        return np.array(vecs)

    async def mock_embed_q(text, **kw):
        seed = sum(ord(c) for c in text) % (2**31)
        rng = np.random.RandomState(seed)
        return rng.randn(DIM).astype(np.float32)

    monkeypatch.setattr("src.memory.extract_session", mock_extract)
    monkeypatch.setattr("src.memory.consolidate_session", mock_consolidate)
    monkeypatch.setattr("src.memory.cross_session_entity", mock_cross)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda f, l: None)
    monkeypatch.setattr("src.memory.embed_texts", mock_embed)
    monkeypatch.setattr("src.memory.embed_query", mock_embed_q)


def test_real_pipeline_store_recall_routing(tmp_path, monkeypatch):
    """End-to-end: store() real data, recall() through real retrieval + routing.

    Mocks only LLM extraction and embedding API calls.
    Retrieval (retrieve_adaptive), query-type detection, and prompt routing
    all run through real code paths.
    """
    _patch_llm_and_embeddings(monkeypatch)

    ms = MemoryServer(str(tmp_path), "pipeline_test")

    # Store a session through the real store() path
    store_result = asyncio.run(ms.store(
        content="User: What language is the agent written in?\n"
                "Assistant: The agent executor is written in Rust.\n"
                "User: And the memory subsystem?\n"
                "Assistant: The memory subsystem is Python-based.",
        session_num=1,
        session_date="2024-06-01",
    ))
    assert store_result.get("facts_extracted", 0) > 0

    # build_index runs real embedding + data_dict construction
    asyncio.run(ms.build_index())

    # recall() runs real retrieval + real routing — nothing mocked here
    result = asyncio.run(ms.recall("What language is the agent written in?"))

    # Core recall shape assertions
    assert "context" in result
    assert "query_type" in result
    assert "recommended_prompt_type" in result
    assert "use_tool" in result
    assert isinstance(result["use_tool"], bool)

    # The real retrieval+rendering path for this corpus should stay hybrid
    assert result["recommended_prompt_type"] == "hybrid"

    # Retrieved facts should exist (real retrieval found something)
    assert len(result.get("retrieved", [])) > 0

    # Context should contain actual content from stored facts
    assert len(result["context"]) > 0
