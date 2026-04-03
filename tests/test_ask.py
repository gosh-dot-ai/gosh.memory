"""Phase 3: ask() + Model Profiles tests (Units 1-4)."""

import asyncio
import json
from types import SimpleNamespace

import pytest
import numpy as np

from src.episode_features import extract_query_features
from src.memory import (
    MemoryServer,
    _augment_commonality_facts,
    _build_raw_commonality_support_items,
    _build_local_anchor_support_items,
    _context_has_source_excerpts,
    _extract_conversation_support_lines,
    _extract_conversation_windows,
)


DIM = 3072

PROFILES = {
    1: "fast",
    2: "fast",
    3: "balanced",
    4: "max",
    5: "max",
}

PROFILE_CONFIGS = {
    "fast": {"model": "openai/gpt-4o-mini", "context_window": 128000,
             "max_output_tokens": 2000, "max_output_tokens_summarize": 4096,
             "thinking_overhead": 0, "input_cost_per_1k": 0.15, "output_cost_per_1k": 0.60},
    "balanced": {"model": "openai/gpt-4.1", "context_window": 128000,
                 "max_output_tokens": 2000, "max_output_tokens_summarize": 4096,
                 "thinking_overhead": 0, "input_cost_per_1k": 2.0, "output_cost_per_1k": 8.0},
    "max": {"model": "anthropic/claude-opus-4-6", "context_window": 200000,
            "max_output_tokens": 4096, "max_output_tokens_summarize": 8192,
            "thinking_overhead": 0, "input_cost_per_1k": 15.0, "output_cost_per_1k": 75.0},
}


def _make_fact(fid="f1", kind="fact", session=1, complexity=0.10):
    return {
        "fact": f"Test fact {fid}", "kind": kind,
        "id": fid, "conv_id": "ask_test", "session": session,
        "agent_id": "default", "swarm_id": "default", "scope": "swarm-shared",
        "owner_id": "system", "read": ["agent:PUBLIC"], "write": ["agent:PUBLIC"],
        "created_at": "2024-01-01T00:00:00+00:00",
        "_session_content_complexity": complexity,
        "entities": [], "tags": [],
    }


def _patch_all(monkeypatch):
    async def mock_embed(texts, **kw):
        return np.random.randn(len(texts), DIM).astype(np.float32)

    async def mock_embed_q(text, **kw):
        return np.random.randn(DIM).astype(np.float32)

    async def mock_extract(**kwargs):
        sn = kwargs.get("session_num", 1)
        return ("conv", sn, "2024-06-01", [
            {"id": f"f{sn}_0", "fact": f"Fact {sn}", "kind": "event",
             "entities": [], "tags": [], "session": sn}], [])

    async def mock_consolidate(**kwargs):
        return ("conv", 1, "2024-06-01", [
            {"id": "c0", "fact": "Cons", "kind": "summary",
             "entities": [], "tags": []}])

    async def mock_cross(**kwargs):
        return ("conv", "e", [
            {"id": "x0", "fact": "Cross", "kind": "profile",
             "entities": [], "tags": []}])

    monkeypatch.setattr("src.memory.extract_session", mock_extract)
    monkeypatch.setattr("src.memory.consolidate_session", mock_consolidate)
    monkeypatch.setattr("src.memory.cross_session_entity", mock_cross)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda f, l: None)
    monkeypatch.setattr("src.memory.embed_texts", mock_embed)
    monkeypatch.setattr("src.memory.embed_query", mock_embed_q)


def _patch_call_oai(monkeypatch, answer="Test answer"):
    """Mock call_oai to return a fixed answer."""
    async def mock_call_oai(model, prompt, max_tokens=300, **kw):
        return answer
    monkeypatch.setattr("src.memory.call_oai", mock_call_oai)


def _patch_call_oai_capture(monkeypatch, answer="Test answer"):
    """Mock call_oai and capture request params."""
    seen = {}

    async def mock_call_oai(model, prompt, max_tokens=300, **kw):
        seen["model"] = model
        seen["prompt"] = prompt
        seen["max_tokens"] = max_tokens
        seen["kwargs"] = kw
        return answer

    monkeypatch.setattr("src.memory.call_oai", mock_call_oai)
    return seen


def _patch_recall(ms, *, query_type="default", recommended_profile="fast",
                  prompt_type="lookup", use_tool=False,
                  retrieval_families=None, search_family="auto", retrieved_count=1):
    async def mock_recall(*args, **kwargs):
        return {
            "context": "Context block",
            "query_type": query_type,
            "recommended_profile": recommended_profile,
            "use_tool": use_tool,
            "recommended_prompt_type": prompt_type,
            "sessions_in_context": 1,
            "total_sessions": 3,
            "coverage_pct": 33,
            "retrieval_families": list(retrieval_families or ["conversation"]),
            "search_family": search_family,
            "retrieved": [{}] * retrieved_count,
        }

    ms.recall = mock_recall


class _FakeCompletions:
    def __init__(self, responses, seen):
        self._responses = list(responses)
        self._seen = seen

    async def create(self, **kwargs):
        self._seen.append(kwargs)
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses, seen):
        self.chat = SimpleNamespace(completions=_FakeCompletions(responses, seen))


def _tool_call_response(session_id=1):
    msg = SimpleNamespace(
        content=None,
        tool_calls=[SimpleNamespace(
            id="tool_1",
            function=SimpleNamespace(
                name="get_more_context",
                arguments=json.dumps({"session_id": session_id}),
            ),
        )],
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _text_response(text):
    msg = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


# ── Unit 1: ask() method core ──

class TestAskCore:
    def test_context_marker_detects_source_sections(self):
        context = "RETRIEVED FACTS:\n[1] Foo\n\n--- SOURCE DOCUMENT SECTIONS ---\n[Source: bar]\nBaz"
        assert _context_has_source_excerpts(context) is True

    def test_extract_conversation_windows_prefers_fact_supporting_region(self):
        text = "\n".join(
            [
                "user: coupon organization question",
                "assistant: binder tips",
                "user: I use the Cartwheel app from Target.",
                "assistant: Cartwheel is great at Target.",
                "user: I redeemed a $5 coupon on coffee creamer last Sunday.",
                "assistant: Nice surprise coupon.",
                "user: I shop at Target every other week.",
            ]
        )
        facts = [{"fact": "User redeemed a $5 coupon on coffee creamer last Sunday.", "session": 1}]
        snippet = _extract_conversation_windows(text, facts, cap=1000)
        assert "redeemed a $5 coupon on coffee creamer" in snippet
        assert "Cartwheel app from Target" in snippet

    def test_extract_conversation_support_lines_keeps_neighboring_anchor_lines(self):
        text = "\n".join(
            [
                "user: coupon organization question",
                "assistant: binder tips",
                "user: I use the Cartwheel app from Target.",
                "assistant: Cartwheel is great at Target.",
                "user: I redeemed a $5 coupon on coffee creamer last Sunday.",
                "assistant: Nice surprise coupon.",
                "user: I shop at Target every other week.",
            ]
        )
        facts = [{"fact": "User redeemed a $5 coupon on coffee creamer last Sunday.", "session": 1}]
        lines = _extract_conversation_support_lines(text, facts, max_lines=6)
        joined = "\n".join(lines)
        assert "redeemed a $5 coupon on coffee creamer" in joined
        assert "Target" in joined

    def test_build_local_anchor_support_items_promotes_repeated_store_anchor(self):
        text = "\n".join(
            [
                "user: I use the Cartwheel app from Target.",
                "assistant: Cartwheel is great at Target.",
                "user: I redeemed a $5 coupon on coffee creamer last Sunday.",
                "assistant: Nice surprise coupon.",
                "user: I shop at Target every other week.",
            ]
        )
        items = _build_local_anchor_support_items(
            "Where did I redeem a $5 coupon on coffee creamer?",
            [{"fact": "User redeemed a $5 coupon on coffee creamer last Sunday.", "session": 1}],
            [{"content": text}],
        )
        assert items
        assert "Target" in items[0]["text"]

    def test_build_local_anchor_support_items_accepts_single_store_anchor_with_coupon_cue(self):
        text = "\n".join(
            [
                "user: I redeemed a $5 coupon on coffee creamer last Sunday.",
                "assistant: Many retailers, like Target, send exclusive coupons to subscribers.",
            ]
        )
        items = _build_local_anchor_support_items(
            "Where did I redeem a $5 coupon on coffee creamer?",
            [{"fact": "User redeemed a $5 coupon on coffee creamer last Sunday.", "session": 1}],
            [{"content": text}],
        )
        assert items
        assert "Target" in items[0]["text"]

    def test_build_raw_commonality_support_items_finds_shared_job_loss_and_business_start(self):
        sessions = [{
            "content": "\n".join(
                [
                    "Jon: Lost my job as a banker yesterday, so I'm gonna take a shot at starting my own business.",
                    "Gina: Sorry about your job Jon, but starting your own business sounds awesome! Unfortunately, I also lost my job at Door Dash this month.",
                    "Jon: Sorry to hear that! I'm starting a dance studio.",
                ]
            )
        }]
        items = _build_raw_commonality_support_items(
            "What do Jon and Gina both have in common?",
            sessions,
        )
        assert items
        joined = "\n".join(item["text"] for item in items)
        assert "lost" in joined.lower()
        assert "job" in joined.lower()

    def test_augment_commonality_facts_prefers_repeated_concrete_shared_motifs(self):
        facts = [
            {"id": "j1", "fact": "Jon lost his job as a banker yesterday", "entities": ["Jon"], "session": 241, "speaker": "Jon"},
            {"id": "g1", "fact": "Gina lost her job at Door Dash this month", "entities": ["Gina"], "session": 241, "speaker": "Gina"},
            {"id": "j2", "fact": "Jon plans to start his own business", "entities": ["Jon"], "session": 241, "speaker": "Jon"},
            {"id": "g2", "fact": "Gina started her own online clothing store", "entities": ["Gina"], "session": 242, "speaker": "Gina"},
            {"id": "j3", "fact": "Jon's hard work and talent will pay off", "entities": ["Jon"], "session": 254, "speaker": "Gina"},
            {"id": "g3", "fact": "Gina's hard work is paying off", "entities": ["Gina"], "session": 253, "speaker": "Jon"},
        ]

        extras = _augment_commonality_facts(
            "What do Jon and Gina both have in common?",
            [],
            facts,
            limit=4,
        )

        picked = [fact["fact"] for fact in extras]
        joined = "\n".join(picked).lower()
        assert "lost his job" in joined
        assert "lost her job" in joined
        assert "start his own business" in joined
        assert "started her own online clothing store" in joined

    def test_ask_with_profiles_returns_answer(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        _patch_call_oai(monkeypatch, "The user likes dark mode")
        ms = MemoryServer(str(tmp_path), "ask1", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)
        ms._all_granular = [_make_fact()]
        asyncio.run(ms.build_index())

        result = asyncio.run(ms.ask("what does user prefer?"))
        assert "answer" in result
        assert result["answer"] == "The user likes dark mode"
        assert result["budget_exceeded"] is False

    def test_ask_no_profiles_no_model_error(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask2")  # no profiles
        ms._all_granular = [_make_fact()]
        asyncio.run(ms.build_index())

        result = asyncio.run(ms.ask("test"))
        assert result["code"] == "NO_PROFILES"

    def test_ask_no_profiles_with_model_works(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        _patch_call_oai(monkeypatch, "answer via model")
        ms = MemoryServer(str(tmp_path), "ask3")  # no profiles
        ms._all_granular = [_make_fact()]
        asyncio.run(ms.build_index())

        result = asyncio.run(ms.ask("test", inference_model="openai/gpt-oss-120b"))
        assert result["answer"] == "answer via model"

    def test_ask_strips_think_blocks(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        _patch_call_oai(monkeypatch, "<think>reasoning here</think>Clean answer")
        ms = MemoryServer(str(tmp_path), "ask4", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)
        ms._all_granular = [_make_fact()]
        asyncio.run(ms.build_index())

        result = asyncio.run(ms.ask("test"))
        assert "<think>" not in result["answer"]
        assert result["answer"] == "Clean answer"

    def test_ask_strips_unclosed_think_blocks(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        _patch_call_oai(monkeypatch, "<think>reasoning here without close")
        ms = MemoryServer(str(tmp_path), "ask4b", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)
        ms._all_granular = [_make_fact()]
        asyncio.run(ms.build_index())

        result = asyncio.run(ms.ask("test"))
        assert result["answer"] == ""

    def test_recall_still_works(self, tmp_path, monkeypatch):
        """recall() must still work independently of ask()."""
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask5", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)
        ms._all_granular = [_make_fact()]
        asyncio.run(ms.build_index())

        result = asyncio.run(ms.recall("test"))
        assert "context" in result
        assert "answer" not in result  # recall != ask

    def test_ask_exposes_execution_telemetry(self, tmp_path, monkeypatch):
        _patch_call_oai(monkeypatch, "telemetry answer")
        ms = MemoryServer(str(tmp_path), "ask_telem", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)
        _patch_recall(
            ms,
            recommended_profile="fast",
            prompt_type="lookup",
            retrieval_families=["conversation"],
            search_family="conversation",
            retrieved_count=3,
        )

        result = asyncio.run(ms.ask("test"))

        assert result["telemetry_version"] == 1
        assert result["answer"] == "telemetry answer"
        assert result["estimated_cost"] > 0
        assert result["retrieval_families"] == ["conversation"]
        assert result["search_family"] == "conversation"
        assert result["retrieved_count"] == 3

    def test_build_payload_messages_routes_compositional_queries_to_leaf_prompt(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_compositional_prompt", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        messages = ms._build_payload_messages(
            prompt_type="hybrid",
            context="Context block\n\nRAW SLOT CANDIDATES:\n[Q1] wildlife documentary",
            query="What kind of project is Gina doing?",
            recall_result={
                "sessions_in_context": 1,
                "total_sessions": 3,
                "coverage_pct": 33,
                "query_operator_plan": {
                    "slot_query": {"enabled": True},
                    "list_set": {"enabled": False},
                    "ordinal": {"enabled": False},
                    "commonality": {"enabled": False},
                    "compare_diff": {"enabled": False},
                },
            },
            speakers="Gina, Jon",
        )

        content = messages[0]["content"]
        assert "slot-filling or attribute question" in content
        assert "RAW SLOT CANDIDATES" in content


    def test_normalize_grounded_answer_marks_unresolved_slot_not_mentioned(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_slot_unresolved", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        normalized = ms._normalize_grounded_answer(
            "What temporary job did Jon take to help cover expenses?",
            "temporary job",
            {
                "context": "RETRIEVED FACTS:\n[1] Jon took a temporary job to help cover expenses.",
                "retrieved": [
                    {"fact": "Jon took a temporary job to help cover expenses."}
                ],
            },
        )

        assert normalized == "Not mentioned in the provided context."


    def test_normalize_grounded_answer_blocks_slot_rescue_when_query_qualifiers_are_missing(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_slot_qualifier_guard", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        normalized = ms._normalize_grounded_answer(
            "What is John's main focus in international politics?",
            "Not mentioned in the provided context.",
            {
                "context": "RETRIEVED FACTS:\n[1] Improving education and infrastructure are John's main focuses.",
                "retrieved": [
                    {"fact": "Improving education and infrastructure are John's main focuses."},
                    {"fact": "John is hoping to get into local politics."},
                ],
            },
        )

        assert normalized == "Not mentioned in the provided context."

    def test_normalize_grounded_answer_blocks_direct_slot_answer_when_specific_qualifier_is_missing(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_slot_direct_qualifier_guard", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        normalized = ms._normalize_grounded_answer(
            "What is John's main focus in international politics?",
            "Improving education and infrastructure",
            {
                "context": "RETRIEVED FACTS:\n[1] Improving education and infrastructure are John's main focuses.",
                "retrieved": [
                    {"fact": "Improving education and infrastructure are John's main focuses."},
                    {"fact": "John is hoping to get into local politics."},
                ],
            },
        )

        assert normalized == "Not mentioned in the provided context."

    def test_normalize_grounded_answer_allows_slot_rescue_when_only_generic_modifier_is_missing(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_slot_generic_modifier_guard", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        normalized = ms._normalize_grounded_answer(
            "What temporary job did Jon take to cover expenses?",
            "Not mentioned in the provided context.",
            {
                "context": "RETRIEVED FACTS:\n[1] Jon took a Door Dash job to cover expenses.",
                "retrieved": [
                    {"fact": "Jon took a Door Dash job to cover expenses."},
                ],
            },
        )

        assert normalized == "Door Dash"

    def test_normalize_grounded_answer_allows_time_scoped_activity_slot_rescue_with_recreational_scaffold(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_slot_recreational_scaffold", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        normalized = ms._normalize_grounded_answer(
            "Which recreational activity was James pursuing on March 16, 2022?",
            "Not mentioned in the provided context.",
            {
                "context": "RETRIEVED FACTS:\n[1] James: I will be looking forward to it. By the way, yesterday I went bowling and got 2 strikes. I love bowling! Resolved date: March 16, 2022 (March 2022).",
                "retrieved": [
                    {"fact": "James: I will be looking forward to it. By the way, yesterday I went bowling and got 2 strikes. I love bowling! Resolved date: March 16, 2022 (March 2022)."},
                ],
            },
        )

        assert normalized == "bowling"

    def test_normalize_grounded_answer_allows_time_scoped_project_slot_rescue_with_beginning_scaffold(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_slot_beginning_scaffold", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        normalized = ms._normalize_grounded_answer(
            "What kind of project was Jolene working on in the beginning of January 2023?",
            "Not mentioned in the provided context.",
            {
                "context": "RETRIEVED FACTS:\n[1] Jolene finished an electrical engineering project last week.",
                "retrieved": [
                    {"fact": "Jolene finished an electrical engineering project last week."},
                ],
            },
        )

        assert normalized == "electrical engineering"

    def test_normalize_grounded_answer_preserves_grounded_specific_slot_value(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_slot_grounded", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        normalized = ms._normalize_grounded_answer(
            "What kind of project is Gina doing?",
            "electrical engineering",
            {
                "context": "RETRIEVED FACTS:\n[1] Gina is doing an electrical engineering project for school.",
                "retrieved": [
                    {"fact": "Gina is doing an electrical engineering project for school."}
                ],
            },
        )

        assert normalized == "electrical engineering"

    def test_normalize_grounded_answer_preserves_favorite_title_when_context_grounded(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_slot_favorite", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        normalized = ms._normalize_grounded_answer(
            "What is one of her favorite movies?",
            "Eternal Sunshine of the Spotless Mind",
            {
                "context": (
                    "RETRIEVED FACTS:\n"
                    "[1] Eternal Sunshine of the Spotless Mind is one of her favorite movies."
                ),
                "retrieved": [
                    {
                        "fact": "Eternal Sunshine of the Spotless Mind is one of her favorite movies."
                    }
                ],
            },
        )

        assert normalized == "Eternal Sunshine of the Spotless Mind"

    def test_normalize_grounded_answer_preserves_grounded_commonality_items_without_entity_noise(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_commonality_multi_item", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        normalized = ms._normalize_grounded_answer(
            "What kind of interests do Joanna and Nate share?",
            "Joanna and Nate share two main interests: watching movies and dairy-free desserts.",
            {
                "context": (
                    "RETRIEVED FACTS:\n"
                    "[1] Joanna enjoys watching movies.\n"
                    "[2] Nate enjoys watching movies.\n"
                    "[3] Joanna tries to make dairy-free desserts just as delicious as non-dairy ones.\n"
                    "[4] Nate started teaching people how to make dairy-free desserts."
                ),
                "retrieved": [
                    {"fact": "Joanna enjoys watching movies."},
                    {"fact": "Nate enjoys watching movies."},
                    {"fact": "Joanna tries to make dairy-free desserts just as delicious as non-dairy ones."},
                    {"fact": "Nate started teaching people how to make dairy-free desserts."},
                ],
            },
        )

        assert normalized == "watching movies, dairy-free desserts"


    def test_normalize_grounded_answer_rescues_commonality_interest_items_from_support_facts(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_commonality_interest_rescue", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        normalized = ms._normalize_grounded_answer(
            "What kind of interests do Joanna and Nate share?",
            "Not mentioned in the provided context.",
            {
                "context": (
                    "RETRIEVED FACTS:\n"
                    "[1] Joanna enjoys watching movies.\n"
                    "[2] Nate enjoys watching movies.\n"
                    "[3] Joanna tries to make dairy-free desserts just as delicious as non-dairy ones.\n"
                    "[4] Nate started teaching people how to make dairy-free desserts."
                ),
                "retrieved": [
                    {"fact": "Joanna enjoys watching movies."},
                    {"fact": "Nate enjoys watching movies."},
                    {"fact": "Joanna tries to make dairy-free desserts just as delicious as non-dairy ones."},
                    {"fact": "Nate started teaching people how to make dairy-free desserts."},
                ],
            },
        )

        assert "movies" in normalized
        assert "desserts" in normalized
        assert normalized != "Not mentioned in the provided context."

    def test_normalize_grounded_answer_preserves_multiple_grounded_slot_candidates(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_slot_multi_item", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        normalized = ms._normalize_grounded_answer(
            "What are two of John's goals?",
            "John wants to improve his shooting percentage and win a championship.",
            {
                "context": (
                    "RETRIEVED FACTS:\n"
                    "[1] John wants to improve his shooting percentage.\n"
                    "[2] John wants to win a championship.\n\n"
                    "RAW SLOT CANDIDATES:\n"
                    "[Q1] improve shooting percentage\n"
                    "[Q2] winning championship"
                ),
                "retrieved": [
                    {"fact": "John wants to improve his shooting percentage."},
                    {"fact": "John wants to win a championship."},
                ],
            },
        )

        assert normalized == "improve shooting percentage, winning championship"

    def test_normalize_grounded_answer_uses_raw_slot_candidates_for_negative_answer(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_slot_raw_candidate", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        normalized = ms._normalize_grounded_answer(
            "What is one of Joanna's favorite movies?",
            "Not mentioned in the provided context.",
            {
                "context": (
                    "RETRIEVED FACTS:\n"
                    "[1] Joanna enjoys watching movies.\n\n"
                    "RAW SLOT CANDIDATES:\n"
                    "[Q1] Eternal Sunshine of the Spotless Mind (from [Turn query]: eternal sunshine of the spotless mind movie poster) "
                    "[Episode: q45_locomo_conv_42_cat4_e0001] [Local evidence: It's one of my favorites.]"
                ),
                "retrieved": [
                    {"fact": "Joanna enjoys watching movies."}
                ],
            },
        )

        assert normalized == "Eternal Sunshine of the Spotless Mind"

    def test_normalize_grounded_answer_ignores_raw_context_noise_for_unresolved_slot(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_slot_context_noise", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        normalized = ms._normalize_grounded_answer(
            "What temporary job did Jon take to cover expenses?",
            (
                "Jon took a temporary job to cover expenses while working on his dance studio. "
                "The retrieved facts and raw context explicitly state he secured a temp job "
                "but do not specify the exact role or industry."
            ),
            {
                "context": (
                    "RETRIEVED FACTS:\n"
                    "[4] (S10) Jon got a temporary job.\n"
                    "[13] (S16) Jon's job loss was hard [Episode: q45_locomo_conv_30_cat5_e0016]\n"
                    "--- SOURCE EPISODE RAW TEXT ---\n"
                    "Jon: The dance studio is on tenuous grounds right now, but I'm staying positive. "
                    "I got a temp job to help cover expenses while I look for investors.\n"
                    "--- SOURCE DOCUMENT SECTIONS ---\n"
                    "Section: investors and expenses planning."
                ),
                "retrieved": [],
            },
        )

        assert normalized == "Not mentioned in the provided context."

    def test_normalize_grounded_answer_rejects_meta_explanatory_slot_answer_without_fill_candidate(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_slot_meta_explanation", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        normalized = ms._normalize_grounded_answer(
            "What temporary job did Jon take to cover expenses?",
            (
                "Answer: Jon took a temporary job to help cover expenses while he works on his dance studio and seeks investors. "
                "This is explicitly stated in the retrieved fact [8] and confirmed in the RAW CONTEXT of Episode q45_locomo_conv_30_cat5_e0010."
            ),
            {
                "context": "RETRIEVED FACTS:\n[8] Jon got a temp job to help cover expenses.",
                "retrieved": [{"fact": "Jon got a temp job to help cover expenses."}],
            },
        )

        assert normalized == "Not mentioned in the provided context."

    def test_normalize_grounded_answer_does_not_ground_slot_from_other_persons_fact(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_slot_other_person", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        normalized = ms._normalize_grounded_answer(
            "What temporary job did Jon take to cover expenses?",
            "Door Dash",
            {
                "context": (
                    "RETRIEVED FACTS:\n"
                    "[1] Jon got a temp job to help cover expenses.\n"
                    "[2] Gina lost her job at Door Dash this month."
                ),
                "retrieved": [
                    {"fact": "Jon got a temp job to help cover expenses."},
                    {"fact": "Gina lost her job at Door Dash this month."},
                ],
            },
        )

        assert normalized == "Not mentioned in the provided context."

    def test_normalize_grounded_answer_uses_retrieved_facts_section_when_retrieved_empty(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_slot_context_fallback", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        normalized = ms._normalize_grounded_answer(
            "What is one of her favorite movies?",
            "Eternal Sunshine of the Spotless Mind",
            {
                "context": (
                    "RETRIEVED FACTS:\n"
                    "[1] Eternal Sunshine of the Spotless Mind is one of her favorite movies.\n"
                    "--- SOURCE EPISODE RAW TEXT ---\n"
                    "She said Eternal Sunshine of the Spotless Mind is one of her favorites."
                ),
                "retrieved": [],
            },
        )

        assert normalized == "Eternal Sunshine of the Spotless Mind"

    def test_relative_temporal_answer_derives_last_week_from_episode_source_date(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_relative_anchor", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        answer = ms._derive_relative_temporal_deterministic_answer(
            query="When did Andrew start his new job as a financial analyst?",
            resolved_facts=[
                {
                    "fact": "Andrew started a new job as a Financial Analyst last week.",
                    "metadata": {"episode_id": "ep1"},
                }
            ],
            episode_lookup={
                "ep1": {"source_date": "5:17 pm on 27 March, 2023"}
            },
        )

        assert answer == "the week before March 27, 2023"

    def test_relative_temporal_answer_skips_explicit_date_queries(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_relative_anchor_skip", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        answer = ms._derive_relative_temporal_deterministic_answer(
            query="When was James bowling on March 16, 2022?",
            resolved_facts=[
                {
                    "fact": "James went bowling yesterday.",
                    "metadata": {"episode_id": "ep1"},
                }
            ],
            episode_lookup={
                "ep1": {"source_date": "9:00 pm on 17 March, 2022"}
            },
        )

        assert answer is None

    def test_duration_temporal_answer_derives_year_from_duration_fact(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_duration_anchor", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        answer = ms._derive_duration_temporal_deterministic_answer(
            query="Which year did Audrey adopt the first three of her dogs?",
            resolved_facts=[
                {
                    "fact": "Audrey has had Pepper, Precious, and Panda for 3 years.",
                    "metadata": {"episode_id": "ep1"},
                }
            ],
            episode_lookup={
                "ep1": {"source_date": "1:21 pm on 21 January, 2023"}
            },
        )

        assert answer == "2020"

    def test_normalize_grounded_answer_extracts_country_from_explicit_surface(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_country_surface", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        normalized = ms._normalize_grounded_answer(
            "In what country did Jolene's mother buy her the pendant?",
            "The context only mentions that she got the pendant in Paris, France.",
            {
                "context": (
                    "RETRIEVED FACTS:\n"
                    "[1] She gave it to me in 2010 in Paris.\n"
                    "[2] Jolene bought it in Paris in 2022."
                ),
                "retrieved": [
                    {"fact": "She gave it to me in 2010 in Paris."},
                    {"fact": "Jolene bought it in Paris, France in 2022."},
                ],
            },
        )

        assert normalized == "France"

    def test_time_scoped_acquisition_answer_derives_items_from_month_scoped_source_facts(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_monthly_acquisition", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)
        ms._all_granular = [
            {
                "id": "f1",
                "fact": "Calvin has a new mansion",
                "metadata": {"episode_id": "ep1", "episode_source_id": "src1"},
            },
            {
                "id": "f2",
                "fact": "The speaker lived in a Japanese mansion with an epic cityscape.",
                "metadata": {"episode_id": "ep2", "episode_source_id": "src1"},
            },
            {
                "id": "f3",
                "fact": "Calvin acquired a Ferrari 488 GTB",
                "metadata": {"episode_id": "ep12", "episode_source_id": "src1"},
            },
        ]
        answer = ms._derive_time_scoped_acquisition_deterministic_answer(
            query="What items did Calvin buy in March 2023?",
            query_features=extract_query_features("What items did Calvin buy in March 2023?"),
            packet={"retrieved_episode_ids": ["ep1", "ep12"]},
            episode_lookup={
                "ep1": {"source_id": "src1", "source_date": "11:53 am on 23 March, 2023"},
                "ep2": {"source_id": "src1", "source_date": "1:00 pm on 2 April, 2023"},
                "ep12": {"source_id": "src1", "source_date": "4:45 pm on 26 March, 2023"},
            },
        )

        assert answer == "Japanese mansion, Ferrari 488 GTB"

    def test_time_scoped_activity_acquisition_answer_derives_hobby_from_month_scoped_source_facts(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_monthly_hobby_acquisition", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)
        ms._all_granular = [
            {
                "id": "f1",
                "fact": "Sam loves hiking.",
                "metadata": {"episode_id": "ep1", "episode_source_id": "src1"},
            },
            {
                "id": "f2",
                "fact": "Sam was thinking about trying painting.",
                "metadata": {"episode_id": "ep1", "episode_source_id": "src1"},
            },
            {
                "id": "f3",
                "fact": "Sam has new hobbies.",
                "metadata": {"episode_id": "ep1", "episode_source_id": "src1"},
            },
        ]
        answer = ms._derive_time_scoped_activity_acquisition_deterministic_answer(
            query="Which hobby did Sam take up in May 2023?",
            query_features=extract_query_features("Which hobby did Sam take up in May 2023?"),
            packet={"retrieved_episode_ids": ["ep1"]},
            episode_lookup={
                "ep1": {"source_id": "src1", "source_date": "11:53 am on 23 May, 2023"},
            },
        )

        assert answer == "painting"


    def test_normalize_grounded_answer_treats_unspecified_slot_explanation_as_not_mentioned(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_slot_unspecified_detail", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        normalized = ms._normalize_grounded_answer(
            "What temporary job did Jon take to cover expenses?",
            (
                "Jon took a temporary job to help cover expenses while he worked on his dance studio, "
                "but the specific type of job is not detailed in the provided context."
            ),
            {
                "context": "RETRIEVED FACTS:\n[1] Jon got a temp job to help cover expenses.",
                "retrieved": [{"fact": "Jon got a temp job to help cover expenses."}],
            },
        )

        assert normalized == "Not mentioned in the provided context."

    def test_duration_temporal_answer_uses_fact_session_date_when_episode_lookup_missing(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_duration_fact_date", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        answer = ms._derive_duration_temporal_deterministic_answer(
            query="Which year did Audrey adopt the first three of her dogs?",
            resolved_facts=[
                {
                    "fact": "Audrey has had Pepper, Precious, and Panda for 3 years.",
                    "session_date": "1:10 pm on 27 March, 2023",
                }
            ],
            episode_lookup={},
        )

        assert answer == "2020"

    def test_month_temporal_answer_uses_last_month_scope_from_episode_raw_text(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_month_scope", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        answer = ms._derive_month_temporal_deterministic_answer(
            query="In which month's game did John achieve a career-high score in points?",
            resolved_facts=[
                {
                    "fact": "John scored 40 points last week, which was his personal best.",
                    "metadata": {"episode_id": "ep1"},
                }
            ],
            episode_lookup={
                "ep1": {
                    "source_date": "4:21 pm on 16 July, 2023",
                    "raw_text": (
                        "So much has happened in the last month. "
                        "Last week I scored 40 points, my highest ever."
                    ),
                }
            },
        )

        assert answer == "June 2023"

    def test_first_window_temporal_answer_uses_temporal_notes_interval(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_first_window", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        answer = ms._derive_first_window_temporal_deterministic_answer(
            query="When did Calvin first travel to Tokyo?",
            context=(
                "TEMPORAL NOTES:\n"
                "[T1] First-mention window: earliest surfaced dated support is 2023-04-20, "
                "with the previous dated episode on 2023-03-26."
            ),
        )

        assert answer == "between 26 March and 20 April 2023"

    def test_ask_preserves_temporal_deterministic_answer_for_year_queries_even_when_query_type_is_default(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        _patch_call_oai(monkeypatch, "They")
        ms = MemoryServer(str(tmp_path), "ask_year_default_gate", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        async def mock_recall(*args, **kwargs):
            return {
                "context": "RETRIEVED FACTS:\n[1] Audrey has had Pepper, Precious, and Panda for 3 years.",
                "query_type": "default",
                "recommended_profile": "fast",
                "use_tool": False,
                "recommended_prompt_type": "lookup",
                "sessions_in_context": 1,
                "total_sessions": 1,
                "coverage_pct": 100,
                "retrieval_families": ["conversation"],
                "search_family": "auto",
                "retrieved": [{"fact": "Audrey has had Pepper, Precious, and Panda for 3 years."}],
                "deterministic_answer": "2020",
                "temporal_resolution": {"query_class": "duration-anchor", "deterministic_answer": "2020"},
            }

        ms.recall = mock_recall

        result = asyncio.run(ms.ask("Which year did Audrey adopt the first three of her dogs?"))

        assert result["answer"] == "2020"
        assert result["profile_used"] == "deterministic:temporal_v1"


    def test_normalize_grounded_answer_extracts_country_from_answer_would_be_surface(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_country_answer_surface", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        normalized = ms._normalize_grounded_answer(
            "In what country did Jolene's mother buy her the pendant?",
            "If the intended subject is Deborah, the answer would be France.",
            {
                "context": "RETRIEVED FACTS:\n[1] She gave it to me in 2010 in Paris.",
                "retrieved": [{"fact": "She gave it to me in 2010 in Paris."}],
            },
        )

        assert normalized == "France"

    def test_normalize_grounded_answer_extracts_country_from_answer_prefix(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_country_answer_prefix", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        normalized = ms._normalize_grounded_answer(
            "In what country did Jolene's mother buy her the pendant?",
            "Answer: France. Jolene's mother gave it to her in Paris.",
            {
                "context": "RETRIEVED FACTS:\n[1] She gave it to me in 2010 in Paris.",
                "retrieved": [{"fact": "She gave it to me in 2010 in Paris."}],
            },
        )

        assert normalized == "France"

    def test_normalize_grounded_answer_extracts_country_from_markdown_rich_answer(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_country_markdown_surface", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        normalized = ms._normalize_grounded_answer(
            "Which country do Calvin and Dave want to meet in?",
            "Calvin and Dave plan to meet in **Boston**, which is in the **United States**.",
            {
                "context": "RETRIEVED FACTS:\n[1] Calvin will meet Dave in Boston.",
                "retrieved": [{"fact": "Calvin will meet Dave in Boston."}],
            },
        )

        assert normalized == "United States"


    def test_normalize_grounded_answer_prefers_support_country_when_answer_is_noisy(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_country_support_noisy_answer", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        normalized = ms._normalize_grounded_answer(
            "Which country do Calvin and Dave want to meet in?",
            "mentioned for their meeting",
            {
                "context": "RETRIEVED FACTS:\n[1] Calvin and Dave want to meet in Boston.",
                "retrieved": [{"fact": "Calvin and Dave want to meet in Boston."}],
            },
        )

        assert normalized == "United States"

    def test_normalize_grounded_answer_prefers_support_country_when_answer_mentions_activity(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_country_support_activity_noise", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        normalized = ms._normalize_grounded_answer(
            "Which country was Evan visiting in May 2023?",
            "ski",
            {
                "context": "RETRIEVED FACTS:\n[1] Evan was visiting Canada in May 2023 for a ski trip.",
                "retrieved": [{"fact": "Evan was visiting Canada in May 2023 for a ski trip."}],
            },
        )

        assert normalized == "Canada"

    def test_normalize_grounded_answer_extracts_country_from_support_city_when_answer_negative(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_country_support_city", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        normalized = ms._normalize_grounded_answer(
            "Which country do Calvin and Dave want to meet in?",
            "Not mentioned in the provided context.",
            {
                "context": "RETRIEVED FACTS:\n[1] Calvin will meet Dave in Boston.",
                "retrieved": [{"fact": "Calvin will meet Dave in Boston."}],
            },
        )

        assert normalized == "United States"

    def test_activity_list_deterministic_answer_collects_source_local_activity_candidates(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_activity_list", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)
        ms._all_granular = [
            {
                "fact": "Andrew has been getting into cooking more.",
                "speaker": "Andrew",
                "session_date": "8:32 pm on 3 July, 2023",
                "metadata": {"episode_source_id": "src1"},
            },
            {
                "fact": "My GF and I just had a great experience volunteering at a pet shelter on Monday.",
                "speaker": "Andrew",
                "session_date": "3:52 pm on 27 July, 2023",
                "metadata": {"episode_source_id": "src1"},
            },
            {
                "fact": "Andrew has been taking care of flowers lately.",
                "speaker": "Andrew",
                "session_date": "5:53 pm on 24 September, 2023",
                "metadata": {"episode_source_id": "src1"},
            },
            {
                "fact": "On last Tuesday, Andrew, his girlfriend Toby, and I had a night playing board games.",
                "speaker": "Andrew",
                "session_date": "4:22 pm on 13 October, 2023",
                "metadata": {"episode_source_id": "src1"},
            },
            {
                "fact": "Andrew and his girlfriend attended a wine tasting last weekend.",
                "speaker": "Andrew",
                "session_date": "10:14 am on 24 October, 2023",
                "metadata": {"episode_source_id": "src1"},
            },
            {
                "fact": "Audrey loves cooking.",
                "speaker": "Audrey",
                "session_date": "8:32 pm on 3 July, 2023",
                "metadata": {"episode_source_id": "src1"},
            },
        ]

        answer = ms._derive_activity_list_deterministic_answer(
            query="What kind of indoor activities has Andrew pursued with his girlfriend?",
            query_features=extract_query_features("What kind of indoor activities has Andrew pursued with his girlfriend?"),
            packet={"retrieved_episode_ids": ["ep1"]},
            episode_lookup={"ep1": {"source_id": "src1", "source_date": "1:00 pm on 1 November, 2023"}},
        )

        assert answer is not None
        assert "cooking" in answer
        assert "pet shelter volunteering" in answer
        assert "growing flowers" in answer
        assert "board games" in answer
        assert "wine tasting" in answer

    def test_normalize_grounded_answer_preserves_activity_list_deterministic_answer(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_activity_list_normalize", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        normalized = ms._normalize_grounded_answer(
            "What kind of indoor activities has Andrew pursued with his girlfriend?",
            "growing flowers, cooking, pet shelter volunteering, board games, wine tasting",
            {
                "context": "RETRIEVED FACTS:\n[1] Andrew likes trying new things.",
                "retrieved": [{"fact": "Andrew likes trying new things."}],
                "runtime_trace": {
                    "deterministic_answer": {
                        "kind": "activity_list",
                        "answer": "growing flowers, cooking, pet shelter volunteering, board games, wine tasting",
                    }
                },
            },
        )

        assert normalized == "growing flowers, cooking, pet shelter volunteering, board games, wine tasting"


    def test_normalize_grounded_answer_does_not_ground_slot_from_production_source_sections(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "ask_slot_source_sections", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)

        normalized = ms._normalize_grounded_answer(
            "What is one of her favorite movies?",
            "Eternal Sunshine of the Spotless Mind",
            {
                "context": (
                    "RETRIEVED FACTS:\n"
                    "[1] She likes movies.\n"
                    "--- SOURCE EPISODE RAW TEXT ---\n"
                    "Eternal Sunshine of the Spotless Mind is one of her favorite movies.\n"
                    "--- SOURCE DOCUMENT SECTIONS ---\n"
                    "Favorites section: Eternal Sunshine of the Spotless Mind."
                ),
                "retrieved": [],
            },
        )

        assert normalized == "Not mentioned in the provided context."


# ── Unit 2: Profile-based model selection ──

class TestProfileSelection:
    def test_action_item_uses_max(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        _patch_call_oai(monkeypatch, "deploy answer")
        ms = MemoryServer(str(tmp_path), "prof_max", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)
        ms._all_granular = [_make_fact(kind="action_item", complexity=0.70)]
        asyncio.run(ms.build_index())

        result = asyncio.run(ms.ask("what needs to be done?"))
        assert result["recommended_profile"] == "max"
        assert result["profile_used"] == "max"
        assert result["profile_fallback"] is False

    def test_preference_uses_fast(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        _patch_call_oai(monkeypatch, "dark mode")
        ms = MemoryServer(str(tmp_path), "prof_fast", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)
        ms._all_granular = [_make_fact(kind="preference", complexity=0.10)]
        asyncio.run(ms.build_index())

        result = asyncio.run(ms.ask("what theme?"))
        assert result["recommended_profile"] == "fast"
        assert result["profile_used"] == "fast"

    def test_missing_profile_fallback_cheapest(self, tmp_path, monkeypatch):
        """If recommended level has no profile, fall back to cheapest."""
        _patch_all(monkeypatch)
        _patch_call_oai(monkeypatch, "fallback answer")
        # Only level 5 defined — levels 1-4 missing
        sparse_profiles = {5: "max"}
        sparse_configs = {"max": PROFILE_CONFIGS["max"]}
        ms = MemoryServer(str(tmp_path), "prof_fallback", profiles=sparse_profiles, profile_configs=sparse_configs)
        ms._all_granular = [_make_fact(kind="preference", complexity=0.10)]
        asyncio.run(ms.build_index())

        result = asyncio.run(ms.ask("test"))
        # Level 1 from preference → not in sparse_profiles → fallback
        assert result["profile_fallback"] is True
        assert result["profile_used"] == "max"  # only available


# ── Unit 3: Budget exceeded handling ──

class TestBudgetExceeded:
    def test_low_budget_exceeded(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "budget1", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)
        ms._all_granular = [_make_fact()]
        asyncio.run(ms.build_index())

        # Budget of 0 → always exceeded
        result = asyncio.run(ms.ask("test", shell_budget=0.0))
        assert result["budget_exceeded"] is True
        assert result["answer"] is None

    def test_high_budget_normal(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        _patch_call_oai(monkeypatch, "normal answer")
        ms = MemoryServer(str(tmp_path), "budget2", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)
        ms._all_granular = [_make_fact()]
        asyncio.run(ms.build_index())

        # Generous budget under the new default output budget (2000 tokens).
        result = asyncio.run(ms.ask("test", shell_budget=2.0))
        assert result["budget_exceeded"] is False
        assert result["answer"] == "normal answer"

    def test_nothing_fits_best_effort_none(self, tmp_path, monkeypatch):
        """When no profile fits budget, best_effort_profile should be None."""
        _patch_all(monkeypatch)
        ms = MemoryServer(str(tmp_path), "budget3", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)
        # Large context to inflate cost
        ms._all_granular = [_make_fact()]
        ms._raw_sessions = [{"content": "x" * 100000}]
        asyncio.run(ms.build_index())

        result = asyncio.run(ms.ask("test", shell_budget=0.0))
        assert result["budget_exceeded"] is True
        assert result["best_effort_profile"] is None

    def test_no_budget_no_check(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        _patch_call_oai(monkeypatch, "no budget check")
        ms = MemoryServer(str(tmp_path), "budget4", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)
        ms._all_granular = [_make_fact()]
        asyncio.run(ms.build_index())

        # No shell_budget → no check
        result = asyncio.run(ms.ask("test"))
        assert result["budget_exceeded"] is False
        assert result["answer"] == "no budget check"


# ── Unit 4: ask() with tool-use ──

class TestAskToolUse:
    def test_use_tool_in_response_openai_compatible(self, tmp_path, monkeypatch):
        """Non-Anthropic models should attempt OpenAI-compatible tool use."""
        seen = []
        fake_client = _FakeClient(
            [_tool_call_response(1), _text_response("Answer after tool")],
            seen,
        )
        monkeypatch.setattr("src.memory._get_client", lambda model: fake_client)
        ms = MemoryServer(str(tmp_path), "tool1", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)
        ms._raw_sessions = [{"content": "Full session text"}]
        _patch_recall(ms, recommended_profile="fast", prompt_type="lookup")

        result = asyncio.run(ms.ask("test", use_tool=True))

        assert result["answer"] == "Answer after tool"
        assert result["use_tool"] is True
        assert result["tool_called"] is True
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0]["tool"] == "get_more_context"
        assert len(seen) == 2
        assert "tools" in seen[0]
        assert seen[1]["messages"][-1]["role"] == "tool"

    def test_openai_tool_path_supports_multiple_tool_rounds(self, tmp_path, monkeypatch):
        """OpenAI-compatible tool mode must keep tools enabled across follow-up rounds."""
        seen = []
        fake_client = _FakeClient(
            [_tool_call_response(1), _tool_call_response(1), _text_response("Final answer")],
            seen,
        )
        monkeypatch.setattr("src.memory._get_client", lambda model: fake_client)
        ms = MemoryServer(str(tmp_path), "tool1c", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)
        ms._raw_sessions = [{"content": "Full session text"}]
        _patch_recall(ms, recommended_profile="fast", prompt_type="summarize_with_metadata", use_tool=True)

        result = asyncio.run(ms.ask("test", use_tool=True))

        assert result["answer"] == "Final answer"
        assert result["tool_called"] is True
        assert len(result["tool_results"]) == 2
        assert len(seen) == 3
        assert "tools" in seen[1]
        assert seen[1]["tool_choice"] == "auto"

    def test_openai_tool_path_uses_model_specific_token_param(self, tmp_path, monkeypatch):
        """o-series models should use max_completion_tokens and omit temperature."""
        seen = []
        fake_client = _FakeClient([_text_response("No tool used")], seen)
        monkeypatch.setattr("src.memory._get_client", lambda model: fake_client)
        ms = MemoryServer(str(tmp_path), "tool1b")
        _patch_recall(ms, recommended_profile=None, prompt_type="lookup")

        result = asyncio.run(ms.ask(
            "test", use_tool=True, inference_model="openai/o4-mini",
        ))

        assert result["answer"] == "No tool used"
        assert result["tool_called"] is False
        assert "max_completion_tokens" in seen[0]
        assert "max_tokens" not in seen[0]
        assert "temperature" not in seen[0]

    def test_tool_use_false_explicit(self, tmp_path, monkeypatch):
        _patch_all(monkeypatch)
        _patch_call_oai(monkeypatch, "no tool")
        ms = MemoryServer(str(tmp_path), "tool2", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)
        ms._all_granular = [_make_fact()]
        asyncio.run(ms.build_index())

        result = asyncio.run(ms.ask("test", use_tool=False))
        assert result["use_tool"] is False
        assert result["tool_called"] is False


class TestAskMaxTokens:
    def test_max_tokens_none_resolves_from_profile_config(self, tmp_path, monkeypatch):
        seen = _patch_call_oai_capture(monkeypatch, "profile answer")
        ms = MemoryServer(
            str(tmp_path), "tok1",
            profiles={1: "fast"},
            profile_configs={"fast": {"model": "openai/gpt-4o-mini", "max_output_tokens": 3000}},
        )
        _patch_recall(ms, recommended_profile="fast", prompt_type="lookup")

        result = asyncio.run(ms.ask("test"))

        assert result["answer"] == "profile answer"
        assert seen["max_tokens"] == 3000

    def test_max_tokens_summarize_escalation(self, tmp_path, monkeypatch):
        seen = _patch_call_oai_capture(monkeypatch, "summary answer")
        ms = MemoryServer(
            str(tmp_path), "tok2",
            profiles={1: "fast"},
            profile_configs={"fast": {
                "model": "openai/gpt-4o-mini",
                "max_output_tokens": 2000,
                "max_output_tokens_summarize": 4096,
            }},
        )
        _patch_recall(ms, recommended_profile="fast", prompt_type="summarize_with_metadata")

        asyncio.run(ms.ask("test"))
        assert seen["max_tokens"] == 4096

    def test_explicit_max_tokens_overrides_profile(self, tmp_path, monkeypatch):
        seen = _patch_call_oai_capture(monkeypatch, "override answer")
        ms = MemoryServer(
            str(tmp_path), "tok3",
            profiles={1: "fast"},
            profile_configs={"fast": {"model": "openai/gpt-4o-mini", "max_output_tokens": 3000}},
        )
        _patch_recall(ms, recommended_profile="fast", prompt_type="lookup")

        asyncio.run(ms.ask("test", max_tokens=1000))
        assert seen["max_tokens"] == 1000

    def test_no_profile_falls_back_to_2000(self, tmp_path, monkeypatch):
        seen = _patch_call_oai_capture(monkeypatch, "fallback answer")
        ms = MemoryServer(str(tmp_path), "tok4")
        _patch_recall(ms, recommended_profile=None, prompt_type="lookup")

        asyncio.run(ms.ask("test", inference_model="openai/gpt-4o-mini"))
        assert seen["max_tokens"] == 2000


# ── Profile helper tests ──

class TestProfileHelpers:
    def test_has_profiles(self, tmp_path):
        ms = MemoryServer(str(tmp_path), "h1", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)
        assert ms._has_profiles() is True

    def test_has_profiles_false(self, tmp_path):
        ms = MemoryServer(str(tmp_path), "h2")
        assert ms._has_profiles() is False

    def test_get_profile_config(self, tmp_path):
        ms = MemoryServer(str(tmp_path), "h3", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)
        cfg = ms._get_profile_config("fast")
        assert cfg is not None
        assert cfg["model"] == "openai/gpt-4o-mini"
        assert cfg["input_cost_per_1k"] == 0.15

    def test_get_profile_config_missing(self, tmp_path):
        ms = MemoryServer(str(tmp_path), "h4", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)
        assert ms._get_profile_config("nonexistent") is None

    def test_list_profile_names(self, tmp_path):
        ms = MemoryServer(str(tmp_path), "h5", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)
        names = ms._list_profile_names()
        assert "fast" in names
        assert "balanced" in names
        assert "max" in names

    def test_cheapest_profile(self, tmp_path):
        ms = MemoryServer(str(tmp_path), "h6", profiles=PROFILES, profile_configs=PROFILE_CONFIGS)
        assert ms._cheapest_profile() == "fast"
