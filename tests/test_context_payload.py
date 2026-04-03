"""Context payload runtime tests for recall()/ask() middle-ground contract."""

import asyncio

import numpy as np

from src.memory import MemoryServer
import src.mcp_server as mcp_mod


DIM = 3072

PROFILES = {1: "fast", 2: "fast", 3: "balanced", 4: "max", 5: "max"}
PROFILE_CONFIGS = {
    "fast": {
        "model": "openai/gpt-4o-mini",
        "context_window": 128000,
        "max_output_tokens": 2000,
        "thinking_overhead": 0,
        "input_cost_per_1k": 0.15,
        "output_cost_per_1k": 0.60,
    },
    "balanced": {
        "model": "google/gemini-2.0-flash",
        "context_window": 128000,
        "max_output_tokens": 2000,
        "thinking_overhead": 0,
        "input_cost_per_1k": 0.10,
        "output_cost_per_1k": 0.40,
    },
    "max": {
        "model": "anthropic/claude-sonnet-4-6",
        "context_window": 200000,
        "max_output_tokens": 4096,
        "thinking_overhead": 0,
        "input_cost_per_1k": 3.0,
        "output_cost_per_1k": 15.0,
    },
}


def _patch_all(monkeypatch):
    async def mock_extract(**kwargs):
        sn = kwargs.get("session_num", 1)
        return ("conv", sn, "2024-06-01", [
            {"id": f"f{sn}_0", "fact": f"Fact {sn}", "kind": "fact",
             "entities": [], "tags": [], "session": sn}], [])

    async def mock_consolidate(**kwargs):
        return ("conv", 1, "2024-06-01", [])

    async def mock_cross(**kwargs):
        return ("conv", "e", [])

    async def mock_embed(texts, **kw):
        return np.random.randn(len(texts), DIM).astype(np.float32)

    async def mock_embed_q(text, **kw):
        return np.random.randn(DIM).astype(np.float32)

    monkeypatch.setattr("src.memory.extract_session", mock_extract)
    monkeypatch.setattr("src.memory.consolidate_session", mock_consolidate)
    monkeypatch.setattr("src.memory.cross_session_entity", mock_cross)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda f, l: None)
    monkeypatch.setattr("src.memory.embed_texts", mock_embed)
    monkeypatch.setattr("src.memory.embed_query", mock_embed_q)


def _make_server(tmp_path, monkeypatch, *, profiles=True, inference_leaf_plugins=None):
    _patch_all(monkeypatch)
    ms = MemoryServer(
        str(tmp_path),
        "ctx_payload",
        profiles=PROFILES if profiles else None,
        profile_configs=PROFILE_CONFIGS if profiles else None,
        inference_leaf_plugins=inference_leaf_plugins,
    )
    asyncio.run(ms.store(
        "User: We chose anodized aluminum for the casing.\nAssistant: Noted.",
        session_num=1,
        session_date="2024-06-01",
    ))
    ms._all_granular[0]["fact"] = "We chose anodized aluminum for the casing."
    ms._all_granular[0]["kind"] = "decision"
    ms._all_granular[0]["status"] = "active"
    ms._all_granular[0]["_session_content_complexity"] = 0.55
    asyncio.run(ms.build_index())
    return ms


def test_recall_payload_present_with_profiles(tmp_path, monkeypatch):
    ms = _make_server(tmp_path, monkeypatch, profiles=True)
    result = asyncio.run(ms.recall("What material did we choose for the casing?"))
    assert "payload" in result
    assert "payload_meta" in result
    assert result["payload"]["model"] == PROFILE_CONFIGS["balanced"]["model"]
    assert result["payload_meta"]["provider_family"] == "google"


def test_recall_payload_absent_without_profiles(tmp_path, monkeypatch):
    ms = _make_server(tmp_path, monkeypatch, profiles=False)
    result = asyncio.run(ms.recall("What material did we choose for the casing?"))
    assert "payload" not in result
    assert "payload_meta" not in result
    assert "complexity_hint" in result


def test_payload_messages_contain_context_and_question(tmp_path, monkeypatch):
    ms = _make_server(tmp_path, monkeypatch, profiles=True)
    result = asyncio.run(ms.recall("What material did we choose for the casing?"))
    message = result["payload"]["messages"][0]["content"]
    assert "anodized aluminum" in message
    assert "What material did we choose for the casing?" in message


def test_payload_messages_use_list_set_prompt_for_list_queries(tmp_path, monkeypatch):
    ms = _make_server(tmp_path, monkeypatch, profiles=False)
    messages = ms._build_payload_messages(
        prompt_type="default",
        context="RETRIEVED FACTS:\n[1] Gina is working on a wildlife documentary project.\n\nRAW SLOT CANDIDATES:\n[Q1] wildlife documentary",
        query="What kind of project is Gina doing?",
        recall_result={
            "sessions_in_context": 1,
            "total_sessions": 8,
            "coverage_pct": 12,
            "query_operator_plan": {
                "slot_query": {"enabled": True},
                "list_set": {"enabled": False},
                "ordinal": {"enabled": False},
                "commonality": {"enabled": False},
                "compare_diff": {"enabled": False},
            },
        },
        speakers="User and Assistant",
    )
    content = messages[0]["content"]
    assert "slot-filling or attribute question" in content
    assert "RAW SLOT CANDIDATES" in content




def test_payload_temperature_from_profile_or_zero(tmp_path, monkeypatch):
    ms = _make_server(tmp_path, monkeypatch, profiles=True)
    ms._profile_configs["balanced"]["temperature"] = 0.3
    result = asyncio.run(ms.recall("What material did we choose for the casing?"))
    assert result["payload"]["temperature"] == 0.3


def test_payload_tools_present_only_when_use_tool(tmp_path):
    ms = MemoryServer(
        str(tmp_path),
        "ctx_payload_tools",
        profiles={1: "fast"},
        profile_configs={"fast": PROFILE_CONFIGS["fast"]},
    )
    recall_result = {
        "context": "Context block",
        "_context_packet": {
            "tier1": [],
            "tier2": [],
            "tier3": [{"text": "Context block", "rank": 0, "source": "fact"}],
            "tier4": [],
        },
        "query_type": "default",
        "recommended_profile": "fast",
        "recommended_prompt_type": "lookup",
        "use_tool": True,
        "sessions_in_context": 1,
        "total_sessions": 1,
        "coverage_pct": 100,
    }
    payload, meta = ms._build_payload(query="Need more detail", recall_result=recall_result)
    assert "tools" in payload
    assert meta["use_tool"] is True

    payload2, meta2 = ms._build_payload(
        query="Need more detail",
        recall_result=recall_result,
        use_tool=False,
    )
    assert "tools" not in payload2
    assert meta2["use_tool"] is False


def test_payload_provider_specific_shapes(tmp_path):
    ms = MemoryServer(
        str(tmp_path),
        "ctx_provider_shapes",
        profiles={1: "fast"},
        profile_configs={"fast": PROFILE_CONFIGS["fast"]},
    )
    messages = [{"role": "user", "content": "hello"}]
    openai_payload, _ = ms._build_provider_payload(
        model="openai/gpt-4o-mini",
        messages=messages,
        max_tokens=200,
        temperature=0,
        use_tool=True,
    )
    anthropic_payload, _ = ms._build_provider_payload(
        model="anthropic/claude-sonnet-4-6",
        messages=messages,
        max_tokens=200,
        temperature=0,
        use_tool=True,
    )
    assert openai_payload["tools"][0]["type"] == "function"
    assert "function" in openai_payload["tools"][0]
    assert anthropic_payload["tools"][0]["input_schema"]["type"] == "object"


def test_context_for_legacy_drops_payload(tmp_path, monkeypatch):
    ms = _make_server(tmp_path, monkeypatch, profiles=True)
    result = asyncio.run(ms.context_for("What material did we choose?", token_budget=4000))
    assert "payload" not in result
    assert "payload_meta" not in result


def test_memory_recall_mcp_returns_payload_when_not_truncated(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    mcp_mod.data_dir = str(tmp_path)
    mcp_mod.registry.clear()
    ms = MemoryServer(
        str(tmp_path),
        "mcp_payload",
        profiles=PROFILES,
        profile_configs=PROFILE_CONFIGS,
    )
    asyncio.run(ms.store(
        "User: Short fact.\nAssistant: saved.",
        session_num=1,
        session_date="2024-06-01",
    ))
    asyncio.run(ms.build_index())
    mcp_mod.registry["mcp_payload"] = ms
    result = asyncio.run(mcp_mod.memory_recall(
        key="mcp_payload",
        query="What is stored?",
        token_budget=200000,
    ))
    assert "payload" in result
    assert "payload_meta" in result


def test_memory_recall_mcp_drops_payload_when_truncated(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    mcp_mod.data_dir = str(tmp_path)
    mcp_mod.registry.clear()
    ms = _make_server(tmp_path, monkeypatch, profiles=True)
    mcp_mod.registry["mcp_trunc"] = ms
    result = asyncio.run(mcp_mod.memory_recall(
        key="mcp_trunc",
        query="What material did we choose for the casing?",
        token_budget=1,
    ))
    assert "payload" not in result
    assert "payload_meta" not in result


def test_truncation_preserves_tier1_and_drops_raw_first(tmp_path):
    ms = MemoryServer(
        str(tmp_path),
        "ctx_trunc",
        profiles={1: "fast"},
        profile_configs={"fast": {
            "model": "openai/gpt-4o-mini",
            "context_window": 400,
            "max_output_tokens": 20,
            "thinking_overhead": 0,
        }},
    )
    recall_result = {
        "context": "",
        "_context_packet": {
            "tier1": [{"text": "[1] critical decision", "rank": 0, "source": "fact"}],
            "tier2": [],
            "tier3": [{"text": "[2] ordinary fact", "rank": 1, "source": "fact"}],
            "tier4": [{"text": "[Raw S1]\n" + ("x" * 800), "rank": 2, "source": "raw"}],
        },
        "query_type": "default",
        "recommended_profile": "fast",
        "recommended_prompt_type": "lookup",
        "use_tool": False,
        "sessions_in_context": 1,
        "total_sessions": 1,
        "coverage_pct": 100,
    }
    payload, meta = ms._build_payload(query="test", recall_result=recall_result)
    assert "[1] critical decision" in recall_result["context"]
    assert "[Raw S1]" not in recall_result["context"]
    assert meta["truncation"]["removed"]["tier4"] >= 1
    assert meta["budget_exceeded"] is False


def test_budget_exceeded_when_tier1_alone_too_large(tmp_path):
    ms = MemoryServer(
        str(tmp_path),
        "ctx_budget_exceeded",
        profiles={1: "fast"},
        profile_configs={"fast": {
            "model": "openai/gpt-4o-mini",
            "context_window": 80,
            "max_output_tokens": 20,
            "thinking_overhead": 0,
        }},
    )
    recall_result = {
        "context": "",
        "_context_packet": {
            "tier1": [{"text": "critical " * 200, "rank": 0, "source": "fact"}],
            "tier2": [],
            "tier3": [],
            "tier4": [],
        },
        "query_type": "default",
        "recommended_profile": "fast",
        "recommended_prompt_type": "lookup",
        "use_tool": False,
        "sessions_in_context": 1,
        "total_sessions": 1,
        "coverage_pct": 100,
    }
    _payload, meta = ms._build_payload(query="test", recall_result=recall_result)
    assert meta["budget_exceeded"] is True


def test_routing_summarize_sets_use_tool_in_payload(tmp_path):
    """When recommended_prompt_type is summarize_with_metadata, payload gets tools."""
    ms = MemoryServer(
        str(tmp_path),
        "ctx_routing_summarize",
        profiles={1: "fast"},
        profile_configs={"fast": PROFILE_CONFIGS["fast"]},
    )
    recall_result = {
        "context": "Context block",
        "_context_packet": {
            "tier1": [],
            "tier2": [],
            "tier3": [{"text": "Context block", "rank": 0, "source": "fact"}],
            "tier4": [],
        },
        "query_type": "summarize",
        "recommended_profile": "fast",
        "recommended_prompt_type": "summarize_with_metadata",
        "use_tool": True,
        "sessions_in_context": 1,
        "total_sessions": 1,
        "coverage_pct": 100,
    }
    payload, meta = ms._build_payload(query="Summarize the project", recall_result=recall_result)
    assert "tools" in payload
    assert meta["use_tool"] is True
    assert meta["prompt_type"] == "summarize_with_metadata"


def test_routing_default_no_tools_in_payload(tmp_path):
    """When use_tool is False, payload has no tools and prompt_type passes through."""
    ms = MemoryServer(
        str(tmp_path),
        "ctx_routing_default",
        profiles={1: "fast"},
        profile_configs={"fast": PROFILE_CONFIGS["fast"]},
    )
    recall_result = {
        "context": "Context block",
        "_context_packet": {
            "tier1": [],
            "tier2": [],
            "tier3": [{"text": "Context block", "rank": 0, "source": "fact"}],
            "tier4": [],
        },
        "query_type": "default",
        "recommended_profile": "fast",
        "recommended_prompt_type": "lookup",
        "use_tool": False,
        "sessions_in_context": 1,
        "total_sessions": 1,
        "coverage_pct": 100,
    }
    payload, meta = ms._build_payload(query="What happened?", recall_result=recall_result)
    assert "tools" not in payload
    assert meta["use_tool"] is False


def test_ask_uses_existing_payload_when_no_overrides(tmp_path):
    ms = MemoryServer(
        str(tmp_path),
        "ctx_ask_reuse",
        profiles={1: "fast"},
        profile_configs={"fast": PROFILE_CONFIGS["fast"]},
    )

    async def fake_recall(*args, **kwargs):
        return {
            "context": "Context block",
            "query_type": "default",
            "recommended_profile": "fast",
            "use_tool": False,
            "recommended_prompt_type": "lookup",
            "sessions_in_context": 1,
            "total_sessions": 1,
            "coverage_pct": 100,
            "payload": {
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 100,
                "temperature": 0,
                "seed": 42,
            },
            "payload_meta": {
                "profile_used": "fast",
                "profile_fallback": False,
                "context_tokens": 10,
                "message_tokens_est": 20,
                "tool_tokens_est": 0,
                "memory_budget": 1000,
                "budget_exceeded": False,
                "prompt_type": "lookup",
                "use_tool": False,
                "truncation": None,
                "provider": "openai",
                "provider_family": "openai_compatible",
            },
        }

    async def fake_send(payload, *, caller_id=None):
        return "payload answer", False, []

    def fail_build_payload(**kwargs):
        raise AssertionError("_build_payload should not be called when ready payload exists")

    ms.recall = fake_recall
    ms._send_payload = fake_send
    ms._build_payload = fail_build_payload

    result = asyncio.run(ms.ask("test"))
    assert result["answer"] == "payload answer"
