"""Tests for src/mcp_server.py — GOSH Memory MCP Server."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time

import numpy as np
import pytest

import src.mcp_server as mod
from src.mcp_server import (
    _get_memory,
    courier_subscribe,
    courier_unsubscribe,
    mcp,
    memory_ask,
    memory_build_index,
    memory_flush,
    memory_get,
    memory_ingest,
    memory_ingest_document,
    memory_list,
    memory_query,
    memory_recall,
    memory_reextract,
    memory_write,
    memory_write_status,
    memory_stats,
    memory_store,
    sse_cleanup,
    sse_endpoint,
)

DIM = 3072


# ── Patches ──

def _patch_extraction(monkeypatch):
    async def mock_extract_session(**kwargs):
        sn = kwargs.get("session_num", 1)
        facts = [
            {"id": f"f{i}", "fact": f"Fact {i}", "kind": "event",
             "entities": ["Alice"], "tags": ["test"], "session": sn}
            for i in range(3)
        ]
        return ("conv", sn, "2024-06-01", facts, [])

    async def mock_consolidate_session(**kwargs):
        return ("conv", 1, "2024-06-01", [
            {"id": "c0", "fact": "Consolidated", "kind": "summary",
             "entities": ["Alice"], "tags": []}
        ])

    async def mock_cross_session_entity(**kwargs):
        return ("conv", "alice", [
            {"id": "x0", "fact": "Cross-session", "kind": "profile",
             "entities": ["Alice"], "tags": []}
        ])

    monkeypatch.setattr("src.memory.extract_session", mock_extract_session)
    monkeypatch.setattr("src.memory.consolidate_session", mock_consolidate_session)
    monkeypatch.setattr("src.memory.cross_session_entity", mock_cross_session_entity)


def _patch_embeddings(monkeypatch):
    async def mock_embed_texts(texts, **kw):
        return np.random.randn(len(texts), DIM).astype(np.float32)

    async def mock_embed_query(text, **kw):
        return np.random.randn(DIM).astype(np.float32)

    monkeypatch.setattr("src.memory.embed_texts", mock_embed_texts)
    monkeypatch.setattr("src.memory.embed_query", mock_embed_query)


def _patch_resolve_supersession(monkeypatch):
    monkeypatch.setattr("src.memory.resolve_supersession", lambda f, l: None)


def _patch_all(monkeypatch):
    _patch_extraction(monkeypatch)
    _patch_embeddings(monkeypatch)
    _patch_resolve_supersession(monkeypatch)


@pytest.fixture(autouse=True)
def reset_state(tmp_path, monkeypatch):
    """Reset module state before each test."""
    mod.data_dir = str(tmp_path)
    mod.registry.clear()
    mod.courier_registry.clear()
    mod.connections.clear()
    mod.sub_to_conn.clear()
    mod._active_connections.clear()
    _patch_all(monkeypatch)
    yield
    # Stop couriers — set flag directly (event loop is closed after asyncio.run)
    for c in mod.courier_registry.values():
        c._running = False


# ── Tests ──

def test_list_tools_returns_all():
    result = asyncio.run(mcp.list_tools())
    names = {t.name for t in result}
    expected = {
        "memory_store", "memory_write", "memory_write_status", "memory_recall", "memory_ingest_document", "memory_ingest",
        "memory_ingest_asserted_facts",
        "memory_build_index", "memory_flush", "memory_stats",
        "memory_reextract", "memory_list", "memory_get",
        "courier_subscribe", "courier_unsubscribe",
        "memory_store_secret", "memory_get_secret",
        "memory_import", "memory_import_history",
        "memory_list_prompts", "memory_get_prompt", "memory_set_prompt",
        "memory_set_config", "memory_get_config",
        "memory_set_profiles", "memory_get_profiles",
        "membership_register", "membership_unregister", "membership_list",
        "memory_ask",
        "memory_edit", "memory_retract", "memory_purge", "memory_get_versions",
        "memory_redact",
        "memory_query", "memory_set_schema", "memory_get_schema",
        "memory_mal_configure", "memory_mal_feedback", "memory_mal_trigger",
        "memory_mal_status", "memory_mal_list_feedback", "memory_mal_get_artifact",
        "memory_mal_rollback",
    }
    assert names == expected


def test_memory_ingest_text_routes_and_tags(tmp_path):
    mod.data_dir = str(tmp_path)
    mod.registry.clear()
    result = asyncio.run(memory_ingest(
        key="ingest_text",
        text="User: hello\nAssistant: hi",
        session_num=1,
        session_date="2024-06-01",
    ))
    assert result["source_family"] == "conversation"
    server = mod.registry["ingest_text"]
    assert server._all_granular
    assert server._episode_corpus["documents"]


def test_memory_ingest_step_trace_preserves_document_family_even_with_session_metadata(tmp_path):
    mod.data_dir = str(tmp_path)
    mod.registry.clear()
    result = asyncio.run(memory_ingest(
        key="ingest_step_trace",
        text="[Step 0] Action: left\nObservation: moved left",
        session_num=1,
        session_date="2024-06-01",
        speakers="Game",
    ))
    assert result["source_family"] == "document"
    assert "step_trace_text" in result["detection_evidence"]["signals"]
    assert "conversation_fields_present" in result["detection_evidence"]["signals"]
    server = mod.registry["ingest_step_trace"]
    assert server._all_granular
    assert server._source_records[result["source_id"]]["family"] == "document"
    assert server._episode_corpus["documents"]


def test_registry_creates_on_demand():
    assert len(mod.registry) == 0
    asyncio.run(memory_store(
        key="test_key", content="Hello", session_num=1,
        session_date="2024-06-01",
    ))
    assert "test_key" in mod.registry


def test_registry_reuses_instance():
    asyncio.run(memory_store(
        key="reuse", content="First", session_num=1,
        session_date="2024-06-01",
    ))
    server1 = mod.registry["reuse"]
    asyncio.run(memory_store(
        key="reuse", content="Second", session_num=2,
        session_date="2024-06-02",
    ))
    server2 = mod.registry["reuse"]
    assert server1 is server2


def test_registry_respects_env_tier_mode(monkeypatch):
    monkeypatch.setenv("GOSH_MEMORY_TIER_MODE", "lazy_tier2_3")
    server = _get_memory("tier_mode_env")
    assert server._tier_mode == "lazy_tier2_3"


def test_memory_tools_reject_empty_key():
    result = asyncio.run(memory_recall(key="", query="test"))
    assert result["code"] == "VALIDATION_ERROR"
    assert "non-empty" in result["error"]


def test_memory_ask_does_not_override_memory_profiles_with_server_default(monkeypatch):
    monkeypatch.setattr(mod.cfg, "inference_model", "anthropic/claude-sonnet-4-6")
    server = _get_memory("ask_profile_default")
    captured = {}

    async def _fake_ask(**kwargs):
        captured.update(kwargs)
        return {"answer": "ok", "profile_used": "fast"}

    monkeypatch.setattr(server, "_has_profiles", lambda: True)
    monkeypatch.setattr(server, "ask", _fake_ask)

    result = asyncio.run(memory_ask(key="ask_profile_default", query="test"))

    assert result["answer"] == "ok"
    assert captured["inference_model"] is None


def test_memory_query_returns_truncated_fact_preview():
    server = _get_memory("query_preview")
    long_text = "A" * 1500
    server._all_granular = [{
        "id": "fact-1",
        "fact": long_text,
        "kind": "task_result",
        "session": 1,
        "conv_id": "query_preview",
        "owner_id": "system",
        "read": ["agent:PUBLIC"],
        "write": ["agent:PUBLIC"],
        "created_at": "2024-01-01T00:00:00+00:00",
        "metadata": {"task_id": "task-1"},
    }]

    result = asyncio.run(memory_query(key="query_preview", filter={"metadata.task_id": "task-1"}))

    assert result["total"] == 1
    assert result["facts"][0]["fact_truncated"] is True
    assert len(result["facts"][0]["fact"]) < len(long_text)


def test_per_call_tagging_and_restore():
    asyncio.run(memory_store(
        key="tag_test", content="Private info", session_num=1,
        session_date="2024-06-01",
        agent_id="agent_x", scope="agent-private", swarm_id="swarm_1",
    ))
    server = mod.registry["tag_test"]
    # Facts should be tagged with per-call values
    last_fact = server._all_granular[-1]
    assert last_fact["agent_id"] == "agent_x"
    assert last_fact["scope"] == "agent-private"
    assert last_fact["swarm_id"] == "swarm_1"
    # Server defaults restored
    assert server.agent_id == "default"
    assert server.scope == "swarm-shared"
    assert server.swarm_id == "default"


def test_memory_flush_blocking(monkeypatch):
    # Store some facts first
    asyncio.run(memory_store(
        key="flush_test", content="Hello", session_num=1,
        session_date="2024-06-01",
    ))

    flush_called = {"called": False}

    async def mock_flush():
        flush_called["called"] = True
        return {"rebuilt": True, "total_consolidated": 3, "total_cross_session": 1}

    mod.registry["flush_test"].flush_background = mock_flush

    result = asyncio.run(memory_flush(key="flush_test"))
    assert result == {"rebuilt": True, "total_consolidated": 3, "total_cross_session": 1}
    assert flush_called["called"]


def test_memory_build_index_blocking():
    asyncio.run(memory_store(
        key="idx_test", content="Hello", session_num=1,
        session_date="2024-06-01",
    ))
    result = asyncio.run(memory_build_index(key="idx_test"))
    assert "granular" in result
    assert "consolidated" in result
    assert "cross_session" in result
    assert result["granular"] >= 3


def test_memory_build_index_no_facts_error():
    # Create empty server
    _get_memory("empty_key")
    result = asyncio.run(memory_build_index(key="empty_key"))
    assert result["code"] == "NO_FACTS"
    assert "error" in result


def test_memory_recall_exposes_actual_injected_episode_ids():
    asyncio.run(memory_store(
        key="mcp_recall_eps",
        content="User: Alice has 3 apples.\nAssistant: noted",
        session_num=1,
        session_date="2024-06-01",
    ))
    asyncio.run(memory_build_index(key="mcp_recall_eps"))

    result = asyncio.run(memory_recall(key="mcp_recall_eps", query="How many apples does Alice have?"))

    assert "actual_injected_episode_ids" in result
    assert result["actual_injected_episode_ids"]
    assert result["actual_injected_episode_ids"] == result["runtime_trace"]["selection"]["actual_injected_episode_ids"]
    assert result["actual_injected_episode_ids"] == ["mcp_recall_eps_e0001"]


def test_memory_recall_exposes_episode_selection_trace():
    asyncio.run(memory_store(
        key="mcp_recall_trace",
        content="User: Alice has 3 apples and 2 pears.\nAssistant: noted",
        session_num=1,
        session_date="2024-06-01",
    ))
    asyncio.run(memory_build_index(key="mcp_recall_trace"))

    result = asyncio.run(memory_recall(key="mcp_recall_trace", query="How many apples does Alice have?"))

    assert result["telemetry_version"] == 1
    assert "retrieved_episode_ids" in result
    assert "selection_scores" in result
    assert result["retrieved_episode_ids"] == result["runtime_trace"]["selection"]["retrieved_episode_ids"]
    assert result["selection_scores"] == result["runtime_trace"]["selection"]["selection_scores"]


def test_memory_stats_exposes_validity_and_cost_summary():
    asyncio.run(memory_store(
        key="mcp_stats",
        content="User: Alice has 3 apples.\nAssistant: noted",
        session_num=1,
        session_date="2024-06-01",
    ))
    asyncio.run(memory_build_index(key="mcp_stats"))

    result = asyncio.run(memory_stats(key="mcp_stats"))

    assert result["telemetry_version"] == 1
    assert result["raw_sessions_count"] == 1
    assert result["source_records_count"] == 1
    assert result["all_raw_sessions_active"] is True
    assert result["logical_source_count"] == 1
    assert result["part_source_count"] == 0
    assert result["raw_session_status_counts"]["active"] == 1
    assert result["process_cost_scope"] == "process"
    assert "process_cost_summary" in result
    assert set(result["process_cost_summary"].keys()) == {
        "input_tokens",
        "output_tokens",
        "embed_tokens",
        "cost_usd",
        "calls",
    }


def test_courier_subscribe_returns_sub_id():
    async def _test():
        # C2: register connection first via SSE so it's in _active_connections
        mod._active_connections["test_conn"] = "test"
        mod.connections["test_conn"] = asyncio.Queue()
        result = await courier_subscribe(
            key="sub_test",
            filter={"kind": "event"},
            connection_id="test_conn",
        )
        return result

    result = asyncio.run(_test())
    sub_id = result["sub_id"]
    assert sub_id.startswith("sub_")
    assert mod.sub_to_conn[sub_id] == "test_conn"


def test_courier_subscribe_rejects_unknown_connection():
    """C2: courier_subscribe must reject unknown connection_ids."""
    async def _test():
        result = await courier_subscribe(
            key="sub_test",
            filter={},
            connection_id="hijacked_conn",
        )
        return result

    result = asyncio.run(_test())
    assert result["code"] == "INVALID_CONNECTION"


def test_courier_unsubscribe_idempotent():
    result = asyncio.run(courier_unsubscribe(sub_id="sub_nonexistent"))
    assert result == {"status": "ok"}


def test_sse_sends_connected_event():
    async def _test():
        response = await sse_endpoint(None)
        assert len(mod.connections) == 1
        conn_id = list(mod.connections.keys())[0]
        queue = mod.connections[conn_id]
        event = queue.get_nowait()
        assert event["type"] == "connected"
        assert event["connection_id"] == conn_id
        assert len(conn_id) > 0

    asyncio.run(_test())


def test_sse_cleanup_removes_subscriptions():
    async def _test():
        # Set up connection
        await sse_endpoint(None)
        conn_id = list(mod.connections.keys())[0]

        # Manually register subscription
        sub_id = "sub_cleanup_test"
        mod.sub_to_conn[sub_id] = conn_id

        assert conn_id in mod.connections
        assert sub_id in mod.sub_to_conn

        # Trigger cleanup
        await sse_cleanup(conn_id)

        assert conn_id not in mod.connections
        assert sub_id not in mod.sub_to_conn

    asyncio.run(_test())


def test_unknown_tool_returns_error():
    async def _test():
        try:
            result = await mcp.call_tool("nonexistent.tool", {})
            # If it returns instead of raising, check for error
            if isinstance(result, dict):
                return result
            text = result[0].text if result else ""
            return {"text": text}
        except Exception as e:
            return {"error": str(e)}

    result = asyncio.run(_test())
    assert "error" in result or "Unknown tool" in str(result)


def test_memory_recall_returns_token_estimate():
    """memory_recall must include token_estimate in response."""
    asyncio.run(memory_store(
        key="tok_est", content="Alice met Bob on Monday.",
        session_num=1, session_date="2024-06-01",
    ))
    result = asyncio.run(memory_recall(
        key="tok_est", query="What happened?",
    ))
    assert "token_estimate" in result
    assert isinstance(result["token_estimate"], int)
    assert result["token_estimate"] == len(result["context"]) // 4


def test_memory_write_exposes_raw_recall_and_status(tmp_path):
    mod.data_dir = str(tmp_path)
    mod.registry.clear()
    result = asyncio.run(memory_write(
        key="write_raw",
        message_id="msg-1",
        session_id="sess-1",
        content="Fresh write about mango orchards",
        content_family="chat",
        timestamp_ms=1712000000000,
    ))
    assert result["message_id"] == "msg-1"
    assert result["extraction_state"] == "pending"

    status = asyncio.run(memory_write_status(key="write_raw", message_id="msg-1"))
    assert status["extraction_state"] == "pending"

    recall = asyncio.run(memory_recall(key="write_raw", query="mango orchards"))
    assert recall["context"].startswith("RECENT RAW WRITES:")
    assert recall["raw_recall_count"] == 1
    assert "mango orchards" in recall["context"].lower()


def test_memory_write_worker_promotes_chat_entry_into_extracted_memory(tmp_path):
    mod.data_dir = str(tmp_path)
    mod.registry.clear()
    asyncio.run(memory_write(
        key="write_worker",
        message_id="msg-2",
        session_id="sess-2",
        content="User: Alice planted tulips.\nAssistant: noted",
        content_family="chat",
        timestamp_ms=1712000001000,
    ))
    server = mod.registry["write_worker"]

    processed = asyncio.run(server.process_write_log_once())
    assert processed == 1

    status = asyncio.run(memory_write_status(key="write_worker", message_id="msg-2"))
    assert status["extraction_state"] == "complete"
    assert server._raw_sessions
    assert any(rs.get("message_id") == "msg-2" for rs in server._raw_sessions)
    assert server._all_granular


def test_memory_write_rejects_unknown_content_family(tmp_path):
    mod.data_dir = str(tmp_path)
    mod.registry.clear()
    result = asyncio.run(memory_write(
        key="write_bad_family",
        message_id="msg-bad",
        session_id="sess-bad",
        content="hello",
        content_family="weird",
        timestamp_ms=1712000002000,
    ))
    assert result["code"] == "VALIDATION_ERROR"
    assert "Unsupported content_family" in result["error"]



def test_memory_write_and_status_support_concurrent_calls(tmp_path):
    mod.data_dir = str(tmp_path)
    mod.registry.clear()
    server = _get_memory("write_concurrent")
    mod._ensure_instance_config(server, "agent:agent-a")
    total = 6

    def _write(i: int) -> dict:
        return asyncio.run(memory_write(
            key="write_concurrent",
            message_id=f"msg-{i}",
            session_id=f"sess-{i}",
            content=f"parallel kiwi write {i}",
            content_family="chat",
            timestamp_ms=1712000003000 + i,
            agent_id="agent-a",
            swarm_id="sw1",
            scope="agent-private",
        ))

    with ThreadPoolExecutor(max_workers=6) as pool:
        write_results = list(pool.map(_write, range(total)))

    assert all(result["inserted"] is True for result in write_results)

    def _status(i: int) -> dict:
        return asyncio.run(memory_write_status(
            key="write_concurrent",
            message_id=f"msg-{i}",
            agent_id="agent-a",
            swarm_id="sw1",
        ))

    with ThreadPoolExecutor(max_workers=6) as pool:
        statuses = list(pool.map(_status, range(total)))

    assert all(status["extraction_state"] == "pending" for status in statuses)
    recall = asyncio.run(memory_recall(
        key="write_concurrent",
        query="parallel kiwi",
        agent_id="agent-a",
        swarm_id="sw1",
    ))
    assert recall["raw_recall_count"] == total
    assert "parallel kiwi write 0" in recall["context"].lower()


def test_memory_write_raw_recall_respects_acl_after_concurrent_writes(tmp_path):
    mod.data_dir = str(tmp_path)
    mod.registry.clear()
    server = _get_memory("write_acl")
    mod._ensure_instance_config(server, "agent:agent-a")
    mod._expand_instance_acl_for_scope(server, scope="swarm-shared", swarm_id="sw1")

    writes = [
        {
            "message_id": "shared-1",
            "session_id": "sess-shared",
            "content": "sharedapple orchard note",
            "agent_id": "agent-a",
            "scope": "swarm-shared",
        },
        {
            "message_id": "private-a",
            "session_id": "sess-private-a",
            "content": "alphaapple orchard note",
            "agent_id": "agent-a",
            "scope": "agent-private",
        },
        {
            "message_id": "private-b",
            "session_id": "sess-private-b",
            "content": "betaapple orchard note",
            "agent_id": "agent-b",
            "scope": "agent-private",
        },
    ]

    def _write(payload: dict) -> dict:
        return asyncio.run(memory_write(
            key="write_acl",
            message_id=payload["message_id"],
            session_id=payload["session_id"],
            content=payload["content"],
            content_family="chat",
            timestamp_ms=1712000004000,
            agent_id=payload["agent_id"],
            swarm_id="sw1",
            scope=payload["scope"],
        ))

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(_write, writes))

    assert all(result["inserted"] is True for result in results)

    recall_a = asyncio.run(memory_recall(key="write_acl", query="orchard", agent_id="agent-a", swarm_id="sw1"))
    recall_b = asyncio.run(memory_recall(key="write_acl", query="orchard", agent_id="agent-b", swarm_id="sw1"))
    recall_c = asyncio.run(memory_recall(key="write_acl", query="orchard", agent_id="agent-c", swarm_id="sw1"))

    assert recall_a["raw_recall_count"] == 2
    assert "sharedapple" in recall_a["context"]
    assert "alphaapple" in recall_a["context"]
    assert "betaapple" not in recall_a["context"]

    assert recall_b["raw_recall_count"] == 2
    assert "sharedapple" in recall_b["context"]
    assert "betaapple" in recall_b["context"]
    assert "alphaapple" not in recall_b["context"]

    assert recall_c["raw_recall_count"] == 1
    assert "sharedapple" in recall_c["context"]
    assert "alphaapple" not in recall_c["context"]
    assert "betaapple" not in recall_c["context"]


def test_memory_write_status_hides_private_write_state_from_other_agents(tmp_path):
    mod.data_dir = str(tmp_path)
    mod.registry.clear()

    shared = asyncio.run(memory_write(
        key="write_status_acl",
        message_id="shared-1",
        session_id="sess-shared",
        content="shared note",
        content_family="chat",
        timestamp_ms=1712000004500,
        agent_id="agent-a",
        swarm_id="sw1",
        scope="swarm-shared",
    ))
    private = asyncio.run(memory_write(
        key="write_status_acl",
        message_id="private-1",
        session_id="sess-private",
        content="private note",
        content_family="chat",
        timestamp_ms=1712000004501,
        agent_id="agent-a",
        swarm_id="sw1",
        scope="agent-private",
    ))

    assert shared["inserted"] is True
    assert private["inserted"] is True

    owner_status = asyncio.run(memory_write_status(
        key="write_status_acl",
        message_id="private-1",
        agent_id="agent-a",
        swarm_id="sw1",
    ))
    other_status = asyncio.run(memory_write_status(
        key="write_status_acl",
        message_id="private-1",
        agent_id="agent-b",
        swarm_id="sw1",
    ))

    assert owner_status["extraction_state"] == "pending"
    assert other_status == {"error": "Write private-1 not found", "code": "NOT_FOUND"}


def test_ensure_instance_config_is_atomic_under_concurrency(tmp_path, monkeypatch):
    mod.data_dir = str(tmp_path)
    mod.registry.clear()
    server = _get_memory("instance_config_race")
    barrier = threading.Barrier(2)
    original = mod._ensure_instance_config

    def _wrapped(server_obj, owner_id):
        barrier.wait(timeout=1)
        return original(server_obj, owner_id)

    monkeypatch.setattr(mod, "_ensure_instance_config", _wrapped)

    with ThreadPoolExecutor(max_workers=2) as pool:
        created = list(pool.map(lambda owner: mod._ensure_instance_config(server, owner), ["agent:a", "agent:b"]))

    assert sum(bool(item) for item in created) == 1
    assert server._instance_config["owner_id"] in {"agent:a", "agent:b"}


def test_memory_write_worker_processes_batch_after_concurrent_writes(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    mod.data_dir = str(tmp_path)
    mod.registry.clear()
    total = 5

    def _write(i: int) -> dict:
        return asyncio.run(memory_write(
            key="write_batch",
            message_id=f"batch-{i}",
            session_id=f"sess-{i}",
            content=f"User: note {i}\nAssistant: ok",
            content_family="chat",
            timestamp_ms=1712000005000 + i,
            agent_id="agent-a",
            swarm_id="sw1",
        ))

    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(_write, range(total)))

    assert all(result["inserted"] is True for result in results)
    server = mod.registry["write_batch"]

    processed = asyncio.run(server.process_write_log_once(batch_size=total))
    assert processed == total
    assert len(server._raw_sessions) == total
    assert len(server._all_granular) == total * 3
    assert not server._storage.list_write_log_entries(states=["pending", "in_progress", "failed"], order="asc")

    for i in range(total):
        status = asyncio.run(memory_write_status(key="write_batch", message_id=f"batch-{i}", agent_id="agent-a", swarm_id="sw1"))
        assert status["extraction_state"] == "complete"



def test_memory_write_worker_soak_hundreds_of_entries(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    mod.data_dir = str(tmp_path)
    mod.registry.clear()
    total = 100

    def _write(i: int) -> dict:
        return asyncio.run(memory_write(
            key="write_soak",
            message_id=f"soak-{i}",
            session_id=f"sess-{i}",
            content=f"User: soak note {i}\nAssistant: ok",
            content_family="chat",
            timestamp_ms=1712000006000 + i,
            agent_id="agent-a",
            swarm_id="sw1",
        ))

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(_write, range(total)))

    assert all(result["inserted"] is True for result in results)
    server = mod.registry["write_soak"]

    processed = 0
    rounds = 0
    while processed < total and rounds < 20:
        processed += asyncio.run(server.process_write_log_once(batch_size=16))
        rounds += 1

    assert processed == total
    assert len(server._raw_sessions) == total
    assert len(server._all_granular) == total * 3
    assert not server._storage.list_write_log_entries(states=["pending", "in_progress", "failed"], order="asc")


def test_memory_write_failed_entries_remain_recallable_under_concurrent_failures(tmp_path, monkeypatch):
    mod.data_dir = str(tmp_path)
    mod.registry.clear()
    total = 6

    def _write(i: int) -> dict:
        return asyncio.run(memory_write(
            key="write_fail",
            message_id=f"fail-{i}",
            session_id=f"sess-{i}",
            content=f"failure papaya note {i}",
            content_family="chat",
            timestamp_ms=1712000007000 + i,
            agent_id="agent-a",
            swarm_id="sw1",
        ))

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(_write, range(total)))

    assert all(result["inserted"] is True for result in results)
    server = mod.registry["write_fail"]

    async def _boom(_entry):
        raise RuntimeError("boom")

    monkeypatch.setattr(server, "_extract_write_log_entry", _boom)
    monkeypatch.setattr(server, "_should_retry_write_entry", lambda entry, now_ms: True)

    for _ in range(3):
        asyncio.run(server.process_write_log_once(batch_size=total))

    for i in range(total):
        status = asyncio.run(memory_write_status(key="write_fail", message_id=f"fail-{i}", agent_id="agent-a", swarm_id="sw1"))
        assert status["extraction_state"] == "failed"
        assert status["extraction_attempts"] == 3

    recall = asyncio.run(memory_recall(key="write_fail", query="papaya", agent_id="agent-a", swarm_id="sw1"))
    assert recall["raw_recall_count"] == total
    assert "failure papaya note 0" in recall["context"].lower()


def test_memory_write_receipt_latency_under_parallel_load(tmp_path):
    mod.data_dir = str(tmp_path)
    mod.registry.clear()
    server = _get_memory("write_latency_parallel")
    mod._ensure_instance_config(server, "agent:agent-a")
    total = 20

    def _timed_write(i: int) -> float:
        start = time.perf_counter()
        result = asyncio.run(memory_write(
            key="write_latency_parallel",
            message_id=f"lat-{i}",
            session_id=f"sess-{i}",
            content=f"parallel latency note {i}",
            content_family="chat",
            timestamp_ms=1712000008000 + i,
            agent_id="agent-a",
            swarm_id="sw1",
            scope="agent-private",
        ))
        assert result["inserted"] is True
        return time.perf_counter() - start

    with ThreadPoolExecutor(max_workers=8) as pool:
        latencies = sorted(pool.map(_timed_write, range(total)))

    p95 = latencies[int(total * 0.95) - 1]
    assert p95 < 0.1

    start = time.perf_counter()
    recall = asyncio.run(memory_recall(
        key="write_latency_parallel",
        query="parallel latency",
        agent_id="agent-a",
        swarm_id="sw1",
    ))
    recall_elapsed = time.perf_counter() - start
    assert recall_elapsed < 0.2
    assert recall["raw_recall_count"] >= 1
    assert "parallel latency note" in recall["context"].lower()
