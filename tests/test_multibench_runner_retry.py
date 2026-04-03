import json
from contextlib import asynccontextmanager

import pytest

from multibench.sprint36 import run_cached_one_each_production as runners
from multibench.sprint36 import run_q45_production_ask as q45_runner


@pytest.mark.asyncio
async def test_ask_with_question_retry_retries_internal_error(monkeypatch):
    calls = []

    async def _fake_memory_ask(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            return {
                "error": "OAI call failed after 5 retries",
                "code": "INTERNAL_ERROR",
                "tool": "memory_ask",
            }
        return {"answer": "ok"}

    monkeypatch.setattr(runners, "memory_ask", _fake_memory_ask)

    result = await runners.ask_with_question_retry(
        key="k",
        query="q",
        inference_model="m",
        search_family="auto",
        attempts=3,
    )

    assert result == {"answer": "ok"}
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_ask_with_question_retry_does_not_retry_non_internal_error(monkeypatch):
    calls = []

    async def _fake_memory_ask(**kwargs):
        calls.append(kwargs)
        return {"error": "bad request", "code": "INVALID_INPUT", "tool": "memory_ask"}

    monkeypatch.setattr(runners, "memory_ask", _fake_memory_ask)

    result = await runners.ask_with_question_retry(
        key="k",
        query="q",
        inference_model="m",
        search_family="auto",
        attempts=3,
    )

    assert result["code"] == "INVALID_INPUT"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_q45_ingest_example_retries_cached_error_payload(monkeypatch, tmp_path):
    qdir = tmp_path / "longmemeval_1e043500"
    ingest_dir = qdir / "ingest"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    part_source_id = "q45_longmemeval_1e043500_p001"
    logical_source_id = "q45_longmemeval_1e043500"
    (ingest_dir / f"{part_source_id}.json").write_text(
        json.dumps(
            {
                "source_id": logical_source_id,
                "part_source_id": part_source_id,
                "part_idx": 1,
                "chars": 10,
                "result": {
                    "error": "unhashable type: 'list'",
                    "code": "INTERNAL_ERROR",
                    "tool": "memory_ingest",
                },
            }
        )
    )
    calls = []

    class _FakeClient:
        async def call_tool(self, name, arguments):
            calls.append((name, arguments))
            return {"facts_extracted": 3, "source_family": "conversation"}

    monkeypatch.setattr(q45_runner, "_assert_store_checkpoint", lambda *args, **kwargs: None)
    example = {
        "benchmark": "longmemeval",
        "qid": "1e043500",
        "source_key": "1e043500",
        "category": "single-session-user",
        "sessions": ["hello world"],
    }

    payloads = await q45_runner.ingest_example(_FakeClient(), "q45_longmemeval_1e043500", example, qdir)

    assert len(calls) == 1
    assert calls[0][0] == "memory_ingest"
    assert calls[0][1]["source_id"] == logical_source_id
    assert payloads[0]["result"]["facts_extracted"] == 3
    saved = json.loads((ingest_dir / f"{part_source_id}.json").read_text())
    assert saved["source_id"] == logical_source_id
    assert saved["part_source_id"] == part_source_id
    assert saved["result"]["facts_extracted"] == 3
    assert "error" not in saved["result"]


@pytest.mark.asyncio
async def test_q45_ask_with_question_retry_applies_backoff(monkeypatch):
    calls = []
    sleeps = []

    class _FakeClient:
        async def call_tool(self, name, arguments):
            calls.append((name, arguments))
            if len(calls) < 3:
                return {"error": "timeout", "code": "INTERNAL_ERROR", "tool": "memory_ask"}
            return {"answer": "ok"}

    async def _fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(q45_runner.asyncio, "sleep", _fake_sleep)

    result = await q45_runner.ask_with_question_retry(
        client=_FakeClient(),
        key="k",
        query="q",
        search_family="auto",
        attempts=3,
    )

    assert result == {"answer": "ok"}
    assert sleeps == [2, 4]


@pytest.mark.asyncio
async def test_q45_ask_with_question_retry_retries_transport_exception(monkeypatch):
    calls = []
    sleeps = []

    class _FakeClient:
        async def call_tool(self, name, arguments):
            calls.append((name, arguments))
            if len(calls) < 3:
                raise RuntimeError("transport boom")
            return {"answer": "ok"}

    async def _fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(q45_runner.asyncio, "sleep", _fake_sleep)

    result = await q45_runner.ask_with_question_retry(
        client=_FakeClient(),
        key="k",
        query="q",
        search_family="auto",
        attempts=3,
    )

    assert result == {"answer": "ok"}
    assert sleeps == [2, 4]


@pytest.mark.asyncio
async def test_q45_ingest_example_retries_cached_zero_fact_payload(monkeypatch, tmp_path):
    qdir = tmp_path / "longmemeval_118b2229"
    ingest_dir = qdir / "ingest"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    part_source_id = "q45_longmemeval_118b2229_p001"
    logical_source_id = "q45_longmemeval_118b2229"
    (ingest_dir / f"{part_source_id}.json").write_text(
        json.dumps(
            {
                "source_id": logical_source_id,
                "part_source_id": part_source_id,
                "part_idx": 1,
                "chars": 10,
                "result": {
                    "facts_extracted": 0,
                    "source_family": "conversation",
                },
            }
        )
    )
    calls = []

    class _FakeClient:
        async def call_tool(self, name, arguments):
            calls.append((name, arguments))
            return {"facts_extracted": 5, "source_family": "conversation"}

    monkeypatch.setattr(q45_runner, "_assert_store_checkpoint", lambda *args, **kwargs: None)
    example = {
        "benchmark": "longmemeval",
        "qid": "118b2229",
        "source_key": "118b2229",
        "category": "single-session-user",
        "sessions": ["hello world"],
    }

    payloads = await q45_runner.ingest_example(_FakeClient(), logical_source_id, example, qdir)

    assert len(calls) == 1
    assert payloads[0]["result"]["facts_extracted"] == 5


@pytest.mark.asyncio
async def test_q45_ingest_example_passes_session_metadata_to_memory_ingest(monkeypatch, tmp_path):
    qdir = tmp_path / "locomo_conv_50_cat1"
    calls = []

    class _FakeClient:
        async def call_tool(self, name, arguments):
            calls.append((name, arguments))
            return {"facts_extracted": 2, "source_family": "conversation"}

    monkeypatch.setattr(q45_runner, "_assert_store_checkpoint", lambda *args, **kwargs: None)
    example = {
        "benchmark": "locomo",
        "qid": "conv_50_cat1",
        "source_key": "conv_50",
        "category": "multi-hop",
        "sessions": ["part one", "part two"],
        "dates": ["2023-03-08", "2023-03-15"],
        "speakers": "Calvin and James",
    }

    payloads = await q45_runner.ingest_example(_FakeClient(), "q45_locomo_conv_50_cat1", example, qdir)

    assert len(calls) == 2
    assert calls[0][1]["source_id"] == "q45_locomo_conv_50_cat1"
    assert calls[0][1]["session_num"] == 1
    assert calls[0][1]["session_date"] == "2023-03-08"
    assert calls[0][1]["speakers"] == "Calvin and James"
    assert calls[1][1]["session_num"] == 2
    assert calls[1][1]["session_date"] == "2023-03-15"
    assert payloads[1]["part_source_id"] == "q45_locomo_conv_50_cat1_p002"


@pytest.mark.asyncio
async def test_q45_ensure_example_extracted_rebuilds_error_build_index(monkeypatch, tmp_path):
    example = {
        "benchmark": "longmemeval",
        "qid": "1e043500",
        "source_key": "1e043500",
        "category": "single-session-user",
        "sessions": ["hello world"],
    }
    qdir, _ = q45_runner._example_qdir(tmp_path, example)
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / "build_index.json").write_text(
        json.dumps({"error": "temporary failure", "code": "INTERNAL_ERROR"})
    )

    async def _fake_ingest_example_impl(client, memory_key, example, qdir, *, force_reingest):
        return [{"source_id": f"{memory_key}_p001", "result": {"facts_extracted": 1}}]

    calls = []

    class _FakeClient:
        async def call_tool(self, name, arguments):
            calls.append((name, arguments))
            return {"indexed": True}

    @asynccontextmanager
    async def _fake_runtime(**kwargs):
        yield _FakeClient()

    monkeypatch.setattr(q45_runner, "_ingest_example_impl", _fake_ingest_example_impl)
    monkeypatch.setattr(q45_runner, "production_case_runtime", _fake_runtime)

    result = await q45_runner.ensure_example_extracted(tmp_path, example)

    assert len(calls) == 1
    assert calls[0][0] == "memory_build_index"
    assert result["build_index"] == {"indexed": True}


def test_q45_store_needs_rebuild_when_store_has_per_part_sources(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    memory_key = "q45_locomo_conv_26_cat1"
    (memory_dir / f"{memory_key}.json").write_text(
        json.dumps(
            {
                "raw_sessions": [
                    {"source_id": f"{memory_key}_p001"},
                    {"source_id": f"{memory_key}_p002"},
                ],
                "source_records": {
                    f"{memory_key}_p001": {"family": "conversation"},
                    f"{memory_key}_p002": {"family": "conversation"},
                },
            }
        )
    )
    example = {"sessions": ["part1", "part2"]}

    assert q45_runner._store_needs_rebuild(memory_dir, memory_key, example) is True


def test_q45_store_does_not_rebuild_when_single_logical_source_is_complete(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    memory_key = "q45_longmemeval_118b2229"
    (memory_dir / f"{memory_key}.json").write_text(
        json.dumps(
            {
                "raw_sessions": [
                    {"source_id": memory_key},
                    {"source_id": memory_key},
                ],
                "source_records": {
                    memory_key: {"family": "conversation"},
                },
            }
        )
    )
    example = {"sessions": ["part1", "part2"]}

    assert q45_runner._store_needs_rebuild(memory_dir, memory_key, example) is False


def test_q45_store_needs_rebuild_when_any_session_is_pending(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    memory_key = "q45_longmemeval_118b2229"
    (memory_dir / f"{memory_key}.json").write_text(
        json.dumps(
            {
                "raw_sessions": [
                    {"source_id": memory_key, "status": "active"},
                    {"source_id": memory_key, "status": "pending_reextract"},
                ],
                "source_records": {
                    memory_key: {"family": "conversation"},
                },
            }
        )
    )
    example = {"sessions": ["part1", "part2"]}

    assert q45_runner._store_needs_rebuild(memory_dir, memory_key, example) is True


def test_q45_build_store_validity_reports_invalid_pending_store(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    memory_key = "q45_longmemeval_118b2229"
    (memory_dir / f"{memory_key}.json").write_text(
        json.dumps(
            {
                "raw_sessions": [
                    {"source_id": memory_key, "status": "active", "part_source_id": f"{memory_key}_p001"},
                    {"source_id": memory_key, "status": "pending_reextract", "part_source_id": f"{memory_key}_p002"},
                ],
                "source_records": {
                    memory_key: {"family": "conversation"},
                },
            }
        )
    )
    ingest_payloads = [
        {"result": {"facts_extracted": 2}, "metadata": {"part_idx": 1, "part_source_id": f"{memory_key}_p001"}},
        {"result": {"facts_extracted": 0}, "metadata": {"part_idx": 2, "part_source_id": f"{memory_key}_p002"}},
    ]

    validity = q45_runner._build_store_validity(
        memory_dir,
        memory_key,
        ingest_payloads,
        {"indexed": True},
    )

    assert validity["store_exists"] is True
    assert validity["raw_sessions_count"] == 2
    assert validity["expected_raw_sessions_count"] == 2
    assert validity["source_records_count"] == 1
    assert validity["all_raw_sessions_active"] is False
    assert validity["part_source_count"] == 2
    assert validity["facts_extracted_total"] == 2
    assert validity["valid_run"] is False
    assert "raw_session_not_active" in validity["rebuild_reasons"]
    assert validity["zero_fact_parts"] == [
        {"part_idx": 2, "part_source_id": f"{memory_key}_p002"}
    ]


def test_q45_build_store_validity_accepts_zero_fact_active_store(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    memory_key = "q45_ama_76"
    (memory_dir / f"{memory_key}.json").write_text(
        json.dumps(
            {
                "raw_sessions": [
                    {"source_id": memory_key, "status": "active", "part_source_id": f"{memory_key}_p001"},
                    {"source_id": memory_key, "status": "active", "part_source_id": f"{memory_key}_p002"},
                ],
                "source_records": {
                    memory_key: {"family": "conversation"},
                },
            }
        )
    )
    ingest_payloads = [
        {"result": {"facts_extracted": 2}, "metadata": {"part_idx": 1, "part_source_id": f"{memory_key}_p001"}},
        {"result": {"facts_extracted": 0}, "metadata": {"part_idx": 2, "part_source_id": f"{memory_key}_p002"}},
    ]

    validity = q45_runner._build_store_validity(
        memory_dir,
        memory_key,
        ingest_payloads,
        {"indexed": True},
    )

    assert validity["store_exists"] is True
    assert validity["all_raw_sessions_active"] is True
    assert validity["valid_run"] is True
    assert "raw_session_not_active" not in validity["rebuild_reasons"]
    assert validity["zero_fact_parts"] == [
        {"part_idx": 2, "part_source_id": f"{memory_key}_p002"}
    ]


def test_q45_store_checkpoint_detects_foreign_future_parts_via_metadata():
    payload = {
        "raw_sessions": [
            {
                "source_id": "q45_ama_12",
                "session_num": 1,
                "status": "active",
                "metadata": {
                    "logical_source_id": "q45_ama_12",
                    "part_source_id": "q45_ama_12_p001",
                },
            },
            {
                "source_id": "q45_ama_74",
                "session_num": 66,
                "status": "active",
                "metadata": {
                    "logical_source_id": "q45_ama_74",
                    "part_source_id": "q45_ama_74_p066",
                },
            },
        ],
        "source_records": {"q45_ama_12": {"family": "conversation"}},
    }

    reasons = q45_runner._store_checkpoint_reasons(
        payload,
        memory_key="q45_ama_12",
        expected_total_parts=10,
        observed_parts=2,
    )

    assert "foreign_source_id" in reasons
    assert "logical_source_id_mismatch" in reasons
    assert "future_session_num_present" in reasons
    assert "impossible_session_num" in reasons
    assert "foreign_part_source_id" in reasons
    assert "future_part_source_id_present" in reasons
    assert "impossible_part_source_id" in reasons


def test_q45_build_store_validity_matches_rebuild_contract_for_wrong_logical_source(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    memory_key = "q45_locomo_conv_26_cat1"
    (memory_dir / f"{memory_key}.json").write_text(
        json.dumps(
            {
                "raw_sessions": [
                    {"source_id": memory_key, "status": "active"},
                    {"source_id": memory_key, "status": "active"},
                ],
                "source_records": {
                    f"{memory_key}_p001": {"family": "conversation"},
                    f"{memory_key}_p002": {"family": "conversation"},
                },
            }
        )
    )
    ingest_payloads = [
        {"result": {"facts_extracted": 2}, "metadata": {"part_idx": 1, "part_source_id": f"{memory_key}_p001"}},
        {"result": {"facts_extracted": 2}, "metadata": {"part_idx": 2, "part_source_id": f"{memory_key}_p002"}},
    ]

    validity = q45_runner._build_store_validity(
        memory_dir,
        memory_key,
        ingest_payloads,
        {"indexed": True},
    )

    assert validity["valid_run"] is False
    assert "logical_source_id_mismatch" in validity["rebuild_reasons"]


def test_q45_store_base_accepts_multiple_source_records_when_one_matches_memory_key():
    memory_key = "q45_locomo_conv_26_cat1"
    payload = {
        "raw_sessions": [
            {"source_id": memory_key, "status": "active"},
            {"source_id": memory_key, "status": "active"},
        ],
        "source_records": {
            memory_key: {"family": "conversation"},
            "550e8400-e29b-41d4-a716-446655440000": {"family": "conversation"},
        },
    }

    reasons = q45_runner._store_base_reasons(
        payload,
        memory_key=memory_key,
        expected_raw_sessions_count=2,
    )

    assert reasons == []


def test_q45_store_accepts_episode_scoped_superseded_history(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    memory_key = "q45_karnali_q_089"
    (memory_dir / f"{memory_key}.json").write_text(
        json.dumps(
            {
                "raw_sessions": [
                    {"source_id": memory_key, "episode_id": f"{memory_key}_e0001", "status": "active"},
                    {"source_id": memory_key, "episode_id": f"{memory_key}_e01", "status": "superseded"},
                    {"source_id": memory_key, "episode_id": f"{memory_key}_e02", "status": "superseded"},
                ],
                "source_records": {
                    memory_key: {"family": "document"},
                },
            }
        )
    )
    example = {"sessions": ["part1", "part2", "part3", "part4"]}
    ingest_payloads = [
        {"result": {"facts_extracted": 10}, "metadata": {"part_idx": 1, "part_source_id": f"{memory_key}_p001"}},
        {"result": {"facts_extracted": 10}, "metadata": {"part_idx": 2, "part_source_id": f"{memory_key}_p002"}},
        {"result": {"facts_extracted": 10}, "metadata": {"part_idx": 3, "part_source_id": f"{memory_key}_p003"}},
        {"result": {"facts_extracted": 10}, "metadata": {"part_idx": 4, "part_source_id": f"{memory_key}_p004"}},
    ]

    assert q45_runner._store_needs_rebuild(memory_dir, memory_key, example) is False

    validity = q45_runner._build_store_validity(
        memory_dir,
        memory_key,
        ingest_payloads,
        {"indexed": True},
    )

    assert validity["valid_run"] is True
    assert validity["rebuild_reasons"] == []
    assert validity["all_raw_sessions_active"] is True


@pytest.mark.asyncio
async def test_q45_run_one_example_backfills_telemetry_for_existing_result(monkeypatch, tmp_path):
    example = {
        "benchmark": "longmemeval",
        "qid": "existing_case",
        "source_key": "existing_case",
        "category": "single-session-user",
        "sessions": ["hello world"],
        "question": "What happened?",
        "gold": "hello",
    }
    qdir, bench_dir_name = q45_runner._example_qdir(tmp_path, example)
    qdir.mkdir(parents=True, exist_ok=True)
    memory_dir = qdir / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    memory_key = f"q45_{bench_dir_name}"
    (memory_dir / f"{memory_key}.json").write_text(
        json.dumps(
            {
                "raw_sessions": [{"source_id": memory_key, "status": "active"}],
                "source_records": {memory_key: {"family": "conversation"}},
            }
        )
    )
    result_payload = {
        "benchmark": example["benchmark"],
        "query_id": example["qid"],
        "source_key": example["source_key"],
        "question": example["question"],
        "gold": example["gold"],
        "category": example["category"],
        "ingest": [{"result": {"facts_extracted": 4}, "metadata": {"part_idx": 1, "part_source_id": f"{memory_key}_p001"}}],
        "build_index": {"indexed": True},
        "ask": {"answer": "hello", "profile_used": "qwen"},
        "recall": {"retrieval_families": ["conversation"], "search_family": "conversation"},
        "judge": {"correct": True},
    }
    (qdir / "result.json").write_text(json.dumps(result_payload))

    result = await q45_runner.run_one_example(tmp_path, example)
    telemetry = json.loads((qdir / "telemetry.json").read_text())

    assert result["telemetry"]["telemetry_version"] == 1
    assert telemetry["validity"]["valid_run"] is True
    assert telemetry["ask"]["profile_used"] == "qwen"


@pytest.mark.asyncio
async def test_q45_run_one_example_replays_valid_store_without_reextract(monkeypatch, tmp_path):
    example = {
        "benchmark": "ama",
        "qid": "replay_valid_store_only",
        "source_key": "replay_valid_store_only",
        "category": "OPENWORLD_QA",
        "sessions": ["part1"],
        "question": "What happened?",
        "gold": "hello",
    }
    qdir, bench_dir_name = q45_runner._example_qdir(tmp_path, example)
    qdir.mkdir(parents=True, exist_ok=True)
    memory_dir = qdir / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    memory_key = f"q45_{bench_dir_name}"
    (memory_dir / f"{memory_key}.json").write_text(
        json.dumps(
            {
                "raw_sessions": [{"source_id": memory_key, "status": "active"}],
                "source_records": {memory_key: {"family": "conversation"}},
            }
        )
    )
    (qdir / "result.json").write_text(
        json.dumps(
            {
                "benchmark": example["benchmark"],
                "query_id": example["qid"],
                "source_key": example["source_key"],
                "question": example["question"],
                "gold": example["gold"],
                "category": example["category"],
                "ingest": [{"result": {"facts_extracted": 3}, "metadata": {"part_idx": 1, "part_source_id": f"{memory_key}_p001"}}],
                "build_index": {"indexed": True},
                "ask": {"error": "skipped because extracted store is invalid", "code": "INVALID_RUN"},
                "recall": {"error": "skipped because extracted store is invalid", "code": "INVALID_RUN"},
                "judge": {"correct": False, "skipped": True, "reason": "invalid_run"},
            }
        )
    )

    async def _unexpected_ensure(*args, **kwargs):
        raise AssertionError("ensure_example_extracted should not run when store is already valid")

    async def _fake_ask_with_question_retry(**kwargs):
        return {"answer": "fresh-answer", "profile_used": "qwen"}

    class _FakeClient:
        async def set_single_inference_profile(self, **kwargs):
            return {"status": "ok"}

        async def call_tool(self, name, arguments):
            assert name == "memory_recall"
            return {"retrieval_families": ["conversation"], "search_family": "conversation"}

    @asynccontextmanager
    async def _fake_runtime(**kwargs):
        yield _FakeClient()

    async def _fake_judge(question, gold, answer, category):
        return {"correct": True}

    monkeypatch.setattr(q45_runner, "ensure_example_extracted", _unexpected_ensure)
    monkeypatch.setattr(q45_runner, "ask_with_question_retry", _fake_ask_with_question_retry)
    monkeypatch.setattr(q45_runner, "production_case_runtime", _fake_runtime)
    monkeypatch.setitem(q45_runner.JUDGE_BY_BENCH, "ama", _fake_judge)

    result = await q45_runner.run_one_example(tmp_path, example)

    assert result["ask"]["answer"] == "fresh-answer"
    assert result["telemetry"]["validity"]["valid_run"] is True
    assert result["judge"]["correct"] is True


@pytest.mark.asyncio
async def test_q45_run_one_example_reruns_when_cached_result_is_invalid(monkeypatch, tmp_path):
    example = {
        "benchmark": "ama",
        "qid": "rerun_invalid_cached_result",
        "source_key": "rerun_invalid_cached_result",
        "category": "OPENWORLD_QA",
        "sessions": ["part1"],
        "question": "What happened?",
        "gold": "hello",
    }
    qdir, bench_dir_name = q45_runner._example_qdir(tmp_path, example)
    qdir.mkdir(parents=True, exist_ok=True)
    memory_dir = qdir / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    memory_key = f"q45_{bench_dir_name}"
    (memory_dir / f"{memory_key}.json").write_text(
        json.dumps(
            {
                "raw_sessions": [],
                "source_records": {},
            }
        )
    )
    (qdir / "result.json").write_text(
        json.dumps(
            {
                "benchmark": example["benchmark"],
                "query_id": example["qid"],
                "source_key": example["source_key"],
                "question": example["question"],
                "gold": example["gold"],
                "category": example["category"],
                "ingest": [{"result": {"facts_extracted": 0}}],
                "build_index": {"indexed": True},
                "ask": {"error": "skipped because extracted store is invalid", "code": "INVALID_RUN"},
                "recall": {"error": "skipped because extracted store is invalid", "code": "INVALID_RUN"},
                "judge": {"correct": False, "skipped": True, "reason": "invalid_run"},
            }
        )
    )

    async def _fake_ensure_example_extracted(run_dir, example):
        (memory_dir / f"{memory_key}.json").write_text(
            json.dumps(
                {
                    "raw_sessions": [{"source_id": memory_key, "status": "active"}],
                    "source_records": {memory_key: {"family": "conversation"}},
                }
            )
        )
        return {
            "ingest": [{"result": {"facts_extracted": 3}, "metadata": {"part_idx": 1, "part_source_id": f"{memory_key}_p001"}}],
            "build_index": {"indexed": True},
        }

    async def _fake_ask_with_question_retry(**kwargs):
        return {"answer": "fresh-answer", "profile_used": "qwen"}

    class _FakeClient:
        async def set_single_inference_profile(self, **kwargs):
            return {"status": "ok"}

        async def call_tool(self, name, arguments):
            assert name == "memory_recall"
            return {"retrieval_families": ["conversation"], "search_family": "conversation"}

    @asynccontextmanager
    async def _fake_runtime(**kwargs):
        yield _FakeClient()

    async def _fake_judge(question, gold, answer, category):
        return {"correct": True}

    monkeypatch.setattr(q45_runner, "ensure_example_extracted", _fake_ensure_example_extracted)
    monkeypatch.setattr(q45_runner, "ask_with_question_retry", _fake_ask_with_question_retry)
    monkeypatch.setattr(q45_runner, "production_case_runtime", _fake_runtime)
    monkeypatch.setitem(q45_runner.JUDGE_BY_BENCH, "ama", _fake_judge)

    result = await q45_runner.run_one_example(tmp_path, example)

    assert result["ask"]["answer"] == "fresh-answer"
    assert result["telemetry"]["validity"]["valid_run"] is True
    assert result["judge"]["correct"] is True


@pytest.mark.asyncio
async def test_q45_run_one_example_writes_canonical_telemetry(monkeypatch, tmp_path):
    example = {
        "benchmark": "longmemeval",
        "qid": "telemetry_case",
        "source_key": "telemetry_case",
        "category": "single-session-user",
        "sessions": ["hello world"],
        "question": "What happened?",
        "gold": "hello",
    }

    async def _fake_ensure_example_extracted(run_dir, example):
        qdir, bench_dir_name = q45_runner._example_qdir(run_dir, example)
        memory_dir = qdir / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        memory_key = f"q45_{bench_dir_name}"
        (memory_dir / f"{memory_key}.json").write_text(
            json.dumps(
                {
                    "raw_sessions": [{"source_id": memory_key, "status": "active"}],
                    "source_records": {memory_key: {"family": "conversation"}},
                }
            )
        )
        return {
            "ingest": [{"result": {"facts_extracted": 4}, "metadata": {"part_idx": 1, "part_source_id": f"{memory_key}_p001"}}],
            "build_index": {"indexed": True},
        }

    async def _fake_ask_with_question_retry(**kwargs):
        return {"answer": "hello", "profile_used": "qwen"}

    class _FakeClient:
        async def set_single_inference_profile(self, **kwargs):
            return {"status": "ok"}

        async def call_tool(self, name, arguments):
            assert name == "memory_recall"
            return {"retrieval_families": ["conversation"], "search_family": "conversation"}

    @asynccontextmanager
    async def _fake_runtime(**kwargs):
        yield _FakeClient()

    async def _fake_judge(question, gold, answer, category):
        return {"correct": True}

    monkeypatch.setattr(q45_runner, "ensure_example_extracted", _fake_ensure_example_extracted)
    monkeypatch.setattr(q45_runner, "ask_with_question_retry", _fake_ask_with_question_retry)
    monkeypatch.setattr(q45_runner, "production_case_runtime", _fake_runtime)
    monkeypatch.setitem(q45_runner.JUDGE_BY_BENCH, "longmemeval", _fake_judge)

    result = await q45_runner.run_one_example(tmp_path, example)
    qdir, _ = q45_runner._example_qdir(tmp_path, example)
    telemetry = json.loads((qdir / "telemetry.json").read_text())

    assert result["telemetry"]["telemetry_version"] == 1
    assert telemetry["telemetry_version"] == 1
    assert telemetry["validity"]["valid_run"] is True
    assert telemetry["recall"]["retrieval_families"] == ["conversation"]
    assert telemetry["ask"]["profile_used"] == "qwen"
    assert telemetry["judge"]["correct"] is True


@pytest.mark.asyncio
async def test_q45_run_one_example_skips_ask_when_store_is_invalid(monkeypatch, tmp_path):
    example = {
        "benchmark": "ama",
        "qid": "invalid_case",
        "source_key": "invalid_case",
        "category": "OPENWORLD_QA",
        "sessions": ["part1", "part2"],
        "question": "What happened?",
        "gold": "hello",
    }

    async def _fake_ensure_example_extracted(run_dir, example):
        qdir, bench_dir_name = q45_runner._example_qdir(run_dir, example)
        memory_dir = qdir / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        memory_key = f"q45_{bench_dir_name}"
        (memory_dir / f"{memory_key}.json").write_text(
            json.dumps(
                {
                    "raw_sessions": [
                        {
                            "source_id": memory_key,
                            "session_num": 1,
                            "status": "active",
                            "metadata": {
                                "logical_source_id": memory_key,
                                "part_source_id": f"{memory_key}_p001",
                            },
                        }
                    ],
                    "source_records": {memory_key: {"family": "conversation"}},
                }
            )
        )
        return {
            "ingest": [
                {"result": {"facts_extracted": 4}, "metadata": {"part_idx": 1, "part_source_id": f"{memory_key}_p001"}},
                {"result": {"facts_extracted": 4}, "metadata": {"part_idx": 2, "part_source_id": f"{memory_key}_p002"}},
            ],
            "build_index": {"indexed": True},
        }

    async def _unexpected(**kwargs):
        raise AssertionError("ask/recall/judge should not be called for invalid extracted stores")

    monkeypatch.setattr(q45_runner, "ensure_example_extracted", _fake_ensure_example_extracted)
    monkeypatch.setattr(q45_runner, "ask_with_question_retry", _unexpected)
    monkeypatch.setitem(q45_runner.JUDGE_BY_BENCH, "ama", _unexpected)

    result = await q45_runner.run_one_example(tmp_path, example)
    qdir, _ = q45_runner._example_qdir(tmp_path, example)
    telemetry = json.loads((qdir / "telemetry.json").read_text())

    assert telemetry["validity"]["valid_run"] is False
    assert "raw_session_count_mismatch" in telemetry["validity"]["rebuild_reasons"]
    assert result["ask"]["code"] == "INVALID_RUN"
    assert result["recall"]["code"] == "INVALID_RUN"
    assert result["judge"]["skipped"] is True
