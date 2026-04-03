"""MAL apply engine tests — SPEC Section 6 + Section 20.

Covers:
- plan_rollback type selection by state diff
- workspace generation build
- retrieval-only apply clones/links unchanged corpus, cache, embeddings,
  bm25, and prompt files into a complete workspace generation
- failure before pointer -> rolled_back
- failure after pointer -> compensating rollback
- prompt reconciliation exact when extraction state differs
- lock semantics
- bounded durable write queue drains against whichever generation is
  live after lock release
"""

import json
from pathlib import Path

import pytest


@pytest.fixture
def data_dir(tmp_path):
    return str(tmp_path)


# ── plan_rollback ──


def test_plan_rollback_immediate_when_only_selectors_differ():
    from src.mal.apply import plan_rollback
    current = {
        "selector_config_overrides": {"entity_phrase_bonus": 3.0},
        "grouping_prompt_mode": "strict_small",
        "size_cap_chars": 12000,
        "extraction_prompts": {},
    }
    target = {
        "selector_config_overrides": {"entity_phrase_bonus": 2.0},
        "grouping_prompt_mode": "strict_small",
        "size_cap_chars": 12000,
        "extraction_prompts": {},
    }
    plan = plan_rollback(current, target)
    assert plan["rollback_type"] == "immediate"
    assert plan["estimated_cost_usd"] == 0.0


def test_plan_rollback_replay_full_when_grouping_differs():
    from src.mal.apply import plan_rollback
    current = {
        "selector_config_overrides": {},
        "grouping_prompt_mode": "strict_small",
        "size_cap_chars": 12000,
        "extraction_prompts": {},
    }
    target = {
        "selector_config_overrides": {},
        "grouping_prompt_mode": "strict_partition",
        "size_cap_chars": 12000,
        "extraction_prompts": {},
    }
    plan = plan_rollback(current, target)
    assert plan["rollback_type"] == "replay_full"
    assert "stage_agent_prompts" in plan["rollback_actions"]


def test_plan_rollback_replay_reextract_when_extraction_differs():
    from src.mal.apply import plan_rollback
    current = {
        "selector_config_overrides": {},
        "grouping_prompt_mode": "strict_small",
        "size_cap_chars": 12000,
        "extraction_prompts": {"conversation_content_type:technical": "prompt v2"},
    }
    target = {
        "selector_config_overrides": {},
        "grouping_prompt_mode": "strict_small",
        "size_cap_chars": 12000,
        "extraction_prompts": {"conversation_content_type:technical": "prompt v1"},
    }
    plan = plan_rollback(current, target)
    assert plan["rollback_type"] == "replay_reextract"
    assert "re-extract-facts" in plan["rollback_actions"]


def test_plan_rollback_grouping_takes_priority_over_extraction():
    from src.mal.apply import plan_rollback
    current = {
        "selector_config_overrides": {},
        "grouping_prompt_mode": "strict_small",
        "size_cap_chars": 12000,
        "extraction_prompts": {"conversation_content_type:technical": "v2"},
    }
    target = {
        "selector_config_overrides": {},
        "grouping_prompt_mode": "metadata_first",
        "size_cap_chars": 8000,
        "extraction_prompts": {"conversation_content_type:technical": "v1"},
    }
    plan = plan_rollback(current, target)
    assert plan["rollback_type"] == "replay_full"


def test_plan_rollback_size_cap_change_is_grouping_change():
    from src.mal.apply import plan_rollback
    current = {
        "selector_config_overrides": {},
        "grouping_prompt_mode": "strict_small",
        "size_cap_chars": 12000,
        "extraction_prompts": {},
    }
    target = {
        "selector_config_overrides": {},
        "grouping_prompt_mode": "strict_small",
        "size_cap_chars": 8000,
        "extraction_prompts": {},
    }
    plan = plan_rollback(current, target)
    assert plan["rollback_type"] == "replay_full"


def test_plan_rollback_identical_states_is_immediate():
    from src.mal.apply import plan_rollback
    state = {
        "selector_config_overrides": {"entity_phrase_bonus": 3.0},
        "grouping_prompt_mode": "strict_small",
        "size_cap_chars": 12000,
        "extraction_prompts": {},
    }
    plan = plan_rollback(state, state)
    assert plan["rollback_type"] == "immediate"


def test_plan_rollback_extraction_includes_selector_when_both_differ():
    from src.mal.apply import plan_rollback
    current = {
        "selector_config_overrides": {"entity_phrase_bonus": 3.0},
        "grouping_prompt_mode": "strict_small",
        "size_cap_chars": 12000,
        "extraction_prompts": {"conversation_content_type:technical": "v2"},
    }
    target = {
        "selector_config_overrides": {"entity_phrase_bonus": 2.0},
        "grouping_prompt_mode": "strict_small",
        "size_cap_chars": 12000,
        "extraction_prompts": {"conversation_content_type:technical": "v1"},
    }
    plan = plan_rollback(current, target)
    assert plan["rollback_type"] == "replay_reextract"
    assert "stage_active_config" in plan["rollback_actions"]


# ── workspace build and promotion ──


def _seed_live_generation(data_dir, key="proj", agent_id="default"):
    """Create a gen_0 with full self-contained content across all 3 prompt planes."""
    gen0 = Path(data_dir) / "mal" / key / agent_id / "gen_0"
    gen0.mkdir(parents=True, exist_ok=True)
    (gen0 / "active_config.json").write_text(json.dumps({"version": 0}))
    # conversation prompts
    conv_dir = gen0 / "prompts" / "conversation"
    conv_dir.mkdir(parents=True, exist_ok=True)
    (conv_dir / "default.md").write_text("base conversation prompt")
    # document block prompts
    block_dir = gen0 / "prompts" / "document" / "block"
    block_dir.mkdir(parents=True, exist_ok=True)
    for name in ("prose_block", "list_block", "table_block", "fallback_block"):
        (block_dir / f"{name}.md").write_text(f"base {name} prompt")
    # document source aggregation prompts
    agg_dir = gen0 / "prompts" / "document" / "source_aggregation"
    agg_dir.mkdir(parents=True, exist_ok=True)
    (agg_dir / "unified_source_aggregation.md").write_text("base aggregation prompt")
    (agg_dir / "unified_source_aggregation_repair.md").write_text("base repair prompt")
    # corpus, cache, embeddings, bm25
    (gen0 / "corpus.json").write_text("[]")
    (gen0 / "cache_derived.json").write_text("{}")
    emb_dir = gen0 / "embeddings"
    emb_dir.mkdir()
    (emb_dir / "atomic.npz").write_text("fake")
    bm25_dir = gen0 / "bm25"
    bm25_dir.mkdir()
    (bm25_dir / "index.json").write_text("{}")
    ptr = Path(data_dir) / "mal" / key / agent_id / "current_gen"
    ptr.write_text("gen_0")
    return gen0


def test_workspace_generation_created_during_apply(data_dir):
    from src.mal.apply import ApplyEngine
    engine = ApplyEngine(data_dir)
    gen_dir = engine.build_workspace("proj", "default", gen_number=1)
    assert gen_dir.exists()
    assert "apply_workspace" in str(gen_dir)
    assert "gen_1" in str(gen_dir)


def test_retrieval_only_workspace_is_complete_generation(data_dir):
    """Retrieval-only apply must clone/link unchanged files into workspace
    so that gen_N is self-contained after promotion (SPEC 1.6, 20).
    All 3 prompt planes must be present."""
    from src.mal.apply import ApplyEngine
    gen0 = _seed_live_generation(data_dir)
    engine = ApplyEngine(data_dir)
    workspace = engine.build_workspace_from_current(
        "proj", "default", gen_number=1,
        new_config={"selector_config_overrides": {"entity_phrase_bonus": 5.0}, "version": 1},
    )
    # config overridden
    assert (workspace / "active_config.json").exists()
    config = json.loads((workspace / "active_config.json").read_text())
    assert config["selector_config_overrides"]["entity_phrase_bonus"] == 5.0
    # corpus/cache/embeddings/bm25 cloned
    assert (workspace / "corpus.json").exists()
    assert (workspace / "corpus.json").read_text() == "[]"
    assert (workspace / "cache_derived.json").exists()
    assert (workspace / "embeddings").exists()
    assert (workspace / "bm25").exists()
    # conversation prompt plane
    assert (workspace / "prompts" / "conversation" / "default.md").exists()
    # document block prompt plane
    assert (workspace / "prompts" / "document" / "block" / "prose_block.md").exists()
    assert (workspace / "prompts" / "document" / "block" / "list_block.md").exists()
    assert (workspace / "prompts" / "document" / "block" / "table_block.md").exists()
    assert (workspace / "prompts" / "document" / "block" / "fallback_block.md").exists()
    # document source aggregation prompt plane
    assert (workspace / "prompts" / "document" / "source_aggregation" / "unified_source_aggregation.md").exists()
    assert (workspace / "prompts" / "document" / "source_aggregation" / "unified_source_aggregation_repair.md").exists()


def test_promote_workspace_moves_to_live(data_dir):
    from src.mal.apply import ApplyEngine
    _seed_live_generation(data_dir)
    engine = ApplyEngine(data_dir)
    workspace = engine.build_workspace_from_current(
        "proj", "default", gen_number=1,
        new_config={"version": 1},
    )
    engine.promote("proj", "default", gen_number=1)
    live_dir = Path(data_dir) / "mal" / "proj" / "default" / "gen_1"
    assert live_dir.exists()
    assert not workspace.exists()
    assert (live_dir / "active_config.json").exists()
    assert (live_dir / "corpus.json").exists()


def test_current_gen_pointer_updated_after_promote(data_dir):
    from src.mal.apply import ApplyEngine
    _seed_live_generation(data_dir)
    engine = ApplyEngine(data_dir)
    engine.build_workspace_from_current("proj", "default", gen_number=1, new_config={"version": 1})
    engine.promote("proj", "default", gen_number=1)
    ptr = Path(data_dir) / "mal" / "proj" / "default" / "current_gen"
    assert ptr.read_text().strip() == "gen_1"


# ── failure before pointer -> rolled_back (SPEC 6.4b) ──


def test_failure_before_pointer_cleans_workspace_and_preserves_live(data_dir):
    from src.mal.apply import ApplyEngine
    gen0 = _seed_live_generation(data_dir)
    engine = ApplyEngine(data_dir)
    workspace = engine.build_workspace_from_current(
        "proj", "default", gen_number=1,
        new_config={"version": 1},
    )
    assert workspace.exists()
    engine.abort_workspace("proj", "default", gen_number=1)
    assert not workspace.exists()
    ptr = Path(data_dir) / "mal" / "proj" / "default" / "current_gen"
    assert ptr.read_text().strip() == "gen_0"
    assert gen0.exists()


# ── failure after pointer -> compensating rollback (SPEC 6.4b) ──


def test_apply_with_post_pointer_failure_sets_apply_failed_and_auto_rollback(data_dir):
    """The full apply-failure path after pointer write (SPEC 6.4b):
    1. apply() promotes gen_1 (pointer written)
    2. a post-promote step raises (simulated)
    3. apply() must set artifact status to apply_failed
    4. apply() must automatically invoke compensating rollback
    5. current_gen must be restored to gen_0
    6. final artifact status must be rolled_back
    """
    from src.mal.apply import ApplyEngine
    gen0 = _seed_live_generation(data_dir)
    engine = ApplyEngine(data_dir)
    ptr = Path(data_dir) / "mal" / "proj" / "default" / "current_gen"

    statuses_seen = []

    def on_status_change(status):
        statuses_seen.append(status)

    result = engine.apply_generation(
        key="proj",
        agent_id="default",
        materialized_state={
            "selector_config_overrides": {"entity_phrase_bonus": 5.0},
            "grouping_prompt_mode": "strict_small",
            "size_cap_chars": 12000,
            "extraction_prompts": {},
        },
        previous_gen=0,
        simulate_post_pointer_failure=True,
        on_status_change=on_status_change,
    )

    assert "apply_failed" in statuses_seen
    assert result["final_status"] == "rolled_back"
    assert ptr.read_text().strip() == "gen_0"
    assert gen0.exists()


def test_compensating_rollback_failure_enters_rollback_failed(data_dir):
    """If the prior live generation is corrupted when compensating rollback
    runs, result must be rollback_failed (SPEC 6.4b).

    Setup: gen_0 live. Apply gen_1 (ok, gen_1 live). Apply gen_2 with
    post-pointer failure. Between pointer write and compensating rollback,
    destroy gen_1 so rollback cannot restore it.

    The key constraint: gen_1 must still be intact when apply_generation
    builds gen_2's workspace (so we don't fail before pointer write).
    The destruction happens via a hook that fires after the pointer is
    written but before compensating rollback reads gen_1.
    """
    import shutil

    from src.mal.apply import ApplyEngine

    _seed_live_generation(data_dir)
    engine = ApplyEngine(data_dir)
    ptr = Path(data_dir) / "mal" / "proj" / "default" / "current_gen"

    # Apply gen_1 successfully (gen_1 is now live)
    engine.build_workspace_from_current("proj", "default", gen_number=1, new_config={"version": 1})
    engine.promote("proj", "default", gen_number=1)
    assert ptr.read_text().strip() == "gen_1"

    gen1_dir = Path(data_dir) / "mal" / "proj" / "default" / "gen_1"
    destroyed = False

    def destroy_restore_target_after_pointer():
        """Called after gen_2 pointer write, before compensating rollback."""
        nonlocal destroyed
        shutil.rmtree(gen1_dir)
        destroyed = True

    # Apply gen_2: workspace builds from gen_1 (intact), pointer writes,
    # then post-pointer failure triggers. Before compensating rollback
    # reads gen_1, the hook destroys it.
    result = engine.apply_generation(
        key="proj",
        agent_id="default",
        materialized_state={
            "selector_config_overrides": {},
            "grouping_prompt_mode": "strict_small",
            "size_cap_chars": 12000,
            "extraction_prompts": {},
        },
        previous_gen=1,
        simulate_post_pointer_failure=True,
        before_compensating_rollback=destroy_restore_target_after_pointer,
    )
    assert destroyed, "hook must have fired between pointer write and rollback"
    assert result["final_status"] == "rollback_failed"


# ── prompt reconciliation during rollback (SPEC 1.5) ──


def test_rollback_with_extraction_diff_includes_prompt_reconciliation(data_dir):
    from src.mal.apply import plan_rollback
    current = {
        "selector_config_overrides": {},
        "grouping_prompt_mode": "strict_small",
        "size_cap_chars": 12000,
        "extraction_prompts": {
            "conversation_content_type:technical": "prompt with extra example",
            "document_block_prompt:prose_block": "doc prompt",
        },
    }
    target = {
        "selector_config_overrides": {},
        "grouping_prompt_mode": "strict_small",
        "size_cap_chars": 12000,
        "extraction_prompts": {
            "conversation_content_type:technical": "original prompt only",
            # document_block_prompt:prose_block is absent -> must be deleted
        },
    }
    plan = plan_rollback(current, target)
    assert plan["rollback_type"] == "replay_reextract"
    assert "stage_agent_prompts" in plan["rollback_actions"]


def test_prompt_reconciliation_physically_deletes_absent_files(data_dir):
    """When rolling back to a state without document_block_prompt:prose_block,
    that file must be physically removed from the target generation (SPEC 1.5)."""
    from src.mal.apply import ApplyEngine
    gen0 = _seed_live_generation(data_dir)
    # Simulate: gen_0 has an extra custom prompt that target state doesn't want
    extra_prompt = gen0 / "prompts" / "document" / "block" / "custom_extra.md"
    extra_prompt.write_text("extra prompt that should be deleted on rollback")
    assert extra_prompt.exists()

    engine = ApplyEngine(data_dir)
    target_extraction_prompts = {
        # custom_extra is NOT in target -> must be deleted
        "conversation_content_type:default": "base conversation prompt",
    }
    workspace = engine.build_workspace_for_rollback(
        "proj", "default", gen_number=1,
        target_materialized_state={
            "selector_config_overrides": {},
            "grouping_prompt_mode": "strict_small",
            "size_cap_chars": 12000,
            "extraction_prompts": target_extraction_prompts,
        },
    )
    # The custom_extra file must NOT exist in the rollback workspace
    assert not (workspace / "prompts" / "document" / "block" / "custom_extra.md").exists()
    # Target prompts that ARE specified must be present
    assert (workspace / "prompts" / "conversation" / "default.md").exists()


# ── lock semantics (SPEC 6.4) ──


def test_lock_prevents_concurrent_apply(data_dir):
    from src.mal.apply import ApplyEngine
    _seed_live_generation(data_dir)
    engine = ApplyEngine(data_dir)
    engine.acquire_lock("proj", "default")
    with pytest.raises(ValueError, match="APPLY_IN_PROGRESS"):
        engine.acquire_lock("proj", "default")
    engine.release_lock("proj", "default")
    engine.acquire_lock("proj", "default")  # should succeed now


# ── write queue ──


def test_write_queue_persists_items(data_dir):
    from src.mal.apply import WriteQueue
    queue = WriteQueue(data_dir, "proj", "default")
    item_id = queue.enqueue({"operation": "store", "content": "hello"})
    items = queue.list_pending()
    assert len(items) == 1
    assert items[0]["operation"] == "store"


def test_write_queue_fifo_order(data_dir):
    from src.mal.apply import WriteQueue
    queue = WriteQueue(data_dir, "proj", "default")
    queue.enqueue({"operation": "store", "content": "first"})
    queue.enqueue({"operation": "store", "content": "second"})
    items = queue.list_pending()
    assert items[0]["content"] == "first"
    assert items[1]["content"] == "second"


def test_write_queue_overflow_returns_error(data_dir):
    from src.mal.apply import WriteQueue
    queue = WriteQueue(data_dir, "proj", "default", max_items=2)
    queue.enqueue({"operation": "store", "content": "1"})
    queue.enqueue({"operation": "store", "content": "2"})
    with pytest.raises(ValueError, match="APPLY_QUEUE_FULL"):
        queue.enqueue({"operation": "store", "content": "3"})


def test_write_queue_drain_removes_items(data_dir):
    from src.mal.apply import WriteQueue
    queue = WriteQueue(data_dir, "proj", "default")
    queue.enqueue({"operation": "store", "content": "hello"})
    drained = queue.drain()
    assert len(drained) == 1
    assert len(queue.list_pending()) == 0


# ── current_gen_dir helper ──


def test_current_gen_dir_defaults_to_gen_0(data_dir):
    from src.mal.apply import current_gen_dir
    gen = current_gen_dir(data_dir, "proj", "default")
    assert gen.name == "gen_0"


def test_current_gen_dir_follows_pointer(data_dir):
    from src.mal.apply import current_gen_dir
    base = Path(data_dir) / "mal" / "proj" / "default"
    base.mkdir(parents=True, exist_ok=True)
    (base / "current_gen").write_text("gen_5")
    gen = current_gen_dir(data_dir, "proj", "default")
    assert gen.name == "gen_5"
