"""MAL apply/rollback engine — workspace, promotion, write queue, locks."""

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

MAL_WRITE_QUEUE_MAX_ITEMS = 1000

_PROMPT_TARGET_PATH_MAP = {
    "conversation_content_type:": ("prompts", "conversation"),
    "document_block_prompt:": ("prompts", "document", "block"),
    "document_source_aggregation_prompt:": ("prompts", "document", "source_aggregation"),
}


def _prompt_file_path(ws: Path, target_key: str) -> Path | None:
    for prefix, parts in _PROMPT_TARGET_PATH_MAP.items():
        if target_key.startswith(prefix):
            name = target_key.split(":", 1)[1]
            return ws.joinpath(*parts) / f"{name}.md"
    return None


def current_gen_dir(data_dir: str, key: str, agent_id: str) -> Path:
    base = Path(data_dir) / "mal" / key / agent_id
    ptr = base / "current_gen"
    gen_name = ptr.read_text().strip() if ptr.exists() else "gen_0"
    return base / gen_name


def plan_rollback(current: dict, target: dict) -> dict:
    selector_changed = current.get("selector_config_overrides") != target.get("selector_config_overrides")
    grouping_changed = (
        current.get("grouping_prompt_mode") != target.get("grouping_prompt_mode")
        or current.get("size_cap_chars") != target.get("size_cap_chars")
    )
    extraction_changed = current.get("extraction_prompts") != target.get("extraction_prompts")

    if not grouping_changed and not extraction_changed:
        return {
            "rollback_type": "immediate",
            "rollback_actions": ["stage_active_config", "rename_to_live"],
            "estimated_cost_usd": 0.0,
        }
    if grouping_changed:
        return {
            "rollback_type": "replay_full",
            "rollback_actions": [
                "stage_active_config", "stage_agent_prompts",
                "re-atoms", "re-episodes", "re-facts",
                "rebuild-index", "rename_to_live",
            ],
            "estimated_cost_usd": None,
        }
    actions = []
    if selector_changed:
        actions.append("stage_active_config")
    actions += ["stage_agent_prompts", "re-extract-facts", "rebuild-index", "rename_to_live"]
    return {
        "rollback_type": "replay_reextract",
        "rollback_actions": actions,
        "estimated_cost_usd": None,
    }


def _write_active_config(ws: Path, materialized_state: dict) -> None:
    config = {
        "selector_config_overrides": materialized_state.get("selector_config_overrides", {}),
        "grouping_prompt_mode": materialized_state.get("grouping_prompt_mode", "strict_small"),
        "size_cap_chars": materialized_state.get("size_cap_chars", 12000),
        "inference_leaf_plugin_overrides": materialized_state.get("inference_leaf_plugin_overrides", {}),
    }
    if materialized_state.get("extraction_model"):
        config["extraction_model"] = materialized_state["extraction_model"]
    (ws / "active_config.json").write_text(json.dumps(config))


def _write_prompts(ws: Path, extraction_prompts: dict) -> None:
    for target_key, prompt_text in extraction_prompts.items():
        path = _prompt_file_path(ws, target_key)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(prompt_text)


def _clone_base_files(current: Path, ws: Path) -> None:
    for item in ("corpus.json", "cache_derived.json"):
        src = current / item
        if src.exists():
            shutil.copy2(str(src), str(ws / item))
    for subdir in ("embeddings", "bm25"):
        src = current / subdir
        if src.exists():
            shutil.copytree(str(src), str(ws / subdir), dirs_exist_ok=True)


class ApplyEngine:

    def __init__(self, data_dir: str):
        self._data_dir = Path(data_dir)
        self._locks: dict[str, bool] = {}

    def _base(self, key: str, agent_id: str) -> Path:
        return self._data_dir / "mal" / key / agent_id

    def _workspace_path(self, key: str, agent_id: str, gen_number: int) -> Path:
        return self._base(key, agent_id) / "apply_workspace" / f"gen_{gen_number}"

    def _live_gen_path(self, key: str, agent_id: str, gen_number: int) -> Path:
        return self._base(key, agent_id) / f"gen_{gen_number}"

    def _lock_key(self, key: str, agent_id: str) -> str:
        return f"{key}/{agent_id}"

    def acquire_lock(self, key: str, agent_id: str) -> None:
        lk = self._lock_key(key, agent_id)
        if self._locks.get(lk):
            raise ValueError("APPLY_IN_PROGRESS")
        self._locks[lk] = True

    def release_lock(self, key: str, agent_id: str) -> None:
        self._locks.pop(self._lock_key(key, agent_id), None)

    def build_workspace(self, key: str, agent_id: str, gen_number: int) -> Path:
        ws = self._workspace_path(key, agent_id, gen_number)
        ws.mkdir(parents=True, exist_ok=True)
        return ws

    def build_workspace_from_current(
        self, key: str, agent_id: str, gen_number: int, new_config: dict = None,
    ) -> Path:
        ws = self.build_workspace(key, agent_id, gen_number)
        current = current_gen_dir(str(self._data_dir), key, agent_id)

        _clone_base_files(current, ws)

        # Copy all prompt trees from current generation
        prompts_src = current / "prompts"
        if prompts_src.exists():
            shutil.copytree(str(prompts_src), str(ws / "prompts"), dirs_exist_ok=True)

        if new_config is not None:
            (ws / "active_config.json").write_text(json.dumps(new_config))
        return ws

    def build_workspace_for_rollback(
        self, key: str, agent_id: str, gen_number: int,
        target_materialized_state: dict,
    ) -> Path:
        ws = self.build_workspace(key, agent_id, gen_number)
        current = current_gen_dir(str(self._data_dir), key, agent_id)

        _clone_base_files(current, ws)

        # Write ONLY target extraction prompts — no copying from current
        _write_prompts(ws, target_materialized_state.get("extraction_prompts", {}))

        _write_active_config(ws, target_materialized_state)
        return ws

    def promote(self, key: str, agent_id: str, gen_number: int) -> None:
        ws = self._workspace_path(key, agent_id, gen_number)
        live = self._live_gen_path(key, agent_id, gen_number)
        ws.rename(live)
        ptr = self._base(key, agent_id) / "current_gen"
        ptr.parent.mkdir(parents=True, exist_ok=True)
        ptr.write_text(f"gen_{gen_number}")

    def abort_workspace(self, key: str, agent_id: str, gen_number: int) -> None:
        ws = self._workspace_path(key, agent_id, gen_number)
        if ws.exists():
            shutil.rmtree(ws)

    def apply_generation(
        self, *, key: str, agent_id: str, materialized_state: dict,
        previous_gen: int,
        simulate_post_pointer_failure: bool = False,
        on_status_change=None,
        before_compensating_rollback=None,
    ) -> dict:
        base = self._base(key, agent_id)
        ptr = base / "current_gen"
        new_gen = previous_gen + 1

        def _set_status(s):
            if on_status_change:
                on_status_change(s)

        try:
            _set_status("applying")

            ws = self.build_workspace_from_current(
                key, agent_id, new_gen,
                new_config=None,
            )

            # Write full config + prompts from materialized_state
            _write_active_config(ws, materialized_state)
            _write_prompts(ws, materialized_state.get("extraction_prompts", {}))

            self.promote(key, agent_id, new_gen)

            if simulate_post_pointer_failure:
                _set_status("apply_failed")

                if before_compensating_rollback:
                    before_compensating_rollback()

                prev_dir = self._live_gen_path(key, agent_id, previous_gen)
                if not prev_dir.exists():
                    _set_status("rollback_failed")
                    return {"final_status": "rollback_failed"}

                ptr.write_text(f"gen_{previous_gen}")
                _set_status("rolled_back")
                return {"final_status": "rolled_back"}

            _set_status("applied")
            return {"final_status": "applied"}

        except Exception:
            self.abort_workspace(key, agent_id, new_gen)
            _set_status("rolled_back")
            return {"final_status": "rolled_back"}


class WriteQueue:

    def __init__(self, data_dir: str, key: str, agent_id: str, max_items: int = MAL_WRITE_QUEUE_MAX_ITEMS):
        self._dir = Path(data_dir) / "mal" / key / agent_id / "write_queue"
        self._max = max_items

    def enqueue(self, item: dict) -> str:
        self._dir.mkdir(parents=True, exist_ok=True)
        if len(list(self._dir.iterdir())) >= self._max:
            raise ValueError("APPLY_QUEUE_FULL")
        item_id = f"wq_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S%f')}_{uuid.uuid4().hex[:6]}"
        item["queued_at"] = datetime.now(timezone.utc).isoformat()
        item["queue_item_id"] = item_id
        (self._dir / f"{item_id}.json").write_text(json.dumps(item))
        return item_id

    def list_pending(self) -> list[dict]:
        if not self._dir.exists():
            return []
        items = []
        for f in sorted(self._dir.iterdir()):
            if f.suffix == ".json":
                items.append(json.loads(f.read_text()))
        return items

    def drain(self) -> list[dict]:
        items = self.list_pending()
        for f in sorted(self._dir.iterdir()):
            if f.suffix == ".json":
                f.unlink()
        return items
