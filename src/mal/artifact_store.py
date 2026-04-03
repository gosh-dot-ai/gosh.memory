"""MAL tuning artifact store — version chain, materialized state."""

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..prompt_registry import BUILTIN_PROMPTS

_EXTRACTION_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts" / "extraction"


class ArtifactStore:

    def __init__(self, data_dir: str):
        self._data_dir = Path(data_dir)
        self._lock = threading.Lock()

    def _artifacts_dir(self, key: str, agent_id: str) -> Path:
        return self._data_dir / "mal" / key / agent_id / "artifacts"

    def _next_version(self, key: str, agent_id: str) -> int:
        d = self._artifacts_dir(key, agent_id)
        if not d.exists():
            return 1
        versions = []
        for f in d.iterdir():
            if f.suffix == ".json":
                try:
                    a = json.loads(f.read_text())
                    versions.append(a.get("version", 0))
                except Exception:
                    pass
        return max(versions, default=0) + 1

    def _previous_materialized_state(self, key: str, agent_id: str) -> dict:
        latest = self.get_latest(key, agent_id)
        if latest:
            return dict(latest["materialized_state"])
        return {
            "selector_config_overrides": {},
            "grouping_prompt_mode": "strict_small",
            "size_cap_chars": 12000,
            "extraction_prompts": {},
        }

    def _compute_materialized_state(self, prev: dict, atom_type: str, atom_payload: dict) -> dict:
        state = {
            "selector_config_overrides": dict(prev.get("selector_config_overrides", {})),
            "grouping_prompt_mode": prev.get("grouping_prompt_mode", "strict_small"),
            "size_cap_chars": prev.get("size_cap_chars", 12000),
            "extraction_prompts": dict(prev.get("extraction_prompts", {})),
            "inference_leaf_plugin_overrides": dict(prev.get("inference_leaf_plugin_overrides", {})),
        }

        if atom_type == "extraction_example_append":
            target = atom_payload["prompt_target"]
            example = atom_payload["example"]
            prev_text = state["extraction_prompts"].get(target)
            if prev_text is None:
                prev_text = self._builtin_for_target(target)
            state["extraction_prompts"][target] = prev_text + "\n\n" + example
        elif atom_type == "grouping_bundle":
            for field, val in atom_payload.items():
                if field == "grouping_prompt_mode":
                    state["grouping_prompt_mode"] = val["new"]
                elif field == "size_cap_chars":
                    state["size_cap_chars"] = val["new"]
        elif atom_type == "inference_leaf_toggle":
            state["inference_leaf_plugin_overrides"][atom_payload["plugin_name"]] = atom_payload["enabled"]
        elif atom_type == "extraction_model_switch":
            state["extraction_model"] = atom_payload["model_id"]
        else:
            for field, val in atom_payload.items():
                state["selector_config_overrides"][field] = val["new"]

        return state

    def _builtin_for_target(self, target: str) -> str:
        if target.startswith("conversation_content_type:"):
            ct = target.split(":", 1)[1]
            return BUILTIN_PROMPTS.get(ct, BUILTIN_PROMPTS["default"])
        if target.startswith("document_block_prompt:"):
            name = target.split(":", 1)[1]
            path = _EXTRACTION_PROMPT_DIR / f"{name}.md"
            if path.exists():
                return path.read_text(encoding="utf-8")
        if target.startswith("document_source_aggregation_prompt:"):
            name = target.split(":", 1)[1]
            path = _EXTRACTION_PROMPT_DIR / f"{name}.md"
            if path.exists():
                return path.read_text(encoding="utf-8")
        return ""

    def create(self, *, key: str, agent_id: str, atom_type: str, atom_payload: dict,
               failure_family: dict, feedback_event_ids: list[str],
               runtime_trace_refs: list[str], independent_failures_evaluated: int,
               score_before: dict, score_after: dict) -> dict:
        with self._lock:
            version = self._next_version(key, agent_id)
            prev_state = self._previous_materialized_state(key, agent_id)
            materialized = self._compute_materialized_state(prev_state, atom_type, atom_payload)

            family_key = f"{failure_family['stage']}|{failure_family['operator_class_or_shape']}|{failure_family['signature']}"

            artifact_id = f"mal_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"
            artifact = {
                "artifact_id": artifact_id,
                "key": key,
                "agent_id": agent_id,
                "version": version,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "accepted",
                "atom_type": atom_type,
                "atom_payload": atom_payload,
                "fields_changed": list(atom_payload.keys()),
                "failure_family": failure_family,
                "failure_family_key": family_key,
                "feedback_event_ids": feedback_event_ids,
                "runtime_trace_refs": runtime_trace_refs,
                "independent_failures_evaluated": independent_failures_evaluated,
                "materialized_state": materialized,
                "score_before": score_before,
                "score_after": score_after,
            }

            d = self._artifacts_dir(key, agent_id)
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{artifact_id}.json").write_text(json.dumps(artifact))
            return artifact

    def get(self, key: str, agent_id: str, artifact_id: str) -> dict | None:
        path = self._artifacts_dir(key, agent_id) / f"{artifact_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def get_latest(self, key: str, agent_id: str) -> dict | None:
        d = self._artifacts_dir(key, agent_id)
        if not d.exists():
            return None
        best = None
        for f in d.iterdir():
            if f.suffix == ".json":
                try:
                    a = json.loads(f.read_text())
                    if best is None or a.get("version", 0) > best.get("version", 0):
                        best = a
                except Exception:
                    pass
        return best

    def update_status(self, key: str, agent_id: str, artifact_id: str, status: str) -> None:
        path = self._artifacts_dir(key, agent_id) / f"{artifact_id}.json"
        a = json.loads(path.read_text())
        a["status"] = status
        path.write_text(json.dumps(a))
