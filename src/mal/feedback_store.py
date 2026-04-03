"""MAL feedback event store — queued -> reserved -> consumed lifecycle."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .control_store import ControlStore

ADAPTATION_VERDICTS = {"bad_answer", "incomplete_answer", "user_correction"}


class FeedbackStore:

    def __init__(self, data_dir: str, control: ControlStore):
        self._data_dir = Path(data_dir)
        self._control = control

    def _feedback_dir(self, key: str, agent_id: str) -> Path:
        return self._data_dir / "mal" / key / agent_id / "feedback"

    def _event_path(self, key: str, agent_id: str, event_id: str) -> Path:
        return self._feedback_dir(key, agent_id) / f"{event_id}.json"

    def submit(self, key: str, agent_id: str, event: dict) -> str:
        if not self._control.is_enabled(key, agent_id):
            raise ValueError("MAL_DISABLED")

        event_id = f"malfb_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        record = {
            "feedback_event_id": event_id,
            "key": key,
            "agent_id": agent_id,
            "signal_source": event.get("signal_source"),
            "verdict": event.get("verdict"),
            "query": event.get("query"),
            "runtime_trace_ref": event.get("runtime_trace_ref"),
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        for opt in ("response_excerpt", "corrected_answer", "retry_chain_id", "source_ids_hint", "runtime_trace"):
            if event.get(opt) is not None:
                record[opt] = event[opt]

        path = self._event_path(key, agent_id, event_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record))
        return event_id

    def get_event(self, key: str, agent_id: str, event_id: str) -> dict:
        path = self._event_path(key, agent_id, event_id)
        return json.loads(path.read_text())

    def _all_events(self, key: str, agent_id: str) -> list[dict]:
        d = self._feedback_dir(key, agent_id)
        if not d.exists():
            return []
        events = []
        for f in sorted(d.iterdir()):
            if f.suffix == ".json":
                events.append(json.loads(f.read_text()))
        return events

    def list_queued(self, key: str, agent_id: str) -> list[dict]:
        return [e for e in self._all_events(key, agent_id) if e["status"] == "queued"]

    def list_trigger_eligible(self, key: str, agent_id: str) -> list[dict]:
        return [
            e for e in self._all_events(key, agent_id)
            if e["status"] == "queued"
            and e.get("runtime_trace_ref")
            and e.get("verdict") in ADAPTATION_VERDICTS
        ]

    def reserve(self, key: str, agent_id: str, event_ids: list[str], run_id: str) -> None:
        for eid in event_ids:
            event = self.get_event(key, agent_id, eid)
            if event["status"] != "queued":
                raise ValueError(f"cannot reserve: event {eid} is {event['status']}, not queued")
            event["status"] = "reserved"
            event["reserved_by_run_id"] = run_id
            self._event_path(key, agent_id, eid).write_text(json.dumps(event))

    def consume(self, key: str, agent_id: str, event_ids: list[str], run_id: str = None) -> None:
        for eid in event_ids:
            event = self.get_event(key, agent_id, eid)
            if event["status"] != "reserved":
                raise ValueError(f"cannot consume: event {eid} is not reserved")
            if run_id is not None and event.get("reserved_by_run_id") != run_id:
                raise ValueError(
                    f"cannot consume: event {eid} reserved by {event.get('reserved_by_run_id')}, "
                    f"not {run_id}"
                )
            event["status"] = "consumed"
            self._event_path(key, agent_id, eid).write_text(json.dumps(event))

    def release(self, key: str, agent_id: str, event_ids: list[str], run_id: str) -> None:
        for eid in event_ids:
            event = self.get_event(key, agent_id, eid)
            if event["status"] != "reserved":
                raise ValueError(f"cannot release: event {eid} is not reserved")
            if event.get("reserved_by_run_id") != run_id:
                raise ValueError(
                    f"cannot release: event {eid} reserved by {event.get('reserved_by_run_id')}, "
                    f"not {run_id}"
                )
            event["status"] = "queued"
            event.pop("reserved_by_run_id", None)
            self._event_path(key, agent_id, eid).write_text(json.dumps(event))

    def count_independent_failures(self, events: list[dict]) -> int:
        seen_chains: set[str] = set()
        seen_traces: set[str] = set()
        count = 0
        for e in events:
            chain_id = e.get("retry_chain_id")
            trace_ref = e.get("runtime_trace_ref", "")
            if chain_id:
                if chain_id not in seen_chains:
                    seen_chains.add(chain_id)
                    count += 1
            else:
                if trace_ref not in seen_traces:
                    seen_traces.add(trace_ref)
                    count += 1
        return count
