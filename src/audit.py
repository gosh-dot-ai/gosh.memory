"""gosh.memory — Audit logging.

Append-only JSONL audit log for security-relevant operations.
"""

import json
from datetime import datetime, timezone
from pathlib import Path


class AuditLog:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = log_dir / "audit.jsonl"

    def log(self, event: str, caller_id: str, details: dict = None) -> None:
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(),
                 "event": event, "caller_id": caller_id}
        if details:
            entry.update(details)
        with open(self._log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
