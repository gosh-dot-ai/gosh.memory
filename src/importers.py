"""Batch history import parsers for GOSH Memory.

Supported formats:
- conversation_json: Claude/ChatGPT export format (array of conversations)
- text: Plain text (single session)
- directory: Multiple files concatenated with ---FILE: name--- markers
"""

import json
import re
from datetime import datetime, timezone

SUPPORTED_FORMATS = {"conversation_json", "text", "directory"}


def parse_conversation_json(content: str) -> list[dict]:
    """Parse conversation_json format (Claude/ChatGPT export)."""
    data = json.loads(content)

    # Accept single object or array
    if isinstance(data, dict):
        data = [data]

    sessions = []
    session_num = 0

    for conv in data:
        messages = conv.get("messages") or conv.get("chat_messages") or []
        if not messages:
            continue

        session_date = _extract_date(conv)

        lines = []
        for msg in messages:
            role = _normalize_role(msg.get("role", "user"))
            text = _extract_content(msg.get("content", ""))
            if text:
                lines.append(f"{role}: {text}")

        if not lines:
            continue

        session_num += 1
        sessions.append({
            "session_num": session_num,
            "session_date": session_date,
            "content": "\n".join(lines),
            "speakers": "User and Assistant",
        })

    return sessions


def parse_text(content: str) -> list[dict]:
    """Parse plain text as a single session."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return [{
        "session_num": 1,
        "session_date": today,
        "content": content.strip(),
        "speakers": "User and Assistant",
    }]


def parse_directory(content: str) -> list[dict]:
    """Parse concatenated files with ---FILE: name--- markers."""
    parts = re.split(r'^---FILE:\s*(.+?)---\s*$', content, flags=re.MULTILINE)

    sessions = []
    session_num = 0

    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts):
            break
        filename = parts[i].strip()
        file_content = parts[i + 1].strip()

        if not file_content:
            continue

        session_date = _extract_date_from_filename(filename)
        session_num += 1

        sessions.append({
            "session_num": session_num,
            "session_date": session_date,
            "content": file_content,
            "speakers": "User and Assistant",
        })

    return sessions


def parse_history(source_format: str, content: str) -> list[dict]:
    """Dispatch to the appropriate parser."""
    if source_format == "conversation_json":
        return parse_conversation_json(content)
    elif source_format == "text":
        return parse_text(content)
    elif source_format == "directory":
        return parse_directory(content)
    else:
        raise ValueError(f"Unknown source_format: {source_format!r}")


# ── Helpers ──

def _normalize_role(role: str) -> str:
    role_lower = role.lower()
    if role_lower in ("user", "human"):
        return "User"
    if role_lower in ("assistant", "ai"):
        return "Assistant"
    if role_lower == "system":
        return "System"
    return role.capitalize()


def _extract_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif isinstance(block, str):
                texts.append(block)
        return " ".join(texts)
    return str(content) if content else ""


def _extract_date(conv: dict) -> str:
    created_at = conv.get("created_at")
    if created_at and isinstance(created_at, str):
        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    create_time = conv.get("create_time")
    if create_time is not None:
        try:
            dt = datetime.fromtimestamp(float(create_time), tz=timezone.utc)
            return dt.strftime("%Y-%m-%d")
        except (ValueError, OSError):
            pass

    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _extract_date_from_filename(filename: str) -> str:
    match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if match:
        return match.group(1)
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")
