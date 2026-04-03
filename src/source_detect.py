#!/usr/bin/env python3
"""Internal source-family detection for unified ingest."""

from __future__ import annotations

import json
import mimetypes
import re
from pathlib import Path

_MEDIA_MIME_PREFIXES = ("audio/", "video/", "image/", "model/")
_MEDIA_EXTS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a",
    ".mp4", ".mov", ".mkv", ".avi", ".webm",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg",
    ".glb", ".gltf", ".obj", ".fbx", ".usd", ".usdz", ".stl",
}
_CODE_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".cpp", ".c",
    ".h", ".hpp", ".rb", ".php", ".cs", ".swift", ".kt", ".scala", ".sql",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".sh",
}
_CONV_ROLE_RE = re.compile(r"^(user|assistant|agent|system|human|ai)\s*:", re.I | re.M)
_NAMED_SPEAKER_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_.\-]{0,31}):\s+\S", re.M)
_STEP_RE = re.compile(r"^\[(step|action|observation|tool)\s+\d+\]", re.I | re.M)
_TRANSCRIPT_BUNDLE_HEADING_RE = re.compile(r"^(dialogue|session|conversation)\s+\d+\s*:\s*$", re.I | re.M)
_HEADING_RE = re.compile(r"^#{1,6}\s+\S", re.M)
_DIFF_RE = re.compile(r"^(diff --git|@@ |--- |\+\+\+ )", re.M)
_NON_DIALOGUE_LABELS = {
    "action",
    "observation",
    "thought",
    "tool",
    "goal",
    "objects on the map",
    "active rules",
}


def _filename_suffix(filename: str | None) -> str:
    if not filename:
        return ""
    return Path(filename).suffix.lower()


def detect_source_family(
    raw_text: str,
    filename: str | None = None,
    mime: str | None = None,
    is_directory: bool = False,
    is_repo: bool = False,
) -> tuple[str, dict]:
    """Return (family, evidence)."""
    signals: list[str] = []
    suffix = _filename_suffix(filename)
    guessed_mime, _ = mimetypes.guess_type(filename or "")
    effective_mime = mime or guessed_mime or ""
    lower = raw_text.lower()

    if effective_mime.startswith(_MEDIA_MIME_PREFIXES) or suffix in _MEDIA_EXTS:
        signals.append(f"media:{effective_mime or suffix}")
        return "media", {"family": "media", "signals": signals}

    if is_directory:
        if is_repo:
            signals.append("repo_markers")
            return "codebase", {"family": "codebase", "signals": signals}
        signals.append("directory_default_document")
        return "document", {"family": "document", "signals": signals}

    role_hits = len(_CONV_ROLE_RE.findall(raw_text))
    transcript_bundle_headings = len(_TRANSCRIPT_BUNDLE_HEADING_RE.findall(raw_text))
    if transcript_bundle_headings >= 2 and role_hits >= 4:
        signals.append(f"transcript_bundle_headings={transcript_bundle_headings}")
        signals.append(f"speaker_roles={role_hits}")
        return "document", {"family": "document", "signals": signals}

    if role_hits >= 2:
        signals.append(f"speaker_roles={role_hits}")
        return "conversation", {"family": "conversation", "signals": signals}

    named_speakers = [
        m.group(1) for m in _NAMED_SPEAKER_RE.finditer(raw_text)
        if m.group(1).strip().lower() not in _NON_DIALOGUE_LABELS
    ]
    unique_speakers = {name.lower() for name in named_speakers}
    if len(named_speakers) >= 4 and len(unique_speakers) >= 2:
        signals.append(f"named_speakers={len(unique_speakers)}")
        signals.append(f"speaker_turns={len(named_speakers)}")
        return "conversation", {"family": "conversation", "signals": signals}

    stripped = raw_text.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) and "role" in parsed[0]:
                signals.append("json_role_array")
                return "conversation", {"family": "conversation", "signals": signals}
            if isinstance(parsed, dict) and any(k in parsed for k in ("messages", "conversation", "turns")):
                signals.append("json_conversation_object")
                return "conversation", {"family": "conversation", "signals": signals}
        except Exception:
            pass

    if suffix in _CODE_EXTS:
        signals.append(f"code_ext:{suffix}")
        return "codebase", {"family": "codebase", "signals": signals}

    if _DIFF_RE.search(raw_text):
        signals.append("diff_markers")
        return "codebase", {"family": "codebase", "signals": signals}

    if _STEP_RE.search(raw_text):
        signals.append("step_trace_text")
        return "document", {"family": "document", "signals": signals}

    if _HEADING_RE.search(raw_text):
        signals.append("markdown_headings")
        return "document", {"family": "document", "signals": signals}

    if any(tok in lower for tok in ("|", "## ", "### ", "<html", "</p>", "table", "json", "log", "appendix")):
        signals.append("document_structure")
        return "document", {"family": "document", "signals": signals}

    signals.append("document_default")
    return "document", {"family": "document", "signals": signals}
