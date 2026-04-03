"""gosh.memory — Identity and content hashing utilities.

Provides deterministic content hashing for dedup and versioning.
"""

import hashlib
from uuid import uuid4


def _generate_artifact_id() -> str:
    """Generate a unique artifact ID."""
    return "art_" + uuid4().hex[:10]


def _generate_version_id() -> str:
    """Generate a unique version ID."""
    return "ver_" + uuid4().hex[:10]


def content_hash_text(text: str) -> str:
    """Hash text content after normalization (BOM strip, line ending normalization)."""
    normalized = text.strip().replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.lstrip("\ufeff")
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def content_hash_bytes(data: bytes) -> str:
    """Hash raw bytes."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def content_hash_git(blob_sha: str) -> str:
    """Wrap a git blob SHA as a content hash."""
    return "git:" + blob_sha
