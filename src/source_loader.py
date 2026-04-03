#!/usr/bin/env python3
"""Transport-level source loading for unified ingest."""

from __future__ import annotations

import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


@dataclass
class LoadedSource:
    raw_text: str
    transport: str
    locator: str
    filename: str | None
    mime: str | None
    is_directory: bool
    is_repo: bool
    fetch_metadata: dict


def _decode_bytes(raw: bytes, encoding_hint: str | None = None) -> str:
    for enc in [encoding_hint, "utf-8", "utf-8-sig", "latin-1"]:
        if not enc:
            continue
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _path_is_repo(path: Path) -> bool:
    return (path / ".git").exists() or any((path / marker).exists() for marker in (
        "pyproject.toml", "package.json", "Cargo.toml", "go.mod", "Makefile", ".gitignore",
    ))


def _load_directory(path: Path) -> LoadedSource:
    entries = []
    for item in sorted(path.rglob("*")):
        if item.is_dir():
            continue
        try:
            rel = item.relative_to(path)
        except ValueError:
            rel = item.name
        entries.append(str(rel))
    listing = "\n".join(entries)
    return LoadedSource(
        raw_text=listing,
        transport="path",
        locator=str(path),
        filename=path.name,
        mime="inode/directory",
        is_directory=True,
        is_repo=_path_is_repo(path),
        fetch_metadata={"entry_count": len(entries)},
    )


def _load_file(path: Path) -> LoadedSource:
    raw = path.read_bytes()
    mime, _ = mimetypes.guess_type(str(path))
    text = _decode_bytes(raw)
    return LoadedSource(
        raw_text=text,
        transport="path",
        locator=str(path),
        filename=path.name,
        mime=mime,
        is_directory=False,
        is_repo=False,
        fetch_metadata={"size_bytes": len(raw)},
    )


async def load_source(
    text: str | None = None,
    path: str | None = None,
    url: str | None = None,
) -> LoadedSource:
    """Load content from exactly one transport."""
    supplied = [value is not None for value in (text, path, url)]
    if sum(supplied) != 1:
        raise ValueError("exactly one of text, path, url must be provided")

    if text is not None:
        return LoadedSource(
            raw_text=text,
            transport="text",
            locator="(inline)",
            filename=None,
            mime="text/plain",
            is_directory=False,
            is_repo=False,
            fetch_metadata={"size_bytes": len(text.encode("utf-8"))},
        )

    if path is not None:
        p = Path(path).expanduser()
        if not p.exists():
            raise ValueError(f"path does not exist: {p}")
        if p.is_dir():
            return _load_directory(p)
        return _load_file(p)

    parsed = urlparse(url or "")
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"unsupported url scheme: {parsed.scheme or '(missing)'}")

    started = time.time()
    req = Request(url, headers={"User-Agent": "gosh-memory/1.0"})
    with urlopen(req, timeout=30) as resp:
        raw = resp.read()
        mime = resp.headers.get_content_type()
        charset = resp.headers.get_content_charset()
        final_url = resp.geturl()
    text = _decode_bytes(raw, charset)
    filename = Path(urlparse(final_url).path).name or None
    return LoadedSource(
        raw_text=text,
        transport="url",
        locator=url or "",
        filename=filename,
        mime=mime,
        is_directory=False,
        is_repo=False,
        fetch_metadata={
            "size_bytes": len(raw),
            "fetched_url": final_url,
            "elapsed_ms": int((time.time() - started) * 1000),
        },
    )
