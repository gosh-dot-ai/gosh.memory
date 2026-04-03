"""gosh.memory — Git repository importer (Class B).

Reads a local or remote git repository and produces normalized sessions
for memory ingestion. One file = one session.

Supports:
  - Local path:  /path/to/repo
  - HTTPS URL:   https://github.com/user/repo
  - SSH URL:     git@github.com:user/repo

Options:
  branch          str   — branch to checkout (default: main/master auto-detect)
  file_patterns   list  — glob patterns to include (default: ["*.md","*.txt","*.py","*.rst"])
  max_files       int   — max files to import (default: 500)
  exclude_dirs    list  — directories to skip (default: [".git","node_modules","__pycache__"])
  content_type    str   — Librarian prompt type for all sessions (default: "technical")
  token           str   — personal access token for private repos (GitHub, GitLab, Bitbucket)
                          injected into clone URL, never logged, never stored in raw sessions
"""

import fnmatch
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PATTERNS = ["*.md", "*.txt", "*.py", "*.rst", "*.yaml", "*.yml", "*.json"]
DEFAULT_EXCLUDE  = [".git", "node_modules", "__pycache__", ".venv", "venv",
                    "dist", "build", ".cache"]
MAX_FILE_SIZE    = 100_000   # bytes — skip files larger than 100KB
MAX_FILES        = 500


def import_git(source_uri: str, options: dict = None) -> list[dict]:
    """Import git repository → list of normalized sessions.

    Returns [{session_num, session_date, content, speakers, content_type}]
    Each file becomes one session.
    """
    opts = options or {}
    branch        = opts.get("branch", None)
    patterns      = opts.get("file_patterns", DEFAULT_PATTERNS)
    max_files     = opts.get("max_files", MAX_FILES)
    exclude_dirs  = set(opts.get("exclude_dirs", DEFAULT_EXCLUDE))
    content_type  = opts.get("content_type", "technical")
    token         = opts.get("token", None)   # never logged, never stored

    # Determine if local or remote
    is_remote = source_uri.startswith(("https://", "git@", "http://"))

    if is_remote:
        return _import_remote(source_uri, branch, patterns,
                              max_files, exclude_dirs, content_type, token)
    else:
        return _import_local(Path(source_uri), branch, patterns,
                             max_files, exclude_dirs, content_type,
                             source_uri=source_uri)


def _import_remote(url: str, branch: str | None, patterns: list,
                   max_files: int, exclude_dirs: set, content_type: str,
                   token: str | None = None) -> list[dict]:
    """Clone remote repo to temp dir, import, cleanup.

    If token is provided, it is injected into HTTPS URLs as:
      https://<token>@github.com/user/repo
    Token is never written to logs, env vars, or raw session records.
    SSH URLs (git@) are used as-is — token is ignored for SSH.
    """
    # H6 fix: use GIT_ASKPASS to pass token instead of embedding in URL.
    # Token in URL is visible in /proc/PID/cmdline to any user on the system.
    # GIT_ASKPASS script echoes the token on demand — never in argv.
    clone_url = url
    askpass_file = None
    clone_env = None

    if token and url.startswith(("https://", "http://")):
        # Write a GIT_ASKPASS helper that echoes the token
        fd, askpass_file = tempfile.mkstemp(prefix="gosh_askpass_", suffix=".sh")
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(f'#!/bin/sh\necho "{token}"\n')
            os.chmod(askpass_file, 0o700)
        except Exception:
            try:
                os.unlink(askpass_file)
            except OSError:
                pass
            askpass_file = None

        if askpass_file:
            # Inject username into URL (token comes via askpass)
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(url)
            authed = parsed._replace(
                netloc=f"oauth2@{parsed.hostname}"
                + (f":{parsed.port}" if parsed.port else "")
            )
            clone_url = urlunparse(authed)
            clone_env = {**os.environ, "GIT_ASKPASS": askpass_file,
                         "GIT_TERMINAL_PROMPT": "0"}
        else:
            # Fallback: embed in URL (old behavior) if askpass setup failed
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(url)
            authed = parsed._replace(
                netloc=f"{token}@{parsed.hostname}"
                + (f":{parsed.port}" if parsed.port else "")
            )
            clone_url = urlunparse(authed)

    tmp = tempfile.mkdtemp(prefix="gosh_memory_git_")
    try:
        cmd = ["git", "clone", "--depth=1"]
        if branch:
            cmd += ["--branch", branch]
        cmd += [clone_url, tmp]

        # capture_output=True prevents token leaking to parent stdout/stderr
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                                env=clone_env)
        if result.returncode != 0:
            # Scrub token from error message before raising
            err = result.stderr.replace(token, "***") if token else result.stderr
            raise RuntimeError(f"git clone failed: {err.strip()}")

        return _import_local(Path(tmp), branch=None,
                             patterns=patterns, max_files=max_files,
                             exclude_dirs=exclude_dirs, content_type=content_type,
                             source_uri=url)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if askpass_file:
            try:
                os.unlink(askpass_file)
            except OSError:
                pass


def _import_local(repo_path: Path, branch: str | None, patterns: list,
                  max_files: int, exclude_dirs: set, content_type: str,
                  source_uri: str = None) -> list[dict]:
    """Read files from local path, build sessions.

    If branch is specified and repo_path is a git repo, clones to a temp
    directory first to avoid modifying the user's working tree.
    """
    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path not found: {repo_path}")

    # If it's a git repo and branch specified, clone to temp dir to avoid
    # changing the user's working tree
    git_dir = repo_path / ".git"
    if branch and git_dir.exists():
        tmp = tempfile.mkdtemp(prefix="gosh_memory_local_")
        try:
            subprocess.run(
                ["git", "clone", "--no-checkout", str(repo_path), tmp],
                capture_output=True, check=True, timeout=120,
            )
            subprocess.run(
                ["git", "-C", tmp, "checkout", branch],
                capture_output=True, check=False,
            )
            return _import_local(
                Path(tmp), branch=None, patterns=patterns,
                max_files=max_files, exclude_dirs=exclude_dirs,
                content_type=content_type,
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # Walk and collect matching files
    files = []
    for f in repo_path.rglob("*"):
        if not f.is_file():
            continue
        # Check any relative path part is excluded
        rel_parts = set(f.relative_to(repo_path).parts)
        if rel_parts & exclude_dirs:
            continue
        # Check pattern
        if not any(fnmatch.fnmatch(f.name, p) for p in patterns):
            continue
        # Skip large files
        if f.stat().st_size > MAX_FILE_SIZE:
            continue
        files.append(f)
        if len(files) >= max_files:
            break

    # Sort for deterministic ordering
    files.sort()

    sessions = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for i, f in enumerate(files):
        try:
            content = f.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            continue
        if not content:
            continue

        # Try to get file modification date from git log
        session_date = _git_file_date(repo_path, f) or today

        # Use relative path as context header
        try:
            rel = f.relative_to(repo_path)
        except ValueError:
            rel = f.name

        # Compute blob SHA for dedup
        blob_sha = _git_blob_sha(repo_path, f) or _content_sha(content)

        sessions.append({
            "session_num":   i + 1,
            "session_date":  session_date,
            "content":       f"# {rel}\n\n{content}",
            "speakers":      "Document",
            "content_type":  content_type,
            "artifact_path": str(rel),
            "blob_sha":      blob_sha,
            "storage_mode":  "inline",
            "source_id":     source_uri or str(repo_path),
        })

    return sessions


def _git_blob_sha(repo_path: Path, file_path: Path) -> str | None:
    """Get git blob SHA for a file. Returns hex string or None."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "hash-object", str(file_path)],
            capture_output=True, text=True, timeout=5,
        )
        sha = result.stdout.strip()
        if sha and len(sha) >= 40:
            return sha
    except Exception:
        pass
    return None


def _content_sha(content: str) -> str:
    """Fallback SHA-256 of content when git hash-object is unavailable."""
    import hashlib
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _git_file_date(repo_path: Path, file_path: Path) -> str | None:
    """Get last commit date for a file. Returns YYYY-MM-DD or None."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "log", "-1",
             "--format=%ai", "--", str(file_path)],
            capture_output=True, text=True, timeout=5
        )
        date_str = result.stdout.strip()
        if date_str:
            return date_str[:10]
    except Exception:
        pass
    return None
