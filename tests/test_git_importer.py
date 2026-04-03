"""Tests for src/git_importer.py — Git repository importer."""

from pathlib import Path

import pytest

from src.git_importer import _import_local, import_git


def test_import_local_basic(tmp_path):
    (tmp_path / "README.md").write_text("# Project\nThis is documentation.")
    (tmp_path / "notes.txt").write_text("Important notes here.")

    sessions = _import_local(tmp_path, branch=None,
                             patterns=["*.md", "*.txt"],
                             max_files=100, exclude_dirs=set(),
                             content_type="technical")
    assert len(sessions) == 2
    assert sessions[0]["content_type"] == "technical"
    assert sessions[0]["session_num"] == 1


def test_import_local_excludes_dirs(tmp_path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "pkg.md").write_text("package docs")
    (tmp_path / "README.md").write_text("real doc")

    sessions = _import_local(tmp_path, branch=None,
                             patterns=["*.md"],
                             max_files=100,
                             exclude_dirs={"node_modules"},
                             content_type="default")
    assert len(sessions) == 1
    assert "README" in sessions[0]["content"]


def test_import_local_skips_empty_files(tmp_path):
    (tmp_path / "empty.md").write_text("")
    (tmp_path / "real.md").write_text("content")
    sessions = _import_local(tmp_path, branch=None,
                             patterns=["*.md"],
                             max_files=100, exclude_dirs=set(),
                             content_type="default")
    assert len(sessions) == 1


def test_import_local_max_files(tmp_path):
    for i in range(10):
        (tmp_path / f"file_{i}.md").write_text(f"content {i}")
    sessions = _import_local(tmp_path, branch=None,
                             patterns=["*.md"],
                             max_files=3, exclude_dirs=set(),
                             content_type="default")
    assert len(sessions) == 3


def test_import_local_includes_path_in_content(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("Guide content.")
    sessions = _import_local(tmp_path, branch=None,
                             patterns=["*.md"],
                             max_files=100, exclude_dirs=set(),
                             content_type="default")
    assert "docs/guide.md" in sessions[0]["content"]


def test_import_local_nonexistent_path():
    with pytest.raises(FileNotFoundError):
        import_git("/nonexistent/path/repo")


def test_import_git_local(tmp_path):
    """import_git with local path must work same as _import_local."""
    (tmp_path / "main.py").write_text("print('hello')")
    sessions = import_git(str(tmp_path), {"file_patterns": ["*.py"]})
    assert len(sessions) == 1
    assert "main.py" in sessions[0]["content"]


def test_token_injected_into_https_url(monkeypatch):
    """Token must NOT appear in clone URL argv (H6 fix: uses GIT_ASKPASS).

    Token is passed via a temporary GIT_ASKPASS script, not embedded in the URL.
    The URL only contains oauth2@ as the username placeholder.
    Token must NOT appear in error messages either.
    """
    calls = []
    call_kwargs = []

    def mock_run(cmd, **kwargs):
        calls.append(cmd)
        call_kwargs.append(kwargs)
        class R:
            returncode = 1
            stderr = "authentication failed"
        return R()

    monkeypatch.setattr("src.git_importer.subprocess.run", mock_run)
    monkeypatch.setattr("src.git_importer.shutil.rmtree", lambda *a, **kw: None)
    monkeypatch.setattr("src.git_importer.tempfile.mkdtemp", lambda **kw: "/tmp/fake")

    with pytest.raises(RuntimeError) as exc:
        import_git("https://github.com/user/private-repo",
                   {"token": "ghp_secret123", "file_patterns": ["*.md"]})

    # Token must NOT appear in clone URL argv (hidden via GIT_ASKPASS)
    clone_cmd = " ".join(calls[0])
    assert "ghp_secret123" not in clone_cmd

    # URL must use oauth2@ placeholder (askpass provides the password)
    assert "oauth2@" in clone_cmd

    # GIT_ASKPASS must be set in the environment
    env = call_kwargs[0].get("env", {})
    assert "GIT_ASKPASS" in env

    # Token must NOT appear in the error message
    assert "ghp_secret123" not in str(exc.value)


def test_token_not_in_raw_sessions(tmp_path):
    """Token must never appear in session content."""
    (tmp_path / "README.md").write_text("Public documentation.")
    sessions = import_git(str(tmp_path),
                          {"token": "secret_token_xyz", "file_patterns": ["*.md"]})
    for s in sessions:
        assert "secret_token_xyz" not in s.get("content", "")
        assert "secret_token_xyz" not in str(s)
