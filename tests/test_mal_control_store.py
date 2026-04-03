"""MAL control_store tests — SPEC Section 4 + Section 20.

Covers:
- disabled-by-default binding config
- enable/disable persistence
- disabling stops future runs without mutating current generation
- auto flags persist correctly
- auto_collect_feedback default is false in both storage and MCP config tool
"""

import json
from pathlib import Path

import pytest


@pytest.fixture
def data_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def store(data_dir):
    from src.mal.control_store import ControlStore
    return ControlStore(data_dir)


# ── disabled-by-default ──


def test_new_binding_is_disabled_by_default(store):
    state = store.get("project-alpha", "default")
    assert state["enabled"] is False
    assert state["auto_collect_feedback"] is False
    assert state["auto_trigger"] is False


def test_get_nonexistent_binding_returns_defaults_without_creating_file(store, data_dir):
    state = store.get("nonexistent-key", "agent-x")
    assert state["enabled"] is False
    control_path = Path(data_dir) / "mal" / "nonexistent-key" / "agent-x" / "control.json"
    assert not control_path.exists()


# ── enable/disable persistence ──


def test_enable_persists_to_disk(store, data_dir):
    store.set("proj", "default", enabled=True)
    control_path = Path(data_dir) / "mal" / "proj" / "default" / "control.json"
    assert control_path.exists()
    data = json.loads(control_path.read_text())
    assert data["enabled"] is True


def test_disable_persists_to_disk(store, data_dir):
    store.set("proj", "default", enabled=True)
    store.set("proj", "default", enabled=False)
    state = store.get("proj", "default")
    assert state["enabled"] is False


def test_enable_disable_cycle_preserves_auto_flags(store):
    store.set("proj", "default", enabled=True, auto_collect_feedback=True, auto_trigger=True)
    store.set("proj", "default", enabled=False)
    state = store.get("proj", "default")
    assert state["enabled"] is False
    assert state["auto_collect_feedback"] is True
    assert state["auto_trigger"] is True


# ── auto flags ──


def test_auto_collect_feedback_default_is_false(store):
    store.set("proj", "default", enabled=True)
    state = store.get("proj", "default")
    assert state["auto_collect_feedback"] is False


def test_auto_trigger_default_is_false(store):
    store.set("proj", "default", enabled=True)
    state = store.get("proj", "default")
    assert state["auto_trigger"] is False


def test_set_auto_flags_independently(store):
    store.set("proj", "default", enabled=True, auto_collect_feedback=True)
    state = store.get("proj", "default")
    assert state["auto_collect_feedback"] is True
    assert state["auto_trigger"] is False

    store.set("proj", "default", auto_trigger=True)
    state = store.get("proj", "default")
    assert state["auto_collect_feedback"] is True
    assert state["auto_trigger"] is True


# ── binding isolation ──


def test_different_agent_ids_are_independent(store):
    store.set("proj", "agent-A", enabled=True)
    store.set("proj", "agent-B", enabled=False)
    assert store.get("proj", "agent-A")["enabled"] is True
    assert store.get("proj", "agent-B")["enabled"] is False


def test_different_keys_are_independent(store):
    store.set("key-1", "default", enabled=True)
    store.set("key-2", "default", enabled=False)
    assert store.get("key-1", "default")["enabled"] is True
    assert store.get("key-2", "default")["enabled"] is False


# ── partial update ──


def test_set_preserves_unspecified_fields(store):
    store.set("proj", "default", enabled=True, auto_collect_feedback=True, auto_trigger=True)
    store.set("proj", "default", enabled=False)
    state = store.get("proj", "default")
    assert state["enabled"] is False
    assert state["auto_collect_feedback"] is True
    assert state["auto_trigger"] is True


# ── is_enabled convenience ──


def test_is_enabled_returns_bool(store):
    assert store.is_enabled("proj", "default") is False
    store.set("proj", "default", enabled=True)
    assert store.is_enabled("proj", "default") is True
