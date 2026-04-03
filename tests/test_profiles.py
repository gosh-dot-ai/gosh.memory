"""Unit 9: Complexity → recommended_profile mapping."""

import asyncio
import json

import numpy as np
import pytest

from src.memory import MemoryServer, _compute_content_complexity

DIM = 3072

PROFILES = {
    1: "fast",     # level 1
    2: "fast",     # level 2
    3: "balanced", # level 3
    4: "max",      # level 4
    5: "max",      # level 5
}


def _patch_all(monkeypatch):
    async def mock_extract(**kwargs):
        sn = kwargs.get("session_num", 1)
        return ("conv", sn, "2024-06-01", [
            {"id": f"f{sn}_0", "fact": f"Fact {sn}", "kind": "event",
             "entities": [], "tags": [], "session": sn}], [])

    async def mock_consolidate(**kwargs):
        return ("conv", 1, "2024-06-01", [
            {"id": "c0", "fact": "Cons", "kind": "summary",
             "entities": [], "tags": []}])

    async def mock_cross(**kwargs):
        return ("conv", "e", [
            {"id": "x0", "fact": "Cross", "kind": "profile",
             "entities": [], "tags": []}])

    async def mock_embed(texts, **kw):
        return np.random.randn(len(texts), DIM).astype(np.float32)

    async def mock_embed_q(text, **kw):
        return np.random.randn(DIM).astype(np.float32)

    monkeypatch.setattr("src.memory.extract_session", mock_extract)
    monkeypatch.setattr("src.memory.consolidate_session", mock_consolidate)
    monkeypatch.setattr("src.memory.cross_session_entity", mock_cross)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda f, l: None)
    monkeypatch.setattr("src.memory.embed_texts", mock_embed)
    monkeypatch.setattr("src.memory.embed_query", mock_embed_q)


def test_profiles_action_item_maps_to_max(tmp_path, monkeypatch):
    """action_item → content_complexity=0.70 → level 4 → 'max'."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "prof1", profiles=PROFILES)

    # Inject a fact with action_item kind
    ms._all_granular = [{
        "fact": "Deploy by Friday", "kind": "action_item",
        "id": "f1", "conv_id": "prof1", "session": 1,
        "agent_id": "default", "swarm_id": "default", "scope": "swarm-shared",
        "owner_id": "system", "read": ["agent:PUBLIC"], "write": ["agent:PUBLIC"],
        "created_at": "2024-01-01T00:00:00+00:00",
        "_session_content_complexity": 0.70,
        "entities": [], "tags": [],
    }]
    asyncio.run(ms.build_index())

    result = asyncio.run(ms.recall("what needs to be done?"))
    assert "recommended_profile" in result
    assert result["recommended_profile"] == "max"


def test_profiles_preference_maps_to_fast(tmp_path, monkeypatch):
    """preference → content_complexity=0.10 → level 1 → 'fast'."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "prof2", profiles=PROFILES)

    ms._all_granular = [{
        "fact": "User likes dark mode", "kind": "preference",
        "id": "f1", "conv_id": "prof2", "session": 1,
        "agent_id": "default", "swarm_id": "default", "scope": "swarm-shared",
        "owner_id": "system", "read": ["agent:PUBLIC"], "write": ["agent:PUBLIC"],
        "created_at": "2024-01-01T00:00:00+00:00",
        "_session_content_complexity": 0.10,
        "entities": [], "tags": [],
    }]
    asyncio.run(ms.build_index())

    result = asyncio.run(ms.recall("what does user prefer?"))
    assert "recommended_profile" in result
    assert result["recommended_profile"] == "fast"


def test_profiles_ignore_session_max_when_retrieved_fact_is_simple(tmp_path, monkeypatch):
    """Routing must follow the retrieved fact set, not a contaminated session-wide max."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "prof_contamination", profiles=PROFILES)

    ms._all_granular = [{
        "fact": "User likes dark mode", "kind": "preference",
        "id": "f1", "conv_id": "prof_contamination", "session": 1,
        "agent_id": "default", "swarm_id": "default", "scope": "swarm-shared",
        "owner_id": "system", "read": ["agent:PUBLIC"], "write": ["agent:PUBLIC"],
        "created_at": "2024-01-01T00:00:00+00:00",
        "_session_content_complexity": 0.70,
        "entities": [], "tags": [],
    }]
    asyncio.run(ms.build_index())

    result = asyncio.run(ms.recall("what does user prefer?"))
    assert result["complexity_hint"]["content_complexity"] == 0.10
    assert result["recommended_profile"] == "fast"


def test_no_profiles_field_absent(tmp_path, monkeypatch):
    """Without profiles config, recommended_profile absent from response."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "prof3")  # no profiles

    ms._all_granular = [{
        "fact": "Test fact", "kind": "fact",
        "id": "f1", "conv_id": "prof3", "session": 1,
        "agent_id": "default", "swarm_id": "default", "scope": "swarm-shared",
        "owner_id": "system", "read": ["agent:PUBLIC"], "write": ["agent:PUBLIC"],
        "created_at": "2024-01-01T00:00:00+00:00",
        "_session_content_complexity": 0.10,
        "entities": [], "tags": [],
    }]
    asyncio.run(ms.build_index())

    result = asyncio.run(ms.recall("test"))
    assert "recommended_profile" not in result


def test_profile_persistence_across_restart(tmp_path):
    ms1 = MemoryServer(str(tmp_path), "profiles_persist")
    asyncio.run(ms1.set_profiles({1: "fast"}, {"fast": {"model": "m"}}))

    ms2 = MemoryServer(str(tmp_path), "profiles_persist")
    assert ms2._profiles == {1: "fast"}
    assert ms2._profile_configs == {"fast": {"model": "m"}}


def test_profile_json_int_key_round_trip(tmp_path):
    ms1 = MemoryServer(str(tmp_path), "profiles_roundtrip")
    asyncio.run(ms1.set_profiles({1: "fast"}, {"fast": {"model": "m"}}))

    ms2 = MemoryServer(str(tmp_path), "profiles_roundtrip")
    first_key = list(ms2._profiles.keys())[0]
    assert isinstance(first_key, int)


@pytest.mark.parametrize(
    ("profiles", "profile_configs", "message"),
    [
        ({6: "fast"}, {"fast": {"model": "m"}}, "profile level must be 1-5"),
        ({0: "fast"}, {"fast": {"model": "m"}}, "profile level must be 1-5"),
        ({1: "fast"}, {"deep": {"model": "m"}}, "but not in profile_configs"),
        ({1: "fast"}, {"fast": {}}, "missing 'model'"),
        ({1: "fast"}, {"fast": {"model": "m", "max_output_tokens": -1}}, "max_output_tokens must be positive int"),
        ({1: "fast"}, {"fast": {"model": "m", "max_output_tokens": "x"}}, "max_output_tokens must be positive int"),
        ({1: "fast"}, {"fast": {"model": "m", "max_output_tokens_summarize": 0}}, "max_output_tokens_summarize must be positive int"),
    ],
)
def test_set_profiles_validation(tmp_path, profiles, profile_configs, message):
    ms = MemoryServer(str(tmp_path), "profiles_validate")
    with pytest.raises(ValueError, match=message):
        asyncio.run(ms.set_profiles(profiles, profile_configs))


def test_old_cache_without_profiles_loads_cleanly(tmp_path):
    cache_path = tmp_path / "old_cache.json"
    cache_path.write_text(json.dumps({
        "granular": [],
        "cons": [],
        "cross": [],
        "tlinks": [],
        "raw_sessions": [],
        "secrets": [],
        "n_sessions": 0,
        "n_sessions_with_facts": 0,
    }))

    ms = MemoryServer(str(tmp_path), "old_cache")
    assert ms._profiles is None
    assert ms._has_profiles() is False


def test_constructor_args_override_saved_profiles(tmp_path):
    ms1 = MemoryServer(str(tmp_path), "profiles_override")
    asyncio.run(ms1.set_profiles({1: "fast"}, {"fast": {"model": "m"}}))

    ms2 = MemoryServer(
        str(tmp_path),
        "profiles_override",
        profiles={1: "deep"},
        profile_configs={"deep": {"model": "m2"}},
    )
    assert ms2._profiles == {1: "deep"}
    assert ms2._profile_configs == {"deep": {"model": "m2"}}


# ── _compute_content_complexity sanity checks ──
# Verify that injected _session_content_complexity values are consistent
# with what _compute_content_complexity would produce for similar facts.

def test_complexity_action_item_matches_injected():
    """_compute_content_complexity for action_item kind returns 0.70,
    matching the value injected in test_profiles_action_item_maps_to_max."""
    facts = [{"fact": "Deploy by Friday", "kind": "action_item", "entities": [], "tags": []}]
    assert _compute_content_complexity(facts) == 0.70


def test_complexity_preference_matches_injected():
    """_compute_content_complexity for preference kind returns 0.10,
    matching the value injected in test_profiles_preference_maps_to_fast."""
    facts = [{"fact": "User likes dark mode", "kind": "preference", "entities": [], "tags": []}]
    assert _compute_content_complexity(facts) == 0.10


def test_complexity_decision_maps_to_balanced(tmp_path, monkeypatch):
    """decision kind → content_complexity=0.50 → level 3 → 'balanced'."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "prof_decision", profiles=PROFILES)

    complexity = _compute_content_complexity(
        [{"fact": "Use PostgreSQL", "kind": "decision", "entities": [], "tags": []}]
    )
    assert complexity == 0.50

    ms._all_granular = [{
        "fact": "Use PostgreSQL", "kind": "decision",
        "id": "f1", "conv_id": "prof_decision", "session": 1,
        "agent_id": "default", "swarm_id": "default", "scope": "swarm-shared",
        "owner_id": "system", "read": ["agent:PUBLIC"], "write": ["agent:PUBLIC"],
        "created_at": "2024-01-01T00:00:00+00:00",
        "_session_content_complexity": complexity,
        "entities": [], "tags": [],
    }]
    asyncio.run(ms.build_index())

    result = asyncio.run(ms.recall("what database?"))
    assert result["recommended_profile"] == "balanced"


def test_multi_constraint_analytics_query_maps_to_strong(tmp_path, monkeypatch):
    """Comparative multi-constraint recommendation questions should escalate to level 4."""
    _patch_all(monkeypatch)
    profiles = {
        1: "fast",
        2: "fast",
        3: "balanced",
        4: "strong",
        5: "max",
    }
    ms = MemoryServer(str(tmp_path), "prof_analytics", profiles=profiles)

    ms._all_granular = [{
        "fact": "Option B is approved for the analytics migration",
        "kind": "decision",
        "id": "f1",
        "conv_id": "prof_analytics",
        "session": 1,
        "agent_id": "default",
        "swarm_id": "default",
        "scope": "swarm-shared",
        "owner_id": "system",
        "read": ["agent:PUBLIC"],
        "write": ["agent:PUBLIC"],
        "created_at": "2024-01-01T00:00:00+00:00",
        "_session_content_complexity": 0.50,
        "entities": ["latency", "ownership", "rollback"],
        "tags": [],
    }]
    asyncio.run(ms.build_index())

    result = asyncio.run(
        ms.recall(
            "For the analytics migration, which option is recommended given latency, ownership, and rollback constraints?"
        )
    )
    assert result["complexity_hint"]["query_complexity"] == 0.70
    assert result["complexity_hint"]["level"] == 4
    assert result["recommended_profile"] == "strong"


def test_complexity_empty_facts_returns_zero():
    """Empty facts list yields 0.0 complexity."""
    assert _compute_content_complexity([]) == 0.0
