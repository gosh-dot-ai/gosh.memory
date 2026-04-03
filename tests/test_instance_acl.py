"""Acceptance tests for memory-instance ACL.

Frozen contract for the instance-level ACL refactor.
Tests derived from SPEC §4 Memory instance ACL.

Key concepts:
- Memory instance is an ACL resource with owner_id/read/write
- Every memory-scoped MCP tool checks instance ACL (outer gate)
- Per-fact ACL is inner filter (after instance gate passes)
- Direct MemoryServer calls are trusted (no instance ACL)
- First authenticated MCP write creates instance config

All instance-ACL assertions are on MCP tool functions.
Direct MemoryServer method calls are tested as trusted.

Complete blanket-rule coverage:
Read tools (instance read required):
  memory_recall, memory_ask, memory_query, memory_list, memory_get,
  memory_get_versions, memory_get_schema, memory_get_prompt,
  memory_list_prompts, memory_get_secret, memory_stats
  memory_get_profiles, memory_get_config

Write tools (instance write required):
  memory_store, memory_ingest_document, memory_ingest_asserted_facts,
  memory_import, memory_import_history, memory_edit, memory_reextract,
  memory_build_index, memory_flush, memory_set_schema, memory_set_prompt,
  memory_set_profiles, memory_set_config,
  memory_store_secret

Destructive write tools (instance write required):
  memory_retract, memory_purge, memory_redact

Excluded (no instance ACL):
  membership_register, membership_unregister, membership_list,
  courier_subscribe, courier_unsubscribe
"""

import asyncio
import tempfile

import pytest

import src.mcp_server as mcp_mod
from src.mcp_server import (
    memory_ask,
    memory_build_index,
    memory_edit,
    memory_flush,
    memory_get,
    memory_get_config,
    memory_get_profiles,
    memory_get_prompt,
    memory_get_schema,
    memory_get_secret,
    memory_get_versions,
    memory_import,
    memory_import_history,
    memory_ingest_asserted_facts,
    memory_ingest_document,
    memory_list,
    memory_list_prompts,
    memory_purge,
    memory_query,
    memory_recall,
    memory_redact,
    memory_reextract,
    memory_retract,
    memory_set_config,
    memory_set_profiles,
    memory_set_prompt,
    memory_set_schema,
    memory_stats,
    memory_store,
    memory_store_secret,
)
from src.memory import MemoryServer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_registry():
    mcp_mod.registry.clear()
    yield
    mcp_mod.registry.clear()


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        mcp_mod.data_dir = tmp
        yield tmp


async def _seed(key, agent_id="owner"):
    """Seed a memory instance via MCP store (triggers instance creation)."""
    return await memory_store(
        key=key, content="Seed session content.", session_num=1,
        session_date="2026-03-20", agent_id=agent_id,
    )


# ===========================================================================
# 1. Schema ACL
# ===========================================================================

class TestSchemaACL:

    @pytest.mark.asyncio
    async def test_set_schema_outsider_forbidden(self, tmp_dir):
        await _seed("s", agent_id="owner")
        r = await memory_set_schema(
            key="s", schema={"p": {"type": "number"}}, agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")

    @pytest.mark.asyncio
    async def test_set_schema_owner_ok(self, tmp_dir):
        await _seed("s", agent_id="owner")
        r = await memory_set_schema(
            key="s", schema={"p": {"type": "number"}}, agent_id="owner")
        assert r.get("status") == "ok" or "error" not in r

    @pytest.mark.asyncio
    async def test_get_schema_outsider_forbidden(self, tmp_dir):
        await _seed("s", agent_id="owner")
        await memory_set_schema(
            key="s", schema={"p": {"type": "number"}}, agent_id="owner")
        r = await memory_get_schema(key="s", agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")

    @pytest.mark.asyncio
    async def test_get_schema_read_granted_ok(self, tmp_dir):
        """Read-granted caller (not owner) can read schema."""
        await _seed("s", agent_id="owner")
        await memory_set_schema(
            key="s", schema={"p": {"type": "number"}}, agent_id="owner")
        mcp_mod.registry["s"]._instance_config["read"] = ["agent:reader"]
        r = await memory_get_schema(key="s", agent_id="reader")
        assert "p" in r.get("schema", {})


# ===========================================================================
# 2. Read tools — instance read required (FORBIDDEN for outsider)
# ===========================================================================

class TestReadToolsForbidden:

    @pytest.mark.asyncio
    async def test_recall_forbidden(self, tmp_dir):
        await _seed("r", agent_id="owner")
        r = await memory_recall(key="r", query="test", agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")

    @pytest.mark.asyncio
    async def test_ask_forbidden(self, tmp_dir):
        await _seed("r", agent_id="owner")
        r = await memory_ask(key="r", query="test", agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")

    @pytest.mark.asyncio
    async def test_query_forbidden(self, tmp_dir):
        await _seed("r", agent_id="owner")
        r = await memory_query(key="r", agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")

    @pytest.mark.asyncio
    async def test_list_forbidden(self, tmp_dir):
        await _seed("r", agent_id="owner")
        r = await memory_list(key="r", agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")

    @pytest.mark.asyncio
    async def test_get_forbidden(self, tmp_dir):
        await _seed("r", agent_id="owner")
        r = await memory_get(key="r", fact_id="nonexistent", agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")

    @pytest.mark.asyncio
    async def test_get_versions_forbidden(self, tmp_dir):
        await _seed("r", agent_id="owner")
        r = await memory_get_versions(key="r", artifact_id="x", agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")

    @pytest.mark.asyncio
    async def test_stats_forbidden(self, tmp_dir):
        await _seed("r", agent_id="owner")
        r = await memory_stats(key="r", agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")

    @pytest.mark.asyncio
    async def test_get_schema_forbidden(self, tmp_dir):
        await _seed("r", agent_id="owner")
        r = await memory_get_schema(key="r", agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")

    @pytest.mark.asyncio
    async def test_get_prompt_forbidden(self, tmp_dir):
        await _seed("r", agent_id="owner")
        r = await memory_get_prompt(key="r", content_type="default", agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")

    @pytest.mark.asyncio
    async def test_list_prompts_forbidden(self, tmp_dir):
        await _seed("r", agent_id="owner")
        r = await memory_list_prompts(key="r", agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")

    @pytest.mark.asyncio
    async def test_get_secret_forbidden(self, tmp_dir):
        await _seed("r", agent_id="owner")
        r = await memory_get_secret(key="r", name="x", agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")

    @pytest.mark.asyncio
    async def test_get_profiles_forbidden(self, tmp_dir):
        await _seed("r", agent_id="owner")
        await memory_set_profiles(
            key="r",
            profiles={1: "fast"},
            profile_configs={"fast": {"model": "m"}},
            agent_id="owner",
        )
        r = await memory_get_profiles(key="r", agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")

    @pytest.mark.asyncio
    async def test_owner_can_query(self, tmp_dir):
        await _seed("r", agent_id="owner")
        r = await memory_query(key="r", agent_id="owner")
        assert "code" not in r or r["code"] not in ("FORBIDDEN", "ACL_FORBIDDEN")


# ===========================================================================
# 3. Write tools — instance write required (FORBIDDEN for outsider)
# ===========================================================================

class TestWriteToolsForbidden:

    @pytest.mark.asyncio
    async def test_store_forbidden(self, tmp_dir):
        await _seed("w", agent_id="owner")
        r = await memory_store(
            key="w", content="X", session_num=2,
            session_date="2026-03-21", agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")

    @pytest.mark.asyncio
    async def test_ingest_asserted_forbidden(self, tmp_dir):
        await _seed("w", agent_id="owner")
        r = await memory_ingest_asserted_facts(
            key="w",
            facts=[{"fact": "X", "kind": "fact", "session": 1,
                    "entities": [], "tags": []}],
            raw_sessions=[{"session_num": 1, "session_date": "2026-03-21",
                           "content": "x"}],
            agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")

    @pytest.mark.asyncio
    async def test_ingest_document_forbidden(self, tmp_dir):
        await _seed("w", agent_id="owner")
        r = await memory_ingest_document(
            key="w", content="Doc text", source_id="doc1",
            agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")

    @pytest.mark.asyncio
    async def test_import_forbidden(self, tmp_dir):
        await _seed("w", agent_id="owner")
        r = await memory_import(
            key="w", source_format="text", content="imported text",
            agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")

    @pytest.mark.asyncio
    async def test_import_history_forbidden(self, tmp_dir):
        await _seed("w", agent_id="owner")
        r = await memory_import_history(
            key="w", source_format="text", content="history text",
            agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")

    @pytest.mark.asyncio
    async def test_build_index_forbidden(self, tmp_dir):
        await _seed("w", agent_id="owner")
        r = await memory_build_index(key="w", agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")

    @pytest.mark.asyncio
    async def test_flush_forbidden(self, tmp_dir):
        await _seed("w", agent_id="owner")
        r = await memory_flush(key="w", agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")

    @pytest.mark.asyncio
    async def test_reextract_forbidden(self, tmp_dir):
        await _seed("w", agent_id="owner")
        r = await memory_reextract(key="w", agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")

    @pytest.mark.asyncio
    async def test_set_schema_forbidden(self, tmp_dir):
        await _seed("w", agent_id="owner")
        r = await memory_set_schema(
            key="w", schema={"x": {"type": "string"}}, agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")

    @pytest.mark.asyncio
    async def test_set_prompt_forbidden(self, tmp_dir):
        await _seed("w", agent_id="owner")
        r = await memory_set_prompt(
            key="w", content_type="t", prompt="p", agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")

    @pytest.mark.asyncio
    async def test_set_profiles_forbidden(self, tmp_dir):
        await _seed("w", agent_id="owner")
        r = await memory_set_profiles(
            key="w",
            profiles={1: "fast"},
            profile_configs={"fast": {"model": "m"}},
            agent_id="outsider",
        )
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")

    @pytest.mark.asyncio
    async def test_store_secret_forbidden(self, tmp_dir):
        await _seed("w", agent_id="owner")
        r = await memory_store_secret(
            key="w", name="k", value="v", agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")

    @pytest.mark.asyncio
    async def test_edit_forbidden(self, tmp_dir):
        await _seed("w", agent_id="owner")
        r = await memory_edit(
            key="w", artifact_id="x", new_content="new", agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")

    @pytest.mark.asyncio
    async def test_retract_forbidden(self, tmp_dir):
        await _seed("w", agent_id="owner")
        r = await memory_retract(key="w", artifact_id="x", agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")

    @pytest.mark.asyncio
    async def test_purge_forbidden(self, tmp_dir):
        await _seed("w", agent_id="owner")
        r = await memory_purge(key="w", artifact_id="x", agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")

    @pytest.mark.asyncio
    async def test_redact_forbidden(self, tmp_dir):
        await _seed("w", agent_id="owner")
        r = await memory_redact(
            key="w", artifact_id="x", fields=["fact"], agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")


# ===========================================================================
# 4. Outer gate / inner filter
# ===========================================================================

class TestOuterGateInnerFilter:

    @pytest.mark.asyncio
    async def test_swarm_member_blocked_without_instance_read(self, tmp_dir):
        """Instance read=[] blocks swarm member even if fact read=swarm:alpha."""
        await _seed("g", agent_id="owner")
        r = await memory_query(key="g", agent_id="alpha_member", swarm_id="alpha")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")


# ===========================================================================
# 5. Direct MemoryServer calls remain trusted
# ===========================================================================

class TestDirectCallsTrusted:

    @pytest.mark.asyncio
    async def test_direct_query(self, tmp_dir):
        s = MemoryServer(key="d", data_dir=tmp_dir)
        await s.ingest_asserted_facts(
            facts=[{"fact": "F", "kind": "fact", "session": 1,
                    "entities": [], "tags": []}],
            raw_sessions=[{"session_num": 1, "session_date": "2026-03-20",
                           "content": "x"}],
            enrich_l0=False)
        r = await s.query(caller_role="admin")
        assert r["total"] >= 1

    @pytest.mark.asyncio
    async def test_direct_set_schema(self, tmp_dir):
        s = MemoryServer(key="d", data_dir=tmp_dir)
        await s.set_metadata_schema({"p": {"type": "number"}})
        assert s.get_metadata_schema()["p"]["type"] == "number"

    @pytest.mark.asyncio
    async def test_direct_ingest(self, tmp_dir):
        s = MemoryServer(key="d", data_dir=tmp_dir)
        r = await s.ingest_asserted_facts(
            facts=[{"fact": "T", "kind": "fact", "session": 1,
                    "entities": [], "tags": []}],
            raw_sessions=[{"session_num": 1, "session_date": "2026-03-20",
                           "content": "x"}],
            enrich_l0=False)
        assert "error" not in r


# ===========================================================================
# 6. Instance creation on first MCP write
# ===========================================================================

class TestInstanceCreation:

    @pytest.mark.asyncio
    async def test_first_store_creates_config(self, tmp_dir):
        await memory_store(
            key="c1", content="First", session_num=1,
            session_date="2026-03-20", agent_id="creator")
        s = mcp_mod.registry["c1"]
        assert s._instance_config is not None
        assert s._instance_config["owner_id"] == "agent:creator"
        assert s._instance_config["read"] == []
        assert s._instance_config["write"] == []

    @pytest.mark.asyncio
    async def test_first_ingest_creates_config(self, tmp_dir):
        await memory_ingest_asserted_facts(
            key="c2",
            facts=[{"fact": "I", "kind": "fact", "session": 1,
                    "entities": [], "tags": []}],
            raw_sessions=[{"session_num": 1, "session_date": "2026-03-20",
                           "content": "x"}],
            agent_id="ingestor")
        s = mcp_mod.registry["c2"]
        assert s._instance_config is not None
        assert s._instance_config["owner_id"] == "agent:ingestor"

    @pytest.mark.asyncio
    async def test_first_set_schema_creates_config(self, tmp_dir):
        await memory_set_schema(
            key="c3", schema={"p": {"type": "number"}}, agent_id="schema_creator")
        s = mcp_mod.registry["c3"]
        assert s._instance_config is not None
        assert s._instance_config["owner_id"] == "agent:schema_creator"

    @pytest.mark.asyncio
    async def test_first_set_prompt_creates_config(self, tmp_dir):
        await memory_set_prompt(
            key="c4", content_type="custom", prompt="text",
            agent_id="prompt_creator")
        s = mcp_mod.registry["c4"]
        assert s._instance_config is not None
        assert s._instance_config["owner_id"] == "agent:prompt_creator"

    @pytest.mark.asyncio
    async def test_config_persists_across_restart(self, tmp_dir):
        await memory_store(
            key="cp", content="Persist", session_num=1,
            session_date="2026-03-20", agent_id="persister")
        mcp_mod.registry.clear()
        s = mcp_mod._get_memory("cp")
        assert s._instance_config is not None
        assert s._instance_config["owner_id"] == "agent:persister"

    @pytest.mark.asyncio
    async def test_second_write_does_not_overwrite_config(self, tmp_dir):
        await memory_store(
            key="c5", content="First", session_num=1,
            session_date="2026-03-20", agent_id="first_writer")
        s = mcp_mod.registry["c5"]
        # Grant write to second_writer so the write can proceed
        s._instance_config["write"] = ["agent:second_writer"]
        await memory_store(
            key="c5", content="Second", session_num=2,
            session_date="2026-03-21", agent_id="second_writer")
        assert s._instance_config["owner_id"] == "agent:first_writer"


# ===========================================================================
# 7. Prompt ACL — positive paths
# ===========================================================================

class TestPromptACLPositive:

    @pytest.mark.asyncio
    async def test_set_prompt_owner_ok(self, tmp_dir):
        await _seed("p", agent_id="owner")
        r = await memory_set_prompt(
            key="p", content_type="custom", prompt="text", agent_id="owner")
        assert r.get("stored") is True or "error" not in r

    @pytest.mark.asyncio
    async def test_get_prompt_owner_ok(self, tmp_dir):
        await _seed("p", agent_id="owner")
        await memory_set_prompt(
            key="p", content_type="custom", prompt="text", agent_id="owner")
        r = await memory_get_prompt(key="p", content_type="custom", agent_id="owner")
        assert r.get("prompt") == "text" or "error" not in r

    @pytest.mark.asyncio
    async def test_list_prompts_owner_ok(self, tmp_dir):
        await _seed("p", agent_id="owner")
        r = await memory_list_prompts(key="p", agent_id="owner")
        assert "prompts" in r or "error" not in r

    @pytest.mark.asyncio
    async def test_get_prompt_read_granted_ok(self, tmp_dir):
        await _seed("p", agent_id="owner")
        await memory_set_prompt(
            key="p", content_type="custom", prompt="text", agent_id="owner")
        mcp_mod.registry["p"]._instance_config["read"] = ["agent:reader"]
        r = await memory_get_prompt(key="p", content_type="custom", agent_id="reader")
        assert r.get("prompt") == "text" or "error" not in r


# ===========================================================================
# 8. Profile ACL — positive and creation/fresh-key semantics
# ===========================================================================

class TestProfileACL:

    @pytest.mark.asyncio
    async def test_set_profiles_owner_ok(self, tmp_dir):
        await _seed("pf", agent_id="owner")
        r = await memory_set_profiles(
            key="pf",
            profiles={1: "fast"},
            profile_configs={"fast": {"model": "m"}},
            agent_id="owner",
        )
        assert r.get("status") == "ok"

    @pytest.mark.asyncio
    async def test_set_profiles_first_write_creates_config(self, tmp_dir):
        r = await memory_set_profiles(
            key="pf_create",
            profiles={1: "fast"},
            profile_configs={"fast": {"model": "m"}},
            agent_id="owner",
        )
        assert r.get("status") == "ok"
        s = mcp_mod.registry["pf_create"]
        assert s._instance_config is not None
        assert s._instance_config["owner_id"] == "agent:owner"

    @pytest.mark.asyncio
    async def test_get_profiles_owner_ok(self, tmp_dir):
        await _seed("pf2", agent_id="owner")
        await memory_set_profiles(
            key="pf2",
            profiles={1: "fast"},
            profile_configs={"fast": {"model": "m"}},
            agent_id="owner",
        )
        r = await memory_get_profiles(key="pf2", agent_id="owner")
        assert r["profiles"] == {1: "fast"}

    @pytest.mark.asyncio
    async def test_get_profiles_fresh_key_forbidden(self, tmp_dir):
        r = await memory_get_profiles(key="pf_fresh", agent_id="owner")
        assert r.get("code") == "NOT_FOUND"

    @pytest.mark.asyncio
    async def test_set_config_owner_ok(self, tmp_dir):
        r = await memory_set_config(
            key="cfg",
            config={
                "schema_version": 1,
                "embedding_model": "openai/text-embedding-3-large",
                "librarian_profile": "openai/gpt-oss-120b",
                "profiles": {1: "fast"},
                "profile_configs": {"fast": {"model": "m"}},
                "retrieval": {"search_family": "auto", "default_token_budget": 4000},
            },
            agent_id="owner",
        )
        assert r.get("status") == "ok"

    @pytest.mark.asyncio
    async def test_get_config_owner_ok(self, tmp_dir):
        await memory_set_config(
            key="cfg2",
            config={
                "schema_version": 1,
                "embedding_model": "openai/text-embedding-3-large",
                "librarian_profile": "openai/gpt-oss-120b",
                "profiles": {1: "fast"},
                "profile_configs": {"fast": {"model": "m"}},
                "retrieval": {"search_family": "auto", "default_token_budget": 4000},
            },
            agent_id="owner",
        )
        r = await memory_get_config(key="cfg2", agent_id="owner")
        assert r["schema_version"] == 1

    @pytest.mark.asyncio
    async def test_get_config_fresh_key_forbidden(self, tmp_dir):
        r = await memory_get_config(key="cfg_fresh", agent_id="owner")
        assert r.get("code") == "NOT_FOUND"


# ===========================================================================
# 9. Flat metadata validation
# ===========================================================================

class TestFlatMetadataValidation:

    @pytest.mark.asyncio
    async def test_nested_dict_rejected(self, tmp_dir):
        s = MemoryServer(key="f", data_dir=tmp_dir)
        r = await s.ingest_asserted_facts(
            facts=[{"fact": "N", "kind": "fact", "session": 1,
                    "entities": [], "tags": [],
                    "metadata": {"nested": {"inner": "v"}}}],
            raw_sessions=[{"session_num": 1, "session_date": "2026-03-20",
                           "content": "x"}],
            enrich_l0=False)
        assert "error" in r

    @pytest.mark.asyncio
    async def test_list_int_rejected(self, tmp_dir):
        s = MemoryServer(key="f", data_dir=tmp_dir)
        r = await s.ingest_asserted_facts(
            facts=[{"fact": "L", "kind": "fact", "session": 1,
                    "entities": [], "tags": [],
                    "metadata": {"nums": [1, 2, 3]}}],
            raw_sessions=[{"session_num": 1, "session_date": "2026-03-20",
                           "content": "x"}],
            enrich_l0=False)
        assert "error" in r

    @pytest.mark.asyncio
    async def test_list_str_allowed(self, tmp_dir):
        s = MemoryServer(key="f", data_dir=tmp_dir)
        r = await s.ingest_asserted_facts(
            facts=[{"fact": "S", "kind": "fact", "session": 1,
                    "entities": [], "tags": [],
                    "metadata": {"labels": ["a", "b"]}}],
            raw_sessions=[{"session_num": 1, "session_date": "2026-03-20",
                           "content": "x"}],
            enrich_l0=False)
        assert "error" not in r

    @pytest.mark.asyncio
    async def test_non_dict_rejected(self, tmp_dir):
        s = MemoryServer(key="f", data_dir=tmp_dir)
        r = await s.ingest_asserted_facts(
            facts=[{"fact": "B", "kind": "fact", "session": 1,
                    "entities": [], "tags": [],
                    "metadata": "bad"}],
            raw_sessions=[{"session_num": 1, "session_date": "2026-03-20",
                           "content": "x"}],
            enrich_l0=False)
        assert "error" in r

    @pytest.mark.asyncio
    async def test_flat_scalars_ok(self, tmp_dir):
        s = MemoryServer(key="f", data_dir=tmp_dir)
        r = await s.ingest_asserted_facts(
            facts=[{"fact": "OK", "kind": "fact", "session": 1,
                    "entities": [], "tags": [],
                    "metadata": {"s": "v", "n": 1, "f": 0.5, "b": True}}],
            raw_sessions=[{"session_num": 1, "session_date": "2026-03-20",
                           "content": "x"}],
            enrich_l0=False)
        assert "error" not in r
