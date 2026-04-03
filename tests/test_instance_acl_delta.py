"""Delta acceptance tests for instance ACL — missing coverage.

5 cases not covered by test_instance_acl.py frozen contract:
1. Instance ACL positive path: agent:PUBLIC in read list
2. Instance ACL positive path: swarm membership
3. First-write instance creation via memory_ingest_document
4. memory_import auth split (token = source auth, agent_key = caller identity)
5. Cross-key prompt isolation (key A prompt not visible on key B)
"""

import asyncio
import tempfile

import pytest

import src.mcp_server as mcp_mod
from src.mcp_server import (
    memory_ask,
    memory_get_config,
    memory_get_profiles,
    memory_get_prompt,
    memory_import,
    memory_ingest_asserted_facts,
    memory_ingest_document,
    memory_list,
    memory_recall,
    memory_set_config,
    memory_set_profiles,
    memory_set_prompt,
    memory_stats,
    memory_store,
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
    """Seed a memory instance via MCP store."""
    return await memory_store(
        key=key, content="Seed session content.", session_num=1,
        session_date="2026-03-20", agent_id=agent_id,
    )


# ===========================================================================
# 1. Instance ACL positive path: agent:PUBLIC
# ===========================================================================

class TestInstanceACLPublicRead:

    @pytest.mark.asyncio
    async def test_public_read_allows_outsider(self, tmp_dir):
        """If instance read=["agent:PUBLIC"], any caller passes read gate."""
        await _seed("pub", agent_id="owner")
        # Grant public read on instance
        mcp_mod.registry["pub"]._instance_config["read"] = ["agent:PUBLIC"]

        # Outsider should now pass instance read gate
        r = await memory_list(key="pub", agent_id="outsider")
        assert r.get("code") not in ("FORBIDDEN", "ACL_FORBIDDEN"), \
            f"agent:PUBLIC in read should allow outsider, got {r}"

    @pytest.mark.asyncio
    async def test_public_read_allows_recall(self, tmp_dir):
        """agent:PUBLIC in instance read allows recall by any caller."""
        await _seed("pub2", agent_id="owner")
        mcp_mod.registry["pub2"]._instance_config["read"] = ["agent:PUBLIC"]

        r = await memory_recall(key="pub2", query="test", agent_id="stranger")
        assert r.get("code") not in ("FORBIDDEN", "ACL_FORBIDDEN"), \
            f"agent:PUBLIC read should allow recall, got {r}"

    @pytest.mark.asyncio
    async def test_public_read_does_not_grant_write(self, tmp_dir):
        """agent:PUBLIC in read does NOT grant write access."""
        await _seed("pub3", agent_id="owner")
        mcp_mod.registry["pub3"]._instance_config["read"] = ["agent:PUBLIC"]
        # write is still [] — outsider should be denied write
        r = await memory_store(
            key="pub3", content="X", session_num=2,
            session_date="2026-03-21", agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN"), \
            f"agent:PUBLIC in read should NOT grant write, got {r}"


# ===========================================================================
# 2. Instance ACL positive path: swarm membership
# ===========================================================================

class TestInstanceACLSwarmMembership:

    @pytest.mark.asyncio
    async def test_swarm_member_passes_read_gate(self, tmp_dir):
        """If instance read=["swarm:alpha"], caller with swarm_id="alpha" passes."""
        await _seed("sw", agent_id="owner")
        mcp_mod.registry["sw"]._instance_config["read"] = ["swarm:alpha"]
        mcp_mod.registry["sw"]._instance_config["_derived_read"] = []

        # Register membership
        server = mcp_mod.registry["sw"]
        server._membership_registry.register("agent:member", "swarm:alpha")

        r = await memory_list(key="sw", agent_id="member")
        assert r.get("code") not in ("FORBIDDEN", "ACL_FORBIDDEN"), \
            f"swarm member should pass read gate, got {r}"

    @pytest.mark.asyncio
    async def test_non_member_blocked(self, tmp_dir):
        """Caller NOT in swarm:alpha is blocked when read=["swarm:alpha"]."""
        await _seed("sw2", agent_id="owner")
        mcp_mod.registry["sw2"]._instance_config["read"] = ["swarm:alpha"]
        mcp_mod.registry["sw2"]._instance_config["_derived_read"] = []

        r = await memory_list(key="sw2", agent_id="nonmember")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN"), \
            f"non-member should be blocked, got {r}"

    @pytest.mark.asyncio
    async def test_swarm_shared_store_auto_grants_instance_read(self, tmp_dir):
        """swarm-shared writes widen instance read for that swarm, not for outsiders."""
        await memory_store(
            key="sw3",
            content="Owner private fact.",
            session_num=1,
            session_date="2026-03-20",
            agent_id="alice",
            scope="agent-private",
        )
        await memory_store(
            key="sw3",
            content="Shared region is eu-west-1.",
            session_num=2,
            session_date="2026-03-20",
            agent_id="alice",
            swarm_id="alpha",
            scope="swarm-shared",
        )

        server = mcp_mod.registry["sw3"]
        assert "swarm:alpha" in server._instance_config["_derived_read"]

        ok = await memory_list(key="sw3", agent_id="bob", swarm_id="alpha")
        assert ok.get("code") not in ("FORBIDDEN", "ACL_FORBIDDEN"), ok

        blocked = await memory_list(key="sw3", agent_id="mallory", swarm_id="beta")
        assert blocked.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN"), blocked

    @pytest.mark.asyncio
    async def test_swarm_shared_store_auto_grants_instance_write(self, tmp_dir):
        """Named swarm-shared content widens instance write only for that swarm."""
        await memory_store(
            key="sw3w",
            content="Shared project board.",
            session_num=1,
            session_date="2026-03-20",
            agent_id="alice",
            swarm_id="alpha",
            scope="swarm-shared",
        )

        server = mcp_mod.registry["sw3w"]
        assert "swarm:alpha" in server._instance_config["_derived_write"]
        assert server._instance_config["write"] == []

        ok = await memory_store(
            key="sw3w",
            content="Bob adds a follow-up note.",
            session_num=2,
            session_date="2026-03-21",
            agent_id="bob",
            swarm_id="alpha",
            scope="swarm-shared",
        )
        assert ok.get("code") not in ("FORBIDDEN", "ACL_FORBIDDEN"), ok

        blocked = await memory_store(
            key="sw3w",
            content="Mallory should not write here.",
            session_num=3,
            session_date="2026-03-21",
            agent_id="mallory",
            swarm_id="beta",
            scope="swarm-shared",
        )
        assert blocked.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN"), blocked

    @pytest.mark.asyncio
    async def test_swarm_shared_instance_write_allows_asserted_task_like_facts(self, tmp_dir):
        """Task-like asserted facts can be written by the collaborating swarm."""
        await memory_store(
            key="sw3task",
            content="Shared task board.",
            session_num=1,
            session_date="2026-03-20",
            agent_id="alice",
            swarm_id="alpha",
            scope="swarm-shared",
        )

        ok = await memory_ingest_asserted_facts(
            key="sw3task",
            agent_id="bob",
            swarm_id="alpha",
            facts=[{
                "id": "task_sw3task_1",
                "fact": "Implement the next watcher step.",
                "kind": "task",
                "session": 1,
                "entities": [],
                "tags": ["task"],
                "scope": "swarm-shared",
                "swarm_id": "alpha",
                "metadata": {"task_id": "task-sw3task-1"},
            }],
        )
        assert ok.get("code") not in ("FORBIDDEN", "ACL_FORBIDDEN"), ok

        blocked = await memory_ingest_asserted_facts(
            key="sw3task",
            agent_id="mallory",
            swarm_id="beta",
            facts=[{
                "id": "task_sw3task_2",
                "fact": "Mallory should not be able to inject this.",
                "kind": "task",
                "session": 1,
                "entities": [],
                "tags": ["task"],
                "scope": "swarm-shared",
                "swarm_id": "beta",
                "metadata": {"task_id": "task-sw3task-2"},
            }],
        )
        assert blocked.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN"), blocked

    @pytest.mark.asyncio
    async def test_swarm_shared_instance_read_allows_recall_and_ask(self, tmp_dir, monkeypatch):
        await memory_store(
            key="sw3b",
            content="Owner private fact.",
            session_num=1,
            session_date="2026-03-20",
            agent_id="alice",
            scope="agent-private",
        )
        await memory_store(
            key="sw3b",
            content="Shared region is eu-west-1.",
            session_num=2,
            session_date="2026-03-20",
            agent_id="alice",
            swarm_id="alpha",
            scope="swarm-shared",
        )

        async def _fake_recall(**kwargs):
            return {
                "context": "Shared region is eu-west-1.",
                "retrieved": [],
                "query_type": "default",
                "complexity_hint": {
                    "score": 0.0,
                    "level": 1,
                    "signals": [],
                    "retrieval_complexity": 0.0,
                    "content_complexity": 0.0,
                    "dominant": "tie",
                },
                "sessions_in_context": 1,
                "total_sessions": 1,
                "coverage_pct": 100.0,
                "raw_budget": 0,
                "recommended_prompt_type": "lookup",
                "use_tool": False,
            }

        async def _fake_ask(**kwargs):
            return {"answer": "Shared region is eu-west-1."}

        monkeypatch.setattr(mcp_mod.registry["sw3b"], "recall", _fake_recall)
        monkeypatch.setattr(mcp_mod.registry["sw3b"], "ask", _fake_ask)

        recall_ok = await memory_recall(
            key="sw3b",
            query="What is the shared region?",
            agent_id="bob",
            swarm_id="alpha",
        )
        assert recall_ok.get("code") not in ("FORBIDDEN", "ACL_FORBIDDEN"), recall_ok
        assert recall_ok.get("context") == "Shared region is eu-west-1."

        ask_ok = await memory_ask(
            key="sw3b",
            query="What is the shared region?",
            agent_id="bob",
            swarm_id="alpha",
        )
        assert ask_ok.get("code") not in ("FORBIDDEN", "ACL_FORBIDDEN"), ask_ok
        assert ask_ok.get("answer") == "Shared region is eu-west-1."

    @pytest.mark.asyncio
    async def test_swarm_shared_instance_read_does_not_expose_key_wide_config(self, tmp_dir):
        await memory_store(
            key="sw4",
            content="Owner private fact.",
            session_num=1,
            session_date="2026-03-20",
            agent_id="alice",
            scope="agent-private",
        )
        await memory_set_config(
            key="sw4",
            config={
                "schema_version": 1,
                "embedding_model": "openai/text-embedding-3-large",
                "librarian_profile": "openai/gpt-oss-120b",
                "profiles": {1: "fast"},
                "profile_configs": {"fast": {"model": "m"}},
                "retrieval": {"search_family": "auto", "default_token_budget": 4000},
            },
            agent_id="alice",
        )
        await memory_set_profiles(
            key="sw4",
            profiles={1: "fast"},
            profile_configs={"fast": {"model": "m"}},
            agent_id="alice",
        )
        await memory_store(
            key="sw4",
            content="Shared region is eu-west-1.",
            session_num=2,
            session_date="2026-03-20",
            agent_id="alice",
            swarm_id="alpha",
            scope="swarm-shared",
        )

        list_ok = await memory_list(key="sw4", agent_id="bob", swarm_id="alpha")
        assert list_ok.get("code") not in ("FORBIDDEN", "ACL_FORBIDDEN"), list_ok

        cfg = await memory_get_config(key="sw4", agent_id="bob", swarm_id="alpha")
        assert cfg.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN"), cfg

        profiles = await memory_get_profiles(key="sw4", agent_id="bob", swarm_id="alpha")
        assert profiles.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN"), profiles

        stats = await memory_stats(key="sw4", agent_id="bob")
        assert stats.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN"), stats


# ===========================================================================
# 3. First-write instance creation via memory_ingest_document
# ===========================================================================

class TestIngestDocumentCreatesInstance:

    @pytest.mark.asyncio
    async def test_ingest_document_creates_config(self, tmp_dir):
        """First authenticated MCP write via memory_ingest_document creates persisted config."""
        r = await memory_ingest_document(
            key="doc1", content="Document content for testing.",
            source_id="test-doc", agent_id="doc_owner")

        assert "error" not in r or r.get("code") not in ("FORBIDDEN", "ACL_FORBIDDEN"), \
            f"First write should succeed, got {r}"

        server = mcp_mod.registry["doc1"]
        assert server._instance_config is not None, \
            "Instance config should be created on first ingest_document"
        assert server._instance_config["owner_id"] == "agent:doc_owner"
        assert server._instance_config["_derived_read"] == ["agent:PUBLIC"]
        assert server._instance_config["_derived_write"] == []
        assert server._instance_config["write"] == []

    @pytest.mark.asyncio
    async def test_ingest_document_config_persists(self, tmp_dir):
        """Instance config from ingest_document survives registry clear + reload."""
        await memory_ingest_document(
            key="doc2", content="Persistent doc.", source_id="doc-persist",
            agent_id="doc_persister")

        mcp_mod.registry.clear()
        server = mcp_mod._get_memory("doc2")
        assert server._instance_config is not None, \
            "Instance config should persist across restart"
        assert server._instance_config["owner_id"] == "agent:doc_persister"

    @pytest.mark.asyncio
    async def test_second_ingest_document_outsider_blocked(self, tmp_dir):
        """After first write creates config, outsider is blocked."""
        await memory_ingest_document(
            key="doc3", content="First.", source_id="d3", agent_id="owner")
        r = await memory_ingest_document(
            key="doc3", content="Second.", source_id="d3b", agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN"), \
            f"Outsider should be blocked after instance creation, got {r}"


# ===========================================================================
# 4. memory_import auth split
# ===========================================================================

class TestImportAuthSplit:

    @pytest.mark.asyncio
    async def test_token_is_source_auth_not_caller(self, tmp_dir):
        """token param is for source/repo auth, not caller identity.
        agent_key is for caller identity. They must be separate."""
        # Seed instance as owner
        await _seed("imp", agent_id="owner")

        # memory_import signature has both `token` and `agent_key`
        # token = source auth (e.g. git clone token)
        # agent_key = caller identity auth
        # Calling with token but wrong agent_id should still be blocked
        r = await memory_import(
            key="imp", source_format="text", content="imported",
            agent_id="outsider", token="some-source-token")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN"), \
            f"token should not grant caller identity, got {r}"

    @pytest.mark.asyncio
    async def test_agent_key_resolves_caller(self, tmp_dir):
        """agent_key param should resolve caller identity separately from token."""
        await _seed("imp2", agent_id="owner")

        # Grant write to agent:importer via instance ACL
        mcp_mod.registry["imp2"]._instance_config["write"] = ["agent:importer"]

        # Use agent_key for identity (not token)
        r = await memory_import(
            key="imp2", source_format="text", content="imported text",
            agent_key="agent:importer",
            token="git-clone-token-for-source")
        # Should pass instance ACL because agent_key resolves to agent:importer
        assert r.get("code") not in ("FORBIDDEN", "ACL_FORBIDDEN"), \
            f"agent_key should resolve caller identity, got {r}"

    @pytest.mark.asyncio
    async def test_token_without_agent_key_uses_agent_id(self, tmp_dir):
        """Without agent_key, caller identity falls back to agent_id param."""
        await _seed("imp3", agent_id="owner")

        # Owner can import (identity from agent_id)
        r = await memory_import(
            key="imp3", source_format="text", content="owner import",
            agent_id="owner", token="source-token")
        assert r.get("code") not in ("FORBIDDEN", "ACL_FORBIDDEN"), \
            f"Owner import should pass, got {r}"


# ===========================================================================
# 5. Cross-key prompt isolation
# ===========================================================================

class TestCrossKeyPromptIsolation:

    @pytest.mark.asyncio
    async def test_prompt_set_on_key_a_not_visible_on_key_b(self, tmp_dir):
        """Custom prompt set on key 'a' must NOT be visible on key 'b'."""
        # Seed both keys
        await _seed("prompt_a", agent_id="owner")
        await _seed("prompt_b", agent_id="owner")

        # Set custom prompt on key a
        r = await memory_set_prompt(
            key="prompt_a", content_type="custom_type",
            prompt="Custom prompt for key A only", agent_id="owner")
        assert r.get("stored") is True or "error" not in r

        # Verify key a has the prompt
        r_a = await memory_get_prompt(
            key="prompt_a", content_type="custom_type", agent_id="owner")
        assert r_a.get("prompt") == "Custom prompt for key A only"

        # Verify key b does NOT have this prompt
        r_b = await memory_get_prompt(
            key="prompt_b", content_type="custom_type", agent_id="owner")
        # Should either be not found or return builtin/different prompt
        assert r_b.get("prompt") != "Custom prompt for key A only", \
            f"Key B should NOT see key A's custom prompt, got {r_b}"

    @pytest.mark.asyncio
    async def test_prompt_isolation_after_restart(self, tmp_dir):
        """Cross-key isolation persists across registry restart."""
        await _seed("iso_a", agent_id="owner")
        await _seed("iso_b", agent_id="owner")

        await memory_set_prompt(
            key="iso_a", content_type="special",
            prompt="A-only prompt", agent_id="owner")

        # Restart
        mcp_mod.registry.clear()

        r_b = await memory_get_prompt(
            key="iso_b", content_type="special", agent_id="owner")
        assert r_b.get("prompt") != "A-only prompt", \
            f"After restart, key B should not see key A's prompt"
