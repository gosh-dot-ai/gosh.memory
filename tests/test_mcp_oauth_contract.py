"""Acceptance contract: MCP OAuth identity resolution.

Freezes the contract for replacing placeholder _TOKEN_IDENTITIES
with real MCP-auth-backed identity resolution (SPEC §3, R35).

NOTE on agent_key: Tests that use agent_key as identity carrier are
marked [TRANSITIONAL]. agent_key is an unverified caller-supplied header.
When _verified_auth_resolver is wired to real MCP OAuth, these tests
must be updated to use the verified auth path.

Contract points:
1. Hardcoded fake token strings must NOT define production identity mapping
2. Canonical owner_id format is enforced
3. Token (auth attempt) wins over agent_key
4. Token (auth attempt) wins over agent_id / swarm_id
5. Unknown/unverified token does NOT impersonate via local stub
6. Admin path still works (GOSH_MEMORY_ADMIN_TOKEN env var)
7. Direct MemoryServer calls remain outside OAuth enforcement
8. ACL-sensitive MCP tools behave correctly once identity is resolved
9. memory_import keeps caller identity separate from source auth
10. Placeholder behavior must not survive in production under a different name
11. _verified_auth_resolver adapter (R35) resolves token → canonical owner_id
"""

import asyncio
import importlib
import os
import tempfile

import pytest

import src.mcp_server as mcp_mod
from src.mcp_server import (
    _TOKEN_IDENTITIES,
    ConnectionContext,
    _resolve_identity,
    memory_import,
    memory_list,
    memory_recall,
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
    return await memory_store(
        key=key, content="Seed.", session_num=1,
        session_date="2026-03-20", agent_id=agent_id,
    )


# ===========================================================================
# CONTRACT 1: Placeholder token map must NOT be production identity source
# ===========================================================================

class TestPlaceholderNotProduction:

    def test_token_identities_is_empty(self):
        """_TOKEN_IDENTITIES must be empty in production."""
        assert len(_TOKEN_IDENTITIES) == 0, (
            f"_TOKEN_IDENTITIES still has {len(_TOKEN_IDENTITIES)} "
            f"hardcoded entries: {list(_TOKEN_IDENTITIES.keys())}."
        )

    def test_fake_token_string_does_not_resolve(self):
        """Arbitrary string 'test-token-alice' must NOT map to user:alice."""
        ctx = _resolve_identity(token="test-token-alice")
        assert ctx.owner_id != "user:alice"

    def test_unknown_token_does_not_impersonate(self):
        """Random bearer token must not impersonate any user."""
        ctx = _resolve_identity(token="random-garbage-token-12345")
        assert ctx.owner_id == "system"
        assert ctx.caller_role == "user"


# ===========================================================================
# CONTRACT 2: Canonical owner_id format
# ===========================================================================

class TestCanonicalIdentity:

    def test_param_derived_identity_is_canonical(self):
        """agent_id param → canonical agent:X format."""
        ctx = _resolve_identity(agent_id="alice")
        assert ctx.owner_id == "agent:alice"
        assert ":" in ctx.owner_id

    def test_agent_key_passthrough_transitional(self):
        """[TRANSITIONAL] agent_key passes through as owner_id."""
        ctx = _resolve_identity(agent_key="user:alice")
        assert ctx.owner_id == "user:alice"

        ctx2 = _resolve_identity(agent_key="agent:bot")
        assert ctx2.owner_id == "agent:bot"

    @pytest.mark.asyncio
    async def test_identity_grants_data_access(self, tmp_dir):
        """End-to-end: resolved identity grants access to owned data."""
        await memory_store(
            key="vauth", content="Alice data.", session_num=1,
            session_date="2026-03-20", agent_id="alice")

        r = await memory_list(key="vauth", agent_id="alice")
        assert r.get("code") not in ("FORBIDDEN", "ACL_FORBIDDEN"), \
            f"Owner should have access, got {r}"

        r2 = await memory_list(key="vauth", agent_id="bob")
        assert r2.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN"), \
            f"Different identity should be blocked, got {r2}"


# ===========================================================================
# CONTRACT 3-4: Token (auth attempt) wins over agent_key and agent_id
# ===========================================================================

class TestAuthPrecedence:

    def test_token_wins_over_agent_key(self):
        """Token present → agent_key is ignored."""
        ctx = _resolve_identity(
            token="test-token-alice",
            agent_key="agent:evil-impersonator",
        )
        assert ctx.owner_id != "agent:evil-impersonator", (
            "agent_key must NOT override token-based auth attempt"
        )

    def test_token_wins_over_agent_id(self):
        """Token present → agent_id is ignored for owner_id."""
        ctx = _resolve_identity(
            token="test-token-alice",
            agent_id="forged-identity",
        )
        assert ctx.owner_id != "agent:forged-identity"

    def test_token_identity_independent_of_swarm_id(self):
        """Token-resolved identity is independent of swarm_id param."""
        ctx1 = _resolve_identity(token="test-token-alice", swarm_id="alpha")
        ctx2 = _resolve_identity(token="test-token-alice", swarm_id="beta")
        assert ctx1.owner_id == ctx2.owner_id


# ===========================================================================
# CONTRACT 5: Admin path
# ===========================================================================

class TestAdminPath:

    def test_admin_token_still_works(self, monkeypatch):
        """GOSH_MEMORY_ADMIN_TOKEN env var → admin role, system owner."""
        monkeypatch.setenv("GOSH_MEMORY_ADMIN_TOKEN", "real-admin-secret")
        importlib.reload(mcp_mod)

        ctx = mcp_mod._resolve_identity(token="real-admin-secret")
        assert ctx.caller_role == "admin"
        assert ctx.owner_id == "system"

        importlib.reload(mcp_mod)

    def test_admin_token_not_in_placeholder_map(self):
        """Admin must come from env var, not from _TOKEN_IDENTITIES."""
        ctx = _resolve_identity(token="test-token-admin")
        assert ctx.caller_role != "admin"


# ===========================================================================
# CONTRACT 6: Direct MemoryServer calls remain trusted
# ===========================================================================

class TestDirectCallsTrusted:

    def test_direct_recall_no_oauth(self, tmp_dir):
        """Direct MemoryServer.recall() does not enforce OAuth."""
        ms = MemoryServer(data_dir=tmp_dir, key="direct")
        asyncio.run(ms.ingest_asserted_facts(
            facts=[{"fact": "Test", "kind": "fact", "session": 1,
                    "entities": [], "tags": []}],
            raw_sessions=[{"session_num": 1, "session_date": "2026-03-20",
                           "content": "test"}],
            enrich_l0=False))
        asyncio.run(ms.build_index())
        r = asyncio.run(ms.recall("test", caller_role="admin"))
        assert "error" not in r

    def test_direct_ingest_no_oauth(self, tmp_dir):
        """Direct MemoryServer.ingest_asserted_facts() no OAuth check."""
        ms = MemoryServer(data_dir=tmp_dir, key="direct2")
        r = asyncio.run(ms.ingest_asserted_facts(
            facts=[{"fact": "X", "kind": "fact", "session": 1,
                    "entities": [], "tags": []}],
            raw_sessions=[{"session_num": 1, "session_date": "2026-03-20",
                           "content": "x"}],
            enrich_l0=False))
        assert "error" not in r


# ===========================================================================
# CONTRACT 7: ACL tools work correctly with resolved identity
# ===========================================================================

class TestACLWithResolvedIdentity:

    @pytest.mark.asyncio
    async def test_recall_uses_resolved_identity_for_acl(self, tmp_dir):
        await _seed("acl1", agent_id="owner")

        r = await memory_recall(key="acl1", query="test", agent_id="owner")
        assert r.get("code") not in ("FORBIDDEN", "ACL_FORBIDDEN")

        r = await memory_recall(key="acl1", query="test", agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")

    @pytest.mark.asyncio
    async def test_list_uses_resolved_identity_for_acl(self, tmp_dir):
        await _seed("acl2", agent_id="owner")
        r = await memory_list(key="acl2", agent_id="outsider")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")


# ===========================================================================
# CONTRACT 8: memory_import auth split
# ===========================================================================

class TestImportAuthSplit:

    @pytest.mark.asyncio
    async def test_token_is_source_auth_not_identity(self, tmp_dir):
        """token param in memory_import is source auth, NOT caller identity."""
        await _seed("imp", agent_id="owner")

        r = await memory_import(
            key="imp", source_format="text", content="imported",
            agent_id="outsider", token="some-source-token")
        assert r.get("code") in ("FORBIDDEN", "ACL_FORBIDDEN")

    @pytest.mark.asyncio
    async def test_agent_id_is_caller_identity(self, tmp_dir):
        """agent_id resolves caller identity, separate from source token."""
        await _seed("imp2", agent_id="owner")

        r = await memory_import(
            key="imp2", source_format="text", content="text",
            agent_id="owner")
        assert r.get("code") not in ("FORBIDDEN", "ACL_FORBIDDEN")


# ===========================================================================
# CONTRACT 9: No placeholder survival under a different name
# ===========================================================================

class TestNoPlaceholderSurvival:

    def test_no_hardcoded_token_to_identity_dict(self):
        """No in-code dict mapping token strings to identities."""
        import inspect
        source = inspect.getsource(mcp_mod)

        assert "test-token-alice" not in source
        assert "test-token-bob" not in source

    def test_no_local_bearer_impersonation(self):
        """No local dict should allow arbitrary bearer tokens to map to users."""
        for fake in ["bearer-user-alice", "oauth-token-123", "sk-test-key"]:
            ctx = _resolve_identity(token=fake)
            assert not ctx.owner_id.startswith("user:")


# ===========================================================================
# CONTRACT 10: Fallback when no auth present
# ===========================================================================

class TestNoAuthFallback:

    def test_no_auth_defaults_to_system(self):
        ctx = _resolve_identity()
        assert ctx.owner_id == "system"
        assert ctx.caller_role == "user"

    def test_agent_id_fallback(self):
        ctx = _resolve_identity(agent_id="alice")
        assert ctx.owner_id == "agent:alice"

    def test_fallback_is_limited(self):
        """agent_id='admin' should NOT grant admin role."""
        ctx = _resolve_identity(agent_id="admin")
        assert ctx.caller_role == "user"
        assert ctx.owner_id == "agent:admin"


# ===========================================================================
# CONTRACT 11: _verified_auth_resolver adapter (R35)
# ===========================================================================

class TestVerifiedAuthAdapter:

    def test_resolver_maps_token_to_owner_id(self, monkeypatch):
        """When _verified_auth_resolver is set, valid token → canonical owner_id."""
        monkeypatch.setattr(mcp_mod, "_verified_auth_resolver",
                            lambda t: "user:alice" if t == "valid" else None)

        ctx = mcp_mod._resolve_identity(token="valid", agent_key="agent:evil")
        assert ctx.owner_id == "user:alice"
        assert ctx.owner_id != "agent:evil"

    def test_resolver_unknown_token_blocks_fallback(self, monkeypatch):
        """Resolver returns None → owner_id=system, agent_key ignored."""
        monkeypatch.setattr(mcp_mod, "_verified_auth_resolver", lambda t: None)

        ctx = mcp_mod._resolve_identity(token="bad", agent_key="agent:evil")
        assert ctx.owner_id == "system"

    def test_resolver_not_set_defaults_to_system(self):
        """No resolver configured + unknown token → system."""
        ctx = _resolve_identity(token="any-token")
        assert ctx.owner_id == "system"

    def test_admin_token_bypasses_resolver(self, monkeypatch):
        """Admin token check happens before _verified_auth_resolver."""
        monkeypatch.setenv("GOSH_MEMORY_ADMIN_TOKEN", "admin-secret")
        importlib.reload(mcp_mod)
        monkeypatch.setattr(mcp_mod, "_verified_auth_resolver",
                            lambda t: "user:hacker")

        ctx = mcp_mod._resolve_identity(token="admin-secret")
        assert ctx.caller_role == "admin"
        assert ctx.owner_id == "system"

        importlib.reload(mcp_mod)


# ===========================================================================
# CONTRACT 12: Resolver output must be canonical
# ===========================================================================

class TestResolverCanonicalValidation:

    def test_resolver_bare_name_rejected_or_normalized(self, monkeypatch):
        """Resolver returning 'alice' (no prefix) must NOT become owner_id.

        Either: _resolve_identity normalizes to canonical form,
        or: rejects and falls back to system.
        'alice' without user:/agent:/swarm: violates canonical principal contract.
        """
        monkeypatch.setattr(mcp_mod, "_verified_auth_resolver",
                            lambda t: "alice")

        ctx = mcp_mod._resolve_identity(token="valid")
        assert ":" in ctx.owner_id or ctx.owner_id == "system", (
            f"Non-canonical resolver output 'alice' accepted as owner_id={ctx.owner_id}. "
            f"Must be user:alice, agent:alice, or rejected to system."
        )
        assert ctx.owner_id != "alice", (
            "Bare 'alice' without canonical prefix must not be stored as owner_id"
        )

    def test_resolver_canonical_output_accepted(self, monkeypatch):
        """Resolver returning 'user:alice' (canonical) is accepted."""
        monkeypatch.setattr(mcp_mod, "_verified_auth_resolver",
                            lambda t: "user:alice")

        ctx = mcp_mod._resolve_identity(token="valid")
        assert ctx.owner_id == "user:alice"

    def test_resolver_empty_string_rejected(self, monkeypatch):
        """Resolver returning '' must not be stored as owner_id."""
        monkeypatch.setattr(mcp_mod, "_verified_auth_resolver",
                            lambda t: "")

        ctx = mcp_mod._resolve_identity(token="valid")
        assert ctx.owner_id == "system"


# ===========================================================================
# CONTRACT 13: memory_import verified-auth path
# ===========================================================================

class TestImportVerifiedAuth:

    @pytest.mark.asyncio
    async def test_import_can_use_verified_auth_for_caller(self, tmp_dir, monkeypatch):
        """memory_import must support verified caller auth separate from source token.

        Currently `token` is reserved for source auth (git clone).
        Verified caller auth must come through a separate channel.
        """
        await _seed("imp_auth", agent_id="owner")

        r = await memory_import(
            key="imp_auth", source_format="text", content="imported",
            agent_id="owner",
            token="git-clone-token")

        assert r.get("code") not in ("FORBIDDEN", "ACL_FORBIDDEN"), \
            f"Owner import should pass, got {r}"
