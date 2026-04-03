"""Unit 4: Cross-session consolidation restricted to ACL domain."""

import asyncio

import numpy as np
import pytest

from src.memory import MemoryServer

DIM = 3072


def _patch_all(monkeypatch):
    async def mock_extract(**kwargs):
        sn = kwargs.get("session_num", 1)
        return ("conv", sn, "2024-06-01", [
            {"id": f"f{sn}_0", "fact": f"Fact {sn}", "kind": "event",
             "entities": ["Alice"], "tags": [], "session": sn}], [])

    async def mock_consolidate(**kwargs):
        sn = kwargs.get("sn", 1)
        sfacts = kwargs.get("session_facts", [])
        # Return cons facts that reflect the input domain
        owner = sfacts[0].get("owner_id", "system") if sfacts else "system"
        return ("conv", sn, "2024-06-01", [
            {"id": f"c{sn}_{owner[:6]}", "fact": f"Cons {owner}",
             "kind": "summary", "entities": ["Alice"], "tags": []}])

    async def mock_cross(**kwargs):
        efacts = kwargs.get("efacts", [])
        owner = efacts[0].get("owner_id", "system") if efacts else "system"
        ename = kwargs.get("ename", "alice")
        return ("conv", ename, [
            {"id": f"x_{ename}_{owner[:6]}", "fact": f"Cross {owner} {ename}",
             "kind": "profile", "entities": [ename], "tags": []}])

    async def mock_extract_source_aggregation_facts(self, **kwargs):
        source_facts = kwargs.get("source_facts", [])
        source_id = kwargs.get("source_id", "source")
        owner = source_facts[0].get("owner_id", "system") if source_facts else "system"
        return [{
            "id": f"substrate_{owner[:6]}",
            "fact": f"Substrate {owner} {source_id}",
            "kind": "fact",
            "entities": ["Alice"],
            "tags": ["substrate"],
            "source_ids": [f["id"] for f in source_facts],
            "metadata": {"source_aggregation": True},
        }]

    async def mock_embed(texts, **kw):
        return np.random.randn(len(texts), DIM).astype(np.float32)

    async def mock_embed_q(text, **kw):
        return np.random.randn(DIM).astype(np.float32)

    monkeypatch.setattr("src.memory.extract_session", mock_extract)
    monkeypatch.setattr("src.memory.consolidate_session", mock_consolidate)
    monkeypatch.setattr("src.memory.cross_session_entity", mock_cross)
    monkeypatch.setattr(MemoryServer, "_extract_source_aggregation_facts", mock_extract_source_aggregation_facts)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda f, l: None)
    monkeypatch.setattr("src.memory.embed_texts", mock_embed)
    monkeypatch.setattr("src.memory.embed_query", mock_embed_q)


def test_rebuild_tiers_groups_by_acl_domain(tmp_path, monkeypatch):
    """Two owners' facts in same store → substrate cross stays ACL-separated."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "acl_dom1")

    # Owner 1: agent:alice
    asyncio.run(ms.store("Alice data 1", session_num=1, session_date="2024-06-01",
                         owner_id="agent:alice", read=["swarm:alpha"], write=[]))
    asyncio.run(ms.store("Alice data 2", session_num=2, session_date="2024-06-02",
                         owner_id="agent:alice", read=["swarm:alpha"], write=[]))

    # Owner 2: agent:bob
    asyncio.run(ms.store("Bob data 1", session_num=3, session_date="2024-06-01",
                         owner_id="agent:bob", read=["swarm:beta"], write=[]))
    asyncio.run(ms.store("Bob data 2", session_num=4, session_date="2024-06-02",
                         owner_id="agent:bob", read=["swarm:beta"], write=[]))

    # Rebuild tiers
    asyncio.run(ms._rebuild_tiers())

    assert ms._all_cons == []

    # Verify cross-session facts also separated
    alice_cross = [f for f in ms._all_cross if f.get("owner_id") == "agent:alice"]
    bob_cross = [f for f in ms._all_cross if f.get("owner_id") == "agent:bob"]

    # Each owner's cross facts should only reference their own data
    for f in alice_cross:
        assert f.get("read") == ["swarm:alpha"]
    for f in bob_cross:
        assert f.get("read") == ["swarm:beta"]


def test_rebuild_tiers_acl_domain_key_includes_read(tmp_path, monkeypatch):
    """Same owner but different read ACL → substrate cross stays separated."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "acl_dom2")

    # Same owner, different read lists
    asyncio.run(ms.store("Data for alpha", session_num=1, session_date="2024-06-01",
                         owner_id="agent:alice", read=["swarm:alpha"], write=[]))
    asyncio.run(ms.store("Data for alpha 2", session_num=2, session_date="2024-06-02",
                         owner_id="agent:alice", read=["swarm:alpha"], write=[]))
    asyncio.run(ms.store("Data for beta", session_num=3, session_date="2024-06-01",
                         owner_id="agent:alice", read=["swarm:beta"], write=[]))
    asyncio.run(ms.store("Data for beta 2", session_num=4, session_date="2024-06-02",
                         owner_id="agent:alice", read=["swarm:beta"], write=[]))

    asyncio.run(ms._rebuild_tiers())

    assert ms._all_cons == []
    alpha_cross = [f for f in ms._all_cross if f.get("read") == ["swarm:alpha"]]
    beta_cross = [f for f in ms._all_cross if f.get("read") == ["swarm:beta"]]
    assert len(alpha_cross) > 0
    assert len(beta_cross) > 0
