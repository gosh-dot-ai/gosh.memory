"""Document ingest metadata + target tests."""

import pytest

import src.mcp_server as mcp_mod
from src.mcp_server import memory_ingest_document
from src.memory import MemoryServer


def _patch_document_extract(monkeypatch):
    async def mock_extract_session(**kwargs):
        sn = kwargs.get("session_num", 1)
        facts = [
            {
                "id": "doc_f0",
                "fact": "Extracted spec fact",
                "kind": "fact",
                "entities": ["spec"],
                "tags": ["doc"],
                "session": sn,
                "metadata": {
                    "section_path": "Extracted Section",
                    "version_status": "current",
                },
            }
        ]
        return ("conv", sn, "2024-06-01", facts, [])

    async def mock_consolidate_session(**kwargs):
        session_facts = kwargs.get("session_facts", [])
        return ("conv", kwargs.get("sn", 1), "2024-06-01", [
            {
                "id": f"cons_{kwargs.get('sn', 1)}",
                "fact": "Consolidated doc fact",
                "kind": "summary",
                "entities": ["spec"],
                "tags": ["doc"],
                "source_ids": [f["id"] for f in session_facts],
            }
        ])

    async def mock_cross_session_entity(**kwargs):
        efacts = kwargs.get("efacts", [])
        return ("conv", "spec", [
            {
                "id": "cross_spec",
                "fact": "Cross-session doc fact",
                "kind": "profile",
                "entities": ["spec"],
                "tags": ["doc"],
                "source_ids": [f["id"] for f in efacts],
            }
        ])

    async def mock_extract_source_aggregation_facts(self, **kwargs):
        source_facts = kwargs.get("source_facts", [])
        if not source_facts:
            return []
        return [{
            "id": "substrate_spec",
            "fact": "Substrate doc fact",
            "kind": "fact",
            "entities": ["spec"],
            "tags": ["doc"],
            "source_ids": [f["id"] for f in source_facts],
            "metadata": {"source_aggregation": True},
        }]

    monkeypatch.setattr("src.memory.extract_session", mock_extract_session)
    monkeypatch.setattr("src.memory.consolidate_session", mock_consolidate_session)
    monkeypatch.setattr("src.memory.cross_session_entity", mock_cross_session_entity)
    monkeypatch.setattr(MemoryServer, "_extract_source_aggregation_facts", mock_extract_source_aggregation_facts)


@pytest.mark.asyncio
async def test_ingest_document_merges_metadata_and_target(tmp_path, monkeypatch):
    _patch_document_extract(monkeypatch)
    ms = MemoryServer(str(tmp_path), "doc_meta")

    n = await ms.ingest_document(
        content="Document body",
        source_id="DOC-001",
        metadata={
            "workflow_id": "wf_17",
            "section_path": "Caller Section",
            "document_source": "caller-doc",
        },
        target=["agent:planner", "agent:coder", "agent:planner"],
    )

    assert n == 1
    fact = ms._all_granular[0]
    assert fact["target"] == ["agent:planner", "agent:coder"]
    assert fact["metadata"]["workflow_id"] == "wf_17"
    assert fact["metadata"]["section_path"] == "Caller Section"
    assert fact["metadata"]["version_status"] == "current"
    assert fact["metadata"]["document_source"] == "caller-doc"

    raw = ms._raw_sessions[0]
    assert raw["metadata"] == {
        "workflow_id": "wf_17",
        "section_path": "Caller Section",
        "document_source": "caller-doc",
    }
    assert raw["target"] == ["agent:planner", "agent:coder"]


@pytest.mark.asyncio
async def test_memory_ingest_document_tool_accepts_metadata_and_target(tmp_path, monkeypatch):
    _patch_document_extract(monkeypatch)
    mcp_mod.data_dir = str(tmp_path)
    mcp_mod.registry.clear()
    mcp_mod.courier_registry.clear()
    mcp_mod.connections.clear()
    mcp_mod.sub_to_conn.clear()
    mcp_mod._active_connections.clear()

    result = await memory_ingest_document(
        key="doc_tool",
        content="Document body",
        source_id="DOC-002",
        metadata={"workflow_id": "wf_tool"},
        target="agent:planner",
    )

    assert result == {"facts_extracted": 1}
    fact = mcp_mod.registry["doc_tool"]._all_granular[0]
    assert fact["target"] == ["agent:planner"]
    assert fact["metadata"]["workflow_id"] == "wf_tool"


@pytest.mark.asyncio
async def test_ingest_document_propagates_target_to_derived_tiers(tmp_path, monkeypatch):
    _patch_document_extract(monkeypatch)
    ms = MemoryServer(str(tmp_path), "doc_meta_derived")

    chunk1 = "First section. " * 600
    chunk2 = "Second section. " * 600
    n = await ms.ingest_document(
        content=f"{chunk1}\n\n{chunk2}",
        source_id="DOC-004",
        target="agent:planner",
    )

    assert n == 2
    assert ms._all_granular
    assert ms._all_cross
    assert all(f.get("target") == ["agent:planner"] for f in ms._all_granular)
    assert ms._all_cons == []
    assert all(f.get("target") == ["agent:planner"] for f in ms._all_cross)


@pytest.mark.asyncio
async def test_document_ingest_without_metadata_still_works(tmp_path, monkeypatch):
    _patch_document_extract(monkeypatch)
    ms = MemoryServer(str(tmp_path), "doc_meta_legacy")

    n = await ms.ingest_document(
        content="Document body",
        source_id="DOC-003",
    )

    assert n == 1
    fact = ms._all_granular[0]
    assert fact["metadata"]["document_source"] == "DOC-003"
    assert "target" not in fact


@pytest.mark.asyncio
async def test_multipart_document_ingest_appends_parts_without_superseding_prior_parts(tmp_path, monkeypatch):
    _patch_document_extract(monkeypatch)
    ms = MemoryServer(str(tmp_path), "doc_meta_parts")

    first = await ms.ingest_document(
        content="[Step 1] take keycard\n[Step 2] open drawer",
        source_id="DOC-PARTS",
        metadata={"part_source_id": "DOC-PARTS_p001", "part_idx": 1},
    )
    second = await ms.ingest_document(
        content="[Step 40] move book 1\n[Step 41] take book 2",
        source_id="DOC-PARTS",
        metadata={"part_source_id": "DOC-PARTS_p002", "part_idx": 2},
    )

    assert first == 1
    assert second == 1
    assert len(ms._raw_sessions) == 2
    assert all(rs.get("status") == "active" for rs in ms._raw_sessions)
    assert [rs.get("part_source_id") for rs in ms._raw_sessions] == [
        "DOC-PARTS_p001",
        "DOC-PARTS_p002",
    ]
    assert [rs.get("session_num") for rs in ms._raw_sessions] == [1, 2]

    docs = [doc for doc in ms._episode_corpus["documents"] if doc["doc_id"] == "document:DOC-PARTS"]
    assert len(docs) == 1
    episodes = docs[0]["episodes"]
    assert [ep["episode_id"] for ep in episodes] == ["DOC-PARTS_e01", "DOC-PARTS_e02"]
    assert episodes[0]["raw_text"].startswith("[Step 1]")
    assert episodes[1]["raw_text"].startswith("[Step 40]")
    assert [
        (ep.get("provenance") or {}).get("multipart_part_key")
        for ep in episodes
    ] == ["DOC-PARTS_p001", "DOC-PARTS_p002"]

    granular_episode_ids = [fact["metadata"]["episode_id"] for fact in ms._all_granular]
    assert granular_episode_ids == ["DOC-PARTS_e01", "DOC-PARTS_e02"]
    assert ms._raw_docs["DOC-PARTS"].startswith("[Step 1]")
    assert "[Step 40]" in ms._raw_docs["DOC-PARTS"]


@pytest.mark.asyncio
async def test_multipart_document_reingest_replaces_only_matching_part(tmp_path, monkeypatch):
    _patch_document_extract(monkeypatch)
    ms = MemoryServer(str(tmp_path), "doc_meta_parts_replace")

    await ms.ingest_document(
        content="[Step 1] initial early step",
        source_id="DOC-PARTS-REPLACE",
        metadata={"part_source_id": "DOC-PARTS-REPLACE_p001", "part_idx": 1},
    )
    await ms.ingest_document(
        content="[Step 40] stable late step",
        source_id="DOC-PARTS-REPLACE",
        metadata={"part_source_id": "DOC-PARTS-REPLACE_p002", "part_idx": 2},
    )
    updated = await ms.ingest_document(
        content="[Step 1] updated early step",
        source_id="DOC-PARTS-REPLACE",
        metadata={"part_source_id": "DOC-PARTS-REPLACE_p001", "part_idx": 1},
    )

    assert updated == 1
    docs = [doc for doc in ms._episode_corpus["documents"] if doc["doc_id"] == "document:DOC-PARTS-REPLACE"]
    assert len(docs) == 1
    episodes = docs[0]["episodes"]
    assert len(episodes) == 2
    assert any(ep["raw_text"].startswith("[Step 40] stable late step") for ep in episodes)
    assert any(ep["raw_text"].startswith("[Step 1] updated early step") for ep in episodes)
    assert not any(ep["raw_text"].startswith("[Step 1] initial early step") for ep in episodes)

    active_sessions = [rs for rs in ms._raw_sessions if rs.get("status") == "active"]
    superseded_sessions = [rs for rs in ms._raw_sessions if rs.get("status") == "superseded"]
    assert len(active_sessions) == 2
    assert len(superseded_sessions) == 1
    assert all(rs.get("part_source_id") for rs in ms._raw_sessions)
    assert "[Step 1] updated early step" in ms._raw_docs["DOC-PARTS-REPLACE"]
    assert "[Step 1] initial early step" not in ms._raw_docs["DOC-PARTS-REPLACE"]
