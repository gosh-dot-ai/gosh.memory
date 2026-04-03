"""Tests for block-based episode extraction helpers."""

from src.episode_extraction import reconstruct_episodes
from src.source_adapters import audit_episode_partition


def _make_block_dicts():
    return [
        {
            "block_id": "DOC-9_b000",
            "order": 0,
            "family": "heading",
            "section_path": "Overview",
            "text": "Overview heading",
            "text_preview": "Overview heading",
            "char_len": len("Overview heading"),
            "raw_span": [0, 16],
        },
        {
            "block_id": "DOC-9_b001",
            "order": 1,
            "family": "paragraph",
            "section_path": "Overview",
            "text": "Route 1 final approved length is 14.3 km.",
            "text_preview": "Route 1 final approved length is 14.3 km.",
            "char_len": len("Route 1 final approved length is 14.3 km."),
            "raw_span": [17, 59],
        },
        {
            "block_id": "DOC-9_b002",
            "order": 2,
            "family": "paragraph",
            "section_path": "Appendix",
            "text": "Footer boilerplate.",
            "text_preview": "Footer boilerplate.",
            "char_len": len("Footer boilerplate."),
            "raw_span": [60, 79],
        },
    ]


def test_reconstruct_episodes_preserves_raw_blocks_and_fills_missing_singletons():
    block_dicts = _make_block_dicts()
    llm_episodes = [
        {
            "topic_key": "route 1 canonical length",
            "state_label": "canonical",
            "currentness": "current",
            "block_ids": ["DOC-9_b000", "DOC-9_b001"],
        }
    ]

    episodes = reconstruct_episodes(
        "DOC-9",
        "2026-01-01",
        block_dicts,
        llm_episodes,
        {"size_cap_chars": 1000, "attach_missing": False, "singleton_missing": True},
    )

    assert len(episodes) == 2
    assert episodes[0]["provenance"]["block_ids"] == ["DOC-9_b000", "DOC-9_b001"]
    assert episodes[0]["raw_text"] == "Overview heading\n\nRoute 1 final approved length is 14.3 km."
    assert episodes[1]["provenance"]["block_ids"] == ["DOC-9_b002"]
    assert episodes[1]["state_label"] == "structural_context"


def test_audit_episode_partition_reports_missing_and_duplicate_blocks():
    block_manifest = {"DOC-9": _make_block_dicts()}
    corpus = {
        "documents": [
            {
                "doc_id": "DOC-9",
                "episodes": [
                    {
                        "episode_id": "DOC-9_e01",
                        "source_type": "document",
                        "source_id": "DOC-9",
                        "topic_key": "route 1 canonical length",
                        "state_label": "canonical",
                        "currentness": "current",
                        "raw_text": "Overview heading\n\nRoute 1 final approved length is 14.3 km.",
                        "provenance": {"block_ids": ["DOC-9_b000", "DOC-9_b001"]},
                    },
                    {
                        "episode_id": "DOC-9_e02",
                        "source_type": "document",
                        "source_id": "DOC-9",
                        "topic_key": "duplicate block",
                        "state_label": "duplicate",
                        "currentness": "unknown",
                        "raw_text": "Route 1 final approved length is 14.3 km.",
                        "provenance": {"block_ids": ["DOC-9_b001"]},
                    },
                ],
            }
        ]
    }

    audit = audit_episode_partition(corpus, block_manifest)

    assert audit["total_missing_blocks"] == 1
    assert audit["total_duplicate_blocks"] == 1
    assert audit["per_doc"]["DOC-9"]["missing_blocks"] == ["DOC-9_b002"]
    assert audit["per_doc"]["DOC-9"]["duplicate_blocks"] == ["DOC-9_b001"]
