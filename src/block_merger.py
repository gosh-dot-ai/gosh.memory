#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Block result merger.

Copyright 2026 (c) Mitja Goroshevsky and GOSH Technology Ltd.
License: MIT.

Merges per-block extraction results into a single session-level result
with sequential fact IDs and remapped temporal links.
"""

from .block_segmenter import Block


def merge_block_results(
    block_results: list[tuple[Block, dict]],
    session_num: int,
) -> dict:
    """Merge block extraction results into session-level output.

    Args:
        block_results: list of (Block, extraction_result) tuples,
            ordered by block.order.
        session_num: session number to set on all facts.

    Returns:
        dict with "facts" and "temporal_links" matching downstream schema.
    """
    # Sort by block order
    block_results = sorted(block_results, key=lambda br: br[0].order)

    all_facts = []
    all_tlinks = []
    fact_counter = 0

    for block, result in block_results:
        facts = result.get("facts", [])
        tlinks = result.get("temporal_links", [])

        # Map block-local IDs to final IDs
        local_to_final: dict[str, str] = {}

        for fact in facts:
            if not isinstance(fact, dict):
                continue
            fact_counter += 1
            final_id = f"f_{fact_counter:02d}"
            local_id = fact.get("local_id") or fact.get("id", "")
            if local_id:
                local_to_final[local_id] = final_id

            # Build merged fact
            merged = {
                "id": final_id,
                "session": session_num,
                "fact": fact.get("fact", ""),
                "kind": fact.get("kind", "fact"),
                "entities": fact.get("entities", []),
                "tags": fact.get("tags", []),
                "depends_on": fact.get("depends_on", []),
                "supersedes_topic": fact.get("supersedes_topic"),
                "confidence": fact.get("confidence"),
                "event_date": fact.get("event_date"),
            }

            # Inherit speaker from block if not set on fact
            if not fact.get("speaker") and block.speaker:
                merged["speaker"] = block.speaker
            elif fact.get("speaker"):
                merged["speaker"] = fact["speaker"]
            else:
                merged["speaker"] = None

            if not fact.get("speaker_role") and block.speaker_role:
                merged["speaker_role"] = block.speaker_role
            elif fact.get("speaker_role"):
                merged["speaker_role"] = fact["speaker_role"]
            else:
                merged["speaker_role"] = None

            # Phase 1B: stamp section metadata from document blocks
            if block.section_path:
                meta = merged.setdefault("metadata", {})
                meta["section_path"] = block.section_path

            # Phase 1B: version_status for superseding facts
            if merged.get("supersedes_topic"):
                meta = merged.setdefault("metadata", {})
                meta["version_status"] = "current"
                meta["version_supersedes"] = merged["supersedes_topic"]

            all_facts.append(merged)

        # Remap temporal links from block-local IDs to final IDs.
        # Preserve links even if one side can't be remapped (legacy compat).
        for tlink in tlinks:
            if not isinstance(tlink, dict):
                continue
            before_local = tlink.get("before", "")
            after_local = tlink.get("after", "")
            before_final = local_to_final.get(before_local, before_local)
            after_final = local_to_final.get(after_local, after_local)
            all_tlinks.append({
                "before": before_final,
                "after": after_final,
                "signal": tlink.get("signal", ""),
                "relation": tlink.get("relation", "before"),
            })

    return {"facts": all_facts, "temporal_links": all_tlinks}
