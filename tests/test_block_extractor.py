"""Tests for block_extractor and block_merger — Stage 2 of ContentIR Phase 1."""

import asyncio
import json
from pathlib import Path

from src.block_extractor import _FAMILY_PROMPT, _load_prompt, extract_block
from src.block_merger import merge_block_results
from src.block_segmenter import Block

# ═══════════════════════════════════════════════════════════════════
# Prompt routing
# ═══════════════════════════════════════════════════════════════════


def test_prompt_routing_prose():
    assert _FAMILY_PROMPT["PROSE"] == "prose_block.md"
    text = _load_prompt("prose_block.md")
    assert "local_id" in text
    assert "temporal_links" in text


def test_prompt_routing_list():
    assert _FAMILY_PROMPT["LIST"] == "list_block.md"
    text = _load_prompt("list_block.md")
    assert "local_id" in text
    assert "temporal_links" in text


def test_prompt_routing_table():
    assert _FAMILY_PROMPT["TABLE"] == "table_block.md"
    text = _load_prompt("table_block.md")
    assert "local_id" in text
    assert "temporal_links" in text


def test_prompt_routing_unknown():
    assert _FAMILY_PROMPT["UNKNOWN"] == "fallback_block.md"
    text = _load_prompt("fallback_block.md")
    assert "local_id" in text
    assert "temporal_links" in text


def test_prompt_routing_kv_uses_prose():
    assert _FAMILY_PROMPT["KV"] == "prose_block.md"


def test_prompt_routing_code_uses_fallback():
    assert _FAMILY_PROMPT["CODE"] == "fallback_block.md"


# ═══════════════════════════════════════════════════════════════════
# Prompt output contract sanity
# ═══════════════════════════════════════════════════════════════════


def test_all_prompts_contain_local_id_and_temporal_links():
    prompt_dir = Path("src/prompts/extraction")
    for name in ["prose_block.md", "list_block.md", "table_block.md", "fallback_block.md"]:
        text = (prompt_dir / name).read_text()
        assert "local_id" in text, f"{name} missing local_id"
        assert "temporal_links" in text, f"{name} missing temporal_links"


# ═══════════════════════════════════════════════════════════════════
# Merger — sequential IDs
# ═══════════════════════════════════════════════════════════════════


def test_merge_sequential_ids():
    """3 blocks with 2, 3, 2 facts → 7 merged facts with f_01..f_07."""
    b1 = Block("PROSE", "text1", 0, (0, 5), "user", "user", None)
    b2 = Block("LIST", "text2", 1, (5, 10), "user", "user", None)
    b3 = Block("PROSE", "text3", 2, (10, 15), "assistant", "assistant", None)

    r1 = {"facts": [
        {"local_id": "b1", "fact": "fact A", "kind": "fact"},
        {"local_id": "b2", "fact": "fact B", "kind": "fact"},
    ], "temporal_links": []}
    r2 = {"facts": [
        {"local_id": "b1", "fact": "item 1", "kind": "count_item"},
        {"local_id": "b2", "fact": "item 2", "kind": "count_item"},
        {"local_id": "b3", "fact": "item 3", "kind": "count_item"},
    ], "temporal_links": []}
    r3 = {"facts": [
        {"local_id": "b1", "fact": "response A", "kind": "fact"},
        {"local_id": "b2", "fact": "response B", "kind": "fact"},
    ], "temporal_links": []}

    merged = merge_block_results([(b1, r1), (b2, r2), (b3, r3)], session_num=5)
    assert len(merged["facts"]) == 7
    ids = [f["id"] for f in merged["facts"]]
    assert ids == ["f_01", "f_02", "f_03", "f_04", "f_05", "f_06", "f_07"]
    for f in merged["facts"]:
        assert f["session"] == 5


# ═══════════════════════════════════════════════════════════════════
# Merger — speaker inheritance
# ═══════════════════════════════════════════════════════════════════


def test_merge_speaker_inheritance():
    """Fact without speaker inherits from block."""
    block = Block("PROSE", "text", 0, (0, 4), "user", "user", None)
    result = {"facts": [
        {"local_id": "b1", "fact": "some fact", "kind": "fact"},
        {"local_id": "b2", "fact": "another", "kind": "fact", "speaker": "assistant"},
    ], "temporal_links": []}

    merged = merge_block_results([(block, result)], session_num=1)
    assert merged["facts"][0]["speaker"] == "user"
    assert merged["facts"][0]["speaker_role"] == "user"
    assert merged["facts"][1]["speaker"] == "assistant"


# ═══════════════════════════════════════════════════════════════════
# Merger — temporal link remap
# ═══════════════════════════════════════════════════════════════════


def test_merge_temporal_remap():
    """Block-local temporal links are remapped to final IDs."""
    block = Block("PROSE", "text", 0, (0, 4), "user", "user", None)
    result = {"facts": [
        {"local_id": "b1", "fact": "first event", "kind": "fact"},
        {"local_id": "b2", "fact": "second event", "kind": "fact"},
    ], "temporal_links": [
        {"before": "b1", "after": "b2", "signal": "then", "relation": "before"},
    ]}

    merged = merge_block_results([(block, result)], session_num=1)
    assert len(merged["temporal_links"]) == 1
    tl = merged["temporal_links"][0]
    assert tl["before"] == "f_01"
    assert tl["after"] == "f_02"
    assert tl["signal"] == "then"


def test_merge_temporal_unresolved_preserved():
    """Temporal links with unresolvable IDs are preserved (legacy compat)."""
    block = Block("PROSE", "text", 0, (0, 4), "user", "user", None)
    result = {"facts": [
        {"local_id": "b1", "fact": "only fact", "kind": "fact"},
    ], "temporal_links": [
        {"before": "b1", "after": "nonexistent", "signal": "after"},
    ]}

    merged = merge_block_results([(block, result)], session_num=1)
    # Link preserved even with unresolvable "after" — legacy compatibility
    assert len(merged["temporal_links"]) == 1
    assert merged["temporal_links"][0]["before"] == "f_01"
    assert merged["temporal_links"][0]["after"] == "nonexistent"


# ═══════════════════════════════════════════════════════════════════
# Extractor — model pass-through
# ═══════════════════════════════════════════════════════════════════


def test_extractor_model_passthrough():
    """Model string passed to extract_block reaches call_extract_fn."""
    captured = {}

    async def mock_extract(model, system, user_msg, max_tokens=4096):
        captured["model"] = model
        captured["system"] = system
        return {"facts": [{"local_id": "b1", "fact": "test", "kind": "fact"}],
                "temporal_links": []}

    block = Block("PROSE", "Hello world", 0, (0, 11), "user", "user", None)
    metadata = {"session_date": "2024-01-01", "session_num": 1, "container_kind": "conversation"}

    asyncio.run(extract_block(block, metadata, model="test-model-v1",
                              call_extract_fn=mock_extract))
    assert captured["model"] == "test-model-v1"


def test_extractor_container_kind_from_metadata():
    """container_kind in prompt comes from session_metadata, not block.family."""
    captured = {}

    async def mock_extract(model, system, user_msg, max_tokens=4096):
        captured["system"] = system
        return {"facts": [], "temporal_links": []}

    block = Block("LIST", "1. item", 0, (0, 7), "user", "user", None)
    metadata = {"session_date": "2024-01-01", "session_num": 1, "container_kind": "conversation"}

    asyncio.run(extract_block(block, metadata, model="test",
                              call_extract_fn=mock_extract))
    assert "Container: conversation" in captured["system"]
    assert "Container: LIST" not in captured["system"]


# ═══════════════════════════════════════════════════════════════════
# Extractor — retry policy
# ═══════════════════════════════════════════════════════════════════


def test_extractor_retry_on_parse_failure():
    """Invalid JSON → retry same prompt → fallback prompt → empty result."""
    call_count = {"n": 0}

    async def mock_extract(model, system, user_msg, max_tokens=4096):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            return "not valid json at all"
        if call_count["n"] == 3:
            return "still broken {"
        return {"facts": [], "temporal_links": []}

    block = Block("PROSE", "garbage", 0, (0, 7), None, None, None)
    metadata = {"session_date": "2024-01-01", "session_num": 1}

    result = asyncio.run(extract_block(block, metadata, model="test",
                                       call_extract_fn=mock_extract))
    assert result["facts"] == []
    assert call_count["n"] == 3  # 1 normal + 1 retry + 1 fallback


def test_extractor_succeeds_on_retry():
    """First attempt fails, retry succeeds."""
    call_count = {"n": 0}

    async def mock_extract(model, system, user_msg, max_tokens=4096):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "broken"
        return json.dumps({"facts": [{"local_id": "b1", "fact": "recovered", "kind": "fact"}],
                           "temporal_links": []})

    block = Block("PROSE", "text", 0, (0, 4), None, None, None)
    metadata = {"session_date": "2024-01-01", "session_num": 1}

    result = asyncio.run(extract_block(block, metadata, model="test",
                                       call_extract_fn=mock_extract))
    assert len(result["facts"]) == 1
    assert result["facts"][0]["fact"] == "recovered"


# ═══════════════════════════════════════════════════════════════════
# Extractor — conservative fallback
# ═══════════════════════════════════════════════════════════════════


def test_unknown_block_garbage_returns_empty():
    """UNKNOWN block with garbage text can return empty facts."""
    async def mock_extract(model, system, user_msg, max_tokens=4096):
        return json.dumps({"facts": [], "temporal_links": []})

    block = Block("UNKNOWN", "asdfghjkl 12345 !!!", 0, (0, 20), None, None, None)
    metadata = {"session_date": "2024-01-01", "session_num": 1}

    result = asyncio.run(extract_block(block, metadata, model="test",
                                       call_extract_fn=mock_extract))
    assert result["facts"] == []
    assert result["temporal_links"] == []


# ═══════════════════════════════════════════════════════════════════
# Extractor — prompt selection by family
# ═══════════════════════════════════════════════════════════════════


def test_extractor_uses_correct_prompt_for_family():
    """Verify that different families load different prompts."""
    captured_systems = []

    async def mock_extract(model, system, user_msg, max_tokens=4096):
        captured_systems.append(system)
        return {"facts": [], "temporal_links": []}

    metadata = {"session_date": "2024-01-01", "session_num": 1}

    for family, expected_file in [("PROSE", "prose_block.md"),
                                   ("LIST", "list_block.md"),
                                   ("TABLE", "table_block.md"),
                                   ("UNKNOWN", "fallback_block.md")]:
        captured_systems.clear()
        block = Block(family, "test content", 0, (0, 12), "user", "user", None)
        asyncio.run(extract_block(block, metadata, model="test",
                                  call_extract_fn=mock_extract))
        expected_text = _load_prompt(expected_file)
        # System prompt should start with the same first line as the template
        first_line = expected_text.split("\n")[0]
        assert captured_systems[0].startswith(first_line), \
            f"{family} → wrong prompt. Got: {captured_systems[0][:80]}"


# ═══════════════════════════════════════════════════════════════════
# Merger — empty blocks
# ═══════════════════════════════════════════════════════════════════


def test_merge_empty_blocks():
    """Empty block results produce empty merged output."""
    block = Block("PROSE", "", 0, (0, 0), None, None, None)
    result = {"facts": [], "temporal_links": []}
    merged = merge_block_results([(block, result)], session_num=1)
    assert merged["facts"] == []
    assert merged["temporal_links"] == []


def test_merge_preserves_block_order():
    """Blocks are merged in order even if passed out of order."""
    b1 = Block("PROSE", "a", 2, (0, 1), None, None, None)
    b2 = Block("PROSE", "b", 0, (1, 2), None, None, None)
    r1 = {"facts": [{"local_id": "x", "fact": "second", "kind": "fact"}], "temporal_links": []}
    r2 = {"facts": [{"local_id": "y", "fact": "first", "kind": "fact"}], "temporal_links": []}
    merged = merge_block_results([(b1, r1), (b2, r2)], session_num=1)
    assert merged["facts"][0]["fact"] == "first"
    assert merged["facts"][1]["fact"] == "second"
