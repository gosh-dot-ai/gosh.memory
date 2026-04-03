"""Tests for document segmentation — Sprint 36 Stage 1."""

from src.block_segmenter import Block, segment_conversation_blocks, segment_document_blocks

# ═══════════════════════════════════════════════════════════════════
# Heading hierarchy
# ═══════════════════════════════════════════════════════════════════


def test_heading_hierarchy():
    text = "# Chapter 1\n\nIntro text.\n\n## Section A\n\nContent A.\n\n## Section B\n\nContent B."
    blocks = segment_document_blocks(text)
    paths = [b.section_path for b in blocks]
    assert "Chapter 1" in paths
    assert "Chapter 1 > Section A" in paths
    assert "Chapter 1 > Section B" in paths


def test_nested_headings():
    text = "# Top\n\n## Mid\n\n### Deep\n\nDeep content."
    blocks = segment_document_blocks(text)
    deep = [b for b in blocks if b.section_path and "Deep" in b.section_path]
    assert len(deep) >= 1
    assert deep[0].section_path == "Top > Mid > Deep"


def test_heading_lines_not_in_block_text():
    text = "# Overview\n\nThis is the overview."
    blocks = segment_document_blocks(text)
    for b in blocks:
        assert not b.text.startswith("# ")


# ═══════════════════════════════════════════════════════════════════
# List handling with lead_in
# ═══════════════════════════════════════════════════════════════════


def test_list_in_document():
    text = "# Requirements\n\nThe following are needed:\n1. Item one\n2. Item two\n3. Item three"
    blocks = segment_document_blocks(text)
    list_blocks = [b for b in blocks if b.family == "LIST"]
    assert len(list_blocks) == 1
    assert list_blocks[0].lead_in is not None
    assert "needed" in list_blocks[0].lead_in.lower()


def test_single_list_item_stays_prose():
    text = "# Notes\n\n- Single item only"
    blocks = segment_document_blocks(text)
    assert not any(b.family == "LIST" for b in blocks)


# ═══════════════════════════════════════════════════════════════════
# Table
# ═══════════════════════════════════════════════════════════════════


def test_table_in_document():
    text = "# Schedule\n\n| Day | Shift | Agent |\n| Mon | Morning | Alice |\n| Tue | Evening | Bob |"
    blocks = segment_document_blocks(text)
    table_blocks = [b for b in blocks if b.family == "TABLE"]
    assert len(table_blocks) == 1
    assert "Alice" in table_blocks[0].text


# ═══════════════════════════════════════════════════════════════════
# KV detection
# ═══════════════════════════════════════════════════════════════════


def test_kv_in_document():
    text = "# Specifications\n\nMaterial: Steel\nPressure: 16 bar\nDiameter: 1200 mm"
    blocks = segment_document_blocks(text)
    kv_blocks = [b for b in blocks if b.family == "KV"]
    assert len(kv_blocks) == 1
    assert "Steel" in kv_blocks[0].text
    assert "16 bar" in kv_blocks[0].text


def test_single_kv_stays_prose():
    text = "# Info\n\nMaterial: Steel\n\nSome other text."
    blocks = segment_document_blocks(text)
    assert not any(b.family == "KV" for b in blocks)


# ═══════════════════════════════════════════════════════════════════
# CODE (fenced blocks)
# ═══════════════════════════════════════════════════════════════════


def test_code_block_in_document():
    text = "# Example\n\nHere is code:\n```python\ndef hello():\n    print('hi')\n```\n\nEnd."
    blocks = segment_document_blocks(text)
    code_blocks = [b for b in blocks if b.family == "CODE"]
    assert len(code_blocks) == 1
    assert "def hello" in code_blocks[0].text


# ═══════════════════════════════════════════════════════════════════
# Long prose sub-segmentation
# ═══════════════════════════════════════════════════════════════════


def test_long_prose_sub_segmented():
    long_text = "# Chapter\n\n" + " ".join(
        [f"This is sentence number {i} about the project." for i in range(80)])
    blocks = segment_document_blocks(long_text)
    prose_blocks = [b for b in blocks if b.family == "PROSE"]
    assert len(prose_blocks) >= 2


# ═══════════════════════════════════════════════════════════════════
# Mixed document
# ═══════════════════════════════════════════════════════════════════


def test_mixed_document():
    text = (
        "# Project Overview\n\n"
        "This project builds a water supply system.\n\n"
        "## Requirements\n\n"
        "Key requirements:\n"
        "1. Capacity: 500 L/s\n"
        "2. Pressure: 16 bar\n"
        "3. Material: DI pipes\n\n"
        "## Specifications\n\n"
        "| Component | Value |\n"
        "| Pipe | DN1200 |\n"
        "| Valve | PN16 |\n\n"
        "## Properties\n\n"
        "Material: Ductile Iron\n"
        "Grade: K9\n"
        "Coating: Zinc + epoxy\n"
    )
    blocks = segment_document_blocks(text)
    families = {b.family for b in blocks}
    assert "PROSE" in families
    assert "LIST" in families
    assert "TABLE" in families
    assert "KV" in families


# ═══════════════════════════════════════════════════════════════════
# Raw span contract
# ═══════════════════════════════════════════════════════════════════


def test_raw_span_contract():
    text = "# Title\n\nSome content.\n\n## Sub\n\n1. Item A\n2. Item B"
    blocks = segment_document_blocks(text)
    for b in blocks:
        start, end = b.raw_span
        assert 0 <= start <= end <= len(text), f"Invalid span {b.raw_span}"
        span_text = text[start:end]
        if b.text.strip():
            assert b.text.strip() in span_text, \
                f"Block text '{b.text[:40]}' not in span '{span_text[:40]}'"


# ═══════════════════════════════════════════════════════════════════
# Conversation backward compatibility
# ═══════════════════════════════════════════════════════════════════


def test_conversation_backward_compat():
    """Conversation blocks still work with section_path=None."""
    text = "user: Hello.\nassistant: Hi there."
    blocks = segment_conversation_blocks(text)
    for b in blocks:
        assert b.section_path is None
        assert b.speaker is not None


# ═══════════════════════════════════════════════════════════════════
# Document blocks have speaker=None
# ═══════════════════════════════════════════════════════════════════


def test_document_blocks_no_speaker():
    text = "# Title\n\nContent here."
    blocks = segment_document_blocks(text)
    for b in blocks:
        assert b.speaker is None
        assert b.speaker_role is None


# ═══════════════════════════════════════════════════════════════════
# Karnali snippet smoke check
# ═══════════════════════════════════════════════════════════════════


def test_karnali_snippet():
    """Smoke test with a Karnali-like document structure."""
    text = (
        "# Karnali River Basin Water Supply\n\n"
        "## 1. Project Background\n\n"
        "The Karnali River Basin Water Supply Project aims to provide\n"
        "reliable drinking water to communities in far-western Nepal.\n"
        "The project was initiated in 2024.\n\n"
        "## 2. Technical Specifications\n\n"
        "Pipe material: Ductile Iron (DI)\n"
        "Nominal diameter: DN1200\n"
        "Pressure class: PN16\n"
        "Coating: Zinc + bituminous\n\n"
        "## 3. Key Components\n\n"
        "The system includes:\n"
        "1. Intake structure at river\n"
        "2. Treatment plant (500 L/s capacity)\n"
        "3. Transmission pipeline (12.4 km)\n"
        "4. Distribution network\n\n"
        "## 4. Cost Summary\n\n"
        "| Item | Cost (TKT) |\n"
        "| Pipes | 1,850,000 |\n"
        "| Treatment | 2,300,000 |\n"
        "| Labor | 1,200,000 |\n"
    )
    blocks = segment_document_blocks(text)
    families = {b.family for b in blocks}
    assert "PROSE" in families
    assert "KV" in families
    assert "LIST" in families
    assert "TABLE" in families

    # Check section paths
    paths = {b.section_path for b in blocks if b.section_path}
    assert any("Technical Specifications" in p for p in paths)
    assert any("Key Components" in p for p in paths)
    assert any("Cost Summary" in p for p in paths)

    # KV block should be in Technical Specifications section
    kv = [b for b in blocks if b.family == "KV"]
    assert len(kv) >= 1
    assert "Technical Specifications" in (kv[0].section_path or "")


# ═══════════════════════════════════════════════════════════════════
# No headings → single section
# ═══════════════════════════════════════════════════════════════════


def test_no_headings():
    text = "Just plain text without headings.\n\nAnother paragraph."
    blocks = segment_document_blocks(text)
    assert len(blocks) >= 1
    assert all(b.section_path is None for b in blocks)


# ═══════════════════════════════════════════════════════════════════
# Content before first heading
# ═══════════════════════════════════════════════════════════════════


def test_content_before_first_heading():
    text = "Preamble text.\n\n# Chapter 1\n\nChapter content."
    blocks = segment_document_blocks(text)
    preamble = [b for b in blocks if b.section_path is None]
    chapter = [b for b in blocks if b.section_path == "Chapter 1"]
    assert len(preamble) >= 1
    assert len(chapter) >= 1
    assert "Preamble" in preamble[0].text


# ═══════════════════════════════════════════════════════════════════
# Order is sequential
# ═══════════════════════════════════════════════════════════════════


def test_order_sequential():
    text = "# A\n\nText A.\n\n# B\n\nText B.\n\n# C\n\nText C."
    blocks = segment_document_blocks(text)
    orders = [b.order for b in blocks]
    assert orders == sorted(orders)
    assert len(set(orders)) == len(orders)
