"""Unit 7: Three new inference prompts exist with correct placeholders."""

from pathlib import Path

PROMPTS_DIR = Path("src/prompts/inference")


def test_hybrid_prompt():
    p = PROMPTS_DIR / "hybrid.md"
    assert p.exists()
    text = p.read_text()
    assert "{context}" in text
    assert "{question}" in text
    assert "CONFLICT RESOLUTION" in text


def test_list_set_prompt():
    p = PROMPTS_DIR / "list_set.md"
    assert p.exists()
    text = p.read_text()
    assert "{context}" in text
    assert "{question}" in text
    assert "list or set of items" in text


def test_tool_prompt():
    p = PROMPTS_DIR / "tool.md"
    assert p.exists()
    text = p.read_text()
    assert "{context}" in text
    assert "{question}" in text
    assert "{sessions_in_context}" in text
    assert "{total_sessions}" in text
    assert "CONFLICT RESOLUTION" in text
    assert "get_more_context" in text


def test_summarize_with_metadata_prompt():
    p = PROMPTS_DIR / "summarize_with_metadata.md"
    assert p.exists()
    text = p.read_text()
    assert "{context}" in text
    assert "{question}" in text
    assert "{total_sessions}" in text
    assert "{sessions_in_context}" in text
    assert "{coverage_pct" in text
