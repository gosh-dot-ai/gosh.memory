"""Unit 9: Tool-use inference — get_more_context tool."""

from src.inference import GET_CONTEXT_TOOL, execute_tool, get_inf_prompt, get_more_context


def test_get_more_context_valid():
    """Valid session_id → returns session text."""
    sessions = [{"content": "Hello world"}, {"content": "Goodbye"}]
    result = get_more_context(1, sessions)
    assert "Full text of Session 1" in result["result"]
    assert "Hello world" in result["result"]


def test_get_more_context_out_of_range():
    """Invalid session_id → not found."""
    sessions = [{"content": "only one"}]
    result = get_more_context(999, sessions)
    assert "not found" in result["result"]


def test_get_more_context_truncation():
    """Session text > 15000 chars → truncated."""
    sessions = [{"content": "A" * 20000}]
    result = get_more_context(1, sessions)
    assert "[...truncated]" in result["result"]
    # Content part should be at most 15000 A's
    assert result["result"].count("A") <= 15000


def test_get_more_context_zero():
    """session_id=0 → not found."""
    sessions = [{"content": "text"}]
    result = get_more_context(0, sessions)
    assert "not found" in result["result"]


def test_execute_tool_get_more_context():
    """execute_tool dispatches to get_more_context."""
    result = execute_tool("get_more_context", {"session_id": 1, "raw_sessions": [{"content": "hi"}]})
    assert "Session 1" in result


def test_existing_tools_still_work():
    """date_diff and count_items still in registry and work."""
    result = execute_tool("date_diff", {"date1": "2024-01-01", "date2": "2024-01-31", "unit": "days"})
    assert "30" in result

    result = execute_tool("count_items", {"items": ["a", "b", "c"]})
    assert "3" in result


def test_get_inf_prompt_new_types():
    """New prompt types (hybrid, tool, summarize_with_metadata) are loadable."""
    for name in ["hybrid", "tool", "summarize_with_metadata"]:
        p = get_inf_prompt(name)
        assert len(p) > 0, f"Prompt '{name}' is empty"


def test_get_context_tool_schema():
    """GET_CONTEXT_TOOL has correct structure."""
    assert GET_CONTEXT_TOOL["name"] == "get_more_context"
    assert "session_id" in GET_CONTEXT_TOOL["input_schema"]["properties"]
