"""Unit 6: Tool naming — underscore convention."""

import asyncio

import pytest

from src.mcp_server import mcp


def test_all_tools_use_underscore_naming():
    """All registered MCP tools must use underscore naming (no dots)."""
    tools = asyncio.run(mcp.list_tools())
    for t in tools:
        assert "." not in t.name, (
            f"Tool '{t.name}' uses dot notation — must use underscore"
        )


def test_required_tools_registered():
    """All required active MCP tools are registered."""
    tools = asyncio.run(mcp.list_tools())
    tool_names = {t.name for t in tools}
    required = {
        "memory_store", "memory_recall", "memory_ingest_document", "memory_ingest",
        "memory_build_index", "memory_flush", "memory_stats",
        "memory_reextract", "memory_list", "memory_get",
        "memory_import",
        "courier_subscribe", "courier_unsubscribe",
    }
    missing = required - tool_names
    assert not missing, f"Missing tools: {missing}"
