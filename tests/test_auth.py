"""Tests for C1: Token-based authentication for MCP server."""

import pytest


@pytest.mark.asyncio
async def test_unauthorized_without_token():
    """Routes other than /health must return 401 without valid token."""
    import tempfile

    from httpx import ASGITransport, AsyncClient
    with tempfile.TemporaryDirectory() as tmp:
        from src.mcp_server import SERVER_TOKEN, create_app
        app = create_app(app_data_dir=tmp)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            # /health should work without token
            r = await c.get("/health")
            assert r.status_code == 200

            # /admin/reload should require token
            r = await c.post("/admin/reload", json={"key": "x"})
            assert r.status_code == 401

            # /mcp/sse should require token
            # (SSE endpoint returns streaming, but 401 comes first)
            r = await c.get("/mcp/sse")
            assert r.status_code == 401


@pytest.mark.asyncio
async def test_authorized_with_valid_token():
    """Routes accept requests with valid x-server-token header."""
    import tempfile

    from httpx import ASGITransport, AsyncClient
    with tempfile.TemporaryDirectory() as tmp:
        from src.mcp_server import SERVER_TOKEN, create_app
        app = create_app(app_data_dir=tmp)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            headers = {"x-server-token": SERVER_TOKEN}

            # /admin/reload with valid token should succeed
            r = await c.post("/admin/reload", json={"key": "nonexistent"},
                             headers=headers)
            assert r.status_code == 200
            assert r.json()["status"] == "reloaded"


@pytest.mark.asyncio
async def test_wrong_token_rejected():
    """Wrong token should get 401."""
    import tempfile

    from httpx import ASGITransport, AsyncClient
    with tempfile.TemporaryDirectory() as tmp:
        from src.mcp_server import create_app
        app = create_app(app_data_dir=tmp)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            headers = {"x-server-token": "wrong-token-value"}
            r = await c.post("/admin/reload", json={"key": "x"},
                             headers=headers)
            assert r.status_code == 401
