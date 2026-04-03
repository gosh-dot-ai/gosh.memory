"""Tests for /health endpoint."""
import pytest


@pytest.mark.asyncio
async def test_health_endpoint():
    """GET /health must return 200 with {"status": "ok"}."""
    import tempfile

    from httpx import ASGITransport, AsyncClient
    with tempfile.TemporaryDirectory() as tmp:
        from src.mcp_server import create_app
        app = create_app(app_data_dir=tmp)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
