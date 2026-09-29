from httpx import ASGITransport, AsyncClient

from dataforge_backend.main import app


async def test_root_returns_welcome_message() -> None:
    # ASGITransport does not run the lifespan, so no database is needed here.
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to DataForge.Backend"}
