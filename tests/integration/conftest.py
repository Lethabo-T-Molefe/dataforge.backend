"""A fresh test database, migrated and seeded through the real pipeline, and an HTTP client for the API."""

from collections.abc import AsyncIterator, Awaitable, Callable

import pytest
from httpx import ASGITransport, AsyncClient

from dataforge_backend.core.database import SessionLocal, engine
from dataforge_backend.devtools import reset_database
from dataforge_backend.main import app
from dataforge_backend.seed import seed_demo


@pytest.fixture(scope="session", autouse=True)
async def database() -> AsyncIterator[None]:
    await reset_database(engine)
    async with SessionLocal() as session:
        await seed_demo(session)
    yield
    await engine.dispose()


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        yield http


SignIn = Callable[[str], Awaitable[dict[str, str]]]


@pytest.fixture
def sign_in(client: AsyncClient) -> SignIn:
    """Headers for the seeded demo member of a role."""

    async def _sign_in(role: str) -> dict[str, str]:
        response = await client.post("/api/v1/auth/demo", json={"role": role})
        assert response.status_code == 200, response.text
        return {"Authorization": f"Bearer {response.json()['accessToken']}"}

    return _sign_in
