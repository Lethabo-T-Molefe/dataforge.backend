"""A fresh test database, migrated and seeded through the real pipeline, and an HTTP client for the API."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from dataforge_backend.core.database import SessionLocal, engine
from dataforge_backend.main import app
from dataforge_backend.seed import seed_demo

ROOT = Path(__file__).resolve().parents[2]


async def _reset_database() -> None:
    async with engine.begin() as conn:
        # DROP ... CASCADE removes the append-only audit table along with everything else in the schema.
        await conn.execute(text("DROP SCHEMA IF EXISTS app, bronze, silver, gold CASCADE"))
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "alembic"))
    # Alembic's env runs its own event loop, so it runs in a worker thread.
    await asyncio.to_thread(command.upgrade, config, "head")


@pytest.fixture(scope="session", autouse=True)
async def database() -> AsyncIterator[None]:
    await _reset_database()
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
