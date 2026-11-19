"""Development helpers shared by the integration tests and the evaluation commands.

Resetting drops every schema, so it is only allowed on a database whose name marks it as disposable
(it contains "test" or "eval"). The development and hosted databases are never reset by these helpers.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import asyncpg
from sqlalchemy import text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncEngine

REPO = Path(__file__).resolve().parents[2]
_DISPOSABLE_MARKERS = ("test", "eval")


def is_disposable(url: URL) -> bool:
    name = (url.database or "").lower()
    return any(marker in name for marker in _DISPOSABLE_MARKERS)


def alembic_upgrade(revision: str = "head") -> None:
    from alembic import command
    from alembic.config import Config

    config = Config(str(REPO / "alembic.ini"))
    config.set_main_option("script_location", str(REPO / "alembic"))
    command.upgrade(config, revision)


async def ensure_database(engine: AsyncEngine) -> None:
    """Create the database on a server that does not have it yet (the first run on a new machine)."""
    url = engine.url
    admin = await asyncpg.connect(
        host=url.host, port=url.port, user=url.username, password=url.password, database="postgres"
    )
    try:
        exists = await admin.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", url.database)
        if not exists:
            # The name comes from this project's own settings, and is quoted as an identifier.
            name = str(url.database).replace('"', '""')
            await admin.execute(f'CREATE DATABASE "{name}"')
    finally:
        await admin.close()


async def reset_database(engine: AsyncEngine) -> None:
    """Drop and rebuild every schema, then migrate to head. Refuses a database that is not disposable."""
    if not is_disposable(engine.url):
        raise RuntimeError(
            f"Refusing to reset {engine.url.database!r}: only a database whose name contains "
            "'test' or 'eval' is reset."
        )
    await ensure_database(engine)
    async with engine.begin() as conn:
        # DROP ... CASCADE removes the append-only audit table along with everything else in the schema.
        await conn.execute(text("DROP SCHEMA IF EXISTS app, bronze, silver, gold CASCADE"))
    # Alembic's env runs its own event loop, so it runs in a worker thread.
    await asyncio.to_thread(alembic_upgrade, "head")
