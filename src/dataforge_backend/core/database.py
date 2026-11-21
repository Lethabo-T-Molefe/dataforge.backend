import time
from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy import MetaData, event, exc
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from .config import settings

# Application metadata lives in the "app" schema. Data itself lives in bronze, silver and gold.
APP_SCHEMA = "app"

_NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
)

# A pooled connection that sat idle may have been closed by the database or a pooler in between, so it is
# checked before use. Only an idle one: the check (pool_pre_ping) is three round trips, about half a second
# per request when the database is on another continent, and a connection in steady use is known to be alive.
PING_AFTER_IDLE_SECONDS = 60.0


@event.listens_for(engine.sync_engine, "checkin")
def _mark_idle(dbapi_connection: Any, record: Any) -> None:
    record.info["idle_since"] = time.monotonic()


@event.listens_for(engine.sync_engine, "checkout")
def _ping_if_idle(dbapi_connection: Any, record: Any, proxy: Any) -> None:
    idle_since = record.info.get("idle_since")
    if idle_since is None or time.monotonic() - idle_since < PING_AFTER_IDLE_SECONDS:
        return
    try:
        dbapi_connection.ping()
    except Exception as error:
        # The pool discards this connection and checks out a fresh one.
        raise exc.DisconnectionError() from error


SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    metadata = MetaData(schema=APP_SCHEMA, naming_convention=_NAMING_CONVENTION)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency that yields a database session per request."""
    async with SessionLocal() as session:
        yield session
