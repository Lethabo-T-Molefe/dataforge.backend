"""Durable per-key rate limits for authentication endpoints.

SlowAPI's in-memory counters race under concurrent requests and reset when a replica restarts.
These buckets live in Postgres and are taken with a row lock, so the documented limits hold.
"""

from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.errors import RateLimited
from ..core.security import utcnow
from ..infrastructure.db.models import RateBucket

_MESSAGE = "Too many requests. Wait a moment and try again."


async def consume(session: AsyncSession, key: str, *, limit: int, window_seconds: int) -> None:
    """Count one hit against `key`. Raises RateLimited when the window is already full.

    Commits immediately so the counter survives when the rest of the request rolls back
    (wrong password, validation error, etc.).
    """
    now = utcnow()
    window = timedelta(seconds=window_seconds)
    for _ in range(3):
        row = (
            await session.execute(select(RateBucket).where(RateBucket.key == key).with_for_update())
        ).scalar_one_or_none()
        if row is None:
            session.add(RateBucket(key=key, window_start=now, count=1))
            try:
                await session.commit()
                return
            except IntegrityError:
                await session.rollback()
                continue
        if now - row.window_start >= window:
            row.window_start = now
            row.count = 1
            await session.commit()
            return
        if row.count >= limit:
            raise RateLimited(_MESSAGE)
        row.count += 1
        await session.commit()
        return
    raise RateLimited(_MESSAGE)
