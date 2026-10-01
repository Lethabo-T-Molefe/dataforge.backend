"""Fair-use quotas: sandbox compute, storage and AI queries per member per month."""

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.errors import QuotaExceeded
from ..core.security import utcnow
from ..domain.roles import DEFAULT_QUOTAS, QuotaLimits, Role
from ..infrastructure.db.models import Dataset, Usage, User
from .common import Actor

_GB = 1024**3


def period() -> str:
    return utcnow().strftime("%Y-%m")


def limits_for(user: User) -> QuotaLimits:
    base = DEFAULT_QUOTAS[Role(user.role)]
    # Only numbers override a role's limit; an absent or null entry means the role's default applies.
    overrides = {k: v for k, v in (user.quota_overrides or {}).items() if v is not None}
    return QuotaLimits(
        compute_seconds=overrides.get("computeSeconds", base.compute_seconds),
        storage_bytes=overrides.get("storageBytes", base.storage_bytes),
        ai_queries=overrides.get("aiQueries", base.ai_queries),
    )


async def usage_row(session: AsyncSession, user_id: str) -> Usage:
    row = (
        await session.execute(select(Usage).where(Usage.user_id == user_id, Usage.period == period()))
    ).scalar_one_or_none()
    if row is None:
        row = Usage(user_id=user_id, period=period(), compute_seconds=0.0, ai_queries=0)
        session.add(row)
        await session.flush()
    return row


async def storage_used(session: AsyncSession, user_id: str) -> int:
    total = await session.execute(
        select(func.coalesce(func.sum(Dataset.size_bytes), 0)).where(
            Dataset.owner_id == user_id, Dataset.erased_at.is_(None)
        )
    )
    return int(total.scalar() or 0)


async def check_ai(session: AsyncSession, actor: Actor) -> None:
    user = await session.get(User, actor.id)
    if user is None:
        return
    limit = limits_for(user).ai_queries
    if limit is None:
        return
    used = (await usage_row(session, actor.id)).ai_queries
    if used >= limit:
        raise QuotaExceeded(
            f"You have used all {limit} assistant questions for this month. Ask an administrator to raise your quota."
        )


async def check_compute(session: AsyncSession, actor: Actor) -> None:
    user = await session.get(User, actor.id)
    if user is None:
        return
    limit = limits_for(user).compute_seconds
    if limit is None:
        return
    used = (await usage_row(session, actor.id)).compute_seconds
    if used >= limit:
        raise QuotaExceeded(f"You have used your {limit // 3600} sandbox hours for this month.")


async def check_storage(session: AsyncSession, actor: Actor, incoming: int) -> None:
    user = await session.get(User, actor.id)
    if user is None:
        return
    limit = limits_for(user).storage_bytes
    if limit is None:
        return
    if await storage_used(session, actor.id) + incoming > limit:
        raise QuotaExceeded(f"This upload would take you past your {limit / _GB:.0f} GB storage quota.")


async def add_compute(session: AsyncSession, user_id: str | None, milliseconds: int) -> None:
    if not user_id or user_id == "system":
        return
    row = await usage_row(session, user_id)
    row.compute_seconds = float(row.compute_seconds) + milliseconds / 1000


async def add_ai_query(session: AsyncSession, user_id: str) -> None:
    row = await usage_row(session, user_id)
    row.ai_queries = int(row.ai_queries) + 1


def _hours(seconds: float) -> str:
    hours = seconds / 3600
    return f"{hours:.0f}" if hours >= 10 or hours == int(hours) else f"{hours:.1f}"


def _gb(value: int) -> str:
    gb = value / _GB
    return f"{gb:.0f}" if gb >= 10 or gb == int(gb) else f"{gb:.1f}"


async def summary(session: AsyncSession, user: User) -> dict[str, Any]:
    """Quotas worded the way the frontend shows them, plus the raw numbers."""
    limits = limits_for(user)
    usage = await usage_row(session, user.id)
    storage = await storage_used(session, user.id)
    compute, ai = float(usage.compute_seconds), int(usage.ai_queries)
    if limits.compute_seconds is None:
        words = {"compute": "unlimited", "storage": "unlimited", "ai": "unlimited"}
        short = dict(words)
    else:
        words = {
            "compute": f"{_hours(compute)} of {_hours(limits.compute_seconds)} sandbox hours",
            "storage": f"{_gb(storage)} GB of {_gb(limits.storage_bytes or 0)} GB",
            "ai": f"{ai} of {limits.ai_queries} queries",
        }
        short = {
            "compute": f"{_hours(compute)} / {_hours(limits.compute_seconds)} h",
            "storage": f"{_gb(storage)} / {_gb(limits.storage_bytes or 0)} GB",
            "ai": f"{ai} / {limits.ai_queries}",
        }
    return {
        **words,
        "short": short,
        "used": {"computeSeconds": round(compute, 1), "storageBytes": storage, "aiQueries": ai},
        "limits": {
            "computeSeconds": limits.compute_seconds,
            "storageBytes": limits.storage_bytes,
            "aiQueries": limits.ai_queries,
        },
        "period": usage.period,
    }
