"""Notifications: what a member should know about, shown in the app and, for alerts, sent by email.

Notifications are added to the caller's session and saved with the action that caused them, so a member
is never told about something that was rolled back.
"""

from collections.abc import Iterable
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..core.logging import get_logger
from ..core.security import utcnow
from ..domain.roles import Role
from ..infrastructure import email as mail
from ..infrastructure.db.models import Notification, User
from .common import Actor, humanize

log = get_logger(__name__)
KINDS = ("alert", "approval", "decision", "share", "refresh")


def notify(
    session: AsyncSession,
    user_ids: Iterable[str | None],
    kind: str,
    title: str,
    body: str = "",
    link: str | None = None,
    *,
    skip: str | None = None,
) -> None:
    """Tell members something. `skip` is the member who did it, who does not need telling."""
    for user_id in dict.fromkeys(u for u in user_ids if u and u not in ("system", "embed", skip)):
        session.add(Notification(user_id=user_id, kind=kind, title=title[:300], body=body, link=link))


async def stewards(session: AsyncSession) -> list[str]:
    """The members who decide on data: stewards, or administrators when there are none."""
    for role in (Role.STEWARD, Role.ADMIN):
        rows = await session.execute(select(User.id).where(User.role == role.value, User.status == "active"))
        ids = [r[0] for r in rows]
        if ids:
            return ids
    return []


async def email(
    session: AsyncSession, user_id: str | None, subject: str, body: str, link: str | None
) -> None:
    """Email a verified, active member. A failed send is logged and never fails the action behind it."""
    user = await session.get(User, user_id) if user_id else None
    if user is None or user.status != "active" or user.email_verified_at is None:
        return
    text = f"Hello {user.name},\n\n{body}\n" + (
        f"\nOpen it in DataForge: {settings.FRONTEND_URL}{link}\n" if link else ""
    )
    try:
        await mail.send_email(user.email, subject, text)
    except Exception as error:  # noqa: BLE001 - the notification in the app still stands
        log.warning("notification_email_failed", to=user.email, error=str(error))


def notification_dict(n: Notification) -> dict[str, Any]:
    return {
        "id": n.id,
        "kind": n.kind,
        "title": n.title,
        "body": n.body,
        "link": n.link,
        "read": n.read_at is not None,
        "at": n.created_at.isoformat(),
        "when": humanize(n.created_at),
    }


async def _unread(session: AsyncSession, actor: Actor) -> int:
    count = await session.execute(
        select(func.count())
        .select_from(Notification)
        .where(Notification.user_id == actor.id, Notification.read_at.is_(None))
    )
    return int(count.scalar() or 0)


async def list_for(session: AsyncSession, actor: Actor, limit: int = 30) -> dict[str, Any]:
    rows = await session.execute(
        select(Notification)
        .where(Notification.user_id == actor.id)
        .order_by(Notification.created_at.desc())
        .limit(max(1, min(limit, 100)))
    )
    return {"items": [notification_dict(n) for n in rows.scalars()], "unread": await _unread(session, actor)}


async def mark_read(session: AsyncSession, actor: Actor, ids: list[str] | None) -> dict[str, Any]:
    """Mark the given notifications read, or all of them when no ids are given."""
    stmt = update(Notification).where(Notification.user_id == actor.id, Notification.read_at.is_(None))
    if ids is not None:
        stmt = stmt.where(Notification.id.in_(ids))
    await session.execute(stmt.values(read_at=utcnow()))
    await session.commit()
    return {"unread": await _unread(session, actor)}
