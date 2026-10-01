"""The audit trail: every security- and data-relevant action, tagged with the member and the resource.

Entries are append-only. A database trigger rejects UPDATE and DELETE on the table, so no role, including
administrators, can alter history through the application or through SQL.
"""

import csv
import io
from datetime import datetime
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..domain.roles import Permission
from ..infrastructure.db.models import AuditEvent
from .common import Actor


async def record(
    session: AsyncSession,
    actor: Actor | None,
    action: str,
    resource_type: str,
    resource_id: str | None = None,
    resource_label: str = "",
    detail: str = "",
    meta: dict[str, Any] | None = None,
) -> None:
    """Add an entry to the current transaction, so it commits (or rolls back) with the action it describes."""
    session.add(
        AuditEvent(
            actor_id=actor.member_id if actor else None,
            actor_label=actor.name if actor else "System",
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_label=resource_label[:300],
            detail=detail[:4000],
            request_id=actor.request_id if actor else None,
            ip=actor.ip if actor else None,
            meta=meta,
        )
    )


def _when(moment: datetime, now: datetime) -> str:
    if moment.date() == now.date():
        return moment.astimezone().strftime("%H:%M")
    days = (now.date() - moment.date()).days
    if days == 1:
        return "Yesterday"
    if days < 7:
        return f"{days} days ago"
    return moment.strftime("%d %b %Y")


def to_dict(event: AuditEvent, now: datetime) -> dict[str, Any]:
    return {
        "id": str(event.id),
        "when": _when(event.occurred_at, now),
        "occurredAt": event.occurred_at.isoformat(),
        "actor": event.actor_label,
        "actorId": event.actor_id,
        "action": event.action,
        "resource": event.resource_label or event.resource_type,
        "resourceType": event.resource_type,
        "resourceId": event.resource_id,
        "detail": event.detail,
        "requestId": event.request_id,
    }


async def list_events(
    session: AsyncSession,
    actor: Actor,
    *,
    query: str | None = None,
    action: str | None = None,
    actor_id: str | None = None,
    resource_id: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[AuditEvent], int]:
    """Members see their own trail; stewards, auditors and administrators see everyone's."""
    stmt = select(AuditEvent)
    if not actor.can(Permission.VIEW_ALL_AUDIT):
        stmt = stmt.where(AuditEvent.actor_id == actor.id)
    elif actor_id:
        stmt = stmt.where(AuditEvent.actor_id == actor_id)
    if action:
        stmt = stmt.where(AuditEvent.action == action)
    if resource_id:
        stmt = stmt.where(AuditEvent.resource_id == resource_id)
    if since:
        stmt = stmt.where(AuditEvent.occurred_at >= since)
    if until:
        stmt = stmt.where(AuditEvent.occurred_at < until)
    if query:
        like = f"%{query}%"
        stmt = stmt.where(
            or_(
                AuditEvent.action.ilike(like),
                AuditEvent.actor_label.ilike(like),
                AuditEvent.resource_label.ilike(like),
                AuditEvent.detail.ilike(like),
            )
        )
    total = int((await session.execute(select(func.count()).select_from(stmt.subquery()))).scalar() or 0)
    rows = await session.execute(stmt.order_by(AuditEvent.id.desc()).limit(min(limit, 1000)).offset(offset))
    return list(rows.scalars()), total


def _cell(value: str) -> str:
    """Stop a spreadsheet reading member-written text (a reason, a name) as a formula."""
    return f"'{value}" if value.startswith(("=", "+", "-", "@", "\t", "\r")) else value


def to_csv(events: list[AuditEvent]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "id",
            "occurred_at",
            "actor",
            "action",
            "resource_type",
            "resource_id",
            "resource",
            "detail",
            "request_id",
        ]
    )
    for e in events:
        writer.writerow(
            [
                e.id,
                e.occurred_at.isoformat(),
                _cell(e.actor_label),
                e.action,
                e.resource_type,
                e.resource_id or "",
                _cell(e.resource_label),
                _cell(e.detail),
                e.request_id or "",
            ]
        )
    return buffer.getvalue()
