"""Audit, search, integrations (Superset, Jupyter, Slack) and health."""

from datetime import datetime
from typing import Any
from urllib.parse import parse_qs

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ....application import audit, inbox, integrations, search, slack
from ....application.common import Actor
from ....core.config import settings
from ....core.errors import Unauthorized
from ....core.security import utcnow
from ....domain.roles import Permission
from ....infrastructure import email as mail
from ....infrastructure.notifications import verify_slack_signature
from ..deps import current_actor, get_db, limiter
from ..schemas import ContactIn, NotificationsReadIn, OpenSqlIn

router = APIRouter(tags=["Platform"])


# ------------------------------------------------------------------ notifications


@router.get("/notifications")
async def notifications(
    limit: int = 30, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """The member's latest notifications, newest first, and how many are unread."""
    return await inbox.list_for(session, actor, limit)


@router.post("/notifications/read")
async def read_notifications(
    body: NotificationsReadIn, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    return await inbox.mark_read(session, actor, body.ids)


# ------------------------------------------------------------------ audit


@router.get("/audit/events")
async def audit_events(
    q: str | None = None,
    action: str | None = None,
    actor_id: str | None = None,
    resource_id: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 100,
    offset: int = 0,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """The immutable audit trail. Your own events, or everyone's for stewards, auditors and administrators."""
    events, total = await audit.list_events(
        session,
        actor,
        query=q,
        action=action,
        actor_id=actor_id,
        resource_id=resource_id,
        since=since,
        until=until,
        limit=limit,
        offset=offset,
    )
    now = utcnow()
    return {"total": total, "items": [audit.to_dict(e, now) for e in events]}


@router.get("/audit/export")
async def audit_export(
    since: datetime | None = None,
    until: datetime | None = None,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> Response:
    events, _ = await audit.list_events(session, actor, since=since, until=until, limit=1000)
    await audit.record(
        session, actor, "Exported audit log", "audit", None, "Audit trail", f"{len(events)} events"
    )
    await session.commit()
    return Response(
        audit.to_csv(events),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="dataforge-audit.csv"'},
    )


# ------------------------------------------------------------------ search


@router.get("/search")
async def search_catalogue(
    q: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> list[dict[str, Any]]:
    return await search.search(session, actor, q)


# ------------------------------------------------------------------ integrations


@router.get("/integrations/status")
async def integration_status(actor: Actor = Depends(current_actor)) -> dict[str, Any]:
    return await integrations.status()


@router.post("/integrations/superset/sign-in")
async def superset_sign_in(
    next: str = "/superset/welcome/",
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, str]:  # noqa: A002
    """A two-minute, single-use link that signs you in to Superset with your DataForge role."""
    return await integrations.superset_sign_in(session, actor, next)


@router.get("/integrations/superset/dashboards")
async def superset_dashboards(actor: Actor = Depends(current_actor)) -> list[dict[str, Any]]:
    return await integrations.superset_dashboards(actor)


@router.post("/integrations/superset/dashboards/{dashboard_id}/guest-token")
async def superset_guest_token(
    dashboard_id: int, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """A guest token for embedding a Superset dashboard, carrying your role's row policies."""
    return await integrations.superset_guest_token(session, actor, dashboard_id)


@router.post("/integrations/superset/sync")
async def superset_sync(
    actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    actor.require(Permission.MANAGE_POLICIES, "Only a Data Steward can resynchronise Superset.")
    return await integrations.sync_all(session)


@router.post("/integrations/jupyter/sign-in")
async def jupyter_sign_in(
    actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, str]:
    """A single-use link to your Jupyter folder. Kernels query DataForge with your permissions."""
    return await integrations.jupyter_sign_in(session, actor)


@router.post("/integrations/jupyter/notebooks/{notebook_id}")
async def open_notebook_in_jupyter(
    notebook_id: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, str]:
    return await integrations.open_notebook(session, actor, notebook_id)


@router.post("/integrations/jupyter/sql")
async def open_sql_in_jupyter(
    body: OpenSqlIn, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, str]:
    """Take generated SQL (a chart, an answer) into a new Jupyter notebook."""
    return await integrations.open_sql(session, actor, body.title, body.sql, body.note)


@router.post("/integrations/slack/command", include_in_schema=False)
async def slack_command(request: Request, session: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Slack slash command endpoint. Requests must carry a valid Slack signature."""
    body = await request.body()
    if not verify_slack_signature(
        body, request.headers.get("X-Slack-Request-Timestamp"), request.headers.get("X-Slack-Signature")
    ):
        raise Unauthorized("Invalid Slack signature.")
    form = {k: v[0] for k, v in parse_qs(body.decode("utf-8")).items()}
    return await slack.handle_command(
        session, form.get("team_id", ""), form.get("user_id", ""), form.get("text", "")
    )


# ------------------------------------------------------------------ contact


@router.post("/contact", status_code=202, tags=["Contact"])
@limiter.limit("3/minute;20/day")
async def contact(request: Request, body: ContactIn) -> dict[str, bool]:
    """The public contact form. The message is emailed to the platform contact; nothing is stored."""
    await mail.send_contact_message(body.name, body.email, body.organisation, body.message)
    return {"accepted": True}


# ------------------------------------------------------------------ health


@router.get("/health", tags=["Health"])
async def health(session: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Liveness and readiness: the database, and whether the sandbox role exists."""
    database = "ok"
    sandbox = "ok"
    try:
        await session.execute(text("SELECT 1"))
        role = await session.execute(
            text("SELECT 1 FROM pg_roles WHERE rolname = :r"), {"r": settings.SANDBOX_ROLE}
        )
        if role.first() is None:
            sandbox = "missing role: run migrations"
    except Exception:  # noqa: BLE001 - health reports rather than raises
        database = "unreachable"
    return {
        "status": "ok" if database == "ok" and sandbox == "ok" else "degraded",
        "database": database,
        "sandbox": sandbox,
        "version": "1.0.0",
    }
