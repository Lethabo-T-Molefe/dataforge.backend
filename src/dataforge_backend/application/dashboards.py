"""Dashboards of saved charts. Every chart is computed for the person viewing it, under their own role, so a
shared or embedded dashboard never shows data the viewer may not see."""

from datetime import timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.errors import Forbidden, NotFound, Refused, ValidationFailed
from ..core.security import create_token, decode_token, utcnow
from ..domain.analysis import AnalysisError, config_from_dict
from ..domain.roles import Permission
from ..domain.sql_guard import QueryRefused
from ..infrastructure.db.models import Dashboard, DashboardChart
from . import analysis, audit, catalogue, workbench
from .common import Actor, can_see, humanize, shared_ids, slugify
from .datasets import share_resource, shares_of, unshare_resource

VISIBILITY_LABELS = {"organisation": "Organisation", "team": "Team", "private": "Private"}
EMBED_HOURS = 12


def embed_viewer() -> Actor:
    """Embedded dashboards render as the least-privileged analytical role: organisation-wide data only."""
    return Actor(
        id="embed",
        name="Embedded viewer",
        email="embed@dataforge.local",
        role="executive",
        team=None,
        verified=True,
        mfa_enabled=False,
    )


async def _render_chart(session: AsyncSession, actor: Actor, chart: DashboardChart) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": chart.id,
        "title": chart.title,
        "type": chart.type,
        "metric": chart.metric,
        "sql": chart.sql,
        "explanation": chart.explanation,
        "source": chart.source,
        "config": chart.config,
        "points": [],
    }
    try:
        if chart.source == "build" and chart.config:
            datasets = {d.id: d for d in await catalogue.analysis_catalogue(session, actor)}
            dataset = datasets.get(str(chart.config.get("datasetId")))
            if dataset is None:
                return {**base, "unavailable": "You do not have access to the data behind this chart."}
            config = config_from_dict(dataset, chart.config)
            result = await analysis.execute(session, actor, dataset, config)
            visual = result["visual"]
            if visual["kind"] in ("bars", "contribution"):
                points = [{"name": p["name"], "value": p["value"]} for p in visual["points"]]
            elif visual["kind"] == "lines":
                series = visual["series"][0] if visual["series"] else None
                points = (
                    [{"name": p["name"], "value": p.get(series)} for p in visual["points"]] if series else []
                )
            elif visual["kind"] == "histogram":
                points = visual["bins"]
            else:
                points = [{"name": p["name"], "value": p["y"]} for p in visual["points"]]
            return {
                **base,
                "points": points,
                "sql": result["sql"],
                "format": result["format"],
                "unit": result.get("measureUnit"),
                "reading": result["reading"],
            }
        output = await workbench.run_sql(session, actor, chart.sql, 200, source="dashboard")
        if output["refused"]:
            return {**base, "unavailable": output["explanation"]}
        rows = output["rows"]
        if len(output["columns"]) < 2:
            return {**base, "unavailable": "A SQL chart needs a label column and a value column."}
        return {
            **base,
            "points": [
                {"name": str(r[0]), "value": r[1] if isinstance(r[1], int | float) else None} for r in rows
            ],
        }
    except (AnalysisError, QueryRefused, Refused) as error:
        return {
            **base,
            "unavailable": str(getattr(error, "message", None) or getattr(error, "reason", None) or error),
        }


async def _visible(session: AsyncSession, actor: Actor, dashboard: Dashboard) -> bool:
    if actor.id == "embed":
        return dashboard.embed_enabled and dashboard.visibility == "organisation"
    return can_see(
        actor,
        dashboard.owner_id,
        dashboard.visibility,
        dashboard.team,
        await shared_ids(session, actor, "dashboard"),
        dashboard.id,
    )


async def _get(session: AsyncSession, actor: Actor, dashboard_id: str) -> Dashboard:
    dashboard = await session.get(Dashboard, dashboard_id)
    if dashboard is None or not await _visible(session, actor, dashboard):
        raise NotFound("Dashboard not found.")
    await session.refresh(dashboard, ["charts"])
    return dashboard


def _require_owner(actor: Actor, dashboard: Dashboard) -> None:
    if dashboard.owner_id != actor.id and actor.role != "admin":
        raise Forbidden("Only the owner can change this dashboard.")


async def to_dict(
    session: AsyncSession, actor: Actor, dashboard: Dashboard, render: bool = True
) -> dict[str, Any]:
    shared = await shares_of(session, "dashboard", dashboard.id)
    shared_with = ["Organisation"] if dashboard.visibility == "organisation" else [s["name"] for s in shared]
    if dashboard.visibility == "team" and dashboard.team:
        shared_with = [f"Team · {dashboard.team}", *shared_with]
    charts = [await _render_chart(session, actor, c) for c in dashboard.charts] if render else []
    return {
        "id": dashboard.id,
        "name": dashboard.name,
        "owner": dashboard.owner_label,
        "ownerId": dashboard.owner_id,
        "visibility": VISIBILITY_LABELS[dashboard.visibility],
        "description": dashboard.description,
        "charts": charts,
        "updated": humanize(dashboard.updated_at),
        "updatedAt": dashboard.updated_at.isoformat(),
        "sharedWith": shared_with,
        "sharedMembers": shared,
        "embedEnabled": dashboard.embed_enabled,
        "canEdit": dashboard.owner_id == actor.id or actor.role == "admin",
        "supersetUuid": dashboard.superset_uuid,
    }


async def list_dashboards(session: AsyncSession, actor: Actor) -> list[dict[str, Any]]:
    rows = (await session.execute(select(Dashboard).order_by(Dashboard.updated_at.desc()))).scalars().all()
    out = []
    for dashboard in rows:
        if await _visible(session, actor, dashboard):
            await session.refresh(dashboard, ["charts"])
            out.append(
                {
                    "id": dashboard.id,
                    "name": dashboard.name,
                    "owner": dashboard.owner_label,
                    "visibility": VISIBILITY_LABELS[dashboard.visibility],
                    "updated": humanize(dashboard.updated_at),
                    "charts": len(dashboard.charts),
                    "description": dashboard.description,
                    "ownerId": dashboard.owner_id,
                    "canEdit": dashboard.owner_id == actor.id or actor.role == "admin",
                }
            )
    return out


async def get(session: AsyncSession, actor: Actor, dashboard_id: str) -> dict[str, Any]:
    dashboard = await _get(session, actor, dashboard_id)
    result = await to_dict(session, actor, dashboard)
    if actor.id not in ("embed",):
        await audit.record(
            session,
            actor,
            "Opened dashboard",
            "dashboard",
            dashboard.id,
            dashboard.name,
            f"{VISIBILITY_LABELS[dashboard.visibility]} share",
        )
    await session.commit()
    return result


async def create(
    session: AsyncSession, actor: Actor, name: str, description: str, visibility: str
) -> dict[str, Any]:
    actor.require(Permission.MANAGE_DASHBOARDS, "Your role cannot create dashboards.")
    if visibility not in VISIBILITY_LABELS:
        raise ValidationFailed("Visibility is private, team or organisation.")
    base = slugify(name or "dashboard", 60)
    dashboard_id, n = base, 2
    while await session.get(Dashboard, dashboard_id) is not None:
        dashboard_id, n = f"{base[:55]}-{n}", n + 1
    dashboard = Dashboard(
        id=dashboard_id,
        name=name.strip()[:200] or "Untitled dashboard",
        description=description[:2000],
        owner_id=actor.id,
        owner_label=actor.name,
        visibility=visibility,
        team=actor.team,
    )
    session.add(dashboard)
    await audit.record(
        session,
        actor,
        "Created dashboard",
        "dashboard",
        dashboard.id,
        dashboard.name,
        VISIBILITY_LABELS[visibility],
    )
    await session.commit()
    await session.refresh(dashboard, ["charts"])
    return await to_dict(session, actor, dashboard)


async def update(
    session: AsyncSession, actor: Actor, dashboard_id: str, changes: dict[str, Any]
) -> dict[str, Any]:
    dashboard = await _get(session, actor, dashboard_id)
    _require_owner(actor, dashboard)
    if changes.get("name"):
        dashboard.name = str(changes["name"]).strip()[:200]
    if changes.get("description") is not None:
        dashboard.description = str(changes["description"])[:2000]
    if changes.get("visibility"):
        visibility = str(changes["visibility"]).lower()
        if visibility not in VISIBILITY_LABELS:
            raise ValidationFailed("Visibility is private, team or organisation.")
        dashboard.visibility = visibility
        if visibility != "organisation":
            dashboard.embed_enabled = False
    dashboard.updated_at = utcnow()
    await audit.record(session, actor, "Changed dashboard", "dashboard", dashboard.id, dashboard.name)
    await session.commit()
    return await to_dict(session, actor, dashboard)


async def delete(session: AsyncSession, actor: Actor, dashboard_id: str) -> None:
    dashboard = await _get(session, actor, dashboard_id)
    _require_owner(actor, dashboard)
    await session.delete(dashboard)
    await audit.record(session, actor, "Deleted dashboard", "dashboard", dashboard_id, dashboard.name)
    await session.commit()


async def add_chart(
    session: AsyncSession, actor: Actor, dashboard_id: str, body: dict[str, Any]
) -> dict[str, Any]:
    dashboard = await _get(session, actor, dashboard_id)
    _require_owner(actor, dashboard)
    chart_type = str(body.get("type", "bar"))
    if chart_type not in ("bar", "line", "area"):
        raise ValidationFailed("Chart type is bar, line or area.")
    title = str(body.get("title", "")).strip()[:200] or "Untitled chart"
    config = body.get("config")
    sql = str(body.get("sql") or "")
    if config:
        datasets = {d.id: d for d in await catalogue.analysis_catalogue(session, actor)}
        dataset = datasets.get(str(config.get("datasetId")))
        if dataset is None:
            raise NotFound("That dataset is not available for analysis.")
        try:
            parsed = config_from_dict(dataset, config)
        except AnalysisError as error:
            raise ValidationFailed(str(error)) from error
        if parsed.method not in ("compare", "trend", "summarise", "explain"):
            raise ValidationFailed("Dashboards hold comparisons, trends, summaries and contributions.")
        measure = dataset.measures[parsed.measure]
        chart = DashboardChart(
            dashboard_id=dashboard.id,
            position=len(dashboard.charts),
            title=title,
            type=chart_type,
            metric=measure.label,
            source="build",
            config=parsed.to_dict(),
            explanation=str(body.get("explanation") or measure.description or ""),
        )
    elif sql:
        actor.require(Permission.RUN_SQL)
        check = await workbench.run_sql(session, actor, sql, 5, source="dashboard")
        if check["refused"]:
            raise ValidationFailed(check["explanation"])
        chart = DashboardChart(
            dashboard_id=dashboard.id,
            position=len(dashboard.charts),
            title=title,
            type=chart_type,
            metric=str(body.get("metric") or title),
            source="sql",
            sql=sql,
            explanation=str(body.get("explanation") or ""),
        )
    else:
        raise ValidationFailed("Give a chart configuration or a SQL query.")
    session.add(chart)
    dashboard.updated_at = utcnow()
    await audit.record(session, actor, "Added chart", "dashboard", dashboard.id, dashboard.name, title)
    await session.commit()
    await session.refresh(dashboard, ["charts"])
    return await to_dict(session, actor, dashboard)


async def remove_chart(
    session: AsyncSession, actor: Actor, dashboard_id: str, chart_id: str
) -> dict[str, Any]:
    dashboard = await _get(session, actor, dashboard_id)
    _require_owner(actor, dashboard)
    chart = next((c for c in dashboard.charts if c.id == chart_id), None)
    if chart is None:
        raise NotFound("Chart not found.")
    await session.delete(chart)
    await session.flush()
    await session.refresh(dashboard, ["charts"])
    for index, remaining in enumerate(dashboard.charts):
        remaining.position = index
    dashboard.updated_at = utcnow()
    await audit.record(
        session, actor, "Removed chart", "dashboard", dashboard.id, dashboard.name, chart.title
    )
    await session.commit()
    return await to_dict(session, actor, dashboard)


async def share(session: AsyncSession, actor: Actor, dashboard_id: str, emails: list[str]) -> dict[str, Any]:
    dashboard = await _get(session, actor, dashboard_id)
    _require_owner(actor, dashboard)
    await share_resource(session, actor, "dashboard", dashboard.id, dashboard.name, emails)
    return await to_dict(session, actor, dashboard, render=False)


async def unshare(session: AsyncSession, actor: Actor, dashboard_id: str, user_id: str) -> dict[str, Any]:
    dashboard = await _get(session, actor, dashboard_id)
    _require_owner(actor, dashboard)
    await unshare_resource(session, actor, "dashboard", dashboard.id, dashboard.name, user_id)
    return await to_dict(session, actor, dashboard, render=False)


async def set_embedding(
    session: AsyncSession, actor: Actor, dashboard_id: str, enabled: bool
) -> dict[str, Any]:
    dashboard = await _get(session, actor, dashboard_id)
    _require_owner(actor, dashboard)
    if enabled and dashboard.visibility != "organisation":
        raise ValidationFailed("Only organisation-wide dashboards can be embedded.")
    dashboard.embed_enabled = enabled
    await audit.record(
        session,
        actor,
        "Enabled embedding" if enabled else "Disabled embedding",
        "dashboard",
        dashboard.id,
        dashboard.name,
    )
    await session.commit()
    result: dict[str, Any] = {"embedEnabled": enabled}
    if enabled:
        token = create_token(
            dashboard.id, "embed", expires_in=timedelta(hours=EMBED_HOURS), extra={"dashboard": dashboard.id}
        )
        result.update(
            token=token,
            expiresInHours=EMBED_HOURS,
            path=f"/api/v1/embed/dashboards/{dashboard.id}?token={token}",
        )
    return result


async def embedded(session: AsyncSession, dashboard_id: str, token: str) -> dict[str, Any]:
    claims = decode_token(token, "embed")
    if claims.extra.get("dashboard") != dashboard_id:
        raise NotFound("Dashboard not found.")
    viewer = embed_viewer()
    dashboard = await _get(session, viewer, dashboard_id)
    result = await to_dict(session, viewer, dashboard)
    result.pop("sharedMembers", None)
    result["canEdit"] = False
    return result
