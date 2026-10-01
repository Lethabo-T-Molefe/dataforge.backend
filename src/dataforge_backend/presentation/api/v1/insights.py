"""Metrics, dashboards (and their embeds), monitors and alerts, and predictions."""

from typing import Any

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from ....application import dashboards, metrics, monitors, predictions
from ....application.common import Actor
from ..deps import current_actor, data_actor, get_db
from ..schemas import ChartIn, DashboardIn, DashboardUpdateIn, EmbedIn, FreeBody, ScoreIn, ShareIn

router = APIRouter(tags=["Metrics, dashboards, monitors and predictions"])


# ------------------------------------------------------------------ semantic layer


@router.get("/metrics/definitions")
async def metric_definitions(
    actor: Actor = Depends(data_actor), session: AsyncSession = Depends(get_db)
) -> list[dict[str, Any]]:
    return await metrics.list_definitions(session, actor)


@router.post("/metrics/definitions", status_code=status.HTTP_201_CREATED)
async def define_metric(
    body: FreeBody, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Define a metric once: additive (SUM of a column), count, distinct count, or a ratio of two columns."""
    return await metrics.create(session, actor, body.model_dump())


@router.patch("/metrics/definitions/{metric_id}")
async def change_metric(
    metric_id: str,
    body: FreeBody,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await metrics.update(session, actor, metric_id, body.model_dump())


@router.post("/metrics/definitions/{metric_id}/publish")
async def publish_metric(
    metric_id: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    return await metrics.set_status(session, actor, metric_id, "published")


@router.post("/metrics/definitions/{metric_id}/unpublish")
async def unpublish_metric(
    metric_id: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    return await metrics.set_status(session, actor, metric_id, "draft")


@router.delete("/metrics/definitions/{metric_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_metric(
    metric_id: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> None:
    await metrics.delete(session, actor, metric_id)


@router.get("/metrics/summaries")
async def metric_summaries(
    actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> list[dict[str, Any]]:
    """Headline figures: each shared metric's latest month against the month before."""
    return await metrics.summaries(session, actor)


# ------------------------------------------------------------------ dashboards


@router.get("/dashboards")
async def list_dashboards(
    actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> list[dict[str, Any]]:
    return await dashboards.list_dashboards(session, actor)


@router.post("/dashboards", status_code=status.HTTP_201_CREATED)
async def create_dashboard(
    body: DashboardIn, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    return await dashboards.create(session, actor, body.name, body.description, body.visibility)


@router.get("/dashboards/{dashboard_id}")
async def get_dashboard(
    dashboard_id: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Charts are computed for you, under your role: a shared dashboard never shows data you may not see."""
    return await dashboards.get(session, actor, dashboard_id)


@router.patch("/dashboards/{dashboard_id}")
async def update_dashboard(
    dashboard_id: str,
    body: DashboardUpdateIn,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await dashboards.update(session, actor, dashboard_id, body.model_dump(exclude_unset=True))


@router.delete("/dashboards/{dashboard_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dashboard(
    dashboard_id: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> None:
    await dashboards.delete(session, actor, dashboard_id)


@router.post("/dashboards/{dashboard_id}/charts", status_code=status.HTTP_201_CREATED)
async def add_chart(
    dashboard_id: str,
    body: ChartIn,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await dashboards.add_chart(session, actor, dashboard_id, body.model_dump(exclude_none=True))


@router.delete("/dashboards/{dashboard_id}/charts/{chart_id}")
async def remove_chart(
    dashboard_id: str,
    chart_id: str,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await dashboards.remove_chart(session, actor, dashboard_id, chart_id)


@router.post("/dashboards/{dashboard_id}/shares")
async def share_dashboard(
    dashboard_id: str,
    body: ShareIn,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await dashboards.share(session, actor, dashboard_id, [str(e) for e in body.emails])


@router.delete("/dashboards/{dashboard_id}/shares/{user_id}")
async def unshare_dashboard(
    dashboard_id: str,
    user_id: str,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await dashboards.unshare(session, actor, dashboard_id, user_id)


@router.post("/dashboards/{dashboard_id}/embed")
async def embed_dashboard(
    dashboard_id: str,
    body: EmbedIn,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Turn embedding on (organisation dashboards only) and get a signed, expiring embed link."""
    return await dashboards.set_embedding(session, actor, dashboard_id, body.enabled)


@router.get("/embed/dashboards/{dashboard_id}", tags=["Embedding"])
async def embedded_dashboard(
    dashboard_id: str, token: str, response: Response, session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Read-only dashboard for another internal application. Shows organisation-wide data only."""
    response.headers["Cache-Control"] = "private, max-age=60"
    return await dashboards.embedded(session, dashboard_id, token)


# ------------------------------------------------------------------ monitors and alerts


@router.get("/monitors")
async def list_monitors(
    actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> list[dict[str, Any]]:
    return await monitors.list_monitors(session, actor)


@router.post("/monitors", status_code=status.HTTP_201_CREATED)
async def create_monitor(
    body: FreeBody, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Watch a metric: rule kinds change (per cent), above / below (a value), or band (standard deviations)."""
    return await monitors.create(session, actor, body.model_dump())


@router.patch("/monitors/{monitor_id}")
async def update_monitor(
    monitor_id: str,
    body: FreeBody,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await monitors.update(session, actor, monitor_id, body.model_dump())


@router.delete("/monitors/{monitor_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_monitor(
    monitor_id: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> None:
    await monitors.delete(session, actor, monitor_id)


@router.post("/monitors/{monitor_id}/check")
async def check_monitor(
    monitor_id: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    return await monitors.check_now(session, actor, monitor_id)


@router.get("/alerts")
async def list_alerts(
    actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> list[dict[str, Any]]:
    return await monitors.list_alerts(session, actor)


@router.get("/alerts/{alert_id}")
async def get_alert(
    alert_id: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """An alert with its plain-language root cause: the main contributing factors and their figures."""
    return await monitors.get_alert(session, actor, alert_id)


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    return await monitors.acknowledge(session, actor, alert_id)


# ------------------------------------------------------------------ predictions


@router.get("/predictions/models")
async def models(
    actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> list[dict[str, Any]]:
    return await predictions.list_models(session, actor)


@router.post("/predictions/models", status_code=status.HTTP_201_CREATED)
async def publish_model(
    body: FreeBody, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Publish a model version (Model Curator). Factor weights must add up to one."""
    return await predictions.create_version(session, actor, body.model_dump())


@router.post("/predictions/score")
async def score(
    body: ScoreIn, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Score a cohort: a 0–100 score, a band, and the ranked factors behind it, with the model version."""
    return await predictions.score(session, actor, body.model_id, body.cohort)


@router.get("/predictions")
async def prediction_history(
    actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> list[dict[str, Any]]:
    return await predictions.history(session, actor)


@router.get("/predictions/{prediction_id}/export")
async def export_prediction(
    prediction_id: str,
    format: str = "pdf",
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> Response:  # noqa: A002
    content, media, filename = await predictions.export(session, actor, prediction_id, format)
    return Response(
        content, media_type=media, headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
