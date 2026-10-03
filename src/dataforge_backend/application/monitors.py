"""Metric monitors and alerts: watch a shared metric, raise an alert when it moves, and explain why."""

from datetime import timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.errors import Forbidden, NotFound, ValidationFailed
from ..core.logging import get_logger
from ..core.security import utcnow
from ..domain.analysis import MONTH_DIMENSION, AnalysisDataset
from ..domain.monitoring import (
    CADENCES,
    GroupChange,
    MonitorError,
    MonitorRule,
    RootCauseInput,
    evaluate,
    root_cause,
    validate_rule,
)
from ..domain.roles import Permission
from ..infrastructure import notifications
from ..infrastructure.db.models import Alert, Monitor, User
from . import analysis, audit, catalogue, inbox
from .common import SYSTEM, Actor, humanize

log = get_logger(__name__)
CADENCE_LABELS = {"hourly": "Hourly", "daily": "Daily", "weekly": "Weekly"}


def _rule(monitor: Monitor) -> MonitorRule:
    rule = monitor.rule or {}
    return MonitorRule(
        str(rule.get("kind", "change")), float(rule.get("threshold", 15)), int(rule.get("window", 6))
    )


def monitor_dict(monitor: Monitor, dataset: AnalysisDataset | None) -> dict[str, Any]:
    measure = dataset.measures.get(monitor.measure) if dataset else None
    unit = measure.unit if measure else "count"
    scope = " · ".join(monitor.filters.values()) if monitor.filters else ""
    return {
        "id": monitor.id,
        "name": monitor.name,
        "metric": (measure.label if measure else monitor.measure) + (f" · {scope}" if scope else ""),
        "threshold": _rule(monitor).describe(unit),
        "cadence": CADENCE_LABELS.get(monitor.cadence, monitor.cadence.capitalize()),
        "owner": monitor.owner_label,
        "ownerId": monitor.owner_id,
        "status": monitor.status,
        "lastChecked": humanize(monitor.last_checked_at),
        "datasetId": monitor.dataset_id,
        "measure": monitor.measure,
        "filters": monitor.filters,
        "rule": monitor.rule,
        "notifySlack": monitor.notify_slack,
        "notifyEmail": monitor.notify_email,
        "available": dataset is not None,
    }


def alert_dict(alert: Alert) -> dict[str, Any]:
    return {
        "id": alert.id,
        "monitorId": alert.monitor_id,
        "metric": alert.metric,
        "message": alert.message,
        "when": humanize(alert.created_at),
        "createdAt": alert.created_at.isoformat(),
        "severity": alert.severity,
        "rootCause": alert.root_cause or [],
        "explore": alert.explore,
        "acknowledged": alert.acknowledged_at is not None,
        "month": alert.month,
        "previousMonth": alert.previous_month,
    }


async def _datasets(session: AsyncSession, actor: Actor) -> dict[str, AnalysisDataset]:
    return {d.id: d for d in await catalogue.analysis_catalogue(session, actor)}


async def list_monitors(session: AsyncSession, actor: Actor) -> list[dict[str, Any]]:
    datasets = await _datasets(session, actor)
    rows = await session.execute(select(Monitor).order_by(Monitor.created_at))
    return [
        monitor_dict(m, datasets.get(m.dataset_id))
        for m in rows.scalars()
        if m.dataset_id in datasets or m.owner_id == actor.id
    ]


def _validated(
    dataset: AnalysisDataset, body: dict[str, Any]
) -> tuple[str, dict[str, str], MonitorRule, str]:
    measure = str(body.get("measure", ""))
    if measure not in dataset.measures:
        raise ValidationFailed(f"{measure or 'That measure'} is not a measure of {dataset.name}.")
    if not dataset.has_time:
        raise ValidationFailed(f"{dataset.name} has no months, so it cannot be monitored over time.")
    filters: dict[str, str] = {}
    for dim, value in (body.get("filters") or {}).items():
        if dim not in dataset.dimensions or (
            dataset.dimensions[dim].values and value not in dataset.dimensions[dim].values
        ):
            raise ValidationFailed(f"{dim} = {value} is not a valid filter.")
        filters[dim] = str(value)
    rule_body = body.get("rule") or {}
    try:
        rule = validate_rule(
            MonitorRule(
                str(rule_body.get("kind", "change")),
                float(rule_body.get("threshold", 15)),
                int(rule_body.get("window", 6)),
            )
        )
    except (MonitorError, TypeError, ValueError) as error:
        raise ValidationFailed(str(error)) from error
    cadence = str(body.get("cadence", "daily")).lower()
    if cadence not in CADENCES:
        raise ValidationFailed("Cadence is hourly, daily or weekly.")
    return measure, filters, rule, cadence


async def create(session: AsyncSession, actor: Actor, body: dict[str, Any]) -> dict[str, Any]:
    actor.require(Permission.MANAGE_MONITORS, "Your role cannot create monitors.")
    datasets = await _datasets(session, actor)
    dataset = datasets.get(str(body.get("datasetId")))
    if dataset is None:
        raise NotFound("That dataset is not available for analysis.")
    measure, filters, rule, cadence = _validated(dataset, body)
    monitor = Monitor(
        name=str(body.get("name") or f"{dataset.measures[measure].label} watch").strip()[:200],
        dataset_id=dataset.id,
        measure=measure,
        filters=filters,
        rule={"kind": rule.kind, "threshold": rule.threshold, "window": rule.window},
        cadence=cadence,
        owner_id=actor.id,
        owner_label=actor.name,
        notify_slack=bool(body.get("notifySlack", False)),
        notify_email=bool(body.get("notifyEmail", True)),
    )
    session.add(monitor)
    await session.flush()
    await audit.record(
        session,
        actor,
        "Created monitor",
        "monitor",
        monitor.id,
        monitor.name,
        rule.describe(dataset.measures[measure].unit),
    )
    await session.commit()
    return monitor_dict(monitor, dataset)


async def _own(session: AsyncSession, actor: Actor, monitor_id: str) -> Monitor:
    monitor = await session.get(Monitor, monitor_id)
    if monitor is None:
        raise NotFound("Monitor not found.")
    if monitor.owner_id != actor.id and actor.role != "admin":
        raise Forbidden("Only the monitor's owner can change it.")
    return monitor


async def update(
    session: AsyncSession, actor: Actor, monitor_id: str, body: dict[str, Any]
) -> dict[str, Any]:
    monitor = await _own(session, actor, monitor_id)
    datasets = await _datasets(session, actor)
    dataset = datasets.get(monitor.dataset_id)
    if "status" in body:
        if body["status"] not in ("watching", "paused"):
            raise ValidationFailed("Status is watching or paused.")
        monitor.status = body["status"]
    if dataset is not None and any(k in body for k in ("measure", "filters", "rule", "cadence")):
        merged = {
            "measure": monitor.measure,
            "filters": monitor.filters,
            "rule": monitor.rule,
            "cadence": monitor.cadence,
            **body,
        }
        measure, filters, rule, cadence = _validated(dataset, merged)
        monitor.measure, monitor.filters, monitor.cadence = measure, filters, cadence
        monitor.rule = {"kind": rule.kind, "threshold": rule.threshold, "window": rule.window}
    if body.get("name"):
        monitor.name = str(body["name"]).strip()[:200]
    if "notifySlack" in body:
        monitor.notify_slack = bool(body["notifySlack"])
    if "notifyEmail" in body:
        monitor.notify_email = bool(body["notifyEmail"])
    await audit.record(session, actor, "Changed monitor", "monitor", monitor.id, monitor.name, monitor.status)
    await session.commit()
    return monitor_dict(monitor, dataset)


async def delete(session: AsyncSession, actor: Actor, monitor_id: str) -> None:
    monitor = await _own(session, actor, monitor_id)
    await session.delete(monitor)
    await audit.record(session, actor, "Deleted monitor", "monitor", monitor_id, monitor.name)
    await session.commit()


async def _root_cause(
    session: AsyncSession, actor: Actor, dataset: AnalysisDataset, monitor: Monitor, before: str, after: str
) -> list[dict[str, str]]:
    measure = dataset.measures[monitor.measure]
    dims = [d for d in dataset.dimensions if d not in monitor.filters]
    breakdowns: dict[str, list[GroupChange]] = {}
    for dim in dims:
        rows = await analysis.query_groups(
            session,
            actor,
            dataset,
            {"v": measure.id},
            [dim, MONTH_DIMENSION],
            monitor.filters,
            pair=(before, after),
            parts=["v"],
            kind="monitor",
        )
        groups: dict[str, dict[str, float]] = {}
        for row in rows:
            entry = groups.setdefault(str(row["d0"]), {})
            key = "a" if row["d1"] == before else "b"
            entry[f"{key}_num"] = float(row["v_num"] or 0)
            entry[f"{key}_den"] = float(row["v_den"] or 0)
            entry[key] = float(row["v"] or 0)
        breakdowns[dataset.dimensions[dim].label] = [
            GroupChange(
                group=name,
                before=e.get("a_num", 0.0) if measure.is_ratio else e.get("a", 0.0),
                after=e.get("b_num", 0.0) if measure.is_ratio else e.get("b", 0.0),
                den_before=e.get("a_den", 0.0),
                den_after=e.get("b_den", 0.0),
            )
            for name, e in groups.items()
        ]
    totals = await analysis.query_groups(
        session,
        actor,
        dataset,
        {"v": measure.id},
        [MONTH_DIMENSION],
        monitor.filters,
        pair=(before, after),
        kind="monitor",
    )
    by_month = {str(r["d0"]): float(r["v"] or 0) for r in totals}
    data = RootCauseInput(
        label=measure.label,
        unit=measure.unit,
        is_ratio=measure.is_ratio,
        month_before=dataset.month_label(before),
        month_after=dataset.month_label(after),
        total_before=by_month.get(before, 0.0),
        total_after=by_month.get(after, 0.0),
        breakdowns=breakdowns,
    )
    # A total that is a volume × a rate (revenue = enrolments × average fee) also splits into those two effects.
    if measure.additive and measure.column:
        rate = next(
            (
                m
                for m in dataset.measures.values()
                if m.is_ratio and m.numerator == measure.column and m.denominator != "*"
            ),
            None,
        )
        volume = next(
            (m for m in dataset.measures.values() if rate and m.additive and m.column == rate.denominator),
            None,
        )
        if rate and volume:
            vol_rows = await analysis.query_groups(
                session,
                actor,
                dataset,
                {"v": volume.id},
                [MONTH_DIMENSION],
                monitor.filters,
                pair=(before, after),
                kind="monitor",
            )
            vols = {str(r["d0"]): float(r["v"] or 0) for r in vol_rows}
            data.volume_label, data.rate_label, data.rate_unit = volume.label, rate.label, rate.unit
            data.volume_before, data.volume_after = vols.get(before), vols.get(after)
    return root_cause(data)


async def evaluate_monitor(session: AsyncSession, monitor: Monitor) -> dict[str, Any]:
    """Evaluate one monitor as its owner (so the owner's row policy applies) and raise an alert if it trips."""
    owner = await session.get(User, monitor.owner_id) if monitor.owner_id else None
    actor = Actor.from_user(owner) if owner is not None and owner.status == "active" else SYSTEM
    datasets = await _datasets(session, actor)
    dataset = datasets.get(monitor.dataset_id)
    monitor.last_checked_at = utcnow()
    if dataset is None or monitor.measure not in dataset.measures:
        await session.commit()
        return {
            "monitorId": monitor.id,
            "triggered": False,
            "message": "The monitored data is no longer available.",
        }
    measure = dataset.measures[monitor.measure]
    rows = await analysis.query_groups(
        session, actor, dataset, {"v": measure.id}, [MONTH_DIMENSION], monitor.filters, kind="monitor"
    )
    series = sorted((str(r["d0"]), float(r["v"])) for r in rows if r["v"] is not None)
    result = evaluate(series, _rule(monitor), measure.unit, dataset.month_label)
    monitor.last_value = result.value
    alert_out = None
    if result.triggered and result.month and result.previous_month:
        existing = await session.execute(
            select(Alert.id).where(Alert.monitor_id == monitor.id, Alert.month == result.month)
        )
        if existing.first() is None:
            cause = await _root_cause(session, actor, dataset, monitor, result.previous_month, result.month)
            scope = " · ".join(monitor.filters.values())
            first_dim = next((d for d in dataset.dimensions if d not in monitor.filters), None)
            alert = Alert(
                monitor_id=monitor.id,
                metric=f"{measure.label}{f' · {scope}' if scope else ''}",
                message=result.message,
                severity=result.severity,
                month=result.month,
                previous_month=result.previous_month,
                value=result.value,
                previous=result.previous,
                root_cause=cause,
                explore={
                    "dataset": dataset.id,
                    "method": "explain" if measure.additive else "trend",
                    "measure": measure.id,
                    **({"dimension": first_dim} if first_dim else {}),
                },
            )
            session.add(alert)
            await session.flush()
            await audit.record(
                session, SYSTEM, "Raised alert", "monitor", monitor.id, alert.metric, result.message
            )
            inbox.notify(
                session,
                [monitor.owner_id],
                "alert",
                f"Alert: {alert.metric}",
                result.message,
                f"/monitors/{alert.id}",
            )
            alert_out = alert
    monitor.last_month = result.month
    await session.commit()
    if alert_out is not None:
        lines = "\n".join(
            f"• {c['factor']}: {c['contribution']} ({c['detail']})" for c in alert_out.root_cause
        )
        if monitor.notify_slack:
            await notifications.post_slack(f"*{alert_out.metric}*: {alert_out.message}\n{lines}")
        if monitor.notify_email:
            await inbox.email(
                session,
                monitor.owner_id,
                f"DataForge alert: {alert_out.metric}"[:150],
                f"{alert_out.message}\n\nWhat drove it:\n{lines.replace('• ', '- ') or '- No breakdown available'}",
                f"/monitors/{alert_out.id}",
            )
    return {
        "monitorId": monitor.id,
        "triggered": result.triggered,
        "message": result.message,
        "alert": alert_dict(alert_out) if alert_out else None,
    }


async def check_now(session: AsyncSession, actor: Actor, monitor_id: str) -> dict[str, Any]:
    monitor = await session.get(Monitor, monitor_id)
    if monitor is None:
        raise NotFound("Monitor not found.")
    if monitor.owner_id != actor.id and actor.role != "admin":
        raise Forbidden("Only the owner can check this monitor now.")
    return await evaluate_monitor(session, monitor)


async def evaluate_due(session: AsyncSession) -> list[dict[str, Any]]:
    now = utcnow()
    rows = await session.execute(select(Monitor).where(Monitor.status == "watching"))
    results = []
    for monitor in rows.scalars().all():
        due = (
            monitor.last_checked_at is None
            or monitor.last_checked_at + timedelta(seconds=CADENCES.get(monitor.cadence, 86_400)) <= now
        )
        if due:
            try:
                results.append(await evaluate_monitor(session, monitor))
            except Exception as error:  # noqa: BLE001 - one failing monitor must not stop the others
                await session.rollback()
                log.warning("monitor_evaluation_failed", monitor=monitor.id, error=str(error))
    return results


async def list_alerts(session: AsyncSession, actor: Actor, limit: int = 50) -> list[dict[str, Any]]:
    datasets = await _datasets(session, actor)
    rows = await session.execute(
        select(Alert, Monitor)
        .outerjoin(Monitor, Monitor.id == Alert.monitor_id)
        .order_by(Alert.created_at.desc())
        .limit(limit)
    )
    return [
        alert_dict(a)
        for a, m in rows
        if m is not None and (m.dataset_id in datasets or m.owner_id == actor.id)
    ]


async def get_alert(session: AsyncSession, actor: Actor, alert_id: str) -> dict[str, Any]:
    alert = await session.get(Alert, alert_id)
    monitor = await session.get(Monitor, alert.monitor_id) if alert and alert.monitor_id else None
    datasets = await _datasets(session, actor)
    if (
        alert is None
        or monitor is None
        or (monitor.dataset_id not in datasets and monitor.owner_id != actor.id)
    ):
        raise NotFound("Alert not found.")
    await audit.record(session, actor, "Viewed root cause", "alert", alert.id, alert.metric)
    await session.commit()
    return alert_dict(alert)


async def acknowledge(session: AsyncSession, actor: Actor, alert_id: str) -> dict[str, Any]:
    await get_alert(session, actor, alert_id)
    alert = await session.get(Alert, alert_id)
    assert alert is not None
    alert.acknowledged_by, alert.acknowledged_at = actor.id, utcnow()
    await audit.record(session, actor, "Acknowledged alert", "alert", alert.id, alert.metric)
    await session.commit()
    return alert_dict(alert)
