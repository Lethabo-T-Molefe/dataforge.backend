"""The scheduled anomaly scan: find unusual months in governed metrics nobody set a monitor on, and explain them.

For every approved dataset with enough months, the scan looks at each governed metric overall and split by up
to two small dimensions (twelve values or fewer), and judges each of the last few months against the months
before it (domain.anomaly). Unusual months become alerts with an automatic root cause, an insight card on the
dataset, and a notification to its owner and the stewards.

Safeguards against noise and over-reach:

* Only governed metrics: a shared definition, not every numeric column.
* Support floors: a rate needs a denominator of at least 30 a month, a count a typical level of at least 10.
* The strongest few findings per dataset per run, each group and month reported once.
* Restricted-personal datasets are left to monitors someone sets up deliberately.
* The scan computes over every row, so a role with a row policy on a dataset does not see its scan alerts.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..core.logging import get_logger
from ..core.security import utcnow
from ..domain import anomaly
from ..domain.analysis import MONTH_DIMENSION, AnalysisDataset
from ..domain.analysis.format import format_value, js_fixed
from ..infrastructure.db.models import Alert, AuditEvent, Dataset, Insight
from . import analysis, audit, catalogue, inbox
from .common import SYSTEM, Actor
from .monitors import alert_dict, comparison_month, root_cause_for

log = get_logger(__name__)

SCAN_ACTION = "Ran anomaly scan"
LOOKBACK_MONTHS = 3
MAX_GROUPS = 12
MAX_SPLITS = 2
MAX_FINDINGS = 3
MIN_DENOMINATOR = 30
MIN_LEVEL = 10
MIN_MONTHS = anomaly.MIN_TREND_HISTORY + 2


@dataclass
class Finding:
    dataset: AnalysisDataset
    measure_id: str
    filters: dict[str, str]
    month: str
    verdict: anomaly.Detection
    months: list[str]
    previous: str
    support: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, tuple[tuple[str, str], ...], str]:
        return (self.dataset.id, tuple(sorted(self.filters.items())), self.month)


def _series(rows: list[dict[str, Any]], key: str | None) -> dict[str, list[tuple[str, float, float]]]:
    """Group rows into (month, value, denominator) series, keyed by group (or one series when key is None)."""
    out: dict[str, list[tuple[str, float, float]]] = {}
    for row in rows:
        if row.get("v") is None:
            continue
        group = "" if key is None else str(row[key])
        month = str(row["d1" if key is not None else "d0"])
        out.setdefault(group, []).append((month, float(row["v"]), float(row.get("v_den") or 0)))
    return {g: sorted(points) for g, points in out.items()}


def _supported(points: list[tuple[str, float, float]], ratio: bool) -> tuple[bool, float]:
    if ratio:
        level = statistics.median(p[2] for p in points)
        return level >= MIN_DENOMINATOR, level
    level = statistics.median(abs(p[1]) for p in points)
    return level >= MIN_LEVEL, level


def _judge(
    dataset: AnalysisDataset,
    measure_id: str,
    filters: dict[str, str],
    points: list[tuple[str, float, float]],
    ratio: bool,
    k: float,
) -> list[Finding]:
    if len(points) < MIN_MONTHS:
        return []
    supported, level = _supported(points, ratio)
    if not supported:
        return []
    months = [p[0] for p in points]
    series = [(p[0], p[1]) for p in points]
    found = []
    for t in range(max(MIN_MONTHS - 1, len(series) - LOOKBACK_MONTHS), len(series)):
        verdict = anomaly.detect(series[: t + 1], k)
        if verdict.unusual and verdict.month:
            found.append(
                Finding(dataset, measure_id, filters, verdict.month, verdict, months, months[t - 1], level)
            )
    return found


async def _candidates(
    session: AsyncSession, actor: Actor, dataset: AnalysisDataset, k: float
) -> list[Finding]:
    governed = [m for m in dataset.measures.values() if m.governed]
    splits = [d for d in dataset.dimensions.values() if 2 <= len(d.values) <= MAX_GROUPS][:MAX_SPLITS]
    findings: list[Finding] = []
    for measure in governed:
        ratio = measure.is_ratio
        parts = ["v"] if ratio else None
        rows = await analysis.query_groups(
            session, actor, dataset, {"v": measure.id}, [MONTH_DIMENSION], {}, parts=parts, kind="monitor"
        )
        for points in _series(rows, None).values():
            findings += _judge(dataset, measure.id, {}, points, ratio, k)
        for dim in splits:
            rows = await analysis.query_groups(
                session,
                actor,
                dataset,
                {"v": measure.id},
                [dim.id, MONTH_DIMENSION],
                {},
                parts=parts,
                kind="monitor",
            )
            for group, points in _series(rows, "d0").items():
                findings += _judge(dataset, measure.id, {dim.id: group}, points, ratio, k)
    # One report per group and month (a count and its rate often move together): the strongest wins.
    best: dict[tuple[str, tuple[tuple[str, str], ...], str], Finding] = {}
    for finding in findings:
        z = abs(finding.verdict.z or 0)
        if finding.key not in best or z > abs(best[finding.key].verdict.z or 0):
            best[finding.key] = finding
    return sorted(best.values(), key=lambda f: -abs(f.verdict.z or 0))[:MAX_FINDINGS]


def _wording(finding: Finding) -> tuple[str, str, str]:
    dataset = finding.dataset
    measure = dataset.measures[finding.measure_id]
    verdict = finding.verdict
    scope = " · ".join(finding.filters.values())
    metric = f"{measure.label}{f' · {scope}' if scope else ''}"
    where = f" in {scope}" if scope else ""
    moved = "rose" if verdict.direction == "above" else "fell"
    value = format_value(float(verdict.value or 0), measure.unit)
    expected = format_value(float(verdict.expected or 0), measure.unit)
    message = (
        f"{measure.label}{where} {moved} to {value} in {dataset.month_label(finding.month)}, against an expected "
        f"{expected}: {js_fixed(abs(verdict.z or 0), 1)} robust standard deviations {verdict.direction} it."
    )
    return metric, message, f"{measure.label}{where} was unusual in {dataset.month_label(finding.month)}"


async def scan(
    session: AsyncSession, *, k: float | None = None, dataset_ids: set[str] | None = None
) -> dict[str, Any]:
    """Scan every approved dataset (or the ones named) once, raise alerts for new findings, and report."""
    k = k if k is not None else settings.ANOMALY_Z
    actor = SYSTEM
    raised: list[Alert] = []
    scanned = 0
    for dataset in await catalogue.analysis_catalogue(session, actor):
        if dataset_ids is not None and dataset.id not in dataset_ids:
            continue
        row = await session.get(Dataset, dataset.id)
        if row is None or row.classification == "restricted-personal" or not dataset.has_time:
            continue
        if len(dataset.months) < MIN_MONTHS:
            continue
        scanned += 1
        try:
            findings = await _candidates(session, actor, dataset, k)
        except Exception as error:  # noqa: BLE001 - one dataset failing must not stop the scan
            log.warning("anomaly_scan_dataset_failed", dataset=dataset.id, error=str(error))
            await session.rollback()
            continue
        for finding in findings:
            metric, message, title = _wording(finding)
            exists = await session.execute(
                select(Alert.id).where(
                    Alert.dataset_id == dataset.id,
                    Alert.source == "scan",
                    Alert.month == finding.month,
                    Alert.metric == metric,
                )
            )
            if exists.first() is not None:
                continue
            before = comparison_month(
                finding.months, finding.month, finding.previous, finding.verdict.to_dict()
            )
            cause = await root_cause_for(
                session, actor, dataset, finding.measure_id, finding.filters, before, finding.month
            )
            measure = dataset.measures[finding.measure_id]
            next_dim = next((d for d in dataset.dimensions if d not in finding.filters), None)
            severity = "high" if abs(finding.verdict.z or 0) >= 1.5 * k else "low"
            alert = Alert(
                monitor_id=None,
                dataset_id=dataset.id,
                source="scan",
                metric=metric,
                message=message,
                severity=severity,
                month=finding.month,
                previous_month=before,
                value=finding.verdict.value,
                previous=finding.verdict.expected,
                series=list(finding.verdict.points),
                detector=finding.verdict.to_dict(),
                root_cause=cause,
                explore={
                    "dataset": dataset.id,
                    "method": "explain" if measure.additive else "trend",
                    "measure": measure.id,
                    "filters": finding.filters,
                    **({"dimension": next_dim} if next_dim else {}),
                },
            )
            session.add(alert)
            await session.flush()
            first = cause[0] if cause else None
            session.add(
                Insight(
                    dataset_id=dataset.id,
                    kind="anomaly",
                    title=title[:300],
                    body=message
                    + (f" Largest driver: {first['factor']}, {first['contribution']}." if first else ""),
                    suggestion="Open the alert to see what drove it.",
                    action="alert",
                    target={"alertId": alert.id},
                )
            )
            await audit.record(session, actor, "Detected anomaly", "dataset", dataset.id, metric, message)
            inbox.notify(
                session,
                [row.owner_id, *await inbox.stewards(session)],
                "alert",
                f"Unusual month: {metric}",
                message,
                f"/monitors/{alert.id}",
            )
            raised.append(alert)
    await audit.record(
        session,
        actor,
        SCAN_ACTION,
        "scan",
        None,
        "Anomaly scan",
        f"{scanned} datasets scanned, {len(raised)} new findings",
    )
    await session.commit()
    for alert in raised:
        dataset_row = await session.get(Dataset, alert.dataset_id) if alert.dataset_id else None
        if dataset_row is not None:
            await inbox.email(
                session,
                dataset_row.owner_id,
                f"DataForge: unusual month in {alert.metric}"[:150],
                alert.message,
                f"/monitors/{alert.id}",
            )
    return {"scanned": scanned, "found": [alert_dict(a) for a in raised]}


async def scan_due(session: AsyncSession) -> bool:
    """Whether a scan is due. The append-only audit trail doubles as the scan log."""
    if settings.ANOMALY_SCAN_HOURS <= 0:
        return False
    last = (
        await session.execute(
            select(func.max(AuditEvent.occurred_at)).where(AuditEvent.action == SCAN_ACTION)
        )
    ).scalar()
    return last is None or last + timedelta(hours=settings.ANOMALY_SCAN_HOURS) <= utcnow()


async def feed(session: AsyncSession, actor: Actor, limit: int = 20) -> dict[str, Any]:
    """What the scan has found that this member may see, newest first, with when it last ran."""
    from .monitors import list_alerts

    found = await list_alerts(session, actor, limit=limit, source="scan")
    last = (
        await session.execute(
            select(func.max(AuditEvent.occurred_at)).where(AuditEvent.action == SCAN_ACTION)
        )
    ).scalar()
    return {
        "alerts": found,
        "lastScan": last.isoformat() if last else None,
        "everyHours": settings.ANOMALY_SCAN_HOURS,
        "method": anomaly.METHOD_LABELS["seasonal"],
    }
