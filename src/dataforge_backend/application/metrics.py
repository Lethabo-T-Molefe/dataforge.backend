"""The semantic layer's shared metrics: define, publish, find where they are used, and headline figures."""

import asyncio
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.errors import Conflict, Forbidden, NotFound, ValidationFailed
from ..core.security import utcnow
from ..domain.analysis import AnalysisDataset, AnalysisError, BuildConfig, MeasureDef, config_from_dict
from ..domain.analysis.format import format_delta, format_percent, format_value
from ..domain.roles import Permission
from ..domain.semantic import MetricError, MetricSpec, expression_for, validate_metric
from ..domain.sql_guard import TableGrant
from ..infrastructure.db.models import DashboardChart, MetricDefinition, ModelVersion, Monitor
from . import analysis, audit, catalogue, integrations, quotas
from .common import Actor, slugify

GRAIN_DEFAULT = "dataset rows"


async def _used_by(session: AsyncSession, key: str) -> int:
    charts = await session.execute(
        select(func.count()).select_from(DashboardChart).where(DashboardChart.config["measure"].astext == key)
    )
    monitors = await session.execute(select(func.count()).select_from(Monitor).where(Monitor.measure == key))
    models = 0
    factor_lists: list[list[dict[str, Any]]] = list(
        (await session.execute(select(ModelVersion.factors).where(ModelVersion.status == "active"))).scalars()
    )
    for factors in factor_lists:
        models += sum(1 for f in factors if f.get("measure") == key)
    return int(charts.scalar() or 0) + int(monitors.scalar() or 0) + models


async def definition_dict(
    session: AsyncSession, metric: MetricDefinition, applicable: list[str] | None = None
) -> dict[str, Any]:
    spec = catalogue.metric_spec(metric)
    return {
        "id": metric.id,
        "key": metric.key,
        "name": metric.name,
        "owner": metric.owner_label,
        "ownerId": metric.owner_id,
        "expression": expression_for(spec),
        "description": metric.description,
        "usedBy": await _used_by(session, metric.key),
        "grain": metric.grain or GRAIN_DEFAULT,
        "status": metric.status,
        "unit": metric.unit,
        "kind": metric.kind,
        "column": metric.column,
        "numerator": metric.numerator,
        "denominator": metric.denominator,
        "synonyms": metric.synonyms or [],
        "version": metric.version,
        "datasets": applicable or [],
        "publishedAt": metric.published_at.isoformat() if metric.published_at else None,
    }


async def list_definitions(session: AsyncSession, actor: Actor) -> list[dict[str, Any]]:
    rows = (
        (
            await session.execute(
                select(MetricDefinition).order_by(MetricDefinition.status.desc(), MetricDefinition.name)
            )
        )
        .scalars()
        .all()
    )
    # Drafts are visible to their owner and to those who can publish.
    visible = [
        m
        for m in rows
        if m.status == "published" or m.owner_id == actor.id or actor.can(Permission.PUBLISH_METRICS)
    ]
    datasets = await catalogue.queryable_datasets(session, actor)
    out = []
    for metric in visible:
        spec = catalogue.metric_spec(metric)
        applicable = []
        for dataset in datasets:
            fields = dataset.fields or {}
            types = {c["name"]: c["type"] for c in fields.get("columns", [])}
            try:
                validate_metric(spec, types)
                applicable.append(dataset.id)
            except MetricError:
                continue
        out.append(await definition_dict(session, metric, applicable))
    return out


def _spec_from(body: dict[str, Any], key: str) -> MetricSpec:
    return MetricSpec(
        key=key,
        name=str(body.get("name", "")).strip(),
        unit=str(body.get("unit", "count")),
        kind=str(body.get("kind", "additive")),
        description=str(body.get("description", "")).strip(),
        column=body.get("column") or None,
        numerator=body.get("numerator") or None,
        denominator=body.get("denominator") or None,
        synonyms=tuple(str(s).strip() for s in body.get("synonyms") or [] if str(s).strip()),
    )


async def create(session: AsyncSession, actor: Actor, body: dict[str, Any]) -> dict[str, Any]:
    actor.require(Permission.DEFINE_METRICS, "Your role cannot define metrics.")
    key = str(body.get("key") or slugify(str(body.get("name", "")), 60).replace("-", "_"))
    try:
        spec = validate_metric(_spec_from(body, key))
    except MetricError as error:
        raise ValidationFailed(str(error)) from error
    if (
        await session.execute(select(MetricDefinition).where(MetricDefinition.key == key))
    ).scalar_one_or_none():
        raise Conflict(f"A metric with the key {key} already exists. Metrics are defined once.")
    metric_id = slugify(spec.name, 60)
    n = 2
    while await session.get(MetricDefinition, metric_id) is not None:
        metric_id, n = f"{slugify(spec.name, 55)}-{n}", n + 1
    metric = MetricDefinition(
        id=metric_id,
        key=key,
        name=spec.name,
        description=spec.description,
        owner_id=actor.id,
        owner_label=actor.name,
        unit=spec.unit,
        kind=spec.kind,
        column=spec.column,
        numerator=spec.numerator,
        denominator=spec.denominator,
        synonyms=list(spec.synonyms),
        grain=str(body.get("grain") or GRAIN_DEFAULT),
        status="draft",
    )
    session.add(metric)
    await audit.record(
        session, actor, "Defined metric", "metric", metric.id, metric.name, expression_for(spec)
    )
    await session.commit()
    return await definition_dict(session, metric)


async def update(session: AsyncSession, actor: Actor, metric_id: str, body: dict[str, Any]) -> dict[str, Any]:
    metric = await session.get(MetricDefinition, metric_id)
    if metric is None:
        raise NotFound("Metric not found.")
    if metric.status == "published":
        actor.require(
            Permission.PUBLISH_METRICS,
            "A published metric can only be changed by a Data Steward, because every answer uses it.",
        )
    elif metric.owner_id != actor.id and not actor.can(Permission.PUBLISH_METRICS):
        raise Forbidden("Only the metric's owner or a steward can change it.")
    merged = {**(await definition_dict(session, metric)), **body}
    try:
        spec = validate_metric(_spec_from(merged, metric.key))
    except MetricError as error:
        raise ValidationFailed(str(error)) from error
    before = expression_for(catalogue.metric_spec(metric))
    metric.name, metric.description, metric.unit, metric.kind = (
        spec.name,
        spec.description,
        spec.unit,
        spec.kind,
    )
    metric.column, metric.numerator, metric.denominator = spec.column, spec.numerator, spec.denominator
    metric.synonyms = list(spec.synonyms)
    metric.grain = str(merged.get("grain") or metric.grain)
    metric.version += 1
    await audit.record(
        session,
        actor,
        "Changed metric",
        "metric",
        metric.id,
        metric.name,
        f"{before} → {expression_for(spec)} (v{metric.version})",
    )
    await session.commit()
    return await definition_dict(session, metric)


async def set_status(session: AsyncSession, actor: Actor, metric_id: str, status: str) -> dict[str, Any]:
    actor.require(Permission.PUBLISH_METRICS, "Only a Data Steward can publish metrics.")
    metric = await session.get(MetricDefinition, metric_id)
    if metric is None:
        raise NotFound("Metric not found.")
    if status not in ("published", "draft"):
        raise ValidationFailed("Status is published or draft.")
    metric.status = status
    if status == "published":
        metric.published_by, metric.published_at = actor.id, utcnow()
    await audit.record(
        session,
        actor,
        "Published metric" if status == "published" else "Unpublished metric",
        "metric",
        metric.id,
        metric.name,
    )
    await session.commit()
    integrations.in_background(integrations.sync_all_quietly())
    return await definition_dict(session, metric)


async def delete(session: AsyncSession, actor: Actor, metric_id: str) -> None:
    metric = await session.get(MetricDefinition, metric_id)
    if metric is None:
        raise NotFound("Metric not found.")
    if metric.status == "published":
        raise Conflict("Unpublish the metric before deleting it.")
    if metric.owner_id != actor.id and not actor.can(Permission.PUBLISH_METRICS):
        raise Forbidden("Only the owner or a steward can delete this metric.")
    if await _used_by(session, metric.key):
        raise Conflict("Dashboards, monitors or models still use this metric.")
    await session.delete(metric)
    await audit.record(session, actor, "Deleted metric", "metric", metric_id, metric.name)
    await session.commit()


async def summaries(session: AsyncSession, actor: Actor, limit: int = 4) -> list[dict[str, Any]]:
    """Headline figures: the latest month of each shared metric against the month before.

    The sandbox queries for the next few candidates run at once, and grants and row policies are read once,
    because each query is a network round trip to the database. Candidates are still taken in order, so the
    figures are the same as running them one by one.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    candidates: list[tuple[AnalysisDataset, MeasureDef, BuildConfig]] = []
    for dataset in await catalogue.analysis_catalogue(session, actor):
        if not dataset.has_time or len(dataset.months) < 2:
            continue
        for measure in dataset.measures.values():
            if not measure.governed:
                continue
            try:
                config = config_from_dict(
                    dataset,
                    {
                        "method": "trend",
                        "measure": measure.id,
                        "from": dataset.months[-2],
                        "to": dataset.months[-1],
                    },
                )
            except AnalysisError:
                continue
            candidates.append((dataset, measure, config))
    if not candidates:
        await session.commit()
        return out

    policies = await catalogue.policies_for_role(session, actor.role)
    grants: dict[str, TableGrant] | None = None
    definitions: dict[str, MetricDefinition] = {}
    compute_ms: int | None = None
    queue = list(candidates)
    while queue and len(out) < limit:
        batch: list[tuple[AnalysisDataset, MeasureDef, BuildConfig]] = []
        later: list[tuple[AnalysisDataset, MeasureDef, BuildConfig]] = []
        while queue and len(batch) < limit - len(out):
            candidate = queue.pop(0)
            if candidate[1].id in seen:
                continue
            if any(candidate[1].id == b[1].id for b in batch):
                later.append(candidate)  # used only if the one already in this batch gives no figure
            else:
                batch.append(candidate)
        queue = later + queue
        if not batch:
            break
        runs = asyncio.gather(
            *(analysis.run_sandboxed(actor, d, c, policies.get(d.id), kind="analysis") for d, _m, c in batch),
            return_exceptions=True,
        )
        # The sandbox uses its own connections, so the session reads what is left while the queries run.
        if grants is None:
            grants = await catalogue.grants_for(session, actor, policies=policies)
            published = select(MetricDefinition).where(MetricDefinition.status == "published")
            definitions = {m.key: m for m in (await session.execute(published)).scalars()}
        results = await runs
        for (dataset, measure, config), run in zip(batch, results, strict=True):
            if len(out) >= limit or measure.id in seen:
                continue
            if isinstance(run, AnalysisError):
                continue
            if isinstance(run, BaseException):
                raise run
            compute_ms = (compute_ms or 0) + run.duration_ms  # counted as before: once the sandbox ran
            try:
                result = await analysis.finish(
                    session,
                    actor,
                    dataset,
                    config,
                    policies.get(dataset.id),
                    run,
                    grants,
                    count_compute=False,
                )
            except AnalysisError:
                continue
            values = [p.get(measure.label) for p in result["visual"]["points"]]
            if len(values) < 2 or not all(isinstance(v, int | float) for v in values[-2:]):
                continue
            a, b = float(values[-2]), float(values[-1])
            delta = (
                format_delta(b - a, measure.unit)
                if measure.is_ratio
                else format_percent((b - a) / abs(a) * 100 if a else float("nan"))
            )
            definition = definitions.get(measure.id)
            out.append(
                {
                    "name": measure.label,
                    "key": measure.id,
                    "value": format_value(b, measure.unit),
                    "delta": delta,
                    "trend": "up" if b >= a else "down",
                    "definition": definition.description if definition else measure.description,
                    "month": dataset.month_label(dataset.months[-1]),
                    "datasetId": dataset.id,
                }
            )
            seen.add(measure.id)
    if compute_ms is not None:
        await quotas.add_compute(session, actor.member_id, compute_ms)
    await session.commit()
    return out
