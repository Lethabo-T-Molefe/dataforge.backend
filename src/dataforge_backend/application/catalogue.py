"""The catalogue: what each member may query, and the analysis fields each approved dataset offers.

Fields (dimensions, time column, months) are computed from the warehouse table when a dataset is approved
or refreshed and cached on the dataset. Measures are derived at request time from the published metric
definitions, so publishing a metric makes it available everywhere at once.
"""

from dataclasses import replace
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from ..domain.analysis import AnalysisDataset
from ..domain.semantic import ColumnInfo, MetricSpec, build_analysis_dataset, derive_fields
from ..domain.sql_guard import TableGrant
from ..infrastructure import warehouse
from ..infrastructure.db.models import Dataset, MetricDefinition, RowPolicy
from .common import Actor, can_see_dataset, is_queryable, shared_ids


def metric_spec(metric: MetricDefinition) -> MetricSpec:
    return MetricSpec(
        key=metric.key,
        name=metric.name,
        unit=metric.unit,
        kind=metric.kind,
        description=metric.description,
        column=metric.column,
        numerator=metric.numerator,
        denominator=metric.denominator,
        synonyms=tuple(metric.synonyms or []),
        metric_id=metric.id,
    )


async def published_metrics(session: AsyncSession) -> list[MetricSpec]:
    rows = await session.execute(
        select(MetricDefinition)
        .where(MetricDefinition.status == "published")
        .order_by(MetricDefinition.created_at)
    )
    return [metric_spec(m) for m in rows.scalars()]


async def compute_fields(conn: AsyncConnection, table: str, pii: dict[str, str]) -> dict[str, Any]:
    """Column statistics and months for a warehouse table, cached as JSON on the dataset."""
    infos, rows = await warehouse.column_infos(conn, table, pii)
    probe = derive_fields(infos, rows, [])
    months: list[str] = []
    if probe.time_column and probe.time_kind:
        months = await warehouse.distinct_months(conn, table, probe.time_column, probe.time_kind)
    return {
        "table": table,
        "rows": rows,
        "columns": [
            {
                "name": i.name,
                "type": i.type,
                "distinct": i.distinct,
                "pii": i.pii,
                "values": list(i.values),
                "monthLike": i.month_like,
            }
            for i in infos
        ],
        "months": months,
    }


def column_infos(fields: dict[str, Any]) -> list[ColumnInfo]:
    return [
        ColumnInfo(
            c["name"],
            c["type"],
            int(c["distinct"]),
            bool(c["pii"]),
            tuple(c.get("values") or ()),
            bool(c.get("monthLike")),
        )
        for c in fields.get("columns", [])
    ]


def analysis_dataset(dataset: Dataset, metrics: list[MetricSpec]) -> AnalysisDataset | None:
    if not dataset.fields or not is_queryable(dataset):
        return None
    fields = dataset.fields
    derived = derive_fields(column_infos(fields), int(fields.get("rows", 0)), metrics)
    if not derived.measures:
        return None
    built = build_analysis_dataset(
        dataset.id, dataset.name, fields["table"], derived, list(fields.get("months") or [])
    )
    if dataset.default_measure and dataset.default_measure in built.measures:
        built = replace(built, default_measure=dataset.default_measure)
    return built


async def visible_datasets(session: AsyncSession, actor: Actor) -> list[Dataset]:
    shared = await shared_ids(session, actor, "dataset")
    rows = await session.execute(
        select(Dataset).where(Dataset.erased_at.is_(None)).order_by(Dataset.updated_at.desc())
    )
    return [d for d in rows.scalars() if can_see_dataset(actor, d, shared)]


async def queryable_datasets(session: AsyncSession, actor: Actor) -> list[Dataset]:
    return [d for d in await visible_datasets(session, actor) if is_queryable(d)]


async def policy_for(session: AsyncSession, dataset_id: str, role: str) -> str | None:
    row = await session.execute(
        select(RowPolicy.predicate).where(RowPolicy.dataset_id == dataset_id, RowPolicy.role == role)
    )
    return row.scalar_one_or_none()


async def grants_for(session: AsyncSession, actor: Actor) -> dict[str, TableGrant]:
    """The tables a member's SQL may read, with masked columns blocked and their row policy attached."""
    datasets = await queryable_datasets(session, actor)
    policies = {
        p.dataset_id: p.predicate
        for p in (await session.execute(select(RowPolicy).where(RowPolicy.role == actor.role))).scalars()
    }
    grants: dict[str, TableGrant] = {}
    for dataset in datasets:
        table = dataset.qualified_table()
        if table:
            grants[table] = TableGrant(table, frozenset(dataset.pii_columns or {}), policies.get(dataset.id))
    return grants


async def analysis_catalogue(session: AsyncSession, actor: Actor) -> list[AnalysisDataset]:
    metrics = await published_metrics(session)
    out = []
    for dataset in await queryable_datasets(session, actor):
        built = analysis_dataset(dataset, metrics)
        if built is not None:
            out.append(built)
    return out
