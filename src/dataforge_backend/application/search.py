"""Catalogue search across datasets, columns, metrics, dashboards and notebooks, within what a member may see.

PostgreSQL full-text search ranks the matches (with expression indexes created by the migration); each hit
is then checked against the member's visibility rules before it is returned.
"""

from typing import Any

from sqlalchemy import Boolean, Text, cast, func, literal_column, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..infrastructure.db.models import Dashboard, Dataset, MetricDefinition, Notebook
from . import catalogue, dashboards, workbench
from .common import Actor


def _vector(*columns: Any) -> Any:
    joined = func.concat_ws(" ", *[func.coalesce(c, "") for c in columns])
    return func.to_tsvector(literal_column("'english'"), joined)


async def search(session: AsyncSession, actor: Actor, query: str, limit: int = 20) -> list[dict[str, Any]]:
    text = " ".join(query.split())[:200]
    if len(text) < 2:
        return []
    ts_query = func.websearch_to_tsquery(literal_column("'english'"), text)
    prefix = f"%{text}%"
    results: list[dict[str, Any]] = []

    visible = {d.id for d in await catalogue.visible_datasets(session, actor)}
    dataset_vector = _vector(
        Dataset.name,
        Dataset.description,
        Dataset.source_name,
        cast(Dataset.columns, Text),
    )
    rows = await session.execute(
        select(Dataset, func.ts_rank(dataset_vector, ts_query).label("rank"))
        .where(
            or_(dataset_vector.op("@@", return_type=Boolean)(ts_query), Dataset.name.ilike(prefix)),
            Dataset.erased_at.is_(None),
        )
        .order_by(literal_column("rank").desc())
        .limit(limit)
    )
    for dataset, rank in rows:
        if dataset.id not in visible:
            continue
        matched = [c["name"] for c in dataset.columns or [] if text.lower() in c["name"].lower()]
        results.append(
            {
                "type": "dataset",
                "id": dataset.id,
                "title": dataset.name,
                "detail": f"{dataset.layer.capitalize()} · {dataset.status}"
                + (f" · columns: {', '.join(matched[:3])}" if matched else ""),
                "href": f"/datasets/{dataset.id}",
                "rank": float(rank or 0) + (0.5 if text.lower() in dataset.name.lower() else 0),
            }
        )

    metric_vector = _vector(
        MetricDefinition.name,
        MetricDefinition.description,
        MetricDefinition.key,
        cast(MetricDefinition.synonyms, Text),
    )
    rows = await session.execute(
        select(MetricDefinition, func.ts_rank(metric_vector, ts_query).label("rank"))
        .where(
            or_(metric_vector.op("@@", return_type=Boolean)(ts_query), MetricDefinition.name.ilike(prefix))
        )
        .limit(limit)
    )
    for metric, rank in rows:
        if metric.status != "published" and metric.owner_id != actor.id:
            continue
        results.append(
            {
                "type": "metric",
                "id": metric.id,
                "title": metric.name,
                "detail": metric.description,
                "href": f"/semantic#{metric.id}",
                "rank": float(rank or 0) + 0.3,
            }
        )

    visible_dashboards = {d["id"] for d in await dashboards.list_dashboards(session, actor)}
    dashboard_vector = _vector(Dashboard.name, Dashboard.description)
    rows = await session.execute(
        select(Dashboard, func.ts_rank(dashboard_vector, ts_query).label("rank"))
        .where(or_(dashboard_vector.op("@@", return_type=Boolean)(ts_query), Dashboard.name.ilike(prefix)))
        .limit(limit)
    )
    for dashboard, rank in rows:
        if dashboard.id in visible_dashboards:
            results.append(
                {
                    "type": "dashboard",
                    "id": dashboard.id,
                    "title": dashboard.name,
                    "detail": dashboard.description,
                    "href": f"/dashboards/{dashboard.id}",
                    "rank": float(rank or 0),
                }
            )

    visible_notebooks = {n["id"] for n in await workbench.list_notebooks(session, actor)}
    notebook_rows = await session.execute(select(Notebook).where(Notebook.name.ilike(prefix)).limit(limit))
    for notebook in notebook_rows.scalars():
        if notebook.id in visible_notebooks:
            results.append(
                {
                    "type": "notebook",
                    "id": notebook.id,
                    "title": notebook.name,
                    "detail": f"Notebook by {notebook.owner_label}",
                    "href": f"/analyse/code?notebook={notebook.id}",
                    "rank": 0.2,
                }
            )

    results.sort(key=lambda r: -r["rank"])
    return results[:limit]
