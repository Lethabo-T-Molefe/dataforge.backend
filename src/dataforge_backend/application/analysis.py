"""Point-and-click analysis: the fields a member may use, running a method, and saved explorations."""

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.errors import NotFound, Refused, ValidationFailed
from ..domain.analysis import (
    ANALYSIS_METHODS,
    ANALYSIS_MODES,
    AnalysisDataset,
    AnalysisError,
    BuildConfig,
    build_sql,
    config_from_dict,
    default_config,
    interpret,
    plan,
)
from ..domain.analysis.sql import group_query
from ..domain.roles import Permission
from ..domain.sql_guard import QueryRefused, guard_select
from ..infrastructure.db.models import Exploration
from . import audit, catalogue, quotas
from .common import Actor, humanize
from .runtime import sandbox


async def catalogue_for(session: AsyncSession, actor: Actor) -> dict[str, Any]:
    # This lists what a member may analyse. For a role that does not analyse (an auditor) that is nothing,
    # which is an answer rather than an error. Running an analysis is still refused for those roles.
    datasets = await catalogue.analysis_catalogue(session, actor) if actor.can(Permission.ANALYSE) else []
    return {
        "methods": ANALYSIS_METHODS,
        "modes": ANALYSIS_MODES,
        "datasets": [d.to_dict() for d in datasets],
    }


async def dataset_for(session: AsyncSession, actor: Actor, dataset_id: str) -> AnalysisDataset:
    for dataset in await catalogue.analysis_catalogue(session, actor):
        if dataset.id == dataset_id:
            return dataset
    raise NotFound("That dataset is not approved for analysis, or you cannot see it.")


async def defaults(
    session: AsyncSession, actor: Actor, dataset_id: str, method: str, presets: dict[str, str | None]
) -> dict[str, Any]:
    actor.require(Permission.ANALYSE)
    dataset = await dataset_for(session, actor, dataset_id)
    try:
        return default_config(dataset, method, presets).to_dict()
    except AnalysisError as error:
        raise ValidationFailed(str(error)) from error


async def execute(
    session: AsyncSession,
    actor: Actor,
    dataset: AnalysisDataset,
    config: BuildConfig,
    *,
    kind: str = "analysis",
) -> dict[str, Any]:
    """Run one analysis in the sandbox under the member's row policy, then interpret it."""
    dataset_row = await catalogue.policy_for(session, dataset.id, actor.role)
    query = plan(dataset, config, policy=dataset_row)
    result = await sandbox.run_select(
        query.sql, user_id=actor.member_id, kind=kind, params=query.params, max_rows=20_000
    )
    await quotas.add_compute(session, actor.member_id, result.duration_ms)
    rows = [dict(zip(result.columns, row, strict=True)) for row in result.rows]
    output = interpret(dataset, config, rows)
    # The SQL shown is validated like any member query, so opening it in the workbench runs as-is.
    grants = await catalogue.grants_for(session, actor)
    try:
        guard_select(output["sql"], grants)
        output["sqlRunnable"] = True
    except QueryRefused as refusal:
        output["sqlRunnable"] = False
        output["notes"] = [*output["notes"], f"The SQL cannot run in the workbench as-is: {refusal.reason}"]
    output["rowPolicy"] = bool(dataset_row)
    if dataset_row:
        output["notes"] = [
            *output["notes"],
            "Rows your role is not permitted to see were left out before anything was computed.",
        ]
    output["dataset"] = {"id": dataset.id, "name": dataset.name, "table": dataset.table}
    output["measureUnit"] = dataset.measures[config.measure].unit
    return output


async def query_groups(
    session: AsyncSession,
    actor: Actor,
    dataset: AnalysisDataset,
    measures: dict[str, str],
    dims: list[str],
    filters: dict[str, str],
    *,
    date_from: str | None = None,
    date_to: str | None = None,
    pair: tuple[str, str] | None = None,
    parts: list[str] | None = None,
    rate_method: str = "pooled",
    kind: str = "analysis",
) -> list[dict[str, Any]]:
    """Values of measures per group, under the member's row policy. `measures` maps an alias to a measure id.

    Rows carry d0, d1, … for the dimensions, each alias, and <alias>_num / <alias>_den for aliases in `parts`.
    """
    months = dataset.months
    body: dict[str, Any] = {
        "method": "compare",
        "measure": next(iter(measures.values())),
        "filters": filters,
        "from": date_from or (months[0] if months else ""),
        "to": date_to or (months[-1] if months else ""),
        "rateMethod": rate_method,
    }
    if pair:
        body.update(periodA=pair[0], periodB=pair[1])
    try:
        config = config_from_dict(dataset, body)
    except AnalysisError as error:
        raise ValidationFailed(str(error)) from error
    values = {alias: (dataset.measures[m], rate_method) for alias, m in measures.items()}
    part_defs = {alias: dataset.measures[measures[alias]] for alias in parts or []}
    policy = await catalogue.policy_for(session, dataset.id, actor.role)
    query = group_query(
        dataset, config, dims, values, period="pair" if pair else "range", parts=part_defs, policy=policy
    )
    result = await sandbox.run_select(
        query.sql, user_id=actor.member_id, kind=kind, params=query.params, max_rows=50_000
    )
    await quotas.add_compute(session, actor.member_id, result.duration_ms)
    return [dict(zip(result.columns, row, strict=True)) for row in result.rows]


async def run(session: AsyncSession, actor: Actor, body: dict[str, Any]) -> dict[str, Any]:
    actor.require(Permission.ANALYSE)
    actor.require_verified()
    await quotas.check_compute(session, actor)
    dataset_id = str(body.get("datasetId") or body.get("dataset_id") or "")
    dataset = await dataset_for(session, actor, dataset_id)
    try:
        config = config_from_dict(dataset, body)
    except AnalysisError as error:
        raise ValidationFailed(str(error)) from error
    try:
        output = await execute(session, actor, dataset, config)
    except QueryRefused as error:
        raise Refused(error.reason) from error
    await audit.record(
        session,
        actor,
        "Ran analysis",
        "dataset",
        dataset.id,
        dataset.name,
        output["title"],
        {"config": config.to_dict()},
    )
    await session.commit()
    return output


async def save_exploration(
    session: AsyncSession, actor: Actor, name: str, config: dict[str, Any]
) -> dict[str, Any]:
    actor.require(Permission.ANALYSE)
    dataset = await dataset_for(session, actor, str(config.get("datasetId", "")))
    try:
        parsed = config_from_dict(dataset, config)
    except AnalysisError as error:
        raise ValidationFailed(str(error)) from error
    exploration = Exploration(
        owner_id=actor.id,
        name=name.strip()[:200] or "Untitled exploration",
        config=parsed.to_dict(),
        sql=build_sql(dataset, parsed),
    )
    session.add(exploration)
    await session.flush()
    await audit.record(session, actor, "Saved exploration", "exploration", exploration.id, exploration.name)
    await session.commit()
    return exploration_dict(exploration)


def exploration_dict(e: Exploration) -> dict[str, Any]:
    return {
        "id": e.id,
        "name": e.name,
        "config": e.config,
        "sql": e.sql,
        "updated": humanize(e.updated_at),
        "updatedAt": e.updated_at.isoformat(),
    }


async def list_explorations(session: AsyncSession, actor: Actor) -> list[dict[str, Any]]:
    rows = await session.execute(
        select(Exploration).where(Exploration.owner_id == actor.id).order_by(Exploration.updated_at.desc())
    )
    return [exploration_dict(e) for e in rows.scalars()]


async def delete_exploration(session: AsyncSession, actor: Actor, exploration_id: str) -> None:
    exploration = await session.get(Exploration, exploration_id)
    if exploration is None or exploration.owner_id != actor.id:
        raise NotFound("Exploration not found.")
    await session.delete(exploration)
    await audit.record(session, actor, "Deleted exploration", "exploration", exploration_id, exploration.name)
    await session.commit()
