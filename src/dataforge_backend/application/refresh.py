"""Refreshing connected data: re-read the source, rerun the saved recipe, and keep the dataset current.

A refresh lands the source's current rows as a new raw version and reruns the dataset's latest applied
cleaning recipe on them. An approved dataset keeps its approval only when the cleaned result has the same
columns and types as before and no new personal information; anything else goes back to a steward, who
is told. Each dataset refreshes in one transaction, so readers see the old data or the new, never a gap.

Refreshes run as the dataset's owner, so the owner's quotas apply, and on the schedule a steward sets on
the connector (or when someone presses "Sync now").
"""

import asyncio
from datetime import timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.errors import AppError, Forbidden
from ..core.logging import get_logger
from ..core.security import utcnow
from ..domain.governance import DatasetStatus, Layer
from ..domain.roles import Permission
from ..infrastructure import warehouse
from ..infrastructure.db.models import Connector, Dataset, DatasetVersion, User
from . import audit, cleaning, connectors, datasets, inbox
from .common import SYSTEM, Actor, is_queryable

log = get_logger(__name__)


def _outcome(dataset: Dataset, status: str, detail: str) -> dict[str, Any]:
    return {"datasetId": dataset.id, "name": dataset.name, "status": status, "detail": detail}


async def refresh_dataset(session: AsyncSession, connector: Connector, dataset: Dataset) -> dict[str, Any]:
    """Refresh one dataset and commit. Statuses: current | review | raw | skipped."""
    owner = await session.get(User, dataset.owner_id) if dataset.owner_id else None
    if owner is None or owner.status != "active":
        return _outcome(dataset, "skipped", "The dataset has no active owner to refresh it as.")
    actor = Actor.from_user(owner)
    conn = await session.connection()
    was_approved = is_queryable(dataset)
    was_gold = dataset.layer == Layer.GOLD.value
    silver = f"silver.{dataset.table_name}"
    gold = f"gold.{dataset.table_name}"
    types_before = await warehouse.column_types(conn, silver) if was_approved else {}
    gold_columns = list(await warehouse.column_types(conn, gold)) if was_gold else None
    personal_before = set(dataset.pii_columns or {})
    decided_by, decided_at = dataset.decided_by, dataset.decided_at
    recipe = await cleaning.latest_applied(session, dataset)

    frame = await connectors.read_source(connector, dataset.source_name.rsplit(" · ", 1)[-1], actor)
    checksum = await asyncio.to_thread(cleaning.frame_checksum, frame)
    landed = await session.execute(
        select(DatasetVersion.checksum).where(
            DatasetVersion.dataset_id == dataset.id,
            DatasetVersion.version == dataset.current_version,
            DatasetVersion.layer == "bronze",
        )
    )
    if checksum == landed.scalar_one_or_none():
        # Nothing changed at the source: no new version, so raw history only grows when the data does.
        return _outcome(dataset, "unchanged", "The source has not changed since it was last read.")
    await datasets.ingest_frame(
        session,
        actor,
        frame,
        name=dataset.name,
        description=dataset.description,
        source="connector",
        source_name=dataset.source_name,
        classification=dataset.classification,
        visibility=dataset.visibility,
        connector_id=connector.id,
        dataset_id=dataset.id,
        source_checksum=checksum,
    )
    if recipe is None:
        inbox.notify(
            session,
            [dataset.owner_id],
            "refresh",
            f"{dataset.name} has new raw data",
            f"{connector.name} was read again. It has no saved recipe yet, so clean it before it can be analysed.",
            f"/clean/{dataset.id}",
        )
        await session.commit()
        return _outcome(dataset, "raw", "New raw version; no recipe to rerun.")

    copied = await cleaning.copy_applied(session, actor, dataset, recipe)
    result = await cleaning.apply_recipe_to_new_version(session, actor, dataset, copied)
    personal_now = set(dataset.pii_columns or {})
    reasons = []
    if not was_approved:
        reasons.append("it was not approved before")
    elif result["types"] != types_before:
        reasons.append("its columns or their types changed")
    if personal_now - personal_before:
        reasons.append(
            f"new personal information was found ({', '.join(sorted(personal_now - personal_before))})"
        )

    if not reasons:
        # Same recipe, same shape, nothing new to protect: the steward's decision still describes this data.
        await warehouse.grant_sandbox(conn, silver)
        dataset.status = DatasetStatus.APPROVED.value
        dataset.decided_by, dataset.decided_at = decided_by, decided_at
        dataset.decision_reason = (
            "Kept on refresh: same recipe, same columns and types, no new personal information"
        )
        await datasets.refresh_fields(session, dataset)
        if was_gold:
            await datasets.write_gold(session, dataset, gold_columns, None)
        await audit.record(
            session,
            SYSTEM,
            "Kept approval on refresh",
            "dataset",
            dataset.id,
            dataset.name,
            f"{result['rows']} rows from {connector.source_path}",
            {"checksum": result["checksum"]},
        )
        status, detail = "current", f"{result['rows']} rows; approval kept."
    else:
        why = " and ".join(reasons)
        inbox.notify(
            session,
            [*await inbox.stewards(session), dataset.owner_id],
            "refresh",
            f"{dataset.name} needs review after a refresh",
            f"It was refreshed from {connector.name}, but {why}, so it waits for a steward's decision.",
            f"/datasets/{dataset.id}",
        )
        status, detail = "review", f"Waiting for review: {why}."
    await session.commit()
    datasets.sync_superset_later(dataset.id)
    return _outcome(dataset, status, detail)


async def refresh_connector(session: AsyncSession, connector: Connector) -> list[dict[str, Any]]:
    """Refresh every dataset read from a connector. One failing dataset does not stop the others."""
    connector_id = connector.id
    outcomes = []
    for dataset_id in [d.id for d in await connectors.datasets_of(session, connector)]:
        dataset = await session.get(Dataset, dataset_id)
        if dataset is None:
            continue
        name, owner_id = dataset.name, dataset.owner_id
        try:
            connector = await connectors.get(session, connector_id)
            outcomes.append(await refresh_dataset(session, connector, dataset))
        except Exception as error:  # noqa: BLE001 - the failure is recorded and reported instead
            # Rolling back expires every loaded object, so only the ids captured above are used after it.
            await session.rollback()
            message = error.message if isinstance(error, AppError) else "An unexpected error stopped it."
            log.warning("refresh_failed", connector=connector_id, dataset=dataset_id, error=str(error))
            connector = await connectors.get(session, connector_id)
            connector.last_error = f"{name}: {message}"[:2000]
            inbox.notify(
                session,
                [*await inbox.stewards(session), owner_id],
                "refresh",
                f"Refreshing {name} failed",
                f"{connector.name}: {message} The data stays as it was.",
                f"/datasets/{dataset_id}",
            )
            await session.commit()
            outcomes.append({"datasetId": dataset_id, "name": name, "status": "failed", "detail": message})
    connector = await connectors.get(session, connector_id)
    connector.last_sync_at = utcnow()
    if all(o["status"] != "failed" for o in outcomes):
        connector.last_error = None
    await audit.record(
        session,
        SYSTEM,
        "Refreshed connector",
        "connector",
        connector.id,
        connector.source_path,
        ", ".join(f"{o['name']}: {o['status']}" for o in outcomes) or "No datasets read from it",
    )
    await session.commit()
    return outcomes


async def sync(session: AsyncSession, actor: Actor, connector_id: str) -> list[dict[str, Any]]:
    """Refresh a connector now. Its requester or a steward may ask."""
    connector = await connectors.get(session, connector_id)
    if connector.owner_id != actor.id and not actor.can(Permission.MANAGE_CONNECTORS):
        raise Forbidden("Only the connector's requester or a steward can refresh it.")
    if connector.status != "connected":
        raise Forbidden("This connector is not connected yet. A steward approves it first.")
    await audit.record(
        session, actor, "Asked to refresh connector", "connector", connector.id, connector.source_path
    )
    await session.commit()
    return await refresh_connector(session, connector)


async def refresh_due(session: AsyncSession) -> list[dict[str, Any]]:
    """Refresh every connector whose schedule says it is due."""
    now = utcnow()
    rows = await session.execute(
        select(Connector).where(Connector.refresh_cadence.is_not(None), Connector.status == "connected")
    )
    results = []
    for connector in rows.scalars().all():
        interval = timedelta(seconds=connectors.REFRESH_CADENCES.get(connector.refresh_cadence or "", 86_400))
        if connector.last_sync_at is None or connector.last_sync_at + interval <= now:
            results.append(
                {"connectorId": connector.id, "datasets": await refresh_connector(session, connector)}
            )
    return results
