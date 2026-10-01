"""Datasets: ingestion, the governed journey (bronze → silver → gold), sharing and erasure."""

import hashlib
from typing import Any

import pandas as pd
from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..core.errors import Conflict, Forbidden, NotFound, ValidationFailed
from ..core.logging import get_logger
from ..core.security import utcnow
from ..domain.analysis.sql import ident
from ..domain.governance import (
    CLASSIFICATION_LABELS,
    LAYER_LABELS,
    Classification,
    DatasetSource,
    DatasetStatus,
    Layer,
    Visibility,
    journey,
    next_step,
)
from ..domain.insights import detect_insights
from ..domain.pii import PiiKind, detect_pii, mask_value
from ..domain.profiling import column_summaries, profile_frame
from ..domain.roles import Permission, Role
from ..infrastructure import storage, warehouse
from ..infrastructure.db.models import (
    Dataset,
    DatasetVersion,
    Insight,
    LineageEdge,
    RowPolicy,
    Share,
    User,
)
from . import audit, catalogue, integrations, quotas
from .common import SYSTEM, Actor, can_see_dataset, humanize, is_queryable, shared_ids, slugify, snake
from .runtime import sandbox

log = get_logger(__name__)


# ------------------------------------------------------------------ shapes


def summary(dataset: Dataset, actor: Actor) -> dict[str, Any]:
    step = next_step(Layer(dataset.layer), DatasetStatus(dataset.status), Role(actor.role))
    return {
        "id": dataset.id,
        "name": dataset.name,
        "description": dataset.description,
        "owner": dataset.owner_label or "Unknown",
        "ownerId": dataset.owner_id,
        "source": dataset.source,
        "sourceName": dataset.source_name,
        "layer": dataset.layer,
        "layerLabel": LAYER_LABELS[Layer(dataset.layer)],
        "classification": dataset.classification,
        "classificationLabel": CLASSIFICATION_LABELS[Classification(dataset.classification)],
        "status": dataset.status,
        "visibility": dataset.visibility,
        "rows": dataset.rows,
        "version": dataset.current_version,
        "table": dataset.qualified_table(),
        "queryable": is_queryable(dataset),
        "updated": humanize(dataset.updated_at),
        "updatedAt": dataset.updated_at.isoformat(),
        "quality": dataset.quality or {},
        "columns": dataset.columns or [],
        "piiColumns": dataset.pii_columns or {},
        "insights": [insight_dict(i) for i in dataset.insights if i.status == "open"],
        "nextStep": {"kind": step.kind, "label": step.label, "reason": step.reason},
        "journey": journey(
            Layer(dataset.layer), DatasetStatus(dataset.status), DatasetSource(dataset.source)
        ),
        "decision": (
            {
                "reason": dataset.decision_reason,
                "at": dataset.decided_at.isoformat() if dataset.decided_at else None,
            }
            if dataset.decided_at
            else None
        ),
    }


def insight_dict(insight: Insight) -> dict[str, Any]:
    return {
        "id": insight.id,
        "kind": insight.kind,
        "title": insight.title,
        "body": insight.body,
        "suggestion": insight.suggestion,
        "action": insight.action,
        "target": insight.target,
        "status": insight.status,
    }


# ------------------------------------------------------------------ lookup


async def get_visible(session: AsyncSession, actor: Actor, dataset_id: str) -> Dataset:
    dataset = await session.get(Dataset, dataset_id)
    if dataset is None or not can_see_dataset(actor, dataset, await shared_ids(session, actor, "dataset")):
        raise NotFound("Dataset not found.")
    await session.refresh(dataset, ["insights"])
    return dataset


async def list_datasets(session: AsyncSession, actor: Actor) -> list[dict[str, Any]]:
    datasets = await catalogue.visible_datasets(session, actor)
    for d in datasets:
        await session.refresh(d, ["insights"])
    return [summary(d, actor) for d in datasets]


async def _unique_id(session: AsyncSession, name: str) -> tuple[str, str]:
    base = slugify(name, 60)
    table_base = snake(name, 50)
    candidate, table, n = base, table_base, 2
    while True:
        taken = await session.execute(
            select(func.count())
            .select_from(Dataset)
            .where((Dataset.id == candidate) | (Dataset.table_name == table))
        )
        if not taken.scalar():
            return candidate, table
        candidate, table, n = f"{base[:55]}-{n}", f"{table_base[:45]}_{n}", n + 1


async def record_version(session: AsyncSession, dataset: Dataset, layer: str, **fields: Any) -> None:
    """The table a layer holds for the current raw version. Rewriting a layer (a rerun) updates its record."""
    existing = (
        await session.execute(
            select(DatasetVersion).where(
                DatasetVersion.dataset_id == dataset.id,
                DatasetVersion.version == dataset.current_version,
                DatasetVersion.layer == layer,
            )
        )
    ).scalar_one_or_none()
    if existing is None:
        session.add(
            DatasetVersion(dataset_id=dataset.id, version=dataset.current_version, layer=layer, **fields)
        )
        return
    for name, value in fields.items():
        setattr(existing, name, value)
    existing.created_at = utcnow()


# ------------------------------------------------------------------ ingestion


def _analyse_frame(
    frame: pd.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str], list[Any]]:
    """Profile, PII scan and insights. CPU-bound: runs in the Python sandbox."""
    findings = detect_pii(frame)
    profile, facts = profile_frame(frame, findings)
    columns = column_summaries(frame, profile)
    insights = detect_insights(frame, profile, facts)
    pii = {f.column: f.kind.value for f in findings}
    return profile, columns, pii, insights


async def ingest_frame(
    session: AsyncSession,
    actor: Actor,
    frame: pd.DataFrame,
    *,
    name: str,
    description: str,
    source: str,
    source_name: str,
    classification: str,
    visibility: str,
    connector_id: str | None = None,
    landing_file: tuple[str, bytes] | None = None,
    dataset_id: str | None = None,
    preferred_id: str | None = None,
    table_name: str | None = None,
) -> Dataset:
    """Land raw data in bronze, scan it for personal information and profile it. A new version when re-ingested."""
    if frame.empty:
        raise ValidationFailed("There are no rows to bring in.")
    try:
        chosen = Classification(classification)
        wanted_visibility = Visibility(visibility)
    except ValueError as error:
        raise ValidationFailed("Choose a valid classification and visibility.") from error
    frame = warehouse.normalise_columns(frame.astype(object))

    existing = await session.get(Dataset, dataset_id) if dataset_id else None
    if existing is None:
        new_id, generated_table = await _unique_id(session, preferred_id or name)
        table_name = table_name or generated_table
        version = 1
    else:
        new_id, table_name, version = existing.id, existing.table_name, existing.current_version + 1

    (profile, columns, pii, insights), elapsed = await sandbox.run_python(
        lambda: _analyse_frame(frame),
        user_id=actor.member_id,
        kind="profile",
        description=f"Profile and PII scan of {source_name}",
    )
    await quotas.add_compute(session, actor.member_id, elapsed)

    # Classification is mandatory and never lower than what the data holds.
    notes: list[str] = []
    if pii and chosen != Classification.RESTRICTED_PERSONAL:
        notes.append(
            f"Classified as restricted-personal because it holds personal information ({', '.join(pii)})."
        )
        chosen = Classification.RESTRICTED_PERSONAL

    landing: tuple[str, str] | None = None
    if landing_file is not None:
        path, checksum = storage.store(new_id, version, landing_file[0], landing_file[1])
        landing = (str(path), checksum)

    conn = await session.connection()
    bronze_table = f"{table_name}_v{version}"
    rows = await warehouse.write_bronze(conn, bronze_table, frame)
    size = await warehouse.table_size(conn, f"bronze.{bronze_table}")

    if existing is None:
        dataset = Dataset(
            id=new_id,
            name=name,
            description=description,
            owner_id=actor.id if actor.id != "system" else None,
            owner_label=actor.name,
            source=source,
            source_name=source_name,
            connector_id=connector_id,
            layer=Layer.BRONZE.value,
            classification=chosen.value,
            status=DatasetStatus.PENDING.value,
            visibility=wanted_visibility.value,
            team=actor.team,
            table_name=table_name,
            current_version=version,
        )
        session.add(dataset)
    else:
        dataset = existing
        dataset.current_version = version
        dataset.layer = Layer.BRONZE.value
        dataset.status = DatasetStatus.PENDING.value
        dataset.classification = chosen.value
        dataset.decided_at = None
        dataset.decided_by = None
        dataset.decision_reason = None
        await session.execute(
            delete(Insight).where(Insight.dataset_id == dataset.id, Insight.status == "open")
        )
    dataset.rows = rows
    dataset.size_bytes = size
    dataset.profile = profile
    dataset.quality = profile["quality"]
    dataset.columns = columns
    dataset.pii_columns = pii
    dataset.fields = None
    await session.flush()

    session.add(
        DatasetVersion(
            dataset_id=dataset.id,
            version=version,
            layer="bronze",
            table=f"bronze.{bronze_table}",
            rows=rows,
            column_types={c: "text" for c in frame.columns},
            source_file=landing[0] if landing else None,
            checksum=landing[1] if landing else None,
            created_by=actor.id if actor.id != "system" else None,
        )
    )
    for insight in insights:
        session.add(
            Insight(
                dataset_id=dataset.id,
                kind=insight.kind,
                title=insight.title,
                body=insight.body,
                suggestion=insight.suggestion,
                action=insight.action,
                target={"value": insight.target} if insight.target else None,
            )
        )
    session.add(
        LineageEdge(
            dataset_id=dataset.id,
            source=f"{source}:{source_name}",
            target=f"bronze.{bronze_table}",
            transform=source,
            detail=f"{rows} rows landed as raw text (version {version})",
            actor_id=actor.id if actor.id != "system" else None,
        )
    )
    verb = "Uploaded dataset" if source == "upload" else "Imported from connector"
    await audit.record(
        session,
        actor,
        verb,
        "dataset",
        dataset.id,
        source_name,
        f"{rows:,} rows landed in bronze".replace(",", " "),
    )
    await audit.record(
        session,
        SYSTEM,
        "Profiled dataset",
        "dataset",
        dataset.id,
        name,
        f"Completeness {profile['quality']['completeness']}%, conformance {profile['quality']['conformance']}%",
    )
    if pii:
        await audit.record(
            session,
            SYSTEM,
            "Masked columns",
            "dataset",
            dataset.id,
            name,
            f"{', '.join(pii)} detected as personal information",
            {"pii": pii},
        )
    for note in notes:
        await audit.record(session, SYSTEM, "Raised classification", "dataset", dataset.id, name, note)
    return dataset


async def upload(
    session: AsyncSession,
    actor: Actor,
    *,
    filename: str,
    content: bytes,
    name: str | None,
    description: str,
    classification: str,
    visibility: str,
    dataset_id: str | None = None,
) -> Dataset:
    actor.require(Permission.UPLOAD)
    actor.require_verified()
    if len(content) > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise ValidationFailed(f"Files up to {settings.MAX_UPLOAD_MB} MB can be uploaded.")
    await quotas.check_storage(session, actor, len(content))
    if dataset_id:
        existing = await get_visible(session, actor, dataset_id)
        if existing.owner_id != actor.id and actor.role != "admin":
            raise Forbidden("Only the owner can add a new version.")
    frame, _ = await sandbox.run_python(
        lambda: storage.parse_upload(filename, content),
        user_id=actor.member_id,
        kind="profile",
        description=f"Parse {filename}",
    )
    label = name or filename.rsplit(".", 1)[0].replace("_", " ").strip().capitalize()
    dataset = await ingest_frame(
        session,
        actor,
        frame,
        name=label,
        description=description,
        source="upload",
        source_name=filename,
        classification=classification,
        visibility=visibility,
        landing_file=(filename, content),
        dataset_id=dataset_id,
    )
    await session.commit()
    await session.refresh(dataset, ["insights"])
    return dataset


# ------------------------------------------------------------------ governance


async def refresh_fields(session: AsyncSession, dataset: Dataset) -> None:
    table = dataset.qualified_table()
    if table is None:
        dataset.fields = None
        return
    conn = await session.connection()
    dataset.fields = await catalogue.compute_fields(conn, table, dataset.pii_columns or {})


async def approve(session: AsyncSession, actor: Actor, dataset_id: str, reason: str) -> Dataset:
    actor.require(Permission.APPROVE, "Only a Data Steward can approve datasets.")
    dataset = await get_visible(session, actor, dataset_id)
    if dataset.layer == Layer.BRONZE.value:
        raise Conflict("Clean this dataset before approving it. Raw (bronze) data is never queryable.")
    if dataset.status == DatasetStatus.APPROVED.value:
        raise Conflict("This dataset is already approved.")
    table = dataset.qualified_table()
    assert table
    conn = await session.connection()
    if not await warehouse.table_exists(conn, table):
        raise Conflict("The cleaned table is missing. Run the cleaning recipe again.")
    await warehouse.grant_sandbox(conn, table)
    dataset.status = DatasetStatus.APPROVED.value
    dataset.decided_by = actor.id
    dataset.decided_at = utcnow()
    dataset.decision_reason = reason or "Quality profile reviewed"
    await refresh_fields(session, dataset)
    await audit.record(
        session,
        actor,
        "Approved dataset",
        "dataset",
        dataset.id,
        dataset.name,
        f"{dataset.decision_reason}, classification {dataset.classification}",
    )
    await session.commit()
    sync_superset_later(dataset.id)
    return dataset


async def reject(session: AsyncSession, actor: Actor, dataset_id: str, reason: str) -> Dataset:
    actor.require(Permission.APPROVE, "Only a Data Steward can reject datasets.")
    if not reason.strip():
        raise ValidationFailed("Give the reason, so the owner knows what to fix.")
    dataset = await get_visible(session, actor, dataset_id)
    table = dataset.qualified_table()
    conn = await session.connection()
    if table:
        await warehouse.revoke_sandbox(conn, table)
    dataset.status = DatasetStatus.REJECTED.value
    dataset.decided_by = actor.id
    dataset.decided_at = utcnow()
    dataset.decision_reason = reason.strip()
    dataset.fields = None
    await audit.record(
        session, actor, "Rejected dataset", "dataset", dataset.id, dataset.name, reason.strip()
    )
    await session.commit()
    sync_superset_later(dataset.id)
    return dataset


async def update_settings(
    session: AsyncSession, actor: Actor, dataset_id: str, changes: dict[str, Any]
) -> Dataset:
    dataset = await get_visible(session, actor, dataset_id)
    is_owner = dataset.owner_id == actor.id
    if not (is_owner or actor.role == "admin" or actor.can(Permission.MANAGE_POLICIES)):
        raise Forbidden("Only the owner or a steward can change this dataset.")
    described: list[str] = []
    if "name" in changes and changes["name"]:
        dataset.name = str(changes["name"]).strip()[:200]
        described.append("renamed")
    if "description" in changes and changes["description"] is not None:
        dataset.description = str(changes["description"])[:4000]
        described.append("description")
    if changes.get("classification"):
        wanted = Classification(changes["classification"])
        if dataset.pii_columns and wanted != Classification.RESTRICTED_PERSONAL and dataset.layer == "bronze":
            raise ValidationFailed("Raw data with personal information stays restricted-personal.")
        dataset.classification = wanted.value
        described.append(f"classification {wanted.value}")
    if changes.get("visibility"):
        dataset.visibility = Visibility(changes["visibility"]).value
        described.append(f"visibility {dataset.visibility}")
    if changes.get("defaultMeasure") is not None:
        dataset.default_measure = changes["defaultMeasure"] or None
    await audit.record(
        session, actor, "Updated dataset", "dataset", dataset.id, dataset.name, ", ".join(described)
    )
    await session.commit()
    sync_superset_later(dataset.id)
    return dataset


async def share(session: AsyncSession, actor: Actor, dataset_id: str, emails: list[str]) -> list[str]:
    dataset = await get_visible(session, actor, dataset_id)
    if dataset.owner_id != actor.id and actor.role != "admin":
        raise Forbidden("Only the owner can share this dataset.")
    return await share_resource(session, actor, "dataset", dataset.id, dataset.name, emails)


async def share_resource(
    session: AsyncSession, actor: Actor, kind: str, resource_id: str, label: str, emails: list[str]
) -> list[str]:
    added: list[str] = []
    for email in {e.strip().lower() for e in emails if e.strip()}:
        user = (
            await session.execute(select(User).where(User.email == email, User.status == "active"))
        ).scalar_one_or_none()
        if user is None:
            raise NotFound(f"No active member with the email {email}.")
        exists = await session.execute(
            select(Share.id).where(
                Share.resource_type == kind, Share.resource_id == resource_id, Share.user_id == user.id
            )
        )
        if exists.first() is None:
            session.add(
                Share(resource_type=kind, resource_id=resource_id, user_id=user.id, created_by=actor.id)
            )
            added.append(user.name)
    if added:
        await audit.record(
            session, actor, f"Shared {kind}", kind, resource_id, label, f"With {', '.join(added)}"
        )
    await session.commit()
    return added


async def unshare_resource(
    session: AsyncSession, actor: Actor, kind: str, resource_id: str, label: str, user_id: str
) -> None:
    await session.execute(
        delete(Share).where(
            Share.resource_type == kind, Share.resource_id == resource_id, Share.user_id == user_id
        )
    )
    await audit.record(
        session, actor, f"Stopped sharing {kind}", kind, resource_id, label, f"Member {user_id}"
    )
    await session.commit()


async def shares_of(session: AsyncSession, kind: str, resource_id: str) -> list[dict[str, str]]:
    rows = await session.execute(
        select(User.id, User.name, User.email)
        .join(Share, Share.user_id == User.id)
        .where(Share.resource_type == kind, Share.resource_id == resource_id)
    )
    return [{"id": r[0], "name": r[1], "email": r[2]} for r in rows]


async def decide_insight(
    session: AsyncSession, actor: Actor, dataset_id: str, insight_id: str, decision: str
) -> Insight:
    await get_visible(session, actor, dataset_id)
    insight = await session.get(Insight, insight_id)
    if insight is None or insight.dataset_id != dataset_id:
        raise NotFound("Insight not found.")
    if decision not in ("accepted", "dismissed"):
        raise ValidationFailed("Accept or dismiss the insight.")
    insight.status = decision
    insight.decided_by = actor.id
    await audit.record(
        session, actor, f"{decision.capitalize()} insight", "dataset", dataset_id, insight.title
    )
    await session.commit()
    return insight


# ------------------------------------------------------------------ reading


async def preview_rows(
    session: AsyncSession, actor: Actor, dataset_id: str, layer: str | None, limit: int = 50
) -> dict[str, Any]:
    """A few rows to look at. Personal columns are always masked, even in bronze."""
    dataset = await get_visible(session, actor, dataset_id)
    limit = max(1, min(limit, 200))
    if layer in (None, "", "current"):
        table = dataset.qualified_table() or f"bronze.{dataset.table_name}_v{dataset.current_version}"
    elif layer == "bronze":
        table = f"bronze.{dataset.table_name}_v{dataset.current_version}"
    else:
        table = f"{layer}.{dataset.table_name}"
    conn = await session.connection()
    if not await warehouse.table_exists(conn, table):
        raise NotFound(f"This dataset has no {layer or 'current'} table yet.")
    frame = await warehouse.read_frame(conn, table, limit)
    pii = dataset.pii_columns or {}
    rows: list[list[Any]] = []
    for record in frame.itertuples(index=False, name=None):
        out = []
        for column, value in zip(frame.columns, record, strict=True):
            if column in pii and value is not None and "•" not in str(value):
                value = mask_value(value, PiiKind(pii[column]))
            out.append(warehouse.jsonable(value))
        rows.append(out)
    return {"table": table, "columns": list(frame.columns), "rows": rows, "masked": sorted(pii)}


async def lineage(session: AsyncSession, actor: Actor, dataset_id: str) -> list[dict[str, Any]]:
    await get_visible(session, actor, dataset_id)
    rows = await session.execute(
        select(LineageEdge, User.name)
        .outerjoin(User, User.id == LineageEdge.actor_id)
        .where(LineageEdge.dataset_id == dataset_id)
        .order_by(LineageEdge.created_at)
    )
    return [
        {
            "id": e.id,
            "source": e.source,
            "target": e.target,
            "transform": e.transform,
            "detail": e.detail,
            "recipeId": e.recipe_id,
            "actor": name or "System",
            "at": e.created_at.isoformat(),
            "when": humanize(e.created_at),
        }
        for e, name in rows
    ]


async def versions(session: AsyncSession, actor: Actor, dataset_id: str) -> list[dict[str, Any]]:
    await get_visible(session, actor, dataset_id)
    rows = await session.execute(
        select(DatasetVersion)
        .where(DatasetVersion.dataset_id == dataset_id)
        .order_by(DatasetVersion.version, DatasetVersion.created_at)
    )
    return [
        {
            "version": v.version,
            "layer": v.layer,
            "table": v.table,
            "rows": v.rows,
            "checksum": v.checksum,
            "at": v.created_at.isoformat(),
        }
        for v in rows.scalars()
    ]


# ------------------------------------------------------------------ promotion and erasure


async def promote(session: AsyncSession, actor: Actor, dataset_id: str, columns: list[str] | None) -> Dataset:
    """Silver → gold: an analytics-ready copy, optionally with fewer columns. Personal columns never reach gold."""
    actor.require(Permission.APPROVE, "Only a Data Steward can promote data to gold.")
    dataset = await get_visible(session, actor, dataset_id)
    if dataset.layer != Layer.SILVER.value or dataset.status != DatasetStatus.APPROVED.value:
        raise Conflict("Only approved silver data can be promoted to gold.")
    conn = await session.connection()
    types = await warehouse.column_types(conn, f"silver.{dataset.table_name}")
    keep = [c for c in (columns or list(types)) if c in types and c not in (dataset.pii_columns or {})]
    if not keep:
        raise ValidationFailed("Choose at least one column that is not personal information.")
    selected = ", ".join(ident(c) for c in keep)
    gold = f"gold.{ident(dataset.table_name)}"
    await conn.execute(text(f"DROP TABLE IF EXISTS {gold}"))
    await conn.execute(
        text(f"CREATE TABLE {gold} AS SELECT {selected} FROM silver.{ident(dataset.table_name)}")
    )
    await warehouse.revoke_sandbox(conn, f"silver.{dataset.table_name}")
    await warehouse.grant_sandbox(conn, f"gold.{dataset.table_name}")
    dataset.layer = Layer.GOLD.value
    dataset.pii_columns = {}
    rows = int((await conn.execute(text(f"SELECT COUNT(*) FROM {gold}"))).scalar() or 0)
    await record_version(
        session,
        dataset,
        "gold",
        table=f"gold.{dataset.table_name}",
        rows=rows,
        column_types={c: types[c] for c in keep},
        created_by=actor.id,
    )
    session.add(
        LineageEdge(
            dataset_id=dataset.id,
            source=f"silver.{dataset.table_name}",
            target=f"gold.{dataset.table_name}",
            transform="promote",
            detail=f"{len(keep)} columns, personal columns left out",
            actor_id=actor.id,
        )
    )
    await refresh_fields(session, dataset)
    await audit.record(
        session, actor, "Promoted to gold", "dataset", dataset.id, dataset.name, f"Columns: {', '.join(keep)}"
    )
    await session.commit()
    sync_superset_later(dataset.id)
    return dataset


async def erase(session: AsyncSession, actor: Actor, dataset_id: str, reason: str) -> dict[str, Any]:
    """Right to erasure: drop every layer's table and the landed files, and return a verifiable record."""
    dataset = await session.get(Dataset, dataset_id)
    if dataset is None or dataset.erased_at is not None:
        raise NotFound("Dataset not found.")
    if dataset.owner_id != actor.id and actor.role != "admin":
        raise Forbidden("Only the owner or an administrator can erase a dataset.")
    conn = await session.connection()
    versions_rows = (
        (await session.execute(select(DatasetVersion.table).where(DatasetVersion.dataset_id == dataset.id)))
        .scalars()
        .all()
    )
    tables = sorted(set(versions_rows) | {f"silver.{dataset.table_name}", f"gold.{dataset.table_name}"})
    dropped = []
    for table in tables:
        if await warehouse.table_exists(conn, table):
            await warehouse.drop_table(conn, table)
            dropped.append(table)
    storage.remove_dataset_files(dataset.id)
    await session.execute(delete(RowPolicy).where(RowPolicy.dataset_id == dataset.id))
    await session.execute(
        delete(Share).where(Share.resource_type == "dataset", Share.resource_id == dataset.id)
    )
    dataset.erased_at = utcnow()
    dataset.status = DatasetStatus.REJECTED.value
    dataset.profile = None
    dataset.columns = []
    dataset.fields = None
    dataset.rows = 0
    dataset.size_bytes = 0
    receipt = {
        "datasetId": dataset.id,
        "erasedAt": dataset.erased_at.isoformat(),
        "tablesDropped": dropped,
        "filesRemoved": True,
        "reason": reason,
    }
    receipt["receipt"] = hashlib.sha256(repr(sorted(receipt.items())).encode()).hexdigest()
    session.add(
        LineageEdge(
            dataset_id=dataset.id,
            source=", ".join(dropped) or dataset.table_name,
            target="(erased)",
            transform="erase",
            detail=reason,
            actor_id=actor.id,
        )
    )
    await audit.record(
        session,
        actor,
        "Erased dataset",
        "dataset",
        dataset.id,
        dataset.name,
        reason or "Right to erasure",
        receipt,
    )
    await session.commit()
    sync_superset_later(dataset.id)
    return receipt


# ------------------------------------------------------------------ Superset sync hook


def sync_superset_later(dataset_id: str) -> None:
    """Keep Superset in step, after the response. Failures are logged and never block governance actions."""
    integrations.in_background(integrations.sync_dataset_quietly(dataset_id))
