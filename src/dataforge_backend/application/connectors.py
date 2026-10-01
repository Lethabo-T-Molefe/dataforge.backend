"""Governed connectors to live sources: PostgreSQL databases, PostgreSQL-compatible warehouses and spreadsheets.

Any member may request a connector; a steward approves it. Credentials are encrypted at rest and never
returned. Imported tables go through the same pipeline as uploads: bronze, PII scan, profile, approval.
"""

from typing import Any

import pandas as pd
from openpyxl import load_workbook
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..core.crypto import decrypt_secret, encrypt_secret
from ..core.errors import Forbidden, NotFound, ValidationFailed
from ..core.security import utcnow
from ..domain.roles import Permission
from ..infrastructure import external_db
from ..infrastructure.db.models import Connector, Dataset
from . import audit
from .common import Actor, humanize, slugify
from .datasets import ingest_frame, summary
from .runtime import sandbox

KINDS = ("postgres", "warehouse", "spreadsheet")
KIND_LABELS = {"postgres": "PostgreSQL", "warehouse": "Warehouse", "spreadsheet": "Spreadsheet"}


def connector_dict(connector: Connector, actor: Actor) -> dict[str, Any]:
    config = connector.config or {}
    return {
        "id": connector.id,
        "name": connector.name,
        "kind": connector.kind,
        "status": connector.status,
        "tables": connector.tables,
        "owner": connector.owner_label or "Unknown",
        "ownerId": connector.owner_id,
        "lastSync": humanize(connector.last_sync_at),
        "sourcePath": connector.source_path,
        "config": {
            k: v
            for k, v in config.items()
            if k in ("host", "port", "database", "schema", "user", "sslmode", "filename")
        },
        "lastError": connector.last_error,
        "hasCredentials": connector.secret_enc is not None,
        "canManage": actor.can(Permission.MANAGE_CONNECTORS),
    }


def _source_config(connector: Connector) -> external_db.SourceConfig:
    config = connector.config or {}
    if not connector.secret_enc:
        raise ValidationFailed("This connector has no credentials.")
    return external_db.SourceConfig(
        host=str(config.get("host", "")),
        port=int(config.get("port", 5432)),
        database=str(config.get("database", "")),
        user=str(config.get("user", "")),
        password=decrypt_secret(connector.secret_enc),
        schema=str(config.get("schema", "public")),
        sslmode=str(config.get("sslmode", "prefer")),
    )


def _spreadsheet_path(connector: Connector) -> str:
    return str(
        settings.STORAGE_DIR
        / "connectors"
        / connector.id
        / str((connector.config or {}).get("filename", "workbook.xlsx"))
    )


async def list_connectors(session: AsyncSession, actor: Actor) -> list[dict[str, Any]]:
    rows = await session.execute(select(Connector).order_by(Connector.created_at))
    connectors = rows.scalars().all()
    if not actor.can(Permission.MANAGE_CONNECTORS):
        connectors = [c for c in connectors if c.status == "connected" or c.owner_id == actor.id]
    return [connector_dict(c, actor) for c in connectors]


async def _get(session: AsyncSession, connector_id: str) -> Connector:
    connector = await session.get(Connector, connector_id)
    if connector is None:
        raise NotFound("Connector not found.")
    return connector


async def request(session: AsyncSession, actor: Actor, body: dict[str, Any]) -> dict[str, Any]:
    """Request (or, for a steward, create) a connector. It is read-only by construction."""
    if not (actor.can(Permission.REQUEST_CONNECTORS) or actor.can(Permission.MANAGE_CONNECTORS)):
        raise Forbidden("Your role cannot connect sources.")
    kind = str(body.get("kind", ""))
    if kind not in KINDS:
        raise ValidationFailed("Choose PostgreSQL, a warehouse or a spreadsheet.")
    name = str(body.get("name", "")).strip()[:200]
    if not name:
        raise ValidationFailed("Name the connector.")
    base = slugify(name, 60)
    connector_id, n = base, 2
    while await session.get(Connector, connector_id) is not None:
        connector_id, n = f"{base[:55]}-{n}", n + 1
    config: dict[str, Any]
    secret = None
    if kind == "spreadsheet":
        filename = str(body.get("filename") or "workbook.xlsx")
        config = {"filename": filename}
        source_path = f"Spreadsheet · {filename}"
    else:
        required = ("host", "database", "user")
        if any(not body.get(k) for k in required):
            raise ValidationFailed("Give the host, database and a read-only user.")
        config = {
            "host": str(body["host"]),
            "port": int(body.get("port") or 5432),
            "database": str(body["database"]),
            "schema": str(body.get("schema") or "public"),
            "user": str(body["user"]),
            "sslmode": str(body.get("sslmode") or "prefer"),
        }
        secret = encrypt_secret(str(body.get("password") or ""))
        source_path = f"{KIND_LABELS[kind]} · {config['database']}.{config['schema']}"
    connector = Connector(
        id=connector_id,
        name=name,
        kind=kind,
        status="pending",
        owner_id=actor.id,
        owner_label=actor.name,
        config=config,
        secret_enc=secret,
        source_path=source_path,
    )
    session.add(connector)
    await audit.record(
        session,
        actor,
        "Requested connector",
        "connector",
        connector.id,
        source_path,
        "Read-only; credentials encrypted at rest",
    )
    await session.commit()
    if actor.can(Permission.MANAGE_CONNECTORS) and kind != "spreadsheet":
        return await approve(session, actor, connector.id)
    return connector_dict(connector, actor)


async def approve(session: AsyncSession, actor: Actor, connector_id: str) -> dict[str, Any]:
    actor.require(Permission.MANAGE_CONNECTORS, "Only a Data Steward can approve connectors.")
    connector = await _get(session, connector_id)
    await test(session, actor, connector.id)
    connector = await _get(session, connector_id)
    connector.approved_by = actor.id
    await audit.record(
        session,
        actor,
        "Connected source",
        "connector",
        connector.id,
        connector.source_path,
        f"Status {connector.status}",
    )
    await session.commit()
    return connector_dict(connector, actor)


async def test(session: AsyncSession, actor: Actor, connector_id: str) -> dict[str, Any]:
    connector = await _get(session, connector_id)
    if connector.owner_id != actor.id and not actor.can(Permission.MANAGE_CONNECTORS):
        raise Forbidden("Only the requester or a steward can test this connector.")
    try:
        tables = await _tables(connector)
        connector.status = (
            "connected" if connector.approved_by or actor.can(Permission.MANAGE_CONNECTORS) else "pending"
        )
        connector.tables = len(tables)
        connector.last_error = None
    except ValidationFailed as error:
        connector.status = "error"
        connector.last_error = error.message
    await session.commit()
    return connector_dict(connector, actor)


async def _tables(connector: Connector) -> list[dict[str, Any]]:
    if connector.kind == "spreadsheet":
        path = _spreadsheet_path(connector)
        try:
            book = load_workbook(path, read_only=True)
        except FileNotFoundError as error:
            raise ValidationFailed("Upload the spreadsheet first.") from error
        return [{"name": sheet.title, "columns": sheet.max_column or 0} for sheet in book.worksheets]
    return await external_db.list_tables(_source_config(connector))


async def list_tables(session: AsyncSession, actor: Actor, connector_id: str) -> list[dict[str, Any]]:
    connector = await _get(session, connector_id)
    if connector.status != "connected" and not actor.can(Permission.MANAGE_CONNECTORS):
        raise ValidationFailed("This connector is not approved yet.")
    return await _tables(connector)


async def upload_workbook(
    session: AsyncSession, actor: Actor, connector_id: str, filename: str, content: bytes
) -> dict[str, Any]:
    connector = await _get(session, connector_id)
    if connector.kind != "spreadsheet":
        raise ValidationFailed("Only spreadsheet connectors take a file.")
    if connector.owner_id != actor.id and not actor.can(Permission.MANAGE_CONNECTORS):
        raise Forbidden("Only the requester or a steward can replace the workbook.")
    if not filename.lower().endswith((".xlsx", ".xlsm")):
        raise ValidationFailed("Upload an Excel workbook (.xlsx).")
    connector.config = {**(connector.config or {}), "filename": "workbook.xlsx"}
    folder = settings.STORAGE_DIR / "connectors" / connector.id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "workbook.xlsx").write_bytes(content)
    connector.source_path = f"Spreadsheet · {filename}"
    await audit.record(session, actor, "Uploaded workbook", "connector", connector.id, filename)
    await session.commit()
    return await test(session, actor, connector.id)


async def import_table(
    session: AsyncSession, actor: Actor, connector_id: str, body: dict[str, Any]
) -> dict[str, Any]:
    actor.require(Permission.UPLOAD, "Your role cannot bring data in.")
    actor.require_verified()
    connector = await _get(session, connector_id)
    if connector.status != "connected":
        raise ValidationFailed("This connector is not connected yet. A steward approves it first.")
    table = str(body.get("table", ""))
    if connector.kind == "spreadsheet":
        path = _spreadsheet_path(connector)
        frame, _ = await sandbox.run_python(
            lambda: pd.read_excel(path, sheet_name=table, dtype=object),
            user_id=actor.member_id,
            kind="profile",
            description=f"Read sheet {table}",
        )
    else:
        frame = await external_db.read_table(_source_config(connector), table)
    dataset = await ingest_frame(
        session,
        actor,
        frame,
        name=str(body.get("name") or f"{connector.name} {table}".strip()),
        description=str(body.get("description") or f"Imported from {connector.source_path}, table {table}."),
        source="connector",
        source_name=f"{connector.source_path} · {table}",
        classification=str(body.get("classification") or "internal-confidential"),
        visibility=str(body.get("visibility") or "private"),
        connector_id=connector.id,
        dataset_id=body.get("datasetId"),
        landing_file=None,
    )
    connector.last_sync_at = utcnow()
    await session.commit()
    await session.refresh(dataset, ["insights"])
    return summary(dataset, actor)


async def sync(session: AsyncSession, actor: Actor, connector_id: str) -> list[dict[str, Any]]:
    """Re-read every dataset that came from this connector, as new versions."""
    connector = await _get(session, connector_id)
    rows = await session.execute(
        select(Dataset).where(Dataset.connector_id == connector.id, Dataset.erased_at.is_(None))
    )
    out = []
    for dataset in rows.scalars().all():
        table = dataset.source_name.rsplit(" · ", 1)[-1]
        out.append(
            await import_table(
                session,
                actor,
                connector.id,
                {
                    "table": table,
                    "datasetId": dataset.id,
                    "name": dataset.name,
                    "classification": dataset.classification,
                    "visibility": dataset.visibility,
                },
            )
        )
    await audit.record(
        session,
        actor,
        "Synced connector",
        "connector",
        connector.id,
        connector.source_path,
        f"{len(out)} datasets refreshed",
    )
    await session.commit()
    return out


async def delete(session: AsyncSession, actor: Actor, connector_id: str) -> None:
    actor.require(Permission.MANAGE_CONNECTORS, "Only a Data Steward can remove connectors.")
    connector = await _get(session, connector_id)
    await session.delete(connector)
    await audit.record(session, actor, "Removed connector", "connector", connector_id, connector.source_path)
    await session.commit()
