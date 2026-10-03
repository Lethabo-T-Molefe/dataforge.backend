"""Governed connectors to live sources: PostgreSQL databases, PostgreSQL-compatible warehouses and spreadsheets.

Any member may request a connector; a steward approves it. Credentials are encrypted at rest and never
returned. Imported tables go through the same pipeline as uploads: bronze, PII scan, profile, approval.
"""

import asyncio
import io
from typing import Any
from zipfile import BadZipFile

import pandas as pd
from openpyxl import load_workbook
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.crypto import decrypt_secret, encrypt_secret
from ..core.errors import Forbidden, NotFound, ValidationFailed
from ..core.security import utcnow
from ..domain.roles import Permission
from ..infrastructure import external_db, storage
from ..infrastructure.db.models import Connector, Dataset
from . import audit, inbox
from .cleaning import frame_checksum
from .common import Actor, humanize, slugify
from .datasets import ingest_frame, summary
from .runtime import sandbox

KINDS = ("postgres", "warehouse", "spreadsheet")
REFRESH_CADENCES = {"hourly": 3600, "daily": 86_400, "weekly": 604_800}
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
        "refreshCadence": connector.refresh_cadence,
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


def _workbook_key(connector: Connector) -> str:
    config = connector.config or {}
    # Workbooks uploaded before keys were recorded sit at the connector's folder under their file name.
    return str(config.get("key") or f"connectors/{connector.id}/{config.get('filename', 'workbook.xlsx')}")


async def _workbook(connector: Connector) -> bytes:
    try:
        return await storage.read(_workbook_key(connector))
    except FileNotFoundError as error:
        raise ValidationFailed("Upload the spreadsheet first.") from error


def _sheets(content: bytes) -> list[dict[str, Any]]:
    book = load_workbook(io.BytesIO(content), read_only=True)
    try:
        return [{"name": sheet.title, "columns": sheet.max_column or 0} for sheet in book.worksheets]
    finally:
        book.close()


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
    if not actor.can(Permission.MANAGE_CONNECTORS):
        inbox.notify(
            session,
            await inbox.stewards(session),
            "approval",
            f"{actor.name} asked to connect {name}",
            f"{source_path}. Test and approve it before anyone imports from it.",
            "/datasets/connect",
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
        content = await _workbook(connector)
        try:
            return await asyncio.to_thread(_sheets, content)
        except (OSError, ValueError, KeyError, BadZipFile) as error:
            raise ValidationFailed("The workbook could not be read. Upload it again as .xlsx.") from error
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
    previous = (connector.config or {}).get("key")
    key = await storage.store_workbook(connector.id, content)
    connector.config = {**(connector.config or {}), "filename": filename, "key": key}
    if previous and previous != key:
        await storage.remove(str(previous))
    connector.source_path = f"Spreadsheet · {filename}"
    await audit.record(session, actor, "Uploaded workbook", "connector", connector.id, filename)
    await session.commit()
    return await test(session, actor, connector.id)


async def read_source(connector: Connector, table: str, actor: Actor) -> pd.DataFrame:
    """One table (or sheet) of the source, read-only, as raw object columns."""
    if connector.kind == "spreadsheet":
        content = await _workbook(connector)
        if table not in {s["name"] for s in await asyncio.to_thread(_sheets, content)}:
            raise ValidationFailed(f"{table} is not a sheet in this workbook.")
        frame: pd.DataFrame
        frame, _ = await sandbox.run_python(
            storage.read_sheet,
            content,
            table,
            user_id=actor.member_id,
            kind="profile",
            description=f"Read sheet {table}",
            isolated=True,
        )
        return frame
    return await external_db.read_table(_source_config(connector), table)


async def set_refresh(
    session: AsyncSession, actor: Actor, connector_id: str, cadence: str | None
) -> dict[str, Any]:
    """How often a steward wants this source re-read. Refreshes keep approvals only when nothing changed shape."""
    actor.require(Permission.MANAGE_CONNECTORS, "Only a Data Steward schedules refreshes.")
    connector = await _get(session, connector_id)
    if cadence is not None and cadence not in REFRESH_CADENCES:
        raise ValidationFailed("Refresh hourly, daily, weekly, or only when asked.")
    if cadence and connector.kind == "spreadsheet":
        raise ValidationFailed(
            "A workbook only changes when a new one is uploaded, so it is not refreshed on a schedule."
        )
    connector.refresh_cadence = cadence
    await audit.record(
        session,
        actor,
        "Scheduled refresh" if cadence else "Stopped scheduled refresh",
        "connector",
        connector.id,
        connector.source_path,
        cadence or "Only when asked",
    )
    await session.commit()
    return connector_dict(connector, actor)


async def import_table(
    session: AsyncSession, actor: Actor, connector_id: str, body: dict[str, Any]
) -> dict[str, Any]:
    actor.require(Permission.UPLOAD, "Your role cannot bring data in.")
    actor.require_verified()
    connector = await _get(session, connector_id)
    if connector.status != "connected":
        raise ValidationFailed("This connector is not connected yet. A steward approves it first.")
    table = str(body.get("table", ""))
    frame = await read_source(connector, table, actor)
    checksum = await asyncio.to_thread(frame_checksum, frame)
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
        source_checksum=checksum,
    )
    connector.last_sync_at = utcnow()
    await session.commit()
    await session.refresh(dataset, ["insights"])
    return summary(dataset, actor)


async def get(session: AsyncSession, connector_id: str) -> Connector:
    return await _get(session, connector_id)


async def datasets_of(session: AsyncSession, connector: Connector) -> list[Dataset]:
    rows = await session.execute(
        select(Dataset).where(Dataset.connector_id == connector.id, Dataset.erased_at.is_(None))
    )
    return list(rows.scalars().all())


async def delete(session: AsyncSession, actor: Actor, connector_id: str) -> None:
    actor.require(Permission.MANAGE_CONNECTORS, "Only a Data Steward can remove connectors.")
    connector = await _get(session, connector_id)
    if connector.kind == "spreadsheet":
        await storage.remove_connector_files(connector.id)
    await session.delete(connector)
    await audit.record(session, actor, "Removed connector", "connector", connector_id, connector.source_path)
    await session.commit()
