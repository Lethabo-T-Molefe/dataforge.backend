from typing import Any

from fastapi import APIRouter, Depends, File, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from ....application import connectors, refresh
from ....application.common import Actor
from ....core.config import settings
from ....core.errors import ValidationFailed
from ..deps import current_actor, get_db
from ..schemas import ConnectorIn, ConnectorRefreshIn, ImportIn

router = APIRouter(prefix="/connectors", tags=["Connectors"])


@router.get("")
async def list_connectors(
    actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> list[dict[str, Any]]:
    return await connectors.list_connectors(session, actor)


@router.post("", status_code=status.HTTP_201_CREATED)
async def request_connector(
    body: ConnectorIn, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Request a read-only connector. A steward's request is tested and connected at once."""
    data = body.model_dump(exclude_none=True)
    if "schema_name" in data:
        data["schema"] = data.pop("schema_name")
    return await connectors.request(session, actor, data)


@router.post("/{connector_id}/approve")
async def approve(
    connector_id: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    return await connectors.approve(session, actor, connector_id)


@router.post("/{connector_id}/test")
async def test(
    connector_id: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    return await connectors.test(session, actor, connector_id)


@router.get("/{connector_id}/tables")
async def tables(
    connector_id: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> list[dict[str, Any]]:
    return await connectors.list_tables(session, actor, connector_id)


@router.post("/{connector_id}/workbook")
async def upload_workbook(
    connector_id: str,
    file: UploadFile = File(...),
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    limit = settings.MAX_UPLOAD_MB * 1024 * 1024
    content = await file.read(limit + 1)
    if len(content) > limit:
        raise ValidationFailed(f"Files up to {settings.MAX_UPLOAD_MB} MB can be uploaded.")
    return await connectors.upload_workbook(
        session, actor, connector_id, file.filename or "workbook.xlsx", content
    )


@router.post("/{connector_id}/import", status_code=status.HTTP_201_CREATED)
async def import_table(
    connector_id: str,
    body: ImportIn,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Bring one source table in as a dataset. It lands in bronze and is profiled like an upload."""
    return await connectors.import_table(session, actor, connector_id, body.model_dump(exclude_none=True))


@router.post("/{connector_id}/sync")
async def sync(
    connector_id: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> list[dict[str, Any]]:
    """Refresh now: re-read every dataset from this source and rerun its saved recipe."""
    return await refresh.sync(session, actor, connector_id)


@router.patch("/{connector_id}")
async def set_refresh(
    connector_id: str,
    body: ConnectorRefreshIn,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """How often a steward wants the source re-read (null: only when someone asks)."""
    return await connectors.set_refresh(session, actor, connector_id, body.refresh_cadence)


@router.delete("/{connector_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete(
    connector_id: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> None:
    await connectors.delete(session, actor, connector_id)
