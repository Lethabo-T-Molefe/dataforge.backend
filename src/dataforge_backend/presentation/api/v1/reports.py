"""Governed reports: templates, generation, exports with provenance, and a reproducibility check."""

from typing import Any

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from ....application import reports
from ....application.common import Actor
from ..deps import current_actor, get_db
from ..schemas import FreeBody

router = APIRouter(tags=["Reports"])


@router.get("/reports/templates")
async def report_templates(actor: Actor = Depends(current_actor)) -> list[dict[str, Any]]:
    return reports.templates()


@router.post("/reports", status_code=status.HTTP_201_CREATED)
async def generate_report(
    body: FreeBody, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Compute every figure of a template in the sandbox, with its SQL kept beside it."""
    return await reports.generate(session, actor, body.model_dump())


@router.get("/reports")
async def list_reports(
    actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> list[dict[str, Any]]:
    return await reports.list_reports(session, actor)


@router.get("/reports/{report_id}")
async def get_report(
    report_id: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    return await reports.get(session, actor, report_id)


@router.post("/reports/{report_id}/audit")
async def audit_report(
    report_id: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Re-run every figure's SQL and say whether each published value reproduces."""
    return await reports.audit_report(session, actor, report_id)


@router.get("/reports/{report_id}/export")
async def export_report(
    report_id: str,
    format: str = "pdf",  # noqa: A002
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> Response:
    content, media, filename = await reports.export(session, actor, report_id, format)
    return Response(
        content, media_type=media, headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
