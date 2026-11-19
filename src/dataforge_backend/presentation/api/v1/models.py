"""Model versions: review (approve, reject, retire), and trained outcome models (train, score, governed views)."""

from typing import Any

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from ....application import outcome_models, predictions
from ....application.common import Actor
from ..deps import current_actor, get_db
from ..schemas import FreeBody

router = APIRouter(tags=["Models"])


@router.get("/predictions/models/train/options")
async def train_options(
    datasetId: str,  # noqa: N803 - the query parameter is camelCase like the rest of the API
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """The outcome columns, suggested protected attributes and columns a curator can train on."""
    return await outcome_models.options(session, actor, datasetId)


@router.post("/predictions/models/train/preview")
async def train_preview(
    body: FreeBody, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Each column's role in the model and why, before anything is trained."""
    return await outcome_models.preview(session, actor, body.model_dump())


@router.post("/predictions/models/train", status_code=status.HTTP_201_CREATED)
async def train(
    body: FreeBody, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> list[dict[str, Any]]:
    """Train and validate a model for a yes/no outcome. Each algorithm is saved as a draft for review."""
    return await outcome_models.train(session, actor, body.model_dump())


@router.get("/predictions/models/{model_id}")
async def model_detail(
    model_id: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """A trained model in full: card, metrics against baselines, fairness, drift and the last scoring run."""
    return await outcome_models.detail(session, actor, model_id)


@router.post("/predictions/models/{model_id}/approve")
async def approve(
    model_id: str,
    body: FreeBody,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Approve a draft for use. Refused for the person who built or trained it."""
    return await predictions.approve(session, actor, model_id, body.model_dump().get("note"))


@router.post("/predictions/models/{model_id}/reject")
async def reject(
    model_id: str,
    body: FreeBody,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await predictions.reject(session, actor, model_id, body.model_dump().get("reason"))


@router.post("/predictions/models/{model_id}/retire")
async def retire(
    model_id: str,
    body: FreeBody,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await predictions.retire(session, actor, model_id, body.model_dump().get("reason"))


@router.post("/predictions/models/{model_id}/score")
async def score_records(
    model_id: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Score the records whose outcome is not known yet with an approved trained model."""
    return await outcome_models.run_scoring(session, actor, model_id)


@router.get("/predictions/models/{model_id}/scored")
async def scored(
    model_id: str,
    by: str | None = None,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Flagged records by group and the highest scores, under the viewer's row policy."""
    return await outcome_models.scored_view(session, actor, model_id, by)


@router.get("/predictions/models/{model_id}/card")
async def model_card(
    model_id: str,
    format: str = "json",  # noqa: A002
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> Any:
    if format == "pdf":
        content, filename = await outcome_models.card_pdf(session, actor, model_id)
        return Response(
            content,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    detail = await outcome_models.detail(session, actor, model_id)
    return detail["card"]
