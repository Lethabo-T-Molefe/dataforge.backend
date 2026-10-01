from typing import Any

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from ....application import datasets
from ....application.common import Actor
from ....core.config import settings
from ....core.errors import Forbidden, ValidationFailed
from ..deps import current_actor, data_actor, get_db
from ..schemas import DatasetUpdateIn, DecisionIn, EraseIn, InsightDecisionIn, PromoteIn, ShareIn

router = APIRouter(prefix="/datasets", tags=["Datasets"])


async def _read_upload(file: UploadFile) -> bytes:
    limit = settings.MAX_UPLOAD_MB * 1024 * 1024
    content = await file.read(limit + 1)
    if len(content) > limit:
        raise ValidationFailed(f"Files up to {settings.MAX_UPLOAD_MB} MB can be uploaded.")
    return content


@router.get("")
async def list_datasets(
    actor: Actor = Depends(data_actor), session: AsyncSession = Depends(get_db)
) -> list[dict[str, Any]]:
    return await datasets.list_datasets(session, actor)


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload(
    file: UploadFile = File(...),
    name: str | None = Form(default=None, max_length=200),
    description: str = Form(default="", max_length=4000),
    classification: str = Form(default="internal-confidential"),
    visibility: str = Form(default="private"),
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Upload a CSV, Excel or JSON file. It lands in bronze, is scanned for personal information and profiled."""
    content = await _read_upload(file)
    dataset = await datasets.upload(
        session,
        actor,
        filename=file.filename or "upload.csv",
        content=content,
        name=name,
        description=description,
        classification=classification,
        visibility=visibility,
    )
    return datasets.summary(dataset, actor)


@router.get("/{dataset_id}")
async def get_dataset(
    dataset_id: str, actor: Actor = Depends(data_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    dataset = await datasets.get_visible(session, actor, dataset_id)
    return {**datasets.summary(dataset, actor), "profile": dataset.profile}


@router.patch("/{dataset_id}")
async def update_dataset(
    dataset_id: str,
    body: DatasetUpdateIn,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    dataset = await datasets.update_settings(
        session, actor, dataset_id, body.model_dump(exclude_unset=True, by_alias=True)
    )
    await session.refresh(dataset, ["insights"])
    return datasets.summary(dataset, actor)


@router.post("/{dataset_id}/versions", status_code=status.HTTP_201_CREATED)
async def new_version(
    dataset_id: str,
    file: UploadFile = File(...),
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Upload a new version of the same dataset. Its saved recipe can then be rerun on it."""
    existing = await datasets.get_visible(session, actor, dataset_id)
    content = await _read_upload(file)
    dataset = await datasets.upload(
        session,
        actor,
        filename=file.filename or "upload.csv",
        content=content,
        name=existing.name,
        description=existing.description,
        classification=existing.classification,
        visibility=existing.visibility,
        dataset_id=dataset_id,
    )
    return datasets.summary(dataset, actor)


@router.get("/{dataset_id}/versions")
async def versions(
    dataset_id: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> list[dict[str, Any]]:
    return await datasets.versions(session, actor, dataset_id)


@router.get("/{dataset_id}/profile")
async def profile(
    dataset_id: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    dataset = await datasets.get_visible(session, actor, dataset_id)
    return dataset.profile or {}


@router.get("/{dataset_id}/rows")
async def rows(
    dataset_id: str,
    layer: str | None = None,
    limit: int = 50,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """A few rows to look at. Personal columns are masked in every layer."""
    return await datasets.preview_rows(session, actor, dataset_id, layer, limit)


@router.get("/{dataset_id}/lineage")
async def lineage(
    dataset_id: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> list[dict[str, Any]]:
    return await datasets.lineage(session, actor, dataset_id)


@router.post("/{dataset_id}/approve")
async def approve(
    dataset_id: str,
    body: DecisionIn,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    dataset = await datasets.approve(session, actor, dataset_id, body.reason)
    await session.refresh(dataset, ["insights"])
    return datasets.summary(dataset, actor)


@router.post("/{dataset_id}/reject")
async def reject(
    dataset_id: str,
    body: DecisionIn,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    dataset = await datasets.reject(session, actor, dataset_id, body.reason)
    await session.refresh(dataset, ["insights"])
    return datasets.summary(dataset, actor)


@router.post("/{dataset_id}/promote")
async def promote(
    dataset_id: str,
    body: PromoteIn,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    dataset = await datasets.promote(session, actor, dataset_id, body.columns)
    await session.refresh(dataset, ["insights"])
    return datasets.summary(dataset, actor)


@router.get("/{dataset_id}/shares")
async def shares(
    dataset_id: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> list[dict[str, str]]:
    await datasets.get_visible(session, actor, dataset_id)
    return await datasets.shares_of(session, "dataset", dataset_id)


@router.post("/{dataset_id}/shares")
async def share(
    dataset_id: str,
    body: ShareIn,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    added = await datasets.share(session, actor, dataset_id, [str(e) for e in body.emails])
    return {"added": added, "shares": await datasets.shares_of(session, "dataset", dataset_id)}


@router.delete("/{dataset_id}/shares/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def unshare(
    dataset_id: str,
    user_id: str,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> None:
    dataset = await datasets.get_visible(session, actor, dataset_id)
    if dataset.owner_id != actor.id and actor.role != "admin":
        raise Forbidden("Only the owner can change who this dataset is shared with.")
    await datasets.unshare_resource(session, actor, "dataset", dataset_id, dataset.name, user_id)


@router.post("/{dataset_id}/insights/{insight_id}")
async def decide_insight(
    dataset_id: str,
    insight_id: str,
    body: InsightDecisionIn,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    insight = await datasets.decide_insight(session, actor, dataset_id, insight_id, body.decision)
    return datasets.insight_dict(insight)


@router.post("/{dataset_id}/erase")
async def erase(
    dataset_id: str,
    body: EraseIn,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Right to erasure. Irreversible: type the dataset id in `confirm`. Returns a verifiable receipt."""
    if body.confirm != dataset_id:
        raise ValidationFailed("Type the dataset id to confirm erasure.")
    return await datasets.erase(session, actor, dataset_id, body.reason)
