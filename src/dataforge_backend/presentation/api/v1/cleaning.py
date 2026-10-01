from typing import Any

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from ....application import cleaning
from ....application.common import Actor
from ....application.datasets import get_visible
from ....domain.cleaning import CHAPTERS, METHODS
from ..deps import current_actor, get_db
from ..schemas import PreviewIn, RecipeRefIn, ReorderIn, StepIn, StepUpdateIn, VisibilityIn

router = APIRouter(prefix="/cleaning", tags=["Cleaning"])


@router.get("/catalogue")
async def method_catalogue(actor: Actor = Depends(current_actor)) -> dict[str, Any]:
    """Every chapter and method, independent of a dataset."""
    return {
        "chapters": [
            {
                "id": c.id,
                "kind": c.kind,
                "number": c.number,
                "title": c.title,
                "plain": c.plain,
                "expert": c.expert,
                "recommendedMethodId": c.recommended_method_id,
            }
            for c in CHAPTERS
        ],
        "methods": [
            {
                "id": m.id,
                "chapterId": m.chapter_id,
                "name": m.name,
                "audience": m.audience,
                "recommended": m.recommended,
                "plain": m.plain,
                "expert": m.expert,
                "caution": m.caution,
                "operation": m.operation,
            }
            for m in METHODS
        ],
    }


@router.get("/recipes/shared")
async def shared_recipes(
    actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> list[dict[str, Any]]:
    return await cleaning.shared_recipes(session, actor)


@router.post("/recipes/{recipe_id}/visibility")
async def share_recipe(
    recipe_id: str,
    body: VisibilityIn,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await cleaning.share_recipe(session, actor, recipe_id, body.visibility)


@router.get("/{dataset_id}/studio")
async def studio(
    dataset_id: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Chapters with this dataset's findings, and each method with defaults, code and a before/after preview."""
    return await cleaning.studio(session, actor, dataset_id)


@router.post("/{dataset_id}/preview")
async def preview(
    dataset_id: str,
    body: PreviewIn,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await cleaning.preview(session, actor, dataset_id, body.method_id, body.params)


@router.get("/{dataset_id}/recipe")
async def recipe(
    dataset_id: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any] | None:
    """Your draft recipe if you may clean this dataset; otherwise the latest recipe, or null if there is none."""
    dataset = await get_visible(session, actor, dataset_id)
    current = await cleaning.recipe_for(session, actor, dataset)
    await session.commit()
    return cleaning.recipe_dict(current) if current else None


@router.post("/{dataset_id}/recipe/steps", status_code=status.HTTP_201_CREATED)
async def add_step(
    dataset_id: str,
    body: StepIn,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await cleaning.add_step(session, actor, dataset_id, body.method_id, body.params, body.status)


@router.patch("/{dataset_id}/recipe/steps/{step_id}")
async def update_step(
    dataset_id: str,
    step_id: str,
    body: StepUpdateIn,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await cleaning.update_step(session, actor, dataset_id, step_id, body.params, body.status)


@router.delete("/{dataset_id}/recipe/steps/{step_id}")
async def remove_step(
    dataset_id: str,
    step_id: str,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await cleaning.remove_step(session, actor, dataset_id, step_id)


@router.post("/{dataset_id}/recipe/reorder")
async def reorder(
    dataset_id: str,
    body: ReorderIn,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await cleaning.reorder(session, actor, dataset_id, body.step_ids)


@router.post("/{dataset_id}/recipe/import")
async def import_recipe(
    dataset_id: str,
    body: RecipeRefIn,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await cleaning.import_recipe(session, actor, dataset_id, body.recipe_id)


@router.post("/{dataset_id}/run")
async def run(
    dataset_id: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Run the whole recipe on the raw data and write the cleaned table to silver, pending approval."""
    return await cleaning.run(session, actor, dataset_id)


@router.get("/{dataset_id}/recipes")
async def history(
    dataset_id: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> list[dict[str, Any]]:
    return await cleaning.history(session, actor, dataset_id)


@router.post("/{dataset_id}/recipes/{recipe_id}/rerun")
async def rerun(
    dataset_id: str,
    recipe_id: str,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Rerun a saved recipe on the current raw version and report whether the output reproduced exactly."""
    return await cleaning.rerun(session, actor, dataset_id, recipe_id)
