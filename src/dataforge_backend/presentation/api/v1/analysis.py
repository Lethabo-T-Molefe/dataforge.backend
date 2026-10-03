from typing import Any

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from ....application import analysis, assistant, workbench
from ....application.common import Actor
from ..deps import current_actor, data_actor, get_db
from ..schemas import AskIn, CellIn, ExplorationIn, FreeBody, NotebookIn, NotebookUpdateIn, SqlIn

router = APIRouter(tags=["Analysis"])


# ------------------------------------------------------------------ point-and-click analysis


@router.get("/analysis/catalogue")
async def analysis_catalogue(
    actor: Actor = Depends(data_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Methods, modes, and every approved dataset's measures, dimensions, grains and months."""
    return await analysis.catalogue_for(session, actor)


@router.get("/analysis/defaults")
async def analysis_defaults(
    dataset: str,
    method: str = "compare",
    measure: str | None = None,
    dimension: str | None = None,
    actor: Actor = Depends(data_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Sensible starting settings for a method on a dataset, optionally seeded with a measure and dimension."""
    return await analysis.defaults(
        session, actor, dataset, method, {"measure": measure, "dimension": dimension}
    )


@router.post("/analysis/run")
async def run_analysis(
    body: FreeBody, actor: Actor = Depends(data_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Run one method. Returns the reading, method, caveats, chart, table, figures and the SQL behind them."""
    return await analysis.run(session, actor, body.model_dump(by_alias=False))


@router.get("/explorations")
async def explorations(
    actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> list[dict[str, Any]]:
    return await analysis.list_explorations(session, actor)


@router.post("/explorations", status_code=status.HTTP_201_CREATED)
async def save_exploration(
    body: ExplorationIn, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    return await analysis.save_exploration(session, actor, body.name, body.config)


@router.delete("/explorations/{exploration_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_exploration(
    exploration_id: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> None:
    await analysis.delete_exploration(session, actor, exploration_id)


# ------------------------------------------------------------------ assistant


@router.post("/assistant/ask")
async def ask(
    body: AskIn, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Ask in plain English. The answer comes with the SQL that ran and an explanation, or a clear refusal.

    `contextId` is the id of the member's previous answer, so a follow-up ("what about Gauteng?") builds on it.
    """
    return await assistant.ask(session, actor, body.question, context_id=body.context_id)


@router.get("/assistant/history")
async def assistant_history(
    actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> list[dict[str, Any]]:
    return await assistant.history(session, actor)


@router.get("/assistant/suggestions")
async def assistant_suggestions(
    actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> list[str]:
    return await assistant.suggestions(session, actor)


# ------------------------------------------------------------------ SQL workbench


@router.post("/sql/run")
async def run_sql(
    body: SqlIn, actor: Actor = Depends(data_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Run a read-only query over your permitted tables in the sandbox. Refusals explain why."""
    return await workbench.run_sql(session, actor, body.sql, body.max_rows)


@router.get("/sql/tables")
async def sql_tables(
    actor: Actor = Depends(data_actor), session: AsyncSession = Depends(get_db)
) -> list[dict[str, Any]]:
    return await workbench.tables(session, actor)


@router.get("/sql/tables/{table}/columns")
async def sql_table_columns(
    table: str, actor: Actor = Depends(data_actor), session: AsyncSession = Depends(get_db)
) -> list[dict[str, Any]]:
    return await workbench.table_columns(session, actor, table)


# ------------------------------------------------------------------ notebooks


@router.get("/notebooks")
async def notebooks(
    actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> list[dict[str, Any]]:
    return await workbench.list_notebooks(session, actor)


@router.post("/notebooks", status_code=status.HTTP_201_CREATED)
async def create_notebook(
    body: NotebookIn, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    return await workbench.create_notebook(session, actor, body.name, body.cells)


@router.get("/notebooks/{notebook_id}")
async def get_notebook(
    notebook_id: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    return await workbench.get_notebook(session, actor, notebook_id)


@router.patch("/notebooks/{notebook_id}")
async def update_notebook(
    notebook_id: str,
    body: NotebookUpdateIn,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await workbench.update_notebook(session, actor, notebook_id, body.name, body.visibility)


@router.delete("/notebooks/{notebook_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_notebook(
    notebook_id: str, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> None:
    await workbench.delete_notebook(session, actor, notebook_id)


@router.post("/notebooks/{notebook_id}/cells", status_code=status.HTTP_201_CREATED)
async def add_cell(
    notebook_id: str,
    body: CellIn,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await workbench.save_cell(
        session, actor, notebook_id, None, body.kind, body.source, None, body.position
    )


@router.put("/notebooks/{notebook_id}/cells/{cell_id}")
async def save_cell(
    notebook_id: str,
    cell_id: str,
    body: CellIn,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Edit a cell. Send the revision you edited; a newer revision from someone else returns 409 with theirs."""
    return await workbench.save_cell(
        session, actor, notebook_id, cell_id, body.kind, body.source, body.revision, None
    )


@router.delete("/notebooks/{notebook_id}/cells/{cell_id}")
async def delete_cell(
    notebook_id: str,
    cell_id: str,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await workbench.delete_cell(session, actor, notebook_id, cell_id)


@router.post("/notebooks/{notebook_id}/cells/{cell_id}/run")
async def run_cell(
    notebook_id: str,
    cell_id: str,
    actor: Actor = Depends(current_actor),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return await workbench.run_cell(session, actor, notebook_id, cell_id)
