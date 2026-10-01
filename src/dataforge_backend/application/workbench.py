"""The SQL workbench and DataForge notebooks: member SQL, validated and run in the sandbox."""

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.errors import Conflict, Forbidden, NotFound, Refused, ValidationFailed
from ..core.security import utcnow
from ..domain.roles import Permission
from ..domain.sql_guard import QueryRefused, guard_select
from ..infrastructure.db.models import Notebook, NotebookCell, User
from . import audit, catalogue, quotas
from .common import Actor, can_see, humanize, shared_ids, slugify
from .runtime import sandbox

DEFAULT_MAX_ROWS = 1000


def _refusal(sql: str, reason: str, suggestion: str = "") -> dict[str, Any]:
    return {
        "sql": sql,
        "refused": True,
        "explanation": f"{reason}{f' {suggestion}' if suggestion else ''}",
        "columns": [],
        "rows": [],
        "truncated": False,
    }


async def run_sql(
    session: AsyncSession, actor: Actor, sql: str, max_rows: int | None = None, *, source: str = "workbench"
) -> dict[str, Any]:
    """Validate against the member's permitted tables, then run read-only in the sandbox."""
    actor.require(Permission.RUN_SQL, "Your role cannot run SQL. Ask a question or build a chart instead.")
    actor.require_verified()
    await quotas.check_compute(session, actor)
    grants = await catalogue.grants_for(session, actor)
    try:
        guarded = guard_select(sql, grants)
    except QueryRefused as refusal:
        await audit.record(
            session,
            actor,
            "Refused query",
            "sandbox",
            None,
            source.capitalize(),
            refusal.reason,
            {"sql": sql[:4000]},
        )
        await session.commit()
        return _refusal(sql, refusal.reason, refusal.suggestion)
    limit = max(1, min(max_rows or DEFAULT_MAX_ROWS, 50_000))
    try:
        result = await sandbox.run_select(guarded.sql, user_id=actor.member_id, kind="sql", max_rows=limit)
    except Refused as error:
        await audit.record(
            session,
            actor,
            "Query failed",
            "sandbox",
            None,
            source.capitalize(),
            error.message,
            {"sql": sql[:4000]},
        )
        await session.commit()
        return _refusal(sql, error.message, str((error.details or {}).get("suggestion", "")))
    await quotas.add_compute(session, actor.member_id, result.duration_ms)
    await audit.record(
        session,
        actor,
        "Ran SQL",
        "sandbox",
        result.session_id,
        ", ".join(guarded.tables) or source.capitalize(),
        f"{len(result.rows)} rows in {result.duration_ms} ms",
        {"sql": sql[:4000]},
    )
    await session.commit()
    policies = (
        f" Row policies applied on {', '.join(guarded.policies_applied)}." if guarded.policies_applied else ""
    )
    return {
        "sql": sql,
        "executedSql": guarded.sql,
        "refused": False,
        "explanation": f"Ran under a least-privilege, read-only identity against {', '.join(guarded.tables) or 'no tables'}. "
        f"Personal columns stay masked.{policies}",
        "columns": result.columns,
        "rows": result.rows,
        "truncated": result.truncated,
        "durationMs": result.duration_ms,
        "sandboxSession": result.session_id,
    }


async def tables(session: AsyncSession, actor: Actor) -> list[dict[str, Any]]:
    """Tables the member may query, with their columns, for the workbench's schema browser."""
    out = []
    for dataset in await catalogue.queryable_datasets(session, actor):
        table = dataset.qualified_table()
        if not table:
            continue
        columns = (dataset.fields or {}).get("columns") or []
        policy = await catalogue.policy_for(session, dataset.id, actor.role)
        out.append(
            {
                "table": table,
                "datasetId": dataset.id,
                "name": dataset.name,
                "rows": dataset.rows,
                "rowPolicy": policy is not None,
                "columns": [
                    {"name": c["name"], "type": c["type"], "masked": c["name"] in (dataset.pii_columns or {})}
                    for c in columns
                ],
            }
        )
    return out


async def table_columns(session: AsyncSession, actor: Actor, table: str) -> list[dict[str, Any]]:
    for entry in await tables(session, actor):
        if entry["table"] == table:
            return [c for c in entry["columns"] if not c["masked"]]
    raise NotFound(f"{table} is not one of the tables you may read.")


# ------------------------------------------------------------------ notebooks


def cell_dict(cell: NotebookCell) -> dict[str, Any]:
    out: dict[str, Any] = {"id": cell.id, "kind": cell.kind, "source": cell.source, "revision": cell.revision}
    if cell.output is not None:
        out["output"] = cell.output
    return out


def notebook_dict(notebook: Notebook) -> dict[str, Any]:
    return {
        "id": notebook.id,
        "name": notebook.name,
        "owner": notebook.owner_label,
        "ownerId": notebook.owner_id,
        "visibility": notebook.visibility,
        "updated": humanize(notebook.updated_at),
        "updatedAt": notebook.updated_at.isoformat(),
        "jupyterPath": notebook.jupyter_path,
        "cells": [cell_dict(c) for c in notebook.cells],
    }


async def _visible_notebook(
    session: AsyncSession, actor: Actor, notebook_id: str, write: bool = False
) -> Notebook:
    notebook = await session.get(Notebook, notebook_id)
    shared = await shared_ids(session, actor, "notebook")
    owner = await session.get(User, notebook.owner_id) if notebook and notebook.owner_id else None
    if notebook is None or not can_see(
        actor, notebook.owner_id, notebook.visibility, owner.team if owner else None, shared, notebook.id
    ):
        raise NotFound("Notebook not found.")
    if write and notebook.owner_id != actor.id and notebook.id not in shared and actor.role != "admin":
        raise Forbidden("Only the owner and members it is shared with can edit this notebook.")
    await session.refresh(notebook, ["cells"])
    return notebook


async def list_notebooks(session: AsyncSession, actor: Actor) -> list[dict[str, Any]]:
    shared = await shared_ids(session, actor, "notebook")
    rows = await session.execute(select(Notebook).order_by(Notebook.updated_at.desc()))
    out = []
    for notebook in rows.scalars():
        owner = await session.get(User, notebook.owner_id) if notebook.owner_id else None
        if can_see(
            actor, notebook.owner_id, notebook.visibility, owner.team if owner else None, shared, notebook.id
        ):
            await session.refresh(notebook, ["cells"])
            out.append(notebook_dict(notebook))
    return out


async def get_notebook(session: AsyncSession, actor: Actor, notebook_id: str) -> dict[str, Any]:
    return notebook_dict(await _visible_notebook(session, actor, notebook_id))


async def create_notebook(
    session: AsyncSession,
    actor: Actor,
    name: str,
    cells: list[dict[str, Any]] | None = None,
    *,
    notebook_id: str | None = None,
    visibility: str = "private",
) -> dict[str, Any]:
    actor.require(Permission.WRITE_CODE, "Notebooks are for roles that write code.")
    if visibility not in ("private", "team", "organisation"):
        raise ValidationFailed("Visibility is private, team or organisation.")
    base = slugify(notebook_id or name or "notebook", 60)
    candidate, n = base, 2
    while await session.get(Notebook, candidate) is not None:
        candidate, n = f"{base[:55]}-{n}", n + 1
    notebook = Notebook(
        id=candidate,
        name=name.strip()[:200] or "Untitled notebook",
        owner_id=actor.id,
        owner_label=actor.name,
        visibility=visibility,
    )
    session.add(notebook)
    for index, cell in enumerate(cells or [{"kind": "markdown", "source": ""}]):
        kind = cell.get("kind", "sql")
        if kind not in ("markdown", "sql"):
            raise ValidationFailed("Cells are markdown or sql.")
        session.add(
            NotebookCell(
                notebook_id=notebook.id,
                position=index,
                kind=kind,
                source=str(cell.get("source", "")),
                updated_by=actor.id,
            )
        )
    await audit.record(session, actor, "Created notebook", "notebook", notebook.id, notebook.name)
    await session.commit()
    return await get_notebook(session, actor, notebook.id)


async def update_notebook(
    session: AsyncSession, actor: Actor, notebook_id: str, name: str | None, visibility: str | None
) -> dict[str, Any]:
    notebook = await _visible_notebook(session, actor, notebook_id, write=True)
    if name:
        notebook.name = name.strip()[:200]
    if visibility:
        if notebook.owner_id != actor.id:
            raise Forbidden("Only the owner changes who can see a notebook.")
        if visibility not in ("private", "team", "organisation"):
            raise ValidationFailed("Visibility is private, team or organisation.")
        notebook.visibility = visibility
    await audit.record(session, actor, "Updated notebook", "notebook", notebook.id, notebook.name)
    await session.commit()
    return await get_notebook(session, actor, notebook.id)


async def save_cell(
    session: AsyncSession,
    actor: Actor,
    notebook_id: str,
    cell_id: str | None,
    kind: str,
    source: str,
    revision: int | None,
    position: int | None,
) -> dict[str, Any]:
    """Create or edit a cell. Edits carry the revision they were based on, so concurrent edits never overwrite silently."""
    notebook = await _visible_notebook(session, actor, notebook_id, write=True)
    if kind not in ("markdown", "sql"):
        raise ValidationFailed("Cells are markdown or sql.")
    if cell_id:
        cell = next((c for c in notebook.cells if c.id == cell_id), None)
        if cell is None:
            raise NotFound("Cell not found.")
        if revision is not None and revision != cell.revision:
            raise Conflict(
                "Someone else changed this cell. Reload to see their version.",
                details={"current": cell_dict(cell)},
            )
        cell.kind, cell.source = kind, source
        cell.revision += 1
        cell.updated_by = actor.id
    else:
        index = len(notebook.cells) if position is None else max(0, min(position, len(notebook.cells)))
        for other in notebook.cells:
            if other.position >= index:
                other.position += 1
        cell = NotebookCell(
            notebook_id=notebook.id, position=index, kind=kind, source=source, updated_by=actor.id
        )
        session.add(cell)
    notebook.updated_at = utcnow()
    await session.commit()
    return await get_notebook(session, actor, notebook.id)


async def delete_cell(session: AsyncSession, actor: Actor, notebook_id: str, cell_id: str) -> dict[str, Any]:
    notebook = await _visible_notebook(session, actor, notebook_id, write=True)
    cell = next((c for c in notebook.cells if c.id == cell_id), None)
    if cell is None:
        raise NotFound("Cell not found.")
    await session.delete(cell)
    await session.flush()
    await session.refresh(notebook, ["cells"])
    for index, remaining in enumerate(notebook.cells):
        remaining.position = index
    await session.commit()
    return await get_notebook(session, actor, notebook.id)


async def run_cell(session: AsyncSession, actor: Actor, notebook_id: str, cell_id: str) -> dict[str, Any]:
    notebook = await _visible_notebook(session, actor, notebook_id)
    cell = next((c for c in notebook.cells if c.id == cell_id), None)
    if cell is None:
        raise NotFound("Cell not found.")
    if cell.kind != "sql":
        raise ValidationFailed("Only SQL cells run.")
    output = await run_sql(session, actor, cell.source, 500, source="notebook")
    cell = await session.get(NotebookCell, cell_id)
    assert cell is not None
    cell.output = {
        k: v
        for k, v in output.items()
        if k in ("sql", "refused", "explanation", "columns", "rows", "truncated")
    }
    await session.commit()
    return output


async def delete_notebook(session: AsyncSession, actor: Actor, notebook_id: str) -> None:
    notebook = await _visible_notebook(session, actor, notebook_id)
    if notebook.owner_id != actor.id and actor.role != "admin":
        raise Forbidden("Only the owner can delete a notebook.")
    await session.delete(notebook)
    await audit.record(session, actor, "Deleted notebook", "notebook", notebook_id, notebook.name)
    await session.commit()
