"""Guided cleaning: the studio, the draft recipe, previews on a sample, and the full run to silver.

Bronze is never changed. A recipe is an ordered list of recorded steps; running it re-derives silver from
bronze, so cleaning is reversible and a saved recipe rerun on the same input gives the same output (the
output checksum is recorded to prove it).
"""

import hashlib
import json
from collections import OrderedDict
from typing import Any

import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..core.errors import Forbidden, NotFound, ValidationFailed
from ..core.security import utcnow
from ..domain.analysis.sql import ident
from ..domain.cleaning import (
    CHAPTERS,
    CHAPTERS_BY_ID,
    METHODS,
    CleaningContext,
    Method,
    OperationError,
    RecipeStepSpec,
    apply_recipe,
    chapter_finding,
    code_for,
    enforce_masking,
    is_applicable,
    method_or_error,
    preview_step,
    type_for_silver,
)
from ..domain.governance import DatasetStatus, Layer
from ..domain.pii import detect_pii
from ..domain.profiling import profile_frame
from ..domain.roles import Permission
from ..infrastructure import warehouse
from ..infrastructure.db.models import CleaningRecipe, CleaningStep, Dataset, LineageEdge
from . import audit, catalogue, quotas
from .common import Actor, humanize
from .datasets import get_visible, record_version, sync_superset_later
from .runtime import sandbox

SAMPLE_ROWS = 400
_STUDIO_CACHE: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_CACHE_SIZE = 32


# ------------------------------------------------------------------ helpers


def _bronze_table(dataset: Dataset) -> str:
    return f"bronze.{dataset.table_name}_v{dataset.current_version}"


def _salt(dataset: Dataset) -> str:
    """Pseudonyms are stable within a dataset and differ between datasets."""
    key = settings.SECRETS_KEY or settings.JWT_SECRET
    return hashlib.sha256(f"{key}:{dataset.id}".encode()).hexdigest()[:32]


async def reference_sets(
    session: AsyncSession, actor: Actor, columns: list[str], exclude: str | None = None
) -> dict[str, set[str]]:
    """Master key lists from approved datasets that share a key-like column name, for referential checks."""
    keyish = [c for c in columns if c.endswith(("_id", "_code", "_key"))]
    if not keyish:
        return {}
    out: dict[str, set[str]] = {}
    conn = await session.connection()
    for dataset in await catalogue.queryable_datasets(session, actor):
        if dataset.id == exclude or not dataset.fields:
            continue
        names = {c["name"] for c in dataset.fields.get("columns", [])}
        table = dataset.qualified_table()
        for column in keyish:
            if column in names and table:
                result = await conn.execute(
                    text(
                        f"SELECT DISTINCT {ident(column)}::text FROM {table} WHERE {ident(column)} IS NOT NULL LIMIT 200000"
                    )
                )
                out[f"{dataset.id}.{column}"] = {str(r[0]) for r in result}
    return out


async def _context(
    session: AsyncSession, actor: Actor, dataset: Dataset, columns: list[str]
) -> CleaningContext:
    return CleaningContext(
        pii_columns=dict(dataset.pii_columns or {}),
        references=await reference_sets(session, actor, columns, exclude=dataset.id),
        seed=0,
        salt=_salt(dataset),
    )


def _requested_method(method_id: str) -> Method:
    """A method named by the caller. Stored steps use method_or_error: their ids were checked when saved."""
    try:
        return method_or_error(method_id)
    except OperationError as error:
        raise NotFound(str(error)) from error


def _can_clean(actor: Actor, dataset: Dataset) -> None:
    actor.require(Permission.CLEAN, "Your role cannot clean data.")
    actor.require_verified()
    if dataset.owner_id != actor.id and not actor.can(Permission.APPROVE):
        raise Forbidden("Only the owner or a steward can clean this dataset.")


def may_clean(actor: Actor, dataset: Dataset) -> bool:
    """Whether _can_clean would pass, for deciding what to show rather than what to refuse."""
    return (
        actor.can(Permission.CLEAN)
        and actor.verified
        and (dataset.owner_id == actor.id or actor.can(Permission.APPROVE))
    )


def step_dict(step: CleaningStep) -> dict[str, Any]:
    method = method_or_error(step.method_id)
    chapter = CHAPTERS_BY_ID[method.chapter_id]
    return {
        "stepId": step.id,
        "methodId": step.method_id,
        "chapterId": step.chapter_id,
        "title": chapter.title,
        "option": method.name,
        "status": step.status,
        "params": step.params,
        "summary": step.summary,
        "code": code_for(method.operation, step.params),
        "position": step.position,
    }


def recipe_dict(recipe: CleaningRecipe) -> dict[str, Any]:
    return {
        "id": recipe.id,
        "datasetId": recipe.dataset_id,
        "version": recipe.version,
        "name": recipe.name,
        "status": recipe.status,
        "visibility": recipe.visibility,
        "steps": [step_dict(s) for s in recipe.steps],
        "savedAt": (recipe.applied_at or recipe.updated_at).isoformat(),
        "saved": humanize(recipe.applied_at or recipe.updated_at),
        "outputTable": recipe.output_table,
        "outputRows": recipe.output_rows,
        "outputChecksum": recipe.output_checksum,
        "code": recipe_code(recipe),
    }


def recipe_code(recipe: CleaningRecipe) -> str:
    """The whole recipe as one pandas script, so an expert can take it into a notebook."""
    parts = [
        "import pandas as pd",
        "",
        f"# Cleaning recipe v{recipe.version} for {recipe.dataset_id}",
        "df = raw.copy()",
    ]
    for step in recipe.steps:
        if step.status != "accepted":
            continue
        method = method_or_error(step.method_id)
        parts += ["", f"# {step.position + 1}. {method.name}", code_for(method.operation, step.params)]
    return "\n".join(parts)


async def draft_recipe(session: AsyncSession, actor: Actor, dataset: Dataset) -> CleaningRecipe:
    """The editable recipe. After a run, the next edit starts a new version from the applied one."""
    rows = await session.execute(
        select(CleaningRecipe)
        .where(CleaningRecipe.dataset_id == dataset.id)
        .order_by(CleaningRecipe.version.desc())
    )
    latest = rows.scalars().first()
    if latest is not None and latest.status == "draft":
        await session.refresh(latest, ["steps"])
        return latest
    recipe = CleaningRecipe(
        dataset_id=dataset.id,
        version=(latest.version + 1) if latest else 1,
        name=f"{dataset.name} recipe",
        created_by=actor.id,
    )
    session.add(recipe)
    await session.flush()
    if latest is not None:
        await session.refresh(latest, ["steps"])
        for step in latest.steps:
            session.add(
                CleaningStep(
                    recipe_id=recipe.id,
                    position=step.position,
                    method_id=step.method_id,
                    chapter_id=step.chapter_id,
                    params=step.params,
                    status=step.status,
                    summary=step.summary,
                )
            )
        await session.flush()
    await session.refresh(recipe, ["steps"])
    return recipe


async def recipe_for(session: AsyncSession, actor: Actor, dataset: Dataset) -> CleaningRecipe | None:
    """The recipe a member works from: their draft if they may clean, otherwise the latest one, read-only.

    Looking at a dataset must not write anything, so members who cannot clean it never start a draft.
    """
    if may_clean(actor, dataset):
        return await draft_recipe(session, actor, dataset)
    rows = await session.execute(
        select(CleaningRecipe)
        .where(CleaningRecipe.dataset_id == dataset.id)
        .order_by(CleaningRecipe.version.desc())
    )
    latest = rows.scalars().first()
    if latest is not None:
        await session.refresh(latest, ["steps"])
    return latest


def _specs(recipe: CleaningRecipe | None) -> list[RecipeStepSpec]:
    return [RecipeStepSpec(s.method_id, s.params, s.status) for s in recipe.steps] if recipe else []


def _signature(dataset: Dataset, recipe: CleaningRecipe | None) -> str:
    steps = recipe.steps if recipe else []
    body = json.dumps([[s.method_id, s.params, s.status] for s in steps], sort_keys=True, default=str)
    return f"{dataset.id}:{dataset.current_version}:{hashlib.sha1(body.encode()).hexdigest()}"


def _sample_profile(frame: pd.DataFrame, ctx: CleaningContext) -> dict[str, Any]:
    findings = [f for f in detect_pii(frame) if f.column in ctx.pii_columns] or detect_pii(frame)
    references = {
        name.split(".", 1)[1]: (name, keys)
        for name, keys in ctx.references.items()
        if name.split(".", 1)[1] in frame.columns
    }
    profile, _ = profile_frame(frame, findings, references)
    return profile


def _build_studio(
    bronze: pd.DataFrame,
    recipe_specs: list[RecipeStepSpec],
    ctx: CleaningContext,
    full_profile: dict[str, Any],
) -> dict[str, Any]:
    """Methods with their defaults, code and a preview on the sample as the recipe so far has left it."""
    sample = bronze.head(SAMPLE_ROWS).copy()
    current, _ = apply_recipe(sample, recipe_specs, ctx) if recipe_specs else (sample, [])
    # Defaults come from the profile of the data as it stands after the recipe so far.
    profile = _sample_profile(current, ctx) if recipe_specs else full_profile
    available = list(current.columns)
    methods: list[dict[str, Any]] = []
    for method in METHODS:
        params = method.defaults(profile, available)
        applicable = is_applicable(method, params)
        preview: dict[str, Any] = {"before": [], "after": []}
        error = None
        if applicable:
            try:
                preview = preview_step(current, method, params, ctx)
            except OperationError as exc:
                applicable, error = False, str(exc)
        methods.append(
            {
                "id": method.id,
                "chapterId": method.chapter_id,
                "name": method.name,
                "audience": method.audience,
                "recommended": method.recommended,
                "plain": method.plain,
                "expert": method.expert,
                "caution": method.caution,
                "operation": method.operation,
                "params": params,
                "applicable": applicable,
                "unavailableReason": error
                or (None if applicable else "Nothing in this data needs this method."),
                "code": code_for(method.operation, params),
                "preview": preview,
                "affected": ", ".join(preview.get("affected") or []) or "—",
            }
        )
    chapters = [
        {
            "id": c.id,
            "kind": c.kind,
            "number": c.number,
            "title": c.title,
            "plain": c.plain,
            "expert": c.expert,
            "finding": chapter_finding(c.id, profile),
            "recommendedMethodId": c.recommended_method_id,
        }
        for c in CHAPTERS
    ]
    return {"chapters": chapters, "methods": methods, "sampleSize": len(sample), "columnsNow": available}


async def _load_bronze(session: AsyncSession, dataset: Dataset, limit: int | None = None) -> pd.DataFrame:
    conn = await session.connection()
    table = _bronze_table(dataset)
    if not await warehouse.table_exists(conn, table):
        raise NotFound("The raw table for this dataset is missing.")
    return await warehouse.read_frame(conn, table, limit)


# ------------------------------------------------------------------ studio and recipe


async def studio(session: AsyncSession, actor: Actor, dataset_id: str) -> dict[str, Any]:
    dataset = await get_visible(session, actor, dataset_id)
    recipe = await recipe_for(session, actor, dataset)
    await session.commit()
    key = _signature(dataset, recipe)
    cached = _STUDIO_CACHE.get(key)
    if cached is None:
        bronze = await _load_bronze(session, dataset, SAMPLE_ROWS)
        ctx = await _context(session, actor, dataset, list(bronze.columns))
        specs = _specs(recipe)
        built, elapsed = await sandbox.run_python(
            lambda: _build_studio(bronze, specs, ctx, dataset.profile or {}),
            user_id=actor.member_id,
            kind="cleaning",
            description=f"Cleaning studio for {dataset.name}",
            timeout_s=120,
        )
        await quotas.add_compute(session, actor.member_id, elapsed)
        await session.commit()
        _STUDIO_CACHE[key] = built
        cached = built
        while len(_STUDIO_CACHE) > _CACHE_SIZE:
            _STUDIO_CACHE.popitem(last=False)
    return {
        **cached,
        "profile": dataset.profile,
        "recipe": recipe_dict(recipe) if recipe else None,
        "datasetId": dataset.id,
        "layer": dataset.layer,
        "canEdit": may_clean(actor, dataset),
    }


async def preview(
    session: AsyncSession, actor: Actor, dataset_id: str, method_id: str, params: dict[str, Any] | None
) -> dict[str, Any]:
    dataset = await get_visible(session, actor, dataset_id)
    method = _requested_method(method_id)
    recipe = await recipe_for(session, actor, dataset)
    bronze = await _load_bronze(session, dataset, SAMPLE_ROWS)
    ctx = await _context(session, actor, dataset, list(bronze.columns))
    specs = _specs(recipe)

    def run() -> dict[str, Any]:
        current, _ = apply_recipe(bronze, specs, ctx)
        chosen = (
            params
            if params is not None
            else method.defaults(_sample_profile(current, ctx), list(current.columns))
        )
        result = preview_step(current, method, chosen, ctx)
        return {**result, "params": chosen, "code": code_for(method.operation, chosen)}

    try:
        result, elapsed = await sandbox.run_python(
            run, user_id=actor.member_id, kind="cleaning", description=f"Preview {method.name}", timeout_s=120
        )
    except OperationError as error:
        raise ValidationFailed(str(error)) from error
    await quotas.add_compute(session, actor.member_id, elapsed)
    await session.commit()
    return result


async def add_step(
    session: AsyncSession,
    actor: Actor,
    dataset_id: str,
    method_id: str,
    params: dict[str, Any] | None,
    status: str,
) -> dict[str, Any]:
    dataset = await get_visible(session, actor, dataset_id)
    _can_clean(actor, dataset)
    if status not in ("accepted", "skipped"):
        raise ValidationFailed("A step is accepted or skipped.")
    method = _requested_method(method_id)
    if status == "accepted":
        checked = await preview(session, actor, dataset_id, method_id, params)
    else:
        # A skipped step changes nothing, so it is recorded without being tried on the data. That keeps
        # "skip" available when the method has nothing to act on (no orphans to flag, no column to drop).
        checked = {"params": params or {}, "summary": "Skipped"}
    recipe = await draft_recipe(session, actor, dataset)
    step = CleaningStep(
        recipe_id=recipe.id,
        position=len(recipe.steps),
        method_id=method.id,
        chapter_id=method.chapter_id,
        params=checked["params"],
        status=status,
        summary=checked["summary"] if status == "accepted" else "Skipped",
    )
    session.add(step)
    recipe.updated_at = utcnow()
    verb = "Applied cleaning step" if status == "accepted" else "Skipped cleaning step"
    await audit.record(
        session,
        actor,
        verb,
        "dataset",
        dataset.id,
        dataset.name,
        f"{CHAPTERS_BY_ID[method.chapter_id].title} · {method.name}",
        {"params": checked["params"]},
    )
    await session.commit()
    await session.refresh(recipe, ["steps"])
    return recipe_dict(recipe)


async def _own_step(
    session: AsyncSession, actor: Actor, dataset_id: str, step_id: str
) -> tuple[Dataset, CleaningRecipe, CleaningStep]:
    dataset = await get_visible(session, actor, dataset_id)
    _can_clean(actor, dataset)
    recipe = await draft_recipe(session, actor, dataset)
    step = next((s for s in recipe.steps if s.id == step_id), None)
    if step is None:
        # A step id from the applied version: the draft copied it under a new id at the same position.
        raise NotFound("That step is not in the current draft recipe. Reload the recipe.")
    return dataset, recipe, step


async def update_step(
    session: AsyncSession,
    actor: Actor,
    dataset_id: str,
    step_id: str,
    params: dict[str, Any] | None,
    status: str | None,
) -> dict[str, Any]:
    dataset, recipe, step = await _own_step(session, actor, dataset_id, step_id)
    if params is None and status == "accepted" and step.status != "accepted":
        # A skipped step was never tried on the data, so accepting it checks its settings first.
        params = step.params
    if params is not None:
        method = method_or_error(step.method_id)
        bronze = await _load_bronze(session, dataset, SAMPLE_ROWS)
        ctx = await _context(session, actor, dataset, list(bronze.columns))
        earlier = [
            RecipeStepSpec(s.method_id, s.params, s.status)
            for s in recipe.steps
            if s.position < step.position
        ]

        def check() -> dict[str, Any]:
            current, _ = apply_recipe(bronze, earlier, ctx)
            return preview_step(current, method, params, ctx)

        try:
            result, _ = await sandbox.run_python(
                check, user_id=actor.member_id, kind="cleaning", description=f"Check {method.name}"
            )
        except OperationError as error:
            raise ValidationFailed(str(error)) from error
        step.params = params
        step.summary = result["summary"]
    if status is not None:
        if status not in ("accepted", "skipped"):
            raise ValidationFailed("A step is accepted or skipped.")
        step.status = status
    recipe.updated_at = utcnow()
    await audit.record(
        session,
        actor,
        "Changed cleaning step",
        "dataset",
        dataset.id,
        dataset.name,
        f"{step.method_id} ({step.status})",
    )
    await session.commit()
    await session.refresh(recipe, ["steps"])
    return recipe_dict(recipe)


async def remove_step(session: AsyncSession, actor: Actor, dataset_id: str, step_id: str) -> dict[str, Any]:
    dataset, recipe, step = await _own_step(session, actor, dataset_id, step_id)
    await session.delete(step)
    await session.flush()
    await session.refresh(recipe, ["steps"])
    for index, remaining in enumerate(recipe.steps):
        remaining.position = index
    recipe.updated_at = utcnow()
    await audit.record(
        session, actor, "Undid cleaning step", "dataset", dataset.id, dataset.name, step.method_id
    )
    await session.commit()
    await session.refresh(recipe, ["steps"])
    return recipe_dict(recipe)


async def reorder(
    session: AsyncSession, actor: Actor, dataset_id: str, step_ids: list[str]
) -> dict[str, Any]:
    dataset = await get_visible(session, actor, dataset_id)
    _can_clean(actor, dataset)
    recipe = await draft_recipe(session, actor, dataset)
    by_id = {s.id: s for s in recipe.steps}
    if sorted(step_ids) != sorted(by_id):
        raise ValidationFailed("List every step of the recipe exactly once.")
    for index, step_id in enumerate(step_ids):
        by_id[step_id].position = index
    recipe.updated_at = utcnow()
    await audit.record(session, actor, "Reordered cleaning steps", "dataset", dataset.id, dataset.name)
    await session.commit()
    await session.refresh(recipe, ["steps"])
    recipe.steps.sort(key=lambda s: s.position)
    return recipe_dict(recipe)


async def import_recipe(
    session: AsyncSession, actor: Actor, dataset_id: str, recipe_id: str
) -> dict[str, Any]:
    """Start this dataset's draft from another recipe the member can see (shared recipes are reusable)."""
    dataset = await get_visible(session, actor, dataset_id)
    _can_clean(actor, dataset)
    source = await session.get(CleaningRecipe, recipe_id)
    if source is None:
        raise NotFound("Recipe not found.")
    source_dataset = await session.get(Dataset, source.dataset_id)
    visible = (
        source.visibility == "organisation"
        or source.created_by == actor.id
        or (source.visibility == "team" and source_dataset is not None and source_dataset.team == actor.team)
    )
    if not visible:
        raise NotFound("Recipe not found.")
    await session.refresh(source, ["steps"])
    recipe = await draft_recipe(session, actor, dataset)
    for step in list(recipe.steps):
        await session.delete(step)
    await session.flush()
    for step in source.steps:
        session.add(
            CleaningStep(
                recipe_id=recipe.id,
                position=step.position,
                method_id=step.method_id,
                chapter_id=step.chapter_id,
                params=step.params,
                status=step.status,
                summary=step.summary,
            )
        )
    await audit.record(
        session,
        actor,
        "Reused cleaning recipe",
        "dataset",
        dataset.id,
        dataset.name,
        f"From recipe {source.id} v{source.version}",
    )
    await session.commit()
    await session.refresh(recipe, ["steps"])
    return recipe_dict(recipe)


async def share_recipe(
    session: AsyncSession, actor: Actor, recipe_id: str, visibility: str
) -> dict[str, Any]:
    recipe = await session.get(CleaningRecipe, recipe_id)
    if recipe is None or (recipe.created_by != actor.id and actor.role != "admin"):
        raise NotFound("Recipe not found.")
    if visibility not in ("private", "team", "organisation"):
        raise ValidationFailed("Visibility is private, team or organisation.")
    recipe.visibility = visibility
    await audit.record(
        session, actor, "Shared recipe", "recipe", recipe.id, recipe.name, f"Visibility {visibility}"
    )
    await session.commit()
    await session.refresh(recipe, ["steps"])
    return recipe_dict(recipe)


async def history(session: AsyncSession, actor: Actor, dataset_id: str) -> list[dict[str, Any]]:
    await get_visible(session, actor, dataset_id)
    rows = await session.execute(
        select(CleaningRecipe)
        .where(CleaningRecipe.dataset_id == dataset_id)
        .order_by(CleaningRecipe.version.desc())
    )
    out = []
    for recipe in rows.scalars():
        await session.refresh(recipe, ["steps"])
        out.append(recipe_dict(recipe))
    return out


async def shared_recipes(session: AsyncSession, actor: Actor) -> list[dict[str, Any]]:
    rows = await session.execute(
        select(CleaningRecipe).where(
            CleaningRecipe.status == "applied", CleaningRecipe.visibility != "private"
        )
    )
    out = []
    for recipe in rows.scalars():
        dataset = await session.get(Dataset, recipe.dataset_id)
        if recipe.visibility == "team" and (dataset is None or dataset.team != actor.team):
            continue
        await session.refresh(recipe, ["steps"])
        out.append({**recipe_dict(recipe), "datasetName": dataset.name if dataset else recipe.dataset_id})
    return out


# ------------------------------------------------------------------ the full run


def frame_checksum(frame: pd.DataFrame) -> str:
    """A stable fingerprint of a table's content, to show a rerun reproduces the same output."""
    digest = hashlib.sha256()
    digest.update("|".join(map(str, frame.columns)).encode())
    for row in frame.itertuples(index=False, name=None):
        digest.update(("\x1f".join("" if v is None else str(v) for v in row) + "\n").encode())
    return digest.hexdigest()


def _clean_full(bronze: pd.DataFrame, specs: list[RecipeStepSpec], ctx: CleaningContext) -> dict[str, Any]:
    out, outcomes = apply_recipe(bronze, specs, ctx)
    out, masked = enforce_masking(out, ctx.pii_columns)
    typed, types = type_for_silver(out.reset_index(drop=True))
    profile, _ = profile_frame(typed.astype(object).where(typed.notna(), None))
    return {
        "frame": typed,
        "types": types,
        "masked": masked,
        "outcomes": [(o.summary, o.rows_before, o.rows_after, o.cells_changed) for o in outcomes],
        "checksum": frame_checksum(typed),
        "quality": profile["quality"],
    }


async def _apply(
    session: AsyncSession, actor: Actor, dataset: Dataset, recipe: CleaningRecipe
) -> dict[str, Any]:
    bronze = await _load_bronze(session, dataset)
    ctx = await _context(session, actor, dataset, list(bronze.columns))
    specs = _specs(recipe)
    await quotas.check_compute(session, actor)
    try:
        result, elapsed = await sandbox.run_python(
            lambda: _clean_full(bronze, specs, ctx),
            user_id=actor.member_id,
            kind="cleaning",
            description=f"Run recipe v{recipe.version} on {dataset.name}",
            timeout_s=900,
        )
    except OperationError as error:
        raise ValidationFailed(f"A step could not run on the full data: {error}") from error
    await quotas.add_compute(session, actor.member_id, elapsed)
    frame: pd.DataFrame = result["frame"]
    if frame.empty or not len(frame.columns):
        raise ValidationFailed("The recipe removed every row or column. Undo a step and run again.")

    conn = await session.connection()
    silver = f"silver.{dataset.table_name}"
    was_granted = dataset.status == DatasetStatus.APPROVED.value
    rows = await warehouse.write_typed(conn, "silver", dataset.table_name, frame, result["types"])
    if dataset.layer == Layer.GOLD.value:
        await warehouse.drop_table(conn, f"gold.{dataset.table_name}")
    dataset.layer = Layer.SILVER.value
    # New cleaned data needs a fresh decision before anyone can query it.
    dataset.status = DatasetStatus.PENDING.value
    dataset.decided_at = None
    dataset.decision_reason = None
    dataset.fields = None
    dataset.quality = {**(dataset.quality or {}), "afterCleaning": result["quality"]}
    dataset.pii_columns = {c: k for c, k in (dataset.pii_columns or {}).items() if c in frame.columns}
    dataset.size_bytes = await warehouse.table_size(conn, silver) + await warehouse.table_size(
        conn, _bronze_table(dataset)
    )
    if was_granted:
        await warehouse.revoke_sandbox(conn, silver)

    recipe.status = "applied"
    recipe.applied_at = utcnow()
    recipe.output_table = silver
    recipe.output_rows = rows
    recipe.output_checksum = result["checksum"]
    await record_version(
        session,
        dataset,
        "silver",
        table=silver,
        rows=rows,
        column_types=result["types"],
        checksum=result["checksum"],
        recipe_id=recipe.id,
        created_by=actor.id,
    )
    session.add(
        LineageEdge(
            dataset_id=dataset.id,
            source=_bronze_table(dataset),
            target=silver,
            transform="clean",
            recipe_id=recipe.id,
            detail=f"Recipe v{recipe.version}, {len([s for s in specs if s.status == 'accepted'])} steps, {rows} rows",
            actor_id=actor.id,
        )
    )
    await audit.record(
        session,
        actor,
        "Ran cleaning recipe",
        "dataset",
        dataset.id,
        dataset.name,
        f"Recipe v{recipe.version} wrote {rows} rows to {silver}; masked {', '.join(result['masked']) or 'no new columns'}",
        {"checksum": result["checksum"], "steps": [o[0] for o in result["outcomes"]]},
    )
    await session.commit()
    await sync_superset_later(dataset.id)
    return {
        "recipe": recipe_dict(recipe),
        "rows": rows,
        "table": silver,
        "types": result["types"],
        "masked": result["masked"],
        "checksum": result["checksum"],
        "quality": result["quality"],
        "steps": [
            {"summary": s, "rowsBefore": b, "rowsAfter": a, "cellsChanged": c}
            for s, b, a, c in result["outcomes"]
        ],
        "nextStep": "Ask a Data Steward to approve the cleaned data.",
    }


async def run(session: AsyncSession, actor: Actor, dataset_id: str) -> dict[str, Any]:
    dataset = await get_visible(session, actor, dataset_id)
    _can_clean(actor, dataset)
    recipe = await draft_recipe(session, actor, dataset)
    if not any(s.status == "accepted" for s in recipe.steps):
        raise ValidationFailed("Accept at least one step before running the recipe.")
    return await _apply(session, actor, dataset, recipe)


async def rerun(session: AsyncSession, actor: Actor, dataset_id: str, recipe_id: str) -> dict[str, Any]:
    """Rerun an applied recipe on the current raw version. Same input, same recipe, same checksum."""
    dataset = await get_visible(session, actor, dataset_id)
    _can_clean(actor, dataset)
    source = await session.get(CleaningRecipe, recipe_id)
    if source is None or source.dataset_id != dataset.id or source.status != "applied":
        raise NotFound("Applied recipe not found for this dataset.")
    await session.refresh(source, ["steps"])
    previous = source.output_checksum
    recipe = CleaningRecipe(
        dataset_id=dataset.id,
        version=int(
            (
                await session.execute(
                    select(CleaningRecipe.version)
                    .where(CleaningRecipe.dataset_id == dataset.id)
                    .order_by(CleaningRecipe.version.desc())
                )
            )
            .scalars()
            .first()
            or 0
        )
        + 1,
        name=source.name,
        created_by=actor.id,
    )
    session.add(recipe)
    await session.flush()
    # A leftover draft would be the newest version; it is replaced by this rerun.
    drafts = await session.execute(
        select(CleaningRecipe).where(
            CleaningRecipe.dataset_id == dataset.id,
            CleaningRecipe.status == "draft",
            CleaningRecipe.id != recipe.id,
        )
    )
    for draft in drafts.scalars():
        await session.delete(draft)
    for step in source.steps:
        session.add(
            CleaningStep(
                recipe_id=recipe.id,
                position=step.position,
                method_id=step.method_id,
                chapter_id=step.chapter_id,
                params=step.params,
                status=step.status,
                summary=step.summary,
            )
        )
    await session.flush()
    await session.refresh(recipe, ["steps"])
    result = await _apply(session, actor, dataset, recipe)
    result["reproduced"] = previous == result["checksum"] if previous else None
    result["previousChecksum"] = previous
    if previous and previous != result["checksum"]:
        result["note"] = "The output differs from the earlier run because the raw data has a new version."
    return result
