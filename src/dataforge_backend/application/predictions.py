"""Guided scoring: documented model versions, cohort scores with factor breakdowns, and exports.

Every model version, heuristic or trained, starts as a draft. A second person approves it before anyone can
score with it: someone with the approve-models permission (a Data Steward or an Administrator) or another
Model Curator, and never the person who built or trained that version.
"""

import io
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.errors import Conflict, Forbidden, NotFound, ValidationFailed
from ..core.security import utcnow
from ..domain.analysis import AnalysisDataset
from ..domain.analysis.stats import median, quantile, standard_deviation
from ..domain.prediction import (
    FactorInput,
    FactorSpec,
    ModelError,
    ModelSpec,
    factor_from_dict,
    score_cohort,
    validate_model,
)
from ..domain.roles import Permission
from ..infrastructure.db.models import ModelVersion, Prediction, User
from . import analysis, audit, catalogue, inbox
from .common import Actor, humanize

STATUSES = ("draft", "active", "retired", "rejected")
_MIN_NOTE = 20


def is_reviewer(actor: Actor) -> bool:
    """Who sees drafts and decides on them: curators, and members with the approve-models permission."""
    return actor.can(Permission.CURATE_MODELS) or actor.can(Permission.APPROVE_MODELS)


async def _names(session: AsyncSession, ids: set[str]) -> dict[str, str]:
    ids = {i for i in ids if i}
    if not ids:
        return {}
    rows = await session.execute(select(User.id, User.name).where(User.id.in_(ids)))
    return {r[0]: r[1] for r in rows}


def model_spec(model: ModelVersion) -> ModelSpec:
    return ModelSpec(
        name=model.name,
        version=model.version,
        outcome=model.outcome,
        dataset_id=model.dataset_id,
        factors=tuple(factor_from_dict(f) for f in model.factors),
        description=model.description,
    )


def metrics_summary(model: ModelVersion) -> dict[str, Any] | None:
    """The few figures a list needs: test AUC, PR-AUC, recall at the threshold, and the gain over the heuristic."""
    if not model.metrics:
        return None
    at = model.metrics.get("atThreshold") or {}
    return {
        "rocAuc": model.metrics.get("rocAuc"),
        "prAuc": model.metrics.get("prAuc"),
        "recall": at.get("recall"),
        "precision": at.get("precision"),
        "liftOverHeuristic": ((model.card or {}).get("performance") or {}).get("liftOverHeuristic"),
    }


def model_dict(model: ModelVersion, names: dict[str, str] | None = None) -> dict[str, Any]:
    names = names or {}
    return {
        "id": model.id,
        "name": model.name,
        "version": model.version,
        "label": f"{model.name} v{model.version}",
        "outcome": model.outcome,
        "description": model.description,
        "datasetId": model.dataset_id,
        "kind": model.kind or "heuristic",
        "algorithm": model.algorithm,
        "factors": model.factors,
        "weightsSum": round(sum(float(f["weight"]) for f in model.factors), 6),
        "status": model.status,
        "target": model.target,
        "threshold": model.threshold,
        "metrics": metrics_summary(model),
        "fairnessFlagged": bool((model.fairness or {}).get("flagged")),
        "driftStatus": (model.drift or {}).get("status"),
        "createdBy": names.get(model.created_by or "", None),
        "trainedBy": names.get(model.trained_by or "", None),
        "approvedBy": names.get(model.approved_by or "", None),
        "approvedAt": model.approved_at.isoformat() if model.approved_at else None,
        "decisionNote": model.decision_note,
        "createdAt": model.created_at.isoformat(),
        "reviewedAt": model.reviewed_at.isoformat() if model.reviewed_at else None,
    }


async def list_models(session: AsyncSession, actor: Actor) -> list[dict[str, Any]]:
    actor.require(Permission.PREDICT)
    datasets = {d.id for d in await catalogue.analysis_catalogue(session, actor)}
    stmt = select(ModelVersion).order_by(ModelVersion.name, ModelVersion.created_at.desc())
    if not is_reviewer(actor):
        stmt = stmt.where(ModelVersion.status == "active")
    models = [m for m in (await session.execute(stmt)).scalars() if m.dataset_id in datasets]
    names = await _names(
        session, {u for m in models for u in (m.created_by, m.trained_by, m.approved_by) if u is not None}
    )
    return [model_dict(m, names) for m in models]


async def _reviewable(session: AsyncSession, actor: Actor, model_id: str) -> ModelVersion:
    if not is_reviewer(actor):
        raise Forbidden("Only a Data Steward, an Administrator or a Model Curator decides on model versions.")
    actor.require_verified()
    model = await session.get(ModelVersion, model_id)
    if model is None:
        raise NotFound("Model version not found.")
    return model


async def approve(session: AsyncSession, actor: Actor, model_id: str, note: str | None) -> dict[str, Any]:
    """Approve a draft for use. The approver is never the person who built or trained it."""
    model = await _reviewable(session, actor, model_id)
    if model.status != "draft":
        raise Conflict(f"This version is {model.status}; only a draft is approved.")
    if actor.id in {model.created_by, model.trained_by}:
        raise Forbidden("A model version is approved by someone other than the person who built it.")
    note = (note or "").strip()
    if (model.fairness or {}).get("flagged") and len(note) < _MIN_NOTE:
        raise ValidationFailed(
            "The fairness audit raised a flag. Approving needs a note of at least 20 characters saying why "
            "the model may still be used and how the flagged groups will be followed up."
        )
    for previous in (
        await session.execute(
            select(ModelVersion).where(
                ModelVersion.name == model.name, ModelVersion.status == "active", ModelVersion.id != model.id
            )
        )
    ).scalars():
        previous.status = "retired"
    model.status = "active"
    model.approved_by = actor.id
    model.approved_at = utcnow()
    model.reviewed_by = actor.id
    model.reviewed_at = model.approved_at
    model.decision_note = note or None
    label = f"{model.name} v{model.version}"
    await audit.record(session, actor, "Approved model version", "model", model.id, label, note)
    inbox.notify(
        session,
        [model.created_by, model.trained_by],
        "model",
        f"{label} was approved",
        note or f"{actor.name} approved it for use.",
        f"/models/{model.id}",
        skip=actor.id,
    )
    await session.commit()
    names = await _names(session, {u for u in (model.created_by, model.trained_by, model.approved_by) if u})
    return model_dict(model, names)


async def reject(session: AsyncSession, actor: Actor, model_id: str, reason: str | None) -> dict[str, Any]:
    model = await _reviewable(session, actor, model_id)
    if model.status != "draft":
        raise Conflict(f"This version is {model.status}; only a draft is rejected.")
    if actor.id in {model.created_by, model.trained_by}:
        raise Forbidden("A model version is decided on by someone other than the person who built it.")
    reason = (reason or "").strip()
    if not reason:
        raise ValidationFailed("Say why the version is rejected, so its author can fix it.")
    model.status = "rejected"
    model.reviewed_by = actor.id
    model.reviewed_at = utcnow()
    model.decision_note = reason
    label = f"{model.name} v{model.version}"
    await audit.record(session, actor, "Rejected model version", "model", model.id, label, reason)
    inbox.notify(
        session,
        [model.created_by, model.trained_by],
        "model",
        f"{label} was rejected",
        reason,
        f"/models/{model.id}",
        skip=actor.id,
    )
    await session.commit()
    return model_dict(model, await _names(session, {u for u in (model.created_by, model.trained_by) if u}))


async def retire(session: AsyncSession, actor: Actor, model_id: str, reason: str | None) -> dict[str, Any]:
    model = await _reviewable(session, actor, model_id)
    if model.status != "active":
        raise Conflict(f"This version is {model.status}; only an active version is retired.")
    model.status = "retired"
    model.decision_note = (reason or "").strip() or model.decision_note
    await audit.record(
        session,
        actor,
        "Retired model version",
        "model",
        model.id,
        f"{model.name} v{model.version}",
        reason or "",
    )
    await session.commit()
    return model_dict(model)


async def create_version(session: AsyncSession, actor: Actor, body: dict[str, Any]) -> dict[str, Any]:
    """A new model version. Weights must add up to one and every factor must read a measure of the dataset."""
    actor.require(Permission.CURATE_MODELS, "Only a Model Curator can publish model versions.")
    dataset = await analysis.dataset_for(session, actor, str(body.get("datasetId", "")))
    try:
        factors = tuple(factor_from_dict(f) for f in body.get("factors") or [])
        spec = validate_model(
            ModelSpec(
                str(body.get("name", "")).strip(),
                str(body.get("version", "")).strip(),
                str(body.get("outcome", "")).strip(),
                dataset.id,
                factors,
                str(body.get("description", "")),
            ),
            set(dataset.measures),
        )
    except ModelError as error:
        raise ValidationFailed(str(error)) from error
    if not spec.name or not spec.version or not spec.outcome:
        raise ValidationFailed("Give the model a name, a version and the outcome it scores.")
    if (
        await session.execute(
            select(ModelVersion).where(ModelVersion.name == spec.name, ModelVersion.version == spec.version)
        )
    ).scalar_one_or_none():
        raise Conflict(f"{spec.label} already exists. Versions are immutable; publish a new version number.")
    # A draft until someone else approves it; approving retires the version it replaces.
    model = ModelVersion(
        name=spec.name,
        version=spec.version,
        outcome=spec.outcome,
        description=spec.description,
        dataset_id=dataset.id,
        factors=[f.to_dict() for f in spec.factors],
        status="draft",
        kind="heuristic",
        created_by=actor.id,
    )
    session.add(model)
    await session.flush()
    await audit.record(
        session,
        actor,
        "Submitted model version for review",
        "model",
        model.id,
        spec.label,
        f"{len(factors)} factors, weights sum to 1.00",
    )
    inbox.notify(
        session,
        await inbox.stewards(session),
        "model",
        f"{spec.label} is waiting for review",
        f"{actor.name} submitted a documented heuristic for {dataset.name}.",
        f"/models/{model.id}",
        skip=actor.id,
    )
    await session.commit()
    return model_dict(model, await _names(session, {actor.id}))


def score_windows(months: list[str]) -> tuple[tuple[str, str], tuple[str, str]]:
    """(recent, before). recent is the last three months; before is the three before that."""
    recent = (months[-3], months[-1]) if len(months) >= 3 else (months[0], months[-1])
    before = (months[-6], months[-4]) if len(months) >= 6 else (months[0], months[max(0, len(months) - 4)])
    return recent, before


async def _factor_input(
    session: AsyncSession, actor: Actor, dataset: AnalysisDataset, factor: FactorSpec, filters: dict[str, str]
) -> FactorInput:
    months = list(dataset.months)
    measure = dataset.measures[factor.measure]
    dims = list(dataset.dimensions)
    recent, before = score_windows(months)

    async def population_values(window: tuple[str, str]) -> dict[tuple[str, ...], float]:
        rows = await analysis.query_groups(
            session,
            actor,
            dataset,
            {"v": measure.id},
            dims,
            {},
            date_from=window[0],
            date_to=window[1],
            kind="prediction",
        )
        return {
            tuple(str(r[f"d{i}"]) for i in range(len(dims))): float(r["v"])
            for r in rows
            if r["v"] is not None
        }

    async def cohort_value(window: tuple[str, str]) -> float:
        rows = await analysis.query_groups(
            session,
            actor,
            dataset,
            {"v": measure.id},
            [],
            filters,
            date_from=window[0],
            date_to=window[1],
            kind="prediction",
        )
        return float(rows[0]["v"]) if rows and rows[0]["v"] is not None else float("nan")

    if factor.window == "trend":
        now, then = await population_values(recent), await population_values(before)
        # Each group's change, matched by the group's own key.
        population = [now[key] - then[key] for key in now if key in then]
        cohort = await cohort_value(recent) - await cohort_value(before)
    else:
        window = recent if factor.window == "recent" else (months[0], months[-1])
        population = list((await population_values(window)).values())
        cohort = await cohort_value(window)
    centre = median(population) if population else float("nan")
    iqr = quantile(population, 0.75) - quantile(population, 0.25) if population else 0.0
    # The interquartile range scaled to a standard deviation is robust to extreme groups.
    spread = iqr / 1.349 if iqr > 0 else (standard_deviation(population) if len(population) > 1 else 0.0)
    return FactorInput(cohort=cohort, centre=centre, spread=spread, units=len(population), unit=measure.unit)


def prediction_dict(prediction: Prediction, model: ModelVersion) -> dict[str, Any]:
    return {
        "id": prediction.id,
        "subject": prediction.subject,
        "outcome": model.outcome,
        "score": prediction.score,
        "band": prediction.band,
        "modelVersion": f"{model.name} v{model.version}",
        "modelId": model.id,
        "cohort": prediction.cohort,
        "factors": prediction.factors,
        "createdAt": prediction.created_at.isoformat(),
        "when": humanize(prediction.created_at),
    }


async def score(
    session: AsyncSession, actor: Actor, model_id: str | None, cohort: dict[str, str]
) -> dict[str, Any]:
    actor.require(Permission.PREDICT)
    actor.require_verified()
    if model_id:
        model = await session.get(ModelVersion, model_id)
        # Drafts and retired versions are not for scoring, except by the people reviewing them.
        if model is not None and model.status != "active" and not is_reviewer(actor):
            model = None
    else:
        # Cohort scoring uses documented heuristics. Trained models score records (see outcome_models).
        model = (
            (
                await session.execute(
                    select(ModelVersion)
                    .where(ModelVersion.status == "active", ModelVersion.kind == "heuristic")
                    .order_by(ModelVersion.created_at.desc())
                )
            )
            .scalars()
            .first()
        )
    if model is None:
        raise NotFound(
            "No model is available. A Model Curator publishes one and a second person approves it."
        )
    if model.kind == "trained":
        raise ValidationFailed("A trained model scores records, not cohorts. Open it from Models to score.")
    dataset = await analysis.dataset_for(session, actor, model.dataset_id)
    spec = model_spec(model)
    for dim, value in cohort.items():
        if dim not in dataset.dimensions or value not in dataset.dimensions[dim].values:
            raise ValidationFailed(f"{dim} = {value} is not a cohort of {dataset.name}.")
    if not dataset.has_time:
        raise ValidationFailed("Scoring compares recent months, and this dataset has none.")
    inputs = {f.key: await _factor_input(session, actor, dataset, f, cohort) for f in spec.factors}
    subject = " · ".join(cohort.values()) if cohort else f"All of {dataset.name}"
    try:
        result = score_cohort(spec, inputs, f"Cohort · {subject}")
    except ModelError as error:
        raise ValidationFailed(str(error)) from error
    prediction = Prediction(
        model_version_id=model.id,
        requested_by=actor.member_id,
        subject=result["subject"],
        cohort=cohort,
        score=result["score"],
        band=result["band"],
        factors=result["factors"],
    )
    session.add(prediction)
    await audit.record(
        session,
        actor,
        "Scored prediction",
        "model",
        model.id,
        spec.label,
        f"{result['subject']}: {result['score']} ({result['band']})",
    )
    await session.commit()
    return prediction_dict(prediction, model)


async def history(session: AsyncSession, actor: Actor, limit: int = 20) -> list[dict[str, Any]]:
    rows = await session.execute(
        select(Prediction, ModelVersion)
        .join(ModelVersion, ModelVersion.id == Prediction.model_version_id)
        .where(Prediction.requested_by == actor.id)
        .order_by(Prediction.created_at.desc())
        .limit(limit)
    )
    return [prediction_dict(p, m) for p, m in rows]


async def _owned(session: AsyncSession, actor: Actor, prediction_id: str) -> tuple[Prediction, ModelVersion]:
    prediction = await session.get(Prediction, prediction_id)
    if prediction is None or (
        prediction.requested_by != actor.id and not actor.can(Permission.VIEW_ALL_AUDIT)
    ):
        raise NotFound("Prediction not found.")
    model = await session.get(ModelVersion, prediction.model_version_id)
    assert model is not None
    return prediction, model


def _pdf(data: dict[str, Any]) -> bytes:
    from fpdf import FPDF

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_fill_color(245, 245, 235)
    pdf.rect(0, 0, 210, 297, "F")
    pdf.set_text_color(55, 61, 64)
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, "DataForge prediction", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(82, 88, 91)
    pdf.multi_cell(
        0,
        6,
        f"{data['subject']}\n{data['outcome']}\nModel {data['modelVersion']} · {data['createdAt'][:16].replace('T', ' ')} UTC",
    )
    pdf.ln(4)
    pdf.set_text_color(233, 80, 54)
    pdf.set_font("Helvetica", "B", 40)
    pdf.cell(40, 18, str(data["score"]))
    pdf.set_text_color(55, 61, 64)
    pdf.set_font("Helvetica", "", 14)
    pdf.cell(0, 18, f"/ 100 · {data['band']}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Factors, largest influence first", new_x="LMARGIN", new_y="NEXT")
    for factor in data["factors"]:
        pdf.set_font("Helvetica", "B", 11)
        arrow = "raises" if factor["direction"] == "up" else "lowers"
        pdf.cell(
            0,
            7,
            f"{factor['name']}  (weight {factor['weight']:.2f}, {arrow} the score)",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5, factor["reading"])
        pdf.ln(1)
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(107, 113, 117)
    pdf.multi_cell(
        0,
        5,
        "Scores describe cohorts, not individual people. A score is a likelihood, not a certainty. "
        "The factor weights are documented with the model version and add up to 1.",
    )
    return bytes(pdf.output())


def _xlsx(data: dict[str, Any]) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill

    book = Workbook()
    sheet = book.active
    sheet.title = "Prediction"
    header = Font(bold=True, color="373D40")
    fill = PatternFill("solid", fgColor="EBEBD9")
    for row in [
        ("Subject", data["subject"]),
        ("Outcome", data["outcome"]),
        ("Score", data["score"]),
        ("Band", data["band"]),
        ("Model version", data["modelVersion"]),
        ("Scored at (UTC)", data["createdAt"]),
    ]:
        sheet.append(row)
        sheet.cell(sheet.max_row, 1).font = header
    sheet.append([])
    sheet.append(["Factor", "Weight", "Direction", "Contribution", "Reading"])
    for cell in sheet[sheet.max_row]:
        cell.font, cell.fill = header, fill
    for factor in data["factors"]:
        sheet.append(
            [
                factor["name"],
                factor["weight"],
                factor["direction"],
                factor.get("contribution"),
                factor["reading"],
            ]
        )
    sheet.column_dimensions["A"].width = 30
    sheet.column_dimensions["E"].width = 90
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


async def export(session: AsyncSession, actor: Actor, prediction_id: str, fmt: str) -> tuple[bytes, str, str]:
    prediction, model = await _owned(session, actor, prediction_id)
    data = prediction_dict(prediction, model)
    if fmt == "pdf":
        content, media, extension = _pdf(data), "application/pdf", "pdf"
    elif fmt == "xlsx":
        content, media, extension = (
            _xlsx(data),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "xlsx",
        )
    else:
        raise ValidationFailed("Export as pdf or xlsx.")
    await audit.record(
        session,
        actor,
        "Exported prediction",
        "prediction",
        prediction.id,
        data["subject"],
        f"As {extension.upper()}",
    )
    await session.commit()
    return content, media, f"dataforge-prediction-{prediction.id[:8]}.{extension}"
