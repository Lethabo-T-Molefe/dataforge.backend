"""Trained outcome models: options, training, review, scoring and the governed views of the scores.

Generic: any approved dataset with a column that records an outcome. Governance, in the order it applies:

* Only a Model Curator trains, and only on data they may analyse, under their own row policy.
* Personal information is never read for training or scoring; protected attributes are read only to audit
  fairness, and are never stored with the scores.
* Training runs in a worker process with a time limit; the fitted model is stored encrypted, with its hash.
* A trained version is a draft until someone else approves it (see predictions.approve).
* Scores are written to a silver table without personal or protected columns. Each viewer reads it through the
  sandbox under their own row policy, and sees aggregates with small groups suppressed.
"""

from __future__ import annotations

import io
from dataclasses import replace
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..core.crypto import decrypt_bytes, encrypt_bytes
from ..core.errors import Conflict, Forbidden, NotFound, Refused, ValidationFailed
from ..core.security import utcnow
from ..domain import outcome_model
from ..domain.analysis.sql import ident, qualified
from ..domain.prediction import ModelError
from ..domain.roles import Permission
from ..infrastructure import warehouse
from ..infrastructure.db.models import Dataset, DatasetVersion, ModelVersion, ScoringRun
from . import analysis, audit, catalogue, inbox, predictions, quotas
from .common import Actor
from .runtime import sandbox

SMALL_CELL = 5
TOP_RECORDS = 50
_READ_TIMEOUT_MS = 60_000
_SCORE_ROWS = 200_000


# ------------------------------------------------------------------ helpers


async def _dataset(session: AsyncSession, actor: Actor, dataset_id: str) -> tuple[Any, Dataset]:
    """The analysis view and the row of a dataset the member may analyse."""
    analysed = await analysis.dataset_for(session, actor, dataset_id)
    row = await session.get(Dataset, dataset_id)
    if row is None or not row.fields:
        raise NotFound("That dataset is not approved for analysis, or you cannot see it.")
    return analysed, row


def _spec(body: dict[str, Any], time_column: str | None) -> outcome_model.TrainingSpec:
    try:
        spec = outcome_model.spec_from_dict(
            {
                **body,
                "cvFolds": body.get("cvFolds") or settings.MODEL_CV_FOLDS,
                "maxRows": min(int(body.get("maxRows") or settings.MODEL_MAX_ROWS), settings.MODEL_MAX_ROWS),
            }
        )
    except ModelError as error:
        raise ValidationFailed(str(error)) from error
    return replace(spec, time_column=time_column)


def _plan(row: Dataset, spec: outcome_model.TrainingSpec) -> list[outcome_model.ColumnPlan]:
    fields = row.fields or {}
    try:
        return outcome_model.suggest_columns(
            list(fields.get("columns") or []), int(fields.get("rows") or 0), spec
        )
    except ModelError as error:
        raise ValidationFailed(str(error)) from error


def _versions(spec: outcome_model.TrainingSpec) -> dict[str, str]:
    """One version per algorithm: 1.0 alone, or 1.0-lr and 1.0-gb side by side."""
    if len(spec.algorithms) == 1:
        return {spec.algorithms[0]: spec.version}
    suffix = {"logistic": "lr", "hist_gb": "gb"}
    return {a: f"{spec.version}-{suffix[a]}" for a in spec.algorithms}


async def _read(
    session: AsyncSession,
    actor: Actor,
    table: str,
    columns: list[str],
    *,
    where: str | None = None,
    params: dict[str, Any] | None = None,
    policy: str | None = None,
    max_rows: int,
    total_rows: int,
) -> tuple[pd.DataFrame, bool]:
    """Read named columns through the read-only sandbox. Never `SELECT *`, so unplanned columns stay put."""
    conditions = [c for c in (where, f"({policy})" if policy else None) if c]
    sql = f"SELECT {', '.join(f't.{ident(c)}' for c in columns)} FROM {qualified(table)} t"
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    if total_rows > max_rows:
        sql += " ORDER BY random()"
    try:
        result = await sandbox.run_select(
            sql,
            user_id=actor.member_id,
            kind="prediction",
            params=params,
            max_rows=max_rows,
            timeout_ms=_READ_TIMEOUT_MS,
        )
    except Refused as error:
        raise ValidationFailed(f"The data could not be read for the model: {error}") from error
    await quotas.add_compute(session, actor.member_id, result.duration_ms)
    return pd.DataFrame(result.rows, columns=result.columns), result.truncated


async def _trained_model(session: AsyncSession, actor: Actor, model_id: str) -> ModelVersion:
    model = await session.get(ModelVersion, model_id)
    if model is None or model.kind != "trained":
        raise NotFound("Trained model not found.")
    if model.status != "active" and not predictions.is_reviewer(actor):
        raise NotFound("Trained model not found.")
    # The member must be able to analyse the dataset the model belongs to.
    await analysis.dataset_for(session, actor, model.dataset_id)
    return model


def _meta(model: ModelVersion) -> dict[str, Any]:
    target = model.target or {}
    return {
        "sha256": model.artifact_sha256,
        "runtime": model.runtime,
        "threshold": model.threshold,
        "outcome": {
            "column": target.get("column"),
            "positive": target.get("positive"),
            "unlabelled": tuple(target.get("unlabelled") or ()),
            "label": target.get("label") or "",
            "entity": target.get("entity") or "record",
        },
    }


async def _artifact(session: AsyncSession, model: ModelVersion) -> bytes:
    stored = (
        await session.execute(select(ModelVersion.artifact).where(ModelVersion.id == model.id))
    ).scalar_one_or_none()
    if not stored:
        raise Conflict("This model version has no stored model.")
    return decrypt_bytes(stored)


# ------------------------------------------------------------------ options and preview


async def options(session: AsyncSession, actor: Actor, dataset_id: str) -> dict[str, Any]:
    """What a curator can choose for a dataset: outcome columns, suggested protected attributes, every column."""
    actor.require(Permission.CURATE_MODELS, "Only a Model Curator trains models.")
    analysed, row = await _dataset(session, actor, dataset_id)
    columns = list((row.fields or {}).get("columns") or [])
    return {
        "dataset": {"id": row.id, "name": row.name, "rows": row.rows, "timeColumn": analysed.time_column},
        "months": list(analysed.months),
        "outcomes": outcome_model.outcome_candidates(columns),
        "protected": outcome_model.suggest_protected(columns),
        "columns": [
            {"name": c["name"], "type": c["type"], "distinct": c["distinct"], "personal": bool(c.get("pii"))}
            for c in columns
        ],
        "algorithms": [{"id": k, "label": v} for k, v in outcome_model.ALGORITHMS.items()],
        "thresholdPolicies": list(outcome_model.THRESHOLD_POLICIES),
    }


async def preview(session: AsyncSession, actor: Actor, body: dict[str, Any]) -> dict[str, Any]:
    """The column plan for a request, with each column's reason, before anything is trained."""
    actor.require(Permission.CURATE_MODELS, "Only a Model Curator trains models.")
    analysed, row = await _dataset(session, actor, str(body.get("datasetId") or ""))
    spec = _spec(body, analysed.time_column)
    plan = _plan(row, spec)
    return {
        "plan": [p.to_dict() for p in plan],
        "features": sum(1 for p in plan if p.role == "feature"),
        "notes": [
            "Leakage and missing values are checked on the data when the model trains; a column can still be "
            "left out then, with its reason."
        ],
    }


# ------------------------------------------------------------------ training


async def train(session: AsyncSession, actor: Actor, body: dict[str, Any]) -> list[dict[str, Any]]:
    """Train each requested algorithm on the dataset and save each as a draft for review."""
    actor.require(Permission.CURATE_MODELS, "Only a Model Curator trains models.")
    actor.require_verified()
    await quotas.check_compute(session, actor)
    analysed, row = await _dataset(session, actor, str(body.get("datasetId") or ""))
    spec = _spec(body, analysed.time_column)
    plan = _plan(row, spec)
    versions = _versions(spec)
    taken = (
        await session.execute(
            select(ModelVersion.version).where(
                ModelVersion.name == spec.name, ModelVersion.version.in_(list(versions.values()))
            )
        )
    ).scalars()
    clash = sorted(taken)
    if clash:
        raise Conflict(
            f"{spec.name} already has version {', '.join(clash)}. Versions are immutable; choose a new version."
        )

    # Read only what the plan uses. Personal and excluded columns never leave the warehouse.
    wanted = [p.name for p in plan if p.role in ("feature", "target", "protected", "key", "time")]
    table = row.qualified_table()
    if table is None:
        raise ValidationFailed("This dataset has no table to train on.")
    policy = await catalogue.policy_for(session, row.id, actor.role)
    frame, truncated = await _read(
        session,
        actor,
        table,
        wanted,
        policy=policy,
        max_rows=spec.max_rows * 2,
        total_rows=int((row.fields or {}).get("rows") or row.rows),
    )
    try:
        result, elapsed = await sandbox.run_python(
            outcome_model.train,
            frame,
            spec,
            plan,
            user_id=actor.member_id,
            kind="prediction",
            description=f"Train {spec.name} v{spec.version} on {row.name}",
            timeout_s=settings.MODEL_TRAIN_TIMEOUT_S,
            isolated=True,
        )
    except ModelError as error:
        raise ValidationFailed(str(error)) from error
    await quotas.add_compute(session, actor.member_id, elapsed)
    if truncated:
        result["data"]["notes"].append(
            "The dataset is larger than the training limit; a random sample was read."
        )
    if policy:
        result["data"]["notes"].append("Trained only on the rows the curator's role may see.")

    version_row = (
        await session.execute(
            select(DatasetVersion).where(
                DatasetVersion.dataset_id == row.id,
                DatasetVersion.version == row.current_version,
                DatasetVersion.layer == row.layer,
            )
        )
    ).scalar_one_or_none()
    outcome = spec.outcome
    target = {
        **spec.to_dict()["outcome"],
        "keyColumn": next((p["name"] for p in result["plan"] if p["role"] == "key"), None),
        "timeColumn": spec.time_column,
        "trainUntil": spec.train_until,
        "thresholdPolicy": spec.threshold_policy,
        "thresholdTarget": spec.threshold_target,
    }
    context = outcome_model.CardContext(row.name, row.current_version, actor.name, utcnow().isoformat())
    created: list[ModelVersion] = []
    for fitted in result["models"]:
        version = versions[fitted["algorithm"]]
        card = outcome_model.model_card(replace(spec, version=version), result, fitted, context)
        card["importance"] = fitted["importance"]
        model = ModelVersion(
            name=spec.name,
            version=version,
            outcome=f"Likelihood that the {outcome.entity} {outcome.sentence()}",
            description=str(body.get("description") or card["predicts"]),
            dataset_id=row.id,
            factors=[],
            status="draft",
            kind="trained",
            algorithm=fitted["algorithm"],
            target=target,
            features=result["plan"],
            protected=list(spec.protected),
            metrics=fitted["metrics"],
            fairness=fitted["fairness"],
            card=card,
            reference=result["reference"],
            runtime=result["runtime"],
            threshold=float(fitted["threshold"]),
            training_dataset_version_id=version_row.id if version_row else None,
            training_rows=int(result["data"]["labelled"]),
            created_by=actor.id,
            trained_by=actor.id,
            artifact=encrypt_bytes(fitted["artifact"]),
            artifact_sha256=fitted["artifactSha256"],
        )
        session.add(model)
        created.append(model)
    await session.flush()
    for model in created:
        await audit.record(
            session,
            actor,
            "Trained model version",
            "model",
            model.id,
            f"{model.name} v{model.version}",
            f"{outcome_model.ALGORITHMS[model.algorithm or 'logistic']} on {row.name}: "
            f"test ROC-AUC {(model.metrics or {}).get('rocAuc')}, {model.training_rows} labelled records",
        )
    labels = ", ".join(f"v{m.version}" for m in created)
    inbox.notify(
        session,
        await inbox.stewards(session),
        "model",
        f"{spec.name} {labels} is waiting for review",
        f"{actor.name} trained it on {row.name}. Check its performance and fairness before approving it.",
        f"/models/{created[0].id}",
        skip=actor.id,
    )
    await session.commit()
    names = {actor.id: actor.name}
    return [predictions.model_dict(m, names) for m in created]


# ------------------------------------------------------------------ one model in full


async def detail(session: AsyncSession, actor: Actor, model_id: str) -> dict[str, Any]:
    actor.require(Permission.PREDICT)
    model = await _trained_model(session, actor, model_id)
    reviewer = predictions.is_reviewer(actor)
    run = (
        (
            await session.execute(
                select(ScoringRun)
                .where(ScoringRun.model_version_id == model.id)
                .order_by(ScoringRun.created_at.desc())
            )
        )
        .scalars()
        .first()
    )
    names = await predictions._names(
        session, {u for u in (model.created_by, model.trained_by, model.approved_by) if u}
    )
    fairness = model.fairness or {}
    out = {
        **predictions.model_dict(model, names),
        "card": model.card,
        "fullMetrics": model.metrics,
        "features": model.features,
        # Group-level fairness figures are for the people deciding on the model.
        "fairness": fairness
        if reviewer
        else {"audited": fairness.get("audited"), "flags": fairness.get("flags")},
        "drift": model.drift,
        "lastScoring": _run_dict(run) if run else None,
        "canApprove": reviewer
        and model.status == "draft"
        and actor.id not in {model.created_by, model.trained_by},
        "canScore": reviewer and model.status == "active",
        "isAuthor": actor.id in {model.created_by, model.trained_by},
    }
    return out


def _run_dict(run: ScoringRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "rows": run.rows_scored,
        "flagged": run.flagged,
        "threshold": run.threshold,
        "datasetVersion": run.dataset_version,
        "drift": run.drift,
        "at": run.created_at.isoformat(),
    }


# ------------------------------------------------------------------ scoring


async def run_scoring(session: AsyncSession, actor: Actor, model_id: str) -> dict[str, Any]:
    """Score the records whose outcome is not known yet (or every record) and store the scores."""
    if not predictions.is_reviewer(actor):
        raise Forbidden(
            "Only a Model Curator, a Data Steward or an Administrator runs a model over a dataset."
        )
    actor.require_verified()
    model = await _trained_model(session, actor, model_id)
    if model.status != "active":
        raise Conflict("Only an approved model scores records. Ask a reviewer to approve this version.")
    analysed, row = await _dataset(session, actor, model.dataset_id)
    table = row.qualified_table()
    if table is None:
        raise ValidationFailed("This dataset has no table to score.")
    target = model.target or {}
    plan = model.features or []
    features = [p["name"] for p in plan if p["role"] == "feature"]
    key = target.get("keyColumn")
    protected = [c for c in (model.protected or []) if c]
    keep = [
        d.column
        for d in analysed.dimensions.values()
        if d.column not in protected and d.column != target.get("column")
    ]
    time_column = analysed.time_column
    columns = list(
        dict.fromkeys(
            [*([key] if key else []), *features, *keep, *protected, *([time_column] if time_column else [])]
        )
    )
    where, params = None, {}
    unlabelled = list(target.get("unlabelled") or [])
    if unlabelled:
        names = [f"u{i}" for i in range(len(unlabelled))]
        where = f"t.{ident(target['column'])}::text IN ({', '.join(':' + n for n in names)})"
        params = dict(zip(names, unlabelled, strict=True))
    frame, truncated = await _read(
        session,
        actor,
        table,
        columns,
        where=where,
        params=params,
        max_rows=_SCORE_ROWS,
        total_rows=0,
    )
    if frame.empty:
        raise ValidationFailed("There are no records to score: every record already has a known outcome.")
    raw = await _artifact(session, model)
    try:
        scored, elapsed = await sandbox.run_python(
            outcome_model.score,
            frame,
            raw,
            _meta(model),
            user_id=actor.member_id,
            kind="prediction",
            description=f"Score {row.name} with {model.name} v{model.version}",
            timeout_s=settings.MODEL_TRAIN_TIMEOUT_S,
            isolated=True,
        )
    except ModelError as error:
        raise ValidationFailed(str(error)) from error
    await quotas.add_compute(session, actor.member_id, elapsed)

    drift = outcome_model.psi(model.reference or {}, frame)
    flagged = pd.Series(scored["flagged"], index=frame.index)
    selection = {}
    for column in protected:
        if column in frame.columns:
            rates = flagged.groupby(frame[column].fillna("(blank)").astype(str)).agg(["mean", "count"])
            selection[column] = [
                {"group": str(g), "selectionRate": round(float(r["mean"]), 4), "n": int(r["count"])}
                for g, r in rates.iterrows()
                if int(r["count"]) >= SMALL_CELL
            ]

    # The stored scores: no personal or protected columns, so row policies and masking stay simple.
    out = pd.DataFrame(index=frame.index)
    types: dict[str, str] = {}
    if key:
        out["record"] = frame[key].astype(str)
        types["record"] = "text"
    for column in keep + ([time_column] if time_column else []):
        out[column] = frame[column].map(lambda v: None if v is None else str(v))
        types[column] = "text"
    out["score"] = scored["probability"]
    out["band"] = scored["band"]
    out["flagged"] = scored["flagged"]
    types.update(score="numeric", band="text", flagged="boolean")
    for i in range(3):
        out[f"driver_{i + 1}"] = [d[i]["text"] if len(d) > i else None for d in scored["drivers"]]
        types[f"driver_{i + 1}"] = "text"
    name = f"{row.table_name[:40]}__{model.id[:8]}_scores"
    conn = await session.connection()
    await warehouse.write_typed(conn, "silver", name, out.reset_index(drop=True), types)
    await warehouse.grant_sandbox(conn, f"silver.{name}")

    run = ScoringRun(
        model_version_id=model.id,
        dataset_id=row.id,
        dataset_version=row.current_version,
        requested_by=actor.member_id,
        scores_table=f"silver.{name}",
        population={"unlabelled": unlabelled, "truncated": truncated},
        rows_scored=len(out),
        flagged=int(flagged.sum()),
        threshold=model.threshold,
        summary={"bands": out["band"].value_counts().to_dict()},
        fairness={"selection": selection},
        drift=drift,
    )
    session.add(run)
    model.drift = drift
    model.drift_checked_at = utcnow()
    await audit.record(
        session,
        actor,
        "Scored records with model",
        "model",
        model.id,
        f"{model.name} v{model.version}",
        f"{len(out)} records from {row.name}, {int(flagged.sum())} flagged; drift {drift['status']}",
    )
    if drift["status"] == "drift":
        worst = drift["columns"][0]
        inbox.notify(
            session,
            [model.trained_by, row.owner_id],
            "model",
            f"{model.name} v{model.version}: today's records have drifted",
            f"{worst['label']} has moved since training (PSI {worst['psi']}). Review the model before relying on it.",
            f"/models/{model.id}",
        )
    await session.commit()
    return _run_dict(run)


# ------------------------------------------------------------------ the governed view of the scores


def _suppress(count: int) -> int | str:
    return f"<{SMALL_CELL}" if 0 < count < SMALL_CELL else count


async def scored_view(session: AsyncSession, actor: Actor, model_id: str, by: str | None) -> dict[str, Any]:
    """Flagged records by group, and the highest scores, under the viewer's own row policy."""
    actor.require(Permission.PREDICT)
    actor.require(Permission.ANALYSE)
    model = await _trained_model(session, actor, model_id)
    analysed, row = await _dataset(session, actor, model.dataset_id)
    run = (
        (
            await session.execute(
                select(ScoringRun)
                .where(ScoringRun.model_version_id == model.id)
                .order_by(ScoringRun.created_at.desc())
            )
        )
        .scalars()
        .first()
    )
    target = model.target or {}
    empty = {"model": predictions.model_dict(model), "scored": False, "groups": [], "records": []}
    if run is None:
        return {**empty, "message": "This model has not scored any records yet."}
    table = run.scores_table
    conn = await session.connection()
    if not await warehouse.table_exists(conn, table):
        return {**empty, "message": "The stored scores are gone; run the model again."}
    columns = set((await warehouse.column_types(conn, table)).keys())
    dims = [d.column for d in analysed.dimensions.values() if d.column in columns]
    group = by if by in dims else (dims[0] if dims else None)
    policy = await catalogue.policy_for(session, model.dataset_id, actor.role)
    where = f" WHERE ({policy})" if policy else ""
    flagged_where = f" WHERE t.flagged AND ({policy})" if policy else " WHERE t.flagged"
    try:
        totals = await sandbox.run_select(
            f"SELECT COUNT(*) AS n, SUM(CASE WHEN t.flagged THEN 1 ELSE 0 END) AS f FROM {qualified(table)} t{where}",
            user_id=actor.member_id,
            kind="prediction",
        )
        groups: list[dict[str, Any]] = []
        if group:
            result = await sandbox.run_select(
                f"SELECT t.{ident(group)} AS k, COUNT(*) AS n, SUM(CASE WHEN t.flagged THEN 1 ELSE 0 END) AS f, "
                f"AVG(t.score) AS s FROM {qualified(table)} t{where} GROUP BY 1 ORDER BY 3 DESC, 1",
                user_id=actor.member_id,
                kind="prediction",
            )
            for k, n, f, s in result.rows:
                if int(n or 0) < SMALL_CELL:
                    continue
                groups.append(
                    {
                        "group": k,
                        "records": int(n),
                        "flagged": _suppress(int(f or 0)),
                        "averageScore": round(float(s or 0), 4),
                    }
                )
        records: list[dict[str, Any]] = []
        if "record" in columns:
            shown = [c for c in dims if c != group][:3]
            select_list = ", ".join(
                ["t.record", *(f"t.{ident(c)}" for c in [group, *shown] if c), "t.score", "t.band"]
                + [f"t.driver_{i}" for i in (1, 2, 3)]
            )
            result = await sandbox.run_select(
                f"SELECT {select_list} FROM {qualified(table)} t{flagged_where} ORDER BY t.score DESC LIMIT {TOP_RECORDS}",
                user_id=actor.member_id,
                kind="prediction",
            )
            for values in result.rows:
                record = dict(zip(result.columns, values, strict=True))
                record["record"] = str(record["record"])[:8]
                record["drivers"] = [
                    record.pop(f"driver_{i}") for i in (1, 2, 3) if record.get(f"driver_{i}")
                ]
                records.append(record)
    except Refused:
        # A row policy that names a column the scores do not carry: show nothing rather than too much.
        return {**empty, "message": "Your role's row filter cannot be applied to the stored scores."}
    n, f = totals.rows[0] if totals.rows else (0, 0)
    await audit.record(
        session,
        actor,
        "Viewed scored records",
        "model",
        model.id,
        f"{model.name} v{model.version}",
        f"Grouped by {group}" if group else "",
    )
    await session.commit()
    entity = target.get("entity") or "record"
    return {
        "model": predictions.model_dict(model),
        "scored": True,
        "title": f"{entity.capitalize()}s likely to {_infinitive(target.get('label') or '')}".strip(),
        "entity": entity,
        "groupedBy": group,
        "dimensions": dims,
        "total": int(n or 0),
        "flagged": _suppress(int(f or 0)),
        "threshold": model.threshold,
        "groups": groups,
        "records": records,
        "recordsShown": "record" in columns,
        "rowPolicy": bool(policy),
        "scoredAt": run.created_at.isoformat(),
        "note": (
            f"Scores rank {entity}s for follow-up by a person. Groups under {SMALL_CELL} records are hidden, and no "
            "personal or protected details are stored with the scores."
        ),
    }


def _infinitive(label: str) -> str:
    """'drops out' → 'drop out', 'churns' → 'churn', for a heading."""
    first, _, rest = label.partition(" ")
    if first.endswith("ies"):
        first = first[:-3] + "y"
    elif first.endswith(("ches", "shes", "sses", "xes")):
        first = first[:-2]
    elif first.endswith("s") and not first.endswith("ss"):
        first = first[:-1]
    return f"{first} {rest}".strip()


# ------------------------------------------------------------------ the card as a PDF


async def card_pdf(session: AsyncSession, actor: Actor, model_id: str) -> tuple[bytes, str]:
    from fpdf import FPDF

    actor.require(Permission.PREDICT)
    model = await _trained_model(session, actor, model_id)
    card = model.card or {}
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_text_color(55, 61, 64)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, f"Model card: {card.get('title', model.name)}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)

    def line(label: str, text: str) -> None:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, label, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5, _latin(text))
        pdf.ln(1)

    performance = card.get("performance") or {}
    at = performance.get("atThreshold") or {}
    line("Status", f"{model.status} · {card.get('algorithm', '')}")
    line("Predicts", str(card.get("predicts", "")))
    line("Intended use", str(card.get("intendedUse", "")))
    line("Not for", "; ".join(card.get("notFor") or []))
    line(
        "Performance on held-out records",
        f"ROC-AUC {performance.get('rocAuc')} · PR-AUC {performance.get('prAuc')} · Brier {performance.get('brier')} · "
        f"recall {at.get('recall')} · precision {at.get('precision')} at threshold "
        f"{(card.get('threshold') or {}).get('value')}",
    )
    baselines = card.get("baselines") or {}
    line(
        "Baselines",
        "; ".join(f"{b.get('label')}: ROC-AUC {b.get('rocAuc')}" for b in baselines.values()),
    )
    line("Features", ", ".join(f.get("label", f.get("name", "")) for f in card.get("features") or []))
    line(
        "Left out",
        "; ".join(f"{f.get('label', f.get('name'))}: {f.get('reason')}" for f in card.get("excluded") or []),
    )
    fairness = card.get("fairness") or {}
    line("Fairness", "; ".join(fairness.get("flags") or []) or "No flags raised on the audited attributes.")
    line("Limitations", " ".join(card.get("limitations") or []))
    line("Human in the loop", str(card.get("humanInTheLoop", "")))
    approved = (
        f"Approved {model.approved_at:%Y-%m-%d}. {model.decision_note or ''}"
        if model.approved_at
        else "Not approved."
    )
    line("Decision", approved)
    buffer = io.BytesIO(bytes(pdf.output()))
    await audit.record(
        session, actor, "Exported model card", "model", model.id, f"{model.name} v{model.version}", "As PDF"
    )
    await session.commit()
    return buffer.getvalue(), f"dataforge-model-card-{model.name}-{model.version}.pdf"


def _latin(text: str) -> str:
    """fpdf's core fonts are Latin-1: keep the meaning of the few characters outside it."""
    return (
        text.replace("—", "-")
        .replace("–", "-")
        .replace("·", "-")
        .replace("’", "'")
        .replace("≥", ">=")
        .replace("≤", "<=")
        .encode("latin-1", "replace")
        .decode("latin-1")
    )
