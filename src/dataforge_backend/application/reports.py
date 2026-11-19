"""Governed reports: generate from a template, export with provenance, and re-check that every figure reproduces."""

from __future__ import annotations

import io
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.errors import NotFound, ValidationFailed
from ..domain import reporting
from ..domain.analysis import AnalysisDataset
from ..domain.analysis.format import format_value
from ..domain.roles import Permission
from ..infrastructure.db.models import MetricDefinition, ModelVersion, Report, ScoringRun, User
from . import analysis, audit, quotas
from .common import Actor, humanize
from .outcome_models import _latin
from .runtime import sandbox

DEFAULT_MONTHS = 12


def templates() -> list[dict[str, Any]]:
    auto = {
        "id": "auto",
        "title": "Governed summary of one dataset",
        "description": "Every shared metric on a dataset, in total and by its main groupings.",
        "roles": {"primary": "Any approved dataset with shared metrics"},
        "sections": [],
    }
    return [t.to_dict() for t in reporting.TEMPLATES.values()] + [auto]


def _period(dataset: AnalysisDataset, body: dict[str, Any]) -> tuple[str, str]:
    months = list(dataset.months)
    if not months:
        raise ValidationFailed(f"{dataset.name} has no months, so a reporting period cannot be chosen.")
    period = body.get("period") or {}
    start = str(period.get("from") or months[max(0, len(months) - DEFAULT_MONTHS)])
    end = str(period.get("to") or months[-1])
    if start > end:
        raise ValidationFailed("The period starts after it ends.")
    return start, end


def _clip(dataset: AnalysisDataset, start: str, end: str) -> tuple[str, str] | None:
    inside = [m for m in dataset.months if start <= m <= end]
    return (inside[0], inside[-1]) if inside else None


async def _metric(session: AsyncSession, metric_id: str | None) -> dict[str, Any] | None:
    if not metric_id:
        return None
    metric = await session.get(MetricDefinition, metric_id)
    return {"name": metric.name, "version": metric.version} if metric else None


async def _measure_figure(
    session: AsyncSession,
    actor: Actor,
    spec: reporting.FigureSpec,
    section: str,
    dataset: AnalysisDataset,
    period: tuple[str, str],
) -> dict[str, Any] | str:
    """A computed figure, or the reason it was skipped."""
    if spec.measure not in dataset.measures:
        return f"{spec.label}: {dataset.name} has no shared metric {spec.measure}."
    if spec.group_by and spec.group_by not in dataset.dimensions:
        return f"{spec.label}: {dataset.name} has no {spec.group_by.replace('_', ' ')}."
    clipped = _clip(dataset, *period)
    if clipped is None:
        return f"{spec.label}: {dataset.name} has no data in the period."
    measure = dataset.measures[spec.measure]
    trace: list[dict[str, Any]] = []
    rows = await analysis.query_groups(
        session,
        actor,
        dataset,
        {"v": measure.id},
        [spec.group_by] if spec.group_by else [],
        dict(spec.filters),
        date_from=clipped[0],
        date_to=clipped[1],
        parts=["v"] if measure.is_ratio else None,
        kind="report",
        trace=trace,
    )
    out_rows = []
    for row in rows:
        raw = row.get("v")
        value = float(raw) if raw is not None else None
        support = float(row["v_den"]) if row.get("v_den") is not None else None
        shown, suppressed = reporting.suppress(value, support, not measure.is_ratio)
        out_rows.append(
            {
                "group": str(row["d0"]) if spec.group_by else None,
                "value": shown,
                "display": f"<{reporting.SUPPRESS_BELOW}"
                if suppressed
                else ("â€”" if shown is None else format_value(shown, measure.unit)),
                "suppressed": suppressed,
            }
        )
    out_rows.sort(key=lambda r: str(r["group"] or ""))
    evidence = trace[0] if trace else {}
    return {
        "id": spec.id,
        "label": spec.label,
        "section": section,
        "kind": "metric",
        "dataset": dataset.id,
        "datasetName": dataset.name,
        "measure": measure.id,
        "measureLabel": measure.label,
        "unit": measure.unit,
        "metric": await _metric(session, measure.metric_id) if measure.governed else None,
        "groupBy": spec.group_by,
        "period": {"from": clipped[0], "to": clipped[1]},
        "rows": out_rows,
        "sql": evidence.get("sql", ""),
        "params": evidence.get("params", {}),
        "sessionId": evidence.get("sessionId"),
        "rowPolicy": evidence.get("rowPolicy", False),
    }


async def _model_figure(
    session: AsyncSession, spec: reporting.ModelFigureSpec, section: str
) -> dict[str, Any] | str:
    model = (
        (
            await session.execute(
                select(ModelVersion)
                .where(ModelVersion.name == spec.model, ModelVersion.status == "active")
                .order_by(ModelVersion.approved_at.desc())
            )
        )
        .scalars()
        .first()
    )
    if model is None:
        return f"{spec.label}: no approved {spec.model} model."
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
    if run is None:
        return f"{spec.label}: {model.name} v{model.version} has not scored any records yet."
    flagged, suppressed = reporting.suppress(float(run.flagged), None, True)
    return {
        "id": spec.id,
        "label": spec.label,
        "section": section,
        "kind": "model",
        "dataset": run.dataset_id,
        "model": {"id": model.id, "label": f"{model.name} v{model.version}"},
        "rows": [
            {
                "group": None,
                "value": flagged,
                "display": f"<{reporting.SUPPRESS_BELOW}"
                if suppressed
                else f"{run.flagged:,}".replace(",", " "),
                "suppressed": suppressed,
            }
        ],
        "scoringRun": {"id": run.id, "at": run.created_at.isoformat(), "rows": run.rows_scored},
        "sql": "",
        "params": {},
        "sessionId": None,
    }


async def generate(session: AsyncSession, actor: Actor, body: dict[str, Any]) -> dict[str, Any]:
    actor.require(Permission.ANALYSE, "Your role cannot produce reports.")
    actor.require_verified()
    await quotas.check_compute(session, actor)
    template_id = str(body.get("templateId") or "")
    roles = {str(k): str(v) for k, v in (body.get("datasets") or {}).items() if v}
    if "primary" not in roles:
        raise ValidationFailed("Choose the dataset the report is about.")
    datasets = {
        role: await analysis.dataset_for(session, actor, dataset_id) for role, dataset_id in roles.items()
    }
    primary = datasets["primary"]
    if template_id == "auto":
        governed = [(m.id, m.label) for m in primary.measures.values() if m.governed]
        if not governed:
            raise ValidationFailed(f"{primary.name} has no shared metrics to report on.")
        dims = [d for d, info in primary.dimensions.items() if 2 <= len(info.values) <= 12]
        template = reporting.auto_template(primary.name, governed, dims)
    elif template_id in reporting.TEMPLATES:
        template = reporting.TEMPLATES[template_id]
    else:
        raise NotFound("No such report template.")
    period = _period(primary, body)

    figures: list[dict[str, Any]] = []
    skipped: list[str] = []
    for section in template.sections:
        for spec in section.figures:
            if isinstance(spec, reporting.ModelFigureSpec):
                result = await _model_figure(session, spec, section.title)
            else:
                dataset = datasets.get(spec.role)
                result = (
                    await _measure_figure(session, actor, spec, section.title, dataset, period)
                    if dataset is not None
                    else f"{spec.label}: no dataset was chosen for {template.roles.get(spec.role, spec.role)}."
                )
            if isinstance(result, str):
                skipped.append(result)
            else:
                figures.append(result)

    totals = {
        f["id"]: f["rows"][0]["display"]
        for f in figures
        if len(f["rows"]) == 1 and f["rows"][0]["group"] is None
    }
    sentences = reporting.usable_narrative(template, set(totals))
    problems = reporting.validate_narrative(sentences, set(totals))
    if problems:
        raise ValidationFailed("The narrative failed its checks: " + "; ".join(problems))
    values = {
        **totals,
        "PERIOD_FROM": primary.month_label(period[0]),
        "PERIOD_TO": primary.month_label(period[1]),
    }
    narrative = [
        {"text": text, "template": sentence}
        for text, sentence in zip(reporting.fill(sentences, values), sentences, strict=True)
    ]
    report = Report(
        template_id=template.id,
        title=str(body.get("title") or template.title)[:300],
        period={"from": period[0], "to": period[1]},
        datasets=roles,
        figures=figures,
        narrative=narrative,
        narrative_source="template",
        created_by=actor.member_id,
    )
    session.add(report)
    await session.flush()
    await audit.record(
        session,
        actor,
        "Generated report",
        "report",
        report.id,
        report.title,
        f"{len(figures)} figures, {len(skipped)} skipped; {period[0]} to {period[1]}",
    )
    await session.commit()
    out = report_dict(report, {actor.id: actor.name})
    out["skipped"] = skipped
    out["disclaimer"] = template.disclaimer
    return out


def report_dict(report: Report, names: dict[str, str] | None = None) -> dict[str, Any]:
    template = reporting.TEMPLATES.get(report.template_id)
    return {
        "id": report.id,
        "templateId": report.template_id,
        "title": report.title,
        "period": report.period,
        "datasets": report.datasets,
        "figures": report.figures or [],
        "narrative": report.narrative or [],
        "narrativeSource": report.narrative_source,
        "disclaimer": template.disclaimer if template else "",
        "createdBy": (names or {}).get(report.created_by or "", None),
        "createdAt": report.created_at.isoformat(),
        "when": humanize(report.created_at),
    }


async def _readable(session: AsyncSession, actor: Actor, report_id: str) -> Report:
    report = await session.get(Report, report_id)
    if report is None or (report.created_by != actor.id and not actor.can(Permission.VIEW_ALL_AUDIT)):
        raise NotFound("Report not found.")
    return report


async def get(session: AsyncSession, actor: Actor, report_id: str) -> dict[str, Any]:
    report = await _readable(session, actor, report_id)
    names = {}
    if report.created_by:
        user = await session.get(User, report.created_by)
        names = {user.id: user.name} if user else {}
    return report_dict(report, names)


async def list_reports(session: AsyncSession, actor: Actor) -> list[dict[str, Any]]:
    stmt = select(Report).order_by(Report.created_at.desc()).limit(50)
    if not actor.can(Permission.VIEW_ALL_AUDIT):
        stmt = stmt.where(Report.created_by == actor.id)
    return [
        {k: v for k, v in report_dict(r).items() if k not in ("figures", "narrative")}
        for r in (await session.execute(stmt)).scalars()
    ]


async def audit_report(session: AsyncSession, actor: Actor, report_id: str) -> dict[str, Any]:
    """Re-run every figure's stored SQL and check each published value comes out the same."""
    report = await _readable(session, actor, report_id)
    checked, reproduced, mismatches = 0, 0, []
    for figure in report.figures or []:
        if not figure.get("sql"):
            continue
        checked += 1
        result = await sandbox.run_select(
            figure["sql"], user_id=actor.member_id, kind="report", params=figure.get("params") or {}
        )
        rows = [dict(zip(result.columns, row, strict=True)) for row in result.rows]
        fresh = {(str(r["d0"]) if figure.get("groupBy") else None): r.get("v") for r in rows}
        ok = True
        for row in figure["rows"]:
            if row["suppressed"] or row["value"] is None:
                continue
            value = fresh.get(row["group"])
            if value is None or abs(float(value) - float(row["value"])) > 1e-9 * max(
                1.0, abs(float(row["value"]))
            ):
                ok = False
                mismatches.append(
                    {"figure": figure["id"], "group": row["group"], "published": row["value"], "now": value}
                )
        reproduced += int(ok)
    narrative_problems = reporting.validate_narrative(
        [p.get("template", "") for p in report.narrative or []], {f["id"] for f in report.figures or []}
    )
    await audit.record(
        session,
        actor,
        "Audited report",
        "report",
        report.id,
        report.title,
        f"{reproduced} of {checked} reproduced",
    )
    await session.commit()
    return {
        "reportId": report.id,
        "figuresChecked": checked,
        "reproduced": reproduced,
        "mismatches": mismatches,
        "narrativeChecks": narrative_problems,
        "suppressedCells": sum(r["suppressed"] for f in report.figures or [] for r in f["rows"]),
    }


# ------------------------------------------------------------------ exports


def _safe(value: Any) -> Any:
    """Stop a spreadsheet reading a label from the data as a formula."""
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
        return f"'{value}"
    return value


def _line(pdf: Any, width: float, height: float, text: str) -> None:
    """One block of text, then back to the left margin for the next."""
    pdf.multi_cell(width, height, text, new_x="LMARGIN", new_y="NEXT")


def _pdf(data: dict[str, Any]) -> bytes:
    from fpdf import FPDF

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()
    pdf.set_text_color(55, 61, 64)
    pdf.set_font("Helvetica", "B", 18)
    _line(pdf, 0, 9, _latin(data["title"]))
    pdf.set_font("Helvetica", "", 10)
    period = data["period"] or {}
    _line(
        pdf,
        0,
        5,
        _latin(f"Period {period.get('from')} to {period.get('to')}. Generated {data['createdAt'][:10]}."),
    )
    if data.get("disclaimer"):
        pdf.set_font("Helvetica", "I", 9)
        _line(pdf, 0, 5, _latin(data["disclaimer"]))
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 11)
    for paragraph in data["narrative"]:
        _line(pdf, 0, 6, _latin(paragraph["text"]))
        pdf.ln(1)
    section = None
    for figure in data["figures"]:
        if figure["section"] != section:
            section = figure["section"]
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 13)
            _line(pdf, 0, 7, _latin(section))
        pdf.set_font("Helvetica", "B", 10)
        _line(pdf, 0, 6, _latin(f"{figure['id']}  {figure['label']}"))
        pdf.set_font("Helvetica", "", 10)
        for row in figure["rows"][:20]:
            label = row["group"] or "Total"
            _line(pdf, 0, 5, _latin(f"    {label}: {row['display']}"))
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Figure provenance", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    for figure in data["figures"]:
        metric = figure.get("metric") or {}
        source = (
            f"model {figure['model']['label']}, scoring run {figure['scoringRun']['id']}"
            if figure.get("kind") == "model"
            else f"{figure.get('datasetName')} Â· metric {metric.get('name', figure.get('measure'))} "
            f"v{metric.get('version', '-')} Â· sandbox session {figure.get('sessionId')}"
        )
        _line(pdf, 0, 4, _latin(f"{figure['id']} {figure['label']}: {source}"))
        if figure.get("sql"):
            _line(pdf, 0, 4, _latin(f"    {figure['sql'][:600]}"))
        pdf.ln(1)
    return bytes(pdf.output())


def _xlsx(data: dict[str, Any]) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Font

    book = Workbook()
    sheet = book.active
    sheet.title = "Report"
    bold = Font(bold=True)
    sheet.append([_safe(data["title"])])
    sheet.append([f"Period {data['period'].get('from')} to {data['period'].get('to')}"])
    if data.get("disclaimer"):
        sheet.append([data["disclaimer"]])
    sheet.append([])
    for paragraph in data["narrative"]:
        sheet.append([_safe(paragraph["text"])])
    sheet.append([])
    sheet.append(["Section", "Figure", "Label", "Group", "Value", "Shown as"])
    for cell in sheet[sheet.max_row]:
        cell.font = bold
    for figure in data["figures"]:
        for row in figure["rows"]:
            sheet.append(
                [
                    figure["section"],
                    figure["id"],
                    figure["label"],
                    _safe(row["group"] or "Total"),
                    row["value"],
                    row["display"],
                ]
            )
    provenance = book.create_sheet("Provenance")
    provenance.append(["Figure", "Dataset", "Metric", "Version", "Sandbox session", "SQL", "Parameters"])
    for cell in provenance[1]:
        cell.font = bold
    for figure in data["figures"]:
        metric = figure.get("metric") or {}
        provenance.append(
            [
                figure["id"],
                figure.get("datasetName") or figure.get("dataset"),
                metric.get("name") or figure.get("measure") or (figure.get("model") or {}).get("label"),
                metric.get("version"),
                figure.get("sessionId") or (figure.get("scoringRun") or {}).get("id"),
                figure.get("sql"),
                str(figure.get("params") or {}),
            ]
        )
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


async def export(session: AsyncSession, actor: Actor, report_id: str, fmt: str) -> tuple[bytes, str, str]:
    data = await get(session, actor, report_id)
    if fmt == "pdf":
        content, media = _pdf(data), "application/pdf"
    elif fmt == "xlsx":
        content, media = _xlsx(data), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        raise ValidationFailed("Export as pdf or xlsx.")
    await audit.record(
        session, actor, "Exported report", "report", report_id, data["title"], f"As {fmt.upper()}"
    )
    await session.commit()
    return content, media, f"dataforge-report-{report_id[:8]}.{fmt}"
