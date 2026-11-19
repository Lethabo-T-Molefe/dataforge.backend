"""The AI assistant: plan, ground, compose, validate, run in the sandbox, review, explain, and record.

The plan comes from Claude when a key is configured, otherwise from the rule-based planner; either way it
is validated against the member's own catalogue. The SQL is composed by the analysis engine, never by the
model, and runs under the member's role. A question that cannot be grounded is refused, not guessed.
"""

import time
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..core.errors import Conflict, ValidationFailed
from ..core.logging import get_logger
from ..domain.analysis import AnalysisDataset, AnalysisError, config_from_dict
from ..domain.analysis.format import format_delta, format_percent, format_value
from ..domain.assistant import (
    AssistantPlan,
    Catalogue,
    Refusal,
    catalogue_prompt,
    detect_injection,
    plan_from_model,
    plan_question,
    plan_schema,
    previous_prompt,
    refusal,
)
from ..domain.qualitative import Route, early_route, is_general
from ..domain.roles import Permission
from ..domain.sql_guard import QueryRefused
from ..domain.trace import Trace
from ..infrastructure import llm
from ..infrastructure.db.models import AssistantQuery, Dataset, MetricDefinition, User
from . import analysis, audit, catalogue, qualitative, quotas
from .common import Actor, humanize

log = get_logger(__name__)


async def _catalogue(session: AsyncSession, actor: Actor, dataset_id: str | None = None) -> Catalogue:
    """What the member may ask about. With `dataset_id`, only that dataset: the one they chose."""
    datasets = await catalogue.analysis_catalogue(session, actor)
    if dataset_id:
        datasets = [d for d in datasets if d.id == dataset_id]
        if not datasets:
            raise Conflict(
                "That dataset is not ready to analyse yet. Open it to see its next step, or choose another."
            )
    synonyms: dict[str, list[str]] = {}
    for metric in (
        await session.execute(select(MetricDefinition).where(MetricDefinition.status == "published"))
    ).scalars():
        synonyms[metric.key] = [metric.name, *(metric.synonyms or [])]
    return Catalogue(datasets, synonyms)


async def _previous(
    session: AsyncSession, actor: Actor, context_id: str | None, cat: Catalogue
) -> tuple[AssistantPlan, str] | None:
    """The member's own earlier answer that a new question may follow up, with its question.

    An earlier answer about a dataset outside the current catalogue (another dataset was chosen) is ignored,
    so a follow-up never drifts back to it.
    """
    if not context_id:
        return None
    earlier = await session.get(AssistantQuery, context_id)
    if earlier is None or earlier.user_id != actor.id or earlier.refused or not earlier.plan:
        return None
    plan = AssistantPlan.from_dict(earlier.plan)
    if plan is None or plan.dataset_id not in cat.datasets:
        return None
    return plan, earlier.question


PLANNERS = ("rules", "llm")


async def _plan(
    question: str,
    cat: Catalogue,
    previous: tuple[AssistantPlan, str] | None = None,
    planner: str | None = None,
) -> tuple[AssistantPlan | Refusal, str, str | None, str | None]:
    """(plan, planner, model, note). Safety checks run first and cannot be overridden by the model.

    `planner` forces one planner (the evaluation harness compares them): "rules" never calls the model, and
    "llm" requires one to be configured. The note says why the rules planned when the model was expected to.
    """
    earlier = previous[0] if previous else None
    first = plan_question(question, cat, earlier)
    if isinstance(first, Refusal) and first.reason in ("injection", "personal"):
        return first, "rules", None, "Safety checks run before any model and cannot be overridden."
    if planner == "rules":
        return first, "rules", None, "The rule-based planner was chosen for this question."
    if not settings.llm_enabled or not cat.datasets:
        if planner == "llm":
            raise ValidationFailed("No language model is configured, so the model planner cannot be used.")
        return first, "rules", None, "No language model is configured, so the rule-based planner planned."
    try:
        proposal, model = await llm.propose_plan(
            question,
            catalogue_prompt(cat),
            plan_schema(cat),
            previous_prompt(previous[1], previous[0]) if previous else None,
        )
    except llm.PlanUnavailable as error:
        log.info("assistant_fallback_to_rules", reason=str(error))
        return (
            first,
            "rules",
            None,
            f"The model was unavailable ({error}), so the rule-based planner planned.",
        )
    return plan_from_model(proposal, cat, earlier), "model", model, None


async def _metric_record(session: AsyncSession, metric_id: str | None) -> dict[str, Any] | None:
    """The governed metric behind a measure: its version, and who published it when."""
    if not metric_id:
        return None
    metric = await session.get(MetricDefinition, metric_id)
    if metric is None:
        return None
    publisher = await session.get(User, metric.published_by) if metric.published_by else None
    return {
        "id": metric.id,
        "name": metric.name,
        "version": metric.version,
        "publishedBy": publisher.name if publisher else metric.owner_label,
        "publishedAt": metric.published_at.date().isoformat() if metric.published_at else None,
    }


def _chart(result: dict[str, Any]) -> dict[str, Any] | None:
    """The answer's chart, kept small: at most 60 points."""
    visual = result.get("visual")
    if not isinstance(visual, dict):
        return None
    trimmed = dict(visual)
    for key in ("points", "bins"):
        if isinstance(trimmed.get(key), list):
            trimmed[key] = trimmed[key][:60]
    return {"visual": trimmed, "format": result.get("format"), "axisFormat": result.get("axisFormat")}


_REFUSAL_STEP = {
    "injection": ("screen", "Checked the question for attempts to change the assistant's instructions"),
    "personal": ("screen", "Checked the question for requests about named people"),
}


def _config_body(plan: AssistantPlan, dataset: AnalysisDataset) -> dict[str, Any]:
    body: dict[str, Any] = {
        "datasetId": dataset.id,
        "method": plan.method,
        "measure": plan.measure,
        "filters": plan.filters,
    }
    optional = {
        "dimension": plan.dimension,
        "split": plan.split,
        "measureY": plan.measure_y,
        "from": plan.date_from,
        "to": plan.date_to,
        "periodA": plan.period_a,
        "periodB": plan.period_b,
    }
    body.update({key: value for key, value in optional.items() if value})
    return body


def _sentences(text: str, count: int) -> str:
    return ". ".join(text.split(". ")[:count]).rstrip(".") + "."


def _answer(plan: AssistantPlan, dataset: AnalysisDataset, result: dict[str, Any]) -> str:
    measure = dataset.measures[plan.measure]
    figures = {f["label"]: f for f in result.get("figures", [])}
    if result.get("empty"):
        return str(result["reading"])
    if plan.method == "relationship":
        return _sentences(str(result["reading"]), 1)
    if plan.method == "explain":
        # The total change, then the group that drove most of it.
        return _sentences(str(result["reading"]), 2)
    if plan.method == "compare" and "Highest" in figures and "Lowest" in figures:
        top, bottom = figures["Highest"], figures["Lowest"]
        return f"{top['detail']} leads on {measure.label.lower()} at {top['value']}, with {bottom['detail']} lowest at {bottom['value']}."
    if plan.figure:
        rows = result.get("table", {}).get("rows") or []
        values = [r for r in rows if len(r) > 1 and r[1] not in ("—", None)]
        if len(values) >= 2:
            (first_month, first_value), (last_month, last_value) = values[-2][:2], values[-1][:2]
            raw = [p.get(measure.label) for p in result["visual"]["points"]]
            numbers = [v for v in raw if isinstance(v, int | float)]
            if len(numbers) >= 2:
                a, b = numbers[-2], numbers[-1]
                change = (
                    format_delta(b - a, measure.unit)
                    if measure.is_ratio
                    else format_percent((b - a) / abs(a) * 100 if a else float("nan"))
                )
                direction = "up" if b > a else "down" if b < a else "unchanged"
                return f"{measure.label} was {format_value(b, measure.unit)} in {last_month}, {direction} {change.lstrip('+−')} from {first_month}."
            return (
                f"{measure.label} was {last_value} in {last_month}, against {first_value} in {first_month}."
            )
        if values:
            return f"{measure.label} was {values[-1][1]} in {values[-1][0]}."
    return str(result["reading"]).split(". ")[0].rstrip(".") + "."


def _explanation(plan: AssistantPlan, dataset: AnalysisDataset, result: dict[str, Any]) -> str:
    measure = dataset.measures[plan.measure]
    source = (
        f"the shared metric {measure.label.lower()}{f' ({measure.description.rstrip(".")})' if measure.description else ''}"
        if measure.governed
        else f"the average of {measure.numerator}"
        if measure.is_average
        else f"the column total {measure.label.lower()}"
    )
    parts = [f"I used {source} from {dataset.name}.", result.get("method", "")]
    if plan.method == "relationship":
        parts.append("Two measures moving together is not proof that one causes the other.")
    if plan.follows_up:
        parts.insert(0, "This follows on from your previous question.")
    if result.get("rowPolicy"):
        parts.append("Rows you are not permitted to see were excluded before anything was computed.")
    parts.append("Open the SQL to see exactly what ran, or take it further in Analyse.")
    return " ".join(p for p in parts if p)


def _examples(cat: Catalogue) -> list[str]:
    return [q for q in cat.example_questions() if "rows are there" not in q]


def turn_dict(q: AssistantQuery) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": q.id,
        "question": q.question,
        "answer": q.answer,
        "explanation": q.explanation,
        "sql": q.sql,
        "refused": q.refused,
        "groundedIn": q.grounded_in or [],
        "planner": q.planner,
        "followsUp": bool((q.plan or {}).get("followsUp")),
        "when": humanize(q.created_at),
    }
    if q.result_table:
        out["table"] = q.result_table
    if q.build:
        out["build"] = q.build
    out["trace"] = q.trace or []
    if q.chart:
        out["chart"] = q.chart
    return out


async def ask(
    session: AsyncSession,
    actor: Actor,
    question: str,
    channel: str = "web",
    context_id: str | None = None,
    dataset_id: str | None = None,
    planner: str | None = None,
) -> dict[str, Any]:
    """Answer a question. `context_id` is the member's previous answer, which a follow-up builds on.

    `dataset_id` is the dataset the member chose: the rules and the model may only plan against it.
    `planner` forces "rules" or "llm" (used by the evaluation harness); normally the model plans when one is
    configured. The answer carries a trace of every governance step it went through.
    """
    actor.require(Permission.ANALYSE)
    actor.require_verified()
    question = " ".join(question.split())
    if not question:
        raise ValidationFailed("Type a question.")
    if planner is not None and planner not in PLANNERS:
        raise ValidationFailed(f"Planner is one of {', '.join(PLANNERS)}.")
    question = question[:2000]
    await quotas.check_ai(session, actor)
    started = time.perf_counter()
    trace = Trace()
    cat = await _catalogue(session, actor, dataset_id)
    suspected = detect_injection(question)
    if suspected:
        await audit.record(
            session, actor, "Suspected prompt injection", "assistant", None, "AI assistant", question[:500]
        )

    previous = await _previous(session, actor, context_id, cat)
    # Questions that are not about a number (how two categories relate, what people wrote, what the dataset
    # is) go to the qualitative answers. Safety refusals (personal details, injection) still come first.
    target = qualitative.in_scope(cat)
    first = plan_question(question, cat, previous[0] if previous else None)
    safe = not (isinstance(first, Refusal) and first.reason in ("injection", "personal"))
    if safe:
        trace.step(
            "screen",
            "Checked the question",
            "No attempt to change the assistant's instructions, and no request about a named person.",
        )
    else:
        assert isinstance(first, Refusal)
        key, label = _REFUSAL_STEP[first.reason]
        trace.step(key, label, first.explanation, status="blocked", reason=first.reason)
    names = ", ".join(d.name for d in cat.datasets.values())
    trace.step(
        "catalogue",
        "Limited to data you may use",
        (
            f"Only the dataset you chose: {names}."
            if dataset_id
            else f"{len(cat.datasets)} dataset(s) your role ({actor.role_label}) may analyse."
        ),
        datasets=[d.id for d in cat.datasets.values()],
        role=actor.role_label,
    )
    if target is not None and safe and planner is None:
        row = await session.get(Dataset, target.id)
        early = early_route(question, row.profile) if row is not None and row.profile else None
        if early is None and isinstance(first, Refusal) and is_general(question):
            early = Route("overview")
        if early is not None:
            record = AssistantQuery(user_id=actor.member_id, question=question, channel=channel)
            used = await qualitative.respond(
                session, actor, target, early, question, record, examples=_examples(cat)
            )
            if used is not None:
                record.planner = "model" if used else "rules"
                _qualitative_steps(trace, early, used)
                return await _finish(
                    session, actor, record, question, started, record.planner, used, channel, trace
                )

    planned, planner_used, model, note = await _plan(question, cat, previous, planner)
    record = AssistantQuery(user_id=actor.member_id, question=question, planner=planner_used, channel=channel)
    if safe:
        trace.step(
            "planner",
            "Planned the analysis",
            note
            or f"{model} proposed a structured plan, choosing only from your permitted datasets, measures and "
            "values. It wrote no SQL and saw no rows.",
            planner=planner_used,
            model=model,
        )

    if isinstance(planned, AssistantPlan):
        dataset = cat.datasets[planned.dataset_id]
        trace.step("plan", "The plan", "", plan=planned.to_dict())
        try:
            config = config_from_dict(dataset, _config_body(planned, dataset))
            result = await analysis.execute(session, actor, dataset, config, kind="assistant")
        except (AnalysisError, QueryRefused) as error:
            log.info("assistant_plan_not_runnable", error=str(error))
            planned = refusal("ungrounded", cat)
            trace.step(
                "validation",
                "Checked the plan against your data",
                f"The plan could not be run on permitted data: {error}",
                status="blocked",
            )
        else:
            if result.get("empty"):
                planned = Refusal(
                    "ungrounded",
                    result["reading"],
                    "No rows matched, so there is nothing to report. Try a wider period.",
                )
                trace.step(
                    "validation", "Checked the plan against your data", "No rows matched.", status="blocked"
                )
            else:
                measures = [dataset.measures[planned.measure]]
                if planned.method == "relationship":
                    measures.append(dataset.measures[config.measure_y])
                record.plan = planned.to_dict()
                record.answer = _answer(planned, dataset, result)
                record.explanation = _explanation(planned, dataset, result)
                record.sql = result["sql"]
                record.grounded_in = [
                    f"Metric · {measures[0].label}"
                    if measures[0].governed
                    else f"Column · {measures[0].label}",
                    dataset.table,
                    *[f"Metric · {m.label}" if m.governed else f"Column · {m.label}" for m in measures[1:]],
                    *(["Role row filters"] if result.get("rowPolicy") else []),
                ]
                record.result_table = result["table"]
                record.chart = _chart(result)
                record.build = {
                    "method": planned.method,
                    "dataset": dataset.id,
                    "measure": planned.measure,
                    **({"dimension": planned.dimension} if planned.dimension else {}),
                    **({"measureY": config.measure_y} if planned.method == "relationship" else {}),
                }
                await _answered_steps(session, trace, dataset, measures, planned, result)

    if (
        isinstance(planned, Refusal)
        and planned.reason == "ungrounded"
        and target is not None
        and planner is None
    ):
        # Not a calculation: let the model say what kind of question it is (a cross-tab, text themes, a
        # question about the dataset itself) and answer that, rather than refusing.
        row = await session.get(Dataset, target.id)
        if row is not None and row.profile:
            routed, route_model = await qualitative.route(question, row)
            used = await qualitative.respond(
                session,
                actor,
                target,
                routed,
                question,
                record,
                examples=_examples(cat),
                lead=(
                    "I couldn't turn that into a calculation on this dataset, so here is what it holds."
                    if routed.kind == "overview"
                    else None
                ),
            )
            if used is not None:
                record.planner = "model" if (used or route_model) else "rules"
                _qualitative_steps(trace, routed, used or route_model)
                return await _finish(
                    session,
                    actor,
                    record,
                    question,
                    started,
                    record.planner,
                    used or route_model,
                    channel,
                    trace,
                )

    if isinstance(planned, Refusal):
        record.refused = True
        record.refusal_reason = planned.reason
        record.answer = planned.answer
        record.explanation = planned.explanation
        record.sql = ""
        record.grounded_in = []
        if not trace.blocked:
            trace.step(
                "validation",
                "Checked the plan against your data",
                planned.explanation
                or "No permitted data answers this, so the assistant refused rather than guessed.",
                status="blocked",
                reason=planned.reason,
            )

    return await _finish(session, actor, record, question, started, planner_used, model, channel, trace)


def _qualitative_steps(trace: Trace, route: Route, model: str | None) -> None:
    detail = {
        "crosstab": "A cross-tabulation of two category columns, counted by DataForge with a chi-square test.",
        "themes": (
            "Themes in a written-text column: a sample of answers was labelled, and DataForge counted the "
            "labels. Personal details were removed before any text left the platform."
        ),
        "profile": "Answered from the dataset's quality profile.",
        "overview": "Answered from what the dataset holds: its columns, measures and quality.",
    }.get(route.kind, "Answered from the dataset's profile.")
    trace.step(
        "planner",
        "Routed to a descriptive answer",
        detail,
        route=route.kind,
        planner="model" if model else "rules",
        model=model,
    )


async def _answered_steps(
    session: AsyncSession,
    trace: Trace,
    dataset: AnalysisDataset,
    measures: list[Any],
    planned: AssistantPlan,
    result: dict[str, Any],
) -> None:
    checks = [f"{dataset.name} is approved and you may analyse it."]
    metrics = []
    for measure in measures:
        record = await _metric_record(session, measure.metric_id) if measure.governed else None
        if record:
            metrics.append(record)
            checks.append(
                f"{measure.label} is the shared metric '{record['name']}' (version {record['version']}, "
                f"published by {record['publishedBy']}"
                + (f" on {record['publishedAt']}" if record["publishedAt"] else "")
                + "), so it is computed the same way everywhere."
            )
        else:
            checks.append(f"{measure.label} is a column of the dataset, not a shared metric.")
    if planned.filters:
        checks.append(
            "Every filter value exists in the data: "
            + ", ".join(f"{k} = {v}" for k, v in planned.filters.items())
            + "."
        )
    trace.step("validation", "Checked the plan against your data", " ".join(checks), metrics=metrics or None)
    row = await session.get(Dataset, dataset.id)
    masked = sorted((row.pii_columns or {}) if row else {})
    privacy = (
        f"Personal columns stay masked and cannot be queried: {', '.join(masked)}."
        if masked
        else "The dataset has no personal columns."
    )
    if result.get("rowPolicy"):
        privacy += " Your role's row filter was applied before anything was computed."
    trace.step(
        "privacy", "Kept personal data out", privacy, masked=masked, rowPolicy=bool(result.get("rowPolicy"))
    )
    trace.step(
        "sql",
        "Wrote the SQL",
        "DataForge's analysis engine composed this SQL from the validated plan, and it passed the same checks as "
        "any member's query.",
        sql=result.get("sql"),
        runnable=result.get("sqlRunnable"),
    )
    sandbox = result.get("sandbox") or {}
    caps = sandbox.get("caps") or {}
    trace.step(
        "sandbox",
        "Ran it in the sandbox",
        f"Read-only, as the {caps.get('role', 'sandbox')} role, with a {int(caps.get('timeoutMs', 0)) // 1000}-second "
        f"limit; {sandbox.get('rows', 0)} row(s) in {sandbox.get('durationMs', 0)} ms.",
        sessionId=sandbox.get("sessionId"),
        caps=caps or None,
    )


async def _finish(
    session: AsyncSession,
    actor: Actor,
    record: AssistantQuery,
    question: str,
    started: float,
    planner: str,
    model: str | None,
    channel: str,
    trace: Trace | None = None,
) -> dict[str, Any]:
    """Record the answered (or refused) question, count it against the quota, audit it and return it."""
    record.latency_ms = int((time.perf_counter() - started) * 1000)
    session.add(record)
    await quotas.add_ai_query(session, actor.id)
    action = "Refused AI query" if record.refused else "Ran AI query"
    detail = question if not record.refused else f"{question} ({record.refusal_reason})"
    event = await audit.record(
        session,
        actor,
        action,
        "assistant",
        None,
        record.grounded_in[1] if len(record.grounded_in) > 1 else "AI assistant",
        detail[:500],
        {"planner": planner, "model": model, "channel": channel},
    )
    await session.flush()
    if trace is not None:
        trace.step(
            "audit",
            "Recorded it",
            f"Written to the append-only audit trail as event {event.id}, with the planner"
            + (f" ({model})" if model else "")
            + " and the channel.",
            eventId=str(event.id),
        )
        record.trace = trace.to_list()
    await session.commit()
    return turn_dict(record)


async def history(session: AsyncSession, actor: Actor, limit: int = 20) -> list[dict[str, Any]]:
    rows = await session.execute(
        select(AssistantQuery)
        .where(AssistantQuery.user_id == actor.id)
        .order_by(AssistantQuery.created_at.desc())
        .limit(limit)
    )
    return [turn_dict(q) for q in rows.scalars()]


async def suggestions(session: AsyncSession, actor: Actor, dataset_id: str | None = None) -> list[str]:
    try:
        cat = await _catalogue(session, actor, dataset_id)
    except Conflict:
        return []
    examples = cat.example_questions()
    if dataset_id:
        examples = ["What is this dataset about?", *examples]
    return examples[:6]
