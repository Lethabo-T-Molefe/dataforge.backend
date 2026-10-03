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
from ..core.errors import ValidationFailed
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
from ..domain.roles import Permission
from ..domain.sql_guard import QueryRefused
from ..infrastructure import llm
from ..infrastructure.db.models import AssistantQuery, MetricDefinition
from . import analysis, audit, catalogue, quotas
from .common import Actor, humanize

log = get_logger(__name__)


async def _catalogue(session: AsyncSession, actor: Actor) -> Catalogue:
    datasets = await catalogue.analysis_catalogue(session, actor)
    synonyms: dict[str, list[str]] = {}
    for metric in (
        await session.execute(select(MetricDefinition).where(MetricDefinition.status == "published"))
    ).scalars():
        synonyms[metric.key] = [metric.name, *(metric.synonyms or [])]
    return Catalogue(datasets, synonyms)


async def _previous(
    session: AsyncSession, actor: Actor, context_id: str | None
) -> tuple[AssistantPlan, str] | None:
    """The member's own earlier answer that a new question may follow up, with its question."""
    if not context_id:
        return None
    earlier = await session.get(AssistantQuery, context_id)
    if earlier is None or earlier.user_id != actor.id or earlier.refused or not earlier.plan:
        return None
    plan = AssistantPlan.from_dict(earlier.plan)
    return (plan, earlier.question) if plan else None


async def _plan(
    question: str, cat: Catalogue, previous: tuple[AssistantPlan, str] | None = None
) -> tuple[AssistantPlan | Refusal, str, str | None]:
    """(plan, planner, model). Safety checks run first and cannot be overridden by the model."""
    earlier = previous[0] if previous else None
    first = plan_question(question, cat, earlier)
    if isinstance(first, Refusal) and first.reason in ("injection", "personal"):
        return first, "rules", None
    if not settings.llm_enabled or not cat.datasets:
        return first, "rules", None
    try:
        proposal, model = await llm.propose_plan(
            question,
            catalogue_prompt(cat),
            plan_schema(cat),
            previous_prompt(previous[1], previous[0]) if previous else None,
        )
    except llm.PlanUnavailable as error:
        log.info("assistant_fallback_to_rules", reason=str(error))
        return first, "rules", None
    return plan_from_model(proposal, cat, earlier), "model", model


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
    return out


async def ask(
    session: AsyncSession,
    actor: Actor,
    question: str,
    channel: str = "web",
    context_id: str | None = None,
) -> dict[str, Any]:
    """Answer a question. `context_id` is the member's previous answer, which a follow-up builds on."""
    actor.require(Permission.ANALYSE)
    actor.require_verified()
    question = " ".join(question.split())
    if not question:
        raise ValidationFailed("Type a question.")
    question = question[:2000]
    await quotas.check_ai(session, actor)
    started = time.perf_counter()
    cat = await _catalogue(session, actor)
    if detect_injection(question):
        await audit.record(
            session, actor, "Suspected prompt injection", "assistant", None, "AI assistant", question[:500]
        )

    planned, planner, model = await _plan(question, cat, await _previous(session, actor, context_id))
    record = AssistantQuery(user_id=actor.member_id, question=question, planner=planner, channel=channel)

    if isinstance(planned, AssistantPlan):
        dataset = cat.datasets[planned.dataset_id]
        try:
            config = config_from_dict(dataset, _config_body(planned, dataset))
            result = await analysis.execute(session, actor, dataset, config, kind="assistant")
        except (AnalysisError, QueryRefused) as error:
            log.info("assistant_plan_not_runnable", error=str(error))
            planned = refusal("ungrounded", cat)
        else:
            if result.get("empty"):
                planned = Refusal(
                    "ungrounded",
                    result["reading"],
                    "No rows matched, so there is nothing to report. Try a wider period.",
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
                record.build = {
                    "method": planned.method,
                    "dataset": dataset.id,
                    "measure": planned.measure,
                    **({"dimension": planned.dimension} if planned.dimension else {}),
                    **({"measureY": config.measure_y} if planned.method == "relationship" else {}),
                }

    if isinstance(planned, Refusal):
        record.refused = True
        record.refusal_reason = planned.reason
        record.answer = planned.answer
        record.explanation = planned.explanation
        record.sql = ""
        record.grounded_in = []

    record.latency_ms = int((time.perf_counter() - started) * 1000)
    session.add(record)
    await quotas.add_ai_query(session, actor.id)
    action = "Refused AI query" if record.refused else "Ran AI query"
    detail = question if not record.refused else f"{question} ({record.refusal_reason})"
    await audit.record(
        session,
        actor,
        action,
        "assistant",
        None,
        record.grounded_in[1] if len(record.grounded_in) > 1 else "AI assistant",
        detail[:500],
        {"planner": planner, "model": model, "channel": channel},
    )
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


async def suggestions(session: AsyncSession, actor: Actor) -> list[str]:
    cat = await _catalogue(session, actor)
    return cat.example_questions()[:6]
