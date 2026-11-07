"""Answer the assistant's qualitative questions: an overview, a profile question, a cross-tab or text themes.

Queries run in the sandbox under the member's read-only role and their row policy, exactly like an analysis.
The model reads column summaries, and for themes a random sample of one free-text column with contact
details removed; it never sees a personal column's values, and it never writes SQL.
"""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from ..core.logging import get_logger
from ..domain.analysis import AnalysisDataset
from ..domain.analysis.sql import ident, qualified
from ..domain.assistant import Catalogue
from ..domain.overview import fallback_narrative, overview_facts, overview_table, schema_prompt
from ..domain.qualitative import (
    THEME_SAMPLE,
    Route,
    crosstab,
    is_profile_question,
    keyword_themes,
    profile_answer,
    profile_prompt,
    routing_prompt,
    scrub,
    themes_result,
    validate_route,
)
from ..infrastructure import llm
from ..infrastructure.db.models import AssistantQuery, Dataset
from ..infrastructure.warehouse import jsonable
from . import catalogue, quotas
from .common import Actor
from .runtime import sandbox

log = get_logger(__name__)


def in_scope(cat: Catalogue) -> AnalysisDataset | None:
    """The one dataset a qualitative answer is about: the member's choice, or their only dataset."""
    return next(iter(cat.datasets.values())) if len(cat.datasets) == 1 else None


async def route(question: str, dataset: Dataset) -> tuple[Route, str | None]:
    """Ask the model what kind of question this is. Without a model, the overview answers."""
    profile = dataset.profile or {}
    try:
        proposal, model = await llm.route_question(
            question,
            routing_prompt(
                dataset.name, profile, schema_prompt(dataset.name, dataset.description or "", profile)
            ),
        )
    except llm.PlanUnavailable as error:
        log.info("assistant_route_without_model", reason=str(error))
        return Route("profile" if is_profile_question(question) else "overview"), None
    return validate_route(proposal, profile), model


def _fill(
    record: AssistantQuery, *, answer: str, explanation: str, table: Any, sql: str, grounded: list[str]
) -> None:
    record.answer = answer
    record.explanation = explanation
    record.result_table = table
    record.sql = sql
    record.grounded_in = grounded
    record.plan = None
    record.refused = False
    record.refusal_reason = None


async def respond(
    session: AsyncSession,
    actor: Actor,
    target: AnalysisDataset,
    route_: Route,
    question: str,
    record: AssistantQuery,
    *,
    examples: list[str],
    lead: str | None = None,
) -> str | None:
    """Fill `record` with the answer for `route_`. Returns the model used ("" for none), or None if it could not."""
    dataset = await session.get(Dataset, target.id)
    if dataset is None or not dataset.profile:
        return None
    if route_.kind == "crosstab" and route_.column_a and route_.column_b:
        return await _crosstab(session, actor, target, dataset, route_.column_a, route_.column_b, record)
    if route_.kind == "themes" and route_.text_column:
        return await _themes(session, actor, target, dataset, route_.text_column, question, record)
    if route_.kind == "profile":
        try:
            answer, model = await llm.answer_from_profile(
                question, profile_prompt(dataset.name, dataset.description or "", dataset.profile)
            )
        except llm.PlanUnavailable as error:
            log.info("assistant_profile_without_model", reason=str(error))
            ruled = profile_answer(question, dataset.profile)
            if ruled is None:
                return await _overview(dataset, target, record, examples, lead)
            answer, model = ruled, ""
        _fill(
            record,
            answer=answer,
            explanation=(
                f"Answered from the profile taken when {dataset.name} was brought in (column types, blanks, "
                f"ranges and most common values); no query ran. {overview_facts(dataset.profile, dataset.pii_columns or {})}"
            ),
            table=overview_table(dataset.profile),
            sql="",
            grounded=["Dataset profile", target.table],
        )
        return model
    return await _overview(dataset, target, record, examples, lead)


async def _overview(
    dataset: Dataset, target: AnalysisDataset, record: AssistantQuery, examples: list[str], lead: str | None
) -> str:
    profile = dataset.profile or {}
    model = ""
    try:
        narrative, model = await llm.describe_dataset(
            schema_prompt(dataset.name, dataset.description or "", profile)
        )
    except llm.PlanUnavailable as error:
        log.info("assistant_overview_without_model", reason=str(error))
        narrative = fallback_narrative(dataset.name, profile)
    ask_next = f" Try asking: {' · '.join(examples[:3])}" if examples else ""
    _fill(
        record,
        answer=f"{lead} {narrative}" if lead else narrative,
        explanation=(
            f"{overview_facts(profile, dataset.pii_columns or {})} This overview comes from the profile taken when "
            f"{dataset.name} was brought in; no query ran.{ask_next}"
        ),
        table=overview_table(profile),
        sql="",
        grounded=["Dataset profile", target.table],
    )
    return model


def _where(policy: str | None, *conditions: str) -> str:
    parts = [*conditions, *([f"({policy})"] if policy else [])]
    return f"WHERE  {' AND '.join(parts)}\n" if parts else ""


async def _crosstab(
    session: AsyncSession,
    actor: Actor,
    target: AnalysisDataset,
    dataset: Dataset,
    a: str,
    b: str,
    record: AssistantQuery,
) -> str:
    policy = await catalogue.policy_for(session, target.id, actor.role)
    sql = (
        f"-- Cross-tab · {dataset.name}\n"
        f"SELECT {ident(a)}::text AS {ident(a)},\n       {ident(b)}::text AS {ident(b)},\n       COUNT(*) AS n\n"
        f"FROM   {qualified(target.table)}\n{_where(policy)}GROUP  BY 1, 2\nORDER  BY 3 DESC;"
    )
    result = await sandbox.run_select(sql, user_id=actor.member_id, kind="assistant", max_rows=5_000)
    await quotas.add_compute(session, actor.member_id, result.duration_ms)
    read = crosstab([(row[0], row[1], row[2]) for row in result.rows], a, b)
    notes = " Rows your role is not permitted to see were left out first." if policy else ""
    _fill(
        record,
        answer=read["answer"],
        explanation=read["explanation"] + notes,
        table=read["table"],
        sql=sql,
        grounded=[f"Column · {a}", target.table, f"Column · {b}", *(["Role row filters"] if policy else [])],
    )
    return ""


async def _themes(
    session: AsyncSession,
    actor: Actor,
    target: AnalysisDataset,
    dataset: Dataset,
    column: str,
    question: str,
    record: AssistantQuery,
) -> str:
    policy = await catalogue.policy_for(session, target.id, actor.role)
    col = ident(column)
    sql = (
        f"-- Sample of written answers · {dataset.name}\n"
        f"SELECT {col}::text AS {col}\nFROM   {qualified(target.table)}\n"
        f"{_where(policy, f'{col} IS NOT NULL', f'length(trim({col}::text)) > 0')}"
        f"ORDER  BY random()\nLIMIT  {THEME_SAMPLE};"
    )
    result = await sandbox.run_select(sql, user_id=actor.member_id, kind="assistant", max_rows=THEME_SAMPLE)
    await quotas.add_compute(session, actor.member_id, result.duration_ms)
    entries = [scrub(str(jsonable(row[0]))) for row in result.rows if row and row[0]]
    label = column.replace("_", " ")
    if not entries:
        _fill(
            record,
            answer=f"{label.capitalize()} has no written entries you are permitted to read.",
            explanation="",
            table=None,
            sql=sql,
            grounded=[f"Column · {column}", target.table],
        )
        return ""
    model = ""
    try:
        coded, model = await llm.code_themes(column, entries, question)
        read = themes_result(coded, entries)
        if not read["table"]["rows"] or read["labelled"] < len(entries) // 2:
            raise llm.PlanUnavailable("the model labelled too few entries")
        method = (
            f"I read a random sample of {len(entries)} entries from {label}. The AI model gave each one a theme "
            f"and a sentiment ({read['labelled']} labelled), and DataForge counted them, so the shares are exact "
            "for the sample and estimates for the whole column. The examples are real entries. Email addresses, "
            "links and long numbers were removed before the model read anything."
        )
    except llm.PlanUnavailable as error:
        log.info("assistant_themes_without_model", reason=str(error))
        read = keyword_themes(entries, column)
        model = ""
        method = (
            f"I counted the words and phrases that recur across a random sample of {len(entries)} entries from "
            f"{label}. Shares are of the sample, so treat them as estimates."
        )
    _fill(
        record,
        answer=read["answer"],
        explanation=method,
        table=read["table"],
        sql=sql,
        grounded=[f"Column · {column}", target.table, *(["Role row filters"] if policy else [])],
    )
    return model
