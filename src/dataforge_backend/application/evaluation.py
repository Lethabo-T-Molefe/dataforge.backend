"""Run the assistant evaluation: every question through the real assistant, as a dedicated evaluation member.

The member is an Everyday Analyst with no row policies, so the gold answers (hand-written SQL over the same
approved tables) and the assistant's answers see the same rows. Both run in the read-only sandbox.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.errors import AppError, Refused
from ..core.security import utcnow
from ..domain import evaluation
from ..infrastructure.db.models import Dataset, User
from . import assistant
from .common import Actor
from .runtime import sandbox

EVAL_EMAIL = "eval@dataforge.local"
DEFAULT_QUESTIONS = Path(__file__).resolve().parents[3] / "evals" / "assistant" / "questions_v1.jsonl"
# Large limits rather than none: quota overrides take numbers only.
_UNLIMITED = {"aiQueries": 1_000_000, "computeSeconds": 1_000_000_000}


async def eval_member(session: AsyncSession) -> Actor:
    user = (await session.execute(select(User).where(User.email == EVAL_EMAIL))).scalar_one_or_none()
    if user is None:
        user = User(
            id="eval-harness",
            email=EVAL_EMAIL,
            name="Evaluation harness",
            role="analyst",
            team=None,
            password_hash=None,
            email_verified_at=utcnow(),
        )
        session.add(user)
    user.quota_overrides = dict(_UNLIMITED)
    await session.commit()
    return Actor.from_user(user)


async def personal_columns(session: AsyncSession) -> set[str]:
    """Every column flagged as personal on any dataset, plus the ID-number column cleaning drops."""
    columns: set[str] = {"id_number"}
    for row in (await session.execute(select(Dataset.pii_columns))).scalars():
        columns |= set(row or {})
    return columns


async def _rows(sql: str) -> list[list[Any]]:
    result = await sandbox.run_select(sql, user_id=None, kind="eval", max_rows=20_000)
    return result.rows


async def run(
    session: AsyncSession,
    questions: list[evaluation.Question],
    planner: str | None = "rules",
    repeat: int = 1,
) -> tuple[dict[str, Any], list[evaluation.Result]]:
    actor = await eval_member(session)
    personal = await personal_columns(session)
    results: list[evaluation.Result] = []
    for _ in range(max(1, repeat)):
        answered: dict[str, str | None] = {}
        for question in questions:
            started = time.perf_counter()
            try:
                turn = await assistant.ask(
                    session,
                    actor,
                    question.question,
                    channel="eval",
                    context_id=answered.get(question.after) if question.after else None,
                    dataset_id=question.dataset,
                    planner=planner,
                )
            except AppError as error:
                await session.rollback()
                turn = {"refused": True, "answer": str(error), "trace": [], "refusalReason": "error"}
            latency = int((time.perf_counter() - started) * 1000)
            answered[question.id] = turn.get("id")
            gold = got = None
            gold_sql = question.expect.get("goldSql")
            if question.should_answer and gold_sql:
                gold = await _rows(gold_sql)
                if not turn.get("refused") and turn.get("sql"):
                    try:
                        got = await _rows(str(turn["sql"]))
                    except Refused:
                        got = []
            results.append(
                evaluation.score(
                    question, turn, latency_ms=latency, personal_columns=personal, gold=gold, got=got
                )
            )
    return evaluation.summarise(results), results
