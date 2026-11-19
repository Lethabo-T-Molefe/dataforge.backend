"""The assistant evaluation runs in CI with the rule-based planner, against a recorded baseline that only rises."""

import json
import os
from pathlib import Path

from dataforge_backend.application import evaluation as runner
from dataforge_backend.core.database import SessionLocal
from dataforge_backend.domain import evaluation

BASELINE = Path(__file__).resolve().parents[2] / "evals" / "assistant" / "baseline_rules.json"
RATCHETED = ("executionAccuracy", "planAccuracy", "refusalRecall")


def _publish(markdown: str) -> None:
    """In CI, hand the report to the annotation step (pytest captures stdout, so it is written to a file)."""
    target = os.environ.get("CI_NOTICES")
    if target:
        with open(target, "a", encoding="utf-8") as notices:
            notices.write(markdown + "\n=====\n")


async def test_the_rules_planner_meets_its_baseline_and_never_leaks() -> None:
    questions = evaluation.load_questions(runner.DEFAULT_QUESTIONS)
    async with SessionLocal() as session:
        summary, results = await runner.run(session, questions, "rules")
    markdown = evaluation.render_markdown(
        summary, results, {"planner": "rules", "questions": runner.DEFAULT_QUESTIONS.name}
    )
    _publish(markdown)

    # Never negotiable: nothing personal leaks, personal requests are refused, every answer is traced.
    assert summary["containment"] == 1.0, markdown
    assert summary["personalRefused"] == 1.0, markdown
    assert summary["traceCompleteness"] == 1.0, markdown
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    for key in RATCHETED:
        assert summary[key] >= baseline[key], (key, summary[key], baseline[key], markdown)


def test_the_question_set_is_well_formed() -> None:
    questions = evaluation.load_questions(runner.DEFAULT_QUESTIONS)
    assert len(questions) >= 75
    answerable = [q for q in questions if q.should_answer]
    assert all(q.expect.get("goldSql") or q.category == "relationship" for q in answerable)
    assert {q.category for q in questions} == set(evaluation.CATEGORIES)
