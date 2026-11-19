"""A governed report: every figure computed in the sandbox with its SQL kept, reproducible, and exportable."""

import re
from typing import Any

from httpx import AsyncClient
from sqlalchemy import text

from dataforge_backend.core.database import SessionLocal

from .conftest import SignIn


async def _ok(response: Any, code: int = 200) -> Any:
    assert response.status_code == code, response.text
    return response.json()


async def test_the_learner_progress_report_is_governed_and_reproducible(
    client: AsyncClient, sign_in: SignIn
) -> None:
    analyst = await sign_in("analyst")
    templates = await _ok(await client.get("/api/v1/reports/templates", headers=analyst))
    assert {"skills-progress", "auto"} <= {t["id"] for t in templates}

    report = await _ok(
        await client.post(
            "/api/v1/reports",
            headers=analyst,
            json={
                "templateId": "skills-progress",
                "datasets": {"primary": "learner-progress", "flows": "learner-progress-monthly"},
                "period": {"from": "2025-01", "to": "2025-06"},
            },
        ),
        201,
    )
    figures = {f["id"]: f for f in report["figures"]}
    assert {"F01", "F04", "F06", "F10", "F16", "F18"} <= set(figures)
    for figure in report["figures"]:
        if figure["kind"] == "metric":
            assert figure["sql"].upper().startswith("SELECT") and figure["sessionId"], figure["id"]
    assert figures["F04"]["metric"]["name"] == "Drop-out rate"
    assert "not an official SETA" in report["disclaimer"]

    # The headline rate is exactly what the data says for learners who started in the period.
    async with SessionLocal() as session:
        expected = (
            await session.execute(
                text(
                    "SELECT SUM(dropped)::float / NULLIF(SUM(starts), 0) FROM silver.learner_progress "
                    "WHERE start_month BETWEEN '2025-01' AND '2025-06'"
                )
            )
        ).scalar_one()
    assert abs(figures["F04"]["rows"][0]["value"] - expected) < 1e-9

    # The narrative's numbers all come from figures.
    prose = " ".join(p["text"] for p in report["narrative"])
    assert figures["F01"]["rows"][0]["display"] in prose
    for paragraph in report["narrative"]:
        assert not re.search(r"\d", re.sub(r"\{\{[A-Z0-9_]+\}\}", "", paragraph["template"]))

    checked = await _ok(await client.post(f"/api/v1/reports/{report['id']}/audit", headers=analyst))
    assert checked["figuresChecked"] > 10 and checked["reproduced"] == checked["figuresChecked"], checked
    assert checked["narrativeChecks"] == []

    pdf = await client.get(f"/api/v1/reports/{report['id']}/export?format=pdf", headers=analyst)
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    xlsx = await client.get(f"/api/v1/reports/{report['id']}/export?format=xlsx", headers=analyst)
    assert xlsx.status_code == 200 and xlsx.content[:2] == b"PK"

    # Only its author (and those who audit everything) can open it.
    other = await sign_in("practitioner")
    assert (await client.get(f"/api/v1/reports/{report['id']}", headers=other)).status_code == 404
    auditor = await sign_in("auditor")
    assert (await client.get(f"/api/v1/reports/{report['id']}", headers=auditor)).status_code == 200


async def test_any_dataset_gets_a_governed_summary(client: AsyncClient, sign_in: SignIn) -> None:
    analyst = await sign_in("analyst")
    report = await _ok(
        await client.post(
            "/api/v1/reports",
            headers=analyst,
            json={"templateId": "auto", "datasets": {"primary": "regional-revenue"}},
        ),
        201,
    )
    measures = {f["measure"] for f in report["figures"]}
    assert {"revenue", "avg_fee"} <= measures
    assert report["narrative"] and "{{" not in report["narrative"][0]["text"]
