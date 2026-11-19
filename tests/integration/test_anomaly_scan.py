"""The scheduled anomaly scan finds the planted unusual month, explains it, and tells the right people."""

from typing import Any

from httpx import AsyncClient

from .conftest import SignIn


async def _ok(response: Any, code: int = 200) -> Any:
    assert response.status_code == code, response.text
    return response.json()


async def test_the_scan_finds_explains_and_reports_the_planted_anomaly(
    client: AsyncClient, sign_in: SignIn
) -> None:
    analyst = await sign_in("analyst")
    steward = await sign_in("steward")
    assert (await client.post("/api/v1/insights/scan", headers=analyst)).status_code == 403

    first = await _ok(await client.post("/api/v1/insights/scan", headers=steward))
    assert first["scanned"] >= 1
    kzn = [a for a in first["found"] if "KwaZulu-Natal" in a["metric"] and a["month"] == "2026-08"]
    assert kzn, [(a["metric"], a["month"]) for a in first["found"]]
    alert = kzn[0]
    assert alert["source"] == "scan" and alert["datasetId"] == "learner-progress-monthly"
    assert alert["detector"]["method"] == "seasonal" and alert["detector"]["z"] > 3.5
    # Explained against the same month last year, and the SETA behind it named first.
    assert alert["previousMonth"] == "2025-08"
    assert "W&RSETA" in alert["rootCause"][0]["factor"], alert["rootCause"]
    assert len(alert["series"]) >= 24 and alert["series"][-1]["anomaly"] is True
    # Restricted-personal data is left to deliberate monitors.
    assert all(a["datasetId"] != "learner-progress" for a in first["found"])

    again = await _ok(await client.post("/api/v1/insights/scan", headers=steward))
    assert not [a for a in again["found"] if a["metric"] == alert["metric"] and a["month"] == "2026-08"]

    feed = await _ok(await client.get("/api/v1/insights/feed", headers=analyst))
    assert alert["id"] in {a["id"] for a in feed["alerts"]} and feed["lastScan"]
    detail = await _ok(await client.get(f"/api/v1/alerts/{alert['id']}", headers=analyst))
    assert detail["series"] and detail["rootCause"]

    dataset = await _ok(await client.get("/api/v1/datasets/learner-progress-monthly", headers=analyst))
    assert any(i["kind"] == "anomaly" for i in dataset["insights"])

    owner = await sign_in("practitioner")
    inbox = await _ok(await client.get("/api/v1/notifications", headers=owner))
    assert any(n["title"].startswith("Unusual month") for n in inbox["items"])


async def test_an_unusual_month_monitor_judges_the_latest_month(client: AsyncClient, sign_in: SignIn) -> None:
    analyst = await sign_in("analyst")
    created = await _ok(
        await client.post(
            "/api/v1/monitors",
            headers=analyst,
            json={
                "name": "KZN drop-outs",
                "datasetId": "learner-progress-monthly",
                "measure": "monthly_dropouts",
                "filters": {"province": "KwaZulu-Natal"},
                "rule": {"kind": "anomaly", "threshold": 3.5},
                "cadence": "daily",
            },
        ),
        201,
    )
    assert "unusually far from its expected level" in created["threshold"]
    checked = await _ok(await client.post(f"/api/v1/monitors/{created['id']}/check", headers=analyst))
    # October 2026 is an ordinary month; the planted anomaly was August.
    assert checked["triggered"] is False and "expected range" in checked["message"]
    too_sensitive = await client.post(
        "/api/v1/monitors",
        headers=analyst,
        json={
            "name": "x",
            "datasetId": "learner-progress-monthly",
            "measure": "monthly_dropouts",
            "rule": {"kind": "anomaly", "threshold": 1},
            "cadence": "daily",
        },
    )
    assert too_sensitive.status_code == 422
