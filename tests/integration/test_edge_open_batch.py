"""Open catalogue cases: connectors, monitors, predictions, governance, recipes, stories."""

import io

from httpx import AsyncClient

from catalogue_progress import marks

from .conftest import SignIn

CSV = "a,b\n1,2\n3,4\n"
PII_CSV = "notes\na@x.org\nb@y.org\nc@z.org\n"


@marks("K-07", "K-10", "K-11", "K-13", "K-14")
async def test_connector_access_and_schedule(client: AsyncClient, sign_in: SignIn) -> None:
    analyst = await sign_in("analyst")
    other = await sign_in("practitioner")
    executive = await sign_in("executive")
    steward = await sign_in("steward")

    pending = await client.post(
        "/api/v1/connectors",
        headers=analyst,
        json={
            "name": "Locked PG",
            "kind": "postgres",
            "host": "db.example",
            "database": "demo",
            "user": "reader",
            "password": "secret",
        },
    )
    assert pending.status_code == 201, pending.text
    cid = pending.json()["id"]
    assert (await client.post(f"/api/v1/connectors/{cid}/test", headers=other)).status_code == 403
    assert (
        await client.post(
            f"/api/v1/connectors/{cid}/import",
            headers=analyst,
            json={"table": "fees", "name": "Too soon"},
        )
    ).status_code == 422
    assert (
        await client.post(
            f"/api/v1/connectors/{cid}/import",
            headers=executive,
            json={"table": "fees", "name": "Exec"},
        )
    ).status_code == 403
    assert (
        await client.post(f"/api/v1/connectors/{cid}/sync", headers=other)
    ).status_code == 403
    assert (
        await client.post(f"/api/v1/connectors/{cid}/sync", headers=analyst)
    ).status_code in (403, 422)

    sheet = await client.post(
        "/api/v1/connectors", headers=analyst, json={"name": "Sheet sched", "kind": "spreadsheet"}
    )
    assert sheet.status_code == 201
    sid = sheet.json()["id"]
    assert (
        await client.post(
            f"/api/v1/connectors/{cid}/workbook",
            headers=analyst,
            files={"file": ("a.xlsx", io.BytesIO(b"PK"), "application/octet-stream")},
        )
    ).status_code == 422
    bad_cadence = await client.patch(
        f"/api/v1/connectors/{sid}", headers=steward, json={"refreshCadence": "monthly"}
    )
    assert bad_cadence.status_code == 422
    sheet_sched = await client.patch(
        f"/api/v1/connectors/{sid}", headers=steward, json={"refreshCadence": "daily"}
    )
    assert sheet_sched.status_code == 422


@marks("MO-02", "MO-05", "MO-13", "MO-15")
async def test_monitor_and_alert_edges(client: AsyncClient, sign_in: SignIn) -> None:
    analyst = await sign_in("analyst")
    missing = await client.post(
        "/api/v1/monitors",
        headers=analyst,
        json={
            "datasetId": "no-such-dataset",
            "measure": "revenue",
            "rule": {"kind": "above", "threshold": 1},
            "cadence": "daily",
        },
    )
    assert missing.status_code == 404
    bad_measure = await client.post(
        "/api/v1/monitors",
        headers=analyst,
        json={
            "datasetId": "regional-revenue",
            "measure": "not_a_measure",
            "rule": {"kind": "above", "threshold": 1},
            "cadence": "daily",
        },
    )
    assert bad_measure.status_code == 422
    bad_cadence = await client.post(
        "/api/v1/monitors",
        headers=analyst,
        json={
            "datasetId": "regional-revenue",
            "measure": "revenue",
            "rule": {"kind": "above", "threshold": 1},
            "cadence": "yearly",
        },
    )
    assert bad_cadence.status_code == 422
    created = await client.post(
        "/api/v1/monitors",
        headers=analyst,
        json={
            "datasetId": "regional-revenue",
            "measure": "revenue",
            "rule": {"kind": "above", "threshold": 1},
            "cadence": "daily",
            "name": "Edge watch",
        },
    )
    assert created.status_code == 201, created.text
    mid = created.json()["id"]
    assert (await client.post(f"/api/v1/monitors/missing-monitor/check", headers=analyst)).status_code == 404
    bad_status = await client.patch(
        f"/api/v1/monitors/{mid}", headers=analyst, json={"status": "archived"}
    )
    assert bad_status.status_code == 422
    paused = await client.patch(
        f"/api/v1/monitors/{mid}", headers=analyst, json={"status": "paused"}
    )
    assert paused.status_code == 200
    assert (await client.post("/api/v1/alerts/missing-alert/acknowledge", headers=analyst)).status_code == 404
    alerts = (await client.get("/api/v1/alerts", headers=analyst)).json()
    if alerts:
        first = await client.post(
            f"/api/v1/alerts/{alerts[0]['id']}/acknowledge", headers=analyst
        )
        assert first.status_code == 200
        again = await client.post(
            f"/api/v1/alerts/{alerts[0]['id']}/acknowledge", headers=analyst
        )
        assert again.status_code == 200


@marks("PR-01", "PR-03", "PR-04", "PR-06", "PR-10")
async def test_prediction_publish_score_export(client: AsyncClient, sign_in: SignIn) -> None:
    analyst = await sign_in("analyst")
    steward = await sign_in("steward")
    curator = await sign_in("curator")
    body = {
        "name": "edge-heuristic",
        "version": "1.0",
        "outcome": "complete",
        "datasetId": "course-completions",
        "factors": [
            {"key": "a", "name": "A", "weight": 0.5, "direction": "up", "measure": "enrolments"},
            {"key": "b", "name": "B", "weight": 0.5, "direction": "down", "measure": "enrolments"},
        ],
    }
    assert (await client.post("/api/v1/predictions/models", headers=analyst, json=body)).status_code == 403
    assert (await client.post("/api/v1/predictions/models", headers=steward, json=body)).status_code == 403
    ok = await client.post("/api/v1/predictions/models", headers=curator, json=body)
    assert ok.status_code == 201, ok.text
    dup = await client.post("/api/v1/predictions/models", headers=curator, json=body)
    assert dup.status_code == 409
    v2 = {**body, "version": "1.1"}
    assert (await client.post("/api/v1/predictions/models", headers=curator, json=v2)).status_code == 201
    assert (
        await client.post(
            "/api/v1/predictions/score",
            headers=analyst,
            json={"modelId": "no-such-model", "cohort": {}},
        )
    ).status_code == 404
    bad_cohort = await client.post(
        "/api/v1/predictions/score",
        headers=analyst,
        json={"cohort": {"course": "Not A Real Course"}},
    )
    assert bad_cohort.status_code == 422
    scored = await client.post(
        "/api/v1/predictions/score",
        headers=analyst,
        json={"cohort": {"course": "Stewardship 101"}},
    )
    assert scored.status_code == 200, scored.text
    pid = scored.json()["id"]
    assert (
        await client.get(f"/api/v1/predictions/{pid}/export?format=csv", headers=analyst)
    ).status_code == 422
    assert (
        await client.get(f"/api/v1/predictions/missing-pred/export?format=pdf", headers=analyst)
    ).status_code == 404


@marks("G-06", "G-21", "G-22", "C-14", "RC-01", "RC-03", "D-08", "ST-05", "ST-08", "SL-01", "ER-09")
async def test_governance_recipe_dashboard_stories(client: AsyncClient, sign_in: SignIn) -> None:
    analyst = await sign_in("analyst")
    other = await sign_in("practitioner")
    steward = await sign_in("steward")

    pii = await client.post(
        "/api/v1/datasets",
        headers=analyst,
        files={"file": ("notes.csv", io.BytesIO(PII_CSV.encode()), "text/csv")},
        data={"name": "Notes tip", "classification": "internal-general", "visibility": "organisation"},
    )
    assert pii.status_code == 201, pii.text
    assert pii.json().get("classification") in ("restricted-pii", "Restricted · PII", "restricted") or "pii" in str(
        pii.json().get("classification", "")
    ).lower() or pii.json().get("piiColumns")

    upload = await client.post(
        "/api/v1/datasets",
        headers=analyst,
        files={"file": ("gov.csv", io.BytesIO(CSV.encode()), "text/csv")},
        data={"name": "Gov tip", "classification": "internal-general", "visibility": "organisation"},
    )
    assert upload.status_code == 201, upload.text
    dataset_id = upload.json()["id"]
    studio = (await client.get(f"/api/v1/cleaning/{dataset_id}/studio", headers=analyst)).json()
    method = next(m for m in studio["methods"] if m["applicable"])
    added = await client.post(
        f"/api/v1/cleaning/{dataset_id}/recipe/steps",
        headers=analyst,
        json={"methodId": method["id"]},
    )
    assert added.status_code == 201, added.text
    run = await client.post(f"/api/v1/cleaning/{dataset_id}/run", headers=analyst)
    assert run.status_code == 200, run.text
    approve = await client.post(
        f"/api/v1/datasets/{dataset_id}/approve", headers=steward, json={"reason": "ok"}
    )
    assert approve.status_code == 200, approve.text
    reject = await client.post(
        f"/api/v1/datasets/{dataset_id}/reject", headers=steward, json={"reason": "revisit"}
    )
    assert reject.status_code == 200, reject.text
    after = (await client.get(f"/api/v1/datasets/{dataset_id}", headers=analyst)).json()
    assert after["status"] == "rejected" and after.get("queryable") is False

    lineage = await client.get(f"/api/v1/datasets/{dataset_id}/lineage", headers=analyst)
    assert lineage.status_code == 200 and lineage.json()
    private = await client.post(
        "/api/v1/datasets",
        headers=analyst,
        files={"file": ("hid.csv", io.BytesIO(CSV.encode()), "text/csv")},
        data={"name": "Private lineage", "classification": "internal-general", "visibility": "private"},
    )
    assert private.status_code == 201
    priv_id = private.json()["id"]
    assert (await client.get(f"/api/v1/datasets/{priv_id}/lineage", headers=other)).status_code == 404

    assert (
        await client.post(
            f"/api/v1/datasets/{dataset_id}/insights/random-id",
            headers=analyst,
            json={"decision": "accepted"},
        )
    ).status_code == 404
    assert (
        await client.post(
            f"/api/v1/datasets/{dataset_id}/insights/random-id",
            headers=analyst,
            json={"decision": "maybe"},
        )
    ).status_code in (404, 422)

    recipes = (await client.get(f"/api/v1/cleaning/{dataset_id}/recipes", headers=analyst)).json()
    draft = next((r for r in recipes if r.get("status") != "applied"), None)
    applied = next((r for r in recipes if r.get("status") == "applied"), None)
    if draft:
        assert (
            await client.post(
                f"/api/v1/cleaning/{dataset_id}/recipes/{draft['id']}/rerun", headers=analyst
            )
        ).status_code == 404
    assert (
        await client.post(
            f"/api/v1/cleaning/{priv_id}/recipe/import",
            headers=analyst,
            json={"recipeId": "missing-recipe"},
        )
    ).status_code == 404
    if applied:
        assert (
            await client.post(
                f"/api/v1/cleaning/{priv_id}/recipe/import",
                headers=other,
                json={"recipeId": applied["id"]},
            )
        ).status_code in (403, 404)

    me_other = (await client.get("/api/v1/auth/me", headers=other)).json()
    share = await client.post(
        f"/api/v1/datasets/{priv_id}/shares",
        headers=analyst,
        json={"emails": [me_other["email"]]},
    )
    assert share.status_code == 200, share.text
    assert (await client.get(f"/api/v1/datasets/{priv_id}", headers=other)).status_code == 200
    unshare = await client.delete(
        f"/api/v1/datasets/{priv_id}/shares/{me_other['id']}", headers=analyst
    )
    assert unshare.status_code == 204
    assert (await client.get(f"/api/v1/datasets/{priv_id}", headers=other)).status_code == 404

    dash = await client.post(
        "/api/v1/dashboards",
        headers=analyst,
        json={"name": "Private embed", "visibility": "private"},
    )
    assert dash.status_code == 201, dash.text
    embed = await client.post(
        f"/api/v1/dashboards/{dash.json()['id']}/embed",
        headers=analyst,
        json={"enabled": True},
    )
    assert embed.status_code == 422

    link = await client.post("/api/v1/auth/slack-link", headers=analyst)
    assert link.status_code == 200, link.text
    code = link.json()["code"]
    assert len(code) == 9 and code[4] == "-"
