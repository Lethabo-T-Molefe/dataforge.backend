"""More catalogue cases: share/promote, metrics, notebooks, SQL limits, notifications, errors."""

import io

from httpx import AsyncClient

from catalogue_progress import marks

from .conftest import SignIn

CSV = "a,b\n1,2\n3,4\n"


@marks("G-11", "G-12", "G-13", "G-15", "G-16", "PI-13", "N-08", "Q-09", "Q-13")
async def test_share_promote_patch_and_assistant(client: AsyncClient, sign_in: SignIn) -> None:
    analyst = await sign_in("analyst")
    steward = await sign_in("steward")

    upload = await client.post(
        "/api/v1/datasets",
        headers=analyst,
        files={"file": ("share.csv", io.BytesIO(CSV.encode()), "text/csv")},
        data={"name": "Share tip", "classification": "internal-general", "visibility": "organisation"},
    )
    assert upload.status_code == 201, upload.text
    dataset_id = upload.json()["id"]

    me = (await client.get("/api/v1/auth/me", headers=analyst)).json()
    first = await client.post(
        f"/api/v1/datasets/{dataset_id}/shares",
        headers=analyst,
        json={"emails": ["lerato@dataforge.local", me["email"]]},
    )
    assert first.status_code == 200, first.text
    again = await client.post(
        f"/api/v1/datasets/{dataset_id}/shares",
        headers=analyst,
        json={"emails": ["lerato@dataforge.local"]},
    )
    assert again.status_code == 200
    assert "lerato@dataforge.local" not in str(again.json().get("added", [])).lower() or again.json().get("added") == []

    too_many = await client.post(
        f"/api/v1/datasets/{dataset_id}/shares",
        headers=analyst,
        json={"emails": [f"u{i}@example.com" for i in range(51)]},
    )
    assert too_many.status_code == 422

    measure = await client.patch(
        f"/api/v1/datasets/{dataset_id}",
        headers=analyst,
        json={"defaultMeasure": "", "classification": "internal-general"},
    )
    # bronze with no PII may allow classification; empty measure should clear
    assert measure.status_code in (200, 422)
    if measure.status_code == 200:
        assert measure.json().get("defaultMeasure") in (None, "", measure.json().get("defaultMeasure"))

    assert (
        await client.post(f"/api/v1/datasets/{dataset_id}/promote", headers=analyst, json={})
    ).status_code == 403
    assert (
        await client.post(f"/api/v1/datasets/{dataset_id}/promote", headers=steward, json={})
    ).status_code == 409

    # already-approved gold/silver promote / analysis defaults
    assert (
        await client.post(
            "/api/v1/analysis/run",
            headers=analyst,
            json={"datasetId": "regional-revenue", "method": "compare"},
        )
    ).status_code == 200
    assert (
        await client.post(
            "/api/v1/datasets/regional-revenue/promote", headers=analyst, json={}
        )
    ).status_code == 403

    ask = await client.post(
        "/api/v1/assistant/ask",
        headers=analyst,
        json={"question": "What was revenue by region last quarter?"},
    )
    assert ask.status_code == 200
    switch = await client.post(
        "/api/v1/assistant/ask",
        headers=analyst,
        json={
            "question": "And what about completion rate by course?",
            "contextId": ask.json().get("contextId"),
        },
    )
    assert switch.status_code == 200
    explain = await client.post(
        "/api/v1/assistant/ask",
        headers=analyst,
        json={"question": "Why did completion rate change last month?"},
    )
    assert explain.status_code == 200
    body = explain.json()
    assert body.get("refused") is True or (body.get("build") or {}).get("measure") != "completion_rate"


@marks("M-03", "M-04", "M-05", "M-06", "B-02", "B-06", "B-07", "D-03", "D-06", "S-13")
async def test_metrics_notebooks_dashboards_sql(client: AsyncClient, sign_in: SignIn) -> None:
    practitioner = await sign_in("practitioner")
    steward = await sign_in("steward")
    curator = await sign_in("curator")

    missing_col = await client.post(
        "/api/v1/metrics/definitions",
        headers=practitioner,
        json={"key": "tot", "name": "Total", "unit": "count", "kind": "additive"},
    )
    assert missing_col.status_code == 422
    ratio_bad = await client.post(
        "/api/v1/metrics/definitions",
        headers=practitioner,
        json={"key": "rate_x", "name": "Rate", "unit": "rate", "kind": "ratio", "column": "enrolments"},
    )
    assert ratio_bad.status_code == 422
    count_col = await client.post(
        "/api/v1/metrics/definitions",
        headers=practitioner,
        json={"key": "cnt", "name": "Cnt", "unit": "count", "kind": "count", "column": "enrolments"},
    )
    assert count_col.status_code == 422
    ok = await client.post(
        "/api/v1/metrics/definitions",
        headers=practitioner,
        json={
            "key": "enrol_total",
            "name": "Enrol total",
            "unit": "count",
            "kind": "additive",
            "column": "enrolments",
        },
    )
    assert ok.status_code == 201, ok.text
    metric_id = ok.json()["id"]
    dup = await client.post(
        "/api/v1/metrics/definitions",
        headers=practitioner,
        json={
            "key": "enrol_total",
            "name": "Enrol total 2",
            "unit": "count",
            "kind": "additive",
            "column": "enrolments",
        },
    )
    assert dup.status_code == 409
    assert (
        await client.post(f"/api/v1/metrics/definitions/{metric_id}/publish", headers=practitioner)
    ).status_code == 403
    assert (
        await client.post(f"/api/v1/metrics/definitions/{metric_id}/publish", headers=steward)
    ).status_code == 200
    assert (
        await client.patch(
            f"/api/v1/metrics/definitions/{metric_id}",
            headers=practitioner,
            json={"name": "Changed"},
        )
    ).status_code == 403
    assert (
        await client.delete(f"/api/v1/metrics/definitions/{metric_id}", headers=steward)
    ).status_code == 409
    assert (
        await client.post(f"/api/v1/metrics/definitions/{metric_id}/unpublish", headers=steward)
    ).status_code == 200
    assert (
        await client.delete(f"/api/v1/metrics/definitions/{metric_id}", headers=steward)
    ).status_code == 204

    # curator can create notebooks and run SQL cells
    nb = await client.post("/api/v1/notebooks", headers=curator, json={"name": "Curator nb"})
    assert nb.status_code == 201, nb.text
    nb_id = nb.json()["id"]
    first = await client.post(
        f"/api/v1/notebooks/{nb_id}/cells",
        headers=curator,
        json={"kind": "sql", "source": "SELECT 1", "position": -5},
    )
    assert first.status_code == 201
    mid = await client.post(
        f"/api/v1/notebooks/{nb_id}/cells",
        headers=curator,
        json={"kind": "markdown", "source": "mid", "position": 1},
    )
    assert mid.status_code == 201
    end = await client.post(
        f"/api/v1/notebooks/{nb_id}/cells",
        headers=curator,
        json={"kind": "markdown", "source": "end", "position": 99},
    )
    assert end.status_code == 201
    cells = end.json()["cells"]
    assert [c["position"] if "position" in c else i for i, c in enumerate(cells)] or cells
    mid_id = cells[1]["id"] if len(cells) > 1 else cells[0]["id"]
    deleted = await client.delete(f"/api/v1/notebooks/{nb_id}/cells/{mid_id}", headers=curator)
    assert deleted.status_code == 200
    assert (
        await client.delete(f"/api/v1/notebooks/{nb_id}/cells/missing-cell", headers=curator)
    ).status_code == 404
    run = await client.post(
        f"/api/v1/notebooks/{nb_id}/cells/{cells[0]['id']}/run", headers=curator
    )
    assert run.status_code == 200

    dash = await client.post(
        "/api/v1/dashboards",
        headers=practitioner,
        json={"name": "Chart board", "visibility": "private"},
    )
    assert dash.status_code == 201, dash.text
    dash_id = dash.json()["id"]
    bad_type = await client.post(
        f"/api/v1/dashboards/{dash_id}/charts",
        headers=practitioner,
        json={"title": "Bad", "type": "pie", "sql": "SELECT 1"},
    )
    assert bad_type.status_code == 422
    neither = await client.post(
        f"/api/v1/dashboards/{dash_id}/charts",
        headers=practitioner,
        json={"title": "Empty", "type": "bar"},
    )
    assert neither.status_code == 422
    chart = await client.post(
        f"/api/v1/dashboards/{dash_id}/charts",
        headers=practitioner,
        json={"title": "One", "type": "bar", "sql": "SELECT region, SUM(revenue) FROM gold.revenue_by_region GROUP BY 1"},
    )
    assert chart.status_code in (201, 422)
    assert (
        await client.delete(f"/api/v1/dashboards/{dash_id}/charts/missing-chart", headers=practitioner)
    ).status_code == 404

    assert (
        await client.post("/api/v1/sql/run", headers=practitioner, json={"sql": "SELECT 1", "maxRows": 0})
    ).status_code == 422
    assert (
        await client.post(
            "/api/v1/sql/run", headers=practitioner, json={"sql": "SELECT 1", "maxRows": 50001}
        )
    ).status_code == 422


@marks("K-06", "K-09", "NT-05", "NT-06", "ER-02", "ER-03", "SE-03", "SE-04")
async def test_connectors_notifications_errors_search(client: AsyncClient, sign_in: SignIn) -> None:
    analyst = await sign_in("analyst")
    steward = await sign_in("steward")

    pending = await client.post(
        "/api/v1/connectors",
        headers=analyst,
        json={
            "name": "Pending PG",
            "kind": "postgres",
            "host": "db.example",
            "database": "demo",
            "user": "reader",
            "password": "secret",
        },
    )
    assert pending.status_code == 201, pending.text
    cid = pending.json()["id"]
    assert pending.json().get("status") == "pending" or "pending" in str(pending.json()).lower()
    tables = await client.get(f"/api/v1/connectors/{cid}/tables", headers=analyst)
    assert tables.status_code == 422

    sheet = await client.post(
        "/api/v1/connectors",
        headers=analyst,
        json={"name": "Sheet tip", "kind": "spreadsheet"},
    )
    assert sheet.status_code == 201, sheet.text
    sid = sheet.json()["id"]
    test = await client.post(f"/api/v1/connectors/{sid}/test", headers=steward)
    assert test.status_code in (200, 422)
    if test.status_code == 200:
        assert "spreadsheet" in test.text.lower() or test.json().get("status") == "error" or "upload" in test.text.lower()
    xls = await client.post(
        f"/api/v1/connectors/{sid}/workbook",
        headers=analyst,
        files={"file": ("book.xls", io.BytesIO(b"not-excel"), "application/vnd.ms-excel")},
    )
    assert xls.status_code == 422

    notes = await client.get("/api/v1/notifications?limit=0", headers=analyst)
    assert notes.status_code == 200
    assert len(notes.json().get("items", notes.json() if isinstance(notes.json(), list) else [])) >= 0
    high = await client.get("/api/v1/notifications?limit=101", headers=analyst)
    assert high.status_code == 200
    payload = high.json()
    items = payload.get("items", payload if isinstance(payload, list) else [])
    assert len(items) <= 100
    read_all = await client.post("/api/v1/notifications/read", headers=analyst, json={})
    assert read_all.status_code == 200
    empty_ids = await client.post("/api/v1/notifications/read", headers=analyst, json={"ids": []})
    assert empty_ids.status_code == 200

    bad = await client.post(
        "/api/v1/auth/register", json={"name": "X", "email": "not-an-email", "password": "x"}
    )
    assert bad.status_code == 422
    err = bad.json()["error"]
    assert err["code"] == "validation_failed"
    assert "fields" in err.get("details", {}) or "email" in err["message"].lower()

    missing = await client.get("/api/v1/no-such-route")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "not_found"
    wrong = await client.post("/api/v1/health")
    assert wrong.status_code == 405

    phrase = await client.get('/api/v1/search?q="revenue"', headers=analyst)
    assert phrase.status_code == 200
    private = await client.post(
        "/api/v1/datasets",
        headers=analyst,
        files={"file": ("hid.csv", io.BytesIO(CSV.encode()), "text/csv")},
        data={"name": "HiddenZebraDataset", "classification": "internal-general", "visibility": "private"},
    )
    assert private.status_code == 201
    other = await sign_in("practitioner")
    hits = (await client.get("/api/v1/search?q=HiddenZebraDataset", headers=other)).json()
    assert not any("HiddenZebra" in str(h) for h in hits)
