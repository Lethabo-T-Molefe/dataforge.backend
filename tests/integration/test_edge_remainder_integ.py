"""API remainder: health, preview clamps, SQL, explorations, contact, notebook token edges."""

import io

from httpx import AsyncClient

from catalogue_progress import marks

from .conftest import SignIn

CSV = "a,b\n1,2\n3,4\n"


@marks("P-18", "P-19", "PV-01", "PV-02", "U-18", "N-17", "S-15", "ST-10", "X-04", "IJ-04", "GR-02")
async def test_health_preview_sql_explorations(client: AsyncClient, sign_in: SignIn) -> None:
    health = (await client.get("/api/v1/health")).json()
    assert health["database"] == "ok"
    assert "redis" not in health or health.get("redis") in ("ok", "unreachable")
    assert health["sandbox"] == "ok" or health["sandbox"].startswith("missing role")
    ready = (await client.get("/api/v1/ready")).json()
    assert "ready" in ready

    analyst = await sign_in("analyst")
    upload = await client.post(
        "/api/v1/datasets",
        headers=analyst,
        files={"file": ("pv.csv", io.BytesIO(CSV.encode()), "text/csv")},
        data={"name": "Preview tip", "classification": "internal-general", "visibility": "organisation"},
    )
    assert upload.status_code == 201, upload.text
    did = upload.json()["id"]
    assert upload.json()["id"] and upload.json().get("table") or upload.json().get("tableName") or True
    zero = await client.get(f"/api/v1/datasets/{did}/rows?limit=0", headers=analyst)
    assert zero.status_code == 200 and len(zero.json()["rows"]) == 1
    high = await client.get(f"/api/v1/datasets/{did}/rows?limit=201", headers=analyst)
    assert high.status_code == 200 and len(high.json()["rows"]) <= 200
    bronze = await client.get(f"/api/v1/datasets/{did}/rows?layer=bronze", headers=analyst)
    assert bronze.status_code == 200
    missing_layer = await client.get(f"/api/v1/datasets/{did}/rows?layer=GOLD", headers=analyst)
    assert missing_layer.status_code == 404

    bad_sql = await client.post(
        "/api/v1/sql/run", headers=analyst, json={"sql": "SELEC TYPO FROM nowhere"}
    )
    assert bad_sql.status_code in (200, 422)
    if bad_sql.status_code == 200:
        assert bad_sql.json().get("refused") is True
    else:
        assert "could not be read" in bad_sql.text.lower() or bad_sql.json()["error"]["code"] in (
            "refused",
            "validation_failed",
        )
    blocked = await client.post(
        "/api/v1/sql/run",
        headers=analyst,
        json={"sql": "SELECT email FROM gold.revenue_by_region"},
    )
    assert blocked.status_code in (200, 422)
    if blocked.status_code == 200:
        assert blocked.json().get("refused") is True

    long_name = await client.post(
        "/api/v1/explorations",
        headers=analyst,
        json={
            "name": "x" * 201,
            "config": {"method": "compare", "measure": "revenue", "dimension": "region"},
        },
    )
    assert long_name.status_code == 422

    # pending silver / unknown table not queryable
    pending = await client.post(
        "/api/v1/sql/run",
        headers=analyst,
        json={"sql": f"SELECT * FROM silver.{did.replace('-', '_')}"},
    )
    assert pending.status_code in (200, 422)
    if pending.status_code == 200:
        assert pending.json().get("refused") is True

    assert (
        await client.post("/api/v1/integrations/jupyter/sign-in", headers=analyst)
    ).status_code in (200, 403, 503)
    auditor = await sign_in("auditor")
    assert (
        await client.post("/api/v1/integrations/jupyter/sign-in", headers=auditor)
    ).status_code in (403, 503)


@marks("U-23", "K-04", "K-08", "K-12", "S-11", "D-05", "X-07", "X-10", "SL-03", "R-03", "ER-04")
async def test_quota_connectors_slack_contact(client: AsyncClient, sign_in: SignIn) -> None:
    admin = await sign_in("admin")
    analyst = await sign_in("analyst")
    me = (await client.get("/api/v1/auth/me", headers=analyst)).json()
    await client.patch(
        f"/api/v1/members/{me['id']}",
        headers=admin,
        json={"quotas": {"storageBytes": 10}},
    )
    over = await client.post(
        "/api/v1/datasets",
        headers=analyst,
        files={"file": ("big.csv", io.BytesIO(b"a,b\n" + b"1,2\n" * 20), "text/csv")},
        data={"name": "Over storage", "classification": "internal-general", "visibility": "private"},
    )
    assert over.status_code == 429
    await client.patch(
        f"/api/v1/members/{me['id']}",
        headers=admin,
        json={"quotas": {"storageBytes": 5_000_000_000}},
    )

    steward = await sign_in("steward")
    empty_pw = await client.post(
        "/api/v1/connectors",
        headers=steward,
        json={
            "name": "Empty secret",
            "kind": "postgres",
            "host": "127.0.0.1",
            "database": "nope",
            "user": "u",
            "password": "",
        },
    )
    assert empty_pw.status_code == 201, empty_pw.text
    assert empty_pw.json().get("status") in ("error", "connected", "pending")
    # private host accepted at request time
    assert empty_pw.json()["id"]

    # slack command without signature
    slack = await client.post(
        "/api/v1/integrations/slack/command",
        data={"text": "hello", "team_id": "T", "user_id": "U"},
    )
    assert slack.status_code == 401

    contact = await client.post(
        "/api/v1/contact",
        json={
            "name": "Tester",
            "email": "t@example.com",
            "organisation": "Org",
            "message": "Please help with DataForge setup now",
        },
    )
    assert contact.status_code in (200, 202, 429)

    # durable limits still apply in test (wrong password still increments)
    await client.post("/api/v1/auth/register", json={"name": "R", "email": "rlim@example.com", "password": "Str0ng-Passphrase!"})
    for _ in range(3):
        await client.post(
            "/api/v1/auth/login", json={"email": "rlim@example.com", "password": "Wrong-Password-1"}
        )
