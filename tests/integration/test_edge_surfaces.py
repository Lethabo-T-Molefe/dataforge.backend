"""Catalogue cases across cleaning, governance, notebooks, metrics, search, and connectors."""

import io

from httpx import AsyncClient

from catalogue_progress import marks
from dataforge_backend.core.config import settings

from .conftest import SignIn

PASSWORD = "Str0ng-Passphrase!"
CSV = "a,b\n1,2\n3,4\n"


@marks("U-02", "U-04", "N-02", "Q-02", "Q-06", "Q-08")
async def test_upload_analysis_and_assistant_edges(client: AsyncClient, sign_in: SignIn) -> None:
    reg = await client.post(
        "/api/v1/auth/register",
        json={"name": "Unverified User", "email": "unverified-up@example.com", "password": PASSWORD},
    )
    assert reg.status_code == 201
    login = await client.post(
        "/api/v1/auth/login", json={"email": "unverified-up@example.com", "password": PASSWORD}
    )
    headers = {"Authorization": f"Bearer {login.json()['accessToken']}"}
    blocked = await client.post(
        "/api/v1/datasets",
        headers=headers,
        files={"file": ("a.csv", io.BytesIO(CSV.encode()), "text/csv")},
        data={"name": "Blocked", "classification": "internal-general", "visibility": "private"},
    )
    assert blocked.status_code == 403

    analyst = await sign_in("analyst")
    huge = b"x" * (settings.MAX_UPLOAD_MB * 1024 * 1024 + 1)
    over = await client.post(
        "/api/v1/datasets",
        headers=analyst,
        files={"file": ("big.csv", io.BytesIO(huge), "text/csv")},
        data={"name": "Big", "classification": "internal-general", "visibility": "private"},
    )
    assert over.status_code == 422

    pending = await client.post(
        "/api/v1/analysis/run",
        headers=analyst,
        json={"datasetId": "learner-enrolments", "method": "compare"},
    )
    assert pending.status_code == 404

    ask = await client.post(
        "/api/v1/assistant/ask",
        headers=analyst,
        json={"question": "What was revenue by region last quarter?"},
    )
    assert ask.status_code == 200 and ask.json()["refused"] is False
    follow = await client.post(
        "/api/v1/assistant/ask",
        headers=analyst,
        json={"question": "What about Gauteng?", "contextId": ask.json().get("contextId")},
    )
    assert follow.status_code == 200
    ungrounded = await client.post(
        "/api/v1/assistant/ask",
        headers=analyst,
        json={"question": "How many cats own yachts in Atlantis?"},
    )
    assert ungrounded.status_code == 200 and ungrounded.json()["refused"] is True


@marks("C-02", "C-04", "C-05", "C-06", "C-07", "C-11", "G-03", "G-08", "G-09", "G-10", "G-14")
async def test_cleaning_and_governance_edges(client: AsyncClient, sign_in: SignIn) -> None:
    analyst = await sign_in("analyst")
    other = await sign_in("practitioner")
    steward = await sign_in("steward")

    private = await client.post(
        "/api/v1/datasets",
        headers=analyst,
        files={"file": ("priv.csv", io.BytesIO(CSV.encode()), "text/csv")},
        data={"name": "Private tip", "classification": "internal-general", "visibility": "private"},
    )
    assert private.status_code == 201, private.text
    dataset_id = private.json()["id"]

    assert (await client.get(f"/api/v1/cleaning/{dataset_id}/studio", headers=other)).status_code == 404
    assert (await client.get(f"/api/v1/datasets/{dataset_id}", headers=other)).status_code == 404

    # Private datasets hide existence from non-owners (404). Org-visible + non-owner share is 403.
    assert (
        await client.post(
            f"/api/v1/datasets/{dataset_id}/shares",
            headers=other,
            json={"emails": ["thandi@dataforge.local"]},
        )
    ).status_code == 404
    org = await client.post(
        "/api/v1/datasets",
        headers=analyst,
        files={"file": ("org.csv", io.BytesIO(CSV.encode()), "text/csv")},
        data={"name": "Org tip", "classification": "internal-general", "visibility": "organisation"},
    )
    assert org.status_code == 201
    assert (
        await client.post(
            f"/api/v1/datasets/{org.json()['id']}/shares",
            headers=other,
            json={"emails": ["thandi@dataforge.local"]},
        )
    ).status_code == 403
    assert (
        await client.post(
            f"/api/v1/datasets/{dataset_id}/shares",
            headers=analyst,
            json={"emails": ["nobody-here@example.com"]},
        )
    ).status_code == 404
    assert (
        await client.post(
            f"/api/v1/datasets/{dataset_id}/shares",
            headers=analyst,
            json={"emails": ["lerato@dataforge.local"]},
        )
    ).status_code == 200

    auditor = await sign_in("auditor")
    assert (await client.get(f"/api/v1/datasets/{dataset_id}", headers=auditor)).status_code == 200
    assert (
        await client.post(
            f"/api/v1/datasets/{dataset_id}/approve", headers=auditor, json={"reason": "no"}
        )
    ).status_code == 403

    patch = await client.patch(
        f"/api/v1/datasets/{dataset_id}",
        headers=analyst,
        json={"name": "  Renamed tip  ", "description": "x" * 10},
    )
    assert patch.status_code == 200 and patch.json()["name"] == "Renamed tip"
    assert (
        await client.patch(f"/api/v1/datasets/{dataset_id}", headers=other, json={"name": "Nope"})
    ).status_code == 404
    assert (
        await client.patch(f"/api/v1/datasets/{org.json()['id']}", headers=other, json={"name": "Nope"})
    ).status_code == 403

    studio = (await client.get(f"/api/v1/cleaning/{dataset_id}/studio", headers=analyst)).json()
    method = next(m for m in studio["methods"] if m["applicable"])
    assert (
        await client.post(
            f"/api/v1/cleaning/{dataset_id}/recipe/steps",
            headers=analyst,
            json={"methodId": method["id"], "status": "pending", "params": method["params"]},
        )
    ).status_code == 422
    added = await client.post(
        f"/api/v1/cleaning/{dataset_id}/recipe/steps",
        headers=analyst,
        json={"methodId": method["id"], "status": "accepted", "params": method["params"]},
    )
    assert added.status_code == 201, added.text
    step_id = added.json()["steps"][-1]["stepId"]
    assert (
        await client.delete(f"/api/v1/cleaning/{dataset_id}/recipe/steps/missing-step", headers=analyst)
    ).status_code == 404
    assert (
        await client.post(
            f"/api/v1/cleaning/{dataset_id}/recipe/reorder",
            headers=analyst,
            json={"stepIds": [step_id, step_id]},
        )
    ).status_code == 422

    recipe = (await client.get(f"/api/v1/cleaning/{dataset_id}/recipe", headers=analyst)).json()
    for s in recipe.get("steps") or []:
        await client.patch(
            f"/api/v1/cleaning/{dataset_id}/recipe/steps/{s['stepId']}",
            headers=analyst,
            json={"status": "skipped"},
        )
    assert (await client.post(f"/api/v1/cleaning/{dataset_id}/run", headers=analyst)).status_code == 422

    preview = await client.post(
        f"/api/v1/cleaning/{dataset_id}/preview",
        headers=analyst,
        json={"methodId": "drop-columns", "params": {"columns": ["no_such_column"]}},
    )
    assert preview.status_code in (404, 422)

    assert (
        await client.post(
            "/api/v1/datasets/course-completions/approve", headers=steward, json={"reason": "again"}
        )
    ).status_code == 409


@marks(
    "B-01", "B-03", "B-04", "B-05", "M-01", "M-02", "D-01", "D-02",
    "K-02", "K-03", "SE-02", "AU-03", "AU-04", "Z-06",
)
async def test_notebooks_metrics_dashboards_connectors_search_audit(
    client: AsyncClient, sign_in: SignIn
) -> None:
    analyst = await sign_in("analyst")
    practitioner = await sign_in("practitioner")
    executive = await sign_in("executive")

    assert (
        await client.post("/api/v1/notebooks", headers=analyst, json={"name": "Nope"})
    ).status_code == 403
    notebook = await client.post(
        "/api/v1/notebooks", headers=practitioner, json={"name": "Scratch"}
    )
    assert notebook.status_code == 201, notebook.text
    nb_id = notebook.json()["id"]
    assert (
        await client.patch(
            f"/api/v1/notebooks/{nb_id}", headers=practitioner, json={"visibility": "public"}
        )
    ).status_code == 422
    assert (
        await client.post(
            f"/api/v1/notebooks/{nb_id}/cells",
            headers=practitioner,
            json={"kind": "python", "source": "1"},
        )
    ).status_code == 422
    cell = await client.post(
        f"/api/v1/notebooks/{nb_id}/cells",
        headers=practitioner,
        json={"kind": "markdown", "source": "hi"},
    )
    assert cell.status_code == 201, cell.text
    cell_id = cell.json()["cells"][-1]["id"]
    conflict = await client.put(
        f"/api/v1/notebooks/{nb_id}/cells/{cell_id}",
        headers=practitioner,
        json={"kind": "markdown", "source": "bye", "revision": 999},
    )
    assert conflict.status_code == 409

    assert (
        await client.post(
            "/api/v1/metrics/definitions",
            headers=analyst,
            json={"key": "bad", "name": "Bad", "unit": "count", "kind": "additive", "column": "x"},
        )
    ).status_code == 403
    assert (
        await client.post(
            "/api/v1/metrics/definitions",
            headers=practitioner,
            json={
                "key": "Bad Key!",
                "name": "Bad",
                "unit": "count",
                "kind": "additive",
                "column": "enrolments",
            },
        )
    ).status_code == 422

    assert (
        await client.post(
            "/api/v1/dashboards",
            headers=executive,
            json={"name": "Exec board", "visibility": "private"},
        )
    ).status_code == 403
    assert (
        await client.post(
            "/api/v1/dashboards", headers=analyst, json={"name": "", "visibility": "private"}
        )
    ).status_code == 422
    assert (
        await client.post(
            "/api/v1/dashboards", headers=analyst, json={"name": "Board", "visibility": "Private"}
        )
    ).status_code == 422

    assert (
        await client.post("/api/v1/connectors", headers=analyst, json={"name": "Nope", "kind": "mysql"})
    ).status_code == 422
    assert (
        await client.post("/api/v1/connectors", headers=analyst, json={"name": "", "kind": "postgres"})
    ).status_code == 422
    assert (
        await client.post(
            "/api/v1/connectors",
            headers=analyst,
            json={"name": "Warehouse", "kind": "postgres", "host": "db.example", "database": "x"},
        )
    ).status_code == 422

    short = await client.get("/api/v1/search?q=a", headers=analyst)
    assert short.status_code == 200 and short.json() == []
    cut = await client.get(f"/api/v1/search?q={'x' * 250}", headers=analyst)
    assert cut.status_code == 200

    auditor = await sign_in("auditor")
    events = await client.get(
        "/api/v1/audit/events?limit=2000&since=2099-01-01T00:00:00Z&until=2099-01-02T00:00:00Z",
        headers=auditor,
    )
    assert events.status_code == 200
    body = events.json()
    assert body.get("total", 0) == 0 or body.get("items") == []
    capped = await client.get("/api/v1/audit/events?limit=5000", headers=auditor)
    assert capped.status_code == 200
    assert len(capped.json()["items"]) <= 1000
