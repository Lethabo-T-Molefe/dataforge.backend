"""Trained outcome models end to end: train, second-person review, scoring and the governed views."""

from typing import Any

from httpx import AsyncClient

from .conftest import SignIn

BODY: dict[str, Any] = {
    "datasetId": "learner-progress",
    "name": "test-dropout-risk",
    "version": "1.0",
    "outcome": {
        "column": "outcome",
        "positive": "Dropped out",
        "unlabelled": ["In progress"],
        "label": "drops out",
        "entity": "learner",
    },
    "protected": ["gender", "population_group", "disability", "age_band"],
    "trainUntil": "2025-04",
    "algorithms": ["logistic"],
    "cvFolds": 3,
}
PERSONAL_OR_PROTECTED = {"full_name", "cell_number", "id_number", "learner_id", *BODY["protected"]}


async def _ok(response: Any, code: int = 200) -> Any:
    assert response.status_code == code, response.text
    return response.json()


async def test_the_seeded_model_was_trained_reviewed_and_scored(client: AsyncClient, sign_in: SignIn) -> None:
    curator = await sign_in("curator")
    models = await _ok(await client.get("/api/v1/predictions/models", headers=curator))
    seeded = {m["version"]: m for m in models if m["name"] == "learner-dropout-risk"}
    assert seeded["1.0-lr"]["status"] == "active" and seeded["1.0-lr"]["kind"] == "trained"
    assert (
        seeded["1.0-lr"]["trainedBy"] == "Zola Nkosi" and seeded["1.0-lr"]["approvedBy"] == "Naledi Khumalo"
    )
    assert seeded["1.0-gb"]["status"] == "draft"
    assert seeded["1.0-lr"]["metrics"]["rocAuc"] > 0.65

    analyst = await sign_in("analyst")
    visible = await _ok(await client.get("/api/v1/predictions/models", headers=analyst))
    # Everyone else sees approved versions only.
    assert {m["status"] for m in visible} == {"active"}
    assert "1.0-gb" not in {m["version"] for m in visible if m["name"] == "learner-dropout-risk"}

    detail = await _ok(
        await client.get(f"/api/v1/predictions/models/{seeded['1.0-lr']['id']}", headers=analyst)
    )
    assert detail["card"]["performance"]["rocAuc"] == detail["fullMetrics"]["rocAuc"]
    assert detail["lastScoring"]["rows"] > 0
    # Group-level fairness figures are for reviewers; everyone sees whether flags were raised.
    assert "attributes" not in detail["fairness"] and "flags" in detail["fairness"]
    roles = {p["name"]: p["role"] for p in detail["features"]}
    assert (
        roles["full_name"] == "excluded" and roles["gender"] == "protected" and roles["dropped"] == "excluded"
    )


async def test_training_is_for_curators_and_approval_is_for_someone_else(
    client: AsyncClient, sign_in: SignIn
) -> None:
    analyst = await sign_in("analyst")
    curator = await sign_in("curator")
    steward = await sign_in("steward")
    assert (
        await client.post("/api/v1/predictions/models/train", headers=analyst, json=BODY)
    ).status_code == 403

    options = await _ok(
        await client.get(
            "/api/v1/predictions/models/train/options?datasetId=learner-progress", headers=curator
        )
    )
    assert "outcome" in {o["name"] for o in options["outcomes"]}
    assert set(BODY["protected"]) <= set(options["protected"])
    preview = await _ok(
        await client.post("/api/v1/predictions/models/train/preview", headers=curator, json=BODY)
    )
    assert {p["name"]: p["role"] for p in preview["plan"]}["cell_number"] == "excluded"

    drafts = await _ok(await client.post("/api/v1/predictions/models/train", headers=curator, json=BODY), 201)
    assert len(drafts) == 1 and drafts[0]["status"] == "draft" and drafts[0]["version"] == "1.0"
    model_id = drafts[0]["id"]
    again = await client.post("/api/v1/predictions/models/train", headers=curator, json=BODY)
    assert again.status_code == 409

    # Not by the person who trained it; and not without a reason when rejecting.
    own = await client.post(f"/api/v1/predictions/models/{model_id}/approve", headers=curator, json={})
    assert own.status_code == 403
    no_reason = await client.post(f"/api/v1/predictions/models/{model_id}/reject", headers=steward, json={})
    assert no_reason.status_code == 422
    # A draft cannot score yet.
    early = await client.post(f"/api/v1/predictions/models/{model_id}/score", headers=steward)
    assert early.status_code == 409

    note = "Checked metrics against the heuristic and the fairness tab; follow-up agreed with facilitators."
    approved = await _ok(
        await client.post(
            f"/api/v1/predictions/models/{model_id}/approve", headers=steward, json={"note": note}
        )
    )
    assert approved["status"] == "active" and approved["approvedBy"] == "Naledi Khumalo"

    run = await _ok(await client.post(f"/api/v1/predictions/models/{model_id}/score", headers=steward))
    assert run["rows"] > 0 and run["drift"]["status"] in ("stable", "watch", "drift")
    # The 2026 cohorts log in far more than the training cohorts did.
    assert run["drift"]["columns"][0]["column"] == "lms_logins"

    view = await _ok(
        await client.get(f"/api/v1/predictions/models/{model_id}/scored?by=programme_type", headers=analyst)
    )
    assert view["scored"] is True and view["groupedBy"] == "programme_type"
    assert view["title"] == "Learners likely to drop out"
    assert not PERSONAL_OR_PROTECTED & set(view["dimensions"])
    for record in view["records"]:
        assert not PERSONAL_OR_PROTECTED & set(record)
        assert len(record["record"]) == 8 and record["drivers"]
    assert all(g["records"] >= 5 for g in view["groups"])

    # A trained model scores records, not cohorts.
    cohort = await client.post(
        "/api/v1/predictions/score", headers=analyst, json={"modelId": model_id, "cohort": {}}
    )
    assert cohort.status_code == 422

    pdf = await client.get(f"/api/v1/predictions/models/{model_id}/card?format=pdf", headers=analyst)
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")

    audit = await _ok(await client.get(f"/api/v1/audit/events?resource_id={model_id}", headers=steward))
    actions = {e["action"] for e in audit["items"]}
    assert {"Trained model version", "Approved model version", "Scored records with model"} <= actions


async def test_a_draft_is_hidden_from_members_who_do_not_review_models(
    client: AsyncClient, sign_in: SignIn
) -> None:
    curator = await sign_in("curator")
    analyst = await sign_in("analyst")
    models = await _ok(await client.get("/api/v1/predictions/models", headers=curator))
    draft = next(m for m in models if m["name"] == "learner-dropout-risk" and m["status"] == "draft")
    hidden = await client.get(f"/api/v1/predictions/models/{draft['id']}", headers=analyst)
    assert hidden.status_code == 404
