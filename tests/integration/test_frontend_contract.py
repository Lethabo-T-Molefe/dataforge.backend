"""Behaviour the web app relies on: what viewers may trigger, what an empty value means, and where links lead."""

import csv
import io

from httpx import AsyncClient

from .conftest import SignIn

API = "/api/v1"


async def test_cleaning_studio_never_writes_for_viewers(client: AsyncClient, sign_in: SignIn) -> None:
    executive = await sign_in("executive")
    steward = await sign_in("steward")

    before = (await client.get(f"{API}/cleaning/course-completions/recipes", headers=steward)).json()
    studio = await client.get(f"{API}/cleaning/course-completions/studio", headers=executive)
    assert studio.status_code == 200, studio.text
    body = studio.json()
    # A viewer sees the recipe that produced the table, and cannot edit it.
    assert body["canEdit"] is False
    assert body["recipe"]["status"] == "applied"
    recipe = await client.get(f"{API}/cleaning/course-completions/recipe", headers=executive)
    assert recipe.json()["id"] == body["recipe"]["id"]
    # Looking did not start a draft.
    after = (await client.get(f"{API}/cleaning/course-completions/recipes", headers=steward)).json()
    assert [r["id"] for r in after] == [r["id"] for r in before]

    refused = await client.post(
        f"{API}/cleaning/course-completions/recipe/steps",
        headers=executive,
        json={"methodId": "dup-keep-first"},
    )
    assert refused.status_code == 403


async def test_unknown_method_is_not_found_and_skips_always_save(
    client: AsyncClient, sign_in: SignIn
) -> None:
    analyst = await sign_in("analyst")
    path = f"{API}/cleaning/learner-enrolments"

    unknown = await client.post(f"{path}/preview", headers=analyst, json={"methodId": "no-such-method"})
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "not_found"

    studio = (await client.get(f"{path}/studio", headers=analyst)).json()
    assert studio["canEdit"] is True
    idle = next(m for m in studio["methods"] if not m["applicable"])
    # Accepting a method that has nothing to act on is refused with the reason.
    accepted = await client.post(
        f"{path}/recipe/steps",
        headers=analyst,
        json={"methodId": idle["id"], "status": "accepted", "params": idle["params"]},
    )
    assert accepted.status_code == 422, accepted.text
    # Skipping it is a decision like any other, and is recorded.
    skipped = await client.post(
        f"{path}/recipe/steps",
        headers=analyst,
        json={"methodId": idle["id"], "status": "skipped", "params": idle["params"]},
    )
    assert skipped.status_code == 201, skipped.text
    step = skipped.json()["steps"][-1]
    assert step["status"] == "skipped" and step["summary"] == "Skipped" and step["code"]

    removed = await client.delete(f"{path}/recipe/steps/{step['stepId']}", headers=analyst)
    assert removed.status_code == 200
    assert all(s["stepId"] != step["stepId"] for s in removed.json()["steps"])


async def test_clearing_a_quota_returns_to_the_role_default(client: AsyncClient, sign_in: SignIn) -> None:
    admin = await sign_in("admin")
    members = (await client.get(f"{API}/members", headers=admin)).json()
    analyst = next(m for m in members if m["roleId"] == "analyst")
    default = analyst["limits"]["aiQueries"]
    assert default == 200

    raised = await client.patch(
        f"{API}/members/{analyst['id']}", headers=admin, json={"quotas": {"aiQueries": 500}}
    )
    assert raised.status_code == 200, raised.text
    assert raised.json()["limits"]["aiQueries"] == 500
    # The other limits were not touched.
    assert raised.json()["limits"]["computeSeconds"] == analyst["limits"]["computeSeconds"]

    # An empty value removes the override. It must never read as "unlimited".
    cleared = await client.patch(
        f"{API}/members/{analyst['id']}", headers=admin, json={"quotas": {"aiQueries": None}}
    )
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["limits"]["aiQueries"] == default

    negative = await client.patch(
        f"{API}/members/{analyst['id']}", headers=admin, json={"quotas": {"aiQueries": -1}}
    )
    assert negative.status_code == 422


async def test_dashboard_sharing_can_be_withdrawn(client: AsyncClient, sign_in: SignIn) -> None:
    analyst = await sign_in("analyst")
    curator = await sign_in("curator")
    created = await client.post(
        f"{API}/dashboards", headers=analyst, json={"name": "Shared then withdrawn", "visibility": "private"}
    )
    assert created.status_code == 201, created.text
    dashboard_id = created.json()["id"]
    assert (await client.get(f"{API}/dashboards/{dashboard_id}", headers=curator)).status_code == 404

    shared = await client.post(
        f"{API}/dashboards/{dashboard_id}/shares", headers=analyst, json={"emails": ["zola@dataforge.local"]}
    )
    assert shared.status_code == 200, shared.text
    member = shared.json()["sharedMembers"][0]
    assert (await client.get(f"{API}/dashboards/{dashboard_id}", headers=curator)).status_code == 200

    # Only the owner withdraws a share.
    not_owner = await client.delete(f"{API}/dashboards/{dashboard_id}/shares/{member['id']}", headers=curator)
    assert not_owner.status_code == 403
    withdrawn = await client.delete(f"{API}/dashboards/{dashboard_id}/shares/{member['id']}", headers=analyst)
    assert withdrawn.status_code == 200, withdrawn.text
    assert withdrawn.json()["sharedMembers"] == []
    assert (await client.get(f"{API}/dashboards/{dashboard_id}", headers=curator)).status_code == 404

    assert (await client.delete(f"{API}/dashboards/{dashboard_id}", headers=analyst)).status_code == 204


async def test_search_links_and_connector_owner(client: AsyncClient, sign_in: SignIn) -> None:
    steward = await sign_in("steward")
    hits = (await client.get(f"{API}/search?q=enrolment", headers=steward)).json()
    by_type = {hit["type"]: hit for hit in hits}
    assert by_type["dataset"]["href"].startswith("/datasets/")
    # Links lead to screens the web app has.
    assert by_type["metric"]["href"].startswith("/semantic#")
    notebooks = [hit for hit in hits if hit["type"] == "notebook"]
    assert all(hit["href"].startswith("/analyse/code?notebook=") for hit in notebooks)

    connectors = (await client.get(f"{API}/connectors", headers=steward)).json()
    assert connectors and all("ownerId" in c and "password" not in c["config"] for c in connectors)


async def test_audit_export_cannot_carry_spreadsheet_formulas(client: AsyncClient, sign_in: SignIn) -> None:
    steward = await sign_in("steward")
    # A reason is free text written by a member, and ends up in the audit trail.
    rejected = await client.post(
        f"{API}/datasets/support-tickets/reject",
        headers=steward,
        json={"reason": '=HYPERLINK("http://example.invalid","open me")'},
    )
    assert rejected.status_code == 200, rejected.text

    export = await client.get(f"{API}/audit/export", headers=steward)
    assert export.status_code == 200
    rows = list(csv.reader(io.StringIO(export.text)))
    details = [row[7] for row in rows[1:]]
    assert any(detail.startswith("'=HYPERLINK") for detail in details)
    assert not any(detail.startswith(("=", "+", "-", "@")) for detail in details)


async def test_roles_that_do_not_analyse_get_an_empty_catalogue(client: AsyncClient, sign_in: SignIn) -> None:
    auditor = await sign_in("auditor")
    catalogue = await client.get(f"{API}/analysis/catalogue", headers=auditor)
    assert catalogue.status_code == 200, catalogue.text
    assert catalogue.json()["datasets"] == [] and catalogue.json()["methods"]
    # Listing nothing is not permission to run something.
    run = await client.post(
        f"{API}/analysis/run", headers=auditor, json={"datasetId": "regional-revenue", "method": "compare"}
    )
    assert run.status_code in (403, 404)


async def test_internal_email_domains_can_sign_in(client: AsyncClient) -> None:
    # The address is well formed, so the answer is about the credentials, not the domain.
    response = await client.post(
        f"{API}/auth/login", json={"email": "Thandi@DataForge.local", "password": "not-the-password"}
    )
    assert response.status_code == 401, response.text
    malformed = await client.post(f"{API}/auth/login", json={"email": "thandi@", "password": "x"})
    assert malformed.status_code == 422


async def test_contact_form_reaches_the_platform_contact(client: AsyncClient) -> None:
    from dataforge_backend.infrastructure import email as mail

    mail.sent.clear()
    sent = await client.post(
        f"{API}/contact",
        json={
            "name": "Nomsa  Zulu\nBcc: someone@example.org",
            "email": "nomsa@example.org",
            "organisation": "Ubuntu Learning",
            "message": "We would like to see a demonstration next month.",
        },
    )
    assert sent.status_code == 202, sent.text
    message = mail.sent[-1]
    assert message["Reply-To"] == "nomsa@example.org"
    # Whatever is typed into the name, it stays one header line: no header can be smuggled in.
    assert "\n" not in message["Subject"] and message["Bcc"] is None
    assert "demonstration next month" in message.get_content()

    too_short = await client.post(
        f"{API}/contact", json={"name": "N", "email": "nomsa@example.org", "message": "Hi"}
    )
    assert too_short.status_code == 422
