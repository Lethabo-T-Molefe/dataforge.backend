import re

import pyotp
from httpx import AsyncClient
from sqlalchemy import text

from catalogue_progress import marks
from dataforge_backend.core.database import SessionLocal
from dataforge_backend.infrastructure import email

from .conftest import SignIn

PASSWORD = "Str0ng-Passphrase!"


async def _register(client: AsyncClient, address: str) -> str:
    response = await client.post(
        "/api/v1/auth/register", json={"name": "Kagiso Molefe", "email": address, "password": PASSWORD}
    )
    assert response.status_code == 201, response.text
    message = email.sent[-1]
    token = re.search(r"token=([\w-]+)", message.get_content())
    assert token is not None
    return token.group(1)


@marks("I-01", "I-02", "I-09", "I-11", "I-23", "I-36")
async def test_register_verify_login_and_me(client: AsyncClient) -> None:
    token = await _register(client, "kagiso@example.com")
    assert (
        await client.post(
            "/api/v1/auth/register", json={"name": "K", "email": "kagiso@example.com", "password": PASSWORD}
        )
    ).status_code == 409
    login = await client.post(
        "/api/v1/auth/login", json={"email": "kagiso@example.com", "password": PASSWORD}
    )
    assert login.status_code == 200 and login.json()["mfaRequired"] is False
    headers = {"Authorization": f"Bearer {login.json()['accessToken']}"}
    me = (await client.get("/api/v1/auth/me", headers=headers)).json()
    assert me["verified"] is False and me["roleId"] == "analyst" and me["initials"] == "KM"
    # Unverified members cannot analyse yet.
    blocked = await client.post(
        "/api/v1/analysis/run", headers=headers, json={"datasetId": "course-completions", "method": "compare"}
    )
    assert blocked.status_code == 403
    assert (await client.post("/api/v1/auth/verify-email", json={"token": token})).status_code == 200
    assert (await client.get("/api/v1/auth/me", headers=headers)).json()["verified"] is True


@marks("I-04", "I-19", "ST-02")
async def test_weak_passwords_and_lockout(client: AsyncClient) -> None:
    weak = await client.post(
        "/api/v1/auth/register", json={"name": "Weak", "email": "weak@example.com", "password": "short"}
    )
    assert weak.status_code == 422
    await _register(client, "locky@example.com")
    for _ in range(5):
        wrong = await client.post(
            "/api/v1/auth/login", json={"email": "locky@example.com", "password": "Wrong-Password-1"}
        )
        assert wrong.status_code == 401
    locked = await client.post(
        "/api/v1/auth/login", json={"email": "locky@example.com", "password": PASSWORD}
    )
    assert locked.status_code == 401 and "Try again" in locked.json()["error"]["message"]


@marks("I-25", "I-26", "I-29", "ST-03")
async def test_admin_must_enrol_mfa_before_session(client: AsyncClient) -> None:
    await _register(client, "admin-mfa@example.com")
    async with SessionLocal() as session:
        await session.execute(
            text("UPDATE app.users SET role = 'admin' WHERE email = 'admin-mfa@example.com'")
        )
        await session.commit()
    challenge = (
        await client.post(
            "/api/v1/auth/login", json={"email": "admin-mfa@example.com", "password": PASSWORD}
        )
    ).json()
    assert challenge.get("mfaEnrollmentRequired") is True
    assert "accessToken" not in challenge
    assert challenge["otpauthUri"].startswith("otpauth://")
    totp = pyotp.TOTP(challenge["secret"])
    bad = await client.post(
        "/api/v1/auth/mfa", json={"challengeToken": challenge["challengeToken"], "code": "000000"}
    )
    assert bad.status_code == 401
    # Sign in again for a fresh enrollment challenge after the failed attempt.
    challenge = (
        await client.post(
            "/api/v1/auth/login", json={"email": "admin-mfa@example.com", "password": PASSWORD}
        )
    ).json()
    totp = pyotp.TOTP(challenge["secret"])
    good = await client.post(
        "/api/v1/auth/mfa",
        json={"challengeToken": challenge["challengeToken"], "code": totp.now()},
    )
    assert good.status_code == 200 and good.json()["user"]["mfaEnabled"] is True
    assert good.json()["user"]["roleId"] == "admin"


@marks("I-24", "I-37", "I-38")
async def test_mfa_is_enforced_once_enabled(client: AsyncClient) -> None:
    await _register(client, "mfa@example.com")
    first = (
        await client.post("/api/v1/auth/login", json={"email": "mfa@example.com", "password": PASSWORD})
    ).json()
    headers = {"Authorization": f"Bearer {first['accessToken']}"}
    setup = (await client.post("/api/v1/auth/mfa/setup", headers=headers)).json()
    assert setup["otpauthUri"].startswith("otpauth://")
    totp = pyotp.TOTP(setup["secret"])
    assert (
        await client.post("/api/v1/auth/mfa/enable", headers=headers, json={"code": "000000"})
    ).status_code == 422
    assert (
        await client.post("/api/v1/auth/mfa/enable", headers=headers, json={"code": totp.now()})
    ).status_code == 204
    challenge = (
        await client.post("/api/v1/auth/login", json={"email": "mfa@example.com", "password": PASSWORD})
    ).json()
    assert challenge["mfaRequired"] is True and "accessToken" not in challenge
    bad = await client.post(
        "/api/v1/auth/mfa", json={"challengeToken": challenge["challengeToken"], "code": "123456"}
    )
    assert bad.status_code == 401
    good = await client.post(
        "/api/v1/auth/mfa", json={"challengeToken": challenge["challengeToken"], "code": totp.now()}
    )
    assert good.status_code == 200 and good.json()["user"]["mfaEnabled"] is True


@marks("I-46", "I-48", "Z-01", "ST-04")
async def test_refresh_rotation_detects_reuse(client: AsyncClient) -> None:
    await _register(client, "rotate@example.com")
    login = await client.post(
        "/api/v1/auth/login", json={"email": "rotate@example.com", "password": PASSWORD}
    )
    # The refresh token never reaches page scripts: it is only in an HttpOnly cookie for the auth endpoints.
    assert "refreshToken" not in login.json() and login.json()["accessToken"]
    cookie = login.headers["set-cookie"].lower()
    assert "httponly" in cookie and "path=/api/v1/auth" in cookie and "samesite=lax" in cookie
    first = client.cookies["df_refresh"]
    rotated = await client.post("/api/v1/auth/refresh", json={})
    assert rotated.status_code == 200 and "refreshToken" not in rotated.json()
    second = client.cookies["df_refresh"]
    assert second != first
    client.cookies.clear()
    assert (await client.post("/api/v1/auth/refresh", json={})).status_code == 401
    # Presenting the old token again revokes the whole family, including the newly issued one.
    assert (await client.post("/api/v1/auth/refresh", json={"refreshToken": first})).status_code == 401
    assert (await client.post("/api/v1/auth/refresh", json={"refreshToken": second})).status_code == 401


@marks("I-51")
async def test_signing_out_ends_the_cookie_session(client: AsyncClient) -> None:
    await _register(client, "leave@example.com")
    login = await client.post("/api/v1/auth/login", json={"email": "leave@example.com", "password": PASSWORD})
    access = login.json()["accessToken"]
    token = client.cookies["df_refresh"]
    assert (await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {access}"})).status_code == 200
    assert (await client.post("/api/v1/auth/logout", json={}, headers={"Authorization": f"Bearer {access}"})).status_code == 204
    assert "df_refresh" not in client.cookies
    assert (await client.post("/api/v1/auth/refresh", json={"refreshToken": token})).status_code == 401
    # The access JWT must stop working immediately — not only after it expires.
    assert (await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {access}"})).status_code == 401


@marks("A-01", "G-01")
async def test_roles_gate_endpoints(client: AsyncClient, sign_in: SignIn) -> None:
    analyst = await sign_in("analyst")
    assert (await client.get("/api/v1/auth/me")).status_code == 401
    assert (
        await client.post(
            "/api/v1/datasets/course-completions/approve", headers=analyst, json={"reason": "x"}
        )
    ).status_code == 403
    assert (await client.get("/api/v1/policies", headers=analyst)).status_code == 403
    admin = await sign_in("admin")
    members = (await client.get("/api/v1/members", headers=admin)).json()
    assert {m["roleId"] for m in members} >= {
        "analyst",
        "practitioner",
        "steward",
        "executive",
        "admin",
        "auditor",
        "curator",
    }
    thandi = next(m for m in members if m["email"] == "thandi@dataforge.local")
    assert thandi["quotas"]["ai"].endswith("/ 200")


@marks("AU-01", "AU-05", "AU-07", "ST-14")
async def test_audit_log_cannot_be_altered(client: AsyncClient, sign_in: SignIn) -> None:
    auditor = await sign_in("auditor")
    events = (await client.get("/api/v1/audit/events?limit=500", headers=auditor)).json()
    actions = {e["action"] for e in events["items"]}
    assert {"Uploaded dataset", "Approved dataset", "Ran AI query", "Raised alert"} <= actions
    async with SessionLocal() as session:
        for statement in ("UPDATE app.audit_events SET detail = 'x'", "DELETE FROM app.audit_events"):
            try:
                await session.execute(text(statement))
                raise AssertionError(f"{statement} should have been rejected")
            except Exception as error:  # noqa: BLE001 - asserting the trigger's error
                assert "append-only" in str(error)
                await session.rollback()
    export = await client.get("/api/v1/audit/export", headers=auditor)
    assert export.status_code == 200 and export.text.startswith("id,occurred_at")


@marks("I-15")
async def test_resend_stays_silent_for_unknown_verified_and_disabled(client: AsyncClient, sign_in: SignIn) -> None:
    token = await _register(client, "still-open@example.com")
    assert (await client.post("/api/v1/auth/verify-email", json={"token": token})).status_code == 200
    await _register(client, "held@example.com")
    admin = await sign_in("admin")
    members = (await client.get("/api/v1/members", headers=admin)).json()
    held = next(m for m in members if m["email"] == "held@example.com")
    assert (
        await client.patch(f"/api/v1/members/{held['id']}", headers=admin, json={"status": "disabled"})
    ).status_code == 200
    email.sent.clear()
    for address in ("nobody@example.com", "still-open@example.com", "held@example.com"):
        response = await client.post("/api/v1/auth/resend-verification", json={"email": address})
        assert response.status_code == 202 and response.json() == {"accepted": True}
    assert email.sent == []


@marks("I-17", "I-18", "I-21", "I-42", "I-53", "I-54")
async def test_login_disabled_forgot_and_demo(client: AsyncClient, sign_in: SignIn) -> None:
    unknown = await client.post(
        "/api/v1/auth/login", json={"email": "ghost@example.com", "password": PASSWORD}
    )
    assert unknown.status_code == 401 and "not correct" in unknown.json()["error"]["message"]
    await _register(client, "wrong-pw@example.com")
    email.sent.clear()
    wrong = await client.post(
        "/api/v1/auth/login", json={"email": "wrong-pw@example.com", "password": "Wrong-Password-1"}
    )
    assert wrong.status_code == 401 and "not correct" in wrong.json()["error"]["message"]
    await _register(client, "disabled-now@example.com")
    admin = await sign_in("admin")
    members = (await client.get("/api/v1/members", headers=admin)).json()
    held = next(m for m in members if m["email"] == "disabled-now@example.com")
    assert (
        await client.patch(f"/api/v1/members/{held['id']}", headers=admin, json={"status": "disabled"})
    ).status_code == 200
    disabled = await client.post(
        "/api/v1/auth/login", json={"email": "disabled-now@example.com", "password": PASSWORD}
    )
    assert disabled.status_code == 401 and "disabled" in disabled.json()["error"]["message"]
    email.sent.clear()
    for address in ("nobody@example.com", "disabled-now@example.com"):
        forgot = await client.post("/api/v1/auth/forgot-password", json={"email": address})
        assert forgot.status_code == 202
    assert email.sent == []
    demo = await client.get("/api/v1/auth/demo")
    assert demo.status_code == 200 and "enabled" in demo.json()
    if not demo.json()["enabled"]:
        blocked = await client.post("/api/v1/auth/demo", json={"role": "analyst"})
        assert blocked.status_code == 403


@marks("A-04", "A-05", "Z-08")
async def test_the_last_admin_cannot_be_removed(client: AsyncClient, sign_in: SignIn) -> None:
    admin = await sign_in("admin")
    members = (await client.get("/api/v1/members", headers=admin)).json()
    amina = next(m for m in members if m["email"] == "amina@dataforge.local")
    others = [m for m in members if m["roleId"] == "admin" and m["id"] != amina["id"]]
    if not others:
        promoted = next(m for m in members if m["roleId"] == "analyst")
        assert (
            await client.patch(f"/api/v1/members/{promoted['id']}", headers=admin, json={"role": "admin"})
        ).status_code == 200
        others = [promoted]
    own = await client.patch(f"/api/v1/members/{amina['id']}", headers=admin, json={"status": "disabled"})
    assert own.status_code == 409 and "own account" in own.json()["error"]["message"]
    for other in others:
        assert (
            await client.patch(f"/api/v1/members/{other['id']}", headers=admin, json={"role": "analyst"})
        ).status_code == 200
    last = await client.patch(f"/api/v1/members/{amina['id']}", headers=admin, json={"role": "analyst"})
    assert last.status_code == 409 and "at least one administrator" in last.json()["error"]["message"]
