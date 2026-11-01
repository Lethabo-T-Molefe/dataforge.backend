"""More identity / admin / health catalogue cases against the live API."""

import re
from datetime import UTC, datetime, timedelta

import pyotp
from httpx import AsyncClient
from sqlalchemy import text

from catalogue_progress import marks
from dataforge_backend.core.database import SessionLocal
from dataforge_backend.infrastructure import email

from .conftest import SignIn

PASSWORD = "Str0ng-Passphrase!"
NEW_PASSWORD = "Newer-Passphrase1!"


async def _register(client: AsyncClient, address: str, name: str = "Kagiso Molefe") -> str:
    response = await client.post(
        "/api/v1/auth/register", json={"name": name, "email": address, "password": PASSWORD}
    )
    assert response.status_code == 201, response.text
    token = re.search(r"token=([\w-]+)", email.sent[-1].get_content())
    assert token is not None
    return token.group(1)


@marks("I-12", "I-13", "I-40", "I-41", "I-43", "I-44", "I-45", "I-47")
async def test_verify_reset_password_and_body_refresh(client: AsyncClient) -> None:
    verify = await _register(client, "reset-me@example.com", name="Lethabo Moyo")
    assert (
        await client.post("/api/v1/auth/reset-password", json={"token": verify, "password": NEW_PASSWORD})
    ).status_code == 422
    assert (await client.post("/api/v1/auth/verify-email", json={"token": verify})).status_code == 200

    email.sent.clear()
    assert (await client.post("/api/v1/auth/forgot-password", json={"email": "reset-me@example.com"})).status_code == 202
    reset = re.search(r"token=([\w-]+)", email.sent[-1].get_content())
    assert reset is not None
    assert (
        await client.post("/api/v1/auth/verify-email", json={"token": reset.group(1)})
    ).status_code == 422
    assert (
        await client.post(
            "/api/v1/auth/reset-password",
            json={"token": reset.group(1), "password": "Lethabo-Pass1!"},
        )
    ).status_code == 422
    assert (
        await client.post(
            "/api/v1/auth/reset-password",
            json={"token": reset.group(1), "password": NEW_PASSWORD},
        )
    ).status_code == 204
    assert (
        await client.post(
            "/api/v1/auth/reset-password",
            json={"token": reset.group(1), "password": NEW_PASSWORD},
        )
    ).status_code == 422

    stale = await _register(client, "stale-verify@example.com")
    async with SessionLocal() as session:
        await session.execute(
            text(
                """
                UPDATE app.email_tokens t SET expires_at = :exp
                FROM app.users u WHERE t.user_id = u.id AND u.email = 'stale-verify@example.com'
                """
            ),
            {"exp": datetime.now(UTC) - timedelta(hours=49)},
        )
        await session.commit()
    assert (await client.post("/api/v1/auth/verify-email", json={"token": stale})).status_code == 422

    login = await client.post(
        "/api/v1/auth/login", json={"email": "reset-me@example.com", "password": NEW_PASSWORD}
    )
    assert login.status_code == 200
    access = login.json()["accessToken"]
    headers = {"Authorization": f"Bearer {access}"}
    assert (
        await client.post(
            "/api/v1/auth/password",
            headers=headers,
            json={"currentPassword": "wrong-password-1", "newPassword": PASSWORD},
        )
    ).status_code == 403
    assert (
        await client.post(
            "/api/v1/auth/password",
            headers=headers,
            json={"currentPassword": NEW_PASSWORD, "newPassword": "short"},
        )
    ).status_code == 422
    same = await client.post(
        "/api/v1/auth/password",
        headers=headers,
        json={"currentPassword": NEW_PASSWORD, "newPassword": NEW_PASSWORD},
    )
    assert same.status_code in (200, 204)

    client.cookies.clear()
    login = await client.post(
        "/api/v1/auth/login", json={"email": "reset-me@example.com", "password": NEW_PASSWORD}
    )
    body_token = client.cookies.get("df_refresh")
    client.cookies.clear()
    refreshed = await client.post("/api/v1/auth/refresh", json={"refreshToken": body_token})
    assert refreshed.status_code == 200 and refreshed.json()["accessToken"]


@marks("I-20", "I-30", "I-31", "I-32", "I-33", "I-39", "I-49", "I-55", "I-56")
async def test_lock_expiry_mfa_tokens_demo_and_expired_refresh(
    client: AsyncClient, sign_in: SignIn
) -> None:
    await _register(client, "unlock@example.com")
    for _ in range(5):
        await client.post("/api/v1/auth/login", json={"email": "unlock@example.com", "password": "Wrong-Password-1"})
    async with SessionLocal() as session:
        await session.execute(
            text(
                "UPDATE app.users SET locked_until = :past WHERE email = 'unlock@example.com'"
            ),
            {"past": datetime.now(UTC) - timedelta(minutes=1)},
        )
        await session.commit()
    unlocked = await client.post(
        "/api/v1/auth/login", json={"email": "unlock@example.com", "password": PASSWORD}
    )
    assert unlocked.status_code == 200 and unlocked.json().get("accessToken")

    await _register(client, "mfa-off@example.com")
    first = (
        await client.post("/api/v1/auth/login", json={"email": "mfa-off@example.com", "password": PASSWORD})
    ).json()
    headers = {"Authorization": f"Bearer {first['accessToken']}"}
    setup = (await client.post("/api/v1/auth/mfa/setup", headers=headers)).json()
    totp = pyotp.TOTP(setup["secret"])
    assert (
        await client.post("/api/v1/auth/mfa/enable", headers=headers, json={"code": totp.now()})
    ).status_code == 204
    challenge = (
        await client.post("/api/v1/auth/login", json={"email": "mfa-off@example.com", "password": PASSWORD})
    ).json()
    assert challenge.get("mfaRequired") is True
    mfa_ok = await client.post(
        "/api/v1/auth/mfa",
        json={"challengeToken": challenge["challengeToken"], "code": totp.now()},
    )
    assert mfa_ok.status_code == 200
    headers = {"Authorization": f"Bearer {mfa_ok.json()['accessToken']}"}
    assert (
        await client.post(
            "/api/v1/auth/mfa/disable",
            headers=headers,
            json={"password": "Wrong-Password-1", "code": totp.now()},
        )
    ).status_code == 403
    assert (
        await client.post(
            "/api/v1/auth/mfa/disable",
            headers=headers,
            json={"password": PASSWORD, "code": "000000"},
        )
    ).status_code == 403
    assert (
        await client.post(
            "/api/v1/auth/mfa/disable",
            headers=headers,
            json={"password": PASSWORD, "code": totp.now()},
        )
    ).status_code == 204
    stale_mfa = await client.post(
        "/api/v1/auth/mfa",
        json={"challengeToken": challenge["challengeToken"], "code": totp.now()},
    )
    assert stale_mfa.status_code == 401

    assert (
        await client.post("/api/v1/auth/mfa", json={"challengeToken": first["accessToken"], "code": "123456"})
    ).status_code == 401
    assert (
        await client.get(
            "/api/v1/datasets", headers={"Authorization": f"Bearer {challenge['challengeToken']}"}
        )
    ).status_code == 401

    bad = await client.get("/api/v1/auth/me", headers={"Authorization": "Bearer not-a.jwt.token"})
    assert bad.status_code == 401

    login = await client.post(
        "/api/v1/auth/login", json={"email": "unlock@example.com", "password": PASSWORD}
    )
    access = login.json()["accessToken"]
    async with SessionLocal() as session:
        await session.execute(
            text("UPDATE app.users SET session_version = session_version + 1 WHERE email = 'unlock@example.com'")
        )
        await session.commit()
    assert (
        await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {access}"})
    ).status_code == 401

    client.cookies.clear()
    await client.post("/api/v1/auth/login", json={"email": "unlock@example.com", "password": PASSWORD})
    refresh = client.cookies["df_refresh"]
    async with SessionLocal() as session:
        await session.execute(
            text(
                """
                UPDATE app.refresh_tokens SET expires_at = :past
                WHERE revoked_at IS NULL AND user_id = (
                    SELECT id FROM app.users WHERE email = 'unlock@example.com'
                )
                """
            ),
            {"past": datetime.now(UTC) - timedelta(days=1)},
        )
        await session.commit()
    client.cookies.clear()
    expired = await client.post("/api/v1/auth/refresh", json={"refreshToken": refresh})
    assert expired.status_code == 401
    assert "expired" in expired.json()["error"]["message"].lower() or "sign in" in expired.json()["error"]["message"].lower()

    nope = await client.post("/api/v1/auth/demo", json={"role": "nope"})
    assert nope.status_code == 422
    demo = await sign_in("admin")
    me = (await client.get("/api/v1/auth/me", headers=demo)).json()
    assert me["roleId"] == "admin" and me.get("mfaEnabled") in (True, False)


@marks("A-02", "A-03", "A-06", "A-07", "A-13", "P-17")
async def test_invite_team_policy_delete_and_health(client: AsyncClient, sign_in: SignIn) -> None:
    health = await client.get("/api/v1/health")
    assert health.status_code == 200
    body = health.json()
    assert "status" in body and "ready" in body

    admin = await sign_in("admin")
    existing = await client.post(
        "/api/v1/members",
        headers=admin,
        json={"email": "thandi@dataforge.local", "role": "analyst", "name": "Thandi"},
    )
    assert existing.status_code == 409
    bad_role = await client.post(
        "/api/v1/members",
        headers=admin,
        json={"email": "newbie@example.com", "role": "wizard", "name": "New"},
    )
    assert bad_role.status_code == 422
    invited = await client.post(
        "/api/v1/members",
        headers=admin,
        json={"email": "new-admin@example.com", "role": "admin", "name": "New Admin"},
    )
    assert invited.status_code in (200, 201)

    members = (await client.get("/api/v1/members", headers=admin)).json()
    target = next(m for m in members if m["email"] == "thandi@dataforge.local")
    bad_status = await client.patch(
        f"/api/v1/members/{target['id']}", headers=admin, json={"status": "banished"}
    )
    assert bad_status.status_code == 422
    team = await client.patch(
        f"/api/v1/members/{target['id']}", headers=admin, json={"team": "   "}
    )
    assert team.status_code == 200
    assert next(m for m in (await client.get("/api/v1/members", headers=admin)).json() if m["id"] == target["id"])["team"] in (None, "")

    missing = await client.delete("/api/v1/policies/00000000-0000-0000-0000-000000000000", headers=admin)
    assert missing.status_code == 404
