"""Identity and access: registration, email verification, sign-in with MFA, sessions and passwords."""

import re
from datetime import timedelta
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..core.crypto import decrypt_secret, encrypt_secret
from ..core.errors import Conflict, Forbidden, NotFound, Unauthorized, ValidationFailed
from ..core.security import (
    create_token,
    decode_token,
    hash_password,
    hash_token,
    new_opaque_token,
    new_totp_secret,
    password_needs_rehash,
    totp_uri,
    utcnow,
    verify_password,
    verify_totp,
)
from ..domain.roles import ROLE_LABELS, ROLE_PERMISSIONS, Role
from ..infrastructure import email as mail
from ..infrastructure.db.models import EmailToken, RefreshToken, User, new_id
from . import audit, quotas
from .common import Actor, initials


def _bump_session(user: User) -> None:
    """Invalidate every outstanding access JWT for this member."""
    user.session_version = int(user.session_version or 0) + 1

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
# Hash of a random value, so a sign-in for an unknown email costs the same time as a real check.
_DUMMY_HASH = hash_password(new_opaque_token())


def normalise_email(email: str) -> str:
    return email.strip().lower()


def check_password_strength(password: str, email: str = "", name: str = "") -> None:
    if len(password) < settings.PASSWORD_MIN_LENGTH:
        raise ValidationFailed(f"Use at least {settings.PASSWORD_MIN_LENGTH} characters.")
    if len(password) > 256:
        raise ValidationFailed("Use at most 256 characters.")
    classes = sum(bool(re.search(p, password)) for p in (r"[a-z]", r"[A-Z]", r"\d", r"[^A-Za-z0-9]"))
    if classes < 3:
        raise ValidationFailed("Mix at least three of: lower case, upper case, digits and symbols.")
    lowered = password.lower()
    for part in [email.split("@")[0], *name.lower().split()]:
        if len(part) >= 4 and part.lower() in lowered:
            raise ValidationFailed("Do not include your name or email in your password.")


async def public_user(session: AsyncSession, user: User) -> dict[str, Any]:
    """The signed-in member, in the shape the frontend's CurrentUser expects."""
    role = Role(user.role)
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": ROLE_LABELS[role],
        "roleId": role.value,
        "team": user.team,
        "workspace": f"{user.name.split()[0]}'s workspace",
        "initials": initials(user.name),
        "verified": user.email_verified_at is not None,
        "mfaEnabled": user.mfa_enabled,
        "permissions": sorted(p.value for p in ROLE_PERMISSIONS[role]),
        "quotas": await quotas.summary(session, user),
    }


async def issue_email_token(session: AsyncSession, user: User, purpose: str, hours: int) -> str:
    token = new_opaque_token()
    session.add(
        EmailToken(
            user_id=user.id,
            purpose=purpose,
            token_hash=hash_token(token),
            expires_at=utcnow() + timedelta(hours=hours),
        )
    )
    return token


async def register(
    session: AsyncSession, name: str, email: str, password: str, ip: str | None = None
) -> User:
    name = " ".join(name.split())
    email = normalise_email(email)
    if not name or len(name) > 200:
        raise ValidationFailed("Enter your name.")
    if not _EMAIL.match(email) or len(email) > 320:
        raise ValidationFailed("Enter a valid email address.")
    check_password_strength(password, email, name)
    existing = (await session.execute(select(User).where(User.email == email))).scalar_one_or_none()
    if existing is not None:
        raise Conflict("An account with this email already exists. Sign in instead.")
    user = User(
        id=new_id(), name=name, email=email, password_hash=hash_password(password), role=Role.ANALYST.value
    )
    session.add(user)
    await session.flush()
    token = await issue_email_token(session, user, "verify", settings.EMAIL_TOKEN_HOURS)
    actor = Actor.from_user(user, ip=ip)
    await audit.record(
        session, actor, "Registered", "user", user.id, user.email, "Account created, email not yet verified"
    )
    await session.commit()
    await mail.send_verification(user.email, user.name, token)
    return user


async def verify_email(session: AsyncSession, token: str) -> User:
    row = (
        await session.execute(
            select(EmailToken).where(
                EmailToken.token_hash == hash_token(token), EmailToken.purpose == "verify"
            )
        )
    ).scalar_one_or_none()
    if row is None or row.used_at is not None or row.expires_at < utcnow():
        raise ValidationFailed("This verification link is not valid or has expired. Ask for a new one.")
    user = await session.get(User, row.user_id)
    if user is None:
        raise NotFound("Account not found.")
    row.used_at = utcnow()
    if user.email_verified_at is None:
        user.email_verified_at = utcnow()
    await audit.record(session, Actor.from_user(user), "Verified email", "user", user.id, user.email)
    await session.commit()
    return user


async def resend_verification(session: AsyncSession, email: str) -> None:
    user = (
        await session.execute(select(User).where(User.email == normalise_email(email)))
    ).scalar_one_or_none()
    # The response is the same whether or not the account exists, so emails cannot be enumerated.
    if user is None or user.email_verified_at is not None:
        return
    token = await issue_email_token(session, user, "verify", settings.EMAIL_TOKEN_HOURS)
    await session.commit()
    await mail.send_verification(user.email, user.name, token)


async def _issue_session(
    session: AsyncSession, user: User, ip: str | None, user_agent: str | None, family_id: str | None = None
) -> dict[str, Any]:
    access = create_token(
        user.id,
        "access",
        expires_in=timedelta(minutes=settings.ACCESS_TOKEN_MINUTES),
        role=user.role,
        extra={"sv": int(user.session_version or 0)},
    )
    refresh = new_opaque_token()
    session.add(
        RefreshToken(
            user_id=user.id,
            token_hash=hash_token(refresh),
            family_id=family_id or new_id(),
            expires_at=utcnow() + timedelta(days=settings.REFRESH_TOKEN_DAYS),
            ip=ip,
            user_agent=(user_agent or "")[:300],
        )
    )
    user.last_active_at = utcnow()
    return {
        "accessToken": access,
        "refreshToken": refresh,
        "tokenType": "bearer",
        "expiresIn": settings.ACCESS_TOKEN_MINUTES * 60,
    }


async def login(
    session: AsyncSession, email: str, password: str, ip: str | None, user_agent: str | None
) -> dict[str, Any]:
    user = (
        await session.execute(select(User).where(User.email == normalise_email(email)))
    ).scalar_one_or_none()
    now = utcnow()
    if user is None or user.password_hash is None:
        verify_password(password, _DUMMY_HASH)
        raise Unauthorized("The email or password is not correct.")
    actor = Actor.from_user(user, ip=ip)
    if user.status != "active" or user.erased_at is not None:
        raise Unauthorized("This account is disabled. Contact an administrator.")
    if user.locked_until and user.locked_until > now:
        minutes = max(1, int((user.locked_until - now).total_seconds() // 60) + 1)
        raise Unauthorized(f"Too many failed attempts. Try again in {minutes} minutes.")
    if not verify_password(password, user.password_hash):
        user.failed_logins += 1
        detail = f"Wrong password (attempt {user.failed_logins})"
        if user.failed_logins >= settings.LOGIN_MAX_ATTEMPTS:
            user.locked_until = now + timedelta(minutes=settings.LOGIN_LOCK_MINUTES)
            user.failed_logins = 0
            detail = f"Account locked for {settings.LOGIN_LOCK_MINUTES} minutes after repeated failures"
        await audit.record(session, actor, "Failed sign-in", "user", user.id, user.email, detail)
        await session.commit()
        raise Unauthorized("The email or password is not correct.")
    user.failed_logins = 0
    user.locked_until = None
    if password_needs_rehash(user.password_hash):
        user.password_hash = hash_password(password)
    if user.mfa_enabled:
        challenge = create_token(user.id, "mfa", expires_in=timedelta(minutes=settings.MFA_CHALLENGE_MINUTES))
        await audit.record(
            session, actor, "Password accepted", "user", user.id, user.email, "Waiting for the second factor"
        )
        await session.commit()
        return {"mfaRequired": True, "challengeToken": challenge}
    # Administrators must enrol MFA before a session is issued.
    if user.role == Role.ADMIN.value:
        secret = new_totp_secret()
        user.mfa_secret_enc = encrypt_secret(secret)
        challenge = create_token(
            user.id, "mfa_enroll", expires_in=timedelta(minutes=settings.MFA_CHALLENGE_MINUTES)
        )
        await audit.record(
            session,
            actor,
            "Password accepted",
            "user",
            user.id,
            user.email,
            "Administrator must enrol MFA before signing in",
        )
        await session.commit()
        return {
            "mfaRequired": False,
            "mfaEnrollmentRequired": True,
            "challengeToken": challenge,
            "secret": secret,
            "otpauthUri": totp_uri(secret, user.email),
        }
    tokens = await _issue_session(session, user, ip, user_agent)
    await audit.record(
        session,
        actor,
        "Signed in",
        "Identity provider",
        user.id,
        "Identity provider",
        "Password only (MFA not set up)",
    )
    await session.commit()
    return {"mfaRequired": False, "mfaEnrollmentRequired": False, **tokens, "user": await public_user(session, user)}


async def complete_mfa(
    session: AsyncSession, challenge: str, code: str, ip: str | None, user_agent: str | None
) -> dict[str, Any]:
    claims = decode_token(challenge, ("mfa", "mfa_enroll"))
    user = await session.get(User, claims.subject)
    if user is None or not user.mfa_secret_enc:
        raise Unauthorized("Sign in again.")
    actor = Actor.from_user(user, ip=ip)
    enrolling = claims.token_type == "mfa_enroll"
    if enrolling and user.mfa_enabled:
        raise Unauthorized("Sign in again.")
    if not enrolling and not user.mfa_enabled:
        raise Unauthorized("Sign in again.")
    if not verify_totp(decrypt_secret(user.mfa_secret_enc), code):
        user.failed_logins += 1
        if user.failed_logins >= settings.LOGIN_MAX_ATTEMPTS:
            user.locked_until = utcnow() + timedelta(minutes=settings.LOGIN_LOCK_MINUTES)
            user.failed_logins = 0
        await audit.record(
            session, actor, "Failed second factor", "user", user.id, user.email, "The session was blocked"
        )
        await session.commit()
        raise Unauthorized("That code is not correct. Sign in again to get a new challenge.")
    user.failed_logins = 0
    if enrolling:
        user.mfa_enabled = True
        await audit.record(session, actor, "Turned on MFA", "user", user.id, user.email, "Required for administrator")
    tokens = await _issue_session(session, user, ip, user_agent)
    await audit.record(
        session,
        actor,
        "Signed in",
        "Identity provider",
        user.id,
        "Identity provider",
        "Multi-factor authentication passed" if not enrolling else "Administrator enrolled MFA and signed in",
    )
    await session.commit()
    return {**tokens, "user": await public_user(session, user)}


async def refresh(
    session: AsyncSession, token: str, ip: str | None, user_agent: str | None
) -> dict[str, Any]:
    """Rotate the refresh token. Reusing a rotated token revokes the whole session family (token theft)."""
    row = (
        await session.execute(select(RefreshToken).where(RefreshToken.token_hash == hash_token(token)))
    ).scalar_one_or_none()
    now = utcnow()
    if row is None:
        raise Unauthorized("Sign in again.")
    if row.revoked_at is not None:
        await session.execute(
            update(RefreshToken)
            .where(RefreshToken.family_id == row.family_id, RefreshToken.revoked_at.is_(None))
            .values(revoked_at=now)
        )
        user = await session.get(User, row.user_id)
        if user is not None:
            _bump_session(user)
            await audit.record(
                session,
                Actor.from_user(user, ip=ip),
                "Revoked sessions",
                "user",
                user.id,
                user.email,
                "A used refresh token was presented again",
            )
        await session.commit()
        raise Unauthorized("Sign in again.")
    if row.expires_at < now:
        raise Unauthorized("Your session has expired. Sign in again.")
    user = await session.get(User, row.user_id)
    if user is None or user.status != "active":
        raise Unauthorized("Sign in again.")
    if user.role == Role.ADMIN.value and not user.mfa_enabled:
        raise Unauthorized("Administrators must enrol multi-factor authentication. Sign in again.")
    tokens = await _issue_session(session, user, ip, user_agent, family_id=row.family_id)
    row.revoked_at = now
    await session.commit()
    return {**tokens, "user": await public_user(session, user)}


async def logout(session: AsyncSession, token: str | None, actor: Actor | None) -> None:
    user_id: str | None = actor.id if actor is not None else None
    if token:
        row = (
            await session.execute(select(RefreshToken).where(RefreshToken.token_hash == hash_token(token)))
        ).scalar_one_or_none()
        if row is not None:
            user_id = row.user_id
            await session.execute(
                update(RefreshToken)
                .where(RefreshToken.family_id == row.family_id, RefreshToken.revoked_at.is_(None))
                .values(revoked_at=utcnow())
            )
    if user_id is not None:
        user = await session.get(User, user_id)
        if user is not None:
            _bump_session(user)
    if actor is not None:
        await audit.record(session, actor, "Signed out", "Identity provider", actor.id, "Identity provider")
    await session.commit()


async def start_mfa_setup(session: AsyncSession, actor: Actor) -> dict[str, str]:
    user = await session.get(User, actor.id)
    if user is None:
        raise NotFound("Account not found.")
    if user.mfa_enabled:
        raise Conflict("Multi-factor authentication is already on.")
    secret = new_totp_secret()
    user.mfa_secret_enc = encrypt_secret(secret)
    await session.commit()
    return {"secret": secret, "otpauthUri": totp_uri(secret, user.email)}


async def enable_mfa(session: AsyncSession, actor: Actor, code: str) -> None:
    user = await session.get(User, actor.id)
    if user is None or not user.mfa_secret_enc:
        raise ValidationFailed("Start the set-up first.")
    if not verify_totp(decrypt_secret(user.mfa_secret_enc), code):
        raise ValidationFailed("That code is not correct. Check your authenticator app's clock.")
    user.mfa_enabled = True
    await audit.record(session, actor, "Turned on MFA", "user", user.id, user.email)
    await session.commit()


async def disable_mfa(session: AsyncSession, actor: Actor, password: str, code: str) -> None:
    user = await session.get(User, actor.id)
    if user is None or not user.mfa_enabled or not user.mfa_secret_enc or not user.password_hash:
        raise ValidationFailed("Multi-factor authentication is not on.")
    if not verify_password(password, user.password_hash) or not verify_totp(
        decrypt_secret(user.mfa_secret_enc), code
    ):
        raise Forbidden("The password or code is not correct.")
    user.mfa_enabled = False
    user.mfa_secret_enc = None
    await audit.record(session, actor, "Turned off MFA", "user", user.id, user.email)
    await session.commit()


async def change_password(session: AsyncSession, actor: Actor, current: str, new: str) -> None:
    user = await session.get(User, actor.id)
    if user is None or not user.password_hash or not verify_password(current, user.password_hash):
        raise Forbidden("Your current password is not correct.")
    check_password_strength(new, user.email, user.name)
    user.password_hash = hash_password(new)
    _bump_session(user)
    await session.execute(
        update(RefreshToken)
        .where(RefreshToken.user_id == user.id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=utcnow())
    )
    await audit.record(
        session, actor, "Changed password", "user", user.id, user.email, "Other sessions were signed out"
    )
    await session.commit()


async def forgot_password(session: AsyncSession, email: str) -> None:
    user = (
        await session.execute(select(User).where(User.email == normalise_email(email)))
    ).scalar_one_or_none()
    if user is None or user.status != "active":
        return
    token = await issue_email_token(session, user, "reset", 1)
    await session.commit()
    await mail.send_password_reset(user.email, user.name, token)


async def reset_password(session: AsyncSession, token: str, password: str) -> None:
    row = (
        await session.execute(
            select(EmailToken).where(
                EmailToken.token_hash == hash_token(token), EmailToken.purpose == "reset"
            )
        )
    ).scalar_one_or_none()
    if row is None or row.used_at is not None or row.expires_at < utcnow():
        raise ValidationFailed("This reset link is not valid or has expired.")
    user = await session.get(User, row.user_id)
    if user is None:
        raise NotFound("Account not found.")
    check_password_strength(password, user.email, user.name)
    user.password_hash = hash_password(password)
    user.failed_logins = 0
    user.locked_until = None
    _bump_session(user)
    row.used_at = utcnow()
    await session.execute(
        update(RefreshToken)
        .where(RefreshToken.user_id == user.id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=utcnow())
    )
    await audit.record(session, Actor.from_user(user), "Reset password", "user", user.id, user.email)
    await session.commit()


async def demo_sign_in(
    session: AsyncSession, role: str, ip: str | None, user_agent: str | None
) -> dict[str, Any]:
    """Prototype convenience: sign in as the seeded demo member for a role. Disabled outside DEMO_MODE."""
    if not settings.DEMO_MODE:
        raise Forbidden("Demo sign-in is turned off.")
    try:
        wanted = Role(role)
    except ValueError as error:
        raise ValidationFailed("Unknown role.") from error
    user = (
        (
            await session.execute(
                select(User)
                .where(
                    User.role == wanted.value, User.email.like("%@dataforge.local"), User.status == "active"
                )
                .order_by(User.created_at)
            )
        )
        .scalars()
        .first()
    )
    if user is None:
        raise NotFound("No demo member for that role. Run: dataforge seed-demo")
    tokens = await _issue_session(session, user, ip, user_agent)
    await audit.record(
        session,
        Actor.from_user(user, ip=ip),
        "Signed in",
        "Identity provider",
        user.id,
        "Identity provider",
        "Demo sign-in",
    )
    await session.commit()
    return {**tokens, "user": await public_user(session, user)}


def notebook_token(user: User) -> tuple[str, int]:
    """A longer-lived token for a Jupyter kernel to call the DataForge API as its member."""
    expires = timedelta(hours=settings.NOTEBOOK_TOKEN_HOURS)
    token = create_token(user.id, "notebook", expires_in=expires, role=user.role)
    return token, int((utcnow() + expires).timestamp())
