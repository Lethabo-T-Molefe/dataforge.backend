"""Password hashing, tokens and one-time codes."""

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import jwt
import pyotp
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from .config import settings
from .errors import Unauthorized

# Argon2id is the modern adaptive algorithm recommended for password storage.
_hasher = PasswordHasher()

TokenType = Literal["access", "mfa", "embed", "notebook"]


def utcnow() -> datetime:
    return datetime.now(UTC)


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def password_needs_rehash(password_hash: str) -> bool:
    return _hasher.check_needs_rehash(password_hash)


def new_opaque_token() -> str:
    """A random token for refresh tokens and email links. Only its hash is stored."""
    return secrets.token_urlsafe(40)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TokenClaims:
    subject: str
    token_type: TokenType
    role: str | None
    expires_at: datetime
    extra: dict[str, Any]


def create_token(
    subject: str,
    token_type: TokenType,
    *,
    expires_in: timedelta,
    role: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    now = utcnow()
    payload: dict[str, Any] = {
        "sub": subject,
        "typ": token_type,
        "iss": settings.JWT_ISSUER,
        "iat": int(now.timestamp()),
        "exp": int((now + expires_in).timestamp()),
        "jti": uuid.uuid4().hex,
    }
    if role:
        payload["role"] = role
    if extra:
        payload["ext"] = extra
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str, expected_type: TokenType | tuple[TokenType, ...]) -> TokenClaims:
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
            issuer=settings.JWT_ISSUER,
            options={"require": ["exp", "sub", "typ", "iss"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise Unauthorized("Your session has expired. Sign in again.") from exc
    except jwt.PyJWTError as exc:
        raise Unauthorized("The token is not valid.") from exc
    accepted = expected_type if isinstance(expected_type, tuple) else (expected_type,)
    if payload.get("typ") not in accepted:
        raise Unauthorized("The token is not valid for this use.")
    return TokenClaims(
        subject=str(payload["sub"]),
        token_type=payload["typ"],
        role=payload.get("role"),
        expires_at=datetime.fromtimestamp(int(payload["exp"]), UTC),
        extra=dict(payload.get("ext") or {}),
    )


def new_totp_secret() -> str:
    return str(pyotp.random_base32())


def totp_uri(secret: str, email: str) -> str:
    return str(pyotp.TOTP(secret).provisioning_uri(name=email, issuer_name="DataForge"))


def verify_totp(secret: str, code: str) -> bool:
    cleaned = "".join(ch for ch in code if ch.isdigit())
    if len(cleaned) != 6:
        return False
    # One step of clock drift either way is accepted.
    return bool(pyotp.TOTP(secret).verify(cleaned, valid_window=1))


def create_sso_token(
    subject: str, audience: str, *, role: str, name: str, email: str, extra: dict[str, Any] | None = None
) -> str:
    """A two-minute, single-use sign-in token for Superset or Jupyter, signed with the separate SSO secret."""
    now = utcnow()
    payload: dict[str, Any] = {
        "sub": subject,
        "aud": audience,
        "iss": settings.JWT_ISSUER,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=settings.SSO_TOKEN_SECONDS)).timestamp()),
        "jti": uuid.uuid4().hex,
        "role": role,
        "name": name,
        "email": email,
        **(extra or {}),
    }
    return jwt.encode(payload, settings.SSO_SHARED_SECRET, algorithm="HS256")
