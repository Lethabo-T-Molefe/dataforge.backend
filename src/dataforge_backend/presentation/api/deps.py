"""Request dependencies: the database session, the acting member, and rate limiting."""

from collections.abc import AsyncIterator

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from slowapi import Limiter
from sqlalchemy.ext.asyncio import AsyncSession

from ...application.common import Actor
from ...core.config import settings
from ...core.database import SessionLocal
from ...core.errors import Unauthorized
from ...core.security import TokenType, decode_token, utcnow
from ...infrastructure.db.models import User


def client_ip(request: Request) -> str | None:
    """The real client address behind Render / Cloudflare.

    SlowAPI and audit logs must key on this, not request.client.host (the edge proxy).
    """
    cf = (request.headers.get("cf-connecting-ip") or "").strip()
    if cf:
        return cf
    forwarded = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    if forwarded:
        return forwarded
    return request.client.host if request.client else None


def _rate_limit_key(request: Request) -> str:
    return client_ip(request) or "unknown"


limiter = Limiter(
    key_func=_rate_limit_key,
    default_limits=[settings.RATE_LIMIT_DEFAULT],
    storage_uri=settings.REDIS_URL or "memory://",
    enabled=settings.ENVIRONMENT != "test",
)
_bearer = HTTPBearer(auto_error=False)


async def get_db() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session


async def _resolve(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None,
    session: AsyncSession,
    accepted: tuple[TokenType, ...],
) -> Actor:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise Unauthorized("Sign in to continue.")
    claims = decode_token(credentials.credentials, accepted)
    user = await session.get(User, claims.subject)
    if user is None or user.status != "active" or user.erased_at is not None:
        raise Unauthorized("Your account is not active.")
    # Access JWTs carry session_version; logout / password change bump it so stolen tokens die immediately.
    if claims.token_type == "access":
        token_sv = int(claims.extra.get("sv", 0))
        if token_sv != int(user.session_version or 0):
            raise Unauthorized("Sign in again.")
    # Refresh "last active" at most once a minute.
    if user.last_active_at is None or (utcnow() - user.last_active_at).total_seconds() > 60:
        user.last_active_at = utcnow()
        await session.commit()
    return Actor.from_user(
        user,
        token_type=claims.token_type,
        request_id=getattr(request.state, "request_id", None),
        ip=client_ip(request),
    )


async def current_actor(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    session: AsyncSession = Depends(get_db),
) -> Actor:
    """The member behind a browser session (access tokens only)."""
    return await _resolve(request, credentials, session, ("access",))


async def data_actor(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    session: AsyncSession = Depends(get_db),
) -> Actor:
    """Read-only data endpoints also accept a Jupyter kernel's notebook token."""
    return await _resolve(request, credentials, session, ("access", "notebook"))


async def optional_actor(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    session: AsyncSession = Depends(get_db),
) -> Actor | None:
    if credentials is None:
        return None
    try:
        return await _resolve(request, credentials, session, ("access",))
    except Unauthorized:
        return None
