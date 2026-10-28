from typing import Any

from fastapi import APIRouter, Depends, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from ....application import identity, slack
from ....application.common import Actor
from ....application import rate_limit as durable_limits
from ....core.config import settings
from ....core.errors import NotFound, Unauthorized
from ....infrastructure.db.models import User
from ..deps import client_ip, current_actor, data_actor, get_db, limiter, optional_actor
from ..schemas import (
    ChangePasswordIn,
    CodeIn,
    DemoIn,
    DisableMfaIn,
    EmailIn,
    LoginIn,
    LogoutIn,
    MfaIn,
    RefreshIn,
    RegisterIn,
    ResetIn,
    TokenIn,
)

router = APIRouter(prefix="/auth", tags=["Identity"])
_COOKIE_PATH = f"{settings.API_PREFIX}/auth"


def _agent(request: Request) -> str | None:
    return request.headers.get("User-Agent")


def _hand_over(response: Response, result: dict[str, Any]) -> dict[str, Any]:
    """Move a new session's refresh token out of the body and into an HttpOnly cookie."""
    token = result.pop("refreshToken", None)
    if token:
        response.set_cookie(
            settings.REFRESH_COOKIE_NAME,
            token,
            max_age=settings.REFRESH_TOKEN_DAYS * 86_400,
            path=_COOKIE_PATH,
            httponly=True,
            secure=settings.refresh_cookie_secure,
            samesite=settings.REFRESH_COOKIE_SAMESITE,
        )
    return result


def _presented(request: Request, body_token: str | None) -> str | None:
    return request.cookies.get(settings.REFRESH_COOKIE_NAME) or body_token


@router.post("/register", status_code=status.HTTP_201_CREATED)
@limiter.limit("5/minute")
async def register(
    request: Request, body: RegisterIn, session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Create an account. A verification email is sent; analysis unlocks once the email is verified."""
    register_limit = 5 if settings.ENVIRONMENT == "production" else 120
    await durable_limits.consume(
        session,
        f"auth:register:{client_ip(request) or 'unknown'}",
        limit=register_limit,
        window_seconds=60,
    )
    user = await identity.register(session, body.name, str(body.email), body.password, client_ip(request))
    return {"id": user.id, "email": user.email, "verificationSent": True}


@router.post("/verify-email")
@limiter.limit("20/minute")
async def verify_email(
    request: Request, body: TokenIn, session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    user = await identity.verify_email(session, body.token)
    return {"verified": True, "email": user.email}


@router.post("/resend-verification", status_code=status.HTTP_202_ACCEPTED)
@limiter.limit("3/minute")
async def resend_verification(
    request: Request, body: EmailIn, session: AsyncSession = Depends(get_db)
) -> dict[str, bool]:
    await durable_limits.consume(
        session, f"auth:resend:{client_ip(request) or 'unknown'}", limit=3, window_seconds=60
    )
    await identity.resend_verification(session, str(body.email))
    return {"accepted": True}


@router.post("/login")
@limiter.limit("10/minute")
async def login(
    request: Request, response: Response, body: LoginIn, session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Password sign-in. With MFA on, returns a challenge token for /auth/mfa instead of a session.

    A session is the access token in the body and the refresh token in an HttpOnly cookie.
    """
    # Production keeps a firm 10/min; local and test runs open the window so Playwright evidence suites can finish.
    login_limit = 10 if settings.ENVIRONMENT == "production" else 120
    await durable_limits.consume(
        session, f"auth:login:{client_ip(request) or 'unknown'}", limit=login_limit, window_seconds=60
    )
    result = await identity.login(
        session, str(body.email), body.password, client_ip(request), _agent(request)
    )
    return _hand_over(response, result)


@router.post("/mfa")
@limiter.limit("10/minute")
async def complete_mfa(
    request: Request, response: Response, body: MfaIn, session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    await durable_limits.consume(
        session, f"auth:mfa:{client_ip(request) or 'unknown'}", limit=10, window_seconds=60
    )
    result = await identity.complete_mfa(
        session, body.challenge_token, body.code, client_ip(request), _agent(request)
    )
    return _hand_over(response, result)


@router.post("/refresh")
@limiter.limit("30/minute")
async def refresh(
    request: Request,
    response: Response,
    body: RefreshIn | None = None,
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """A new access token. The refresh token in the cookie is rotated; reusing an old one ends the session."""
    await durable_limits.consume(
        session, f"auth:refresh:{client_ip(request) or 'unknown'}", limit=30, window_seconds=60
    )
    token = _presented(request, body.refresh_token if body else None)
    if not token:
        raise Unauthorized("Sign in again.")
    result = await identity.refresh(session, token, client_ip(request), _agent(request))
    return _hand_over(response, result)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    request: Request,
    response: Response,
    body: LogoutIn | None = None,
    actor: Actor | None = Depends(optional_actor),
    session: AsyncSession = Depends(get_db),
) -> None:
    await identity.logout(session, _presented(request, body.refresh_token if body else None), actor)
    response.delete_cookie(
        settings.REFRESH_COOKIE_NAME,
        path=_COOKIE_PATH,
        httponly=True,
        secure=settings.refresh_cookie_secure,
        samesite=settings.REFRESH_COOKIE_SAMESITE,
    )


@router.get("/me")
async def me(actor: Actor = Depends(data_actor), session: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    user = await session.get(User, actor.id)
    if user is None:
        raise NotFound("Account not found.")
    result = await identity.public_user(session, user)
    await session.commit()
    return result


@router.post("/mfa/setup")
async def start_mfa(
    actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, str]:
    """A new authenticator secret. MFA turns on only after a code from it is confirmed."""
    return await identity.start_mfa_setup(session, actor)


@router.post("/mfa/enable", status_code=status.HTTP_204_NO_CONTENT)
async def enable_mfa(
    body: CodeIn, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> None:
    await identity.enable_mfa(session, actor, body.code)


@router.post("/mfa/disable", status_code=status.HTTP_204_NO_CONTENT)
async def disable_mfa(
    body: DisableMfaIn, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> None:
    await identity.disable_mfa(session, actor, body.password, body.code)


@router.post("/password", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(
    body: ChangePasswordIn, actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> None:
    await identity.change_password(session, actor, body.current_password, body.new_password)


@router.post("/forgot-password", status_code=status.HTTP_202_ACCEPTED)
@limiter.limit("3/minute")
async def forgot_password(
    request: Request, body: EmailIn, session: AsyncSession = Depends(get_db)
) -> dict[str, bool]:
    await durable_limits.consume(
        session, f"auth:forgot:{client_ip(request) or 'unknown'}", limit=3, window_seconds=60
    )
    await identity.forgot_password(session, str(body.email))
    return {"accepted": True}


@router.post("/reset-password", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit("10/minute")
async def reset_password(request: Request, body: ResetIn, session: AsyncSession = Depends(get_db)) -> None:
    await durable_limits.consume(
        session, f"auth:reset:{client_ip(request) or 'unknown'}", limit=10, window_seconds=60
    )
    await identity.reset_password(session, body.token, body.password)


@router.get("/demo")
async def demo_available() -> dict[str, bool]:
    return {"enabled": settings.DEMO_MODE}


@router.post("/demo")
@limiter.limit("30/minute")
async def demo(
    request: Request, response: Response, body: DemoIn, session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Prototype only: sign in as the seeded member for a role. Off unless DEMO_MODE is set."""
    result = await identity.demo_sign_in(session, body.role, client_ip(request), _agent(request))
    return _hand_over(response, result)


@router.post("/slack-link")
async def slack_link(
    actor: Actor = Depends(current_actor), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """A one-time code that links a Slack account to this member."""
    return await slack.create_link_code(session, actor)
