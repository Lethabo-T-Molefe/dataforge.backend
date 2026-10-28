"""Outgoing email: through Resend when RESEND_API_KEY is set, otherwise SMTP.

Without either, messages are written to the log (development only).
"""

import asyncio
import smtplib
from email.message import EmailMessage

import httpx

from ..core.config import settings
from ..core.errors import Unavailable
from ..core.logging import get_logger
from . import http as outbound

log = get_logger(__name__)

# Tests replace this to capture messages.
sent: list[EmailMessage] = []


def _send_sync(message: EmailMessage) -> None:
    assert settings.SMTP_HOST
    with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=20) as client:
        if settings.SMTP_STARTTLS:
            client.starttls()
        if settings.SMTP_USER and settings.SMTP_PASSWORD:
            client.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
        client.send_message(message)


# Top-level domains reserved for local networks, testing and examples (RFC 2606, RFC 6762).
_RESERVED_TLDS = {"local", "test", "example", "invalid", "localhost"}


async def _send_resend(to: str, subject: str, body: str, reply_to: str | None) -> None:
    """Send through Resend's HTTP API, which works where outbound mail ports are blocked."""
    payload: dict[str, object] = {"from": settings.SMTP_FROM, "to": [to], "subject": subject, "text": body}
    if reply_to:
        payload["reply_to"] = reply_to
    try:
        response = await outbound.post(
            "https://api.resend.com/emails",
            timeout=20,
            retryable=True,
            headers={"Authorization": f"Bearer {settings.RESEND_API_KEY}"},
            json=payload,
        )
    except httpx.HTTPError as error:
        log.error("email_not_sent", to=to, subject=subject, error=str(error))
        raise Unavailable("The email could not be sent. Try again shortly.") from error
    if response.status_code >= 300:
        log.error(
            "email_not_sent", to=to, subject=subject, status=response.status_code, body=response.text[:300]
        )
        raise Unavailable("The email could not be sent. Try again shortly.")


async def send_email(to: str, subject: str, body: str, *, reply_to: str | None = None) -> None:
    message = EmailMessage()
    message["From"] = settings.SMTP_FROM
    message["To"] = to
    message["Subject"] = subject
    if reply_to:
        message["Reply-To"] = reply_to
    message.set_content(body)
    sent.append(message)
    del sent[:-50]
    if to.rsplit(".", 1)[-1].lower() in _RESERVED_TLDS:
        # Demo members live at addresses that can never receive mail; sending to them only earns bounces.
        log.info("email_skipped_undeliverable", to=to, subject=subject)
        return
    if settings.RESEND_API_KEY:
        await _send_resend(to, subject, body, reply_to)
        return
    if not settings.SMTP_HOST:
        if settings.ENVIRONMENT == "production":
            log.error("email_not_sent_no_smtp", to=to, subject=subject)
        else:
            log.info("email_outbox", to=to, subject=subject, body=body)
        return
    await asyncio.to_thread(_send_sync, message)


async def send_verification(to: str, name: str, token: str) -> None:
    link = f"{settings.FRONTEND_URL}/verify?token={token}"
    await send_email(
        to,
        "Verify your DataForge email",
        f"Hello {name},\n\nConfirm your email address to start analysing data:\n{link}\n\n"
        f"The link expires in {settings.EMAIL_TOKEN_HOURS} hours. If you did not register, ignore this email.\n",
    )


async def send_password_reset(to: str, name: str, token: str) -> None:
    link = f"{settings.FRONTEND_URL}/reset-password?token={token}"
    await send_email(
        to,
        "Reset your DataForge password",
        f"Hello {name},\n\nChoose a new password here:\n{link}\n\nIf you did not ask for this, ignore this email.\n",
    )


async def send_contact_message(name: str, sender: str, organisation: str, body: str) -> None:
    """A message from the public contact form, delivered to the platform contact with the sender as reply-to."""
    # Header values are single lines: whatever was typed, the name and organisation cannot add headers.
    who = " ".join(name.split())
    where = " ".join(organisation.split())
    await send_email(
        settings.CONTACT_EMAIL,
        f"DataForge contact form: {who}"[:150],
        f"From: {who} <{sender}>\nOrganisation: {where or 'not given'}\n\n{body.strip()}\n",
        reply_to=sender,
    )
