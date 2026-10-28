"""Chat notifications: alerts posted to Slack through an incoming webhook, and Slack request verification."""

import hashlib
import hmac
import time
from typing import Any

import httpx

from ..core.config import settings
from ..core.logging import get_logger
from . import http as outbound

log = get_logger(__name__)


async def post_slack(message: str, blocks: list[dict[str, Any]] | None = None) -> bool:
    if not settings.SLACK_WEBHOOK_URL:
        return False
    payload: dict[str, Any] = {"text": message}
    if blocks:
        payload["blocks"] = blocks
    try:
        response = await outbound.post(
            settings.SLACK_WEBHOOK_URL, timeout=10, retryable=True, json=payload
        )
        return response.status_code < 300
    except httpx.HTTPError as error:
        log.warning("slack_post_failed", error=str(error))
        return False


def verify_slack_signature(body: bytes, timestamp: str | None, signature: str | None) -> bool:
    """Slack signs requests with HMAC-SHA256 of 'v0:{timestamp}:{body}'. Reject stale or unsigned requests."""
    if not settings.SLACK_SIGNING_SECRET or not timestamp or not signature:
        return False
    try:
        if abs(time.time() - int(timestamp)) > 300:
            return False
    except ValueError:
        return False
    base = f"v0:{timestamp}:".encode() + body
    expected = "v0=" + hmac.new(settings.SLACK_SIGNING_SECRET.encode(), base, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)
