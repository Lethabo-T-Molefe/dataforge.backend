"""Shared outbound HTTP: timeouts and bounded retries for transient failures."""

from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx

from ..core.logging import get_logger

log = get_logger(__name__)

_RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_DEFAULT_TIMEOUT = httpx.Timeout(20.0, connect=5.0)


def _backoff(attempt: int) -> float:
    """Exponential backoff with jitter, capped so a burst of retries stays under a few seconds."""
    return min(4.0, (0.25 * (2**attempt)) + random.uniform(0, 0.15))


async def request(
    method: str,
    url: str,
    *,
    timeout: float | httpx.Timeout = _DEFAULT_TIMEOUT,
    retries: int = 2,
    retryable: bool = True,
    **kwargs: Any,
) -> httpx.Response:
    """HTTP request with optional retries on connection errors and retryable status codes.

    Set retryable=False for non-idempotent POSTs that must not run twice.
    """
    attempts = 1 + max(0, retries if retryable else 0)
    last_error: Exception | None = None
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(attempts):
            try:
                response = await client.request(method, url, **kwargs)
            except httpx.HTTPError as error:
                last_error = error
                if attempt + 1 >= attempts:
                    raise
                await asyncio.sleep(_backoff(attempt))
                continue
            if response.status_code in _RETRY_STATUS and attempt + 1 < attempts and retryable:
                log.info(
                    "http_retry",
                    method=method,
                    url=url.split("?", 1)[0],
                    status=response.status_code,
                    attempt=attempt + 1,
                )
                await asyncio.sleep(_backoff(attempt))
                continue
            return response
    assert last_error is not None
    raise last_error


async def get(url: str, **kwargs: Any) -> httpx.Response:
    return await request("GET", url, **kwargs)


async def post(url: str, *, retryable: bool = False, **kwargs: Any) -> httpx.Response:
    return await request("POST", url, retryable=retryable, **kwargs)


def get_sync(url: str, *, timeout: float = 120.0, retries: int = 2, **kwargs: Any) -> httpx.Response:
    """Synchronous GET with the same retry policy (used by the Cloudinary file store)."""
    attempts = 1 + max(0, retries)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = httpx.get(url, timeout=timeout, **kwargs)
        except httpx.HTTPError as error:
            last_error = error
            if attempt + 1 >= attempts:
                raise
            continue
        if response.status_code in _RETRY_STATUS and attempt + 1 < attempts:
            continue
        return response
    assert last_error is not None
    raise last_error
