"""Request ids, security headers, body-size limits, metrics and structured access logs."""

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ..core.config import settings
from ..core.logging import get_logger
from .metrics import observe_request

log = get_logger("dataforge.access")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming = request.headers.get("X-Request-ID", "")
        request_id = (
            incoming if 8 <= len(incoming) <= 64 and incoming.replace("-", "").isalnum() else uuid.uuid4().hex
        )
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        response: Response | None = None
        try:
            limited = _reject_oversized_body(request)
            if limited is not None:
                response = limited
            else:
                response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("request_id")
        assert response is not None
        duration = round((time.perf_counter() - started) * 1000, 1)
        observe_request(request.method, request.url.path, response.status_code, duration)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        if not request.url.path.startswith(("/docs", "/redoc")):
            response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        response.headers.setdefault("Cache-Control", "no-store")
        log.info(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            ms=duration,
            request_id=request_id,
        )
        return response


def _reject_oversized_body(request: Request) -> JSONResponse | None:
    """Refuse non-multipart bodies larger than MAX_JSON_BODY_MB before they are parsed."""
    if request.method in {"GET", "HEAD", "OPTIONS", "TRACE"}:
        return None
    content_type = (request.headers.get("content-type") or "").lower()
    if "multipart/form-data" in content_type:
        return None
    length = request.headers.get("content-length")
    if not length:
        return None
    try:
        size = int(length)
    except ValueError:
        return None
    limit = int(settings.MAX_JSON_BODY_MB * 1024 * 1024)
    if size <= limit:
        return None
    return JSONResponse(
        status_code=413,
        content={
            "error": {
                "code": "payload_too_large",
                "message": f"Request body must be at most {settings.MAX_JSON_BODY_MB:g} MB.",
            }
        },
    )
