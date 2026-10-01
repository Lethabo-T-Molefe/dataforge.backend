"""Application errors. The presentation layer maps each to an HTTP status and a stable error code."""

from typing import Any


class AppError(Exception):
    status_code = 400
    code = "bad_request"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationFailed(AppError):
    status_code = 422
    code = "validation_failed"


class Unauthorized(AppError):
    status_code = 401
    code = "unauthorized"


class Forbidden(AppError):
    status_code = 403
    code = "forbidden"


class NotFound(AppError):
    status_code = 404
    code = "not_found"


class Conflict(AppError):
    status_code = 409
    code = "conflict"


class QuotaExceeded(AppError):
    status_code = 429
    code = "quota_exceeded"


class Refused(AppError):
    """The request is understood but refused on governance grounds (for example, a blocked query)."""

    status_code = 422
    code = "refused"


class Unavailable(AppError):
    status_code = 503
    code = "unavailable"
