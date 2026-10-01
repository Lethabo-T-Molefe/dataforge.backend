"""One error shape for every failure: {"error": {"code", "message", "details"}}."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

from ...core.errors import AppError
from ...core.logging import get_logger

log = get_logger(__name__)


def _body(
    code: str, message: str, request: Request, details: dict[str, object] | None = None
) -> dict[str, object]:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
            "requestId": getattr(request.state, "request_id", None),
        }
    }


def install(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error(request: Request, error: AppError) -> JSONResponse:
        return JSONResponse(
            _body(error.code, error.message, request, error.details), status_code=error.status_code
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        fields = []
        for issue in error.errors():
            location = ".".join(str(p) for p in issue.get("loc", []) if p != "body")
            fields.append({"field": location, "message": issue.get("msg", "Invalid value")})
        first = fields[0] if fields else {"field": "", "message": "Invalid request"}
        message = f"{first['field']}: {first['message']}" if first["field"] else first["message"]
        return JSONResponse(_body("validation_failed", message, request, {"fields": fields}), status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, error: StarletteHTTPException) -> JSONResponse:
        codes = {404: "not_found", 405: "method_not_allowed", 401: "unauthorized", 403: "forbidden"}
        return JSONResponse(
            _body(codes.get(error.status_code, "http_error"), str(error.detail), request),
            status_code=error.status_code,
        )

    @app.exception_handler(RateLimitExceeded)
    async def rate_limited(request: Request, error: RateLimitExceeded) -> JSONResponse:
        return JSONResponse(
            _body("rate_limited", "Too many requests. Wait a moment and try again.", request), status_code=429
        )

    @app.exception_handler(Exception)
    async def unexpected(request: Request, error: Exception) -> JSONResponse:
        log.exception("unhandled_error", path=request.url.path, error=str(error))
        return JSONResponse(
            _body("internal_error", "Something went wrong on our side. The error was logged.", request),
            status_code=500,
        )
