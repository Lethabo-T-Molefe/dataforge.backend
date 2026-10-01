"""The DataForge API: the composition root that wires routers, middleware and background work together."""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy import text

from .application import integrations, monitors
from .core.config import settings
from .core.database import SessionLocal, engine
from .core.logging import configure_logging, get_logger
from .infrastructure.superset import superset
from .presentation.api import errors
from .presentation.api.deps import limiter
from .presentation.api.v1 import router as v1_router
from .presentation.middleware import RequestContextMiddleware

log = get_logger(__name__)
# An arbitrary constant: the advisory lock that makes one API replica the monitor scheduler.
_SCHEDULER_LOCK = 42_1017


async def _monitor_loop() -> None:
    """Evaluate due monitors on an interval. Only the replica holding the advisory lock does the work."""
    while True:
        try:
            async with engine.connect() as conn:
                locked = (
                    await conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": _SCHEDULER_LOCK})
                ).scalar()
                if locked:
                    try:
                        async with SessionLocal() as session:
                            results = await monitors.evaluate_due(session)
                        if results:
                            log.info(
                                "monitors_evaluated",
                                count=len(results),
                                alerts=sum(1 for r in results if r.get("alert")),
                            )
                    finally:
                        await conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": _SCHEDULER_LOCK})
        except Exception as error:  # noqa: BLE001 - the loop must survive a failed round
            log.warning("monitor_loop_failed", error=str(error))
        await asyncio.sleep(settings.MONITOR_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    task = (
        asyncio.create_task(_monitor_loop())
        if settings.MONITOR_INTERVAL_SECONDS > 0 and settings.ENVIRONMENT != "test"
        else None
    )
    log.info("started", project=settings.PROJECT_NAME, environment=settings.ENVIRONMENT)
    yield
    if task is not None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    # Superset syncs still running were started by actions that are already committed: let them finish.
    await integrations.drain()
    await superset.close()
    await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version="1.0.0",
        description="Governed analytics for everyday members and data scientists: ingestion, guided cleaning, "
        "a semantic layer, analysis, an AI assistant, monitors and explainable predictions.",
        lifespan=lifespan,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_url="/openapi.json" if settings.DEBUG else None,
    )
    app.state.limiter = limiter
    errors.install(app)
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID", "Content-Disposition"],
        max_age=600,
    )
    app.add_middleware(RequestContextMiddleware)
    app.include_router(v1_router, prefix=settings.API_PREFIX)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {"message": f"Welcome to {settings.PROJECT_NAME}"}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("dataforge_backend.main:app", host="127.0.0.1", port=8000, reload=settings.DEBUG)
