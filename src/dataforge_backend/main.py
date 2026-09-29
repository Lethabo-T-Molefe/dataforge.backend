from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .core.config import settings
from .core.database import Base, engine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)  # For dev only
    print(f"{settings.PROJECT_NAME} started successfully")
    yield
    # Shutdown
    await engine.dispose()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
)

# Include routers later
# from .presentation.api.v1 import user
# app.include_router(user.router, prefix="/api/v1")


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": f"Welcome to {settings.PROJECT_NAME}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.dataforge_backend.main:app", host="0.0.0.0", port=8000, reload=True)
