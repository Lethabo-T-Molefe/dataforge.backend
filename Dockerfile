# DataForge API
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY alembic.ini ./
COPY alembic ./alembic
COPY src ./src
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 dataforge && mkdir -p /app/var/landing && chown -R dataforge /app/var
USER dataforge

# Production refuses to start with DEBUG or DEMO_MODE on, so the image turns both off.
# WEB_CONCURRENCY is the number of API processes; each keeps its own database pool and sandbox workers.
ENV PATH="/app/.venv/bin:$PATH" \
    ENVIRONMENT=production \
    DEBUG=false \
    DEMO_MODE=false \
    WEB_CONCURRENCY=2 \
    STORAGE_DIR=/app/var/landing

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/api/v1/health').status == 200 else 1)"
# Apply migrations before serving so schema changes ship with the image.
# PORT is set by the host (Render sets it); uvicorn reads WEB_CONCURRENCY for its worker count.
CMD ["sh", "-c", "dataforge migrate && exec uvicorn dataforge_backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers"]
