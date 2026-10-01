# Apache Superset with DataForge's configuration, theme and single sign-on.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 SUPERSET_HOME=/app/superset_home SUPERSET_CONFIG_PATH=/app/superset_config.py

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

WORKDIR /app
COPY requirements.txt constraints-6.1.0.txt ./
RUN uv pip install --system --no-cache -r requirements.txt -c constraints-6.1.0.txt gunicorn gevent

COPY superset_config.py bootstrap.py run.py ./
COPY dataforge_superset ./dataforge_superset

RUN useradd --create-home --uid 10002 superset && mkdir -p /app/superset_home && chown -R superset /app/superset_home
USER superset

EXPOSE 8088
CMD ["sh", "-c", "python bootstrap.py && gunicorn -w 4 -k gevent --timeout 120 -b 0.0.0.0:8088 'superset.app:create_app()'"]
