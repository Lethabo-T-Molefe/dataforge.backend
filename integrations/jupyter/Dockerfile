# JupyterLab for DataForge notebooks: single sign-on, per-member folders and kernels, DataForge theme.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 JUPYTER_WORKSPACE=/workspace JUPYTER_HOST=0.0.0.0 JUPYTER_ALLOW_REMOTE=true

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

WORKDIR /app
COPY requirements.txt ./
RUN uv pip install --system --no-cache -r requirements.txt

COPY jupyter_server_config.py run.py ./
COPY dataforge_jupyter ./dataforge_jupyter
COPY dataforge ./dataforge

RUN useradd --create-home --uid 10003 jupyter && mkdir -p /workspace /app/config && chown -R jupyter /workspace /app/config /usr/local/share/jupyter
USER jupyter

EXPOSE 8888
CMD ["python", "run.py"]
