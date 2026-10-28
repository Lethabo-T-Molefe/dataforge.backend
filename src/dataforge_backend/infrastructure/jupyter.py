"""JupyterLab, driven from DataForge: status, sign-in links and notebooks written into a member's folder."""

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from ..core.config import settings
from ..core.errors import Unavailable
from ..core.logging import get_logger
from . import http as outbound

log = get_logger(__name__)


@dataclass
class JupyterStatus:
    enabled: bool
    reachable: bool
    url: str | None
    detail: str = ""


def _base() -> str:
    return (settings.JUPYTER_URL or "").rstrip("/")


def public_url() -> str:
    return (settings.JUPYTER_PUBLIC_URL or _base()).rstrip("/")


def member_folder(user_id: str) -> str:
    return f"members/df_{user_id}"


def _headers() -> dict[str, str]:
    return {"Authorization": f"token {settings.JUPYTER_API_TOKEN}"}


async def status() -> JupyterStatus:
    if not _base():
        return JupyterStatus(False, False, None, "Set JUPYTER_URL to enable Jupyter.")
    try:
        response = await outbound.get(f"{_base()}/api/status", timeout=5, headers=_headers())
        return JupyterStatus(True, response.status_code == 200, public_url())
    except httpx.HTTPError:
        return JupyterStatus(True, False, public_url(), "Jupyter is not reachable.")


def lab_url(sso_token: str, path: str | None = None) -> str:
    target = f"/lab/tree/{quote(path)}" if path else "/lab"
    return f"{public_url()}{target}?dataforge_sso={sso_token}"


def to_ipynb(title: str, cells: list[dict[str, Any]]) -> dict[str, Any]:
    """A DataForge notebook (markdown and SQL cells) as a Jupyter notebook that runs through the API."""
    source_cells: list[dict[str, Any]] = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": f"# {title}\n\nOpened from DataForge. Queries run in the "
            "DataForge sandbox with your permissions, so masked columns and row policies apply here too.",
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": "import dataforge as df\n%load_ext dataforge",
        },
    ]
    for cell in cells:
        if cell.get("kind") == "markdown":
            source_cells.append({"cell_type": "markdown", "metadata": {}, "source": cell.get("source", "")})
        else:
            source_cells.append(
                {
                    "cell_type": "code",
                    "metadata": {"dataforge": {"kind": "sql"}},
                    "execution_count": None,
                    "outputs": [],
                    "source": f"%%sql\n{cell.get('source', '').strip()}",
                }
            )
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3 (DataForge)", "language": "python"},
            "language_info": {"name": "python"},
            "dataforge": {"source": "dataforge"},
        },
        "cells": source_cells,
    }


async def write_notebook(user_id: str, filename: str, notebook: dict[str, Any]) -> str:
    """Save a notebook into the member's folder and return its path in the Jupyter workspace."""
    if not _base():
        raise Unavailable("Jupyter is not configured.")
    safe = "".join(ch if ch.isalnum() or ch in " ._-" else "_" for ch in filename).strip() or "notebook"
    if not safe.endswith(".ipynb"):
        safe += ".ipynb"
    folder = member_folder(user_id)
    path = f"{folder}/{safe}"
    try:
        for directory in ("members", folder):
            await outbound.request(
                "PUT",
                f"{_base()}/api/contents/{quote(directory)}",
                timeout=15,
                retryable=True,
                headers=_headers(),
                json={"type": "directory"},
            )
        response = await outbound.request(
            "PUT",
            f"{_base()}/api/contents/{quote(path)}",
            timeout=15,
            retryable=False,
            headers=_headers(),
            content=json.dumps({"type": "notebook", "format": "json", "content": notebook}),
        )
    except httpx.HTTPError as error:
        raise Unavailable("Jupyter is not reachable.") from error
    if response.status_code >= 400:
        log.warning("jupyter_write_failed", status=response.status_code, body=response.text[:300])
        raise Unavailable("Jupyter could not save the notebook.")
    return path
