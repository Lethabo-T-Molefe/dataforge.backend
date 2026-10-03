"""Superset and Jupyter, governed by DataForge.

Superset receives only approved, organisation-wide datasets, each with the semantic layer's shared metrics
and the row policies of every role. Members open Superset and Jupyter through short-lived sign-in links;
Jupyter kernels query through the DataForge API with the member's own token.
"""

import asyncio
import hmac
import re
from collections.abc import Coroutine
from typing import Any
from urllib.parse import quote

from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..core.database import SessionLocal
from ..core.errors import Forbidden, NotFound, Unavailable
from ..core.logging import get_logger
from ..core.security import create_sso_token
from ..domain.roles import Permission
from ..domain.semantic import MetricError, expression_for, validate_metric
from ..infrastructure import jupyter
from ..infrastructure.db.models import Dataset, Notebook, RowPolicy, User
from ..infrastructure.superset import superset
from . import audit, catalogue
from .common import Actor, is_queryable
from .identity import notebook_token
from .workbench import get_notebook

log = get_logger(__name__)
SUPERSET_ROLES = {"admin", "steward", "practitioner", "curator", "analyst", "executive"}
JUPYTER_ROLES = {"admin", "steward", "practitioner", "curator"}
_D3_FORMATS = {"money": "R,.0f", "rate": ".1%", "decimal": ",.2f", "count": ",.0f"}
_bi_role_ready = False
# Syncs that are still running. The set keeps them from being garbage-collected before they finish.
_background: set[asyncio.Task[None]] = set()
# One sync at a time, so two governance actions cannot interleave their calls to Superset.
_sync_lock = asyncio.Lock()


def in_background(work: Coroutine[Any, Any, None]) -> None:
    """Run a Superset sync after the response has been sent.

    A sync makes many calls to Superset and can take tens of seconds. The governance action it follows
    (an approval, a cleaning run, publishing a metric) is already committed, so the member is not kept waiting.
    """
    task = asyncio.get_running_loop().create_task(work)
    _background.add(task)
    task.add_done_callback(_background.discard)


async def drain(timeout: float = 20.0) -> None:
    """Let running syncs finish before the service stops."""
    if _background:
        await asyncio.wait(set(_background), timeout=timeout)


async def status() -> dict[str, Any]:
    s, j = await superset.status(), await jupyter.status()
    return {
        "superset": {"enabled": s.enabled, "reachable": s.reachable, "url": s.url, "detail": s.detail},
        "jupyter": {"enabled": j.enabled, "reachable": j.reachable, "url": j.url, "detail": j.detail},
    }


# ------------------------------------------------------------------ Superset


async def ensure_bi_role(session: AsyncSession) -> None:
    """The read-only login Superset uses. It inherits the sandbox role's grants, so it sees approved tables only."""
    role = settings.BI_DB_ROLE
    if not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", role):
        raise ValueError("BI_DB_ROLE must be a simple lower-case identifier.")
    password = settings.BI_DB_PASSWORD.replace("'", "''")
    # The role's comment records a keyed fingerprint of the password last set, so the password is only set
    # again when BI_DB_PASSWORD changes. Resetting it gives it a new salt, which a connection pooler that
    # caches credentials (Supabase's) briefly rejects. The key keeps the fingerprint from revealing it.
    secret = (settings.SECRETS_KEY or settings.JWT_SECRET).encode()
    fingerprint = "dataforge:" + hmac.new(secret, settings.BI_DB_PASSWORD.encode(), "sha256").hexdigest()
    conn = await session.connection()
    current = (
        await conn.execute(
            text("SELECT shobj_description(oid, 'pg_authid') FROM pg_roles WHERE rolname = :r"), {"r": role}
        )
    ).first()
    # A new role has no superuser or role-making rights by default. Naming SUPERUSER in an ALTER is refused
    # on PostgreSQL 16+ unless the caller is a superuser, which a hosted database's owner (Supabase) is not.
    statement = (
        f"CREATE ROLE {role} WITH LOGIN PASSWORD '{password}' NOCREATEDB NOCREATEROLE"
        if current is None
        else None
        if current[0] == fingerprint
        else f"ALTER ROLE {role} WITH LOGIN PASSWORD '{password}'"
    )
    if statement:
        try:
            await conn.execute(text(statement))
        except DBAPIError as error:
            # The statement holds the password, so it must not travel with the error into logs.
            detail = str(error.orig).splitlines()[0] if error.orig else type(error).__name__
            raise RuntimeError(f"The {role} login could not be created or updated: {detail}") from None
        await conn.execute(text(f"COMMENT ON ROLE {role} IS '{fingerprint}'"))
    await conn.execute(text(f"GRANT {settings.SANDBOX_ROLE} TO {role}"))
    await conn.execute(text(f"ALTER ROLE {role} SET default_transaction_read_only = on"))
    await conn.execute(text(f"ALTER ROLE {role} SET statement_timeout = '30s'"))
    await conn.execute(text(f"ALTER ROLE {role} SET search_path = gold, silver"))
    await session.commit()


def _superset_metrics(dataset: Dataset, specs: list[Any]) -> list[dict[str, Any]]:
    types = {c["name"]: c["type"] for c in (dataset.fields or {}).get("columns", [])}
    out = []
    for spec in specs:
        try:
            validate_metric(spec, types)
        except MetricError:
            continue
        out.append(
            {
                "metric_name": spec.key,
                "verbose_name": spec.name,
                "expression": expression_for(spec),
                "description": f"DataForge shared metric. {spec.description}".strip(),
                "d3format": _D3_FORMATS.get(spec.unit),
                "warning_text": None,
            }
        )
    return out


async def sync_dataset(dataset_id: str) -> None:
    """Register, update or remove one dataset in Superset to match its governance state."""
    if not superset.enabled:
        return
    global _bi_role_ready
    async with SessionLocal() as session:
        if not _bi_role_ready:
            await ensure_bi_role(session)
            _bi_role_ready = True
        dataset = await session.get(Dataset, dataset_id)
        if dataset is None:
            return
        wanted = is_queryable(dataset) and dataset.visibility == "organisation" and dataset.fields
        for layer in ("silver", "gold"):
            if not (wanted and dataset.layer == layer):
                await superset.remove_dataset(layer, dataset.table_name)
        if not wanted:
            return
        specs = await catalogue.published_metrics(session)
        superset_id = await superset.sync_dataset(
            dataset.layer,
            dataset.table_name,
            f"{dataset.name}. {dataset.description} (DataForge dataset {dataset.id})",
            _superset_metrics(dataset, specs),
        )
        policies = {
            p.role: p.predicate
            for p in (
                await session.execute(select(RowPolicy).where(RowPolicy.dataset_id == dataset.id))
            ).scalars()
        }
        await superset.sync_row_policies(superset_id, dataset.table_name, policies)


async def sync_all(session: AsyncSession) -> dict[str, Any]:
    await ensure_bi_role(session)
    await superset.ensure_database()
    ids = [
        d.id for d in (await session.execute(select(Dataset).where(Dataset.erased_at.is_(None)))).scalars()
    ]
    synced, failed = [], []
    for dataset_id in ids:
        try:
            await sync_dataset(dataset_id)
            synced.append(dataset_id)
        except Unavailable as error:
            failed.append({"dataset": dataset_id, "error": error.message})
    return {"synced": synced, "failed": failed}


async def sync_all_quietly() -> None:
    try:
        async with _sync_lock, SessionLocal() as session:
            await sync_all(session)
    except Exception as error:  # noqa: BLE001 - Superset being down must not fail the caller
        log.warning("superset_sync_failed", error=str(error))


async def sync_dataset_quietly(dataset_id: str) -> None:
    try:
        async with _sync_lock:
            await sync_dataset(dataset_id)
    except Exception as error:  # noqa: BLE001 - Superset being down must not undo a governance decision
        log.warning("superset_sync_failed", dataset=dataset_id, error=str(error))


def _require_superset(actor: Actor) -> None:
    if actor.role not in SUPERSET_ROLES:
        raise Forbidden("Your role does not use Superset.")
    if not superset.enabled:
        raise Unavailable("Superset is not configured.")


async def superset_sign_in(
    session: AsyncSession, actor: Actor, next_path: str = "/superset/welcome/"
) -> dict[str, str]:
    _require_superset(actor)
    actor.require_verified()
    if not next_path.startswith("/") or next_path.startswith("//"):
        next_path = "/superset/welcome/"
    token = create_sso_token(actor.id, "superset", role=actor.role, name=actor.name, email=actor.email)
    await audit.record(session, actor, "Opened Superset", "integration", None, "Superset", next_path)
    await session.commit()
    return {
        "url": f"{superset.public_url}/dataforge/sso?token={token}&next={quote(next_path)}",
        "expiresIn": str(settings.SSO_TOKEN_SECONDS),
    }


async def superset_dashboards(actor: Actor) -> list[dict[str, Any]]:
    _require_superset(actor)
    return await superset.list_dashboards()


async def superset_guest_token(session: AsyncSession, actor: Actor, dashboard_id: int) -> dict[str, Any]:
    """A guest token to embed one Superset dashboard, carrying the viewer's row policies."""
    _require_superset(actor)
    embedded = await superset.enable_embedding(dashboard_id, list(settings.CORS_ORIGINS))
    rls: list[dict[str, Any]] = []
    for dataset in await catalogue.queryable_datasets(session, actor):
        policy = await catalogue.policy_for(session, dataset.id, actor.role)
        if policy and dataset.visibility == "organisation":
            found = await superset.find_dataset(dataset.layer, dataset.table_name)
            if found:
                rls.append({"clause": policy, "dataset": int(found["id"])})
    names = actor.name.split(" ", 1)
    token = await superset.guest_token(
        embedded,
        {
            "username": f"df_{actor.id}",
            "first_name": names[0],
            "last_name": names[1] if len(names) > 1 else "",
        },
        rls,
    )
    await audit.record(
        session, actor, "Opened embedded dashboard", "integration", str(dashboard_id), "Superset dashboard"
    )
    await session.commit()
    return {"token": token, "embeddedId": embedded, "supersetDomain": superset.public_url}


# ------------------------------------------------------------------ Jupyter


def _require_jupyter(actor: Actor) -> None:
    actor.require(Permission.WRITE_CODE, "Jupyter is for roles that write code.")
    if actor.role not in JUPYTER_ROLES:
        raise Forbidden("Your role does not use Jupyter.")
    if not settings.JUPYTER_URL:
        raise Unavailable("Jupyter is not configured.")


async def jupyter_sign_in(session: AsyncSession, actor: Actor, path: str | None = None) -> dict[str, str]:
    _require_jupyter(actor)
    actor.require_verified()
    user = await session.get(User, actor.id)
    if user is None:
        raise NotFound("Account not found.")
    token, expires = notebook_token(user)
    sso = create_sso_token(
        actor.id,
        "jupyter",
        role=actor.role,
        name=actor.name,
        email=actor.email,
        extra={"nbt": token, "nbt_exp": expires},
    )
    await audit.record(session, actor, "Opened Jupyter", "integration", None, "Jupyter", path or "Workspace")
    await session.commit()
    return {
        "url": jupyter.lab_url(sso, path or jupyter.member_folder(actor.id)),
        "folder": jupyter.member_folder(actor.id),
    }


async def open_notebook(session: AsyncSession, actor: Actor, notebook_id: str) -> dict[str, str]:
    """Copy a DataForge notebook into the member's Jupyter folder and open it."""
    _require_jupyter(actor)
    notebook = await session.get(Notebook, notebook_id)
    if notebook is None:
        raise NotFound("Notebook not found.")
    data = await get_notebook(session, actor, notebook_id)
    path = await jupyter.write_notebook(
        actor.id, notebook.name, jupyter.to_ipynb(notebook.name, data["cells"])
    )
    notebook.jupyter_path = path
    await audit.record(
        session, actor, "Opened notebook in Jupyter", "notebook", notebook.id, notebook.name, path
    )
    await session.commit()
    return await jupyter_sign_in(session, actor, path)


async def open_sql(
    session: AsyncSession, actor: Actor, title: str, sql: str, note: str = ""
) -> dict[str, str]:
    """Progressive disclosure: take generated SQL (a chart, an answer) into a new Jupyter notebook."""
    _require_jupyter(actor)
    cells = ([{"kind": "markdown", "source": note}] if note else []) + [{"kind": "sql", "source": sql}]
    path = await jupyter.write_notebook(
        actor.id, title or "From DataForge", jupyter.to_ipynb(title or "From DataForge", cells)
    )
    await audit.record(session, actor, "Opened SQL in Jupyter", "integration", None, "Jupyter", path)
    await session.commit()
    return await jupyter_sign_in(session, actor, path)
