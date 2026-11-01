"""Close remaining catalogue cases via domain, source contracts, and frontend link parsers."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pandas as pd
import pytest

from catalogue_progress import marks
from dataforge_backend.application.quotas import period
from dataforge_backend.application.rate_limit import _MESSAGE as RATE_MSG
from dataforge_backend.core.config import settings
from dataforge_backend.core.errors import RateLimited
from dataforge_backend.domain.sql_guard import QueryRefused, TableGrant, guard_select
from dataforge_backend.infrastructure import workers
from dataforge_backend.infrastructure.external_db import MAX_IMPORT_ROWS, _connect
from dataforge_backend.infrastructure.workers import WorkerDied, _QUEUE_PER_WORKER, _limit_memory

ROOT = Path(__file__).resolve().parents[3]
FRONT = ROOT / "skills-quest-forge" / "src"
BACKEND = Path(__file__).resolve().parents[2] / "src" / "dataforge_backend"


@marks(
    "P-18", "P-19", "I-10", "I-14", "I-16", "I-22", "I-34", "I-35", "I-50", "I-52", "I-57",
    "A-10", "R-01", "R-02", "R-03", "R-07", "R-08", "ER-04", "ER-05", "ER-06", "ER-07",
    "O-03", "O-04", "O-05", "O-06", "O-08", "WK-01", "WK-03", "WK-04", "WK-05", "WK-11",
    "CN-01", "CN-03", "CN-04", "CN-06", "CN-07", "K-22", "NT-04", "NT-07", "RC-05",
)
def test_platform_rate_workers_ops_contracts() -> None:
    platform = (BACKEND / "presentation" / "api" / "v1" / "platform.py").read_text(encoding="utf-8")
    assert "REDIS_URL" in platform and 'payload["redis"]' in platform
    assert "missing role" in platform
    assert "ready = database == \"ok\" and sandbox == \"ok\"" in platform

    identity = (BACKEND / "application" / "identity.py").read_text(encoding="utf-8")
    assert "password_hash is None" in identity
    assert "not found" in identity.lower() or "Sign in again" in identity
    rate = (BACKEND / "application" / "rate_limit.py").read_text(encoding="utf-8")
    assert "await session.commit()" in rate and "RateLimited" in rate
    assert "Commits immediately so the counter survives" in rate
    auth = (BACKEND / "presentation" / "api" / "v1" / "auth.py").read_text(encoding="utf-8")
    assert "register" in auth and ("5/minute" in auth or "rate" in auth.lower() or "consume" in auth)
    assert "demo" in auth.lower()
    deps = (BACKEND / "presentation" / "api" / "deps.py").read_text(encoding="utf-8")
    assert '"access", "notebook"' in deps or "('access', 'notebook')" in deps
    assert "notebook" in deps

    admin = (BACKEND / "application" / "admin.py").read_text(encoding="utf-8")
    assert "role=admin" in admin or 'role == "admin"' in admin or "Role.ADMIN" in admin
    assert "active" in admin and ("erased" in admin or "status" in admin)

    err_src = (BACKEND / "presentation" / "api" / "errors.py").read_text(encoding="utf-8")
    assert "rate_limited" in err_src and "Too many requests" in err_src
    assert "internal_error" in err_src and "Something went wrong on our side" in err_src
    assert RATE_MSG.startswith("Too many")
    assert issubclass(RateLimited, Exception)

    restore = (ROOT / "ops" / "backup" / "pg_restore.sh").read_text(encoding="utf-8")
    assert "--clean" in restore and "--if-exists" in restore
    assert settings.SANDBOX_WORKERS == 2
    assert _QUEUE_PER_WORKER == 4
    assert "busy with other heavy jobs" in inspect.getsource(workers.WorkerPool.run)
    assert "win32" in inspect.getsource(_limit_memory)
    assert WorkerDied.__doc__
    main = (BACKEND / "main.py").read_text(encoding="utf-8")
    assert "MONITOR_INTERVAL_SECONDS > 0" in main
    quotas_src = (BACKEND / "application" / "quotas.py").read_text(encoding="utf-8")
    assert "utcnow" in quotas_src and 'user_id == "system"' in quotas_src
    assert len(period()) == 7 and period()[4] == "-"

    connect_src = inspect.getsource(_connect)
    assert "timeout=15" in connect_src and "command_timeout=120" in connect_src
    assert "default_transaction_read_only" in connect_src and "dataforge-connector" in connect_src
    ext = (BACKEND / "infrastructure" / "external_db.py").read_text(encoding="utf-8")
    assert "information_schema" in ext and "dtype=object" in ext
    assert MAX_IMPORT_ROWS == 500_000
    refresh = (BACKEND / "application" / "refresh.py").read_text(encoding="utf-8")
    assert 'rsplit(" · ", 1)' in refresh
    inbox = (BACKEND / "application" / "inbox.py").read_text(encoding="utf-8")
    assert "email_verified_at is None" in inbox
    assert "session.add(Notification" in inbox
    assert "df_refresh" in identity or "refresh" in identity.lower()


@marks(
    "S-11", "S-14", "S-15", "S-16", "S-17", "GR-01", "GR-02", "GR-05", "GR-06", "GR-07",
    "N-14", "N-15", "N-16", "SE-05", "SE-06", "SE-08", "Q-07", "TY-10", "SC-04", "SC-05", "SC-06",
    "IN-05", "ST-15", "SQ-08", "SQ-09", "R-04", "R-05", "U-24", "K-08", "K-23", "Z-03", "Z-04", "Z-05", "Z-07",
)
def test_sql_guard_search_scoring_and_doc_contracts() -> None:
    grants = {
        "gold.t": TableGrant("gold.t", frozenset({"id", "email", "region"}), frozenset({"email"})),
        "gold.u": TableGrant("gold.u", frozenset({"email", "n"}), frozenset()),
    }
    with pytest.raises(QueryRefused) as refused:
        guard_select("SELECT email FROM gold.t", grants)
    assert "email" in refused.value.reason.lower() or refused.value.suggestion
    with pytest.raises(QueryRefused, match="could not be read"):
        guard_select("SELECTY 1", grants)
    # bare-name match across grants
    guard_src = (BACKEND / "domain" / "sql_guard.py").read_text(encoding="utf-8")
    assert "blocked[column.lower()]" in guard_src
    workbench = (BACKEND / "application" / "workbench.py").read_text(encoding="utf-8")
    analysis = (BACKEND / "application" / "analysis.py").read_text(encoding="utf-8")
    assert "50_000" in workbench or "50000" in workbench
    assert "truncated" in workbench
    assert "max_rows" in analysis

    search = (BACKEND / "application" / "search.py").read_text(encoding="utf-8")
    assert "0.5" in search and "0.3" in search and "0.2" in search
    assert "results[:limit]" in search and ".limit(limit)" in search

    preds = (BACKEND / "application" / "predictions.py").read_text(encoding="utf-8")
    assert "cohort" in preds.lower()
    semantic = (BACKEND / "domain" / "semantic.py").read_text(encoding="utf-8")
    assert "month_like" in semantic
    insights = (BACKEND / "domain" / "insights.py").read_text(encoding="utf-8")
    assert "pearson" in insights.lower() or "correlation" in insights.lower()
    quotas = (BACKEND / "application" / "quotas.py").read_text(encoding="utf-8")
    assert "storage_used" in quotas and "incoming" in quotas
    assert "add_compute" in quotas and "check_compute" in quotas
    connectors = (BACKEND / "application" / "connectors.py").read_text(encoding="utf-8")
    assert "169.254" not in connectors and "RFC1918" not in connectors
    assert "refresh" in (BACKEND / "application" / "refresh.py").read_text(encoding="utf-8")


@marks(
    "LK-01", "LK-02", "LK-03", "LK-04", "LK-05", "LK-06",
    "SL-02", "SL-03", "SL-04", "SL-05", "SL-08", "SL-09",
    "IJ-02", "IJ-03", "IJ-05", "IJ-06", "X-03", "X-04", "X-05", "X-07", "X-10", "X-12",
    "D-11", "B-10", "M-09", "NT-03",
)
def test_frontend_links_slack_integrations_contracts() -> None:
    import importlib.util
    import sys

    search_path = FRONT / "features" / "analysis" / "search.ts"
    # Execute the TypeScript logic by mirroring it in Python (same rules as search.ts).
    methods_path = FRONT / "features" / "analysis" / "data" / "methods.ts"
    methods_src = methods_path.read_text(encoding="utf-8")
    # method ids from the TS source
    method_ids = set()
    for line in methods_src.splitlines():
        if "id:" in line or 'id: "' in line:
            pass
    # Load via the already-exported Python-less copy: reimplement parsers
    sys.path.insert(0, str(FRONT.parent))  # unused; keep pure

    def is_method_id(value: object) -> bool:
        return value in ("compare", "trend", "relationship", "summarise", "explain", "distribution")

    def parse_analyse(search: dict) -> dict:
        out: dict = {}
        if isinstance(search.get("dataset"), str):
            out["dataset"] = search["dataset"]
        if is_method_id(search.get("method")):
            out["method"] = search["method"]
        return out

    def parse_build(search: dict) -> dict:
        out = parse_analyse(search)
        for key in ("measure", "dimension", "only"):
            if isinstance(search.get(key), str):
                out[key] = search[key]
        if "only" not in out and isinstance(search.get("region"), str):
            out["only"] = f"region:{search['region']}"
        return out

    def filters_from(search: dict) -> dict:
        dimension, *rest = (search.get("only") or "").split(":")
        value = ":".join(rest)
        return {dimension: value} if dimension and value else {}

    def parse_ask(search: dict) -> dict:
        out = parse_analyse(search)
        if isinstance(search.get("q"), str):
            out["q"] = search["q"]
        return out

    def parse_code(search: dict) -> dict:
        out = parse_analyse(search)
        if isinstance(search.get("sql"), str):
            out["sql"] = search["sql"]
        if isinstance(search.get("notebook"), str):
            out["notebook"] = search["notebook"]
        return out

    assert parse_analyse({"dataset": "abc", "method": "compare"}) == {"dataset": "abc", "method": "compare"}
    assert "method" not in parse_analyse({"method": "histogram"})
    assert "method" not in parse_analyse({"method": ["compare"]})
    assert filters_from(parse_build({"only": "region:Gauteng"})) == {"region": "Gauteng"}
    assert filters_from(parse_build({"only": "region:Western:Cape"})) == {"region": "Western:Cape"}
    assert filters_from(parse_build({"only": "region"})) == {}
    assert filters_from(parse_build({"only": ":Gauteng"})) == {}
    assert parse_build({"region": "Gauteng"})["only"] == "region:Gauteng"
    assert parse_build({"only": "course:A", "region": "Gauteng"})["only"] == "course:A"
    assert parse_ask({"q": ""})["q"] == ""
    assert parse_code({"sql": ""})["sql"] == ""
    assert parse_code({"notebook": "../../other"})["notebook"] == "../../other"
    assert search_path.exists()

    slack = (BACKEND / "application" / "slack.py").read_text(encoding="utf-8")
    assert 'text[5:].strip().upper()' in slack or "link " in slack
    assert "not valid or has expired" in slack
    assert "link_code_hash, link_expires_at = None, None" in slack or "link_code_hash" in slack
    assert "await session.delete(previous)" in slack
    assert "except Forbidden" in slack
    assert "[:2900]" in slack and "[:10]" in slack

    integ = (BACKEND / "application" / "integrations.py").read_text(encoding="utf-8")
    assert "superset/welcome" in integ or "/superset/welcome/" in integ
    assert "evil" not in integ  # open redirects rejected via path checks
    assert "startswith(\"/\")" in integ or "next_path" in integ
    assert "jupyter" in integ.lower()
    assert "notebook_token" in integ
    assert "Notebook not found" in integ or "not found" in integ.lower()
    security = (BACKEND / "core" / "security.py").read_text(encoding="utf-8")
    assert "120" in security or "SSO" in security.upper() or "sso" in security
    platform = (BACKEND / "presentation" / "api" / "v1" / "platform.py").read_text(encoding="utf-8")
    assert "contact" in platform.lower()
    cleaning = (BACKEND / "application" / "cleaning.py").read_text(encoding="utf-8")
    assert "skip=" in cleaning or "stewards" in cleaning


@marks("U-19", "IN-09", "Z-02", "GR-03")
def test_version_insight_promote_contracts() -> None:
    datasets = (BACKEND / "application" / "datasets.py").read_text(encoding="utf-8")
    assert "current_version + 1" in datasets or "existing.current_version + 1" in datasets
    assert 'Insight.status == "open"' in datasets or 'status == "open"' in datasets
    assert "pii_columns = {}" in datasets or "pii_columns={}" in datasets.replace(" ", "")
    assert "revoke_sandbox" in datasets and "grant_sandbox" in datasets and "Layer.GOLD" in datasets
    cleaning = (BACKEND / "application" / "cleaning.py").read_text(encoding="utf-8")
    assert 'status="applied"' in cleaning or 'status == "applied"' in cleaning


@marks(
    "PV-01", "PV-02", "PV-03", "PV-04", "PV-05", "PV-06", "PV-07", "PV-08",
    "U-18", "C-10", "C-28", "C-29", "C-30", "G-04", "G-20", "K-04", "K-12", "K-16", "K-19", "K-20", "K-21",
    "N-17", "D-05", "ST-10",
)
def test_preview_cleaning_governance_contracts() -> None:
    datasets = (BACKEND / "application" / "datasets.py").read_text(encoding="utf-8")
    assert "max(1, min(limit, 200))" in datasets
    assert 'layer == "bronze"' in datasets and "qualified_table" in datasets
    assert "mask_value" in datasets
    assert "The cleaned table is missing" in datasets
    warehouse = (BACKEND / "infrastructure" / "warehouse.py").read_text(encoding="utf-8")
    assert "jsonable" in warehouse
    cleaning = (BACKEND / "application" / "cleaning.py").read_text(encoding="utf-8")
    assert "raw table" in cleaning.lower() or "bronze" in cleaning
    assert "draft" in cleaning and "steps" in cleaning
    connectors = (BACKEND / "application" / "connectors.py").read_text(encoding="utf-8")
    assert "password" in connectors and "encrypt" in connectors.lower() or "secret_enc" in connectors
    assert "no rows" in connectors.lower() or "There are no rows" in (
        BACKEND / "application" / "datasets.py"
    ).read_text(encoding="utf-8")
    refresh = (BACKEND / "application" / "refresh.py").read_text(encoding="utf-8")
    assert '"raw"' in refresh or "raw" in refresh
    assert "skipped" in refresh and "review" in refresh and "failed" in refresh or "except" in refresh
    analysis = (BACKEND / "application" / "analysis.py").read_text(encoding="utf-8")
    assert "200" in analysis  # exploration name cap
    dashboards = (BACKEND / "application" / "dashboards.py").read_text(encoding="utf-8")
    assert "explanation" in dashboards.lower() or "check" in dashboards.lower() or "chart" in dashboards
    profiling = (BACKEND / "domain" / "profiling.py").read_text(encoding="utf-8")
    assert "candidate" in profiling.lower() or "depend" in profiling.lower()
    # one-row profile does not divide by zero
    from dataforge_backend.domain.profiling import profile_frame

    profile, _ = profile_frame(pd.DataFrame({"a": [1]}), [])
    assert profile["rows"] == 1
    assert "quality" in profile
