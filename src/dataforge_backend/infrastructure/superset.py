"""Apache Superset, driven from DataForge through its REST API with a service account.

DataForge keeps Superset in step with governance: the warehouse connection uses the read-only BI login,
only approved organisation-wide datasets are registered, each carries the semantic layer's shared metrics,
row policies become Superset row-level security rules, and embedded dashboards use guest tokens that
carry the viewer's row policies.
"""

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlparse

import httpx

from ..core.config import settings
from ..core.errors import Unavailable
from ..core.logging import get_logger

log = get_logger(__name__)
RLS_PREFIX = "dataforge:"


def rison(value: Any) -> str:
    """Encode a value as Rison, the query format Superset's list endpoints read."""
    if value is True:
        return "!t"
    if value is False:
        return "!f"
    if value is None:
        return "!n"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, dict):
        return "(" + ",".join(f"{rison(str(k))}:{rison(v)}" for k, v in value.items()) + ")"
    if isinstance(value, list | tuple):
        return "!(" + ",".join(rison(v) for v in value) + ")"
    text = str(value)
    if (
        text
        and all(ch.isalnum() or ch in "-_./~" for ch in text)
        and not text[0].isdigit()
        and text[0] != "-"
    ):
        return text
    return "'" + text.replace("!", "!!").replace("'", "!'") + "'"


def _q(value: dict[str, Any]) -> str:
    return quote(rison(value), safe="():,!'")


@dataclass
class SupersetStatus:
    enabled: bool
    reachable: bool
    url: str | None
    detail: str = ""


class SupersetClient:
    def __init__(self) -> None:
        self.base = (settings.SUPERSET_URL or "").rstrip("/")
        self._token: str | None = None
        self._csrf: str | None = None
        self._client: httpx.AsyncClient | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.base)

    @property
    def public_url(self) -> str:
        return (settings.SUPERSET_PUBLIC_URL or self.base).rstrip("/")

    async def _http(self) -> httpx.AsyncClient:
        if not self.enabled:
            raise Unavailable("Superset is not configured.")
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base, timeout=30)
        if self._token is None:
            try:
                response = await self._client.post(
                    "/api/v1/security/login",
                    json={
                        "username": settings.SUPERSET_ADMIN_USER,
                        "password": settings.SUPERSET_ADMIN_PASSWORD,
                        "provider": "db",
                        "refresh": False,
                    },
                )
            except httpx.HTTPError as error:
                raise Unavailable("Superset is not reachable.") from error
            if response.status_code != 200:
                raise Unavailable("Superset rejected the DataForge service account.")
            self._token = response.json()["access_token"]
            self._client.headers["Authorization"] = f"Bearer {self._token}"
            csrf = await self._client.get("/api/v1/security/csrf_token/")
            if csrf.status_code == 200:
                self._csrf = csrf.json().get("result")
                self._client.headers["X-CSRFToken"] = self._csrf or ""
                self._client.headers["Referer"] = self.base
        return self._client

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        client = await self._http()
        response = await client.request(method, path, **kwargs)
        if response.status_code == 401:
            # The access token expired: sign in again once.
            self._token = None
            client = await self._http()
            response = await client.request(method, path, **kwargs)
        if response.status_code >= 400:
            log.warning(
                "superset_api_error",
                method=method,
                path=path,
                status=response.status_code,
                body=response.text[:500],
            )
            raise Unavailable(f"Superset refused {method} {path.split('?')[0]} ({response.status_code}).")
        return response.json() if response.content else {}

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._token = None

    async def status(self) -> SupersetStatus:
        if not self.enabled:
            return SupersetStatus(False, False, None, "Set SUPERSET_URL to enable Superset.")
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.base}/health")
            return SupersetStatus(True, response.status_code == 200, self.public_url)
        except httpx.HTTPError:
            return SupersetStatus(True, False, self.public_url, "Superset is not reachable.")

    # ------------------------------------------------------------------ warehouse connection

    def _warehouse_uri(self) -> str:
        parsed = urlparse(settings.sync_database_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 5432
        database = (parsed.path or "/").lstrip("/")
        password = quote(settings.BI_DB_PASSWORD, safe="")
        return f"postgresql+psycopg2://{settings.BI_DB_ROLE}:{password}@{host}:{port}/{database}"

    async def ensure_database(self) -> int:
        found = await self._request(
            "GET",
            f"/api/v1/database/?q={_q({'filters': [{'col': 'database_name', 'opr': 'eq', 'value': settings.SUPERSET_DATABASE_NAME}]})}",
        )
        payload = {
            "database_name": settings.SUPERSET_DATABASE_NAME,
            "sqlalchemy_uri": self._warehouse_uri(),
            "expose_in_sqllab": True,
            "allow_dml": False,
            "allow_file_upload": False,
            "allow_ctas": False,
            "allow_cvas": False,
            "allow_run_async": False,
            "extra": json.dumps(
                {
                    "schemas_allowed_for_file_upload": [],
                    "engine_params": {"connect_args": {"options": "-c statement_timeout=30000"}},
                }
            ),
        }
        if found.get("result"):
            database_id = int(found["result"][0]["id"])
            await self._request("PUT", f"/api/v1/database/{database_id}", json=payload)
            return database_id
        created = await self._request("POST", "/api/v1/database/", json=payload)
        return int(created["id"])

    # ------------------------------------------------------------------ datasets and metrics

    async def find_dataset(self, schema: str, table: str) -> dict[str, Any] | None:
        filters = [
            {"col": "table_name", "opr": "eq", "value": table},
            {"col": "schema", "opr": "eq", "value": schema},
        ]
        found = await self._request("GET", f"/api/v1/dataset/?q={_q({'filters': filters})}")
        return found["result"][0] if found.get("result") else None

    async def sync_dataset(
        self, schema: str, table: str, description: str, metrics: list[dict[str, Any]]
    ) -> int:
        """Register a table and give it the semantic layer's shared metrics. Returns Superset's dataset id."""
        database_id = await self.ensure_database()
        existing = await self.find_dataset(schema, table)
        if existing is None:
            created = await self._request(
                "POST",
                "/api/v1/dataset/",
                json={"database": database_id, "schema": schema, "table_name": table},
            )
            dataset_id = int(created["id"])
        else:
            dataset_id = int(existing["id"])
            await self._request("PUT", f"/api/v1/dataset/{dataset_id}/refresh")
        current = (await self._request("GET", f"/api/v1/dataset/{dataset_id}"))["result"]
        by_name = {m["metric_name"]: m for m in current.get("metrics", [])}
        wanted = []
        for metric in metrics:
            entry = {k: v for k, v in metric.items() if v is not None}
            if metric["metric_name"] in by_name:
                entry["id"] = by_name[metric["metric_name"]]["id"]
            wanted.append(entry)
        # Keep Superset's own count metric; replace everything DataForge manages.
        for name, metric in by_name.items():
            if name == "count":
                wanted.append(
                    {"id": metric["id"], "metric_name": "count", "expression": metric["expression"]}
                )
        await self._request(
            "PUT",
            f"/api/v1/dataset/{dataset_id}?override_columns=false",
            json={"description": description, "metrics": wanted},
        )
        return dataset_id

    async def remove_dataset(self, schema: str, table: str) -> None:
        existing = await self.find_dataset(schema, table)
        if existing is not None:
            await self._request("DELETE", f"/api/v1/dataset/{existing['id']}")

    # ------------------------------------------------------------------ row-level security

    async def _role_id(self, name: str) -> int | None:
        found = await self._request(
            "GET",
            f"/api/v1/security/roles/?q={_q({'filters': [{'col': 'name', 'opr': 'eq', 'value': name}]})}",
        )
        result = found.get("result") or []
        return int(result[0]["id"]) if result else None

    async def sync_row_policies(self, dataset_id: int, table: str, policies: dict[str, str]) -> None:
        """Replace DataForge-managed rules for one dataset: role → predicate."""
        found = await self._request(
            "GET",
            f"/api/v1/rowlevelsecurity/?q={_q({'filters': [{'col': 'name', 'opr': 'sw', 'value': f'{RLS_PREFIX}{table}:'}], 'page_size': 100})}",
        )
        for rule in found.get("result") or []:
            await self._request("DELETE", f"/api/v1/rowlevelsecurity/{rule['id']}")
        for role, clause in policies.items():
            role_id = await self._role_id(f"df_{role}")
            if role_id is None:
                continue
            await self._request(
                "POST",
                "/api/v1/rowlevelsecurity/",
                json={
                    "name": f"{RLS_PREFIX}{table}:{role}",
                    "filter_type": "Regular",
                    "tables": [dataset_id],
                    "roles": [role_id],
                    "clause": clause,
                    "description": f"Managed by DataForge: rows the {role} role may read.",
                },
            )

    # ------------------------------------------------------------------ dashboards and embedding

    async def list_dashboards(self) -> list[dict[str, Any]]:
        found = await self._request(
            "GET",
            f"/api/v1/dashboard/?q={_q({'page_size': 100, 'order_column': 'changed_on_delta_humanized', 'order_direction': 'desc'})}",
        )
        return [
            {
                "id": d["id"],
                "title": d.get("dashboard_title") or "Untitled",
                "url": f"{self.public_url}{d.get('url') or f'/superset/dashboard/{d["id"]}/'}",
                "published": bool(d.get("published")),
                "changed": d.get("changed_on_delta_humanized") or "",
                "owners": [
                    f"{o.get('first_name', '')} {o.get('last_name', '')}".strip()
                    for o in d.get("owners") or []
                ],
            }
            for d in found.get("result") or []
        ]

    async def enable_embedding(self, dashboard_id: int, allowed_domains: list[str]) -> str:
        created = await self._request(
            "POST", f"/api/v1/dashboard/{dashboard_id}/embedded", json={"allowed_domains": allowed_domains}
        )
        return str(created["result"]["uuid"])

    async def guest_token(self, embedded_uuid: str, user: dict[str, str], rls: list[dict[str, Any]]) -> str:
        created = await self._request(
            "POST",
            "/api/v1/security/guest_token/",
            json={"user": user, "resources": [{"type": "dashboard", "id": embedded_uuid}], "rls": rls},
        )
        return str(created["token"])


superset = SupersetClient()
