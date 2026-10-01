"""DataForge from a notebook: governed data as pandas DataFrames.

    import dataforge as df
    df.datasets()                                   # what you may read
    df.sql("SELECT region, SUM(revenue) FROM gold.revenue_by_region GROUP BY 1")
    df.analyse(dataset="course-completions", method="trend", measure="completion_rate")

    %load_ext dataforge
    %%sql
    SELECT course, SUM(enrolments) FROM silver.course_completions GROUP BY 1

Every call goes through the DataForge API with your own notebook token, so your role's permissions,
masking and row policies apply exactly as they do everywhere else. The kernel holds no database password.
"""

import os
from typing import Any

import httpx
import pandas as pd

__all__ = ["DataForgeError", "analyse", "datasets", "metrics", "sql", "table", "whoami"]


class DataForgeError(RuntimeError):
    """The DataForge API refused or could not complete a request."""


def _client() -> httpx.Client:
    token = os.environ.get("DATAFORGE_TOKEN", "")
    if not token:
        raise DataForgeError("No DataForge session in this kernel. Open Jupyter from DataForge (Workbench → Open in Jupyter).")
    base = os.environ.get("DATAFORGE_API_URL", "http://localhost:8000").rstrip("/")
    return httpx.Client(base_url=f"{base}/api/v1", headers={"Authorization": f"Bearer {token}"}, timeout=120)


def _call(method: str, path: str, **kwargs: Any) -> Any:
    with _client() as client:
        response = client.request(method, path, **kwargs)
    if response.status_code == 401:
        raise DataForgeError("Your notebook session has expired. Open Jupyter from DataForge again.")
    try:
        body = response.json()
    except ValueError:
        body = {"message": response.text}
    if response.status_code >= 400:
        error = body.get("error", body) if isinstance(body, dict) else {}
        message = error.get("message") or f"Request failed ({response.status_code})"
        suggestion = (error.get("details") or {}).get("suggestion")
        raise DataForgeError(f"{message}{f' {suggestion}' if suggestion else ''}")
    return body


def whoami() -> dict[str, Any]:
    """The member this kernel acts for."""
    return dict(_call("GET", "/auth/me"))


def datasets() -> pd.DataFrame:
    """Datasets you may see, with their layer, status and the table to query."""
    items = _call("GET", "/datasets")
    return pd.DataFrame(
        [
            {
                "id": d["id"],
                "name": d["name"],
                "layer": d["layer"],
                "status": d["status"],
                "table": d.get("table"),
                "rows": d["rows"],
                "classification": d["classification"],
            }
            for d in items
        ]
    )


def metrics() -> pd.DataFrame:
    """Shared metric definitions from the semantic layer."""
    return pd.DataFrame(_call("GET", "/metrics/definitions"))


def sql(query: str, max_rows: int | None = None) -> pd.DataFrame:
    """Run a read-only query in the sandbox and return the rows."""
    payload: dict[str, Any] = {"sql": query}
    if max_rows:
        payload["maxRows"] = max_rows
    result = _call("POST", "/sql/run", json=payload)
    if result.get("refused"):
        raise DataForgeError(result.get("explanation") or "The query was refused.")
    frame = pd.DataFrame(result["rows"], columns=result["columns"])
    if result.get("truncated"):
        print(f"Showing the first {len(frame)} rows (the sandbox's row cap).")
    return frame


def table(name: str, limit: int = 1000) -> pd.DataFrame:
    """A permitted table (for example gold.revenue_by_region), up to `limit` rows."""
    if not all(part.replace("_", "").isalnum() for part in name.split(".")):
        raise DataForgeError("Give a table name such as gold.revenue_by_region.")
    columns = _call("GET", f"/sql/tables/{name}/columns")
    listed = ", ".join(f'"{c["name"]}"' for c in columns)
    return sql(f"SELECT {listed} FROM {name} LIMIT {int(limit)}")


def analyse(dataset: str, method: str = "compare", **settings: Any) -> dict[str, Any]:
    """Run one of DataForge's analysis methods and return its result (reading, table, figures and SQL)."""
    config = {"datasetId": dataset, "method": method, **settings}
    return dict(_call("POST", "/analysis/run", json=config))


def load_ipython_extension(ipython: Any) -> None:
    """%%sql cell magic: run the cell in the DataForge sandbox and show the rows."""

    def _sql_magic(line: str, cell: str) -> pd.DataFrame:
        frame = sql(cell)
        if line.strip():
            ipython.user_ns[line.strip()] = frame
        return frame

    ipython.register_magic_function(_sql_magic, magic_kind="cell", magic_name="sql")
