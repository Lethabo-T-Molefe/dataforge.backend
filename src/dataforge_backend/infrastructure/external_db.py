"""Read-only connectors to external PostgreSQL-compatible sources (databases and warehouses)."""

from dataclasses import dataclass
from typing import Any

import asyncpg
import pandas as pd

from ..core.errors import ValidationFailed

MAX_IMPORT_ROWS = 500_000


@dataclass(frozen=True)
class SourceConfig:
    host: str
    port: int
    database: str
    user: str
    password: str
    schema: str = "public"
    sslmode: str = "prefer"


async def _connect(config: SourceConfig) -> asyncpg.Connection:
    try:
        conn = await asyncpg.connect(
            host=config.host,
            port=config.port,
            database=config.database,
            user=config.user,
            password=config.password,
            ssl=None if config.sslmode == "disable" else config.sslmode,
            timeout=15,
            command_timeout=120,
            server_settings={
                "default_transaction_read_only": "on",
                "application_name": "dataforge-connector",
            },
        )
    except (OSError, asyncpg.PostgresError, TimeoutError) as error:
        raise ValidationFailed(f"Could not connect to the source: {str(error)[:200]}") from error
    return conn


async def list_tables(config: SourceConfig) -> list[dict[str, Any]]:
    conn = await _connect(config)
    try:
        rows = await conn.fetch(
            "SELECT table_name, (SELECT COUNT(*) FROM information_schema.columns c "
            " WHERE c.table_schema = t.table_schema AND c.table_name = t.table_name) AS columns "
            "FROM information_schema.tables t WHERE table_schema = $1 AND table_type IN ('BASE TABLE', 'VIEW') "
            "ORDER BY table_name",
            config.schema,
        )
        return [{"name": r["table_name"], "columns": int(r["columns"])} for r in rows]
    finally:
        await conn.close()


async def read_table(config: SourceConfig, table: str, limit: int = MAX_IMPORT_ROWS) -> pd.DataFrame:
    """Read one table in a read-only transaction. The name is checked against the source's own table list."""
    tables = {t["name"] for t in await list_tables(config)}
    if table not in tables:
        raise ValidationFailed(f"{table} is not a table in {config.schema}.")
    conn = await _connect(config)
    try:
        async with conn.transaction(readonly=True):
            quoted = f'"{config.schema.replace(chr(34), "")}"."{table.replace(chr(34), "")}"'
            rows = await conn.fetch(f"SELECT * FROM {quoted} LIMIT {int(limit)}")
    finally:
        await conn.close()
    if not rows:
        return pd.DataFrame()
    columns = list(rows[0].keys())
    return pd.DataFrame([tuple(r.values()) for r in rows], columns=columns, dtype=object)
