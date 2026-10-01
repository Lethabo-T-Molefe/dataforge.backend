"""The medallion warehouse in PostgreSQL: bronze (raw text), silver (cleaned, typed), gold (analytics-ready).

Tables are written with COPY, read into pandas as object columns, and granted to the sandbox role only once
approved, so the sandbox can never read raw or unapproved data.
"""

import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from ..core.config import settings
from ..domain.analysis.sql import ident, qualified
from ..domain.semantic import ColumnInfo
from ..domain.values import is_blank

LAYERS = ("bronze", "silver", "gold")
ROW_COLUMN = "_row"
_PG_TYPES = {"text": "text", "integer": "bigint", "numeric": "numeric", "date": "date", "boolean": "boolean"}
_DOMAIN_TYPES = {
    "text": "text",
    "character varying": "text",
    "bigint": "integer",
    "integer": "integer",
    "smallint": "integer",
    "numeric": "numeric",
    "double precision": "numeric",
    "real": "numeric",
    "date": "date",
    "timestamp without time zone": "date",
    "timestamp with time zone": "date",
    "boolean": "boolean",
}


def safe_column(name: Any, taken: set[str]) -> str:
    """A PostgreSQL-friendly column name: lower snake case, unique, at most 63 bytes."""
    base = re.sub(r"[^a-z0-9_]+", "_", str(name).strip().lower()).strip("_") or "column"
    if base[0].isdigit():
        base = f"c_{base}"
    base = base[:60]
    candidate, n = base, 2
    while candidate in taken or candidate == ROW_COLUMN:
        candidate = f"{base[:56]}_{n}"
        n += 1
    taken.add(candidate)
    return candidate


def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    taken: set[str] = set()
    out = frame.copy()
    out.columns = [safe_column(c, taken) for c in frame.columns]
    return out


async def _copy(
    conn: AsyncConnection, schema: str, table: str, columns: list[str], records: list[tuple[Any, ...]]
) -> None:
    raw = await conn.get_raw_connection()
    driver = raw.driver_connection
    assert driver is not None
    await driver.copy_records_to_table(table, records=records, columns=columns, schema_name=schema)


def _text(value: Any) -> str | None:
    if is_blank(value):
        return None
    return str(value)


async def write_bronze(conn: AsyncConnection, table: str, frame: pd.DataFrame) -> int:
    """Raw data exactly as it arrived: every column as text, with the original row number."""
    columns = list(frame.columns)
    ddl = ", ".join([f"{ident(ROW_COLUMN)} bigint"] + [f"{ident(c)} text" for c in columns])
    await conn.execute(text(f"DROP TABLE IF EXISTS bronze.{ident(table)}"))
    await conn.execute(text(f"CREATE TABLE bronze.{ident(table)} ({ddl})"))
    records = [
        (i + 1, *(_text(v) for v in row)) for i, row in enumerate(frame.itertuples(index=False, name=None))
    ]
    await _copy(conn, "bronze", table, [ROW_COLUMN, *columns], records)
    return len(records)


def _convert(value: Any, kind: str) -> Any:
    if value is None or is_blank(value):
        return None
    if kind == "integer":
        return int(value)
    if kind == "numeric":
        return Decimal(str(value))
    if kind == "date":
        return value if isinstance(value, date) else None
    if kind == "boolean":
        return bool(value)
    return str(value)


async def write_typed(
    conn: AsyncConnection, schema: str, table: str, frame: pd.DataFrame, types: dict[str, str]
) -> int:
    """Replace a silver or gold table with typed columns. DDL is transactional, so readers see old or new."""
    if schema not in ("silver", "gold"):
        raise ValueError("Typed tables live in silver or gold.")
    columns = list(frame.columns)
    ddl = ", ".join(f"{ident(c)} {_PG_TYPES.get(types.get(c, 'text'), 'text')}" for c in columns)
    await conn.execute(text(f"DROP TABLE IF EXISTS {ident(schema)}.{ident(table)}"))
    await conn.execute(text(f"CREATE TABLE {ident(schema)}.{ident(table)} ({ddl})"))
    kinds = [types.get(c, "text") for c in columns]
    records = [
        tuple(_convert(v, k) for v, k in zip(row, kinds, strict=True))
        for row in frame.itertuples(index=False, name=None)
    ]
    await _copy(conn, schema, table, columns, records)
    return len(records)


async def read_frame(conn: AsyncConnection, table: str, limit: int | None = None) -> pd.DataFrame:
    """Read a warehouse table into object columns (bronze drops its row-number column)."""
    sql = f"SELECT * FROM {qualified(table)}"
    if table.startswith("bronze."):
        sql += f" ORDER BY {ident(ROW_COLUMN)}"
    if limit:
        sql += f" LIMIT {int(limit)}"
    result = await conn.execute(text(sql))
    columns = list(result.keys())
    rows = result.fetchall()
    frame = pd.DataFrame([tuple(r) for r in rows], columns=columns, dtype=object)
    if ROW_COLUMN in frame.columns:
        frame = frame.drop(columns=[ROW_COLUMN])
    return frame


async def drop_table(conn: AsyncConnection, table: str) -> None:
    await conn.execute(text(f"DROP TABLE IF EXISTS {qualified(table)}"))


async def table_exists(conn: AsyncConnection, table: str) -> bool:
    schema, _, name = table.partition(".")
    result = await conn.execute(
        text("SELECT 1 FROM information_schema.tables WHERE table_schema = :s AND table_name = :n"),
        {"s": schema, "n": name},
    )
    return result.first() is not None


async def grant_sandbox(conn: AsyncConnection, table: str) -> None:
    await conn.execute(text(f"GRANT SELECT ON {qualified(table)} TO {ident(settings.SANDBOX_ROLE)}"))


async def revoke_sandbox(conn: AsyncConnection, table: str) -> None:
    if await table_exists(conn, table):
        await conn.execute(text(f"REVOKE ALL ON {qualified(table)} FROM {ident(settings.SANDBOX_ROLE)}"))


async def table_size(conn: AsyncConnection, table: str) -> int:
    result = await conn.execute(
        text("SELECT pg_total_relation_size(CAST(:t AS regclass))"), {"t": qualified(table)}
    )
    return int(result.scalar() or 0)


async def column_types(conn: AsyncConnection, table: str) -> dict[str, str]:
    schema, _, name = table.partition(".")
    result = await conn.execute(
        text(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = :s AND table_name = :n ORDER BY ordinal_position"
        ),
        {"s": schema, "n": name},
    )
    return {r[0]: _DOMAIN_TYPES.get(r[1], "text") for r in result if r[0] != ROW_COLUMN}


async def column_infos(
    conn: AsyncConnection, table: str, pii: dict[str, str]
) -> tuple[list[ColumnInfo], int]:
    """Distinct counts and most common values per column, for deriving analysis fields."""
    types = await column_types(conn, table)
    rows = int((await conn.execute(text(f"SELECT COUNT(*) FROM {qualified(table)}"))).scalar() or 0)
    infos: list[ColumnInfo] = []
    for name, kind in types.items():
        col = ident(name)
        distinct = int(
            (await conn.execute(text(f"SELECT COUNT(DISTINCT {col}) FROM {qualified(table)}"))).scalar() or 0
        )
        values: tuple[str, ...] = ()
        month_like = False
        if kind in ("text", "boolean") and 0 < distinct <= 200:
            result = await conn.execute(
                text(
                    f"SELECT {col}::text, COUNT(*) FROM {qualified(table)} WHERE {col} IS NOT NULL "
                    f"GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 200"
                )
            )
            values = tuple(str(r[0]) for r in result)
            month_like = bool(values) and all(re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", v) for v in values)
        infos.append(ColumnInfo(name, kind, distinct, pii=name in pii, values=values, month_like=month_like))
    return infos, rows


async def distinct_months(conn: AsyncConnection, table: str, time_column: str, time_kind: str) -> list[str]:
    col = ident(time_column)
    expr = col if time_kind == "month" else f"to_char({col}, 'YYYY-MM')"
    result = await conn.execute(
        text(f"SELECT DISTINCT {expr} FROM {qualified(table)} WHERE {col} IS NOT NULL ORDER BY 1")
    )
    return [str(r[0]) for r in result if r[0]]


def jsonable(value: Any) -> Any:
    """Make a database value safe for a JSON response."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and value != value:  # NaN
        return None
    if isinstance(value, bytes | memoryview):
        return "(binary)"
    return value
