"""The sandbox for SQL: a read-only transaction under a least-privilege role, with a time and row cap.

Every run is recorded as a sandbox session with its caps and outcome, and its duration counts towards the
member's compute quota. Python work (cleaning, profiling, scoring) runs in a worker thread with a wall-clock
cap through `run_python`.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from sqlalchemy import insert, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine

from ..core.config import settings
from ..core.errors import Refused, Unavailable
from ..core.security import utcnow
from ..domain.analysis.sql import ident
from .db.models import SandboxSession, new_id
from .warehouse import jsonable

T = TypeVar("T")


@dataclass
class SandboxResult:
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool
    duration_ms: int
    session_id: str


class Sandbox:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    async def _record(self, session: SandboxSession) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(
                insert(SandboxSession).values(
                    id=session.id,
                    user_id=session.user_id,
                    kind=session.kind,
                    status=session.status,
                    caps=session.caps,
                    statement=session.statement[:20_000],
                    rows_returned=session.rows_returned or 0,
                    duration_ms=session.duration_ms or 0,
                    error=session.error,
                    started_at=session.started_at,
                    ended_at=session.ended_at,
                )
            )

    async def run_select(
        self,
        sql: str,
        *,
        user_id: str | None,
        kind: str = "sql",
        params: dict[str, Any] | None = None,
        max_rows: int | None = None,
        timeout_ms: int | None = None,
    ) -> SandboxResult:
        """Run an already-validated SELECT. Raise Refused on a timeout or a permission failure."""
        max_rows = max_rows or settings.SANDBOX_MAX_ROWS
        timeout_ms = timeout_ms or settings.SANDBOX_STATEMENT_TIMEOUT_MS
        record = SandboxSession(
            id=new_id(),
            user_id=user_id,
            kind=kind,
            caps={
                "timeoutMs": timeout_ms,
                "maxRows": max_rows,
                "role": settings.SANDBOX_ROLE,
                "readOnly": True,
            },
            statement=sql,
            started_at=utcnow(),
        )
        started = time.perf_counter()
        try:
            async with self.engine.connect() as conn:
                async with conn.begin():
                    await conn.execute(text("SET TRANSACTION READ ONLY"))
                    await conn.execute(text(f"SET LOCAL ROLE {ident(settings.SANDBOX_ROLE)}"))
                    await conn.execute(text(f"SET LOCAL statement_timeout = {int(timeout_ms)}"))
                    await conn.execute(text("SET LOCAL idle_in_transaction_session_timeout = 60000"))
                    await conn.execute(text("SET LOCAL search_path TO gold, silver"))
                    await conn.execute(text("SET LOCAL work_mem = '32MB'"))
                    result = await conn.stream(text(sql), params or {})
                    columns = list(result.keys())
                    fetched = await result.fetchmany(max_rows + 1)
                    await result.close()
                    await conn.rollback()
        except DBAPIError as error:
            message = str(error.orig) if error.orig else str(error)
            record.duration_ms = int((time.perf_counter() - started) * 1000)
            record.ended_at = utcnow()
            lowered = message.lower()
            if "statement timeout" in lowered or "canceling statement" in lowered:
                record.status, record.error = "timeout", message
                await self._record(record)
                raise Refused(
                    f"The query ran longer than the sandbox allows ({timeout_ms // 1000} seconds) and was stopped.",
                    details={"suggestion": "Filter to fewer rows or aggregate before joining."},
                ) from error
            if "permission denied" in lowered:
                record.status, record.error = "refused", message
                await self._record(record)
                raise Refused("The sandbox identity is not permitted to read that object.") from error
            if "role" in lowered and "does not exist" in lowered:
                record.status, record.error = "error", message
                await self._record(record)
                raise Unavailable("The sandbox role is missing. Run the database migrations.") from error
            record.status, record.error = "error", message
            await self._record(record)
            first_line = message.split("\n")[0].replace("<class 'asyncpg.exceptions.", "").strip()
            raise Refused(f"The database could not run the query: {first_line}") from error

        truncated = len(fetched) > max_rows
        rows = [[jsonable(v) for v in row] for row in fetched[:max_rows]]
        record.status = "ok"
        record.rows_returned = len(rows)
        record.duration_ms = int((time.perf_counter() - started) * 1000)
        record.ended_at = utcnow()
        await self._record(record)
        return SandboxResult(columns, rows, truncated, record.duration_ms, record.id)

    async def run_python(
        self,
        fn: Callable[[], T],
        *,
        user_id: str | None,
        kind: str,
        description: str,
        timeout_s: float = 300.0,
    ) -> tuple[T, int]:
        """Run CPU-bound Python (pandas, scikit-learn) off the event loop with a wall-clock cap."""
        record = SandboxSession(
            id=new_id(),
            user_id=user_id,
            kind=kind,
            caps={"timeoutS": timeout_s, "isolation": "worker-thread"},
            statement=description,
            started_at=utcnow(),
        )
        started = time.perf_counter()
        try:
            value = await asyncio.wait_for(asyncio.to_thread(fn), timeout=timeout_s)
        except TimeoutError as error:
            record.status, record.error = "timeout", "wall-clock cap reached"
            record.duration_ms = int((time.perf_counter() - started) * 1000)
            record.ended_at = utcnow()
            await self._record(record)
            raise Refused(f"The step took longer than {int(timeout_s)} seconds and was stopped.") from error
        except Exception as error:
            record.status, record.error = "error", str(error)[:2000]
            record.duration_ms = int((time.perf_counter() - started) * 1000)
            record.ended_at = utcnow()
            await self._record(record)
            raise
        record.status = "ok"
        record.duration_ms = int((time.perf_counter() - started) * 1000)
        record.ended_at = utcnow()
        await self._record(record)
        return value, record.duration_ms
