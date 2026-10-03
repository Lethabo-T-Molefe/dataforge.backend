"""Worker processes for heavy Python work (parsing, profiling, full cleaning runs).

A job runs in its own operating-system process, so the API keeps answering while it works, and a job that
reaches its time limit is killed rather than left running. Workers start on demand and are reused; only
SANDBOX_WORKERS jobs run at once and a short queue waits behind them.
"""

import asyncio
import multiprocessing
import sys
from collections.abc import Callable
from functools import partial
from multiprocessing.connection import Connection
from typing import Any

from ..core.config import settings
from ..core.errors import Refused, Unavailable

# Spawned, never forked: a forked child would inherit the API's event loop and database connections.
_context = multiprocessing.get_context("spawn")
# Requests waiting for a worker each hold a database connection, so the queue is kept short.
_QUEUE_PER_WORKER = 4


class WorkerDied(Exception):
    """The worker process ended before it returned a result."""


def _limit_memory(megabytes: int) -> None:
    if sys.platform == "win32":  # Windows has no address-space limit to set
        return
    import resource

    limit = megabytes * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ValueError, OSError):
        pass


def _serve(conn: Connection, memory_mb: int) -> None:
    """A worker's whole life: take (function, arguments), send back (succeeded, value or error)."""
    _limit_memory(memory_mb)
    while True:
        try:
            fn, args = conn.recv()
        except (EOFError, OSError):
            return
        outcome: tuple[bool, Any]
        try:
            outcome = (True, fn(*args))
        except Exception as error:  # noqa: BLE001 - every failure is reported to the API, which re-raises it
            outcome = (False, error)
        try:
            conn.send(outcome)
        except Exception as error:  # noqa: BLE001 - the value or the error could not be pickled
            conn.send((False, RuntimeError(f"The result could not be returned: {error}")))


class _Worker:
    def __init__(self, memory_mb: int) -> None:
        self.conn, child = _context.Pipe()
        self.process = _context.Process(
            target=_serve, args=(child, memory_mb), daemon=True, name="dataforge-worker"
        )
        self.process.start()
        child.close()

    def call(self, fn: Callable[..., Any], args: tuple[Any, ...]) -> tuple[bool, Any]:
        """Blocking: runs in a thread while the event loop waits."""
        try:
            self.conn.send((fn, args))
            outcome: tuple[bool, Any] = self.conn.recv()
        except (EOFError, OSError) as error:
            raise WorkerDied from error
        return outcome

    def kill(self) -> None:
        if self.process.is_alive():
            self.process.kill()

    def close(self) -> None:
        self.kill()
        self.conn.close()


def _discard(worker: _Worker, call: "asyncio.Future[tuple[bool, Any]]") -> None:
    """Once the abandoned call returns (the killed worker's pipe breaks), retrieve its error and close."""
    if not call.cancelled():
        call.exception()
    worker.close()


class WorkerPool:
    def __init__(self, size: int, memory_mb: int) -> None:
        self.size = max(1, size)
        self.memory_mb = memory_mb
        self._idle: list[_Worker] = []
        self._waiting = 0
        self._slots: tuple[asyncio.AbstractEventLoop, asyncio.Semaphore] | None = None

    def _semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._slots is None or self._slots[0] is not loop:
            self._slots = (loop, asyncio.Semaphore(self.size))
        return self._slots[1]

    def _take(self) -> _Worker | None:
        while self._idle:
            worker = self._idle.pop()
            if worker.process.is_alive():
                return worker
            worker.close()
        return None

    async def run(self, fn: Callable[..., Any], args: tuple[Any, ...], timeout_s: float) -> Any:
        """Run fn(*args) in a worker. Raise TimeoutError (after killing the worker) at the time limit."""
        if self._waiting >= self.size * _QUEUE_PER_WORKER:
            raise Unavailable("DataForge is busy with other heavy jobs. Try again in a minute.")
        slots = self._semaphore()
        self._waiting += 1
        try:
            await slots.acquire()
        finally:
            self._waiting -= 1
        try:
            worker = self._take()
            if worker is None:
                worker = await asyncio.to_thread(_Worker, self.memory_mb)
            call = asyncio.get_running_loop().run_in_executor(None, worker.call, fn, args)
            try:
                succeeded, value = await asyncio.wait_for(asyncio.shield(call), timeout_s)
            except BaseException:
                # Time limit, a cancelled request, or a worker that died: the process must not live on.
                worker.kill()
                call.add_done_callback(partial(_discard, worker))
                raise
            self._idle.append(worker)
        except WorkerDied as error:
            raise Refused(
                "The step stopped before it finished. It may have needed more memory than the sandbox allows.",
                details={"suggestion": "Try it on fewer rows or columns."},
            ) from error
        finally:
            slots.release()
        if succeeded:
            return value
        if isinstance(value, MemoryError):
            raise Refused(
                f"The step needed more than the {self.memory_mb} MB of memory the sandbox allows.",
                details={"suggestion": "Try it on fewer rows or columns."},
            ) from value
        assert isinstance(value, BaseException)
        raise value

    def shutdown(self) -> None:
        while self._idle:
            self._idle.pop().close()


pool = WorkerPool(settings.SANDBOX_WORKERS, settings.SANDBOX_WORKER_MEMORY_MB)
