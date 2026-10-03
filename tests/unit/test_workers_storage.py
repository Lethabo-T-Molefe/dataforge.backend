import os
import time
from typing import Any

import pytest

from dataforge_backend.core.config import settings
from dataforge_backend.core.crypto import decrypt_bytes, encrypt_bytes
from dataforge_backend.core.errors import ValidationFailed
from dataforge_backend.infrastructure import storage
from dataforge_backend.infrastructure.storage import CloudinaryStore, LocalStore
from dataforge_backend.infrastructure.workers import WorkerPool

# ------------------------------------------------------------------ worker processes


async def test_a_job_runs_in_a_reused_worker_process() -> None:
    pool = WorkerPool(1, 1024)
    try:
        first = await pool.run(os.getpid, (), 60)
        second = await pool.run(os.getpid, (), 60)
        assert first == second != os.getpid()
        assert await pool.run(sum, ([1, 2, 3],), 60) == 6
    finally:
        pool.shutdown()


async def test_errors_come_back_as_they_were_raised() -> None:
    pool = WorkerPool(1, 1024)
    try:
        with pytest.raises(ValidationFailed, match="not supported"):
            await pool.run(storage.parse_upload, ("data.pdf", b"x"), 60)
        # The worker survives a failed job.
        assert await pool.run(len, ("abc",), 60) == 3
    finally:
        pool.shutdown()


async def test_a_job_past_its_time_limit_is_killed() -> None:
    pool = WorkerPool(1, 1024)
    try:
        pid = await pool.run(os.getpid, (), 60)
        started = time.perf_counter()
        with pytest.raises(TimeoutError):
            await pool.run(time.sleep, (60,), 1)
        assert time.perf_counter() - started < 15
        # The process that ran the job is gone, and the next job gets a new one.
        assert await pool.run(os.getpid, (), 60) != pid
    finally:
        pool.shutdown()


# ------------------------------------------------------------------ file encryption


def test_files_are_encrypted_and_plain_files_still_read() -> None:
    blob = encrypt_bytes(b"name,id_number\nThandi,9001015009087\n")
    assert blob.startswith(b"DFE1") and b"Thandi" not in blob
    assert decrypt_bytes(blob) == b"name,id_number\nThandi,9001015009087\n"
    assert decrypt_bytes(b"stored before encryption") == b"stored before encryption"
    assert encrypt_bytes(b"same") != encrypt_bytes(b"same")


# ------------------------------------------------------------------ the local landing zone


async def test_erasing_a_dataset_never_reaches_other_folders(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "STORAGE_DIR", tmp_path)
    local = LocalStore()
    local.put("connectors/sheet/abc.xlsx", b"workbook")
    local.put("datasets/connectors/v1/a.csv", b"a")
    # Files landed before the datasets/ folder sat directly under the dataset id.
    local.put("old-upload/v1/b.csv", b"b")
    monkeypatch.setattr(storage, "file_store", lambda: local)

    await storage.remove_dataset_files("connectors")
    assert (tmp_path / "connectors/sheet/abc.xlsx").exists()
    assert not (tmp_path / "datasets/connectors").exists()
    await storage.remove_dataset_files("old-upload")
    assert not (tmp_path / "old-upload").exists()


# ------------------------------------------------------------------ Cloudinary


class _Response:
    def __init__(self, status_code: int, content: bytes = b"") -> None:
        self.status_code = status_code
        self.content = content


def test_cloudinary_keeps_private_encrypted_raw_files(monkeypatch: pytest.MonkeyPatch) -> None:
    uploads: list[dict[str, Any]] = []
    held: dict[str, bytes] = {}

    def upload(file: Any, **options: Any) -> dict[str, Any]:
        held[options["public_id"]] = file.read()
        uploads.append(options)
        return {"public_id": options["public_id"]}

    def get(url: str, **_: Any) -> _Response:
        public_id = next((p for p in held if url.endswith(p)), None)
        return _Response(200, held[public_id]) if public_id else _Response(404)

    monkeypatch.setattr(storage.cloudinary.uploader, "upload", upload)
    monkeypatch.setattr(storage.httpx, "get", get)
    store = CloudinaryStore("cloudinary://key:secret@demo-cloud", "dataforge", encrypt=True)

    location = store.put("datasets/d1/v1/learners.csv", b"name\nThandi\n")
    assert location == "cloudinary:dataforge/datasets/d1/v1/learners.csv"
    options = uploads[0]
    assert options["resource_type"] == "raw" and options["type"] == "authenticated"
    assert b"Thandi" not in held["dataforge/datasets/d1/v1/learners.csv"]
    assert store.get("datasets/d1/v1/learners.csv") == b"name\nThandi\n"
    with pytest.raises(FileNotFoundError):
        store.get("datasets/d1/v1/missing.csv")


def test_cloudinary_prefix_deletes_follow_the_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Any]] = []

    def delete_by_prefix(prefix: str, **options: Any) -> dict[str, Any]:
        calls.append((prefix, options.get("next_cursor")))
        return {"next_cursor": "page-2"} if len(calls) == 1 else {}

    monkeypatch.setattr(storage.cloudinary.api, "delete_resources_by_prefix", delete_by_prefix)
    store = CloudinaryStore("cloudinary://key:secret@demo-cloud", "dataforge", encrypt=True)
    store.delete_prefix("datasets/d1")
    # The trailing slash keeps dataset d1 from matching d10.
    assert calls == [("dataforge/datasets/d1/", None), ("dataforge/datasets/d1/", "page-2")]


def test_a_malformed_cloudinary_url_is_refused() -> None:
    with pytest.raises(ValueError, match="cloudinary://"):
        CloudinaryStore("https://example.com", "dataforge", encrypt=True)
