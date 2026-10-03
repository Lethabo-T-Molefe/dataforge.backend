"""The landing zone: uploaded files are kept per dataset and version, then parsed into raw text frames.

Files are kept on local disk, or in Cloudinary when CLOUDINARY_URL is set. In Cloudinary they are raw,
authenticated assets (never reachable by a public URL), and they are encrypted before they leave the API
unless STORAGE_ENCRYPT is off, so the file store only holds ciphertext.
"""

import asyncio
import csv
import hashlib
import io
import json
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlparse

import cloudinary
import cloudinary.api
import cloudinary.exceptions
import cloudinary.uploader
import cloudinary.utils
import httpx
import pandas as pd

from ..core.config import settings
from ..core.crypto import decrypt_bytes, encrypt_bytes
from ..core.errors import Unavailable, ValidationFailed
from ..core.logging import get_logger

log = get_logger(__name__)

SUPPORTED = (".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".jsonl")
# Folder names under the landing zone that a dataset id must never be mistaken for.
_RESERVED = ("datasets", "connectors")
# Cloudinary takes up to 100 MB in one request; larger files go up in chunks.
_SINGLE_REQUEST_BYTES = 20_000_000


# ------------------------------------------------------------------ file stores


class FileStore(Protocol):
    name: str

    def put(self, key: str, content: bytes) -> str: ...

    def get(self, key: str) -> bytes: ...

    def delete(self, key: str) -> None: ...

    def delete_prefix(self, prefix: str) -> None: ...


class LocalStore:
    name = "local"

    def put(self, key: str, content: bytes) -> str:
        path = settings.STORAGE_DIR / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return str(path)

    def get(self, key: str) -> bytes:
        return (settings.STORAGE_DIR / key).read_bytes()

    def delete(self, key: str) -> None:
        (settings.STORAGE_DIR / key).unlink(missing_ok=True)

    def delete_prefix(self, prefix: str) -> None:
        folder = settings.STORAGE_DIR / prefix
        if folder.is_dir():
            shutil.rmtree(folder)


class CloudinaryStore:
    """Raw, authenticated assets under one folder. Every call is a blocking HTTP request."""

    name = "cloudinary"
    _asset: dict[str, Any] = {"resource_type": "raw", "type": "authenticated"}

    def __init__(self, url: str, folder: str, encrypt: bool) -> None:
        parsed = urlparse(url)
        if parsed.scheme != "cloudinary" or not (parsed.username and parsed.password and parsed.hostname):
            raise ValueError("CLOUDINARY_URL must look like cloudinary://<api key>:<api secret>@<cloud name>")
        cloudinary.config(
            cloud_name=parsed.hostname,
            api_key=unquote(parsed.username),
            api_secret=unquote(parsed.password),
            secure=True,
        )
        self.folder = folder.strip("/")
        self.encrypt = encrypt

    def _public_id(self, key: str) -> str:
        return f"{self.folder}/{key}" if self.folder else key

    def put(self, key: str, content: bytes) -> str:
        public_id = self._public_id(key)
        body = encrypt_bytes(content) if self.encrypt else content
        options = {"public_id": public_id, "overwrite": True, "invalidate": True, **self._asset}
        try:
            if len(body) <= _SINGLE_REQUEST_BYTES:
                cloudinary.uploader.upload(io.BytesIO(body), **options)
            else:
                cloudinary.uploader.upload_large(io.BytesIO(body), filename=Path(key).name, **options)
        except cloudinary.exceptions.Error as error:
            log.warning("file_store_put_failed", key=key, error=str(error))
            raise Unavailable(f"The file store did not accept the file: {str(error)[:200]}") from error
        return f"cloudinary:{public_id}"

    def get(self, key: str) -> bytes:
        url, _ = cloudinary.utils.cloudinary_url(self._public_id(key), sign_url=True, **self._asset)
        try:
            response = httpx.get(url, timeout=120, follow_redirects=True)
        except httpx.HTTPError as error:
            raise Unavailable("The file store could not be reached.") from error
        if response.status_code == 404:
            raise FileNotFoundError(key)
        if response.status_code >= 300:
            raise Unavailable(f"The file store refused to return the file (status {response.status_code}).")
        return decrypt_bytes(response.content)

    def delete(self, key: str) -> None:
        try:
            cloudinary.uploader.destroy(self._public_id(key), invalidate=True, **self._asset)
        except cloudinary.exceptions.Error as error:
            raise Unavailable(f"The file store could not remove the file: {str(error)[:200]}") from error

    def delete_prefix(self, prefix: str) -> None:
        cursor: str | None = None
        try:
            # Each call removes up to 1 000 assets and says whether more remain.
            for _ in range(100):
                result = cloudinary.api.delete_resources_by_prefix(
                    f"{self._public_id(prefix)}/", invalidate=True, next_cursor=cursor, **self._asset
                )
                cursor = result.get("next_cursor")
                if not cursor:
                    return
        except cloudinary.exceptions.Error as error:
            raise Unavailable(f"The file store could not remove the files: {str(error)[:200]}") from error
        raise Unavailable("The file store still holds files for this item. Try again.")


@lru_cache(maxsize=1)
def file_store() -> FileStore:
    if settings.CLOUDINARY_URL:
        return CloudinaryStore(settings.CLOUDINARY_URL, settings.CLOUDINARY_FOLDER, settings.STORAGE_ENCRYPT)
    return LocalStore()


def _safe_name(filename: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in Path(filename).name) or "upload"


def _put(key: str, content: bytes) -> tuple[str, str]:
    return file_store().put(key, content), hashlib.sha256(content).hexdigest()


async def store(dataset_id: str, version: int, filename: str, content: bytes) -> tuple[str, str]:
    """Keep an uploaded file exactly as it arrived. Returns where it is kept and its SHA-256."""
    key = f"datasets/{dataset_id}/v{version}/{_safe_name(filename)}"
    return await asyncio.to_thread(_put, key, content)


async def store_workbook(connector_id: str, content: bytes) -> str:
    """Keep a spreadsheet connector's workbook. The key is new for every upload, so a copy is never stale."""
    key = f"connectors/{connector_id}/{hashlib.sha256(content).hexdigest()[:16]}.xlsx"
    await asyncio.to_thread(file_store().put, key, content)
    return key


async def read(key: str) -> bytes:
    """Raise FileNotFoundError when nothing is kept under the key."""
    return await asyncio.to_thread(file_store().get, key)


async def remove(key: str) -> None:
    await asyncio.to_thread(file_store().delete, key)


async def remove_dataset_files(dataset_id: str) -> None:
    store_ = file_store()
    await asyncio.to_thread(store_.delete_prefix, f"datasets/{dataset_id}")
    # Files landed before the datasets/ folder existed sat directly under the dataset's id.
    if store_.name == "local" and dataset_id not in _RESERVED:
        await asyncio.to_thread(store_.delete_prefix, dataset_id)


async def remove_connector_files(connector_id: str) -> None:
    await asyncio.to_thread(file_store().delete_prefix, f"connectors/{connector_id}")


# ------------------------------------------------------------------ parsing


def _decode(content: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValidationFailed("The file's text encoding could not be read. Save it as UTF-8 and upload again.")


def parse_upload(filename: str, content: bytes) -> pd.DataFrame:
    """Parse a file into object columns without guessing types; typing happens during cleaning."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise ValidationFailed(f"{suffix or 'This file type'} is not supported. Upload CSV, Excel or JSON.")
    if not content:
        raise ValidationFailed("The file is empty.")
    try:
        if suffix in (".xlsx", ".xls"):
            frame = pd.read_excel(io.BytesIO(content), dtype=object, sheet_name=0)
        elif suffix in (".json", ".jsonl"):
            raw = _decode(content)
            if suffix == ".jsonl" or raw.lstrip().startswith("{") and "\n{" in raw:
                records = [json.loads(line) for line in raw.splitlines() if line.strip()]
            else:
                data = json.loads(raw)
                records = data if isinstance(data, list) else data.get("rows") or data.get("data") or [data]
            frame = pd.DataFrame.from_records(
                [
                    {k: json.dumps(v) if isinstance(v, dict | list) else v for k, v in r.items()}
                    for r in records
                ]
            ).astype(object)
        else:
            raw = _decode(content)
            sample = raw[:20_000]
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                sep = dialect.delimiter
            except csv.Error:
                sep = "\t" if suffix == ".tsv" else ","
            frame = pd.read_csv(
                io.StringIO(raw), sep=sep, dtype=str, keep_default_na=False, na_values=[]
            ).astype(object)
    except ValidationFailed:
        raise
    except (ValueError, json.JSONDecodeError, pd.errors.ParserError) as error:
        raise ValidationFailed(f"The file could not be read: {str(error)[:200]}") from error
    if frame.empty or len(frame.columns) == 0:
        raise ValidationFailed("The file has no rows.")
    frame = frame.dropna(how="all")
    frame.columns = [str(c).strip() or f"column_{i + 1}" for i, c in enumerate(frame.columns)]
    return frame


def read_sheet(content: bytes, sheet: str) -> pd.DataFrame:
    """One sheet of a workbook as object columns."""
    return pd.read_excel(io.BytesIO(content), sheet_name=sheet, dtype=object)
