"""The landing zone: uploaded files are stored per dataset and version, then parsed into raw text frames."""

import csv
import hashlib
import io
import json
from pathlib import Path

import pandas as pd

from ..core.config import settings
from ..core.errors import ValidationFailed

SUPPORTED = (".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".jsonl")


def landing_path(dataset_id: str, version: int, filename: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in Path(filename).name) or "upload"
    folder = settings.STORAGE_DIR / dataset_id / f"v{version}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / safe


def store(dataset_id: str, version: int, filename: str, content: bytes) -> tuple[Path, str]:
    path = landing_path(dataset_id, version, filename)
    path.write_bytes(content)
    return path, hashlib.sha256(content).hexdigest()


def remove_dataset_files(dataset_id: str) -> None:
    folder = settings.STORAGE_DIR / dataset_id
    if folder.exists():
        for item in sorted(folder.rglob("*"), reverse=True):
            item.unlink() if item.is_file() else item.rmdir()
        folder.rmdir()


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
