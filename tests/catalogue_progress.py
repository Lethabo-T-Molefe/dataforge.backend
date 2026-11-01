"""Marks an edge-case id as checked, and writes the checklist next to edge_case.md.

Covered ids accumulate in a small file so a unit run and an integration run can both tick boxes.
"""

from __future__ import annotations

import asyncio
import functools
import json
import re
from pathlib import Path

COVERED: set[str] = set()
ROOT = Path(__file__).resolve().parents[2]
EDGE = ROOT / "edge_case.md"
CHECKLIST = ROOT / "edge_case_checklist.md"
STATE = Path(__file__).resolve().parent / ".edge_covered.json"
_CASE = re.compile(r"^\*\*([A-Z]+-\d+)\.\s*(.+)$")
_SECTION = re.compile(r"^##\s+(.+)$")


def covered(*ids: str) -> None:
    COVERED.update(ids)


def marks(*ids: str):
    """Decorator: record the ids after the test function returns."""

    def deco(fn):
        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrap(*args, **kwargs):
                result = await fn(*args, **kwargs)
                covered(*ids)
                return result

            return async_wrap

        @functools.wraps(fn)
        def wrap(*args, **kwargs):
            result = fn(*args, **kwargs)
            covered(*ids)
            return result

        return wrap

    return deco


def _load() -> set[str]:
    if not STATE.exists():
        return set()
    try:
        return set(json.loads(STATE.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return set()


def _save(ids: set[str]) -> None:
    STATE.write_text(json.dumps(sorted(ids), indent=0), encoding="utf-8")


def write_checklist(*, reset: bool = False) -> None:
    known = set() if reset else _load()
    known |= COVERED
    _save(known)

    sections: list[tuple[str, list[tuple[str, str]]]] = []
    current = "Catalogue"
    bucket: list[tuple[str, str]] = []
    for line in EDGE.read_text(encoding="utf-8").splitlines():
        section = _SECTION.match(line)
        if section:
            if bucket:
                sections.append((current, bucket))
                bucket = []
            current = section.group(1)
            continue
        case = _CASE.match(line)
        if case:
            label = case.group(2).strip().removesuffix("**").strip()
            if len(label) > 140:
                label = label[:137] + "..."
            bucket.append((case.group(1), label))
    if bucket:
        sections.append((current, bucket))

    total = sum(len(items) for _, items in sections)
    done = sum(1 for _, items in sections for case_id, _ in items if case_id in known)
    percent = round(100 * done / total) if total else 0
    lines = [
        "# Edge-case checklist",
        "",
        "A box is checked when a test for that case has passed in a recent pytest run.",
        "Open boxes are still untested. Re-run the backend tests to refresh this file.",
        "",
        f"**{done} / {total} checked ({percent}%)**",
        "",
    ]
    for title, items in sections:
        section_done = sum(1 for case_id, _ in items if case_id in known)
        lines.append(f"## {title} — {section_done}/{len(items)}")
        lines.append("")
        for case_id, label in items:
            mark = "x" if case_id in known else " "
            lines.append(f"- [{mark}] **{case_id}** — {label}")
        lines.append("")
    CHECKLIST.write_text("\n".join(lines), encoding="utf-8")
