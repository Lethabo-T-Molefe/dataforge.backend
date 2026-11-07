"""A dataset overview for the assistant: what a dataset holds, built from its stored profile.

Asked "what is this dataset about?", the assistant describes the dataset instead of refusing for want of a
metric. Everything here comes from the profile taken when the data was brought in, so no query runs and no row
is read. Personal columns appear only by name, marked as masked: their values never leave the profile.
"""

import re
from typing import Any

from .values import thousands

_OVERVIEW = re.compile(
    r"\b("
    r"about|describe|description|summar(y|ise|ize)|overview|explain (this|the|my) (data|dataset|table|file)"
    r"|tell me (about|more)|what('s| is| are)? in (this|the|my|it)|what (does|do) (this|the|it)\b.*\b(hold|contain|show|have)"
    r"|what (columns|fields|variables|data)|which (columns|fields|variables)|explore|insights?|key (facts|findings)"
    r"|what can (you|i) (tell|learn|ask|find)|interesting|look like|structure|schema"
    r")\b",
    re.IGNORECASE,
)
_NUMERIC = ("integer", "decimal", "numeric")


def wants_overview(question: str) -> bool:
    """Whether a question asks what a dataset is or holds, rather than for a number."""
    return bool(_OVERVIEW.search(question))


def _kind(entry: dict[str, Any]) -> str:
    if entry.get("masked"):
        return "Personal (masked)"
    inferred = str(entry.get("inferredType") or "string")
    return {
        "integer": "Whole number",
        "decimal": "Number",
        "date": "Date",
        "boolean": "Yes / no",
        "mixed": "Mixed",
        "empty": "Empty",
    }.get(inferred, "Text")


def _summary(entry: dict[str, Any]) -> str:
    if entry.get("masked"):
        return f"{str(entry.get('piiKind') or 'personal').replace('_', ' ')} information, masked for analysis"
    parts: list[str] = []
    if entry.get("inferredType") in _NUMERIC and entry.get("min") is not None:
        parts.append(f"{entry['min']} to {entry['max']}, average {entry.get('mean')}")
    else:
        top = [v for v in entry.get("topValues") or [] if v.get("value") not in (None, "")][:3]
        if top:
            parts.append(", ".join(f"{v['value']} ({thousands(int(v['count']))})" for v in top))
        if entry.get("distinct"):
            parts.append(f"{thousands(int(entry['distinct']))} different values")
    if entry.get("nullPct"):
        parts.append(f"{entry['nullPct']}% blank")
    return "; ".join(parts) or "—"


def overview_table(profile: dict[str, Any]) -> dict[str, Any]:
    rows = [[str(entry["name"]), _kind(entry), _summary(entry)] for entry in profile.get("columns") or []]
    return {"columns": ["Column", "Type", "What it holds"], "rows": rows}


def schema_prompt(name: str, description: str, profile: dict[str, Any]) -> str:
    """What the model may see to describe the dataset: names, types and the column summaries above."""
    lines = [f"Dataset: {name}", f"Rows: {profile.get('rows', 0)}"]
    if description:
        lines.append(f"Owner's description: {description[:500]}")
    for entry in profile.get("columns") or []:
        lines.append(f"- {entry['name']} ({_kind(entry)}): {_summary(entry)}")
    return "\n".join(lines)


def _names(entries: list[dict[str, Any]], limit: int = 4) -> str:
    names = [str(e["name"]).replace("_", " ") for e in entries[:limit]]
    more = len(entries) - limit
    if more > 0:
        names.append(f"{more} more")
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def fallback_narrative(name: str, profile: dict[str, Any]) -> str:
    """A plain description when no model is available."""
    columns = profile.get("columns") or []
    numbers = [c for c in columns if c.get("inferredType") in _NUMERIC and not c.get("masked")]
    labels = [
        c
        for c in columns
        if c.get("inferredType") in ("string", "boolean")
        and not c.get("masked")
        and int(c.get("distinct") or 0) <= 60
    ]
    written = [
        c
        for c in columns
        if c.get("inferredType") == "string"
        and not c.get("masked")
        and int(c.get("distinct") or 0) > 60
        and int(c.get("lengthMax") or 0) >= 25
    ]
    dates = [c for c in columns if c.get("inferredType") == "date" and not c.get("masked")]
    parts = [f"{name} has {thousands(int(profile.get('rows', 0)))} rows and {len(columns)} columns."]
    if written:
        parts.append(f"People wrote free text in {_names(written, 2)}, which can be read for themes.")
    if labels:
        parts.append(f"Rows are described by {_names(labels)}.")
    if numbers:
        parts.append(f"It measures {_names(numbers)}.")
    if dates:
        parts.append(f"It is dated by {_names(dates, 2)}, so it can be followed over time.")
    return " ".join(parts)


def overview_facts(profile: dict[str, Any], pii: dict[str, str]) -> str:
    quality = profile.get("quality") or {}
    parts = [
        f"Measured when it was brought in: {quality.get('completeness', '—')}% of cells are filled and "
        f"{quality.get('conformance', '—')}% of values fit their column's type."
    ]
    if quality.get("duplicates"):
        parts.append(f"{thousands(int(quality['duplicates']))} rows repeat exactly.")
    if pii:
        parts.append(
            f"{', '.join(pii)} hold personal information, so they are masked and every answer describes groups."
        )
    return " ".join(parts)
