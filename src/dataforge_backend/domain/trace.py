"""A trace of the governance steps one answer went through, in order, for the member to inspect.

Each step says what was checked or done, whether it passed, and the evidence (the plan, the SQL, the sandbox
session, the audit event). A refused answer ends at the step that blocked it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

STATUSES = ("passed", "blocked", "info")


@dataclass
class Trace:
    steps: list[dict[str, Any]] = field(default_factory=list)

    def step(self, key: str, label: str, detail: str = "", status: str = "passed", **data: Any) -> Trace:
        if status not in STATUSES:
            raise ValueError(f"Unknown trace status {status!r}.")
        entry: dict[str, Any] = {"key": key, "label": label, "status": status, "detail": detail}
        evidence = {k: v for k, v in data.items() if v is not None}
        if evidence:
            entry["data"] = evidence
        self.steps.append(entry)
        return self

    @property
    def blocked(self) -> bool:
        return any(s["status"] == "blocked" for s in self.steps)

    def keys(self) -> list[str]:
        return [s["key"] for s in self.steps]

    def to_list(self) -> list[dict[str, Any]]:
        return [dict(s) for s in self.steps]
