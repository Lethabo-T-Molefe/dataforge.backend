"""Automated insights: patterns, quality issues and anomalies surfaced before anyone asks.

Detection is unsupervised, so it works on data it has never seen. Each insight carries a suggested next step.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import pandas as pd

from .profiling import ColumnFacts
from .values import thousands


@dataclass(frozen=True)
class Insight:
    kind: str  # pattern | quality | anomaly
    title: str
    body: str
    suggestion: str
    action: str  # clean | chart | ask | monitor
    target: str | None = None  # cleaning chapter or column the suggestion acts on

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "title": self.title,
            "body": self.body,
            "suggestion": self.suggestion,
            "action": self.action,
            "target": self.target,
        }


def _pct(value: float) -> str:
    return f"{value:.0f}%" if value >= 10 else f"{value:.1f}%".replace(".0%", "%")


def detect_insights(
    frame: pd.DataFrame, profile: dict[str, Any], facts: dict[str, ColumnFacts]
) -> list[Insight]:
    insights: list[Insight] = []
    n = max(profile["rows"], 1)

    for column in facts.values():
        if len(column.date_shapes) > 1:
            shapes = ", ".join(list(column.date_shapes)[:3])
            empty = (
                f"{_pct(column.null_pct)} of {column.name} values are missing and the rest"
                if column.nulls
                else f"{column.name} values"
            )
            insights.append(
                Insight(
                    "quality",
                    f"{column.name} uses {len(column.date_shapes)} different date formats",
                    f"{empty} mix {shapes}.",
                    "Standardise the date format",
                    "clean",
                    "formats",
                )
            )

    for column in sorted(facts.values(), key=lambda c: c.null_pct, reverse=True):
        if column.null_pct >= 5 and column.pii_kind is None:
            insights.append(
                Insight(
                    "quality",
                    f"{_pct(column.null_pct)} of {column.name} is empty",
                    f"{thousands(column.nulls)} rows have no {column.name}. How you fill them depends on why they are empty.",
                    "Handle the missing values",
                    "clean",
                    "missing",
                )
            )
            break

    wrong_type = [c for c in facts.values() if c.inferred_type == "mixed"]
    if wrong_type:
        names = ", ".join(c.name for c in wrong_type[:3])
        insights.append(
            Insight(
                "quality",
                f"{names} {'mixes' if len(wrong_type) == 1 else 'mix'} numbers or dates with text",
                "Some values cannot be read as their real type, so sums and date filters would skip them.",
                "Give each column a real type",
                "clean",
                "types",
            )
        )

    for column in facts.values():
        if column.pii_kind or column.distinct < 2 or column.distinct > 30 or column.inferred_type != "string":
            continue
        counts = frame[column.name].dropna().astype(str).str.strip().value_counts()
        if counts.empty:
            continue
        share = 100 * counts.iloc[0] / n
        if share >= 35:
            blank = (
                f", and {_pct(column.null_pct)} of rows have no {column.name} at all"
                if column.null_pct >= 5
                else ""
            )
            insights.append(
                Insight(
                    "pattern",
                    f"{counts.index[0]} accounts for {_pct(share)} of rows",
                    f"{column.name} is heavily concentrated{blank}.",
                    f"Chart rows by {column.name}",
                    "chart",
                    column.name,
                )
            )
            break

    numeric = {
        c.name: c.numeric for c in facts.values() if c.numeric is not None and c.numeric.notna().sum() > 10
    }
    best: tuple[float, str, str] | None = None
    for a, b in combinations(list(numeric)[:10], 2):
        pair = pd.DataFrame({"a": numeric[a], "b": numeric[b]}).dropna()
        if len(pair) < 10 or pair["a"].nunique() < 3 or pair["b"].nunique() < 3:
            continue
        r = float(pair["a"].corr(pair["b"], method="spearman"))
        if abs(r) >= 0.7 and (best is None or abs(r) > abs(best[0])):
            best = (r, a, b)
    if best:
        r, a, b = best
        insights.append(
            Insight(
                "pattern",
                f"{a} and {b} move together",
                f"Their rank correlation is {r:.2f}. That is a strong {'positive' if r > 0 else 'negative'} relationship, not proof of cause.",
                "Test the relationship",
                "chart",
                f"{a},{b}",
            )
        )

    duplicates = profile["quality"]["duplicates"]
    if duplicates:
        insights.append(
            Insight(
                "anomaly",
                f"{thousands(duplicates)} repeated rows",
                "Whole rows appear more than once, which inflates counts and totals.",
                "Remove duplicate rows",
                "clean",
                "duplicates",
            )
        )

    extreme = max(facts.values(), key=lambda c: c.outliers, default=None)
    if extreme and extreme.outliers:
        valid = extreme.numeric.dropna() if extreme.numeric is not None else pd.Series(dtype=float)
        biggest = f" The largest is {thousands(round(valid.max()))}." if not valid.empty else ""
        insights.append(
            Insight(
                "anomaly",
                f"{extreme.outliers} {extreme.name} values sit far from the rest",
                f"They are outside the robust fence (3 × IQR or a modified z-score above 3.5).{biggest}",
                "Check values that sit far away",
                "clean",
                "outliers",
            )
        )

    return insights[:6]
