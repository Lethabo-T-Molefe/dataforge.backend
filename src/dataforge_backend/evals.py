"""Reproducible evaluation reports: each run writes a JSON file and a Markdown summary under evals/<suite>/.

Every report records how it was made (command, seed, commit, library versions), so a figure quoted in the
documentation can be traced to the run that produced it and produced again.
"""

from __future__ import annotations

import json
import platform
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Not imported from devtools: the anomaly backtest needs no database stack.
REPO = Path(__file__).resolve().parents[2]
EVALS = REPO / "evals"


def _commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def provenance(command: str, **extra: Any) -> dict[str, Any]:
    return {
        "command": command,
        "generatedAt": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": _commit(),
        "python": platform.python_version(),
        **extra,
    }


def write(suite: str, name: str, data: dict[str, Any], markdown: str, out: Path | None = None) -> Path:
    folder = (out or EVALS) / suite
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    base = folder / f"{name}-{stamp}"
    base.with_suffix(".json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    base.with_suffix(".md").write_text(markdown, encoding="utf-8")
    return base


# ------------------------------------------------------------------ anomaly detection


def anomaly_backtest(synthetic: int = 200) -> tuple[dict[str, Any], str]:
    """The robust detector against the band rule: on the showcase's planted anomaly and on synthetic series."""
    from .domain import anomaly
    from .seed_data import learners

    rows = learners.monthly()
    planted = learners.ANOMALY
    by_province: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    overall: dict[str, float] = defaultdict(float)
    for row in rows:
        by_province[str(row["province"])][str(row["month"])] += float(row["dropouts"])
        overall[str(row["month"])] += float(row["dropouts"])
    showcase: list[dict[str, Any]] = []
    totals = {"robust": [0, 0, 0], "band": [0, 0, 0]}
    for name, series in [("All provinces", overall), *sorted(by_province.items())]:
        truth = {str(planted["month"])} if name == planted["province"] else set()
        scores = anomaly.backtest(sorted(series.items()), truth, start=24)
        entry: dict[str, Any] = {"series": name, "planted": sorted(truth)}
        for method, score in scores.items():
            entry[method] = score.to_dict()
            totals[method][0] += score.true_positives
            totals[method][1] += score.false_positives
            totals[method][2] += score.false_negatives
        showcase.append(entry)
    bench = anomaly.benchmark(synthetic)
    data = {
        "provenance": provenance(
            f"dataforge backtest-anomalies --synthetic {synthetic}", seed="0..{synthetic}"
        ),
        "settings": {"k": anomaly.DEFAULT_K, "bandK": 3.0, "bandWindow": 12, "judgedFrom": "month 25"},
        "showcase": {
            "metric": "Monthly drop-outs by province (learner-progress-monthly)",
            "planted": planted,
            "series": showcase,
            "totals": {
                m: {"truePositives": t[0], "falsePositives": t[1], "falseNegatives": t[2]}
                for m, t in totals.items()
            },
        },
        "synthetic": bench,
    }
    lines = [
        "# Anomaly detection backtest",
        "",
        "Each month is judged only on the months before it, as a monitor would see it. The robust detector",
        "(seasonal-naive with drift and a MAD spread, k = 3.5) is compared with the older band rule",
        "(mean ± 3 standard deviations of the last 12 months).",
        "",
        f"Generated {data['provenance']['generatedAt']} at commit {data['provenance']['commit'] or 'unknown'}.",
        "",
        "## Synthetic series",
        "",
        f"{bench['series']} seeded series with trend, seasonality and noise; one or two planted spikes or dips each.",
        "",
        "| Method | True positives | False positives | Missed | Precision | Recall |",
        "|---|---|---|---|---|---|",
    ]
    for method, label in (("robust", "Robust seasonal detector"), ("band", "Band rule")):
        m = bench[method]
        lines.append(
            f"| {label} | {m['truePositives']} | {m['falsePositives']} | {m['falseNegatives']} | "
            f"{m['precision']:.3f} | {m['recall']:.3f} |"
        )
    lines += [
        "",
        "## Showcase: monthly drop-outs by province",
        "",
        f"Planted: {planted['month']}, {planted['province']}, {planted['seta']} ({planted['reason']})",
        "",
        "| Series | Robust flagged | Band flagged |",
        "|---|---|---|",
    ]
    for entry in showcase:
        lines.append(
            f"| {entry['series']} | {', '.join(entry['robust']['flagged']) or '-'} | "
            f"{', '.join(entry['band']['flagged']) or '-'} |"
        )
    return data, "\n".join(lines) + "\n"
