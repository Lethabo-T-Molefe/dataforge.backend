"""Data profiling: the measurement step before anything is changed.

Produces the column profile, value shapes, how blanks happen (MCAR / MAR), functional dependencies,
candidate keys, orphan checks and the six quality dimensions shown to people before they clean or approve.
"""

import math
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .pii import PiiFinding, PiiKind, mask_value
from .values import (
    date_shape,
    fingerprint,
    is_sentinel,
    parse_bool,
    parse_date,
    parse_number,
    shape,
    thousands,
)

_SAMPLE_FOR_INFERENCE = 20_000
_TYPE_THRESHOLD = 0.9
_MIXED_THRESHOLD = 0.5


@dataclass
class ColumnFacts:
    """Intermediate facts about one column, reused by insights and cleaning defaults."""

    name: str
    inferred_type: str
    nulls: int
    null_pct: float
    distinct: int
    conformance: float
    numeric: pd.Series | None = None
    date_shapes: dict[str, int] = field(default_factory=dict)
    sentinels: int = 0
    outliers: int = 0
    pii_kind: str | None = None
    avg_length: float = 0.0
    variant_groups: int = 0


def _blank_mask(series: pd.Series) -> pd.Series:
    return series.map(lambda v: is_sentinel(v)).astype(bool)


def _fmt(value: float | None) -> str | None:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return None
    if abs(value) >= 1000 or value == int(value):
        return thousands(round(value))
    return f"{value:.2f}"


def infer_type(values: pd.Series) -> tuple[str, float, pd.Series | None]:
    """Infer a column's real type from its non-blank values. Returns (type, conformance, numeric values)."""
    non_blank = values[~_blank_mask(values)]
    if non_blank.empty:
        return "empty", 1.0, None
    sample = non_blank.sample(n=min(len(non_blank), _SAMPLE_FOR_INFERENCE), random_state=0)
    texts = sample.astype(str).str.strip()

    bools = texts.map(parse_bool)
    bool_share = float(bools.notna().mean())
    if bool_share >= _TYPE_THRESHOLD and texts.str.lower().nunique() <= 4:
        return "boolean", bool_share, None

    numbers = texts.map(parse_number)
    number_share = float(numbers.notna().mean())
    dates = texts.map(parse_date)
    date_share = float(dates.notna().mean())
    # Plain digit strings like 20260314 parse as numbers; a column of real dates wins over that.
    if date_share >= _TYPE_THRESHOLD and date_share >= number_share:
        return "date", date_share, None
    if number_share >= _TYPE_THRESHOLD:
        full = non_blank.astype(str).map(parse_number)
        parsed = full.dropna().astype(float)
        is_integer = bool(len(parsed)) and bool(np.all(np.equal(np.mod(parsed.to_numpy(), 1), 0)))
        # Identifiers (0821234567, a 13-digit ID number) are text even though they are all digits.
        leading_zero = float(texts.str.match(r"^0\d").mean())
        long_digits = float(texts.str.fullmatch(r"\d{9,}").mean())
        if leading_zero > 0.5 or long_digits > 0.9:
            return "string", 1.0, None
        return ("integer" if is_integer else "decimal"), number_share, full.astype(float)
    if max(number_share, date_share) >= _MIXED_THRESHOLD:
        return "mixed", max(number_share, date_share), None
    return "string", 1.0, None


def robust_z(numeric: pd.Series) -> pd.Series:
    """Modified z-score (Iglewicz and Hoaglin). When the MAD is zero the mean absolute deviation stands in."""
    x = numeric.dropna()
    if x.empty:
        return pd.Series(0.0, index=numeric.index)
    median = x.median()
    mad = float((x - median).abs().median())
    if mad > 0:
        return 0.6745 * (numeric - median).abs() / mad
    mean_ad = float((x - median).abs().mean())
    if mean_ad > 0:
        return (numeric - median).abs() / (1.253314 * mean_ad)
    return pd.Series(0.0, index=numeric.index)


def _outlier_mask(numeric: pd.Series) -> pd.Series:
    """Tukey's far fence (3 × IQR) or a modified z-score above 3.5. Both are robust to skewed data."""
    x = numeric.dropna()
    if len(x) < 8:
        return pd.Series(False, index=numeric.index)
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    far = (
        (numeric < q1 - 3 * iqr) | (numeric > q3 + 3 * iqr)
        if iqr > 0
        else pd.Series(False, index=numeric.index)
    )
    far = far | (robust_z(numeric) > 3.5)
    return far.fillna(False).astype(bool)


def _patterns(values: pd.Series) -> list[dict[str, Any]]:
    non_blank = values[~_blank_mask(values)].astype(str).str.strip()
    if non_blank.empty:
        return []
    sample = non_blank.head(_SAMPLE_FOR_INFERENCE)
    shapes = sample.map(shape)
    counts = shapes.value_counts()
    total = int(counts.sum())
    scale = len(non_blank) / max(total, 1)
    rows: list[dict[str, Any]] = []
    for pattern, count in counts.head(6).items():
        example = str(sample[shapes == pattern].iloc[0])
        rows.append(
            {
                "pattern": str(pattern)[:40],
                "example": example[:40],
                "count": int(round(int(count) * scale)),
                "rare": bool(int(count) / total < 0.01 and len(counts) > 1),
            }
        )
    return rows


def _missingness(frame: pd.DataFrame, column: str, blank: pd.Series) -> tuple[str, str, str | None]:
    """Classify how a column's blanks happen by testing the blank indicator against every other column."""
    null_share = float(blank.mean())
    if null_share == 0:
        return "none", "Complete.", None
    if null_share > 0.95:
        return "MNAR", "Almost always empty. Check with the source why it is collected at all.", None
    best: tuple[float, str, str] | None = None
    for other in frame.columns:
        if other == column:
            continue
        values = frame[other]
        other_blank = _blank_mask(values)
        observed = values[~other_blank].astype(str).str.strip()
        indicator = blank[~other_blank]
        if observed.empty or indicator.nunique() < 2:
            continue
        distinct = observed.nunique()
        try:
            if 2 <= distinct <= 30:
                table = pd.crosstab(indicator, observed)
                if table.shape[0] < 2 or table.shape[1] < 2:
                    continue
                result = stats.chi2_contingency(table)
                p_value = float(result.pvalue)
                rates = table.loc[True] / table.sum(axis=0) if True in table.index else None
                top = str(rates.idxmax()) if rates is not None else ""
                reason = f"{column} is blank more often when {other} is {top}."
            else:
                numeric = observed.map(parse_number)
                if numeric.notna().mean() < 0.9:
                    continue
                a = numeric[indicator.to_numpy()].dropna()
                b = numeric[~indicator.to_numpy()].dropna()
                if len(a) < 5 or len(b) < 5:
                    continue
                p_value = float(stats.mannwhitneyu(a, b).pvalue)
                direction = "higher" if a.median() > b.median() else "lower"
                reason = f"{column} is blank more often when {other} is {direction}."
        except ValueError:
            continue
        if p_value < 0.001 and (best is None or p_value < best[0]):
            best = (p_value, str(other), reason)
    if best:
        return "MAR", f"{best[2]} Treat the blanks as MAR: fill from the columns that explain them.", best[1]
    return "MCAR", "Blanks do not line up with any other column. Dropping or a simple fill is unbiased.", None


def littles_mcar(frame: pd.DataFrame, numeric_columns: dict[str, pd.Series]) -> dict[str, Any]:
    """Little's MCAR test over the numeric columns, using pairwise estimates of the mean and covariance."""
    names = [name for name, series in numeric_columns.items() if series.notna().sum() > 3][:8]
    if len(names) < 2:
        return {
            "statistic": "n/a",
            "pValue": "n/a",
            "reading": "Needs at least two numeric columns. Each column's blanks are tested on their own instead.",
        }
    data = pd.DataFrame({name: numeric_columns[name] for name in names})
    if not data.isna().any().any():
        return {
            "statistic": "0",
            "pValue": "1",
            "reading": "No numeric column has blanks, so there is nothing to test.",
        }
    mu = data.mean()
    sigma = data.cov()
    d2 = 0.0
    dof = 0
    pattern_keys = data.isna().apply(lambda row: tuple(row.to_numpy()), axis=1)
    for pattern, group in data.groupby(pattern_keys):
        observed = [name for name, missing in zip(names, pattern, strict=True) if not missing]
        if not observed:
            continue
        diff = (group[observed].mean() - mu[observed]).to_numpy()
        cov = sigma.loc[observed, observed].to_numpy()
        try:
            inverse = np.linalg.pinv(cov)
        except np.linalg.LinAlgError:
            continue
        d2 += len(group) * float(diff @ inverse @ diff)
        dof += len(observed)
    dof -= len(names)
    if dof <= 0:
        return {"statistic": "n/a", "pValue": "n/a", "reading": "Too few missingness patterns to test."}
    p_value = float(stats.chi2.sf(d2, dof))
    reading = (
        "We reject MCAR: blanks in the numeric columns relate to values we can see, so treat them as MAR or MNAR."
        if p_value < 0.05
        else "We cannot reject MCAR for the numeric columns: their blanks look random."
    )
    return {"statistic": f"χ² = {d2:.1f}, df = {dof}", "pValue": f"{p_value:.3f}", "reading": reading}


def _dependencies(frame: pd.DataFrame, candidates: list[str]) -> list[dict[str, Any]]:
    """Functional dependencies A → B that hold, or nearly hold, between low-cardinality columns."""
    results: list[dict[str, Any]] = []
    n = len(frame)
    for left, right in combinations(candidates[:12], 2):
        for a, b in ((left, right), (right, left)):
            pair = frame[[a, b]].dropna()
            if pair.empty or pair[a].nunique() >= len(pair):
                continue
            # Any column "determines" a column that barely varies (a flag that is almost always False).
            # Such a rule says nothing, and repairing to it would overwrite the few values that differ.
            tolerance = max(1, int(0.01 * n))
            if len(pair) - int(pair[b].value_counts().iloc[0]) <= tolerance:
                continue
            modes = pair.groupby(a)[b].agg(lambda s: s.value_counts().index[0])
            violations = int((pair[b] != pair[a].map(modes)).sum())
            if violations == 0:
                results.append(
                    {
                        "left": a,
                        "right": b,
                        "hold": True,
                        "support": f"holds on all {thousands(n)} rows",
                        "violations": 0,
                    }
                )
            elif violations <= tolerance:
                results.append(
                    {
                        "left": a,
                        "right": b,
                        "hold": False,
                        "support": f"nearly holds: {violations} rows disagree",
                        "violations": violations,
                    }
                )
    results.sort(key=lambda item: (not item["hold"], item["violations"]))
    return results[:8]


def _candidate_keys(frame: pd.DataFrame, candidates: list[str]) -> list[str]:
    n = len(frame)
    keys = [c for c in frame.columns if frame[c].notna().all() and frame[c].nunique() == n]
    if keys:
        return [str(k) for k in keys[:3]]
    best: tuple[int, tuple[str, ...]] | None = None
    for size in (2, 3):
        for combo in combinations(candidates[:8], size):
            duplicates = n - len(frame.drop_duplicates(subset=list(combo)))
            if best is None or duplicates < best[0]:
                best = (duplicates, combo)
        if best and best[0] == 0:
            break
    if best and best[0] <= max(1, int(0.02 * n)):
        label = " + ".join(best[1])
        return [label if best[0] == 0 else f"{label} (after deduplication, {best[0]} clashes)"]
    return []


def _variant_groups(values: pd.Series) -> int:
    """How many labels have a spelling variant that means the same thing (GP / gp / G.P.)."""
    observed = values[~_blank_mask(values)].astype(str).str.strip()
    distinct = observed.unique()
    if len(distinct) > 300:
        return 0
    keys = Counter(fingerprint(v) for v in distinct)
    return sum(1 for count in keys.values() if count > 1)


def profile_frame(
    frame: pd.DataFrame,
    pii: list[PiiFinding] | None = None,
    references: dict[str, tuple[str, set[str]]] | None = None,
) -> tuple[dict[str, Any], dict[str, ColumnFacts]]:
    """Profile a raw table. Returns the JSON profile shown to people and per-column facts for later steps."""
    pii_by_column = {finding.column: finding for finding in pii or []}
    n = len(frame)
    columns: list[dict[str, Any]] = []
    facts: dict[str, ColumnFacts] = {}
    numeric_columns: dict[str, pd.Series] = {}
    blank_cells = 0
    conforming = 0
    counted = 0

    for column in frame.columns:
        name = str(column)
        series = frame[column]
        blank = _blank_mask(series)
        sentinels = int((blank & series.notna() & (series.astype(str).str.strip() != "")).sum())
        nulls = int(blank.sum())
        blank_cells += nulls
        inferred, conformance, numeric = infer_type(series)
        non_blank = series[~blank].astype(str).str.strip()
        distinct = int(non_blank.nunique())
        conforming += int(round(conformance * (n - nulls)))
        counted += n - nulls
        finding = pii_by_column.get(name)

        column_facts = ColumnFacts(
            name=name,
            inferred_type=inferred,
            nulls=nulls,
            null_pct=round(100 * nulls / n, 1) if n else 0.0,
            distinct=distinct,
            conformance=conformance,
            numeric=numeric,
            sentinels=sentinels,
            pii_kind=finding.kind.value if finding else None,
            avg_length=float(non_blank.str.len().mean()) if not non_blank.empty else 0.0,
        )
        entry: dict[str, Any] = {
            "name": name,
            "declaredType": "text",
            "inferredType": inferred,
            "nulls": nulls,
            "nullPct": column_facts.null_pct,
            "distinct": distinct,
            "conformancePct": round(conformance * 100, 1),
            "sentinels": sentinels,
            "masked": finding is not None,
            "piiKind": finding.kind.value if finding else None,
        }
        if not non_blank.empty:
            entry["lengthMin"] = int(non_blank.str.len().min())
            entry["lengthMax"] = int(non_blank.str.len().max())
        if numeric is not None and numeric.notna().any():
            numeric_columns[name] = numeric
            valid = numeric.dropna()
            entry.update(
                {
                    "min": _fmt(float(valid.min())),
                    "max": _fmt(float(valid.max())),
                    "mean": _fmt(float(valid.mean())),
                    "median": _fmt(float(valid.median())),
                    "std": _fmt(float(valid.std())) if len(valid) > 1 else None,
                }
            )
            column_facts.outliers = int(_outlier_mask(numeric).sum())
            entry["outliers"] = column_facts.outliers
        if inferred in ("date", "mixed"):
            shapes = Counter(s for s in non_blank.head(_SAMPLE_FOR_INFERENCE).map(date_shape) if s)
            column_facts.date_shapes = dict(shapes)
            entry["dateFormats"] = dict(shapes.most_common())
        if inferred == "string" and distinct <= 300:
            column_facts.variant_groups = _variant_groups(series)
            entry["labelVariants"] = column_facts.variant_groups
        top = non_blank.value_counts().head(5)
        entry["topValues"] = [
            {"value": "(masked)" if finding else str(value)[:60], "count": int(count)}
            for value, count in top.items()
        ]
        entry["patterns"] = [] if finding else _patterns(series)
        if 0 < nulls < n:
            missingness, note, depends_on = _missingness(frame, name, blank)
        elif nulls == 0:
            missingness, note, depends_on = "none", "Complete.", None
        else:
            missingness, note, depends_on = "MNAR", "Empty in every row.", None
        entry["missingness"] = missingness
        entry["missingNote"] = note
        entry["missingDependsOn"] = depends_on
        columns.append(entry)
        facts[name] = column_facts

    low_card = [
        c.name
        for c in facts.values()
        if c.pii_kind is None and 2 <= c.distinct <= min(5000, max(2, n // 2)) and c.avg_length <= 40
    ]
    key_candidates = [c.name for c in facts.values() if c.avg_length <= 40]
    duplicate_rows = n - len(frame.drop_duplicates())
    outliers_total = sum(c.outliers for c in facts.values())

    cross_table: list[dict[str, Any]] = []
    for column, (reference_name, keys) in (references or {}).items():
        if column not in frame.columns or not keys:
            continue
        values = frame[column][~_blank_mask(frame[column])].astype(str).str.strip()
        orphans = int((~values.isin(keys)).sum())
        pct = round(100 * orphans / max(len(values), 1), 1)
        cross_table.append(
            {
                "reference": reference_name,
                "parent": reference_name,
                "child": column,
                "orphanPct": pct,
                "orphans": orphans,
                "note": f"{thousands(orphans)} rows point at a {column} that is not in {reference_name}."
                if orphans
                else f"Every {column} exists in {reference_name}.",
            }
        )

    total_cells = n * max(len(frame.columns), 1)
    completeness = round(100 * (1 - blank_cells / total_cells)) if total_cells else 100
    conformance_pct = round(100 * conforming / counted) if counted else 100
    mixed_dates = [c.name for c in facts.values() if len(c.date_shapes) > 1]
    variants = sum(c.variant_groups for c in facts.values())
    inconsistencies = len(mixed_dates) + (1 if variants else 0)
    consistency = "Strong" if inconsistencies == 0 else ("Fair" if inconsistencies == 1 else "Weak")
    orphan_pct = max((item["orphanPct"] for item in cross_table), default=None)

    most_empty = sorted(facts.values(), key=lambda c: c.null_pct, reverse=True)
    empty_note = (
        f"{', '.join(c.name for c in most_empty[:2] if c.null_pct > 0)} drive most of the gap."
        if most_empty and most_empty[0].null_pct > 0
        else "Every cell is filled."
    )
    low_conformance = [c.name for c in facts.values() if c.conformance < 0.98 and c.inferred_type != "string"]
    outlier_columns = [c.name for c in facts.values() if c.outliers]
    dimensions = [
        {"name": "Completeness", "score": f"{completeness}%", "note": empty_note},
        {
            "name": "Validity",
            "score": f"{conformance_pct}%",
            "note": f"{', '.join(low_conformance[:3])} have values that fail their type."
            if low_conformance
            else "Values fit their types.",
        },
        {
            "name": "Uniqueness",
            "score": f"{thousands(duplicate_rows)} clashes" if duplicate_rows else "No repeats",
            "note": "Whole rows repeat exactly." if duplicate_rows else "No row appears twice.",
        },
        {
            "name": "Consistency",
            "score": consistency,
            "note": "; ".join(
                part
                for part in (
                    f"{', '.join(mixed_dates)} mix date formats" if mixed_dates else "",
                    f"{variants} label groups have spelling variants" if variants else "",
                )
                if part
            )
            or "Formats and labels are consistent.",
        },
        {
            "name": "Accuracy",
            "score": f"{thousands(outliers_total)} extremes" if outliers_total else "No extremes",
            "note": f"Far outside the usual range in {', '.join(outlier_columns[:3])}."
            if outlier_columns
            else "No value sits far from the rest.",
        },
        {
            "name": "Integrity",
            "score": f"{orphan_pct}% orphans" if orphan_pct is not None else "Not checked",
            "note": "IDs checked against related tables."
            if cross_table
            else "No related table shares a key column.",
        },
    ]

    profile = {
        "rows": n,
        "columns": columns,
        "littlesMcar": littles_mcar(frame, numeric_columns),
        "dependencies": _dependencies(frame, low_card),
        "candidateKeys": _candidate_keys(frame, key_candidates),
        "crossTable": cross_table,
        "dimensions": dimensions,
        "quality": {
            "completeness": completeness,
            "conformance": conformance_pct,
            "duplicates": duplicate_rows,
            "outliers": outliers_total,
        },
    }
    return profile, facts


def column_summaries(frame: pd.DataFrame, profile: dict[str, Any]) -> list[dict[str, Any]]:
    """The short per-column summary shown on a dataset's page (masked sample included)."""
    summaries = []
    for entry in profile["columns"]:
        name = entry["name"]
        series = frame[name] if name in frame.columns else pd.Series(dtype=object)
        sample_values = [v for v in series.dropna().astype(str).str.strip() if v and not is_sentinel(v)]
        sample = sample_values[0] if sample_values else ""
        if entry["masked"] and sample:
            sample = mask_value(sample, entry["piiKind"] or PiiKind.NAME) or ""
        summaries.append(
            {
                "name": name,
                "type": entry["inferredType"],
                "missingPct": entry["nullPct"],
                "distinct": entry["distinct"],
                "masked": entry["masked"],
                "sample": sample[:40],
            }
        )
    return summaries
