"""Cleaning operations: each is a recorded, parameterised transform from one table to the next.

Operations never touch bronze. They run on a copy, and the saved recipe replays them in order to reproduce
the same silver table from the same input.
"""

import json
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..pii import PiiKind, mask_value, pseudonymise
from ..profiling import robust_z
from ..values import (
    NUMERIC_SENTINELS,
    SENTINELS,
    fingerprint,
    is_blank,
    is_sentinel,
    jaro_winkler,
    normalise_za_phone,
    parse_bool,
    parse_date,
    parse_number,
    sa_id_valid,
    thousands,
)


class OperationError(ValueError):
    """Raised when a step's parameters do not fit the table (for example, an unknown column)."""


@dataclass
class CleaningContext:
    pii_columns: dict[str, str] = field(default_factory=dict)
    # Reference key sets for referential checks, keyed "datasetId.column".
    references: dict[str, set[str]] = field(default_factory=dict)
    seed: int = 0
    salt: str = "dataforge"


@dataclass
class StepOutcome:
    frame: pd.DataFrame
    affected: list[str]
    summary: str
    rows_before: int
    rows_after: int
    cells_changed: int


Operation = Callable[[pd.DataFrame, dict[str, Any], CleaningContext], StepOutcome]
OPERATIONS: dict[str, Operation] = {}


def operation(name: str) -> Callable[[Operation], Operation]:
    def register(fn: Operation) -> Operation:
        OPERATIONS[name] = fn
        return fn

    return register


# ------------------------------------------------------------------ helpers


def _columns(frame: pd.DataFrame, params: dict[str, Any], key: str = "columns") -> list[str]:
    value = params.get(key) or []
    names = [value] if isinstance(value, str) else list(value)
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise OperationError(f"Unknown column: {', '.join(missing)}")
    return names


def _column(frame: pd.DataFrame, params: dict[str, Any], key: str = "column") -> str:
    name = params.get(key)
    if not name or name not in frame.columns:
        raise OperationError(f"Choose a column for '{key}'. {name!r} is not in this table.")
    return str(name)


def _missing(series: pd.Series) -> pd.Series:
    return series.map(lambda v: is_sentinel(v)).astype(bool)


def _changed(before: pd.Series, after: pd.Series) -> int:
    b = before.map(lambda v: None if is_blank(v) else str(v))
    a = after.map(lambda v: None if is_blank(v) else str(v))
    return int((b.fillna("\0") != a.fillna("\0")).sum())


def _outcome(before: pd.DataFrame, after: pd.DataFrame, affected: list[str], summary: str) -> StepOutcome:
    changed = 0
    if len(before) == len(after):
        for column in affected:
            if column in before.columns and column in after.columns:
                changed += _changed(
                    before[column].reset_index(drop=True), after[column].reset_index(drop=True)
                )
            elif column in after.columns:
                changed += int(after[column].notna().sum())
    return StepOutcome(after, affected, summary, len(before), len(after), changed)


def _text(series: pd.Series) -> pd.Series:
    """Values as Python strings, with blanks as "". (In pandas 3, astype(str) keeps missing values missing.)"""
    return series.map(lambda v: "" if is_blank(v) else str(v)).astype(object)


def _numeric(series: pd.Series) -> pd.Series:
    return series.map(parse_number).astype(float)


def _numeric_share(series: pd.Series) -> float:
    observed = series[~_missing(series)]
    return float(_numeric(observed).notna().mean()) if len(observed) else 0.0


def _as_object(values: Any, index: pd.Index) -> pd.Series:
    return pd.Series(values, index=index, dtype=object)


# ------------------------------------------------------------------ profile


@operation("accept_profile")
def accept_profile(frame: pd.DataFrame, params: dict[str, Any], ctx: CleaningContext) -> StepOutcome:
    return StepOutcome(frame, [], "Profile accepted. Nothing was changed.", len(frame), len(frame), 0)


# ------------------------------------------------------------------ missing values


def _fill_numeric_models(
    frame: pd.DataFrame, columns: list[str], strategy: str, params: dict[str, Any], ctx: CleaningContext
) -> pd.DataFrame:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer, KNNImputer
    from sklearn.preprocessing import StandardScaler

    predictors = [
        c
        for c in frame.columns
        if c not in columns and _numeric(frame[c]).notna().mean() >= 0.9 and c not in ctx.pii_columns
    ][:10]
    numeric = pd.DataFrame({c: _numeric(frame[c]) for c in [*columns, *predictors]})
    usable = [c for c in numeric.columns if numeric[c].notna().any()]
    if not usable:
        return frame
    data = numeric[usable].to_numpy(dtype=float)
    if strategy == "knn":
        scaler = StandardScaler()
        filled = KNNImputer(n_neighbors=int(params.get("n_neighbors", 5))).fit_transform(
            scaler.fit_transform(data)
        )
        filled = scaler.inverse_transform(filled)
    elif strategy == "missforest":
        from sklearn.ensemble import RandomForestRegressor

        imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=30, max_depth=12, random_state=ctx.seed, n_jobs=1),
            max_iter=int(params.get("max_iter", 3)),
            random_state=ctx.seed,
        )
        filled = imputer.fit_transform(data)
    else:
        imputer = IterativeImputer(max_iter=int(params.get("max_iter", 10)), random_state=ctx.seed)
        filled = imputer.fit_transform(data)
    out = frame.copy()
    result = pd.DataFrame(filled, columns=usable, index=frame.index)
    for column in columns:
        if column in usable:
            original = numeric[column]
            out[column] = _as_object(
                np.where(original.isna(), result[column].round(4), original), frame.index
            )
    return out


def _fill_categorical_forest(frame: pd.DataFrame, column: str, ctx: CleaningContext) -> pd.Series:
    """MissForest-style fill for a category: a random forest predicts each blank from the other columns."""
    from sklearn.ensemble import RandomForestClassifier

    target = frame[column]
    missing = _missing(target)
    if missing.sum() == 0 or (~missing).sum() < 10:
        return target
    features = pd.DataFrame(index=frame.index)
    for other in frame.columns:
        if other == column or other in ctx.pii_columns:
            continue
        numeric = _numeric(frame[other])
        if numeric.notna().mean() >= 0.9:
            features[other] = numeric.fillna(numeric.median())
        elif frame[other].nunique() <= 50:
            codes = frame[other].map(lambda v: None if is_sentinel(v) else str(v).strip())
            features[other] = pd.Categorical(codes).codes
    if features.empty:
        return target
    model = RandomForestClassifier(n_estimators=30, max_depth=12, random_state=ctx.seed, n_jobs=1)
    model.fit(features[~missing], target[~missing].astype(str).str.strip())
    predicted = pd.Series(model.predict(features[missing]), index=target[missing].index)
    return _as_object(np.where(missing, predicted.reindex(target.index), target), target.index)


@operation("fill_missing")
def fill_missing(frame: pd.DataFrame, params: dict[str, Any], ctx: CleaningContext) -> StepOutcome:
    columns = _columns(frame, params)
    strategy = str(params.get("strategy", "constant"))
    add_indicator = bool(params.get("add_indicator", strategy in ("constant", "unknown")))
    out = frame.copy()
    affected: list[str] = []
    filled_total = 0

    for column in columns:
        missing = _missing(out[column])
        if add_indicator:
            out[f"{column}_was_missing"] = _as_object(missing.to_numpy(), out.index)
            affected.append(f"{column}_was_missing")
        filled_total += int(missing.sum())

    if strategy in ("knn", "iterative", "missforest"):
        # A column is numeric when its observed (non-blank) values parse as numbers.
        numeric_targets = [c for c in columns if _numeric_share(out[c]) >= 0.9]
        if numeric_targets:
            out = _fill_numeric_models(out, numeric_targets, strategy, params, ctx)
        if strategy == "missforest":
            for column in columns:
                if column not in numeric_targets:
                    out[column] = _fill_categorical_forest(out, column, ctx)
        elif set(columns) - set(numeric_targets):
            raise OperationError(f"{strategy} fills numeric columns. Use mode or MissForest for categories.")
    else:
        for column in columns:
            series = out[column]
            missing = _missing(series)
            if strategy in ("constant", "unknown"):
                value = params.get("value", "Unknown")
                out[column] = _as_object(np.where(missing, value, series), out.index)
            elif strategy in ("mean", "median"):
                numeric = _numeric(series)
                stat = float(numeric.mean() if strategy == "mean" else numeric.median())
                out[column] = _as_object(np.where(numeric.isna(), round(stat, 4), numeric), out.index)
            elif strategy == "mode":
                observed = series[~missing].astype(str).str.strip()
                if observed.empty:
                    continue
                out[column] = _as_object(np.where(missing, observed.mode().iloc[0], series), out.index)
            elif strategy == "group_median":
                group_by = _column(out, params, "group_by")
                numeric = _numeric(series)
                medians = numeric.groupby(_text(out[group_by])).transform("median")
                fallback = float(numeric.median())
                fill = medians.fillna(fallback)
                out[column] = _as_object(np.where(numeric.isna(), fill.round(4), numeric), out.index)
            elif strategy in ("ffill", "bfill"):
                order_by = params.get("order_by")
                fill_group = params.get("group_by")
                working = series.mask(missing)
                ordered = out.sort_values(order_by, kind="stable") if order_by else out
                values = working.loc[ordered.index]
                if fill_group:
                    grouped = values.groupby(_text(ordered[fill_group]), sort=False)
                    values = grouped.ffill() if strategy == "ffill" else grouped.bfill()
                else:
                    values = values.ffill() if strategy == "ffill" else values.bfill()
                out[column] = _as_object(values.reindex(out.index), out.index)
            else:
                raise OperationError(f"Unknown fill strategy: {strategy}")
    affected = [*columns, *affected]
    label = {"constant": f"'{params.get('value', 'Unknown')}'", "unknown": "'Unknown'"}.get(
        strategy, strategy.replace("_", " ")
    )
    return _outcome(
        frame,
        out,
        affected,
        f"Filled {thousands(filled_total)} blank cells in {', '.join(columns)} using {label}."
        if columns
        else "No column needed filling.",
    )


@operation("drop_missing_rows")
def drop_missing_rows(frame: pd.DataFrame, params: dict[str, Any], ctx: CleaningContext) -> StepOutcome:
    columns = _columns(frame, params) or list(frame.columns)
    how = params.get("how", "any")
    flags = pd.DataFrame({c: _missing(frame[c]) for c in columns})
    drop = flags.any(axis=1) if how == "any" else flags.all(axis=1)
    out = frame[~drop].copy()
    return _outcome(
        frame, out, [], f"Dropped {thousands(int(drop.sum()))} rows with blanks in {', '.join(columns)}."
    )


@operation("drop_sparse_columns")
def drop_sparse_columns(frame: pd.DataFrame, params: dict[str, Any], ctx: CleaningContext) -> StepOutcome:
    threshold = float(params.get("threshold", 0.5))
    sparse = [c for c in frame.columns if float(_missing(frame[c]).mean()) > threshold]
    out = frame.drop(columns=sparse)
    summary = (
        f"Dropped {', '.join(sparse)} (more than {threshold:.0%} empty)."
        if sparse
        else f"No column is more than {threshold:.0%} empty."
    )
    return _outcome(frame, out, sparse, summary)


# ------------------------------------------------------------------ repeats


@operation("deduplicate")
def deduplicate(frame: pd.DataFrame, params: dict[str, Any], ctx: CleaningContext) -> StepOutcome:
    subset = _columns(frame, params, "subset") or list(frame.columns)
    keep = params.get("keep", "first")
    order_by = params.get("order_by")
    working = frame.sort_values(order_by, kind="stable") if order_by and order_by in frame.columns else frame
    keys = pd.DataFrame(
        {c: working[c].map(lambda v: None if is_sentinel(v) else str(v).strip().lower()) for c in subset},
        index=working.index,
    )
    duplicated = keys.duplicated(keep="last" if keep == "last" else "first")
    out = working[~duplicated]
    out = out.loc[[i for i in frame.index if i in out.index]].copy()
    return _outcome(
        frame,
        out,
        subset,
        f"Removed {thousands(int(duplicated.sum()))} repeated rows (same {', '.join(subset)}), keeping the {keep} one.",
    )


@operation("link_records")
def link_records(frame: pd.DataFrame, params: dict[str, Any], ctx: CleaningContext) -> StepOutcome:
    columns = _columns(frame, params)
    if not columns:
        raise OperationError("Choose the columns that identify a record, such as a name.")
    threshold = float(params.get("threshold", 0.92))
    block_on = params.get("block_on")
    action = params.get("action", "flag")
    text = frame[columns].apply(_text).agg(" ".join, axis=1).str.lower().str.strip()
    blocks = _text(frame[block_on]) if block_on and block_on in frame.columns else text.str[:1]
    parent = {i: i for i in frame.index}

    def find(i: Any) -> Any:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for _, members in text.groupby(blocks):
        ids = list(members.index)[:600]
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                if jaro_winkler(text[ids[x]], text[ids[y]]) >= threshold:
                    parent[find(ids[y])] = find(ids[x])
    clusters = pd.Series({i: find(i) for i in frame.index})
    sizes = clusters.map(clusters.value_counts())
    out = frame.copy()
    if action == "merge":
        out = frame[clusters.to_numpy() == clusters.index.to_numpy()].copy()
        summary = f"Merged {thousands(len(frame) - len(out))} near-duplicate records into the first record of their cluster."
        return _outcome(frame, out, columns, summary)
    codes = pd.Categorical(clusters).codes
    out["record_cluster"] = _as_object(codes, frame.index)
    out["possible_duplicate"] = _as_object((sizes > 1).to_numpy(), frame.index)
    return _outcome(
        frame,
        out,
        ["record_cluster", "possible_duplicate"],
        f"Flagged {thousands(int((sizes > 1).sum()))} rows in {thousands(int(clusters[sizes > 1].nunique()))} likely-same clusters for review.",
    )


# ------------------------------------------------------------------ extreme values


def _outlier_flags(numeric: pd.Series, method: str, k: float | None) -> pd.Series:
    x = numeric.dropna()
    if len(x) < 5:
        return pd.Series(False, index=numeric.index)
    if method == "zscore":
        z = (numeric - x.mean()) / (x.std() or 1)
        return (z.abs() > (k or 3)).fillna(False)
    if method == "mad":
        return (robust_z(numeric) > (k or 3.5)).fillna(False)
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    fence = k or (3.0 if method == "combined" else 1.5)
    # With half the values identical the IQR is zero, and every other value would sit outside the fence.
    if iqr > 0:
        tukey = ((numeric < q1 - fence * iqr) | (numeric > q3 + fence * iqr)).fillna(False)
    else:
        tukey = pd.Series(False, index=numeric.index)
    if method == "combined":
        tukey = tukey | (robust_z(numeric) > 3.5).fillna(False)
    return tukey


@operation("handle_outliers")
def handle_outliers(frame: pd.DataFrame, params: dict[str, Any], ctx: CleaningContext) -> StepOutcome:
    columns = _columns(frame, params)
    method = params.get("method", "combined")
    action = params.get("action", "flag")
    k = params.get("k")
    out = frame.copy()
    affected: list[str] = []
    drop = pd.Series(False, index=frame.index)
    counts: list[str] = []
    for column in columns:
        numeric = _numeric(frame[column])
        flags = _outlier_flags(numeric, method, float(k) if k is not None else None)
        counts.append(f"{int(flags.sum())} in {column}")
        if action == "flag":
            out[f"{column}_outlier"] = _as_object(flags.to_numpy(), frame.index)
            affected.append(f"{column}_outlier")
        elif action == "cap_percentile":
            lower = float(params.get("lower", 0.01))
            upper = float(params.get("upper", 0.99))
            lo, hi = numeric.quantile(lower), numeric.quantile(upper)
            capped = numeric.clip(float(lo), float(hi))
            out[column] = _as_object(np.where(numeric.isna(), frame[column], capped.round(4)), frame.index)
            affected.append(column)
        elif action == "null":
            out[column] = _as_object(frame[column].mask(flags.astype(bool)), frame.index)
            affected.append(column)
        elif action == "drop":
            drop = drop | flags
        else:
            raise OperationError(f"Unknown outlier action: {action}")
    if action == "drop":
        out = out[~drop].copy()
    verb = {"flag": "Flagged", "cap_percentile": "Capped", "null": "Blanked", "drop": "Dropped rows for"}[
        action
    ]
    return _outcome(
        frame,
        out,
        affected or columns,
        f"{verb} extreme values: {', '.join(counts)}." if counts else "No numeric column chosen.",
    )


@operation("isolation_forest")
def isolation_forest(frame: pd.DataFrame, params: dict[str, Any], ctx: CleaningContext) -> StepOutcome:
    from sklearn.ensemble import IsolationForest

    columns = _columns(frame, params)
    if not columns:
        columns = [
            c for c in frame.columns if c not in ctx.pii_columns and _numeric(frame[c]).notna().mean() >= 0.9
        ]
    if not columns:
        raise OperationError("Isolation Forest needs at least one numeric column.")
    numeric = pd.DataFrame({c: _numeric(frame[c]) for c in columns})
    numeric = numeric.fillna(numeric.median())
    contamination = params.get("contamination", "auto")
    model = IsolationForest(
        contamination=contamination if contamination == "auto" else float(contamination),
        random_state=ctx.seed,
    )
    flags = pd.Series(model.fit_predict(numeric.to_numpy()) == -1, index=frame.index)
    out = frame.copy()
    if params.get("action", "flag") == "drop":
        out = out[~flags].copy()
        return _outcome(
            frame,
            out,
            [],
            f"Dropped {int(flags.sum())} rows the forest isolated across {', '.join(columns)}.",
        )
    out["multivariate_outlier"] = _as_object(flags.to_numpy(), frame.index)
    return _outcome(
        frame,
        out,
        ["multivariate_outlier"],
        f"Flagged {int(flags.sum())} rows that stand out across {', '.join(columns)}.",
    )


@operation("smooth_series")
def smooth_series(frame: pd.DataFrame, params: dict[str, Any], ctx: CleaningContext) -> StepOutcome:
    column = _column(frame, params)
    window = max(3, int(params.get("window", 5)))
    order_by = params.get("order_by")
    ordered = frame.sort_values(order_by, kind="stable") if order_by and order_by in frame.columns else frame
    numeric = _numeric(ordered[column])
    smoothed = numeric.rolling(window, center=True, min_periods=1).median()
    out = frame.copy()
    out[column] = _as_object(smoothed.round(4).reindex(frame.index), frame.index)
    return _outcome(
        frame, out, [column], f"Replaced spikes in {column} with a {window}-point rolling median."
    )


# ------------------------------------------------------------------ formats and types


@operation("parse_dates")
def parse_dates(frame: pd.DataFrame, params: dict[str, Any], ctx: CleaningContext) -> StepOutcome:
    columns = _columns(frame, params)
    day_first = bool(params.get("day_first", True))
    out = frame.copy()
    unreadable: list[str] = []
    for column in columns:
        parsed = frame[column].map(lambda v: parse_date(v, day_first=day_first))
        lost = int((parsed.isna() & ~_missing(frame[column])).sum())
        if lost:
            unreadable.append(f"{lost} in {column}")
        out[column] = _as_object(parsed.map(lambda d: d.isoformat() if d else None), frame.index)
    note = f" Could not read {', '.join(unreadable)}; those are now blank." if unreadable else ""
    return _outcome(
        frame,
        out,
        columns,
        f"Wrote {', '.join(columns)} as ISO dates ({'day' if day_first else 'month'} first).{note}",
    )


@operation("parse_numbers")
def parse_numbers(frame: pd.DataFrame, params: dict[str, Any], ctx: CleaningContext) -> StepOutcome:
    columns = _columns(frame, params)
    decimals = int(params.get("decimals", 2))
    out = frame.copy()
    unreadable: list[str] = []
    for column in columns:
        parsed = frame[column].map(parse_number)
        lost = int((parsed.isna() & ~_missing(frame[column])).sum())
        if lost:
            unreadable.append(f"{lost} in {column}")
        out[column] = _as_object(
            parsed.map(lambda v: None if v is None or np.isnan(v) else round(v, decimals)), frame.index
        )
    note = f" Could not read {', '.join(unreadable)}." if unreadable else ""
    return _outcome(
        frame,
        out,
        columns,
        f"Converted {', '.join(columns)} to numbers (currency symbols, spaces and decimal commas handled).{note}",
    )


@operation("normalize_text")
def normalize_text(frame: pd.DataFrame, params: dict[str, Any], ctx: CleaningContext) -> StepOutcome:
    columns = _columns(frame, params)
    case = params.get("case", "none")
    out = frame.copy()

    def clean(value: Any) -> Any:
        if is_blank(value):
            return None
        text = unicodedata.normalize("NFC", str(value))
        text = re.sub(r"\s+", " ", text).strip()
        if case == "lower":
            text = text.lower()
        elif case == "upper":
            text = text.upper()
        elif case == "title":
            text = text.title()
        return text

    for column in columns:
        out[column] = _as_object(frame[column].map(clean), frame.index)
    return _outcome(frame, out, columns, f"Trimmed spaces and normalised Unicode in {', '.join(columns)}.")


@operation("normalize_phones")
def normalize_phones(frame: pd.DataFrame, params: dict[str, Any], ctx: CleaningContext) -> StepOutcome:
    columns = _columns(frame, params)
    code = str(params.get("country_code", "27"))
    out = frame.copy()
    for column in columns:
        out[column] = _as_object(frame[column].map(lambda v: normalise_za_phone(v, code)), frame.index)
    return _outcome(frame, out, columns, f"Wrote {', '.join(columns)} in E.164 form (+{code}…).")


@operation("standardise_formats")
def standardise_formats(frame: pd.DataFrame, params: dict[str, Any], ctx: CleaningContext) -> StepOutcome:
    """Dates to ISO and money or number text to numbers, in one step."""
    date_columns = _columns(frame, params, "dates")
    number_columns = _columns(frame, params, "numbers")
    summaries: list[str] = []
    out = frame
    if date_columns:
        step = parse_dates(out, {"columns": date_columns, "day_first": params.get("day_first", True)}, ctx)
        out, summaries = step.frame, [*summaries, step.summary]
    if number_columns:
        step = parse_numbers(out, {"columns": number_columns, "decimals": params.get("decimals", 2)}, ctx)
        out, summaries = step.frame, [*summaries, step.summary]
    return _outcome(
        frame,
        out,
        [*date_columns, *number_columns],
        " ".join(summaries) or "No date or number column needed standardising.",
    )


@operation("normalise_text_and_phones")
def normalise_text_and_phones(
    frame: pd.DataFrame, params: dict[str, Any], ctx: CleaningContext
) -> StepOutcome:
    text_columns = _columns(frame, params, "text_columns")
    phone_columns = _columns(frame, params, "phone_columns")
    out = frame
    summaries: list[str] = []
    if text_columns:
        step = normalize_text(out, {"columns": text_columns, "case": params.get("case", "none")}, ctx)
        out, summaries = step.frame, [*summaries, step.summary]
    if phone_columns:
        step = normalize_phones(out, {"columns": phone_columns}, ctx)
        out, summaries = step.frame, [*summaries, step.summary]
    return _outcome(
        frame,
        out,
        [*text_columns, *phone_columns],
        " ".join(summaries) or "No text column needed normalising.",
    )


_TYPE_PARSERS: dict[str, Callable[[Any], Any]] = {
    "integer": lambda v: None if (n := parse_number(v)) is None else int(round(n)),
    "decimal": parse_number,
    "date": lambda v: None if (d := parse_date(v)) is None else d.isoformat(),
    "boolean": parse_bool,
    "string": lambda v: None if is_blank(v) else str(v).strip(),
}


@operation("coerce_types")
def coerce_types(frame: pd.DataFrame, params: dict[str, Any], ctx: CleaningContext) -> StepOutcome:
    types: dict[str, str] = dict(params.get("types") or {})
    sentinels = {str(s).strip().lower() for s in params.get("sentinels", sorted(SENTINELS))}
    if params.get("numeric_sentinels", False):
        sentinels |= NUMERIC_SENTINELS
    unknown = [c for c in types if c not in frame.columns]
    if unknown:
        raise OperationError(f"Unknown column: {', '.join(unknown)}")
    out = frame.copy()
    blanked = 0
    for column in frame.columns:
        text = frame[column].map(lambda v: None if is_blank(v) else str(v).strip().lower())
        mask = text.isin(sentinels) & text.notna()
        if column in types and types[column] == "string" and not params.get("clean_text_sentinels", True):
            mask = pd.Series(False, index=frame.index)
        blanked += int(mask.sum())
        out[column] = _as_object(frame[column].mask(mask.astype(bool)), frame.index)
    for column, kind in types.items():
        parser = _TYPE_PARSERS.get(kind)
        if parser is None:
            raise OperationError(f"Unknown type {kind!r}. Use integer, decimal, date, boolean or string.")
        out[column] = _as_object(out[column].map(parser), frame.index)
    typed = ", ".join(f"{c} → {t}" for c, t in types.items())
    return _outcome(
        frame,
        out,
        list(frame.columns),
        f"Turned {thousands(blanked)} placeholder values into blanks. {typed}."
        if typed
        else f"Turned {thousands(blanked)} placeholder values into blanks.",
    )


@operation("flatten_json")
def flatten_json(frame: pd.DataFrame, params: dict[str, Any], ctx: CleaningContext) -> StepOutcome:
    column = _column(frame, params)

    def load(value: Any) -> dict[str, Any]:
        if is_blank(value):
            return {}
        try:
            parsed = json.loads(str(value))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {"value": parsed}

    expanded = pd.json_normalize(frame[column].map(load).tolist()).add_prefix(f"{column}_")
    expanded.index = frame.index
    out = pd.concat([frame.drop(columns=[column]), expanded.astype(object)], axis=1)
    return _outcome(
        frame, out, list(expanded.columns), f"Split {column} into {len(expanded.columns)} columns."
    )


# ------------------------------------------------------------------ labels


@operation("map_values")
def map_values(frame: pd.DataFrame, params: dict[str, Any], ctx: CleaningContext) -> StepOutcome:
    column = _column(frame, params)
    mapping = {str(k).strip().lower(): v for k, v in dict(params.get("mapping") or {}).items()}
    other = params.get("other_label")
    keep = {str(v).strip().lower() for v in mapping.values()}
    out = frame.copy()

    def apply(value: Any) -> Any:
        if is_blank(value):
            return value
        key = str(value).strip().lower()
        if key in mapping:
            return mapping[key]
        if other and key not in keep:
            return other
        return value

    out[column] = _as_object(frame[column].map(apply), frame.index)
    return _outcome(
        frame, out, [column], f"Mapped {len(mapping)} variant labels in {column} to their official names."
    )


@operation("cluster_labels")
def cluster_labels(frame: pd.DataFrame, params: dict[str, Any], ctx: CleaningContext) -> StepOutcome:
    column = _column(frame, params)
    observed = frame[column].dropna().astype(str).str.strip()
    counts = observed.value_counts()
    canonical: dict[str, str] = {}
    for value in counts.index:
        key = fingerprint(value)
        canonical.setdefault(key, str(value))  # most frequent spelling wins
    out = frame.copy()
    out[column] = _as_object(
        frame[column].map(lambda v: v if is_blank(v) else canonical.get(fingerprint(v), v)), frame.index
    )
    merged = len(counts) - len(set(canonical.values()))
    return _outcome(
        frame, out, [column], f"Merged {merged} spelling variants in {column} into their most common form."
    )


# ------------------------------------------------------------------ structure and privacy


@operation("drop_columns")
def drop_columns(frame: pd.DataFrame, params: dict[str, Any], ctx: CleaningContext) -> StepOutcome:
    columns = _columns(frame, params)
    out = frame.drop(columns=columns)
    return _outcome(
        frame, out, columns, f"Dropped {', '.join(columns)}." if columns else "No column dropped."
    )


@operation("rename_columns")
def rename_columns(frame: pd.DataFrame, params: dict[str, Any], ctx: CleaningContext) -> StepOutcome:
    mapping = {str(k): str(v) for k, v in dict(params.get("mapping") or {}).items()}
    unknown = [k for k in mapping if k not in frame.columns]
    if unknown:
        raise OperationError(f"Unknown column: {', '.join(unknown)}")
    bad = [v for v in mapping.values() if not re.fullmatch(r"[a-z][a-z0-9_]{0,62}", v)]
    if bad:
        raise OperationError(
            f"New names must be lower case letters, digits and underscores: {', '.join(bad)}"
        )
    out = frame.rename(columns=mapping)
    return _outcome(
        frame, out, list(mapping.values()), f"Renamed {', '.join(f'{k} → {v}' for k, v in mapping.items())}."
    )


@operation("tidy_reshape")
def tidy_reshape(frame: pd.DataFrame, params: dict[str, Any], ctx: CleaningContext) -> StepOutcome:
    id_columns = _columns(frame, params, "id_columns")
    value_columns = _columns(frame, params, "value_columns")
    if not value_columns:
        raise OperationError("Choose the columns to stack into rows.")
    out = frame.melt(
        id_vars=id_columns,
        value_vars=value_columns,
        var_name=str(params.get("var_name", "variable")),
        value_name=str(params.get("value_name", "value")),
    )
    return _outcome(
        frame,
        out,
        list(out.columns),
        f"Stacked {len(value_columns)} columns into rows ({len(frame)} → {len(out)} rows).",
    )


@operation("mask_columns")
def mask_columns(frame: pd.DataFrame, params: dict[str, Any], ctx: CleaningContext) -> StepOutcome:
    # An earlier step may already have dropped a personal column; that is not an error.
    requested = params.get("columns") or []
    requested = [requested] if isinstance(requested, str) else list(requested)
    columns = [c for c in requested if c in frame.columns]
    gone = [c for c in requested if c not in frame.columns]
    method = params.get("method", "redact")
    out = frame.copy()
    if method == "drop":
        out = frame.drop(columns=columns)
    for column in columns if method != "drop" else []:
        kind = ctx.pii_columns.get(column, PiiKind.NAME.value)
        if method == "hash":
            out[column] = _as_object(frame[column].map(lambda v: pseudonymise(v, ctx.salt)), frame.index)
        else:
            out[column] = _as_object(frame[column].map(lambda v, k=kind: mask_value(v, k)), frame.index)
    verb = {"drop": "Dropped", "hash": "Pseudonymised", "redact": "Masked"}.get(method, "Masked")
    summary = f"{verb} {', '.join(columns)}." if columns else "No personal column left to mask."
    if gone:
        summary += f" Already removed earlier: {', '.join(gone)}."
    return _outcome(frame, out, columns, summary)


@operation("k_anonymity")
def k_anonymity(frame: pd.DataFrame, params: dict[str, Any], ctx: CleaningContext) -> StepOutcome:
    quasi = _columns(frame, params, "quasi_identifiers")
    k = int(params.get("k", 5))
    if not quasi:
        raise OperationError(
            "Choose the quasi-identifiers (columns that together could single a person out)."
        )
    keys = frame[quasi].apply(_text).agg("|".join, axis=1)
    sizes = keys.map(keys.value_counts())
    keep = sizes >= k
    out = frame[keep].copy()
    return _outcome(
        frame, out, quasi, f"Suppressed {thousands(int((~keep).sum()))} rows in groups smaller than k = {k}."
    )


# ------------------------------------------------------------------ rules and constraints


@operation("validate_rules")
def validate_rules(frame: pd.DataFrame, params: dict[str, Any], ctx: CleaningContext) -> StepOutcome:
    rules: list[dict[str, Any]] = list(params.get("rules") or [])
    action = params.get("action", "flag")
    failures = pd.Series("", index=frame.index, dtype=object)
    for index, rule in enumerate(rules):
        column = rule.get("column")
        if column not in frame.columns:
            raise OperationError(f"Rule {index + 1} names unknown column {column!r}.")
        series = frame[column]
        blank = _missing(series)
        kind = rule.get("rule")
        if kind == "not_null":
            failed = blank
        elif kind == "range":
            numeric = _numeric(series)
            low, high = rule.get("min"), rule.get("max")
            failed = pd.Series(False, index=frame.index)
            if low is not None:
                failed |= numeric < float(low)
            if high is not None:
                failed |= numeric > float(high)
            failed &= ~blank
        elif kind == "date_range":
            dates = series.map(parse_date)
            low = parse_date(rule.get("min")) if rule.get("min") else None
            high = parse_date(rule.get("max")) if rule.get("max") else None
            failed = dates.map(
                lambda d, lo=low, hi=high: (
                    d is not None and ((lo is not None and d < lo) or (hi is not None and d > hi))
                )
            ).astype(bool)
        elif kind == "allowed":
            allowed = {str(v).strip().lower() for v in rule.get("values", [])}
            failed = ~blank & ~series.astype(str).str.strip().str.lower().isin(allowed)
        elif kind == "regex":
            pattern = re.compile(str(rule.get("pattern", ".*")))
            failed = ~blank & ~_text(series).map(lambda v, rx=pattern: bool(rx.fullmatch(v.strip())))
        elif kind == "unique":
            failed = ~blank & series.astype(str).str.strip().duplicated(keep=False)
        else:
            raise OperationError(f"Unknown rule {kind!r}.")
        label = rule.get("name") or f"{column}:{kind}"
        failures = failures.where(~failed, failures.map(lambda f, lbl=label: f"{f}; {lbl}" if f else lbl))
    failed_any = failures != ""
    out = frame.copy()
    if action == "drop":
        out = out[~failed_any].copy()
        return _outcome(
            frame,
            out,
            [],
            f"Dropped {int(failed_any.sum())} rows that failed a rule; {int((~failed_any).sum())} pass.",
        )
    out["row_failed_rules"] = _as_object(failed_any.to_numpy(), frame.index)
    out["failed_rules"] = _as_object(failures.where(failed_any, None).to_numpy(), frame.index)
    return _outcome(
        frame,
        out,
        ["row_failed_rules", "failed_rules"],
        f"{thousands(int((~failed_any).sum()))} rows pass all {len(rules)} rules · {thousands(int(failed_any.sum()))} flagged.",
    )


@operation("validate_sa_id")
def validate_sa_id(frame: pd.DataFrame, params: dict[str, Any], ctx: CleaningContext) -> StepOutcome:
    column = params.get("column")
    if not column or column not in frame.columns:
        return StepOutcome(
            frame, [], "No ID column remains, so there is nothing to check.", len(frame), len(frame), 0
        )
    valid = frame[column].map(lambda v: None if is_blank(v) else sa_id_valid(v))
    action = params.get("action", "flag")
    out = frame.copy()
    invalid = valid == False  # noqa: E712 - None (blank) is neither valid nor invalid
    if action == "drop":
        out = out[~invalid].copy()
        return _outcome(frame, out, [], f"Dropped {int(invalid.sum())} rows with an invalid ID number.")
    out[f"{column}_valid"] = _as_object(valid.to_numpy(), frame.index)
    return _outcome(
        frame,
        out,
        [f"{column}_valid"],
        f"{int(invalid.sum())} ID numbers fail the check digit. None were rewritten.",
    )


def _reference(params: dict[str, Any], ctx: CleaningContext) -> set[str]:
    reference = str(params.get("reference") or "")
    if reference not in ctx.references:
        raise OperationError(f"The reference list {reference!r} is not available to this recipe.")
    return ctx.references[reference]


@operation("flag_orphans")
def flag_orphans(frame: pd.DataFrame, params: dict[str, Any], ctx: CleaningContext) -> StepOutcome:
    column = _column(frame, params)
    keys = _reference(params, ctx)
    orphan = frame[column].map(lambda v: not is_blank(v) and str(v).strip() not in keys)
    out = frame.copy()
    out[f"{column}_orphan"] = _as_object(orphan.to_numpy(), frame.index)
    return _outcome(
        frame,
        out,
        [f"{column}_orphan"],
        f"Flagged {thousands(int(orphan.sum()))} rows whose {column} is not in {params.get('reference')}. Values unchanged.",
    )


@operation("repair_constraints")
def repair_constraints(frame: pd.DataFrame, params: dict[str, Any], ctx: CleaningContext) -> StepOutcome:
    determinant = _column(frame, params, "determinant")
    dependent = _column(frame, params, "dependent")
    min_share = float(params.get("min_share", 0.8))
    groups = frame.groupby(_text(frame[determinant]))[dependent]
    majority = groups.agg(lambda s: str(s.astype(str).value_counts().index[0]) if s.notna().any() else "")
    share = groups.agg(
        lambda s: s.astype(str).value_counts(normalize=True).iloc[0] if s.notna().any() else 0.0
    )
    target = _text(frame[determinant]).map(majority)
    confident = _text(frame[determinant]).map(share) >= min_share
    differs = _text(frame[dependent]) != _text(target)
    repair = differs & confident & target.notna()
    out = frame.copy()
    out[dependent] = _as_object(np.where(repair, target, frame[dependent]), frame.index)
    out[f"{dependent}_repaired"] = _as_object(repair.to_numpy(), frame.index)
    review = int((differs & ~confident).sum())
    return _outcome(
        frame,
        out,
        [dependent, f"{dependent}_repaired"],
        f"Repaired {int(repair.sum())} cells so {determinant} → {dependent} holds; {review} conflicting cells left for review.",
    )


@operation("reconcile_master")
def reconcile_master(frame: pd.DataFrame, params: dict[str, Any], ctx: CleaningContext) -> StepOutcome:
    column = _column(frame, params)
    master = sorted(_reference(params, ctx))
    threshold = float(params.get("threshold", 0.9))
    by_fingerprint = {fingerprint(value): value for value in master}
    cache: dict[str, str | None] = {}

    def match(value: Any) -> str | None:
        key = str(value).strip()
        if key in cache:
            return cache[key]
        found = by_fingerprint.get(fingerprint(key))
        if found is None:
            best = max(master, key=lambda m: jaro_winkler(key.lower(), m.lower()), default=None)
            if best is not None and jaro_winkler(key.lower(), best.lower()) >= threshold:
                found = best
        cache[key] = found
        return found

    matched = frame[column].map(lambda v: None if is_blank(v) else match(v))
    out = frame.copy()
    out[column] = _as_object(np.where(matched.notna(), matched, frame[column]), frame.index)
    unmatched = matched.isna() & ~_missing(frame[column])
    out[f"{column}_unmatched"] = _as_object(unmatched.to_numpy(), frame.index)
    return _outcome(
        frame,
        out,
        [column, f"{column}_unmatched"],
        f"Matched {column} to the master list; {int(unmatched.sum())} values had no confident match.",
    )


def run_operation(
    name: str, frame: pd.DataFrame, params: dict[str, Any], ctx: CleaningContext
) -> StepOutcome:
    fn = OPERATIONS.get(name)
    if fn is None:
        raise OperationError(f"Unknown operation {name!r}.")
    return fn(frame, params, ctx)
