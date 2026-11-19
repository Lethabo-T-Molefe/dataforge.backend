"""Trained outcome models: plan the columns, check the data, fit, validate, audit fairness, explain and score.

The engine is generic. A curator picks any approved dataset, a column that records an outcome, and the value
that counts as the outcome happening (for example "Dropped out", "Churned" or 1). Optionally they name
protected attributes, which are audited for fairness and never used to score.

Everything here is a pure function over pandas frames, and everything returned is plain data (dicts, lists,
bytes), so training and scoring can run in a worker process that is killed at its time limit.

Guarantees, each enforced here rather than left to the caller:

* Personal information and protected attributes are never features.
* A column that on its own predicts the outcome almost perfectly is treated as leakage and left out.
* The decision threshold is chosen on out-of-fold predictions, never on the test set.
* Every model is reported against baselines that need no learning, including the platform's own
  documented heuristic method, so an improvement is a measured one.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from .prediction import FactorInput, ModelError, factor_score
from .semantic import humanise

ALGORITHMS = {
    "logistic": "Logistic regression",
    "hist_gb": "Gradient-boosted trees",
}
THRESHOLD_POLICIES = ("max_f1", "recall_at", "top_pct")
NUMERIC_TYPES = ("integer", "bigint", "smallint", "numeric", "double precision", "real", "decimal")
TEXT_TYPES = ("text", "boolean")
TIME_TYPES = ("date", "timestamp", "timestamp with time zone", "timestamp without time zone")

MAX_CATEGORIES = 30
MAX_OUTCOME_VALUES = 10
MIN_LABELLED = 200
MIN_EACH_CLASS = 30
MAX_MISSING_SHARE = 0.6
LEAKAGE_EXCLUDE_AUC = 0.98
LEAKAGE_WARN_AUC = 0.90
SMALL_GROUP = 30
FOUR_FIFTHS = 0.8
RECALL_GAP = 0.10
PSI_WATCH, PSI_DRIFT = 0.1, 0.25
REFERENCE_BINS = 10

_PROTECTED_WORDS = {
    "gender",
    "sex",
    "race",
    "racial",
    "population",
    "ethnic",
    "ethnicity",
    "disability",
    "disabled",
    "age",
    "religion",
    "religious",
    "nationality",
    "marital",
    "pregnancy",
}
_IDENTIFIER = re.compile(r"(^id$|_id$|_number$|^uuid$|_uuid$|_key$)")
_MONTH_PREFIX = re.compile(r"^(\d{4}-\d{2})")


# ------------------------------------------------------------------ specification


@dataclass(frozen=True)
class OutcomeSpec:
    column: str
    positive: str
    unlabelled: tuple[
        str, ...
    ] = ()  # values whose outcome is not known yet: they are scored, not learned from
    label: str = ""  # how the outcome reads in a sentence: "drops out", "churns"
    entity: str = "record"  # what one row is: "learner", "customer"

    def sentence(self) -> str:
        return self.label or f"has {self.column} = {self.positive}"


@dataclass(frozen=True)
class TrainingSpec:
    name: str
    version: str
    outcome: OutcomeSpec
    protected: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    treat_as_category: tuple[str, ...] = ()
    train_until: str | None = None  # YYYY-MM: only records up to this month are learned from
    time_column: str | None = None
    key_column: str | None = None
    algorithms: tuple[str, ...] = ("logistic", "hist_gb")
    threshold_policy: str = "max_f1"
    threshold_target: float | None = None  # the recall to reach (recall_at) or the share to flag (top_pct)
    cv_folds: int = 5
    seed: int = 20261119
    max_rows: int = 50_000

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "outcome": {
                "column": self.outcome.column,
                "positive": self.outcome.positive,
                "unlabelled": list(self.outcome.unlabelled),
                "label": self.outcome.label,
                "entity": self.outcome.entity,
            },
            "protected": list(self.protected),
            "exclude": list(self.exclude),
            "treatAsCategory": list(self.treat_as_category),
            "trainUntil": self.train_until,
            "timeColumn": self.time_column,
            "keyColumn": self.key_column,
            "algorithms": list(self.algorithms),
            "threshold": {"policy": self.threshold_policy, "target": self.threshold_target},
            "cvFolds": self.cv_folds,
            "seed": self.seed,
            "maxRows": self.max_rows,
        }


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(v) for v in value)


def spec_from_dict(body: dict[str, Any]) -> TrainingSpec:
    """Read a training request. Raise ModelError with every problem found."""
    problems: list[str] = []
    outcome = body.get("outcome") or {}
    if not isinstance(outcome, dict):
        outcome = {}
    name = str(body.get("name") or "").strip()
    version = str(body.get("version") or "").strip()
    column = str(outcome.get("column") or "").strip()
    positive = outcome.get("positive")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,80}", name):
        problems.append("Give the model a name of lower-case letters, digits and hyphens.")
    if not re.fullmatch(r"\d{1,3}(\.\d{1,3}){0,2}(-[a-z0-9]{1,12})?", version):
        problems.append("Give a version such as 1.0.")
    if not column:
        problems.append("Choose the column that records the outcome.")
    if positive is None or str(positive).strip() == "":
        problems.append("Choose the value that counts as the outcome happening.")
    threshold = body.get("threshold") or {}
    policy = str(threshold.get("policy") or "max_f1")
    raw_target = threshold.get("target")
    target: float | None = None
    if policy not in THRESHOLD_POLICIES:
        problems.append(f"Threshold policy must be one of {', '.join(THRESHOLD_POLICIES)}.")
    if policy in ("recall_at", "top_pct"):
        try:
            target = float(raw_target) if raw_target is not None else None
        except (TypeError, ValueError):
            target = None
        if target is None or not 0 < target < 1:
            problems.append("Give the threshold target as a share between 0 and 1, for example 0.7.")
    algorithms = _strings(body.get("algorithms")) or ("logistic", "hist_gb")
    unknown = [a for a in algorithms if a not in ALGORITHMS]
    if unknown:
        problems.append(f"Unknown algorithm: {', '.join(unknown)}.")
    train_until = body.get("trainUntil")
    if train_until and not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", str(train_until)):
        problems.append("trainUntil is a month such as 2025-04.")
    folds = int(body.get("cvFolds") or 5)
    if not 3 <= folds <= 10:
        problems.append("Use between 3 and 10 cross-validation folds.")
    if problems:
        raise ModelError(" ".join(problems))
    return TrainingSpec(
        name=name,
        version=version,
        outcome=OutcomeSpec(
            column=column,
            positive=str(positive).strip(),
            unlabelled=_strings(outcome.get("unlabelled")),
            label=str(outcome.get("label") or "").strip(),
            entity=str(outcome.get("entity") or "record").strip() or "record",
        ),
        protected=_strings(body.get("protected")),
        exclude=_strings(body.get("exclude")),
        treat_as_category=_strings(body.get("treatAsCategory")),
        train_until=str(train_until) if train_until else None,
        algorithms=algorithms,
        threshold_policy=policy,
        threshold_target=target,
        cv_folds=folds,
        seed=int(body.get("seed") or 20261119),
        max_rows=int(body.get("maxRows") or 50_000),
    )


# ------------------------------------------------------------------ column plan


@dataclass(frozen=True)
class ColumnPlan:
    name: str
    role: str  # feature | target | protected | key | time | excluded
    kind: str  # numeric | categorical | text | time | other
    reason: str
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "label": humanise(self.name),
            "role": self.role,
            "kind": self.kind,
            "reason": self.reason,
        }
        if self.warning:
            out["warning"] = self.warning
        return out


def _words(name: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9]+", name.lower()) if w}


def suggest_protected(columns: list[dict[str, Any]]) -> list[str]:
    """Columns whose names suggest a protected attribute. Only a suggestion: the curator confirms."""
    return [
        str(c["name"])
        for c in columns
        if not c.get("pii")
        and _words(str(c["name"])) & _PROTECTED_WORDS
        and int(c.get("distinct") or 0) <= 60
    ]


def outcome_candidates(columns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Columns with a handful of values, any of which could record an outcome, with their values."""
    out = []
    for c in columns:
        distinct = int(c.get("distinct") or 0)
        if c.get("pii") or not 2 <= distinct <= MAX_OUTCOME_VALUES:
            continue
        if c.get("type") not in (*TEXT_TYPES, *NUMERIC_TYPES):
            continue
        out.append(
            {"name": c["name"], "label": humanise(str(c["name"])), "values": list(c.get("values") or [])}
        )
    return out


def _is_time(column: dict[str, Any], spec: TrainingSpec) -> bool:
    return (
        column["name"] == spec.time_column
        or bool(column.get("monthLike"))
        or str(column.get("type")) in TIME_TYPES
    )


def suggest_columns(columns: list[dict[str, Any]], rows: int, spec: TrainingSpec) -> list[ColumnPlan]:
    """Decide what each column is for, with a reason a reviewer can read. No data is read here."""
    plan: list[ColumnPlan] = []
    protected = set(spec.protected)
    excluded = set(spec.exclude)
    categories = set(spec.treat_as_category)
    names = {str(c["name"]) for c in columns}
    if spec.outcome.column not in names:
        raise ModelError(f"The dataset has no column {spec.outcome.column}.")
    key_taken = False
    for column in columns:
        name = str(column["name"])
        kind = str(column.get("type") or "text")
        distinct = int(column.get("distinct") or 0)
        if name == spec.outcome.column:
            plan.append(ColumnPlan(name, "target", "categorical", "The outcome the model learns to predict."))
        elif column.get("pii"):
            plan.append(
                ColumnPlan(name, "excluded", "text", "Personal information is never used to score anyone.")
            )
        elif name in protected:
            plan.append(
                ColumnPlan(
                    name,
                    "protected",
                    "categorical",
                    "A protected attribute: audited for fairness, never used to score.",
                )
            )
        elif name in excluded:
            plan.append(ColumnPlan(name, "excluded", "other", "Left out by the curator."))
        elif _is_time(column, spec):
            plan.append(
                ColumnPlan(
                    name,
                    "time",
                    "time",
                    "The time column: it decides which records are old enough to learn from.",
                )
            )
        elif _IDENTIFIER.search(name.lower()) or (rows > 20 and distinct >= 0.95 * rows):
            if not key_taken and (spec.key_column in (None, name)) and distinct >= 0.99 * rows:
                key_taken = True
                plan.append(
                    ColumnPlan(
                        name,
                        "key",
                        "text",
                        "Identifies each record, so a score can be followed up. Never a feature.",
                    )
                )
            else:
                plan.append(ColumnPlan(name, "excluded", "text", "Looks like an identifier, not a signal."))
        elif distinct <= 1:
            plan.append(
                ColumnPlan(name, "excluded", "other", "The same for every record, so it tells nothing.")
            )
        elif kind in NUMERIC_TYPES and not (name in categories and distinct <= MAX_CATEGORIES):
            plan.append(
                ColumnPlan(name, "feature", "numeric", "A number: gaps filled with the median, then scaled.")
            )
        elif kind in TEXT_TYPES or kind in NUMERIC_TYPES:
            if distinct <= MAX_CATEGORIES:
                plan.append(
                    ColumnPlan(
                        name,
                        "feature",
                        "categorical",
                        "A category: each value becomes its own signal; rare values are pooled.",
                    )
                )
            else:
                plan.append(
                    ColumnPlan(
                        name,
                        "excluded",
                        "text",
                        f"Too many different values ({distinct}) to learn from reliably.",
                    )
                )
        else:
            plan.append(ColumnPlan(name, "excluded", "other", f"Columns of type {kind} are not used."))
    missing = sorted(protected - names)
    if missing:
        raise ModelError(f"The dataset has no column {', '.join(missing)} to audit.")
    return plan


# ------------------------------------------------------------------ labels and features


def _normal(value: Any) -> str:
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text.lower()
    return repr(float(number)) if math.isfinite(number) else text.lower()


def labels(frame: pd.DataFrame, outcome: OutcomeSpec) -> tuple[pd.Series, pd.Series]:
    """(labelled mask, y): unlabelled rows have an unknown outcome or a value the curator marked as unknown."""
    raw = frame[outcome.column]
    text = raw.map(lambda v: "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v).strip())
    unknown = {_normal(v) for v in outcome.unlabelled}
    normal = text.map(_normal)
    labelled = (text != "") & ~normal.isin(unknown)
    y = (normal == _normal(outcome.positive)).astype(int)
    return labelled, y


def _month(value: Any) -> str | None:
    if value is None:
        return None
    match = _MONTH_PREFIX.match(str(value))
    return match.group(1) if match else None


def month_of(series: pd.Series) -> pd.Series:
    """YYYY-MM for each value (a month, a date or a timestamp), or None."""
    return series.map(_month)


def _feature_frame(frame: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for name in numeric:
        out[name] = pd.to_numeric(frame[name], errors="coerce").astype(float)
    for name in categorical:
        out[name] = frame[name].map(
            lambda v: (
                "(blank)"
                if v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == ""
                else str(v)
            )
        )
    return out


def _roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, score))


def _univariate(x: pd.Series, y: np.ndarray, categorical: bool) -> tuple[float, np.ndarray]:
    """How well one column alone ranks the outcome (AUC, direction-free), and the score it uses."""
    if categorical:
        rates = pd.Series(y, index=x.index).groupby(x).mean()
        score = x.map(rates).fillna(float(y.mean())).to_numpy(dtype=float)
    else:
        score = x.fillna(x.median()).to_numpy(dtype=float)
    auc = _roc_auc(y, score)
    return max(auc, 1 - auc), score


# ------------------------------------------------------------------ training


def _pipeline(algorithm: str, numeric: list[str], categorical: list[str], seed: int) -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    parts = []
    if numeric:
        parts.append(
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                numeric,
            )
        )
    if categorical:
        parts.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False),
                categorical,
            )
        )
    prep = ColumnTransformer(parts, remainder="drop")
    if algorithm == "logistic":
        model: Any = LogisticRegression(max_iter=2000, C=1.0)
    else:
        model = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.06, max_leaf_nodes=15, early_stopping=True, random_state=seed
        )
    return Pipeline([("prep", prep), ("model", model)])


def _threshold(oof: np.ndarray, y: np.ndarray, policy: str, target: float | None) -> float:
    candidates = np.unique(np.quantile(oof, np.linspace(0.01, 0.99, 99)))
    if policy == "top_pct" and target:
        return float(np.quantile(oof, 1 - target))
    best, best_value = 0.5, -1.0
    positives = max(1, int(y.sum()))
    for t in candidates:
        flagged = oof >= t
        tp = int((flagged & (y == 1)).sum())
        recall = tp / positives
        if policy == "recall_at" and target:
            # The highest threshold that still reaches the recall target flags the fewest records.
            if recall >= target and t > best:
                best, best_value = float(t), recall
            continue
        precision = tp / max(1, int(flagged.sum()))
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        if f1 > best_value:
            best, best_value = float(t), f1
    if policy == "recall_at" and best_value < 0:
        return float(candidates.min())
    return best


def _r(value: float, digits: int = 4) -> float:
    return round(float(value), digits) if math.isfinite(float(value)) else 0.0


def _confusion(y: np.ndarray, flagged: np.ndarray) -> dict[str, Any]:
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    tn = int((~flagged & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": _r(precision),
        "recall": _r(recall),
        "f1": _r(f1),
        "flaggedShare": _r((tp + fp) / max(1, len(y))),
    }


def _curve_points(xs: np.ndarray, ys: np.ndarray, limit: int = 50) -> list[list[float]]:
    if len(xs) <= limit:
        idx = np.arange(len(xs))
    else:
        idx = np.unique(np.linspace(0, len(xs) - 1, limit).round().astype(int))
    return [[_r(xs[i]), _r(ys[i])] for i in idx]


def _evaluate(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_curve

    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    try:
        observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
    except ValueError:
        observed, predicted = np.array([]), np.array([])
    return {
        "rocAuc": _r(_roc_auc(y, p)),
        "prAuc": _r(float(average_precision_score(y, p))),
        "brier": _r(float(brier_score_loss(y, p))),
        "atThreshold": _confusion(y, p >= threshold),
        "roc": _curve_points(fpr, tpr),
        "pr": _curve_points(recall[::-1], precision[::-1]),
        "calibration": [[_r(a), _r(b)] for a, b in zip(predicted, observed, strict=True)],
    }


def _heuristic_scores(
    train: pd.DataFrame, y_train: np.ndarray, test: pd.DataFrame, numeric: list[str]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """The platform's documented heuristic method: an equal-weight robust-z index over the four numeric columns
    that rank the outcome best, each signed by its direction on the training data. Nothing is fitted."""
    ranked = []
    for name in numeric:
        auc = _roc_auc(y_train, train[name].fillna(train[name].median()).to_numpy(dtype=float))
        ranked.append((max(auc, 1 - auc), name, "up" if auc >= 0.5 else "down"))
    chosen = sorted(ranked, reverse=True)[:4]
    if not chosen:
        return np.full(len(test), 0.5), []
    parts = []
    used = []
    for _, name, direction in chosen:
        column = train[name].dropna()
        centre = float(column.median()) if len(column) else 0.0
        spread = float(column.quantile(0.75) - column.quantile(0.25)) / 1.349 if len(column) else 0.0
        values = test[name].fillna(centre).to_numpy(dtype=float)
        parts.append(
            [
                factor_score(FactorInput(float(v), centre, spread, len(column), "decimal"), direction)
                for v in values
            ]
        )
        used.append({"column": name, "direction": direction, "weight": _r(1 / len(chosen))})
    return np.mean(np.array(parts), axis=0), used


def _reference(train: pd.DataFrame, numeric: list[str], categorical: list[str]) -> dict[str, Any]:
    """Training distributions, kept so drift can be measured later without the training data."""
    out: dict[str, Any] = {}
    for name in numeric:
        column = train[name].dropna()
        if column.empty:
            continue
        edges = np.unique(np.quantile(column, np.linspace(0, 1, REFERENCE_BINS + 1))[1:-1])
        counts = np.bincount(
            np.searchsorted(edges, column.to_numpy(), side="right"), minlength=len(edges) + 1
        )
        out[name] = {
            "kind": "numeric",
            "edges": [float(e) for e in edges],
            "shares": [_r(c / len(column), 6) for c in counts],
            "median": float(column.median()),
        }
    for name in categorical:
        shares = train[name].value_counts(normalize=True)
        top = shares.head(MAX_CATEGORIES)
        out[name] = {
            "kind": "categorical",
            "shares": {str(k): _r(v, 6) for k, v in top.items()},
            "mode": str(shares.index[0]) if len(shares) else "",
        }
    return out


def _group_names(pipeline: Any, numeric: list[str], categorical: list[str]) -> list[str]:
    """For each column the model sees after encoding, the original column it came from."""
    out: list[str] = []
    names = list(pipeline.named_steps["prep"].get_feature_names_out())
    for full in names:
        prefix, _, rest = str(full).partition("__")
        if prefix == "num":
            out.append(rest)
        else:
            matches = [c for c in categorical if rest == c or rest.startswith(f"{c}_")]
            out.append(max(matches, key=len) if matches else rest)
    return out


def fairness_report(y: np.ndarray, flagged: np.ndarray, groups: dict[str, pd.Series]) -> dict[str, Any]:
    """Per group: who gets flagged and whose outcomes the model misses.

    For an early-warning model, being flagged means getting support, so the harmful error is a miss (a false
    negative). Recall parity is therefore the main criterion; the four-fifths ratio of selection rates is
    reported too.
    """
    attributes: list[dict[str, Any]] = []
    flags: list[str] = []
    for attribute, series in groups.items():
        values = series.fillna("(blank)").astype(str).to_numpy()
        rows: list[dict[str, Any]] = []
        for value in sorted(set(values)):
            mask = values == value
            n = int(mask.sum())
            positives = int((y[mask] == 1).sum())
            confusion = _confusion(y[mask], flagged[mask])
            small = n < SMALL_GROUP or positives < 10
            fpr = confusion["fp"] / max(1, confusion["fp"] + confusion["tn"])
            rows.append(
                {
                    "group": value,
                    "n": n,
                    "prevalence": _r(positives / max(1, n)),
                    "selectionRate": confusion["flaggedShare"],
                    "recall": confusion["recall"],
                    "falseNegativeRate": _r(1 - confusion["recall"]) if positives else None,
                    "falsePositiveRate": _r(fpr),
                    "precision": confusion["precision"],
                    "tooSmall": small,
                }
            )
        judged = [r for r in rows if not r["tooSmall"]]
        selection_ratio = None
        recall_gap = None
        if len(judged) >= 2:
            rates = [r["selectionRate"] for r in judged]
            selection_ratio = _r(min(rates) / max(rates)) if max(rates) > 0 else 1.0
            recalls = [r["recall"] for r in judged]
            recall_gap = _r(max(recalls) - min(recalls))
        attribute_flags = []
        if selection_ratio is not None and selection_ratio < FOUR_FIFTHS:
            attribute_flags.append(
                f"{humanise(attribute)}: groups are flagged at different rates "
                f"(lowest is {selection_ratio:.0%} of the highest, below the four-fifths guide)."
            )
        if recall_gap is not None and recall_gap > RECALL_GAP:
            attribute_flags.append(
                f"{humanise(attribute)}: the model misses the outcome more often for some groups "
                f"(recall differs by {recall_gap * 100:.0f} points)."
            )
        flags.extend(attribute_flags)
        attributes.append(
            {
                "attribute": attribute,
                "label": humanise(attribute),
                "groups": rows,
                "selectionRateRatio": selection_ratio,
                "recallGap": recall_gap,
                "flags": attribute_flags,
                "tooSmallGroups": [r["group"] for r in rows if r["tooSmall"]],
            }
        )
    return {
        "audited": bool(groups),
        "criterion": "Recall parity (missed outcomes), with the four-fifths selection-rate guide.",
        "attributes": attributes,
        "flags": flags,
        "flagged": bool(flags),
    }


def train(frame: pd.DataFrame, spec: TrainingSpec, plan: list[ColumnPlan]) -> dict[str, Any]:
    """Check the data, then fit and validate each requested algorithm. Raises ModelError with every reason."""
    import joblib
    import sklearn
    from sklearn.model_selection import StratifiedKFold, train_test_split

    labelled, y_all = labels(frame, spec.outcome)
    time_column = next((p.name for p in plan if p.role == "time"), None)
    notes: list[str] = []
    if spec.train_until:
        if time_column is None:
            raise ModelError("trainUntil needs a time column, and this dataset has none.")
        months = month_of(frame[time_column])
        labelled &= months.notna() & (months.fillna("9999-99") <= spec.train_until)
        notes.append(f"Learned only from records up to {spec.train_until}, whose outcomes are settled.")
    data = frame[labelled]
    y = y_all[labelled]
    if len(data) > spec.max_rows:
        fraction = spec.max_rows / len(data)
        data = data.groupby(y, group_keys=False).sample(frac=fraction, random_state=spec.seed)
        y = y.loc[data.index]
        notes.append(
            f"Sampled {len(data)} labelled records, keeping the outcome's share, to stay within limits."
        )

    problems = []
    positives, negatives = int(y.sum()), int((y == 0).sum())
    if len(data) < MIN_LABELLED:
        problems.append(f"Only {len(data)} records have a known outcome; at least {MIN_LABELLED} are needed.")
    if positives < MIN_EACH_CLASS or negatives < MIN_EACH_CLASS:
        problems.append(
            f"The outcome happened {positives} times and did not happen {negatives} times; each needs at least "
            f"{MIN_EACH_CLASS}. Check the value chosen as the outcome."
        )
    if problems:
        raise ModelError(" ".join(problems))
    prevalence = positives / len(data)
    if not 0.01 <= prevalence <= 0.99:
        raise ModelError(
            f"The outcome happens in {prevalence:.1%} of records, too rare or too common to model."
        )

    features = [p for p in plan if p.role == "feature"]
    numeric = [p.name for p in features if p.kind == "numeric"]
    categorical = [p.name for p in features if p.kind == "categorical"]
    X = _feature_frame(data, numeric, categorical)
    y_np = y.to_numpy(dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_np, test_size=0.2, stratify=y_np, random_state=spec.seed
    )

    # Data checks on the training split only, so the test split stays untouched.
    updated: dict[str, ColumnPlan] = {}
    univariate: dict[str, float] = {}
    for p in features:
        series = X_train[p.name]
        missing = float(series.isna().mean()) if p.kind == "numeric" else float((series == "(blank)").mean())
        if missing > MAX_MISSING_SHARE:
            updated[p.name] = replace(
                p, role="excluded", reason=f"Missing in {missing:.0%} of records, too sparse to learn from."
            )
            continue
        auc, _ = _univariate(series, y_train, p.kind == "categorical")
        univariate[p.name] = _r(auc)
        if auc >= LEAKAGE_EXCLUDE_AUC:
            updated[p.name] = replace(
                p,
                role="excluded",
                reason=(
                    f"Possible leakage: on its own it predicts the outcome almost perfectly (AUC {auc:.2f}). "
                    "It is probably recorded after the outcome is known."
                ),
            )
        elif auc >= LEAKAGE_WARN_AUC:
            updated[p.name] = replace(
                p,
                warning=(
                    f"On its own it predicts the outcome very well (AUC {auc:.2f}). Check it is known before the "
                    "outcome, or the model will look better than it is."
                ),
            )
    plan = [updated.get(p.name, p) for p in plan]
    numeric = [p.name for p in plan if p.role == "feature" and p.kind == "numeric"]
    categorical = [p.name for p in plan if p.role == "feature" and p.kind == "categorical"]
    if not numeric and not categorical:
        raise ModelError("No usable feature is left after the checks. See each column's reason.")
    X_train, X_test = X_train[numeric + categorical], X_test[numeric + categorical]

    # Baselines that need no learning. The best single column is chosen among the columns the model uses.
    kept = {name: univariate[name] for name in numeric + categorical if name in univariate}
    best_feature = max(kept, key=lambda k: kept[k]) if kept else None
    baselines: dict[str, Any] = {
        "prevalence": {
            "label": "No model (everyone at the average rate)",
            "rocAuc": 0.5,
            "prAuc": _r(float(y_test.mean())),
        }
    }
    if best_feature:
        if best_feature in categorical:
            rates = pd.Series(y_train, index=X_train.index).groupby(X_train[best_feature]).mean()
            test_score = X_test[best_feature].map(rates).fillna(float(y_train.mean())).to_numpy(dtype=float)
        else:
            test_score = X_test[best_feature].fillna(X_train[best_feature].median()).to_numpy(dtype=float)
        auc = _roc_auc(y_test, test_score)
        baselines["bestSingleColumn"] = {
            "label": f"Best single column ({humanise(best_feature)})",
            "column": best_feature,
            "rocAuc": _r(max(auc, 1 - auc)),
        }
    heuristic, used = _heuristic_scores(X_train, y_train, X_test, numeric)
    if used:
        from sklearn.metrics import average_precision_score

        baselines["heuristic"] = {
            "label": "Documented heuristic (equal-weight robust z over the strongest columns)",
            "columns": used,
            "rocAuc": _r(_roc_auc(y_test, heuristic)),
            "prAuc": _r(float(average_precision_score(y_test, heuristic))),
        }

    folds = StratifiedKFold(n_splits=spec.cv_folds, shuffle=True, random_state=spec.seed)
    reference = _reference(X_train, numeric, categorical)
    group_frames = {name: data.loc[X_test.index, name] for name in spec.protected if name in data.columns}
    models: list[dict[str, Any]] = []
    for algorithm in spec.algorithms:
        oof = np.zeros(len(y_train))
        fold_auc, fold_pr = [], []
        from sklearn.metrics import average_precision_score

        for train_idx, valid_idx in folds.split(X_train, y_train):
            fitted = _pipeline(algorithm, numeric, categorical, spec.seed).fit(
                X_train.iloc[train_idx], y_train[train_idx]
            )
            p = fitted.predict_proba(X_train.iloc[valid_idx])[:, 1]
            oof[valid_idx] = p
            fold_auc.append(_roc_auc(y_train[valid_idx], p))
            fold_pr.append(float(average_precision_score(y_train[valid_idx], p)))
        threshold = _threshold(oof, y_train, spec.threshold_policy, spec.threshold_target)
        pipeline = _pipeline(algorithm, numeric, categorical, spec.seed).fit(X_train, y_train)
        p_test = pipeline.predict_proba(X_test)[:, 1]
        metrics = _evaluate(y_test, p_test, threshold)
        metrics["crossValidation"] = {
            "folds": spec.cv_folds,
            "rocAuc": {"mean": _r(float(np.mean(fold_auc))), "sd": _r(float(np.std(fold_auc)))},
            "prAuc": {"mean": _r(float(np.mean(fold_pr))), "sd": _r(float(np.std(fold_pr)))},
        }
        fairness = fairness_report(y_test, p_test >= threshold, group_frames)
        importance = _importance(pipeline, algorithm, X_test, y_test, numeric, categorical, spec.seed)
        buffer = io.BytesIO()
        joblib.dump(
            {
                "pipeline": pipeline,
                "algorithm": algorithm,
                "numeric": numeric,
                "categorical": categorical,
                "groups": _group_names(pipeline, numeric, categorical),
                "baseline": {
                    **{n: float(X_train[n].median()) for n in numeric},
                    **{c: str(X_train[c].mode().iloc[0]) for c in categorical},
                },
            },
            buffer,
            compress=3,
        )
        artifact = buffer.getvalue()
        models.append(
            {
                "algorithm": algorithm,
                "algorithmLabel": ALGORITHMS[algorithm],
                "threshold": _r(threshold, 6),
                "metrics": metrics,
                "fairness": fairness,
                "importance": importance,
                "artifact": artifact,
                "artifactSha256": hashlib.sha256(artifact).hexdigest(),
            }
        )
    return {
        "plan": [p.to_dict() for p in plan],
        "data": {
            "labelled": int(len(data)),
            "positives": positives,
            "prevalence": _r(prevalence),
            "trainRows": int(len(y_train)),
            "testRows": int(len(y_test)),
            "unlabelled": int((~labelled).sum()),
            "notes": notes,
        },
        "univariate": univariate,
        "baselines": baselines,
        "reference": reference,
        "runtime": {
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "models": models,
    }


def _importance(
    pipeline: Any,
    algorithm: str,
    X: pd.DataFrame,
    y: np.ndarray,
    numeric: list[str],
    categorical: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    """Which columns the model relies on: the drop in test AUC when each is shuffled."""
    base = _roc_auc(y, pipeline.predict_proba(X)[:, 1])
    rng = np.random.default_rng(seed)
    out: list[dict[str, Any]] = []
    for name in numeric + categorical:
        shuffled = X.copy()
        shuffled[name] = rng.permutation(shuffled[name].to_numpy())
        drop = base - _roc_auc(y, pipeline.predict_proba(shuffled)[:, 1])
        out.append({"column": name, "label": humanise(name), "aucDrop": _r(drop)})
    out.sort(key=lambda r: -float(r["aucDrop"]))
    return out


# ------------------------------------------------------------------ artifacts, explanations and scoring


class ArtifactError(ModelError):
    pass


def load_artifact(raw: bytes, sha256: str | None, runtime: dict[str, Any] | None) -> dict[str, Any]:
    """Load a stored model only if it is the exact bytes that were reviewed, under the same library version."""
    import joblib
    import sklearn

    if sha256 and hashlib.sha256(raw).hexdigest() != sha256:
        raise ArtifactError(
            "The stored model does not match the version that was reviewed, so it is not used."
        )
    trained_with = (runtime or {}).get("sklearn")
    if trained_with and trained_with.split(".")[:2] != sklearn.__version__.split(".")[:2]:
        raise ArtifactError(
            f"The model was trained with scikit-learn {trained_with} and this server runs {sklearn.__version__}. "
            "Retrain it before use."
        )
    loaded: dict[str, Any] = joblib.load(io.BytesIO(raw))
    return loaded


def _describe(name: str, value: Any, typical: Any) -> str:
    label = humanise(name)
    if isinstance(value, float):
        shown = f"{value:,.1f}".rstrip("0").rstrip(".")
        typical_shown = f"{float(typical):,.1f}".rstrip("0").rstrip(".")
        return f"{label} is {shown}, against a typical {typical_shown}"
    return f"{label} is {value}" + (f" (most records: {typical})" if typical != value else "")


def explain(
    model: dict[str, Any], X: pd.DataFrame, outcome: OutcomeSpec, top: int = 3
) -> list[list[dict[str, Any]]]:
    """The columns that moved each record's score most, in words.

    Logistic regression: exact contributions on the log-odds scale (coefficient times the encoded value,
    summed back per original column). Trees: occlusion, the change in probability when the column is set to
    its typical training value.
    """
    pipeline = model["pipeline"]
    numeric, categorical = model["numeric"], model["categorical"]
    columns = numeric + categorical
    if model["algorithm"] == "logistic":
        encoded = np.asarray(pipeline.named_steps["prep"].transform(X[columns]), dtype=float)
        coef = pipeline.named_steps["model"].coef_[0]
        groups = model["groups"]
        effects = np.zeros((len(X), len(columns)))
        for j, column in enumerate(groups):
            effects[:, columns.index(column)] += encoded[:, j] * coef[j]
    else:
        base = pipeline.predict_proba(X[columns])[:, 1]
        effects = np.zeros((len(X), len(columns)))
        for k, column in enumerate(columns):
            changed = X[columns].copy()
            changed[column] = model["baseline"][column]
            effects[:, k] = base - pipeline.predict_proba(changed)[:, 1]
    out: list[list[dict[str, Any]]] = []
    who = outcome.entity
    what = outcome.sentence()
    for i in range(len(X)):
        order = np.argsort(-np.abs(effects[i]))[:top]
        drivers = []
        for k in order:
            if abs(effects[i, k]) < 1e-9:
                continue
            column = columns[k]
            value = X.iloc[i][column]
            if isinstance(value, float) and math.isnan(value):
                value = "not recorded"
            raises = effects[i, k] > 0
            drivers.append(
                {
                    "column": column,
                    "label": humanise(column),
                    "effect": _r(float(effects[i, k])),
                    "direction": "raises" if raises else "lowers",
                    "text": (
                        f"{_describe(column, value, model['baseline'][column])}. "
                        f"This {'raises' if raises else 'lowers'} the likelihood that the {who} {what}."
                    ),
                }
            )
        out.append(drivers)
    return out


def band(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "High"
    if probability >= 0.6 * threshold:
        return "Watch"
    return "Low"


def score(frame: pd.DataFrame, raw: bytes, meta: dict[str, Any]) -> dict[str, Any]:
    """Score records with a stored model: probability, band, flagged, and the top three reasons."""
    model = load_artifact(raw, meta.get("sha256"), meta.get("runtime"))
    outcome = OutcomeSpec(**meta["outcome"])
    threshold = float(meta["threshold"])
    X = _feature_frame(frame, model["numeric"], model["categorical"])
    probabilities = model["pipeline"].predict_proba(X)[:, 1]
    reasons = explain(model, X, outcome)
    return {
        "probability": [_r(p, 6) for p in probabilities],
        "band": [band(float(p), threshold) for p in probabilities],
        "flagged": [bool(p >= threshold) for p in probabilities],
        "drivers": reasons,
    }


# ------------------------------------------------------------------ drift


def psi(reference: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    """Population stability index per feature: how far today's records have moved from the training ones."""
    columns: list[dict[str, Any]] = []
    for name, ref in reference.items():
        if name not in frame.columns:
            continue
        if ref["kind"] == "numeric":
            values = pd.to_numeric(frame[name], errors="coerce").dropna().to_numpy()
            if len(values) == 0:
                continue
            counts = np.bincount(
                np.searchsorted(np.array(ref["edges"]), values, side="right"), minlength=len(ref["shares"])
            )
            current = counts / len(values)
            expected = np.array(ref["shares"], dtype=float)
        else:
            shares = frame[name].astype(str).value_counts(normalize=True)
            keys = list(ref["shares"])
            expected = np.array(
                [ref["shares"][k] for k in keys] + [max(0.0, 1 - sum(ref["shares"].values()))]
            )
            current = np.array(
                [float(shares.get(k, 0.0)) for k in keys] + [float(shares[~shares.index.isin(keys)].sum())]
            )
        e = np.clip(expected, 1e-4, None)
        c = np.clip(current, 1e-4, None)
        value = float(np.sum((c - e) * np.log(c / e)))
        status = "drift" if value >= PSI_DRIFT else "watch" if value >= PSI_WATCH else "stable"
        columns.append({"column": name, "label": humanise(name), "psi": _r(value), "status": status})
    columns.sort(key=lambda r: -float(r["psi"]))
    worst = columns[0]["psi"] if columns else 0.0
    return {
        "columns": columns,
        "max": worst,
        "status": "drift" if worst >= PSI_DRIFT else "watch" if worst >= PSI_WATCH else "stable",
        "guide": f"Below {PSI_WATCH} stable, {PSI_WATCH} to {PSI_DRIFT} watch, {PSI_DRIFT} or more has drifted.",
    }


# ------------------------------------------------------------------ the model card


@dataclass
class CardContext:
    dataset_name: str
    dataset_version: int
    trained_by: str
    trained_at: str
    sources: list[str] = field(default_factory=list)


def model_card(
    spec: TrainingSpec, result: dict[str, Any], model: dict[str, Any], ctx: CardContext
) -> dict[str, Any]:
    """What a reviewer needs to decide whether this model may be used, in one document."""
    outcome = spec.outcome
    metrics = model["metrics"]
    heuristic = result["baselines"].get("heuristic", {})
    lift = None
    if heuristic.get("rocAuc"):
        lift = _r(metrics["rocAuc"] - float(heuristic["rocAuc"]))
    limitations = [
        "It learns from past records. If programmes, people or recording practices change, its accuracy can "
        "drop: watch the drift panel.",
        "Only records with a known outcome are learned from. Recent records are scored, not learned from.",
        "A score is a likelihood for prioritising attention, not a fact about a person.",
    ]
    if result["data"]["notes"]:
        limitations.extend(result["data"]["notes"])
    warnings = [p["warning"] for p in result["plan"] if p.get("warning")]
    return {
        "title": f"{spec.name} v{spec.version}",
        "algorithm": ALGORITHMS[model["algorithm"]],
        "predicts": f"Whether the {outcome.entity} {outcome.sentence()} ({outcome.column} = {outcome.positive}).",
        "intendedUse": (
            f"Prioritise which {outcome.entity}s to contact or support first. A person reviews every flagged "
            f"{outcome.entity} and decides what, if anything, to do."
        ),
        "notFor": [
            "Decisions about admission, termination, pay or stipends.",
            "Any decision made automatically, without a person reviewing it (POPIA section 71).",
        ],
        "data": {
            "dataset": ctx.dataset_name,
            "version": ctx.dataset_version,
            **result["data"],
        },
        "features": [p for p in result["plan"] if p["role"] == "feature"],
        "excluded": [p for p in result["plan"] if p["role"] in ("excluded", "protected", "key", "time")],
        "warnings": warnings,
        "performance": {
            "rocAuc": metrics["rocAuc"],
            "prAuc": metrics["prAuc"],
            "brier": metrics["brier"],
            "crossValidation": metrics["crossValidation"],
            "atThreshold": metrics["atThreshold"],
            "liftOverHeuristic": lift,
        },
        "baselines": result["baselines"],
        "threshold": {
            "value": model["threshold"],
            "policy": spec.threshold_policy,
            "target": spec.threshold_target,
            "chosenOn": "out-of-fold predictions on the training split",
        },
        "fairness": {
            "audited": model["fairness"]["audited"],
            "flags": model["fairness"]["flags"],
            "criterion": model["fairness"]["criterion"],
        },
        "limitations": limitations,
        "humanInTheLoop": (
            "Scores are shown to people who decide what to do. Nobody is treated differently by the system "
            "because of a score alone."
        ),
        "trainedBy": ctx.trained_by,
        "trainedAt": ctx.trained_at,
        "runtime": result["runtime"],
    }
