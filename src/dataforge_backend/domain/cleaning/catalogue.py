"""The cleaning catalogue: chapters (families of problems) and methods, each backed by a real operation.

Method ids match the frontend. Default parameters are worked out from the dataset's profile, so the same
catalogue works on any table rather than one example file.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..values import fingerprint, thousands

Profile = dict[str, Any]
Defaults = Callable[[Profile, list[str]], dict[str, Any]]


@dataclass(frozen=True)
class Chapter:
    id: str
    kind: str  # profile | transform | governance
    number: str
    title: str
    plain: str
    expert: str
    recommended_method_id: str


@dataclass(frozen=True)
class Method:
    id: str
    chapter_id: str
    name: str
    audience: str  # everyday | expert
    plain: str
    expert: str
    caution: str
    operation: str
    defaults: Defaults
    recommended: bool = False


CHAPTERS: list[Chapter] = [
    Chapter(
        "profile",
        "profile",
        "01",
        "See what is wrong",
        "Before changing anything, we measure the file. This is profiling. It tells us which columns are messy, how values go missing, and which rows look like duplicates.",
        "Column, pattern, dependency and cross-table profiling are the prerequisite. Little's MCAR test, shape counts and dependency discovery decide which later methods are valid. Bronze is not rewritten.",
        "profile-accept",
    ),
    Chapter(
        "missing",
        "transform",
        "02",
        "Handle missing values",
        "Some cells are empty. What we do depends on why they are empty. If the blank is random, we can fill or drop. If it depends on other columns, we fill from those columns. If people hide the value on purpose, we keep a note that it was missing.",
        "Diagnose MCAR / MAR / MNAR before choosing a method. Deletion is safe under MCAR but wastes rows. MAR supports imputation from observed columns. MNAR needs an indicator, a domain model or a sensitivity analysis. Multiple imputation (MICE) and MissForest belong on the expert path.",
        "missing-unknown-indicator",
    ),
    Chapter(
        "duplicates",
        "transform",
        "03",
        "Find and merge repeats",
        "The same record appears more than once. We keep one row per real record so counts are not inflated.",
        "Exact deduplication on a normalised key is the everyday path. Record linkage needs blocking, field similarity (Jaro-Winkler), a match threshold, then clustering and survivorship.",
        "dup-keep-first",
    ),
    Chapter(
        "outliers",
        "transform",
        "04",
        "Check values that sit far away",
        "A few values are enormous or tiny compared with the rest. That can be a typing error, or a rare real event. We flag first. We only delete when it is clearly impossible.",
        "Prefer a modified z-score (MAD) or Tukey fences over a raw z-score on skewed data. Isolation Forest is available when several columns move together. Treatment: remove, winsorise, blank, or flag and keep.",
        "outlier-flag",
    ),
    Chapter(
        "formats",
        "transform",
        "05",
        "Make formats the same",
        "Dates, money, phones and names should look the same in every row. Then a computer can add them up and a person can read them.",
        "Parse day-first explicitly, emit ISO 8601, convert ZA thousands and decimal commas, normalise Unicode (NFC) and write phones in E.164.",
        "format-date-number",
    ),
    Chapter(
        "types",
        "transform",
        "06",
        "Give each column a real type",
        "A date stored as text cannot be sorted as a date. We turn each column into the type it should be, and we turn fake blanks such as N/A into a real empty cell.",
        "Coerce with errors turned into blanks, then count the new nulls. The sentinel list is recorded in the recipe.",
        "type-coerce-sentinels",
    ),
    Chapter(
        "categories",
        "transform",
        "07",
        "Tidy labels",
        "Variants of the same label, such as GP and Gauteng, should become one. We map variants onto one official list.",
        "Start with a mapping table. Fingerprint keys catch the rest. Rare labels can fold into Other after a steward agrees.",
        "cat-map",
    ),
    Chapter(
        "structure",
        "transform",
        "08",
        "Shape the table",
        "One kind of thing per row, one fact per column. We drop columns we will not use, and we keep personal identifiers out of the clean table.",
        "Remove prohibited or irrelevant columns; reshape wide tables to long (tidy) form.",
        "struct-drop-pii",
    ),
    Chapter(
        "validate",
        "transform",
        "09",
        "Check the rules",
        "After cleaning, we test the file against rules: types, ranges, allowed lists and unique keys. Failing rows are marked, not silently dropped.",
        "Expectation-style checks: type, range, domain, regex, uniqueness, completeness and South African ID checksums on any residual identifier.",
        "validate-rules",
    ),
    Chapter(
        "constraints",
        "transform",
        "10",
        "Repair contradictions",
        "If two columns should agree and they do not, we either flag the row or pick the more trusted value. We do not invent values we do not have.",
        "Functional dependencies, referential integrity against master lists, majority-vote repair for near-holding dependencies, and master-data reconciliation.",
        "constraint-flag-orphans",
    ),
    Chapter(
        "privacy",
        "transform",
        "11",
        "Keep personal information out of silver",
        "Personal columns were masked on arrival. We confirm they stay masked or dropped.",
        "Residual PII scan, masking or pseudonymisation, and generalisation toward k-anonymity when data will leave the organisation. Bronze stays as it arrived.",
        "privacy-keep-masked",
    ),
]

CHAPTERS_BY_ID = {chapter.id: chapter for chapter in CHAPTERS}


# ------------------------------------------------------------------ profile helpers


def _cols(profile: Profile, available: list[str]) -> list[dict[str, Any]]:
    return [c for c in profile.get("columns", []) if c["name"] in available]


def _pii(profile: Profile, available: list[str], kinds: set[str] | None = None) -> list[str]:
    return [
        c["name"]
        for c in _cols(profile, available)
        if c.get("piiKind") and (kinds is None or c["piiKind"] in kinds)
    ]


def _numeric(profile: Profile, available: list[str]) -> list[str]:
    return [
        c["name"]
        for c in _cols(profile, available)
        if c["inferredType"] in ("integer", "decimal") and not c.get("piiKind")
    ]


def _dates(profile: Profile, available: list[str]) -> list[str]:
    return [
        c["name"]
        for c in _cols(profile, available)
        if c["inferredType"] == "date" or (c["inferredType"] == "mixed" and c.get("dateFormats"))
    ]


def _categories(profile: Profile, available: list[str], max_distinct: int = 60) -> list[str]:
    return [
        c["name"]
        for c in _cols(profile, available)
        if c["inferredType"] == "string" and not c.get("piiKind") and 1 < c["distinct"] <= max_distinct
    ]


def _with_blanks(profile: Profile, available: list[str]) -> list[dict[str, Any]]:
    return [c for c in _cols(profile, available) if c["nulls"] > 0 and not c.get("piiKind")]


def _key_columns(profile: Profile, available: list[str]) -> list[str]:
    for key in profile.get("candidateKeys", []):
        names = [part.strip() for part in re.sub(r"\(.*\)", "", key).split("+")]
        if all(name in available for name in names):
            return names
    return []


def _money_like(profile: Profile, available: list[str]) -> list[str]:
    """Text columns whose values are numbers written with currency symbols, spaces or decimal commas."""
    names = []
    for c in _cols(profile, available):
        if c.get("piiKind"):
            continue
        if c["inferredType"] in ("integer", "decimal", "mixed") and any(
            re.search(r"[R$€£\s,]", str(p.get("example", ""))) for p in c.get("patterns", [])
        ):
            names.append(c["name"])
    return names


# ------------------------------------------------------------------ default builders


def _d_none(profile: Profile, available: list[str]) -> dict[str, Any]:
    return {}


def _d_fill_unknown(profile: Profile, available: list[str]) -> dict[str, Any]:
    categorical = [
        c["name"]
        for c in _with_blanks(profile, available)
        if c["inferredType"] in ("string", "mixed", "boolean")
    ]
    return {"columns": categorical, "strategy": "constant", "value": "Unknown", "add_indicator": True}


def _d_drop_rows(profile: Profile, available: list[str]) -> dict[str, Any]:
    blanks = sorted(_with_blanks(profile, available), key=lambda c: c["nullPct"], reverse=True)
    return {"columns": [blanks[0]["name"]] if blanks else [], "how": "any"}


def _d_group_median(profile: Profile, available: list[str]) -> dict[str, Any]:
    numeric = [
        c for c in _with_blanks(profile, available) if c["inferredType"] in ("integer", "decimal", "mixed")
    ]
    categories = _categories(profile, available)
    group_by = None
    for column in numeric:
        depends = column.get("missingDependsOn")
        if depends in categories:
            group_by = depends
            break
    if group_by is None and categories:
        group_by = min(
            categories, key=lambda name: next(c["distinct"] for c in profile["columns"] if c["name"] == name)
        )
    return {"columns": [c["name"] for c in numeric], "strategy": "group_median", "group_by": group_by}


def _d_sparse(profile: Profile, available: list[str]) -> dict[str, Any]:
    return {"threshold": 0.5}


def _d_locf(profile: Profile, available: list[str]) -> dict[str, Any]:
    dates = _dates(profile, available)
    keys = _key_columns(profile, available)
    return {
        "columns": [c["name"] for c in _with_blanks(profile, available)][:3],
        "strategy": "ffill",
        "order_by": dates[0] if dates else None,
        "group_by": keys[0] if keys else None,
    }


def _d_numeric_model(strategy: str) -> Defaults:
    def build(profile: Profile, available: list[str]) -> dict[str, Any]:
        numeric = [
            c["name"] for c in _with_blanks(profile, available) if c["inferredType"] in ("integer", "decimal")
        ]
        return {"columns": numeric, "strategy": strategy, "add_indicator": False}

    return build


def _d_missforest(profile: Profile, available: list[str]) -> dict[str, Any]:
    return {
        "columns": [c["name"] for c in _with_blanks(profile, available)],
        "strategy": "missforest",
        "add_indicator": False,
    }


def _d_dedupe(keep: str) -> Defaults:
    def build(profile: Profile, available: list[str]) -> dict[str, Any]:
        keys = _key_columns(profile, available)
        dates = _dates(profile, available)
        params: dict[str, Any] = {"subset": keys, "keep": keep}
        if keep == "last" and dates:
            params["order_by"] = dates[0]
        return params

    return build


def _d_linkage(profile: Profile, available: list[str]) -> dict[str, Any]:
    names = _pii(profile, available, {"name"})
    if not names:
        names = [
            c["name"]
            for c in _cols(profile, available)
            if c["inferredType"] == "string" and c["distinct"] > 0.5 * max(profile["rows"], 1)
        ][:1]
    return {"columns": names, "threshold": 0.92, "action": "flag"}


def _outlier_columns(profile: Profile, available: list[str]) -> list[str]:
    flagged = [c["name"] for c in _cols(profile, available) if c.get("outliers") and not c.get("piiKind")]
    return flagged or _numeric(profile, available)


def _d_outlier(action: str) -> Defaults:
    def build(profile: Profile, available: list[str]) -> dict[str, Any]:
        params: dict[str, Any] = {
            "columns": _outlier_columns(profile, available),
            "method": "combined",
            "action": action,
        }
        if action == "cap_percentile":
            params.update({"lower": 0.01, "upper": 0.99})
        return params

    return build


def _d_iforest(profile: Profile, available: list[str]) -> dict[str, Any]:
    rows = max(profile["rows"], 1)
    extremes = profile["quality"].get("outliers", 0)
    contamination: float | str = round(min(0.2, max(extremes / rows, 0.001)), 4) if extremes else "auto"
    return {"columns": _numeric(profile, available), "contamination": contamination, "action": "flag"}


def _d_smooth(profile: Profile, available: list[str]) -> dict[str, Any]:
    numeric = _numeric(profile, available)
    dates = _dates(profile, available)
    return {"column": numeric[0] if numeric else None, "window": 5, "order_by": dates[0] if dates else None}


def _d_formats(profile: Profile, available: list[str]) -> dict[str, Any]:
    return {
        "dates": _dates(profile, available),
        "numbers": _money_like(profile, available),
        "day_first": True,
    }


def _d_text_phone(profile: Profile, available: list[str]) -> dict[str, Any]:
    text = [
        c["name"] for c in _cols(profile, available) if c["inferredType"] == "string" and not c.get("piiKind")
    ]
    return {"text_columns": text, "phone_columns": _pii(profile, available, {"phone"}), "case": "none"}


def _d_coerce(profile: Profile, available: list[str]) -> dict[str, Any]:
    types = {}
    for c in _cols(profile, available):
        if c.get("piiKind"):
            continue
        kind = c["inferredType"]
        if kind in ("integer", "decimal", "date", "boolean"):
            types[c["name"]] = kind
        elif kind == "mixed":
            types[c["name"]] = "date" if c.get("dateFormats") else "decimal"
    return {"types": types, "numeric_sentinels": False}


def _d_json(profile: Profile, available: list[str]) -> dict[str, Any]:
    for c in _cols(profile, available):
        if any(str(p.get("example", "")).startswith("{") for p in c.get("patterns", [])):
            return {"column": c["name"]}
    return {"column": None}


def _variant_column(profile: Profile, available: list[str]) -> str | None:
    candidates = [c for c in _cols(profile, available) if c.get("labelVariants")]
    if candidates:
        return str(max(candidates, key=lambda c: c["labelVariants"])["name"])
    categories = _categories(profile, available)
    return categories[0] if categories else None


def _initials(label: str) -> str:
    return "".join(word[0] for word in re.split(r"[\s-]+", label) if word).upper()


def _d_map(profile: Profile, available: list[str]) -> dict[str, Any]:
    column = _variant_column(profile, available)
    mapping: dict[str, str] = {}
    if column:
        entry = next(c for c in profile["columns"] if c["name"] == column)
        values = [v["value"] for v in entry.get("topValues", [])]
        official = values[0] if values else None
        by_key: dict[str, str] = {}
        for value in values:
            by_key.setdefault(fingerprint(value), value)
        for value in values:
            canonical = by_key[fingerprint(value)]
            if canonical != value:
                mapping[value] = canonical
            elif len(value) <= 4 and value.isupper() and official is not None:
                match = next((v for v in values if v != value and _initials(v).startswith(value[:2])), None)
                if match:
                    mapping[value] = match
    return {"column": column, "mapping": mapping}


def _d_fingerprint(profile: Profile, available: list[str]) -> dict[str, Any]:
    return {"column": _variant_column(profile, available)}


def _d_drop_pii(profile: Profile, available: list[str]) -> dict[str, Any]:
    return {"columns": _pii(profile, available, {"national_id", "payment_card"})}


_MONTH_NAMES = re.compile(
    r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*(_?\d{2,4})?$|^\d{4}[_-]?\d{2}$"
)


def _d_tidy(profile: Profile, available: list[str]) -> dict[str, Any]:
    wide = [name for name in available if _MONTH_NAMES.match(name.lower())]
    if len(wide) < 3:
        return {"id_columns": [], "value_columns": []}
    return {
        "id_columns": [name for name in available if name not in wide],
        "value_columns": wide,
        "var_name": "period",
        "value_name": "value",
    }


def _d_rules(profile: Profile, available: list[str]) -> dict[str, Any]:
    rules: list[dict[str, Any]] = []
    for c in _cols(profile, available):
        name = c["name"]
        if c.get("piiKind"):
            continue
        if c["inferredType"] in ("integer", "decimal") and c.get("min") is not None:
            try:
                minimum = float(str(c["min"]).replace(" ", ""))
            except ValueError:
                minimum = 0.0
            if minimum >= 0 or (c.get("outliers") or 0) > 0:
                rules.append({"column": name, "rule": "range", "min": 0, "name": f"{name} is not negative"})
        elif c["inferredType"] == "date" or c.get("dateFormats"):
            rules.append(
                {
                    "column": name,
                    "rule": "date_range",
                    "min": "1900-01-01",
                    "max": "2100-12-31",
                    "name": f"{name} is a plausible date",
                }
            )
        elif c["inferredType"] == "string" and 1 < c["distinct"] <= 15:
            allowed = [
                v["value"] for v in c.get("topValues", []) if v["count"] >= 0.005 * max(profile["rows"], 1)
            ]
            if allowed:
                rules.append(
                    {
                        "column": name,
                        "rule": "allowed",
                        "values": [*allowed, "Unknown"],
                        "name": f"{name} is on the allowed list",
                    }
                )
    keys = _key_columns(profile, available)
    if len(keys) == 1:
        rules.append({"column": keys[0], "rule": "unique", "name": f"{keys[0]} is unique"})
    return {"rules": rules, "action": "flag"}


def _d_sa_id(profile: Profile, available: list[str]) -> dict[str, Any]:
    ids = _pii(profile, available, {"national_id"})
    return {"column": ids[0] if ids else None, "action": "flag"}


def _d_orphans(profile: Profile, available: list[str]) -> dict[str, Any]:
    for check in profile.get("crossTable", []):
        if check["child"] in available:
            return {"column": check["child"], "reference": check.get("reference", check["parent"])}
    return {"column": None, "reference": None}


def _d_repair(profile: Profile, available: list[str]) -> dict[str, Any]:
    for dependency in profile.get("dependencies", []):
        if not dependency["hold"] and dependency["left"] in available and dependency["right"] in available:
            return {"determinant": dependency["left"], "dependent": dependency["right"], "min_share": 0.8}
    return {"determinant": None, "dependent": None}


def _d_reconcile(profile: Profile, available: list[str]) -> dict[str, Any]:
    params = _d_orphans(profile, available)
    return {**params, "threshold": 0.9}


def _d_keep_masked(profile: Profile, available: list[str]) -> dict[str, Any]:
    remaining = [name for name in _pii(profile, available) if name in available]
    return {"columns": remaining, "method": "redact"}


def _d_k_anon(profile: Profile, available: list[str]) -> dict[str, Any]:
    return {"quasi_identifiers": _categories(profile, available, max_distinct=30)[:3], "k": 5}


# ------------------------------------------------------------------ methods


METHODS: list[Method] = [
    Method(
        "profile-accept",
        "profile",
        "Accept the profile and continue",
        "everyday",
        "You have read the measurements. Nothing in bronze changes. We now know which steps are worth running.",
        "Persist the profile: column stats, pattern histogram, Little's test, discovered dependencies and orphan rates. Later steps read this; they do not re-guess.",
        "Skipping the profile means later methods can be invalid (for example, mean-filling blanks that are MNAR).",
        "accept_profile",
        _d_none,
        True,
    ),
    Method(
        "missing-unknown-indicator",
        "missing",
        "Fill with Unknown and add a missing flag",
        "everyday",
        "Empty category cells become the word Unknown. We also tick a box that says the value was missing, so a later model can still see that fact.",
        "Sentinel fill plus a missingness indicator. Valid for MAR when there is no reliable donor column yet. Prefer this over a silent mean fill.",
        "Do not mean-fill a category.",
        "fill_missing",
        _d_fill_unknown,
        True,
    ),
    Method(
        "missing-drop-rows",
        "missing",
        "Drop rows that are missing a value",
        "everyday",
        "Remove every row with a blank in the chosen column. Simple, but you lose those rows.",
        "Listwise deletion. Safe under MCAR; biased under MAR because it changes who is in the table.",
        "Wastes data and changes who is in the file. Only use if those rows are out of scope.",
        "drop_missing_rows",
        _d_drop_rows,
    ),
    Method(
        "missing-group-median",
        "missing",
        "Fill a number with the median of its group",
        "everyday",
        "A missing number takes the middle value of rows in the same group. That is fairer than one number for the whole file.",
        "Group-wise median imputation. Appropriate for MAR when the group explains the value. Median, not mean, because money is skewed.",
        "Shrinks variance a little. Do not use for MNAR income-style fields.",
        "fill_missing",
        _d_group_median,
    ),
    Method(
        "missing-column-drop",
        "missing",
        "Drop a column if it is mostly empty",
        "everyday",
        "If a column is empty in more than half the rows, we stop using it.",
        "Column deletion at a 50% missingness threshold.",
        "You cannot analyse a column you deleted. Check with a steward first.",
        "drop_sparse_columns",
        _d_sparse,
    ),
    Method(
        "missing-locf",
        "missing",
        "Carry the last value forward (time order)",
        "expert",
        "For a value that is written over time, copy the last known one into the gap.",
        "LOCF after a stable sort on the key and time. Dangerous if gaps are long.",
        "Not for a cross-section. Only when rows are a time series per entity.",
        "fill_missing",
        _d_locf,
    ),
    Method(
        "missing-mice",
        "missing",
        "Iterative imputation (MICE)",
        "expert",
        "Predict each blank number from the other numbers, round after round, until the guesses settle.",
        "Chained equations with scikit-learn's IterativeImputer, seeded so the recipe reruns identically. The stored table holds one completed draw; report uncertainty separately.",
        "Still wrong under MNAR without a model for why values are missing.",
        "fill_missing",
        _d_numeric_model("iterative"),
    ),
    Method(
        "missing-knn",
        "missing",
        "KNN imputation",
        "expert",
        "Fill a blank with the average of the most similar rows.",
        "Standardised features, then KNNImputer with k = 5.",
        "Slow on wide tables. Scale first or the distances lie.",
        "fill_missing",
        _d_numeric_model("knn"),
    ),
    Method(
        "missing-missforest",
        "missing",
        "MissForest",
        "expert",
        "A set of decision trees guesses each blank from the other columns.",
        "Random forests per column, for numbers and categories, seeded for a rerunnable recipe.",
        "Can overfit small tables.",
        "fill_missing",
        _d_missforest,
    ),
    Method(
        "dup-keep-first",
        "duplicates",
        "Keep the first exact match",
        "everyday",
        "If the same record appears twice, keep the first row and drop the rest.",
        "Exact key-based deduplication after normalising case and spaces. keep='first' is deterministic for a rerun.",
        "Does not catch spelling variants of the same person. Use linkage for that.",
        "deduplicate",
        _d_dedupe("first"),
        True,
    ),
    Method(
        "dup-keep-last",
        "duplicates",
        "Keep the most recent row",
        "everyday",
        "When two rows clash, keep the one that arrived last.",
        "Survivorship: most recent wins, ordered by the date column when there is one, else file order.",
        "Only if later rows are corrections, not accidental re-exports.",
        "deduplicate",
        _d_dedupe("last"),
    ),
    Method(
        "dup-linkage",
        "duplicates",
        "Near-duplicate linkage",
        "expert",
        "Find records that are probably the same even if the spelling differs, and mark them for review.",
        "Blocking, Jaro-Winkler similarity above 0.92, connected-component clustering. Flag first; merge only after review.",
        "False merges mix two people. Review the flagged clusters.",
        "link_records",
        _d_linkage,
    ),
    Method(
        "outlier-flag",
        "outliers",
        "Flag extreme values, do not delete them",
        "everyday",
        "Mark the strange values so you can see them later. Keep the rows. A huge value might be a real event.",
        "Tukey far fence (3 × IQR) or modified z-score (MAD) above 3.5. The preferred default for genuine rare events.",
        "Do not drop extremes by habit.",
        "handle_outliers",
        _d_outlier("flag"),
        True,
    ),
    Method(
        "outlier-winsor",
        "outliers",
        "Cap at the 1st and 99th percentile",
        "everyday",
        "Pull the most extreme values back to a high-but-possible number so they do not dominate an average.",
        "Winsorise. The bounds are recorded in the recipe.",
        "Hides the original value. Prefer flagging if the value might be real.",
        "handle_outliers",
        _d_outlier("cap_percentile"),
    ),
    Method(
        "outlier-iforest",
        "outliers",
        "Isolation Forest (several columns together)",
        "expert",
        "A model looks at several columns together and marks rows that are easy to separate from the rest.",
        "Isolation Forest with contamination set from the profile, seeded for reproducibility.",
        "Do not treat the score as a label of fraud.",
        "isolation_forest",
        _d_iforest,
    ),
    Method(
        "noise-median",
        "outliers",
        "Smooth spikes with a rolling median",
        "expert",
        "If a number jumps for one step and comes back, replace the jump with the middle of its neighbours.",
        "Centred rolling median over an ordered series (Hampel-style).",
        "Do not smooth a cross-section. Only for an ordered series.",
        "smooth_series",
        _d_smooth,
    ),
    Method(
        "format-date-number",
        "formats",
        "ISO dates and numbers without symbols",
        "everyday",
        "Read every date day-first and write it as 2026-03-14. Strip R and spaces from money, and treat a comma as a decimal when that is how it was written.",
        "Day-first parsing avoids 03/04 ambiguity. Locale-aware thousands (space or comma) and decimal commas.",
        "Never let the parser guess month-first on South African data.",
        "standardise_formats",
        _d_formats,
        True,
    ),
    Method(
        "format-unicode-phone",
        "formats",
        "Unicode, spacing and E.164 phones",
        "expert",
        "Trim spaces, fix broken accents, and write South African numbers as +27…",
        "Whitespace collapse, Unicode NFC, and E.164 phone formatting.",
        "Do not change the case of codes or identifiers.",
        "normalise_text_and_phones",
        _d_text_phone,
    ),
    Method(
        "type-coerce-sentinels",
        "types",
        "Coerce types and turn fake blanks into empty",
        "everyday",
        "N/A, -, and a lone question mark become empty. Then dates are dates and numbers are numbers.",
        "The sentinel list is recorded in the recipe; parse failures become blanks and are counted.",
        "0 can be a real value. Numeric sentinels such as 9999 are only removed if you choose that.",
        "coerce_types",
        _d_coerce,
        True,
    ),
    Method(
        "type-json",
        "types",
        "Flatten nested JSON",
        "expert",
        "If a cell holds a bundle of fields, split them into columns.",
        "json_normalize with a column prefix.",
        "Exploding lists multiplies rows.",
        "flatten_json",
        _d_json,
    ),
    Method(
        "cat-map",
        "categories",
        "Map variants to official names",
        "everyday",
        "Variants of the same label become one. We keep the list so the same fix happens next time.",
        "Deterministic mapping table first; fingerprint clusters for leftovers.",
        "Do not merge two real categories because their names look alike.",
        "map_values",
        _d_map,
        True,
    ),
    Method(
        "cat-fingerprint",
        "categories",
        "Cluster labels by fingerprint",
        "expert",
        "Group labels that are the same once punctuation, order and case are ignored, then keep the most common spelling.",
        "OpenRefine key collision on sorted lower-case tokens.",
        "Western Cape and Eastern Cape must not share a cluster; review before committing.",
        "cluster_labels",
        _d_fingerprint,
    ),
    Method(
        "struct-drop-pii",
        "structure",
        "Drop identifiers from the clean table",
        "everyday",
        "The clean table does not need national ID or card numbers. They stay in bronze if a steward must see the raw file.",
        "Prohibited column removal. Masked display labels such as names can stay masked.",
        "A later join that needs the identifier must go through a steward.",
        "drop_columns",
        _d_drop_pii,
        True,
    ),
    Method(
        "struct-tidy",
        "structure",
        "Reshape wide tables to tidy form",
        "expert",
        "If periods are separate columns, stack them so each row is one period.",
        "melt, with the identifier columns kept.",
        "Pivoting without a unique key duplicates facts.",
        "tidy_reshape",
        _d_tidy,
    ),
    Method(
        "validate-rules",
        "validate",
        "Run the rule pack and flag failures",
        "everyday",
        "We check values against rules: numbers not negative, dates plausible, labels on the allowed list, keys unique. Failing rows are marked, not silently dropped.",
        "Expectation suite: range, domain, regex, uniqueness and completeness rules suggested from the profile.",
        "A failed row stays in silver with a flag so a steward can see it.",
        "validate_rules",
        _d_rules,
        True,
    ),
    Method(
        "validate-sa-id",
        "validate",
        "South African ID check digit",
        "expert",
        "If an ID number is still present, test its check digit and embedded birth date.",
        "Luhn check digit plus a valid YYMMDD birth date. Raw values are never logged.",
        "Fail closed: a bad check digit is a flag, not a rewrite.",
        "validate_sa_id",
        _d_sa_id,
    ),
    Method(
        "constraint-flag-orphans",
        "constraints",
        "Flag orphan identifiers",
        "everyday",
        "If a row points at a record that is not on the master list, we mark it. We do not invent a record.",
        "Referential check against a permitted reference table.",
        "Guessing a missing key is forbidden without a steward.",
        "flag_orphans",
        _d_orphans,
        True,
    ),
    Method(
        "constraint-holoclean",
        "constraints",
        "Majority-vote constraint repair",
        "expert",
        "When two columns should agree and a few rows disagree, set those cells to the value most rows agree on.",
        "Repair near-holding functional dependencies by majority vote where the majority is at least 80%; flag the rest for review.",
        "Can hide a real change. Every repaired cell is flagged, and bronze keeps the original.",
        "repair_constraints",
        _d_repair,
    ),
    Method(
        "reconcile-master",
        "constraints",
        "Prefer the master record when sources clash",
        "expert",
        "Match values to the list we trust and use its spelling, noting any that do not match.",
        "Fingerprint then Jaro-Winkler match against the master list.",
        "Needs an approved master list.",
        "reconcile_master",
        _d_reconcile,
    ),
    Method(
        "privacy-keep-masked",
        "privacy",
        "Keep masks and record the recipe",
        "everyday",
        "Personal columns stay masked. Every accepted step is recorded so someone can replay the work.",
        "No new identifiers in silver; bronze immutable; recipe versioned; lineage bronze → silver.",
        "Do not export bronze to check a name.",
        "mask_columns",
        _d_keep_masked,
        True,
    ),
    Method(
        "privacy-k-anon",
        "privacy",
        "Suppress small groups (k-anonymity)",
        "expert",
        "Remove rows whose combination of details is shared by fewer than k people, so no row picks out a person.",
        "Suppression of equivalence classes smaller than k over the chosen quasi-identifiers.",
        "Harms utility. Use when data will leave the organisation.",
        "k_anonymity",
        _d_k_anon,
    ),
]

METHODS_BY_ID = {method.id: method for method in METHODS}


def methods_for(chapter_id: str) -> list[Method]:
    return [method for method in METHODS if method.chapter_id == chapter_id]


_ALWAYS_APPLICABLE = {"accept_profile", "drop_sparse_columns", "deduplicate"}


def is_applicable(method: Method, params: dict[str, Any]) -> bool:
    """A method applies when its defaults found something to act on in this table."""
    if method.operation in _ALWAYS_APPLICABLE:
        return True
    if method.operation == "map_values":
        return bool(params.get("mapping"))
    if any(isinstance(value, list | dict) and value for value in params.values()):
        return True
    return bool(params.get("column") or params.get("determinant"))


# ------------------------------------------------------------------ findings


def chapter_finding(chapter_id: str, profile: Profile) -> str:
    """What the profile found for this chapter, in plain words."""
    columns = profile.get("columns", [])
    quality = profile.get("quality", {})
    rows = max(profile.get("rows", 0), 1)
    if chapter_id == "profile":
        parts = [f"This file is {quality.get('completeness', 100)}% complete."]
        mixed = [c["name"] for c in columns if len(c.get("dateFormats") or {}) > 1]
        if mixed:
            parts.append(f"{', '.join(mixed)} mix date formats.")
        if quality.get("duplicates"):
            parts.append(f"{quality['duplicates']} rows repeat.")
        if quality.get("outliers"):
            parts.append(f"{quality['outliers']} values sit far outside the rest.")
        return " ".join(parts)
    if chapter_id == "missing":
        blanks = sorted((c for c in columns if c["nulls"]), key=lambda c: c["nullPct"], reverse=True)[:3]
        if not blanks:
            return "No column has blank cells."
        return " ".join(
            f"{c['name']} is blank in {c['nullPct']}% of rows ({c['missingness']}{': depends on ' + c['missingDependsOn'] if c.get('missingDependsOn') else ''})."
            for c in blanks
        )
    if chapter_id == "duplicates":
        keys = profile.get("candidateKeys", [])
        key_note = f" Likely key: {keys[0]}." if keys else ""
        return f"{quality.get('duplicates', 0)} rows repeat exactly.{key_note}"
    if chapter_id == "outliers":
        flagged = [c for c in columns if c.get("outliers")]
        if not flagged:
            return "No value sits far outside the usual range."
        return " ".join(
            f"{c['outliers']} {c['name']} values sit outside a robust fence (max {c.get('max')})."
            for c in flagged[:3]
        )
    if chapter_id == "formats":
        parts = []
        for c in columns:
            formats = c.get("dateFormats") or {}
            if len(formats) > 1:
                parts.append(f"{c['name']} mixes {', '.join(list(formats)[:3])}.")
        for c in columns:
            if c["inferredType"] in ("integer", "decimal", "mixed") and any(
                re.search(r"[R$€£\s,]", str(p.get("example", ""))) for p in c.get("patterns", [])
            ):
                parts.append(f"{c['name']} is written with currency symbols, spaces or decimal commas.")
        return " ".join(parts) or "Formats are already consistent."
    if chapter_id == "types":
        typed = [
            f"{c['name']} looks like {c['inferredType']}"
            for c in columns
            if c["inferredType"] not in ("string", "empty") and not c.get("piiKind")
        ]
        sentinels = sum(c.get("sentinels", 0) for c in columns)
        note = f" {sentinels} cells hold placeholders such as N/A or -." if sentinels else ""
        return (", ".join(typed[:4]) + " but is stored as text." if typed else "Every column is text.") + note
    if chapter_id == "categories":
        variants = [c for c in columns if c.get("labelVariants")]
        if not variants:
            return "No label has an obvious spelling variant."
        return " ".join(
            f"{c['name']} has {c['labelVariants']} groups of variant spellings." for c in variants[:3]
        )
    if chapter_id == "structure":
        pii = [f"{c['name']} ({c['piiKind'].replace('_', ' ')})" for c in columns if c.get("piiKind")]
        return (
            f"Personal columns: {', '.join(pii)}. They are masked; identifiers should not reach silver."
            if pii
            else "No personal column was detected. The table shape looks tidy."
        )
    if chapter_id == "validate":
        return "Rules suggested from the profile: numbers not negative, dates plausible, labels on their observed lists, keys unique."
    if chapter_id == "constraints":
        parts = [
            f"{check['orphanPct']}% of {check['child']} values are not in {check['parent']}."
            for check in profile.get("crossTable", [])
        ]
        near = [d for d in profile.get("dependencies", []) if not d["hold"]]
        if near:
            parts.append(
                f"{near[0]['left']} → {near[0]['right']} nearly holds ({near[0]['violations']} rows disagree)."
            )
        return " ".join(parts) or "No contradiction between columns was found."
    if chapter_id == "privacy":
        pii = [c["name"] for c in columns if c.get("piiKind")]
        return (
            f"{', '.join(pii)} {'is' if len(pii) == 1 else 'are'} masked. The residual scan checks nothing personal leaks into silver."
            if pii
            else f"No personal information was found in {thousands(rows)} rows."
        )
    return ""
