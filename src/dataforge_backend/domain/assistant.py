"""The assistant's planner: ground a plain-English question in a shared metric, or refuse.

The planner never writes SQL. It chooses a dataset, a measure, a method, a grouping, filters and a period
from the fields the member is permitted to read; the analysis engine then composes the SQL. A language
model can propose the plan (see `plan_schema`), but its proposal is validated here exactly like the
rule-based one, so a prompt cannot reach data or fields outside the member's catalogue.

A question can follow up the member's previous answer ("what about Gauteng?", "and by course?"): it keeps
what the previous plan chose and changes only what the new question names.
"""

import json
import re
from dataclasses import dataclass, field, replace
from typing import Any

from .analysis.model import MONTH_DIMENSION, AnalysisDataset

_INJECTION = re.compile(
    r"(ignore|disregard|forget|override)\s+(all\s+|any\s+|the\s+|your\s+|previous\s+|prior\s+|above\s+)*"
    r"(instructions|rules|prompts?|guidelines|permissions|policies)"
    r"|system\s+prompt|developer\s+message|you\s+are\s+now|act\s+as\s+(an?\s+)?(admin|administrator|root|dba|steward)"
    r"|jailbreak|\bDAN\b|pretend\s+(that\s+)?you|reveal\s+(your|the)\s+(prompt|instructions|rules)"
    r"|bypass\s+(the\s+)?(masking|security|permissions|policy|policies|filters?)"
    r"|;\s*(drop|delete|truncate|alter|grant|update|insert)\s|\bunion\s+select\b|--\s*$",
    re.IGNORECASE,
)
_PERSONAL = re.compile(
    r"\b(who\s+(is|are|was|were)|whose|full[_\s]?names?|names?\s+of|id\s*numbers?|identity\s+numbers?|e-?mails?|"
    r"phone|cell\s*numbers?|addresses|home\s+address|which\s+(learners|students|people|members|individuals|customers)|"
    r"list\s+(all\s+|the\s+)?(learners|students|people|members|individuals|customers)|personal\s+(data|details|information))\b",
    re.IGNORECASE,
)
_PREDICT = re.compile(
    r"\b(likely|likelihood|predict\w*|forecast\w*|probabilit\w*|risk\s+of|at\s+risk|next\s+(week|month|quarter|year)|"
    r"will\s+(\w+\s+){1,4}(drop|fall|rise|grow|increase|decrease|leave|churn|complete|fail))\b",
    re.IGNORECASE,
)
_RELATIONSHIP = re.compile(
    r"\b(correlat\w*|relationship|related\s+to|associated\s+with|move\s+together)\b", re.IGNORECASE
)
_EXPLAIN = re.compile(
    r"\b(why\s+(did|has|is|was)|what\s+(caused|drove|explains)|root\s+cause|reason\s+for)\b", re.IGNORECASE
)
_TREND = re.compile(
    r"\b(trend\w*|over\s+time|chang(e|ed|es|ing)|month\s+by\s+month|monthly|since|growth|grow\w*|grew|rise|rose|risen|"
    r"fall|fell|fallen|increas\w*|decreas\w*|moved|movement|progress\w*)\b",
    re.IGNORECASE,
)
# Words that open a follow-up to the previous answer rather than a new question.
_FOLLOW_UP = re.compile(
    r"^(and|but|now|then|also|only|just|what\s+about|how\s+about|same\s+for)\b|\binstead\b|\bthe\s+same\b",
    re.IGNORECASE,
)
_MONTH_WORDS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
_NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "nine": 9, "twelve": 12}

REFUSAL_REASONS = ("personal", "prediction", "relationship", "explain", "ungrounded", "injection")


@dataclass(frozen=True)
class AssistantPlan:
    dataset_id: str
    measure: str
    method: str  # compare | trend | relationship | explain
    dimension: str | None = None
    split: str | None = None
    filters: dict[str, str] = field(default_factory=dict)
    date_from: str | None = None
    date_to: str | None = None
    figure: bool = False  # a single figure: the latest month against the one before
    source: str = "rules"  # rules | model
    measure_y: str | None = None  # relationship: the second measure (None lets the engine pick a partner)
    period_a: str | None = None  # explain: the two months whose difference is split into contributions
    period_b: str | None = None
    follows_up: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "datasetId": self.dataset_id,
            "measure": self.measure,
            "method": self.method,
            "dimension": self.dimension,
            "split": self.split,
            "filters": dict(self.filters),
            "from": self.date_from,
            "to": self.date_to,
            "figure": self.figure,
            "source": self.source,
            "measureY": self.measure_y,
            "periodA": self.period_a,
            "periodB": self.period_b,
            "followsUp": self.follows_up,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssistantPlan | None":
        """A stored plan, or None when it is not one (older records lack the newer keys, which is fine)."""
        try:
            return cls(
                dataset_id=str(data["datasetId"]),
                measure=str(data["measure"]),
                method=str(data["method"]),
                dimension=data.get("dimension"),
                split=data.get("split"),
                filters={str(k): str(v) for k, v in (data.get("filters") or {}).items()},
                date_from=data.get("from"),
                date_to=data.get("to"),
                figure=bool(data.get("figure")),
                source=str(data.get("source") or "rules"),
                measure_y=data.get("measureY"),
                period_a=data.get("periodA"),
                period_b=data.get("periodB"),
            )
        except (KeyError, TypeError, AttributeError):
            return None


@dataclass(frozen=True)
class Refusal:
    reason: str
    answer: str
    explanation: str


def refusal(reason: str, catalogue: "Catalogue | None" = None) -> Refusal:
    examples = catalogue.example_questions() if catalogue else []
    try_this = f" Try: {examples[0].lower().rstrip('?')}?" if examples else ""
    if reason == "personal":
        return Refusal(
            reason,
            "I can't answer that from the data you are permitted to see. Personal information about individual "
            "people is masked for your role, and answers describe groups rather than people.",
            f"I refused rather than guessing.{try_this}",
        )
    if reason == "prediction":
        return Refusal(
            reason,
            "I can't answer that here. Likelihoods are not a defined metric in the semantic layer, and learner-level "
            "personal data is masked for your role.",
            "Use Predict an outcome in Analyse, which scores a cohort with a documented factor breakdown. Or ask a "
            f"Steward to define the metric you need.{try_this}",
        )
    if reason == "relationship":
        return Refusal(
            reason,
            "I could not find two measures from the same dataset to test against each other.",
            "Name both, or open Analyse, choose Test a relationship, and pick the two measures.",
        )
    if reason == "explain":
        return Refusal(
            reason,
            "I can explain a change in a total, such as revenue or enrolments, by splitting it into each group's "
            "contribution. That question is about a rate or a dataset without months, so there is no total to split.",
            "Ask why a total changed, or open the alert in Monitors for its root cause.",
        )
    if reason == "injection":
        return Refusal(
            reason,
            "I can't follow instructions that try to change my rules or permissions. Your role decides what I can read, "
            "and nothing in a question can change that.",
            f"This attempt was recorded.{try_this}",
        )
    return Refusal(
        "ungrounded",
        "I can't ground that request in a defined metric or in data you are allowed to see, so I won't guess at a number.",
        f"Try asking about {catalogue.metric_names() if catalogue else 'a shared metric'}. Those are defined in the semantic layer.",
    )


@dataclass(frozen=True)
class MeasureEntry:
    dataset: AnalysisDataset
    measure: str
    phrases: tuple[str, ...]
    governed: bool


class Catalogue:
    """The datasets and measures one member may ask about, with the words that name them."""

    def __init__(self, datasets: list[AnalysisDataset], synonyms: dict[str, list[str]] | None = None) -> None:
        self.datasets = {d.id: d for d in datasets}
        synonyms = synonyms or {}
        entries: list[MeasureEntry] = []
        for dataset in datasets:
            for m in dataset.measures.values():
                phrases = {m.label.lower(), m.id.replace("_", " ")}
                phrases.update(s.lower() for s in synonyms.get(m.id, []))
                if m.kind == "count" and m.id == "row_count":
                    phrases = {"rows", "row count", "number of rows"}
                entries.append(
                    MeasureEntry(dataset, m.id, tuple(sorted(p for p in phrases if p)), m.governed)
                )
        self.entries = entries

    def metric_names(self) -> str:
        names = list(
            dict.fromkeys(e.dataset.measures[e.measure].label.lower() for e in self.entries if e.governed)
        )
        if not names:
            return "a shared metric"
        return ", ".join(names[:-1]) + (f" or {names[-1]}" if len(names) > 1 else names[0])

    def example_questions(self) -> list[str]:
        out: list[str] = []
        for entry in self.entries:
            if not entry.governed:
                continue
            m = entry.dataset.measures[entry.measure]
            dims = list(entry.dataset.dimensions)
            if dims:
                out.append(f"What is {m.label.lower()} by {entry.dataset.dimensions[dims[0]].label.lower()}?")
            if entry.dataset.has_time:
                out.append(f"How has {m.label.lower()} changed over time?")
        return list(dict.fromkeys(out))


def detect_injection(text: str) -> bool:
    return bool(_INJECTION.search(text))


def _has_phrase(text: str, phrase: str) -> bool:
    words = [re.escape(w) for w in phrase.split()]
    if not words:
        return False
    # Allow a plural or possessive ending on the last word: "enrolment" matches "enrolments".
    pattern = r"\b" + r"\s+".join(words) + r"(s|es|'s)?\b"
    return re.search(pattern, text) is not None


def _match_measure(text: str, catalogue: Catalogue) -> list[tuple[int, MeasureEntry]]:
    scored: list[tuple[int, MeasureEntry]] = []
    for entry in catalogue.entries:
        best = max((len(p) for p in entry.phrases if _has_phrase(text, p)), default=0)
        if best:
            scored.append((best + (1000 if entry.governed else 0), entry))
    scored.sort(key=lambda s: -s[0])
    return scored


def _match_dimension(text: str, dataset: AnalysisDataset) -> str | None:
    for dim_id, dim in dataset.dimensions.items():
        names = {dim.label.lower(), dim.plural.lower(), dim_id.replace("_", " ")}
        for name in names:
            if re.search(
                rf"\b(by|per|each|every|across|which|what|top|between)\s+(the\s+)?{re.escape(name)}\b", text
            ):
                return dim_id
            if re.search(rf"\b{re.escape(name)}\s+(breakdown|split|comparison|ranking)\b", text):
                return dim_id
    if re.search(r"\b(by|per|each)\s+month\b", text) and dataset.has_time:
        return MONTH_DIMENSION
    return None


def _match_filters(text: str, dataset: AnalysisDataset) -> dict[str, str]:
    found: dict[str, str] = {}
    for dim_id, dim in dataset.dimensions.items():
        for value in sorted(dim.values, key=len, reverse=True):
            if len(value) >= 3 and re.search(rf"(?<![\w-]){re.escape(value.lower())}(?![\w-])", text):
                found[dim_id] = value
                break
    return found


def _period(text: str, dataset: AnalysisDataset) -> tuple[str | None, str | None, bool]:
    """(from, to, single month asked for)."""
    months = list(dataset.months)
    if not months:
        return None, None, False
    latest = months[-1]
    if re.search(r"\b(this|last|latest|current|past)\s+month\b|\bright\s+now\b|\bcurrently\b|\bnow\b", text):
        return latest, latest, True
    match = re.search(
        r"\b(?:last|past|previous)\s+(\d+|two|three|four|five|six|nine|twelve)\s+months\b", text
    )
    if match:
        raw = match.group(1)
        n = int(raw) if raw.isdigit() else _NUMBER_WORDS[raw]
        return months[max(0, len(months) - n)], latest, False
    if re.search(r"\b(last|this|previous|past|latest)\s+quarter\b", text):
        return months[max(0, len(months) - 3)], latest, False
    if re.search(r"\b(this|the\s+current)\s+year\b|\byear\s+to\s+date\b|\bytd\b", text):
        year = latest[:4]
        return next(m for m in months if m.startswith(year)), latest, False
    match = re.search(
        r"\b(?:between|from)\s+([a-z]+)(?:\s+(\d{4}))?\s+(?:and|to)\s+([a-z]+)(?:\s+(\d{4}))?", text
    )
    if match and match.group(1) in _MONTH_WORDS and match.group(3) in _MONTH_WORDS:
        low = _resolve_month(match.group(1), match.group(2), months)
        high = _resolve_month(match.group(3), match.group(4), months)
        if low and high:
            return (low, high, False) if low <= high else (high, low, False)
    match = re.search(r"\bsince\s+([a-z]+)(?:\s+(\d{4}))?", text)
    if match and match.group(1) in _MONTH_WORDS:
        month = _resolve_month(match.group(1), match.group(2), months)
        if month:
            return month, latest, False
    match = re.search(r"\bin\s+([a-z]+)(?:\s+(\d{4}))?\b", text)
    if match and match.group(1) in _MONTH_WORDS:
        month = _resolve_month(match.group(1), match.group(2), months)
        if month:
            return month, month, True
    return None, None, False


def _resolve_month(word: str, year: str | None, months: list[str]) -> str | None:
    number = f"{_MONTH_WORDS[word]:02d}"
    candidates = [m for m in months if m.endswith(f"-{number}") and (year is None or m.startswith(year))]
    return candidates[-1] if candidates else None


def _mentioned_measures(text: str, dataset: AnalysisDataset, catalogue: Catalogue) -> list[str]:
    """The dataset's measures the question names, in the order it names them.

    Where two phrases overlap ("completion rate" and "completion"), the longer one wins.
    """
    spans: list[tuple[int, int, bool, str]] = []
    for entry in catalogue.entries:
        if entry.dataset.id != dataset.id:
            continue
        for phrase in entry.phrases:
            words = [re.escape(w) for w in phrase.split()]
            if not words:
                continue
            pattern = r"\b" + r"\s+".join(words) + r"(s|es|'s)?\b"
            spans += [(m.start(), m.end(), entry.governed, entry.measure) for m in re.finditer(pattern, text)]
    kept: list[tuple[int, int, bool, str]] = []
    for span in sorted(spans, key=lambda s: (-(s[1] - s[0]), not s[2], s[0])):
        if not any(span[0] < k[1] and k[0] < span[1] for k in kept):
            kept.append(span)
    return list(dict.fromkeys(s[3] for s in sorted(kept)))


@dataclass(frozen=True)
class _Asked:
    """What a question asks for, read from its words against one dataset."""

    dimension: str | None
    filters: dict[str, str]
    date_from: str | None
    date_to: str | None
    single: bool
    trend: bool
    relationship: bool
    explain: bool

    @property
    def names_anything(self) -> bool:
        return bool(
            self.dimension
            or self.filters
            or self.date_from
            or self.trend
            or self.relationship
            or self.explain
        )


def _read(text: str, dataset: AnalysisDataset) -> _Asked:
    date_from, date_to, single = _period(text, dataset)
    return _Asked(
        dimension=_match_dimension(text, dataset),
        filters=_match_filters(text, dataset),
        date_from=date_from,
        date_to=date_to,
        single=single,
        trend=bool(_TREND.search(text)) and dataset.has_time,
        relationship=bool(_RELATIONSHIP.search(text)),
        explain=bool(_EXPLAIN.search(text)),
    )


def _explain_plan(
    dataset: AnalysisDataset,
    measure: str,
    asked: _Asked,
    catalogue: Catalogue,
    periods: tuple[str | None, str | None] = (None, None),
) -> AssistantPlan | Refusal:
    """Split a change in a total between two months into each group's contribution."""
    if not dataset.has_time or not dataset.measures[measure].additive:
        return refusal("explain", catalogue)
    months = list(dataset.months)
    if asked.date_from and asked.date_to and asked.date_from != asked.date_to:
        a, b = asked.date_from, asked.date_to
    elif asked.date_to or not periods[1]:
        end = asked.date_to or months[-1]
        index = months.index(end) if end in months else len(months) - 1
        a, b = months[max(0, index - 1)], months[index]
    else:
        a, b = periods[0] or months[0], periods[1]
    if a == b:
        return refusal("explain", catalogue)
    groupable = [d for d in dataset.dimensions if d not in asked.filters]
    dimension = asked.dimension if asked.dimension in groupable else next(iter(groupable), None)
    if dimension is None:
        return refusal("explain", catalogue)
    return AssistantPlan(
        dataset.id, measure, "explain", dimension, None, asked.filters, None, None, period_a=a, period_b=b
    )


def _compose(
    dataset: AnalysisDataset,
    measure: str,
    asked: _Asked,
    catalogue: Catalogue,
    *,
    measure_y: str | None = None,
    split: str | None = None,
    periods: tuple[str | None, str | None] = (None, None),
) -> AssistantPlan | Refusal:
    """The plan for a question once its dataset and measure are known."""
    if asked.relationship:
        y = measure_y if measure_y != measure else None
        return AssistantPlan(
            dataset.id,
            measure,
            "relationship",
            None,
            None,
            asked.filters,
            asked.date_from,
            asked.date_to,
            measure_y=y,
        )
    if asked.explain:
        return _explain_plan(dataset, measure, asked, catalogue, periods)
    dimension, filters = asked.dimension, asked.filters
    if dimension == MONTH_DIMENSION or (asked.trend and not asked.single):
        lines = dimension if dimension not in (None, MONTH_DIMENSION) else split
        lines = lines if lines not in filters else None
        return AssistantPlan(
            dataset.id, measure, "trend", None, lines, filters, asked.date_from, asked.date_to
        )
    if dimension and dimension not in filters:
        return AssistantPlan(
            dataset.id, measure, "compare", dimension, None, filters, asked.date_from, asked.date_to
        )
    if dataset.has_time:
        # A single figure: the chosen month (or the latest) against the month before it.
        months = list(dataset.months)
        end = asked.date_to or months[-1]
        index = months.index(end) if end in months else len(months) - 1
        start = months[max(0, index - 1)]
        return AssistantPlan(dataset.id, measure, "trend", None, None, filters, start, end, figure=True)
    first_dim = next((d for d in dataset.dimensions if d not in filters), None)
    if first_dim is None:
        return refusal("ungrounded", catalogue)
    return AssistantPlan(dataset.id, measure, "compare", first_dim, None, filters, None, None)


def _continue(
    dataset: AnalysisDataset,
    measure: str,
    asked: _Asked,
    previous: AssistantPlan,
    catalogue: Catalogue,
) -> AssistantPlan | Refusal:
    """A follow-up: keep what the previous plan chose and change only what the new question names."""
    if measure not in dataset.measures:
        return refusal("ungrounded", catalogue)
    filters = {**previous.filters, **asked.filters}
    if asked.date_from:
        date_from, date_to = asked.date_from, asked.date_to
    elif previous.figure:
        # A figure's two months were derived from its end month; carry the end month only.
        date_from, date_to = None, previous.date_to
    else:
        date_from, date_to = previous.date_from, previous.date_to
    named_method = asked.relationship or asked.explain or asked.trend or asked.dimension == MONTH_DIMENSION
    dimension = asked.dimension
    if not named_method and dimension is None and previous.method == "compare":
        dimension = previous.dimension
    merged = replace(
        asked,
        dimension=dimension,
        filters=filters,
        date_from=date_from,
        date_to=date_to,
        relationship=asked.relationship or (not named_method and previous.method == "relationship"),
        explain=asked.explain or (not named_method and previous.method == "explain"),
        trend=asked.trend
        or (
            not named_method
            and asked.dimension is None
            and previous.method == "trend"
            and not previous.figure
        ),
    )
    if merged.explain and previous.method == "explain" and dimension is None:
        merged = replace(merged, dimension=previous.dimension)
    planned = _compose(
        dataset,
        measure,
        merged,
        catalogue,
        measure_y=previous.measure_y if measure == previous.measure else previous.measure,
        # "How has it changed?" after a comparison by region draws one line per region.
        split=previous.split or (previous.dimension if previous.method == "compare" else None),
        periods=(previous.period_a, previous.period_b) if not asked.date_from else (None, None),
    )
    return replace(planned, follows_up=True) if isinstance(planned, AssistantPlan) else planned


def plan_question(
    question: str, catalogue: Catalogue, previous: AssistantPlan | None = None
) -> AssistantPlan | Refusal:
    """Rule-based planning. Deterministic, so it is also the fallback when no language model is configured.

    `previous` is the member's last answered plan in this conversation, for follow-up questions.
    """
    text = " ".join(question.lower().split())
    if detect_injection(question):
        return refusal("injection", catalogue)
    if _PERSONAL.search(text):
        return refusal("personal", catalogue)
    if _PREDICT.search(text):
        return refusal("prediction", catalogue)

    earlier = catalogue.datasets.get(previous.dataset_id) if previous else None
    matches = _match_measure(text, catalogue)
    if not matches:
        # "What about Gauteng?" names no measure: it can only continue the previous answer.
        if previous is None or earlier is None:
            return refusal("ungrounded", catalogue)
        asked = _read(text, earlier)
        if not asked.names_anything:
            return refusal("ungrounded", catalogue)
        return _continue(earlier, previous.measure, asked, previous, catalogue)

    top_score = matches[0][0]
    tied = [entry for score, entry in matches if score == top_score]
    wants_relationship = bool(_RELATIONSHIP.search(text))

    # Among datasets offering the same metric, prefer the one that has the grouping, values or (for a
    # relationship) the second measure asked about.
    def fit(entry: MeasureEntry) -> int:
        score = (2 if _match_dimension(text, entry.dataset) else 0) + len(_match_filters(text, entry.dataset))
        if wants_relationship:
            score += 3 * len(_mentioned_measures(text, entry.dataset, catalogue))
        return score

    entry = max(tied, key=fit)
    dataset = entry.dataset
    asked = _read(text, dataset)
    if previous is not None and earlier is not None and dataset.id == earlier.id and _FOLLOW_UP.search(text):
        return _continue(dataset, entry.measure, asked, previous, catalogue)
    measure, measure_y = entry.measure, None
    if asked.relationship:
        named = _mentioned_measures(text, dataset, catalogue)
        if entry.measure not in named:
            named.insert(0, entry.measure)
        measure, measure_y = named[0], (named[1] if len(named) > 1 else None)
    return _compose(dataset, measure, asked, catalogue, measure_y=measure_y)


def plan_schema(catalogue: Catalogue) -> dict[str, Any]:
    """JSON schema for a language model's proposed plan. Enums hold only fields the member may read."""
    dataset_ids = sorted(catalogue.datasets)
    measures = sorted({e.measure for e in catalogue.entries})
    dims = sorted({d for ds in catalogue.datasets.values() for d in ds.dimension_ids()})
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "action",
            "refusalReason",
            "datasetId",
            "measure",
            "measureY",
            "method",
            "dimension",
            "filters",
            "period",
            "followsUp",
        ],
        "properties": {
            "action": {"type": "string", "enum": ["answer", "refuse"]},
            "refusalReason": {"type": ["string", "null"], "enum": [*REFUSAL_REASONS, None]},
            "datasetId": {"type": ["string", "null"], "enum": [*dataset_ids, None]},
            "measure": {"type": ["string", "null"], "enum": [*measures, None]},
            "measureY": {"type": ["string", "null"], "enum": [*measures, None]},
            "method": {
                "type": ["string", "null"],
                "enum": ["figure", "compare", "trend", "relationship", "explain", None],
            },
            "dimension": {"type": ["string", "null"], "enum": [*dims, None]},
            "followsUp": {"type": "boolean"},
            "filters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["dimension", "value"],
                    "properties": {
                        "dimension": {"type": "string", "enum": dims or [""]},
                        "value": {"type": "string"},
                    },
                },
            },
            "period": {
                "type": "object",
                "additionalProperties": False,
                "required": ["kind", "months", "from", "to"],
                "properties": {
                    "kind": {"type": "string", "enum": ["all", "latest_month", "last_n_months", "range"]},
                    "months": {"type": ["integer", "null"]},
                    "from": {"type": ["string", "null"]},
                    "to": {"type": ["string", "null"]},
                },
            },
        },
    }


def catalogue_prompt(catalogue: Catalogue) -> str:
    """The grounding a language model sees: permitted datasets, measures, dimensions and their values."""
    lines: list[str] = []
    for dataset in catalogue.datasets.values():
        lines.append(f"Dataset {dataset.id} ({dataset.name})")
        for m in dataset.measures.values():
            tag = "shared metric" if m.governed else "column total"
            adds = "adds up" if m.additive else "rate, does not add up"
            lines.append(f"  measure {m.id}: {m.label} [{m.unit}, {tag}, {adds}] {m.description}".rstrip())
        for d in dataset.dimensions.values():
            values = ", ".join(d.values[:25])
            lines.append(f"  dimension {d.id}: {d.label} (values: {values})")
        if dataset.has_time:
            lines.append(f"  months available: {dataset.months[0]} to {dataset.months[-1]}")
    return "\n".join(lines)


def previous_prompt(question: str, plan: AssistantPlan) -> str:
    """The previous turn as a language model sees it, so a follow-up can keep what it does not change."""
    return f"Question: {question}\nPlan: {json.dumps(plan.to_dict(), sort_keys=True)}"


def plan_from_model(
    data: dict[str, Any], catalogue: Catalogue, previous: AssistantPlan | None = None
) -> AssistantPlan | Refusal:
    """Validate a language model's proposal against the catalogue, exactly like a rule-based plan."""
    if data.get("action") == "refuse":
        reason = data.get("refusalReason")
        return refusal(reason if reason in REFUSAL_REASONS else "ungrounded", catalogue)
    dataset = catalogue.datasets.get(str(data.get("datasetId")))
    measure = str(data.get("measure"))
    if dataset is None or measure not in dataset.measures:
        return refusal("ungrounded", catalogue)
    filters: dict[str, str] = {}
    for item in data.get("filters") or []:
        dim, value = str(item.get("dimension")), str(item.get("value"))
        if dim not in dataset.dimensions:
            return refusal("ungrounded", catalogue)
        match = next((v for v in dataset.dimensions[dim].values if v.lower() == value.lower()), None)
        if match is None:
            return refusal("ungrounded", catalogue)
        filters[dim] = match

    months = list(dataset.months)
    period = data.get("period") or {}
    kind = period.get("kind", "all")
    date_from = date_to = None
    if months:
        if kind == "latest_month":
            date_from = date_to = months[-1]
        elif kind == "last_n_months":
            n = max(1, min(int(period.get("months") or 3), len(months)))
            date_from, date_to = months[-n], months[-1]
        elif kind == "range":
            low, high = period.get("from"), period.get("to")
            date_from = low if low in months else months[0]
            date_to = high if high in months else months[-1]

    method = data.get("method") or "figure"
    dimension = data.get("dimension")
    if dimension is not None and dimension not in dataset.dimension_ids():
        return refusal("ungrounded", catalogue)
    follows = bool(data.get("followsUp")) and previous is not None

    def mark(plan: AssistantPlan) -> AssistantPlan:
        return replace(plan, source="model", follows_up=follows)

    if method == "relationship":
        measure_y = data.get("measureY")
        if measure_y is not None and (measure_y not in dataset.measures or measure_y == measure):
            measure_y = None
        return mark(
            AssistantPlan(
                dataset.id,
                measure,
                "relationship",
                None,
                None,
                filters,
                date_from,
                date_to,
                measure_y=measure_y,
            )
        )
    if method == "explain":
        asked = _Asked(
            dimension if dimension in dataset.dimensions else None,
            filters,
            date_from,
            date_to,
            date_from == date_to,
            False,
            False,
            True,
        )
        explained = _explain_plan(dataset, measure, asked, catalogue)
        return mark(explained) if isinstance(explained, AssistantPlan) else explained
    if method == "trend" and dataset.has_time:
        split = dimension if dimension in dataset.dimensions and dimension not in filters else None
        return mark(AssistantPlan(dataset.id, measure, "trend", None, split, filters, date_from, date_to))
    if method == "compare" and dimension in dataset.dimensions:
        return mark(
            AssistantPlan(dataset.id, measure, "compare", dimension, None, filters, date_from, date_to)
        )
    if dataset.has_time:
        end = date_to or months[-1]
        index = months.index(end)
        figure = AssistantPlan(
            dataset.id,
            measure,
            "trend",
            None,
            None,
            filters,
            months[max(0, index - 1)],
            end,
            figure=True,
        )
        return mark(figure)
    return refusal("ungrounded", catalogue)
