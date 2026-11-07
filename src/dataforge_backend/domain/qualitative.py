"""Qualitative questions: how categories relate, what free text says, and what the dataset itself holds.

The analysis engine answers questions that reduce to a number. These answer the rest: a cross-tab of two
categories with a chi-square test of independence, themes and sentiment in a free-text column, and questions
about the dataset that its profile answers. Everything here is pure: the application layer runs the queries.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import pandas as pd
from scipy import stats

from .values import thousands

MAX_CATEGORIES = 30  # distinct values a column may have to be cross-tabbed
_TABLE_ROWS = 12  # categories shown down the side of a cross-tab; the rest become "Other"
_TABLE_COLUMNS = 6  # categories shown across the top
THEME_SAMPLE = 150

_CROSSTAB_WORDS = re.compile(
    r"\b(relat(ed|ion|ionship)|associat\w*|link(ed)?|connect\w*|depend\w*|independen\w*|cross.?tab\w*"
    r"|break(s)? ?down|broken down|distribut\w*|split|against|versus|vs\.?|differ\w*|mix|make.?up|proportion\w*)\b",
    re.IGNORECASE,
)
_THEME_WORDS = re.compile(
    r"\b(theme\w*|topic\w*|sentiment|opinion\w*|feel(ing|s)?|say(ing|s)?|said|wr(ote|ite|iting)|comment\w*"
    r"|feedback|complain\w*|complaint\w*|review\w*|mention\w*|text|responses?|answers?|keywords?|words?)\b",
    re.IGNORECASE,
)
_GENERAL = re.compile(
    r"\b(about|describe|description|overview|summar(y|ise|ize)|tell me (about|more)|what('s| is) in"
    r"|look like|explore|insights?)\b",
    re.IGNORECASE,
)
_STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before being below between
    both but by can could did do does doing down during each few for from further had has have having he her here
    hers him his how i if in into is it its just me more most my no nor not now of off on once only or other our
    out over own same she should so some such than that the their them then there these they this those through to
    too under until up very was we were what when where which while who whom why will with would you your yours
    also get got really one two much many us im ive dont didnt cant wont isnt its thats very""".split()
)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_URL = re.compile(r"https?://\S+|www\.\S+")
_LONG_NUMBER = re.compile(r"\+?\d[\d\s()-]{7,}\d")


@dataclass(frozen=True)
class Route:
    kind: str  # crosstab | themes | profile | overview
    column_a: str | None = None
    column_b: str | None = None
    text_column: str | None = None


def is_general(question: str) -> bool:
    """A question about what the dataset is in general, answered with its overview."""
    return bool(_GENERAL.search(question))


_PROFILE_WORDS = re.compile(
    r"\b(missing|blanks?|nulls?|empty|columns?|fields?|variables?|quality|clean\w*|dirty|duplicat\w*"
    r"|types?|ranges?|values?|categor\w*|unique|distinct|outliers?|extremes?|errors?|valid\w*"
    r"|complete\w*|consisten\w*|keys?|structure|schema)\b",
    re.IGNORECASE,
)


def is_profile_question(question: str) -> bool:
    """A question about the dataset itself (its columns, blanks, quality) that the profile can answer."""
    return bool(_PROFILE_WORDS.search(question))


def profile_answer(question: str, profile: dict[str, Any]) -> str | None:
    """Answer the common questions about a dataset from its profile without a model, or None."""
    columns = profile.get("columns") or []
    rows = int(profile.get("rows") or 0)
    if re.search(r"\b(missing|blanks?|nulls?|empty|gaps?)\b", question, re.IGNORECASE):
        gaps = sorted((c for c in columns if c.get("nullPct")), key=lambda c: -float(c["nullPct"]))
        if not gaps:
            return f"No column has blanks: all {thousands(rows)} rows are filled in every column."
        listed = ", ".join(
            f"{c['name']} ({c['nullPct']}%, {thousands(int(c.get('nulls') or 0))} rows)" for c in gaps[:8]
        )
        more = f", and {len(gaps) - 8} more" if len(gaps) > 8 else ""
        return f"{len(gaps)} of {len(columns)} columns have blanks: {listed}{more}."
    if re.search(
        r"\b(quality|clean\w*|dirty|errors?|valid\w*|duplicat\w*|outliers?|extremes?)\b", question, re.I
    ):
        quality = profile.get("quality") or {}
        return (
            f"{quality.get('completeness', '—')}% of cells are filled and {quality.get('conformance', '—')}% of "
            f"values fit their column's type. {thousands(int(quality.get('duplicates') or 0))} rows repeat exactly "
            f"and {thousands(int(quality.get('outliers') or 0))} values sit far outside the usual range."
        )
    if re.search(r"\b(columns?|fields?|variables?|structure|schema)\b", question, re.IGNORECASE):
        names = ", ".join(f"{c['name']} ({c.get('inferredType')})" for c in columns[:20])
        return f"It has {len(columns)} columns and {thousands(rows)} rows: {names}."
    return None


# ------------------------------------------------------------------ which columns are which


def category_columns(profile: dict[str, Any]) -> list[str]:
    """Columns with a handful of values (text, flags or small codes) that can be cross-tabbed."""
    return [
        str(c["name"])
        for c in profile.get("columns") or []
        if not c.get("masked") and 2 <= int(c.get("distinct") or 0) <= MAX_CATEGORIES
    ]


def free_text_columns(profile: dict[str, Any]) -> list[str]:
    """Columns of written text: many different values, and long enough to be sentences rather than labels."""
    out = []
    for c in profile.get("columns") or []:
        if c.get("masked") or c.get("inferredType") not in ("string", "mixed"):
            continue
        if int(c.get("distinct") or 0) > MAX_CATEGORIES and int(c.get("lengthMax") or 0) >= 25:
            out.append(str(c["name"]))
    return out


def _forms(name: str) -> list[str]:
    spaced = name.replace("_", " ").lower()
    forms = {name.lower(), spaced}
    if " " not in spaced and len(spaced) >= 5:
        for suffix in ("ed", "ing", "s"):
            if spaced.endswith(suffix) and len(spaced) - len(suffix) >= 4:
                forms.add(spaced[: -len(suffix)])
    return sorted(forms, key=len, reverse=True)


def mentioned(question: str, names: list[str]) -> list[str]:
    """The columns a question names, in the order it names them."""
    lowered = question.lower()
    found: list[tuple[int, str]] = []
    for name in names:
        for form in _forms(name):
            match = re.search(rf"\b{re.escape(form)}\w*\b", lowered)
            if match:
                found.append((match.start(), name))
                break
    return [name for _, name in sorted(found)]


def early_route(question: str, profile: dict[str, Any]) -> Route | None:
    """Route the clearly qualitative questions before the number planner sees them."""
    texts = free_text_columns(profile)
    named_text = mentioned(question, texts)
    if texts and (named_text or _THEME_WORDS.search(question)) and (named_text or len(texts) == 1):
        if named_text or re.search(
            r"\b(say|said|wr(ote|ite)|feedback|comment|theme|sentiment|complain)", question, re.I
        ):
            return Route("themes", text_column=(named_text or texts)[0])
    categories = mentioned(question, category_columns(profile))
    if len(categories) >= 2 and _CROSSTAB_WORDS.search(question):
        # "Is churn related to plan?": compare churn across plans, so the second column named is the rows.
        return Route("crosstab", categories[1], categories[0])
    return None


def validate_route(proposal: dict[str, Any], profile: dict[str, Any]) -> Route:
    """A model's routing, checked against the dataset's real columns. Anything invalid becomes the overview."""
    kind = str(proposal.get("kind") or "overview")
    if kind == "crosstab":
        allowed = category_columns(profile)
        a, b = proposal.get("columnA"), proposal.get("columnB")
        if a in allowed and b in allowed and a != b:
            return Route("crosstab", str(a), str(b))
        return Route("profile")
    if kind == "themes":
        allowed = free_text_columns(profile)
        column = proposal.get("textColumn")
        if column in allowed:
            return Route("themes", text_column=str(column))
        return Route("profile")
    return Route(kind if kind in ("profile", "overview") else "overview")


def routing_prompt(name: str, profile: dict[str, Any], summary: str) -> str:
    categories = category_columns(profile)
    texts = free_text_columns(profile)
    return (
        f"{summary}\n\nCategory columns (for crosstab): {', '.join(categories) or 'none'}\n"
        f"Free-text columns (for themes): {', '.join(texts) or 'none'}"
    )


def profile_prompt(name: str, description: str, profile: dict[str, Any]) -> str:
    """Everything the profile knows, for answering questions about the dataset itself."""
    quality = profile.get("quality") or {}
    lines = [
        f"Dataset: {name}",
        f"Rows: {profile.get('rows', 0)}",
        f"Completeness {quality.get('completeness')}%, conformance {quality.get('conformance')}%, "
        f"{quality.get('duplicates', 0)} duplicate rows, {quality.get('outliers', 0)} extreme values",
    ]
    if description:
        lines.append(f"Owner's description: {description[:500]}")
    for dimension in profile.get("dimensions") or []:
        lines.append(f"Quality · {dimension.get('name')}: {dimension.get('score')} ({dimension.get('note')})")
    for c in profile.get("columns") or []:
        if c.get("masked"):
            lines.append(
                f"- {c['name']}: personal ({c.get('piiKind')}), masked; {c.get('nullPct', 0)}% blank"
            )
            continue
        facts = [
            str(c.get("inferredType")),
            f"{c.get('nullPct', 0)}% blank",
            f"{c.get('distinct', 0)} distinct",
        ]
        if c.get("min") is not None:
            facts.append(
                f"min {c['min']}, max {c['max']}, mean {c.get('mean')}, median {c.get('median')}, "
                f"{c.get('outliers', 0)} extreme"
            )
        top = [f"{v['value']} ({v['count']})" for v in (c.get("topValues") or [])[:5]]
        if top and c.get("min") is None:
            facts.append("most common: " + ", ".join(top))
        if c.get("labelVariants"):
            facts.append(f"{c['labelVariants']} labels with spelling variants")
        if c.get("missingness") not in (None, "none"):
            facts.append(f"blanks: {c.get('missingNote')}")
        lines.append(f"- {c['name']}: " + "; ".join(facts))
    for dependency in profile.get("dependencies") or []:
        lines.append(
            f"Dependency: {dependency['left']} determines {dependency['right']} ({dependency['support']})"
        )
    if profile.get("candidateKeys"):
        lines.append(f"Candidate keys: {', '.join(profile['candidateKeys'])}")
    return "\n".join(lines)


# ------------------------------------------------------------------ cross-tab


def _label(value: Any) -> str:
    return "(blank)" if value is None or str(value).strip() == "" else str(value)


def _strength(v: float) -> str:
    return "strong" if v >= 0.5 else "moderate" if v >= 0.3 else "weak" if v >= 0.1 else "very weak"


def crosstab(rows: list[tuple[Any, Any, Any]], a: str, b: str) -> dict[str, Any]:
    """Counts of a × b with a chi-square test of independence, read in words. `rows` are (a, b, count)."""
    frame = pd.DataFrame([(_label(x), _label(y), int(n or 0)) for x, y, n in rows], columns=["a", "b", "n"])
    frame = frame[(frame["a"] != "(blank)") & (frame["b"] != "(blank)")]
    label_a, label_b = a.replace("_", " "), b.replace("_", " ")
    if frame.empty:
        return {
            "answer": f"There are no rows with both {label_a} and {label_b} filled in, so they cannot be compared.",
            "explanation": "",
            "table": None,
        }
    table = frame.pivot_table(index="a", columns="b", values="n", aggfunc="sum", fill_value=0)
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).index]
    table = table[table.sum(axis=0).sort_values(ascending=False).index]
    total = int(table.to_numpy().sum())

    tested = table.loc[table.sum(axis=1) > 0, table.sum(axis=0) > 0]
    if tested.shape[0] < 2 or tested.shape[1] < 2:
        answer = f"Every row has the same {label_a if tested.shape[0] < 2 else label_b}, so there is no relationship to test."
        p_value, cramers_v, chi2, dof, low_expected = float("nan"), float("nan"), 0.0, 0, 0.0
    else:
        result = stats.chi2_contingency(tested.to_numpy())
        chi2, p_value, dof = float(result.statistic), float(result.pvalue), int(result.dof)
        expected = result.expected_freq
        low_expected = float((expected < 5).mean())
        k = min(tested.shape) - 1
        cramers_v = math.sqrt(chi2 / (total * k)) if total and k else 0.0

        overall = tested.sum(axis=0) / total
        shares = tested.div(tested.sum(axis=1), axis=0)
        gap = (shares - overall).abs()
        row_label, col_label = gap.stack().idxmax()  # type: ignore[misc]
        here = float(shares.to_numpy()[shares.index.get_loc(row_label), shares.columns.get_loc(col_label)])
        there = float(overall.to_numpy()[overall.index.get_loc(col_label)])
        biggest = (
            f"The biggest difference: {label_b} is {col_label} in {here:.0%} of rows where {label_a} is "
            f"{row_label}, against {there:.0%} overall."
        )
        if p_value < 0.05:
            answer = (
                f"Yes, {label_a} and {label_b} are related, but the link is {_strength(cramers_v)} "
                f"(Cramér's V = {cramers_v:.2f}). {biggest}"
            )
        else:
            answer = (
                f"No clear relationship: the mix of {label_b} is about the same for every {label_a} "
                f"(p = {p_value:.2f}). {biggest}"
            )

    explanation_parts = [
        f"I counted the {thousands(total)} rows that have both {label_a} and {label_b}, then ran a chi-square "
        f"test of independence"
        + (f" (χ² = {chi2:.1f}, {dof} degrees of freedom, p = {_p(p_value)})." if dof else ".")
    ]
    explanation_parts.append(
        "Cramér's V runs from 0 (no link) to 1 (one decides the other). A link between two categories does not "
        "show that one causes the other."
    )
    if low_expected > 0.2:
        explanation_parts.append(
            f"{low_expected:.0%} of the cells would expect fewer than 5 rows, so read the p-value with care."
        )

    # Row percentages use each row's total over every column, including columns not shown. Categories past the
    # first few are folded into one extra row, which is kept apart from any real category called "Other".
    groups: list[tuple[str, pd.Series, int]] = [
        (str(label), table.loc[label].iloc[:_TABLE_COLUMNS], int(table.loc[label].sum()))
        for label in table.index[:_TABLE_ROWS]
    ]
    if table.shape[0] > _TABLE_ROWS:
        rest = table.iloc[_TABLE_ROWS:]
        groups.append(
            (
                f"All other {label_a} values",
                rest.iloc[:, :_TABLE_COLUMNS].sum(axis=0),
                int(rest.to_numpy().sum()),
            )
        )
    out_rows = []
    for label, counts, n in groups:
        cells = [f"{(int(v) / n if n else 0):.1%} ({thousands(int(v))})" for v in counts]
        out_rows.append([label, *cells, thousands(n)])
    shown = table.iloc[:, :_TABLE_COLUMNS]
    columns = [label_a.capitalize(), *[f"{label_b.capitalize()}: {c}" for c in shown.columns], "Rows"]
    if table.shape[1] > _TABLE_COLUMNS:
        explanation_parts.append(f"The table shows the {_TABLE_COLUMNS} most common values of {label_b}.")
    return {
        "answer": answer,
        "explanation": " ".join(explanation_parts),
        "table": {"columns": columns, "rows": out_rows},
        "stats": {"chi2": chi2, "dof": dof, "p": p_value, "cramersV": cramers_v, "rows": total},
    }


def _p(p: float) -> str:
    if p != p:
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


# ------------------------------------------------------------------ free text


def scrub(text: str) -> str:
    """Remove contact details and long numbers from free text before a model reads it."""
    text = _EMAIL.sub("[email]", text)
    text = _URL.sub("[link]", text)
    text = _LONG_NUMBER.sub("[number]", text)
    return " ".join(text.split())[:400]


def keyword_themes(entries: list[str], column: str) -> dict[str, Any]:
    """Without a model: the words and two-word phrases that recur across the sample."""
    words: Counter[str] = Counter()
    phrases: Counter[str] = Counter()
    for entry in entries:
        tokens = re.findall(r"[a-z][a-z']+", entry.lower())
        words.update({t for t in tokens if t not in _STOPWORDS and len(t) > 2})
        # Only words that stand next to each other in the text form a phrase.
        phrases.update(
            {
                f"{x} {y}"
                for x, y in zip(tokens, tokens[1:], strict=False)
                if x not in _STOPWORDS and y not in _STOPWORDS and len(x) > 2 and len(y) > 2
            }
        )
    top = [(p, n) for p, n in phrases.most_common(4) if n > 1]
    covered = {w for p, _ in top for w in p.split()}
    top += [(w, n) for w, n in words.most_common(12) if w not in covered][: 8 - len(top)]
    top.sort(key=lambda item: -item[1])
    rows = [[term, thousands(n), f"{n / max(len(entries), 1):.0%}"] for term, n in top]
    lead = ", ".join(f'"{term}"' for term, _ in top[:3])
    return {
        "answer": f"The entries in {column.replace('_', ' ')} most often mention {lead}." if top else "",
        "table": {"columns": ["Word or phrase", "Entries", "Share of sample"], "rows": rows},
    }


_SENTIMENTS = ("positive", "neutral", "negative")


def themes_result(coded: dict[str, Any], entries: list[str]) -> dict[str, Any]:
    """Count the model's per-entry labels into themes, sentiment and real example quotes.

    The model only labels entries; the counting is done here, so every share is exact for the sample and
    every example is an entry that was really written.
    """
    names = [str(t)[:80] for t in coded.get("themes") or [] if isinstance(t, str | int) and str(t).strip()][
        :6
    ]
    counts: Counter[int] = Counter()
    moods: Counter[str] = Counter()
    theme_moods: dict[int, Counter[str]] = {}
    examples: dict[tuple[int, str], str] = {}  # the first entry of each theme and sentiment
    labelled: set[int] = set()
    for label in coded.get("labels") or []:
        if not isinstance(label, dict):
            continue
        try:
            entry, theme = int(label.get("i") or 0), int(label.get("t") or 0)
        except (TypeError, ValueError):
            continue
        if not (1 <= entry <= len(entries)) or not (1 <= theme <= len(names)) or entry in labelled:
            continue
        labelled.add(entry)
        mood = str(label.get("s") or "neutral").lower()
        mood = mood if mood in _SENTIMENTS else "neutral"
        counts[theme] += 1
        moods[mood] += 1
        theme_moods.setdefault(theme, Counter())[mood] += 1
        examples.setdefault((theme, mood), entries[entry - 1])
    rows = []
    for theme, count in counts.most_common():
        main, main_n = theme_moods[theme].most_common(1)[0]
        rows.append(
            [
                names[theme - 1],
                f"{count / len(labelled):.0%} ({count})",
                main if main_n / count >= 0.7 else "mixed",
                examples[(theme, main)][:160],  # a quote that carries the theme's main sentiment
            ]
        )
    mood_text = ""
    if labelled:
        shares = ", ".join(f"{moods[m] / len(labelled):.0%} {m}" for m in _SENTIMENTS)
        mood_text = f" Sentiment in the sample: {shares}."
    return {
        "answer": (str(coded.get("summary") or "").strip()[:600] + mood_text).strip(),
        "table": {"columns": ["Theme", "Share of sample", "Sentiment", "Example"], "rows": rows},
        "labelled": len(labelled),
    }
