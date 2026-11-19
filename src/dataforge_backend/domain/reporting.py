"""Governed reports: templates of figures, small-cell suppression, and narratives that cannot invent numbers.

A report is a template of figures. Each figure is a governed metric, optionally broken down by a dimension, over
the report period, on a dataset the template names by role ("primary", "flows"). The platform computes every
figure in the sandbox and keeps the SQL, parameters and sandbox session with it. A narrative may refer to figures
only through placeholders such as {{F04}}: it is checked to contain no digits of its own before the platform
fills the placeholders in, so no number in the prose can be invented.

The engine is generic; SKILLS_PROGRESS is the skills-development showcase, and auto_template builds a report for
any dataset from the governed metrics it offers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

SUPPRESS_BELOW = 5
_PLACEHOLDER = re.compile(r"\{\{(F\d{2}|PERIOD_FROM|PERIOD_TO)\}\}")


class ReportError(ValueError):
    pass


@dataclass(frozen=True)
class FigureSpec:
    id: str
    label: str
    measure: str
    role: str = "primary"  # which dataset of the template
    group_by: str | None = None
    filters: dict[str, str] = field(default_factory=dict)
    optional: bool = False  # skipped (and noted) when the dataset lacks the measure or dimension


@dataclass(frozen=True)
class ModelFigureSpec:
    id: str
    label: str
    model: str  # model name; its approved version's latest scoring run is used


@dataclass(frozen=True)
class Section:
    title: str
    figures: tuple[FigureSpec | ModelFigureSpec, ...]
    note: str = ""


@dataclass(frozen=True)
class ReportTemplate:
    id: str
    title: str
    description: str
    roles: dict[str, str]  # role -> what the dataset is
    sections: tuple[Section, ...]
    narrative: tuple[str, ...] = ()  # sentences with placeholders, used when no model writes the narrative
    disclaimer: str = ""

    def figure_specs(self) -> list[FigureSpec | ModelFigureSpec]:
        return [f for s in self.sections for f in s.figures]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "roles": self.roles,
            "sections": [{"title": s.title, "figures": [f.id for f in s.figures]} for s in self.sections],
        }


def _fig(id_: str, label: str, measure: str, by: str | None = None, role: str = "primary") -> FigureSpec:
    return FigureSpec(id_, label, measure, role, by, optional=by is not None)


SKILLS_PROGRESS = ReportTemplate(
    id="skills-progress",
    title="Learner progress report",
    description=(
        "Starts, completions and drop-outs by programme and province, the equity breakdown funders ask for, "
        "monthly flows, and how many current learners the early-warning model flags."
    ),
    roles={
        "primary": "Learner records: one row per learner (e.g. Learner progress)",
        "flows": "Monthly progress across the network (e.g. Learner progress, monthly)",
    },
    sections=(
        Section(
            "Starts, completions and drop-outs",
            (
                _fig("F01", "Learners who started", "programme_starts"),
                _fig("F02", "Learners who completed", "programme_completions"),
                _fig("F03", "Learners who dropped out", "learner_dropouts"),
                _fig("F04", "Drop-out rate", "dropout_rate"),
                _fig("F05", "Completion rate", "programme_completion_rate"),
            ),
            "Learners still in progress count as neither completed nor dropped out.",
        ),
        Section(
            "By programme and province",
            (
                _fig("F06", "Drop-out rate by programme", "dropout_rate", "programme_type"),
                _fig("F07", "Starts by programme", "programme_starts", "programme_type"),
                _fig("F08", "Starts by province", "programme_starts", "province"),
                _fig("F09", "Drop-out rate by province", "dropout_rate", "province"),
            ),
        ),
        Section(
            "Equity",
            (
                _fig("F10", "Starts by gender", "programme_starts", "gender"),
                _fig("F11", "Starts by population group", "programme_starts", "population_group"),
                _fig("F12", "Starts by disability", "programme_starts", "disability"),
                _fig("F13", "Starts by age band", "programme_starts", "age_band"),
                _fig("F14", "Drop-out rate by gender", "dropout_rate", "gender"),
                _fig("F15", "Drop-out rate by population group", "dropout_rate", "population_group"),
            ),
            f"Groups with fewer than {SUPPRESS_BELOW} learners are shown as <{SUPPRESS_BELOW}.",
        ),
        Section(
            "Monthly flows",
            (
                FigureSpec("F16", "Drop-outs in the period", "monthly_dropouts", "flows", optional=True),
                FigureSpec("F17", "New starts in the period", "new_starts", "flows", optional=True),
            ),
        ),
        Section(
            "Early warning",
            (
                ModelFigureSpec(
                    "F18", "Current learners flagged as likely to drop out", "learner-dropout-risk"
                ),
            ),
            "From the approved early-warning model's latest run. Flags prioritise support; a person decides.",
        ),
    ),
    narrative=(
        "Between {{PERIOD_FROM}} and {{PERIOD_TO}}, {{F01}} learners started a programme.",
        "So far {{F02}} have completed and {{F03}} have dropped out, a drop-out rate of {{F04}} and a "
        "completion rate of {{F05}}.",
        "Across the network, {{F16}} learners left their programme during the period and {{F17}} started one.",
        "The early-warning model currently flags {{F18}} learners for follow-up by a facilitator.",
    ),
    disclaimer=(
        "Prepared from governed metrics for progress reporting to funders. It is not an official SETA WSP or ATR "
        "submission."
    ),
)


def auto_template(
    dataset_name: str, measures: list[tuple[str, str]], dimensions: list[str]
) -> ReportTemplate:
    """A report for any dataset: each governed metric in total, and by its first two dimensions."""
    figures: list[FigureSpec | ModelFigureSpec] = []
    number = 1
    for key, label in measures:
        figures.append(FigureSpec(f"F{number:02d}", label, key))
        number += 1
        for dim in dimensions[:2]:
            figures.append(
                FigureSpec(f"F{number:02d}", f"{label} by {dim.replace('_', ' ')}", key, group_by=dim)
            )
            number += 1
            if number > 99:
                break
    totals = [f for f in figures if isinstance(f, FigureSpec) and f.group_by is None][:3]
    return ReportTemplate(
        id="auto",
        title=f"{dataset_name}: governed summary",
        description="Every shared metric on the dataset, in total and by its main groupings.",
        roles={"primary": dataset_name},
        sections=(Section("Shared metrics", tuple(figures)),),
        narrative=(
            "Between {{PERIOD_FROM}} and {{PERIOD_TO}}: "
            + "; ".join(f"{f.label.lower()} {{{{{f.id}}}}}" for f in totals)
            + ".",
        )
        if totals
        else (),
    )


TEMPLATES = {SKILLS_PROGRESS.id: SKILLS_PROGRESS}


# ------------------------------------------------------------------ suppression and narrative


def suppress(value: float | None, support: float | None, is_count: bool) -> tuple[float | None, bool]:
    """Hide a figure that rests on fewer than SUPPRESS_BELOW people: a small count, or a rate of a small group."""
    if value is None:
        return None, False
    base = value if is_count else support
    if base is not None and 0 < base < SUPPRESS_BELOW:
        return None, True
    return value, False


def validate_narrative(paragraphs: list[str], known: set[str]) -> list[str]:
    """Problems with a narrative: digits outside placeholders, or placeholders for figures that do not exist."""
    problems = []
    for text in paragraphs:
        for match in _PLACEHOLDER.finditer(text):
            name = match.group(1)
            if name.startswith("F") and name not in known:
                problems.append(f"{{{{{name}}}}} is not a figure in this report.")
        bare = _PLACEHOLDER.sub("", text)
        if re.search(r"\d", bare):
            problems.append(f"Contains a number not taken from a figure: {text[:80]}")
        if re.search(r"\{\{[^}]*\}\}", bare):
            problems.append(f"Contains an unknown placeholder: {text[:80]}")
    return problems


def fill(paragraphs: list[str], values: dict[str, str]) -> list[str]:
    """Put the platform's figures into the narrative. A missing figure reads as 'not available'."""
    return [_PLACEHOLDER.sub(lambda m: values.get(m.group(1), "not available"), text) for text in paragraphs]


def usable_narrative(template: ReportTemplate, available: set[str]) -> list[str]:
    """The template's sentences whose figures were all computed."""
    out = []
    for sentence in template.narrative:
        needed = {m.group(1) for m in _PLACEHOLDER.finditer(sentence) if m.group(1).startswith("F")}
        if needed <= available:
            out.append(sentence)
    return out
