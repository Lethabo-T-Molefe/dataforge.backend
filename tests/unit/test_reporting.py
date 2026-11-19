"""Governed reports: small groups are hidden, and a narrative cannot carry a number of its own."""

from dataforge_backend.domain import reporting


def test_small_counts_and_rates_of_small_groups_are_suppressed() -> None:
    assert reporting.suppress(3, None, True) == (None, True)
    assert reporting.suppress(12, None, True) == (12, False)
    assert reporting.suppress(0, None, True) == (0, False)  # a true zero is not identifying
    assert reporting.suppress(0.5, 4, False) == (None, True)  # a rate over four people
    assert reporting.suppress(0.5, 40, False) == (0.5, False)
    assert reporting.suppress(None, None, True) == (None, False)


def test_a_narrative_may_only_use_placeholders_for_numbers() -> None:
    known = {"F01", "F02"}
    assert (
        reporting.validate_narrative(["Between {{PERIOD_FROM}} and {{PERIOD_TO}}, {{F01}} started."], known)
        == []
    )
    invented = reporting.validate_narrative(["{{F01}} started, up 12% on last year."], known)
    assert invented and "not taken from a figure" in invented[0]
    unknown = reporting.validate_narrative(["{{F09}} completed."], known)
    assert unknown and "not a figure" in unknown[0]
    stray = reporting.validate_narrative(["{{SALARY}} was paid."], known)
    assert stray and "unknown placeholder" in stray[0]


def test_placeholders_are_filled_from_the_platforms_figures() -> None:
    filled = reporting.fill(["{{F01}} of {{F02}}; {{F03}}."], {"F01": "1 200", "F02": "5 000"})
    assert filled == ["1 200 of 5 000; not available."]


def test_sentences_need_every_figure_they_name() -> None:
    template = reporting.SKILLS_PROGRESS
    computed = {f"F{n:02d}" for n in range(1, 16)}  # no monthly flows (F16, F17) or model (F18)
    sentences = reporting.usable_narrative(template, computed)
    assert sentences and all("{{F16}}" not in s and "{{F18}}" not in s for s in sentences)
    assert reporting.validate_narrative(list(template.narrative), {f"F{n:02d}" for n in range(1, 19)}) == []


def test_any_dataset_gets_a_report_from_its_shared_metrics() -> None:
    template = reporting.auto_template(
        "Regional revenue", [("revenue", "Revenue"), ("avg_fee", "Average fee")], ["region"]
    )
    ids = [f.id for f in template.figure_specs()]
    assert ids == ["F01", "F02", "F03", "F04"] and len(set(ids)) == 4
    assert reporting.validate_narrative(list(template.narrative), set(ids)) == []
    assert "skills" not in template.title.lower()
