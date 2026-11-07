"""Qualitative questions: routing, cross-tabs with a chi-square test, and free-text themes."""

from dataforge_backend.domain.qualitative import (
    Route,
    category_columns,
    crosstab,
    early_route,
    free_text_columns,
    is_general,
    keyword_themes,
    mentioned,
    profile_prompt,
    scrub,
    themes_result,
    validate_route,
)

PROFILE = {
    "rows": 400,
    "quality": {"completeness": 98, "conformance": 100, "duplicates": 0, "outliers": 0},
    "columns": [
        {"name": "gender", "inferredType": "string", "distinct": 3, "lengthMax": 6, "nullPct": 0},
        {"name": "churned", "inferredType": "integer", "distinct": 2, "min": "0", "max": "1", "nullPct": 0},
        {"name": "country", "inferredType": "string", "distinct": 8, "lengthMax": 9, "nullPct": 0},
        {"name": "age", "inferredType": "integer", "distinct": 60, "min": "18", "max": "80", "nullPct": 1.0},
        {"name": "feedback", "inferredType": "string", "distinct": 380, "lengthMax": 140, "nullPct": 4.0},
        {"name": "email", "inferredType": "string", "distinct": 400, "masked": True, "piiKind": "email"},
    ],
}


def test_columns_are_sorted_into_categories_and_free_text() -> None:
    assert category_columns(PROFILE) == ["gender", "churned", "country"]
    assert free_text_columns(PROFILE) == ["feedback"]
    assert mentioned("Is churn related to gender?", category_columns(PROFILE)) == ["churned", "gender"]


def test_clear_qualitative_questions_are_routed_before_the_number_planner() -> None:
    assert early_route("Is churn related to gender?", PROFILE) == Route("crosstab", "gender", "churned")
    assert early_route("How does gender break down by country?", PROFILE) == Route(
        "crosstab", "country", "gender"
    )
    assert early_route("What are customers saying in their feedback?", PROFILE) == Route(
        "themes", text_column="feedback"
    )
    assert early_route("What is the sentiment of the feedback?", PROFILE) == Route(
        "themes", text_column="feedback"
    )
    # Questions about a number stay with the analysis engine.
    assert early_route("What is average age by gender?", PROFILE) is None
    assert early_route("How many rows are there by country?", PROFILE) is None
    assert is_general("What is this dataset about?")


def test_a_model_route_is_checked_against_the_real_columns() -> None:
    assert validate_route({"kind": "crosstab", "columnA": "gender", "columnB": "country"}, PROFILE) == Route(
        "crosstab", "gender", "country"
    )
    # A masked or invented column never gets queried.
    assert (
        validate_route({"kind": "crosstab", "columnA": "email", "columnB": "gender"}, PROFILE).kind
        == "profile"
    )
    assert validate_route({"kind": "themes", "textColumn": "email"}, PROFILE).kind == "profile"
    assert validate_route({"kind": "drop table"}, PROFILE).kind == "overview"


def test_the_profile_prompt_never_holds_personal_values() -> None:
    prompt = profile_prompt("Customers", "", PROFILE)
    assert "email: personal (email), masked" in prompt
    assert "feedback" in prompt and "4.0% blank" in prompt


def test_a_crosstab_finds_a_real_relationship() -> None:
    rows = [("Male", "1", 120), ("Male", "0", 80), ("Female", "1", 40), ("Female", "0", 160)]
    read = crosstab(rows, "gender", "churned")
    assert read["answer"].startswith("Yes, gender and churned are related")
    assert read["stats"]["p"] < 0.001 and 0.3 < read["stats"]["cramersV"] < 0.5
    assert read["table"]["columns"][0] == "Gender"
    assert read["table"]["rows"][0][-1] == "200"


def test_a_crosstab_reports_no_relationship_when_the_mix_is_the_same() -> None:
    rows = [("A", "x", 50), ("A", "y", 50), ("B", "x", 51), ("B", "y", 49)]
    read = crosstab(rows, "region", "plan")
    assert read["answer"].startswith("No clear relationship")
    assert read["stats"]["p"] > 0.05


def test_a_crosstab_ignores_blanks_and_handles_a_single_value() -> None:
    read = crosstab([("A", "x", 5), ("A", "y", 5), (None, "x", 9)], "region", "plan")
    assert "same region" in read["answer"]
    assert "no rows" in crosstab([(None, "x", 3)], "region", "plan")["answer"]


def test_free_text_is_scrubbed_and_keyword_coded_without_a_model() -> None:
    assert (
        scrub("Call me on 082 123 4567 or a@b.co.za, see https://x.y")
        == "Call me on [number] or [email], see [link]"
    )
    entries = [
        "Delivery was late again",
        "late delivery and rude staff",
        "Great delivery, fast",
        "late delivery",
    ]
    read = keyword_themes(entries, "feedback")
    assert '"late delivery"' in read["answer"]
    assert read["table"]["rows"][0][0] == "late delivery"
    assert "delivery rude" not in str(read["table"])


def test_model_labels_are_counted_into_exact_themes_with_real_quotes() -> None:
    entries = ["late again", "so late", "never arrived", "love it", "great value"]
    coded = {
        "summary": "Most people complain about late deliveries.",
        "themes": ["Late delivery", "Happy customers", "Never used"],
        "labels": [
            {"i": 1, "t": 1, "s": "negative"},
            {"i": 2, "t": 1, "s": "negative"},
            {"i": 3, "t": 1, "s": "negative"},
            {"i": 4, "t": 2, "s": "positive"},
            {"i": 5, "t": 2, "s": "positive"},
            {"i": 5, "t": 1, "s": "negative"},  # a second label for one entry is ignored
            {"i": 99, "t": 1, "s": "negative"},  # an entry that does not exist is ignored
        ],
    }
    read = themes_result(coded, entries)
    assert read["labelled"] == 5
    assert read["answer"].endswith("40% positive, 0% neutral, 60% negative.")
    assert read["table"]["rows"][0] == ["Late delivery", "60% (3)", "negative", "late again"]
    assert read["table"]["rows"][1] == ["Happy customers", "40% (2)", "positive", "love it"]
    assert len(read["table"]["rows"]) == 2


def test_profile_questions_are_answered_without_a_model() -> None:
    from dataforge_backend.domain.qualitative import is_profile_question, profile_answer

    assert is_profile_question("Which columns have missing values?")
    assert profile_answer("Which columns have missing values?", PROFILE) == (
        "2 of 6 columns have blanks: feedback (4.0%, 0 rows), age (1.0%, 0 rows)."
    )
    assert "98% of cells are filled" in (profile_answer("Is the data clean?", PROFILE) or "")
    assert (profile_answer("What columns are there?", PROFILE) or "").startswith(
        "It has 6 columns and 400 rows"
    )
    assert profile_answer("Why do customers leave?", PROFILE) is None
