import pandas as pd
import pytest

from dataforge_backend.domain.insights import detect_insights
from dataforge_backend.domain.pii import PiiKind, detect_pii, mask_value, pseudonymise
from dataforge_backend.domain.profiling import profile_frame
from dataforge_backend.domain.values import (
    date_shape,
    jaro_winkler,
    normalise_za_phone,
    parse_date,
    parse_number,
    sa_id_valid,
    thousands,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("R 1 250.00", 1250.0),
        ("1 250,00", 1250.0),
        ("R1,250.00", 1250.0),
        ("1,250,000", 1250000.0),
        ("12,5", 12.5),
        ("(40)", -40.0),
        ("1.250.000", 1250000.0),
        ("abc", None),
        ("", None),
    ],
)
def test_parse_number_reads_common_writing(raw: str, expected: float | None) -> None:
    assert parse_number(raw) == expected


def test_parse_date_is_day_first_and_shapes_are_recognised() -> None:
    assert str(parse_date("03/04/2026")) == "2026-04-03"
    assert str(parse_date("March 14 2026")) == "2026-03-14"
    assert str(parse_date("2026/03/14")) == "2026-03-14"
    assert parse_date("not a date") is None
    assert date_shape("14-03-2026") == "dd-mm-yyyy"


def test_south_african_identifiers() -> None:
    assert sa_id_valid("8001015009087")
    assert not sa_id_valid("8001015009088")
    assert normalise_za_phone("082 123 4567") == "+27821234567"
    assert thousands(48213) == "48 213"
    assert jaro_winkler("Thandi Mokoena", "Thandi Mokena") > 0.95


def test_pii_is_detected_by_value_and_name_and_masked() -> None:
    frame = pd.DataFrame(
        {
            "full_name": ["Thandi Mokoena", "Sipho Dlamini"] * 10,
            "id_number": ["8001015009087"] * 20,
            "contact": ["thandi@example.com", "sipho@example.com"] * 10,
            "course_name": ["SQL basics", "Data literacy"] * 10,
        },
        dtype=object,
    )
    found = {f.column: f.kind for f in detect_pii(frame)}
    assert found["full_name"] == PiiKind.NAME
    assert found["id_number"] == PiiKind.NATIONAL_ID
    assert found["contact"] == PiiKind.EMAIL
    assert "course_name" not in found
    assert mask_value("8001015009087", PiiKind.NATIONAL_ID) == "••••••••••087"
    assert "•" in (mask_value("Thandi Mokoena", PiiKind.NAME) or "")
    assert pseudonymise("x", "salt") == pseudonymise("x", "salt") != pseudonymise("x", "other")


def test_profile_measures_quality_and_missingness() -> None:
    frame = pd.DataFrame(
        {
            "learner_id": [f"LR-{i}" for i in range(200)],
            "course": ["SQL basics" if i % 2 else "Data literacy" for i in range(200)],
            "region": ["" if i % 2 else "Gauteng" for i in range(200)],
            "fee_paid": ["R 1 250.00" if i % 10 else "98000" for i in range(200)],
            "enrolled_on": ["2026/03/14" if i % 3 else "14-03-2026" for i in range(200)],
        },
        dtype=object,
    )
    profile, facts = profile_frame(frame, detect_pii(frame))
    columns = {c["name"]: c for c in profile["columns"]}
    assert profile["rows"] == 200
    assert columns["fee_paid"]["inferredType"] in ("integer", "decimal")
    assert columns["enrolled_on"]["inferredType"] == "date"
    assert columns["region"]["nullPct"] == 50.0
    # Region is blank exactly when the course is SQL basics: missing at random, explained by course.
    assert columns["region"]["missingness"] == "MAR"
    assert set(profile["quality"]) >= {"completeness", "conformance", "duplicates", "outliers"}
    kinds = {i.kind for i in detect_insights(frame, profile, facts)}
    assert "quality" in kinds
