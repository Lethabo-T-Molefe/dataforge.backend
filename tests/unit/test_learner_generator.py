"""The skills-development showcase generator: deterministic, learnable but not trivial, and honest about PII."""

import statistics
from collections import defaultdict

import pandas as pd

from dataforge_backend.domain.pii import detect_pii
from dataforge_backend.seed_data import learners as showcase

SIGNALS = [
    "attendance_pct",
    "formative_avg",
    "lms_logins",
    "missed_submissions",
    "late_stipend_payments",
    "stipend_rand_avg",
    "travel_km",
]


def _labelled_early(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Cohorts old enough that every programme has had time to finish: the honest training population."""
    until = showcase.month_index("2025-04")
    return [
        r
        for r in rows
        if r["outcome"] != "In progress" and showcase.month_index(str(r["start_month"])) <= until
    ]


def test_the_generator_is_deterministic() -> None:
    assert showcase.learners(300) == showcase.learners(300)
    assert showcase.monthly()[:500] == showcase.monthly()[:500]
    assert showcase.learners(300, seed=1) != showcase.learners(300, seed=2)


def test_dropout_is_common_enough_to_learn_and_no_single_signal_gives_it_away() -> None:
    early = _labelled_early(showcase.learners(3000))
    share = sum(int(str(r["dropped"])) for r in early) / len(early)
    assert 0.2 <= share <= 0.32, share
    frame = pd.DataFrame(early)
    for column in SIGNALS:
        r = frame[column].astype(float).corr(frame["dropped"].astype(float))
        assert abs(r) < 0.6, (column, r)


def test_outcomes_are_consistent_with_their_indicator_columns() -> None:
    for row in showcase.learners(500):
        assert row["starts"] == 1
        assert row["completed"] == int(row["outcome"] == "Completed")
        assert row["dropped"] == int(row["outcome"] == "Dropped out")
        assert str(row["nqf_level"]).startswith("NQF ")


def test_recent_cohorts_log_in_more_which_is_the_planted_drift() -> None:
    rows = showcase.learners(3000)
    drift_from = showcase.month_index(showcase.DRIFT_FROM)
    before = [float(r["lms_logins"]) for r in rows if showcase.month_index(r["start_month"]) < drift_from]
    after = [float(r["lms_logins"]) for r in rows if showcase.month_index(r["start_month"]) >= drift_from]
    assert statistics.fmean(after) > 1.4 * statistics.fmean(before)


def test_the_monthly_data_carries_the_planted_anomaly() -> None:
    rows = showcase.monthly()
    anomaly = showcase.ANOMALY
    by_month: dict[str, int] = defaultdict(int)
    for row in rows:
        if row["province"] == anomaly["province"] and row["seta"] == anomaly["seta"]:
            by_month[str(row["month"])] += int(row["dropouts"])
    others = [value for month, value in by_month.items() if month != anomaly["month"]]
    assert by_month[str(anomaly["month"])] > 2.5 * statistics.median(others)
    months = {str(row["month"]) for row in rows}
    assert min(months) == showcase.FIRST_MONTH and max(months) == showcase.CUTOFF and len(months) == 34


def test_personal_columns_are_detected_and_the_learner_number_is_not() -> None:
    frame = pd.DataFrame(showcase.learners(200)).fillna("").astype(str).astype(object)
    found = {finding.column: finding.kind.value for finding in detect_pii(frame)}
    assert found.get("id_number") == "national_id"
    assert found.get("full_name") == "name"
    assert found.get("cell_number") == "phone"
    for column in ("learner_id", "employer", "seta", "province", "population_group", "disability"):
        assert column not in found, column
