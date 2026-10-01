import pandas as pd
import pytest

from dataforge_backend.domain.cleaning import (
    METHODS,
    METHODS_BY_ID,
    CleaningContext,
    OperationError,
    RecipeStepSpec,
    apply_recipe,
    code_for,
    enforce_masking,
    is_applicable,
    preview_step,
    run_operation,
    type_for_silver,
)
from dataforge_backend.domain.pii import detect_pii
from dataforge_backend.domain.profiling import profile_frame


@pytest.fixture
def raw() -> pd.DataFrame:
    rows = []
    for i in range(120):
        rows.append(
            {
                "learner_id": f"LR-{100000 + (i if i % 40 else 0)}",
                "full_name": ["Thandi Mokoena", "Sipho Dlamini", "Naledi Khumalo"][i % 3],
                "enrolled_on": ["2026/03/14", "14-03-2026", "March 14 2026", ""][i % 4],
                "course": "SQL basics" if i % 2 else "Data literacy",
                "region": ["Gauteng", "gauteng", "GP", "", "Limpopo"][i % 5],
                "fee_paid": ["R 1 250.00", "1250", "N/A", "1 400,00"][i % 4],
            }
        )
    return pd.DataFrame(rows, dtype=object)


def _ctx(frame: pd.DataFrame) -> CleaningContext:
    return CleaningContext(pii_columns={f.column: f.kind.value for f in detect_pii(frame)})


def test_every_method_has_defaults_code_and_a_working_preview(raw: pd.DataFrame) -> None:
    profile, _ = profile_frame(raw, detect_pii(raw))
    ctx = _ctx(raw)
    for method in METHODS:
        params = method.defaults(profile, list(raw.columns))
        assert isinstance(code_for(method.operation, params), str)
        if is_applicable(method, params):
            preview = preview_step(raw.head(60), method, params, ctx)
            assert preview["summary"], method.id
            assert len(preview["before"]) == len(preview["after"])


def test_formats_become_iso_dates_and_numbers(raw: pd.DataFrame) -> None:
    outcome = run_operation(
        "standardise_formats",
        raw,
        {"dates": ["enrolled_on"], "numbers": ["fee_paid"], "day_first": True},
        _ctx(raw),
    )
    typed, types = type_for_silver(outcome.frame)
    assert types["enrolled_on"] == "date"
    assert types["fee_paid"] in ("integer", "numeric")
    assert {v for v in typed["fee_paid"] if v is not None} <= {1250, 1400}


def test_recipe_is_reproducible_and_bronze_is_untouched(raw: pd.DataFrame) -> None:
    before = raw.copy()
    ctx = _ctx(raw)
    steps = [
        RecipeStepSpec(
            "format-date-number",
            {"dates": ["enrolled_on"], "numbers": ["fee_paid"], "day_first": True},
            "accepted",
        ),
        RecipeStepSpec(
            "cat-map", {"column": "region", "mapping": {"gauteng": "Gauteng", "GP": "Gauteng"}}, "accepted"
        ),
        RecipeStepSpec("dup-keep-first", {"subset": ["learner_id"], "keep": "first"}, "accepted"),
        RecipeStepSpec("missing-column-drop", {"threshold": 0.5}, "skipped"),
    ]
    first, _ = apply_recipe(raw, steps, ctx)
    second, _ = apply_recipe(raw, steps, ctx)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(raw, before)
    assert set(first["region"].dropna()) <= {"Gauteng", "Limpopo", ""}
    assert first["learner_id"].is_unique


def test_masking_is_enforced_before_silver(raw: pd.DataFrame) -> None:
    masked, columns = enforce_masking(raw, {"full_name": "name"})
    assert "full_name" in columns
    assert all("•" in str(v) for v in masked["full_name"])


def test_unknown_columns_are_refused(raw: pd.DataFrame) -> None:
    with pytest.raises(OperationError):
        run_operation("drop_missing_rows", raw, {"columns": ["nope"]}, _ctx(raw))
    assert METHODS_BY_ID["profile-accept"].operation == "accept_profile"
