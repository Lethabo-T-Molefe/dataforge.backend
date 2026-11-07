"""Owners analyse their own data before (or without) cleaning and approval; nobody else does."""

from types import SimpleNamespace

from dataforge_backend.application.common import is_owner_queryable, is_queryable
from dataforge_backend.domain.governance import DatasetSource, DatasetStatus, Layer, journey, next_step
from dataforge_backend.domain.roles import Role
from dataforge_backend.infrastructure.db.models import Dataset

OWNER = SimpleNamespace(id="thandi")
OTHER = SimpleNamespace(id="sipho")


def _raw(**overrides: object) -> Dataset:
    values: dict[str, object] = {
        "id": "pii-demo",
        "table_name": "pii_demo",
        "layer": "bronze",
        "status": "pending",
        "owner_id": "thandi",
        "quality": {"workingCopy": True},
        "fields": {"table": "silver.pii_demo", "columns": []},
    }
    values.update(overrides)
    return Dataset(**values)


def test_raw_upload_reads_the_masked_working_copy_never_bronze() -> None:
    assert _raw().qualified_table() == "silver.pii_demo"
    assert _raw(quality={}).qualified_table() is None


def test_owner_can_analyse_without_cleaning_or_approval() -> None:
    dataset = _raw()
    assert is_owner_queryable(dataset, OWNER)  # type: ignore[arg-type]
    assert is_queryable(dataset, OWNER)  # type: ignore[arg-type]


def test_nobody_else_can_analyse_it_before_approval() -> None:
    dataset = _raw()
    assert not is_queryable(dataset, OTHER)  # type: ignore[arg-type]
    assert not is_queryable(dataset)  # the shared rule, used for Superset and organisation syncs


def test_owner_access_closes_when_rejected_erased_or_not_ready() -> None:
    assert not is_queryable(_raw(status="rejected"), OWNER)  # type: ignore[arg-type]
    assert not is_queryable(_raw(quality={}), OWNER)  # type: ignore[arg-type]
    assert not is_queryable(_raw(fields=None), OWNER)  # type: ignore[arg-type]
    assert not is_queryable(_raw(owner_id=None), OWNER)  # type: ignore[arg-type]


def test_owner_keeps_analysing_cleaned_data_while_it_waits_for_approval() -> None:
    cleaned = _raw(layer="silver", quality={}, status="pending")
    assert is_queryable(cleaned, OWNER)  # type: ignore[arg-type]
    assert not is_queryable(cleaned, OTHER)  # type: ignore[arg-type]


def test_approved_data_is_open_to_everyone_permitted_to_see_it() -> None:
    approved = _raw(layer="silver", quality={}, status="approved")
    assert is_queryable(approved, OTHER)  # type: ignore[arg-type]
    assert is_queryable(approved)


def test_next_step_offers_analysis_to_the_owner_and_cleaning_to_others() -> None:
    owner_step = next_step(Layer.BRONZE, DatasetStatus.PENDING, Role.ANALYST, owner_ready=True)
    assert owner_step.kind == "analyse"
    assert "masked" in owner_step.reason
    assert next_step(Layer.BRONZE, DatasetStatus.PENDING, Role.ANALYST).kind == "clean"
    rejected = next_step(Layer.BRONZE, DatasetStatus.REJECTED, Role.ANALYST, owner_ready=True)
    assert rejected.kind == "rejected"


def test_journey_marks_cleaning_and_approval_as_optional_for_the_owner() -> None:
    steps = journey(Layer.BRONZE, DatasetStatus.PENDING, DatasetSource.UPLOAD, owner_ready=True)
    labels = [step["label"] for step in steps]
    assert labels == ["Brought in", "Analysed by you", "Cleaned", "Approved and shared"]
    assert steps[2]["detail"] == "Optional"
    assert journey(Layer.BRONZE, DatasetStatus.PENDING, DatasetSource.UPLOAD)[1]["label"] == "Cleaned"
