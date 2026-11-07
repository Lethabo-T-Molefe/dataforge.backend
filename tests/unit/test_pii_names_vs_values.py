"""A column's name is only a hint: plain measurements under a personal-sounding name are not personal data."""

import pandas as pd

from dataforge_backend.domain.pii import PiiKind, detect_pii
from dataforge_backend.domain.profiling import profile_frame


def _kinds(frame: pd.DataFrame) -> dict[str, PiiKind]:
    return {finding.column: finding.kind for finding in detect_pii(frame)}


def test_behaviour_rates_named_after_contact_channels_are_not_masked() -> None:
    frame = pd.DataFrame(
        {
            "email_open_rate": ["17.9", "22.4", "8.0", "41.3", "0.5"],
            "mobile_app_usage": ["12", "30", "7", "0", "55"],
            "customer_name_length": ["5", "7", "4", "6", "9"],
        }
    )
    assert _kinds(frame) == {}


def test_real_personal_columns_are_still_found() -> None:
    frame = pd.DataFrame(
        {
            "email": ["a@x.org", "b@y.org", "c@z.org", "d@w.org"],
            "mobile": ["0821234567", "0837654321", "0712345678", "0794561230"],
            "full_name": ["Ayanda Nkosi", "Pieter van Wyk", "Nomsa Dlamini", "Kabelo Mokoena"],
            "contact_email": ["", "", "", ""],
        }
    )
    kinds = _kinds(frame)
    assert kinds["email"] == PiiKind.EMAIL
    assert kinds["mobile"] == PiiKind.PHONE
    assert kinds["full_name"] == PiiKind.NAME
    # An empty column has no values to judge, so its name still marks it as personal.
    assert kinds["contact_email"] == PiiKind.EMAIL


def test_profile_explains_blanks_the_same_way_with_cached_blank_masks() -> None:
    rows = 60
    frame = pd.DataFrame(
        {
            "region": ["Gauteng", "Limpopo", "Free State"] * (rows // 3),
            # Blank only in Limpopo: the profile should tie the blanks to region (MAR).
            "fee": ["" if i % 3 == 1 else str(100 + i) for i in range(rows)],
            "score": [str(i % 7) for i in range(rows)],
        }
    )
    profile, _ = profile_frame(frame)
    fee = next(column for column in profile["columns"] if column["name"] == "fee")
    assert fee["missingness"] == "MAR"
    assert fee["missingDependsOn"] == "region"
