"""Detection and masking of personal information.

Columns are flagged by their name and by the shape of their values. Flagged columns are masked before data
moves beyond the raw layer, so personal information never reaches silver or gold in the clear.
"""

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import pandas as pd

from .values import is_blank, luhn_valid, sa_id_valid


class PiiKind(StrEnum):
    NAME = "name"
    NATIONAL_ID = "national_id"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    PAYMENT_CARD = "payment_card"


@dataclass(frozen=True)
class PiiFinding:
    column: str
    kind: PiiKind
    confidence: float
    reason: str


_NAME_HINTS: list[tuple[re.Pattern[str], PiiKind]] = [
    (re.compile(r"(^|_)(id_?number|id_?no|national_?id|identity|sa_?id|passport)(_|$)"), PiiKind.NATIONAL_ID),
    (re.compile(r"(^|_)(e_?mail)(_|$)"), PiiKind.EMAIL),
    (re.compile(r"(^|_)(phone|cell|mobile|tel|telephone|contact_?number)(_|$)"), PiiKind.PHONE),
    (re.compile(r"(^|_)(full_?name|first_?name|last_?name|surname|given_?name|name)(_|$)"), PiiKind.NAME),
    (re.compile(r"(^|_)(address|street|postal_?address|residential)(_|$)"), PiiKind.ADDRESS),
    (re.compile(r"(^|_)(dob|date_?of_?birth|birth_?date|birthday)(_|$)"), PiiKind.DATE_OF_BIRTH),
    (re.compile(r"(^|_)(card_?number|credit_?card|pan)(_|$)"), PiiKind.PAYMENT_CARD),
]
# Names that contain "name" but describe things, not people.
_NOT_PERSON_NAMES = re.compile(
    r"(course|product|dataset|file|table|column|region|company|org|team|metric|user_?name)"
)

_EMAIL = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$")
_PHONE = re.compile(r"^(\+?27|0)[\s-]?\d{2}[\s-]?\d{3}[\s-]?\d{4}$")
_CARD = re.compile(r"^\d{13,19}$")
_ALREADY_MASKED = "•"


def detect_pii(frame: pd.DataFrame, sample_size: int = 500) -> list[PiiFinding]:
    findings: list[PiiFinding] = []
    for column in frame.columns:
        name = str(column).lower()
        values = [str(v).strip() for v in frame[column].dropna().head(sample_size) if not is_blank(v)]
        finding = _by_values(str(column), values) or _by_name(str(column), name)
        if finding:
            findings.append(finding)
    return findings


def _by_name(column: str, name: str) -> PiiFinding | None:
    for pattern, kind in _NAME_HINTS:
        if pattern.search(name):
            if kind == PiiKind.NAME and _NOT_PERSON_NAMES.search(name):
                continue
            return PiiFinding(column, kind, 0.7, f"The column name suggests {kind.value.replace('_', ' ')}.")
    return None


def _share(values: list[str], test: Any) -> float:
    if not values:
        return 0.0
    return sum(1 for v in values if test(v)) / len(values)


def _by_values(column: str, values: list[str]) -> PiiFinding | None:
    if len(values) < 3:
        return None
    if _share(values, lambda v: _ALREADY_MASKED in v) > 0.5:
        return PiiFinding(column, PiiKind.NAME, 0.9, "Values are already masked.")
    checks: list[tuple[PiiKind, Any, str]] = [
        (
            PiiKind.NATIONAL_ID,
            sa_id_valid,
            "values are 13-digit South African ID numbers with valid check digits",
        ),
        (PiiKind.EMAIL, lambda v: bool(_EMAIL.match(v)), "values are email addresses"),
        (PiiKind.PHONE, lambda v: bool(_PHONE.match(v)), "values are South African phone numbers"),
        (
            PiiKind.PAYMENT_CARD,
            lambda v: bool(_CARD.match(re.sub(r"[\s-]", "", v))) and luhn_valid(re.sub(r"[\s-]", "", v)),
            "values are payment card numbers",
        ),
    ]
    for kind, test, reason in checks:
        share = _share(values, test)
        if share >= 0.6:
            return PiiFinding(column, kind, round(share, 2), f"{round(share * 100)}% of sampled {reason}.")
    return None


def mask_value(value: Any, kind: PiiKind | str) -> str | None:
    """Mask a single value, keeping just enough shape to stay recognisable in a preview."""
    if is_blank(value):
        return None
    text = str(value).strip()
    if _ALREADY_MASKED in text:
        return text
    kind = PiiKind(kind)
    if kind == PiiKind.NAME:
        return " ".join(part[0] + "•" * max(len(part) - 1, 1) for part in text.split() if part)
    if kind == PiiKind.EMAIL and "@" in text:
        local, _, domain = text.partition("@")
        return f"{local[:1]}•••@{domain}"
    digits_or_text = re.sub(r"\s", "", text)
    keep = 3 if kind in (PiiKind.NATIONAL_ID, PiiKind.PHONE, PiiKind.PAYMENT_CARD) else 0
    if keep and len(digits_or_text) > keep:
        return "•" * (len(digits_or_text) - keep) + digits_or_text[-keep:]
    return "•" * min(len(text), 12)


def pseudonymise(value: Any, salt: str) -> str | None:
    """A stable, non-reversible token for joining on a personal key without exposing it."""
    if is_blank(value):
        return None
    return hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()[:16]


def mask_series(series: pd.Series, kind: PiiKind | str) -> pd.Series:
    return series.map(lambda v: mask_value(v, kind))
