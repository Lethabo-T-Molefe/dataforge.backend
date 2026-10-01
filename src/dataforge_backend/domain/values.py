"""Parsing of raw text values, shared by profiling and cleaning so they agree on what a value is."""

import math
import re
from datetime import date, datetime
from functools import lru_cache
from typing import Any

import pandas as pd

SENTINELS = frozenset(
    {"", "n/a", "na", "null", "none", "nil", "-", "--", "?", "nan", "#n/a", "missing", "unknown?"}
)
# Numeric sentinels are only treated as missing when the cleaning step says so (0 can be a real fee).
NUMERIC_SENTINELS = frozenset({"9999", "-999", "-9999", "99999"})

_CURRENCY = re.compile(r"^(?:ZAR|R|USD|\$|EUR|€|GBP|£)\s*", re.IGNORECASE)
_NUMBER_BODY = re.compile(r"^[+-]?[\d\s.,']+$")

_DATE_FORMATS_DAY_FIRST = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%d.%m.%Y",
    "%d %B %Y",
    "%d %b %Y",
    "%B %d %Y",
    "%b %d %Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%d/%m/%Y %H:%M",
)
_DATE_FORMATS_MONTH_FIRST = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m-%d-%Y",
    "%m/%d/%Y",
    "%B %d %Y",
    "%b %d %Y",
    "%B %d, %Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
)


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, str) and value.strip() == ""


def is_sentinel(value: Any, *, include_numeric: bool = False) -> bool:
    if is_blank(value):
        return True
    text = str(value).strip().lower()
    return text in SENTINELS or (include_numeric and text in NUMERIC_SENTINELS)


def thousands(value: float | int) -> str:
    """48213 → "48 213" (South African digit grouping)."""
    return f"{value:,}".replace(",", " ")


def parse_number(value: Any) -> float | None:
    """Parse a number written in any common way: 1250, R 1 250.00, 1 250,00, R1,250.00, (40)."""
    if value is None or is_blank(value):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return None if isinstance(value, float) and math.isnan(value) else float(value)
    return _parse_number_text(str(value).strip())


@lru_cache(maxsize=200_000)
def _parse_number_text(text: str) -> float | None:
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()").strip()
    text = _CURRENCY.sub("", text).replace(" ", " ").strip()
    if text.endswith("%"):
        text = text[:-1].strip()
    if not text or not _NUMBER_BODY.match(text):
        return None
    sign = -1.0 if text.startswith("-") or negative else 1.0
    text = text.lstrip("+-").replace(" ", "").replace("'", "")
    if "," in text and "." in text:
        # Whichever separator comes last is the decimal mark.
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        if re.fullmatch(r"\d{1,3}(,\d{3})+", text):
            # 1,250 and 1,250,000 group thousands.
            text = text.replace(",", "")
        elif text.count(",") == 1:
            # 1250,5 and 12,50 use a decimal comma.
            text = text.replace(",", ".")
        else:
            return None
    elif text.count(".") > 1:
        if re.fullmatch(r"\d{1,3}(\.\d{3})+", text):
            text = text.replace(".", "")
        else:
            head, _, tail = text.rpartition(".")
            text = f"{head.replace('.', '')}.{tail}"
    try:
        return sign * float(text)
    except ValueError:
        return None


def parse_date(value: Any, *, day_first: bool = True) -> date | None:
    """Parse a date written in any of the common shapes. Day-first by default (South African convention)."""
    if value is None or is_blank(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return _parse_date_text(str(value).strip(), day_first)


_ISO_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


@lru_cache(maxsize=200_000)
def _parse_date_text(raw: str, day_first: bool) -> date | None:
    iso = _ISO_DATE.match(raw)
    if iso:
        try:
            return date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        except ValueError:
            return None
    if not any(ch.isdigit() for ch in raw):
        return None
    text = re.sub(r"\s+", " ", raw)
    text = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text)
    for fmt in _DATE_FORMATS_DAY_FIRST if day_first else _DATE_FORMATS_MONTH_FIRST:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def date_shape(value: Any) -> str | None:
    """The written form of a date, used to detect mixed formats (2026/03/14 vs 14-03-2026 vs March 14 2026)."""
    if is_blank(value):
        return None
    text = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return "yyyy-mm-dd"
    if re.fullmatch(r"\d{4}/\d{1,2}/\d{1,2}", text):
        return "yyyy/mm/dd"
    if re.fullmatch(r"\d{1,2}-\d{1,2}-\d{4}", text):
        return "dd-mm-yyyy"
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", text):
        return "dd/mm/yyyy"
    if re.fullmatch(r"[A-Za-z]+ \d{1,2},? \d{4}", text):
        return "month dd yyyy"
    if re.fullmatch(r"\d{1,2} [A-Za-z]+ \d{4}", text):
        return "dd month yyyy"
    return "other"


def parse_bool(value: Any) -> bool | None:
    if is_blank(value):
        return None
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    return None


def shape(value: Any) -> str:
    """Reduce a value to its shape: letters become A (or a), digits become 9. 0821234567 → 9999999999."""
    out = []
    for ch in str(value):
        if ch.isdigit():
            out.append("9")
        elif ch.isalpha():
            out.append("A" if ch.isupper() else "a")
        else:
            out.append(ch)
    return "".join(out)


def fingerprint(value: Any) -> str:
    """OpenRefine-style key: lower case, punctuation removed, unique tokens sorted."""
    text = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in str(value))
    return " ".join(sorted({token for token in text.split() if token}))


def jaro_winkler(a: str, b: str) -> float:
    """Jaro-Winkler similarity in [0, 1]."""
    if a == b:
        return 1.0
    len_a, len_b = len(a), len(b)
    if not len_a or not len_b:
        return 0.0
    window = max(len_a, len_b) // 2 - 1
    matched_a = [False] * len_a
    matched_b = [False] * len_b
    matches = 0
    for i, ch in enumerate(a):
        for j in range(max(0, i - window), min(len_b, i + window + 1)):
            if not matched_b[j] and b[j] == ch:
                matched_a[i] = matched_b[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0
    transpositions = 0
    k = 0
    for i in range(len_a):
        if matched_a[i]:
            while not matched_b[k]:
                k += 1
            if a[i] != b[k]:
                transpositions += 1
            k += 1
    jaro = (matches / len_a + matches / len_b + (matches - transpositions / 2) / matches) / 3
    prefix = 0
    for ch_a, ch_b in zip(a[:4], b[:4], strict=False):
        if ch_a != ch_b:
            break
        prefix += 1
    return jaro + prefix * 0.1 * (1 - jaro)


def luhn_valid(digits: str) -> bool:
    if not digits.isdigit():
        return False
    total = 0
    for index, ch in enumerate(reversed(digits)):
        n = int(ch)
        if index % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def sa_id_valid(value: Any) -> bool:
    """South African ID number: 13 digits, a real birth date, and a Luhn check digit."""
    text = re.sub(r"\s", "", str(value))
    if not re.fullmatch(r"\d{13}", text):
        return False
    try:
        datetime.strptime(text[:6], "%y%m%d")
    except ValueError:
        return False
    return luhn_valid(text)


def normalise_za_phone(value: Any, country_code: str = "27") -> str | None:
    """Write a South African phone number in E.164 form (+27821234567)."""
    if is_blank(value):
        return None
    digits = re.sub(r"\D", "", str(value))
    if digits.startswith("00"):
        digits = digits[2:]
    if digits.startswith(country_code) and len(digits) == len(country_code) + 9:
        return f"+{digits}"
    if digits.startswith("0") and len(digits) == 10:
        return f"+{country_code}{digits[1:]}"
    if len(digits) == 9:
        return f"+{country_code}{digits}"
    return None
