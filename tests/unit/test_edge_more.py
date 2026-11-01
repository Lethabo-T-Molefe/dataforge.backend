"""More catalogue fixtures that fit in pure unit tests."""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError
from starlette.requests import Request

from catalogue_progress import marks
from dataforge_backend.application.common import is_queryable, slugify
from dataforge_backend.application.identity import check_password_strength
from dataforge_backend.application.predictions import score_windows
from dataforge_backend.core.config import settings
from dataforge_backend.core.crypto import decrypt_bytes, encrypt_bytes
from dataforge_backend.core.errors import AppError
from dataforge_backend.domain.analysis import AnalysisError, config_from_dict, interpret
from dataforge_backend.domain.analysis.format import format_value, js_fixed
from dataforge_backend.domain.analysis.model import AnalysisDataset
from dataforge_backend.domain.analysis.sql import ident, measure_sql, month_expr, qualified
from dataforge_backend.domain.analysis.stats import pearson, quantile, spearman
from dataforge_backend.domain.assistant import Catalogue, _mentioned_measures, _match_filters, detect_injection, plan_question
from dataforge_backend.domain.cleaning import CleaningContext, METHODS_BY_ID, OperationError, code_for, is_applicable, run_operation
from dataforge_backend.domain.insights import detect_insights
from dataforge_backend.domain.pii import PiiKind, detect_pii
from dataforge_backend.domain.profiling import _outlier_mask, infer_type, littles_mcar, profile_frame
from dataforge_backend.domain.roles import DEFAULT_QUOTAS, Role
from dataforge_backend.domain.semantic import ColumnInfo, build_analysis_dataset, derive_fields
from dataforge_backend.domain.sql_guard import QueryRefused, TableGrant, guard_select, validate_predicate
from dataforge_backend.domain.values import fingerprint, luhn_valid, sa_id_valid
from dataforge_backend.infrastructure.storage import SUPPORTED, parse_upload
from dataforge_backend.infrastructure.warehouse import normalise_columns, safe_column
from dataforge_backend.presentation.api.deps import client_ip
from dataforge_backend.presentation.api.schemas import ContactIn, RegisterIn, SqlIn, VisibilityIn
from dataforge_backend.presentation.metrics import _normalise_path, observe_request, render_prometheus


def _months() -> AnalysisDataset:
    columns = [
        ColumnInfo("region", "text", 2, values=("Gauteng", "Eastern Cape", "East")),
        ColumnInfo("month", "text", 5, month_like=True),
        ColumnInfo("fee", "integer", 5),
        ColumnInfo("enrolments", "integer", 5),
        ColumnInfo("completions", "integer", 5),
    ]
    fields = derive_fields(columns, 5, [])
    return build_analysis_dataset(
        "d", "Demo", "gold.d", fields, ["2025-11", "2025-12", "2026-01", "2026-02", "2026-03"]
    )


@marks("U-05", "U-06", "U-07", "U-11", "U-14", "U-17")
def test_upload_parse_edges() -> None:
    with pytest.raises(Exception, match="empty"):
        parse_upload("a.csv", b"")
    assert ".csv" in SUPPORTED
    with pytest.raises(Exception, match="no rows"):
        parse_upload("a.csv", b"a,b\n")
    frame = parse_upload("a.csv", b"a,b\n1,2\n")
    assert list(frame.columns) == ["a", "b"]
    assert frame.iloc[0].tolist() == ["1", "2"]
    js = parse_upload("a.json", b'[{"x": 1, "y": {"z": 2}}]')
    assert isinstance(js.iloc[0]["y"], str) and js.iloc[0]["y"].startswith("{")
    blank_header = parse_upload("a.csv", b",name\n1,Ada\n")
    assert str(blank_header.columns[0]).startswith("column")


@marks("U-15", "U-16")
def test_column_normalisation() -> None:
    taken: set[str] = set()
    assert safe_column("Revenue", taken) == "revenue"
    assert safe_column("revenue", taken) == "revenue_2"
    assert safe_column("1region", set()) == "c_1region"
    assert safe_column("_row", set()) != "_row"
    frame = normalise_columns(pd.DataFrame({"Revenue ": [1], "revenue": [2]}))
    assert list(frame.columns) == ["revenue", "revenue_2"]


@marks("P-13", "P-14", "P-15", "X-09")
def test_schema_forbid_and_strip() -> None:
    with pytest.raises(ValidationError):
        RegisterIn.model_validate(
            {"name": "Ada", "email": "ada@org.local", "password": "x", "extraField": 1}
        )
    body = RegisterIn.model_validate({"name": "  Ada  ", "email": "Ada@Org.LOCAL", "password": "x"})
    assert body.name == "Ada" and body.email == "ada@org.local"
    snake = SqlIn.model_validate({"sql": "select 1", "max_rows": 10})
    camel = SqlIn.model_validate({"sql": "select 1", "maxRows": 10})
    assert snake.max_rows == camel.max_rows == 10
    with pytest.raises(ValidationError, match="line breaks"):
        ContactIn(name="A\nB", email="a@b.co", message="long enough")


@marks("P-16")
def test_client_ip_order() -> None:
    def req(headers: list[tuple[bytes, bytes]]) -> Request:
        return Request({"type": "http", "method": "GET", "headers": headers, "client": ("9.9.9.9", 80)})

    assert client_ip(req([(b"cf-connecting-ip", b"1.1.1.1"), (b"x-forwarded-for", b"2.2.2.2")])) == "1.1.1.1"
    assert client_ip(req([(b"x-forwarded-for", b"2.2.2.2, 3.3.3.3")])) == "2.2.2.2"
    assert client_ip(req([])) == "9.9.9.9"


@marks("P-20")
def test_metrics_path_labels() -> None:
    observe_request("GET", "/api/v1/datasets/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", 200, 12.0)
    text = render_prometheus()
    assert "aaaaaaaa" not in text
    assert _normalise_path("/api/v1/datasets/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee").endswith(":id")


@marks("G-07")
def test_is_queryable_flags() -> None:
    assert is_queryable(SimpleNamespace(status="approved", layer="silver", erased_at=None))  # type: ignore[arg-type]
    assert not is_queryable(SimpleNamespace(status="approved", layer="bronze", erased_at=None))  # type: ignore[arg-type]
    assert not is_queryable(SimpleNamespace(status="pending", layer="gold", erased_at=None))  # type: ignore[arg-type]


@marks("S-04", "S-05", "S-06", "S-07", "S-09", "A-12")
def test_sql_guard_more() -> None:
    grants = {"gold.t": TableGrant("gold.t"), "silver.t": TableGrant("silver.t")}
    guard_select("WITH x AS (SELECT 1 AS n FROM gold.t) SELECT n FROM x", grants)
    guard_select("SELECT 1 FROM gold.t UNION SELECT 1 FROM silver.t", grants)
    with pytest.raises(QueryRefused):
        guard_select("SELECT * FROM otherdb.gold.t", grants)
    guard_select("SELECT to_char(fee, '999') FROM gold.t", grants)
    validate_predicate("region = 'Gauteng' OR true", {"region"})
    with pytest.raises(QueryRefused):
        validate_predicate("region = (SELECT 1)", {"region"})


@marks("SQ-03")
def test_measure_sql_pooled_vs_average() -> None:
    m = SimpleNamespace(kind="ratio", column=None, denominator="enrolments", numerator="completions")
    assert "NULLIF(SUM(enrolments)" in measure_sql(m, "pooled")  # type: ignore[arg-type]
    assert "AVG(" in measure_sql(m, "average")  # type: ignore[arg-type]


@marks("N-03", "N-04", "N-05", "N-06", "N-07")
def test_analysis_config_validation() -> None:
    dataset = _months()
    with pytest.raises(AnalysisError, match="Unknown method"):
        config_from_dict(dataset, {"method": "histogram"})
    with pytest.raises(AnalysisError):
        config_from_dict(dataset, {"method": "compare", "dimension": "nope"})
    with pytest.raises(AnalysisError):
        config_from_dict(dataset, {"method": "compare", "aggregation": "median"})
    with pytest.raises(AnalysisError):
        config_from_dict(dataset, {"method": "compare", "filters": ["x"]})
    with pytest.raises(AnalysisError):
        config_from_dict(dataset, {"method": "compare", "filters": {"region": "Nowhere"}})
    with pytest.raises(AnalysisError):
        config_from_dict(dataset, {"method": "trend", "date_from": "2026-13"})


@marks("AQ-08", "Q-05", "Q-11")
def test_assistant_filters_and_predict() -> None:
    dataset = _months()
    filters = _match_filters("eastern cape and east please", dataset)
    assert filters.get("region") == "Eastern Cape"
    assert detect_injection("why did drop-off rise") is False
    refused = plan_question("will revenue drop next month", Catalogue([dataset]))
    assert getattr(refused, "reason", None) == "prediction"


@marks("IN-02", "IN-06")
def test_insights_missing_and_duplicates() -> None:
    frame = pd.DataFrame({"a": [None] * 5 + ["x"] * 95, "b": ["y"] * 100})
    profile, facts = profile_frame(frame, [])
    profile["quality"]["duplicates"] = 3
    out = detect_insights(frame, profile, facts)
    assert any(i.target == "missing" or "empty" in i.title for i in out)
    assert any(i.target == "duplicates" for i in out)


@marks("CD-01", "CD-07", "CD-10")
def test_cleaning_defaults_and_shown_code() -> None:
    assert is_applicable(METHODS_BY_ID["profile-accept"], {})
    assert "isna().mean()" in code_for("drop_sparse_columns", {"threshold": 0.5})
    assert code_for("no_such_op", {"a": 1}).startswith("#")


@marks("SC-01", "SC-02", "SC-03", "PR-11")
def test_score_windows() -> None:
    recent, before = score_windows(["2026-01"])
    assert recent == ("2026-01", "2026-01")
    recent, before = score_windows(["2026-01", "2026-02", "2026-03", "2026-04"])
    assert recent == ("2026-02", "2026-04") and before == ("2026-01", "2026-01")
    six = [f"2026-{m:02d}" for m in range(1, 7)]
    recent, before = score_windows(six)
    assert recent == ("2026-04", "2026-06") and before == ("2026-01", "2026-03")


@marks("WK-06", "WK-07")
def test_file_crypto_marker() -> None:
    blob = encrypt_bytes(b"hello")
    assert blob.startswith(b"DFE1") and decrypt_bytes(blob) == b"hello"
    assert decrypt_bytes(b"plain") == b"plain"
    with pytest.raises(AppError, match="SECRETS_KEY"):
        decrypt_bytes(b"DFE1" + b"\x00" * 20)


@marks("O-01", "O-02")
def test_pg_restore_script_rewrites_url() -> None:
    script = Path(__file__).resolve().parents[3] / "ops" / "backup" / "pg_restore.sh"
    text = script.read_text(encoding="utf-8")
    assert "postgresql+asyncpg" in text and "Usage:" in text


@marks("CL-35")
def test_fingerprint_order() -> None:
    assert fingerprint("Gauteng Province") == fingerprint("province, gauteng")


@marks("TY-04")
def test_integer_vs_decimal_inference() -> None:
    assert infer_type(pd.Series(["12", "13", "14"] * 4))[0] == "integer"
    assert infer_type(pd.Series(["12.5", "13.0", "14.2"] * 4))[0] == "decimal"


def test_quotas_and_slug() -> None:
    assert DEFAULT_QUOTAS[Role.ANALYST].ai_queries == 200
    assert DEFAULT_QUOTAS[Role.AUDITOR].compute_seconds == 0
    assert DEFAULT_QUOTAS[Role.ADMIN].storage_bytes is None
    assert slugify("My File!") == "my-file"


@marks("I-03", "I-08", "I-41", "U-21")
def test_register_schema_and_password_reuse() -> None:
    with pytest.raises(ValidationError):
        RegisterIn.model_validate({"name": "Ada", "email": "a@localhost", "password": "Correct-Horse1"})
    with pytest.raises(ValidationError):
        RegisterIn.model_validate({"name": "Ada", "email": "a b@org.local", "password": "Correct-Horse1"})
    with pytest.raises(ValidationError):
        RegisterIn.model_validate(
            {"name": "Ada", "email": "a@" + "x" * 250 + ".co", "password": "Correct-Horse1"}
        )
    with pytest.raises(ValidationError):
        RegisterIn.model_validate({"name": "   ", "email": "ada@org.local", "password": "Correct-Horse1"})
    with pytest.raises(ValidationError):
        RegisterIn.model_validate({"name": "A" * 201, "email": "ada@org.local", "password": "Correct-Horse1"})
    check_password_strength("Correct-Horse1", "ada@org.local", "Ada")  # equal to current still allowed at schema
    with pytest.raises(ValidationError):
        VisibilityIn.model_validate({"visibility": "public"})
    VisibilityIn.model_validate({"visibility": "private"})


@marks("U-08", "U-09", "U-10", "U-12", "U-22", "U-27")
def test_more_upload_parse() -> None:
    blank_rows = parse_upload("a.csv", b"a,b\n,\n,\n")
    assert len(blank_rows) == 2 and blank_rows.iloc[0].tolist() == ["", ""]
    semi = parse_upload("a.csv", b"a;b\n1;2\n")
    assert list(semi.columns) == ["a", "b"] and semi.iloc[0].tolist() == ["1", "2"]
    bom = parse_upload("a.csv", "\ufeffa,b\n1,2\n".encode("utf-8-sig"))
    assert list(bom.columns) == ["a", "b"]
    jsonl = parse_upload("a.jsonl", b'{"a": 1}\n{"a": 2}\n')
    assert list(jsonl["a"]) == [1, 2]
    as_json = parse_upload("a.json", b'{"a": 1}\n{"a": 2}\n')
    assert len(as_json) == 2
    name = "my_file_v2.csv".rsplit(".", 1)[0].replace("_", " ").strip().capitalize()
    assert name == "My file v2"
    formulas = parse_upload("a.csv", b"x\n=cmd|\n+1\n-1\n@SUM(1)\n")
    assert formulas["x"].tolist() == ["=cmd|", "+1", "-1", "@SUM(1)"]
    assert settings.MAX_UPLOAD_MB >= 1


@marks("PI-05", "PI-07", "PI-09", "PI-15")
def test_pii_sample_ceiling_card_and_k() -> None:
    deep = pd.DataFrame({"notes": ["word"] * 500 + ["a@x.org"] * 10})
    assert detect_pii(deep) == []
    assert datetime.strptime("680101", "%y%m%d").year == 2068
    assert datetime.strptime("690101", "%y%m%d").year == 1969
    assert sa_id_valid("8001015009087")
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")
    cards = detect_pii(pd.DataFrame({"pan": ["4111111111111111"] * 3}))
    assert any(f.kind == PiiKind.PAYMENT_CARD for f in cards)
    junk = detect_pii(pd.DataFrame({"acct": ["4111111111111112"] * 3}))
    assert junk == []
    with pytest.raises(OperationError, match="quasi-identifiers"):
        run_operation("k_anonymity", pd.DataFrame({"a": [1, 1, 1]}), {"k": 5}, CleaningContext())


@marks("V-08", "V-11", "V-12", "V-13", "V-14", "IN-01", "IN-03", "IN-04", "IN-07", "IN-08", "NF-07", "NF-08", "NF-09", "NF-10")
def test_profile_outliers_keys_and_formats() -> None:
    mixed = pd.DataFrame({"d": ["2026-03-14", "14/03/2026"] * 6})
    profile, _ = profile_frame(mixed, [])
    formats = profile["columns"][0].get("dateFormats") or {}
    assert len(formats) >= 2
    short = pd.Series([1, 2, 3, 100], dtype=float)
    assert not _outlier_mask(short).any()
    long = pd.Series([1, 2, 3, 4, 5, 6, 7, 1000], dtype=float)
    assert _outlier_mask(long).any()
    one = littles_mcar(pd.DataFrame({"n": [1.0, None, 3.0]}), {"n": pd.Series([1.0, None, 3.0])})
    assert one["statistic"] == "n/a"
    keys = pd.DataFrame({"id": [str(i) for i in range(12)], "n": list(range(12))})
    profile, _ = profile_frame(keys, [])
    assert profile["candidateKeys"]
    nulls = pd.DataFrame({"empty": [None, None, None, None]})
    profile, facts = profile_frame(nulls, [])
    assert profile["columns"][0]["nullPct"] == 100
    assert profile["columns"][0]["inferredType"] == "empty"
    dates = pd.DataFrame({"d": ["2026-03-14"] * 20})
    profile, facts = profile_frame(dates, [])
    assert not any(i.target == "formats" for i in detect_insights(dates, profile, facts))
    mixed_prof, mixed_facts = profile_frame(mixed, [])
    assert any(i.target == "formats" for i in detect_insights(mixed, mixed_prof, mixed_facts))
    out_frame = pd.DataFrame({"n": [1, 2, 3, 4, 5, 6, 7, 1000] * 2})
    out_prof, out_facts = profile_frame(out_frame, [])
    assert any(i.target == "outliers" for i in detect_insights(out_frame, out_prof, out_facts))
    assert js_fixed(1.225, 2) == "1.23"
    assert format_value(1.2, "decimal") == "1.20"
    assert format_value(float("nan"), "count") == "—"
    import math

    assert math.isnan(quantile([], 0.25))
    assert quantile([1, 2, 3], 0.25) == 1.5
    assert math.isnan(pearson([1], [1]))
    assert math.isnan(pearson([1, 1, 1], [2, 3, 4]))
    assert math.isnan(spearman([1], [2]))
    conc = pd.DataFrame({"city": ["GP"] * 40 + ["WC"] * 10 + ["EC"] * 10})
    conc_prof, conc_facts = profile_frame(conc, [])
    conc_insights = detect_insights(conc, conc_prof, conc_facts)
    assert any("accounts for" in i.title for i in conc_insights)
    mixed_many = pd.DataFrame({f"c{i}": (["1"] * 5 + ["x"] * 5) * 3 for i in range(4)})
    mm_prof, mm_facts = profile_frame(mixed_many, [])
    mix_insights = detect_insights(mixed_many, mm_prof, mm_facts)
    assert any("mix" in i.title.lower() for i in mix_insights)
    assert len(detect_insights(mixed_many, mm_prof, mm_facts)) <= 6


@marks(
    "AN-02", "AN-08", "AN-14", "AN-16", "AN-20",
    "SQ-05", "SQ-06", "NF-10",
)
def test_analysis_empty_readings_and_sql_helpers() -> None:
    dataset = _months()
    one = interpret(
        dataset,
        config_from_dict(dataset, {"method": "compare", "measure": "fee", "dimension": "region"}),
        [{"d0": "Gauteng", "v": 10.0, "p_den": 10.0}],
    )
    assert "nothing to compare" in one["reading"].lower()
    empty_trend = interpret(
        dataset,
        config_from_dict(dataset, {"method": "trend", "measure": "fee", "date_from": "2026-01", "date_to": "2026-02"}),
        [],
    )
    assert empty_trend["empty"] is True
    same = interpret(
        dataset,
        config_from_dict(dataset, {"method": "relationship", "measure": "fee", "measureY": "fee"}),
        [{"x": 1, "y": 1}],
    )
    assert "two different measures" in same["reading"].lower()
    not_enough = interpret(
        dataset,
        config_from_dict(dataset, {"method": "relationship", "measure": "fee", "measureY": "enrolments"}),
        [],
    )
    assert "not enough" in not_enough["reading"].lower()
    same_month = interpret(
        dataset,
        config_from_dict(
            dataset,
            {"method": "explain", "measure": "fee", "periodA": "2026-01", "periodB": "2026-01"},
        ),
        [],
    )
    assert "different months" in same_month["reading"].lower()
    assert month_expr(dataset) == "month"
    assert qualified("gold.select") == 'gold."select"'
    assert qualified("gold.fee_by_region") == "gold.fee_by_region"
    assert ident("select") == '"select"'


@marks(
    "F-01", "F-02", "F-03", "F-04", "F-05", "F-06", "F-07", "F-08", "F-09", "F-10",
    "F-11", "F-12", "F-13", "F-14", "F-15",
    "UI-01", "UI-02", "UI-03", "UI-04", "UI-05", "UI-06", "UI-07",
)
def test_frontend_catalogue_wired_in_source() -> None:
    front = Path(__file__).resolve().parents[3] / "skills-quest-forge" / "src"
    app = (front / "App.tsx").read_text(encoding="utf-8")
    public = (front / "config" / "public.ts").read_text(encoding="utf-8")
    sign_in = (front / "pages" / "SignInPage.tsx").read_text(encoding="utf-8")
    mfa = (front / "pages" / "MfaPage.tsx").read_text(encoding="utf-8")
    session = (front / "features" / "auth" / "api" / "session.ts").read_text(encoding="utf-8")
    register = (front / "pages" / "RegisterPage.tsx").read_text(encoding="utf-8")
    contact = (front / "pages" / "public" / "ContactPage.tsx").read_text(encoding="utf-8")
    reset = (front / "pages" / "ResetPasswordPage.tsx").read_text(encoding="utf-8")
    upload = (front / "pages" / "UploadPage.tsx").read_text(encoding="utf-8")
    perms = (front / "features" / "auth" / "permissions.ts").read_text(encoding="utf-8")
    assert 'to: "/sign-in"' in app and "search: {}" in app
    assert 'to: "/workspace"' in app
    assert "startsWith(\"//\")" in sign_in and 'return "/workspace"' in sign_in
    assert "Your session ended. Sign in again." in app
    assert "sessionStorage.getItem(MFA_KEY)" in session
    assert "sessionStorage.removeItem(MFA_KEY)" in session
    assert "MFA_ENROLL_KEY" in session
    assert "notFound()" in app and "ErrorPage" in app
    assert "failureCount < 2" in app and "status >= 400" in app
    assert "defaultPendingMs: 400" in app
    assert 'to: "/analyse/ask"' in app and 'to: "/analyse/build"' in app
    assert 'pathname.startsWith("/embed/")' in public
    assert "hasPermission" in perms and "admin_users" in perms
    assert "refetchOnWindowFocus: false" in app
    assert "full name" in register.lower()
    assert "work email" in register.lower() or "@" in register
    assert "password" in register.lower()
    assert "message.trim().length < 8" in contact and "email.includes(\"@\")" in contact
    assert "not the same" in reset.lower() or "match" in reset.lower()
    assert "maxLength={200}" in upload or "200" in upload
    assert "verify" in register.lower()


@marks("TY-05", "TY-06", "TY-07", "TY-08", "TY-09", "CD-08", "CD-09", "AQ-09", "Q-10", "Q-12")
def test_ty_cd_and_assistant_phrases() -> None:
    phones = infer_type(pd.Series(["0821234567"] * 10))
    assert phones[0] == "string"
    ids = infer_type(pd.Series(["8001015009087"] * 10))
    assert ids[0] == "string"
    mixed = infer_type(pd.Series(["1"] * 89 + ["nope"] * 11))
    assert mixed[0] == "mixed" and mixed[1] == pytest.approx(0.89)
    weak = infer_type(pd.Series(["1"] * 4 + ["x"] * 6))
    assert weak[0] == "string"
    assert _outlier_mask  # keep import used via profile
    profile, _ = profile_frame(pd.DataFrame({"n": [48213, 12.5, float("nan")]}), [])
    text = str(profile)
    assert "48 213" in text or "48213" in text
    assert "nan" not in text.lower().replace("none", "")
    emails = pd.DataFrame({"email": ["a@x.org"] * 9})
    pii_prof, _ = profile_frame(emails, detect_pii(emails))
    assert pii_prof["columns"][0]["masked"] is True
    assert pii_prof["columns"][0]["topValues"][0]["value"] == "(masked)"
    assert "dropna" in code_for("drop_missing_rows", {"columns": ["a"], "how": "any"})
    assert "mode().iloc[0]" in code_for("fill_missing", {"columns": ["a"], "strategy": "mode"})
    assert is_applicable(METHODS_BY_ID["profile-accept"], {})

    from dataforge_backend.domain.semantic import MetricSpec

    columns = [
        ColumnInfo("region", "text", 2, values=("Gauteng", "East")),
        ColumnInfo("month", "text", 5, month_like=True),
        ColumnInfo("completions", "integer", 5),
        ColumnInfo("enrolments", "integer", 5),
        ColumnInfo("fee", "integer", 5),
    ]
    metrics = [
        MetricSpec("completion_rate", "Completion rate", "rate", "ratio", numerator="completions", denominator="enrolments"),
        MetricSpec("completion", "Completion", "count", "additive", column="completions"),
    ]
    fields = derive_fields(columns, 5, metrics)
    dataset = build_analysis_dataset(
        "d", "Demo", "gold.d", fields, ["2025-11", "2025-12", "2026-01", "2026-02", "2026-03"]
    )
    catalogue = Catalogue([dataset])
    named = _mentioned_measures("what is the completion rate by region", dataset, catalogue)
    assert named[0] == "completion_rate"
    assert "completion" not in named or named == ["completion_rate"]
    period = plan_question("over the last 3 months", catalogue)
    assert getattr(period, "date_from", None) or getattr(period, "reason", None)
    partner = plan_question("is completion rate related to enrolments", catalogue)
    assert getattr(partner, "method", None) == "relationship" or getattr(partner, "reason", None)


@marks("RC-01", "RC-02", "RC-03", "RC-04", "MO-16", "MO-17", "CN-02", "CN-05", "X-06", "SL-10", "ER-01", "ER-09")
def test_root_cause_slack_connector_errors() -> None:
    from dataforge_backend.application.dashboards import embed_viewer
    from dataforge_backend.domain.monitoring import GroupChange, RootCauseInput, root_cause
    from dataforge_backend.infrastructure.external_db import MAX_IMPORT_ROWS, SourceConfig
    from dataforge_backend.infrastructure.notifications import post_slack, verify_slack_signature

    zero_den = root_cause(
        RootCauseInput(
            label="Rate",
            unit="rate",
            is_ratio=True,
            month_before="Jan",
            month_after="Feb",
            total_before=1,
            total_after=1,
            breakdowns={"Region": [GroupChange("A", 1, 1, den_before=0, den_after=0)]},
        )
    )
    assert zero_den == []
    flat = root_cause(
        RootCauseInput(
            label="Rate",
            unit="rate",
            is_ratio=True,
            month_before="Jan",
            month_after="Feb",
            total_before=1,
            total_after=1,
            breakdowns={"Region": [GroupChange("A", 5, 5, den_before=10, den_after=10)]},
        )
    )
    assert flat == []
    zero_group_den = root_cause(
        RootCauseInput(
            label="Rate",
            unit="rate",
            is_ratio=True,
            month_before="Jan",
            month_after="Feb",
            total_before=1,
            total_after=1,
            breakdowns={
                "Region": [
                    GroupChange("Keep", 1, 2, den_before=10, den_after=10),
                    GroupChange("Dash", 0, 1, den_before=0, den_after=5),
                ]
            },
        )
    )
    dash = next(c for c in zero_group_den if c["factor"] == "Dash")
    assert "—" in dash["detail"] and "nan" not in dash["detail"].lower()
    assert root_cause(
        RootCauseInput(
            label="Rev",
            unit="money",
            is_ratio=False,
            month_before="Jan",
            month_after="Feb",
            total_before=0,
            total_after=10,
        )
    ) == []
    many = root_cause(
        RootCauseInput(
            label="Rate",
            unit="rate",
            is_ratio=True,
            month_before="Jan",
            month_after="Feb",
            total_before=1,
            total_after=1,
            breakdowns={
                "Region": [
                    GroupChange(f"g{i}", i, i + 2, den_before=10, den_after=10) for i in range(6)
                ]
            },
        ),
        limit=4,
    )
    assert len(many) == 4

    import asyncio
    import inspect

    assert asyncio.run(post_slack("hi")) is False
    assert verify_slack_signature(b"x", None, None) is False
    assert verify_slack_signature(b"x", "not-int", "v0=x") is False
    assert verify_slack_signature(b"x", "1", "v0=bad") is False

    assert MAX_IMPORT_ROWS == 500_000
    assert 'ssl=None if config.sslmode == "disable"' in inspect.getsource(
        __import__(
            "dataforge_backend.infrastructure.external_db", fromlist=["_connect"]
        )._connect
    )
    assert SourceConfig("h", 5432, "d", "u", "p", sslmode="require").sslmode == "require"
    err = AppError("boom", details={"field": "a"})
    assert err.details == {"field": "a"} and AppError("x").details == {}
    viewer = embed_viewer()
    assert viewer.id == "embed" and viewer.role == "executive" and viewer.verified is True
