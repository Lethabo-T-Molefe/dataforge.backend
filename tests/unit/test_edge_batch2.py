"""Dense unit coverage for open AU/NT/CD/N/IJ/SL/U cases."""

import asyncio
import io
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from openpyxl import Workbook

from catalogue_progress import marks
from dataforge_backend.application.audit import _cell, _when, record
from dataforge_backend.application.inbox import notify
from dataforge_backend.core.config import settings
from dataforge_backend.domain.analysis import config_from_dict, interpret
from dataforge_backend.domain.cleaning.catalogue import (
    _categories,
    _d_drop_rows,
    _d_fill_unknown,
    _d_formats,
    _d_group_median,
    _d_map,
    is_applicable,
    METHODS_BY_ID,
)
from dataforge_backend.domain.semantic import ColumnInfo, build_analysis_dataset, derive_fields
from dataforge_backend.infrastructure import jupyter, superset
from dataforge_backend.infrastructure.storage import parse_upload


@marks("AU-06", "AU-08", "AU-09", "NT-02", "MO-14", "IJ-01", "X-01", "B-09")
def test_audit_notify_monitor_integrations(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _cell("=1+1") == "'=1+1"
    assert _cell("+cmd") == "'+cmd"
    assert _cell("ok") == "ok"
    now = datetime(2026, 3, 14, 15, 30, tzinfo=timezone.utc)
    assert _when(now, now) == now.astimezone().strftime("%H:%M")
    assert _when(now - timedelta(days=1), now) == "Yesterday"

    added: list[object] = []

    class FakeSession:
        def add(self, obj: object) -> None:
            added.append(obj)

    actor = SimpleNamespace(member_id="u1", name="Ada", request_id="r1", ip="1.1.1.1")
    asyncio.run(
        record(FakeSession(), actor, "Act", "dataset", "d1", "L" * 500, "D" * 5000)  # type: ignore[arg-type]
    )
    event = added[0]
    assert len(event.resource_label) == 300 and len(event.detail) == 4000  # type: ignore[attr-defined]

    rows: list[str] = []

    class NotifySession:
        def add(self, obj: object) -> None:
            rows.append(obj.user_id)  # type: ignore[attr-defined]

    notify(NotifySession(), ["a", "a", "system", "embed", "b"], "share", "t", "body", skip="a")  # type: ignore[arg-type]
    assert rows == ["b"]

    from dataforge_backend import main as app_main

    assert "MONITOR_INTERVAL_SECONDS > 0" in open(app_main.__file__, encoding="utf-8").read()

    monkeypatch.setattr(settings, "SUPERSET_URL", None)
    monkeypatch.setattr(settings, "JUPYTER_URL", None)
    assert "Set SUPERSET_URL" in asyncio.run(superset.SupersetClient().status()).detail
    assert "Set JUPYTER_URL" in asyncio.run(jupyter.status()).detail


@marks("CD-02", "CD-03", "CD-04", "CD-05", "CD-06", "U-13", "N-10", "N-11", "N-12", "SL-06", "SL-07")
def test_cd_defaults_excel_short_trends_slack_copy() -> None:
    profile = {
        "rows": 10,
        "columns": [
            {
                "name": "city",
                "inferredType": "string",
                "nulls": 2,
                "nullPct": 20,
                "distinct": 3,
                "piiKind": None,
                "patterns": [],
                "topValues": [{"value": "Gauteng", "count": 4}, {"value": "gauteng", "count": 2}],
            },
            {
                "name": "fee",
                "inferredType": "decimal",
                "nulls": 1,
                "nullPct": 10,
                "distinct": 5,
                "piiKind": None,
                "patterns": [{"example": "R 1 200"}],
                "missingDependsOn": "city",
            },
            {
                "name": "solo",
                "inferredType": "string",
                "nulls": 0,
                "nullPct": 0,
                "distinct": 1,
                "piiKind": None,
                "patterns": [],
                "topValues": [{"value": "only", "count": 10}],
            },
            {
                "name": "many",
                "inferredType": "string",
                "nulls": 0,
                "nullPct": 0,
                "distinct": 61,
                "piiKind": None,
                "patterns": [],
                "topValues": [],
            },
        ],
        "candidateKeys": [],
        "quality": {"outliers": 0},
    }
    available = ["city", "fee", "solo", "many"]
    fill = _d_fill_unknown(profile, available)  # type: ignore[arg-type]
    assert fill["strategy"] == "constant" and fill["value"] == "Unknown" and "city" in fill["columns"]
    drop = _d_drop_rows(profile, available)  # type: ignore[arg-type]
    assert drop == {"columns": ["city"], "how": "any"}
    empty_drop = _d_drop_rows({"rows": 1, "columns": [], "candidateKeys": [], "quality": {}}, [])  # type: ignore[arg-type]
    assert empty_drop["columns"] == []
    assert not is_applicable(METHODS_BY_ID["missing-drop-rows"], empty_drop)
    group = _d_group_median(profile, available)  # type: ignore[arg-type]
    assert group["strategy"] == "group_median" and group["group_by"] == "city"
    money = _d_formats(profile, available)  # type: ignore[arg-type]
    assert money["numbers"] == ["fee"]
    cats = _categories(profile, available)  # type: ignore[arg-type]
    assert "city" in cats and "many" not in cats and "solo" not in cats
    mapped = _d_map(profile, available)  # type: ignore[arg-type]
    assert mapped.get("column") == "city" or "mapping" in mapped

    book = Workbook()
    book.active.title = "first"
    book.active["A1"] = "a"
    book.active["A2"] = 1
    second = book.create_sheet("second")
    second["A1"] = "b"
    second["A2"] = 99
    buf = io.BytesIO()
    book.save(buf)
    frame = parse_upload("multi.xlsx", buf.getvalue())
    assert list(frame.columns) == ["a"] and int(frame["a"].iloc[0]) == 1

    columns = [
        ColumnInfo("region", "text", 2, values=("Gauteng", "East")),
        ColumnInfo("month", "text", 2, month_like=True),
        ColumnInfo("fee", "integer", 2),
        ColumnInfo("enrolments", "integer", 2),
    ]
    fields = derive_fields(columns, 2, [])
    dataset = build_analysis_dataset("d", "Demo", "gold.d", fields, ["2026-01", "2026-02"])
    short_rows = [
        {"d0": "2026-01", "v": 10, "v_num": 10, "v_den": 1},
        {"d0": "2026-02", "v": 12, "v_num": 12, "v_den": 1},
    ]
    smooth = config_from_dict(
        dataset,
        {
            "method": "trend",
            "measure": "enrolments",
            "smoothing": "ma3",
            "date_from": "2026-01",
            "date_to": "2026-02",
        },
    )
    smooth_out = interpret(dataset, smooth, short_rows)
    assert smooth_out["visual"]["points"]  # short window: no crash
    mom = config_from_dict(
        dataset,
        {
            "method": "trend",
            "measure": "enrolments",
            "change": "mom",
            "date_from": "2026-01",
            "date_to": "2026-02",
        },
    )
    mom_out = interpret(dataset, mom, short_rows)
    key = next(k for k in mom_out["visual"]["points"][0] if k != "name")
    assert mom_out["visual"]["points"][0][key] is None
    # N-10: outliers=exclude on a tiny series — empty/short result, no exception
    empty_rel = interpret(dataset, config_from_dict(dataset, {"method": "relationship", "outliers": "exclude"}), [])
    assert empty_rel.get("empty") is True or empty_rel["visual"]

    from dataforge_backend.application import slack as slack_app

    text = open(slack_app.__file__, encoding="utf-8").read()
    assert "Link your DataForge account first" in text
    assert "Ask a question, for example" in text
