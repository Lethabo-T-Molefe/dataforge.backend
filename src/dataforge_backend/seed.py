"""Demo data for the DataForge prototype, created through the real pipeline.

Every dataset is ingested, profiled, cleaned with a recipe, approved and (where relevant) promoted exactly
as a member would do it, so the demo exercises the same code paths as production. Numbers come from the
same seeded generator as the frontend's mock data, so the two agree.
"""

import math
import random
from collections.abc import Callable
from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .application import assistant, cleaning, datasets, monitors, predictions, workbench
from .application.common import Actor
from .core.config import settings
from .core.logging import get_logger
from .core.security import hash_password, utcnow
from .infrastructure.db.models import (
    Connector,
    Dashboard,
    DashboardChart,
    Dataset,
    MetricDefinition,
    Monitor,
    User,
)

log = get_logger(__name__)

MONTHS = ["2025-10", "2025-11", "2025-12", "2026-01", "2026-02", "2026-03"]
REGIONS = ["Gauteng", "Western Cape", "KwaZulu-Natal", "Eastern Cape", "Limpopo", "Free State", "Mpumalanga"]
COURSES = ["Data literacy", "SQL basics", "Python for analysts", "Stewardship 101"]
_REGION_ENROLMENTS = {
    "Gauteng": 3290,
    "Western Cape": 1520,
    "KwaZulu-Natal": 1247,
    "Eastern Cape": 718,
    "Limpopo": 530,
    "Free State": 398,
    "Mpumalanga": 332,
}
_COURSE_SHARE = {
    "Data literacy": 0.4,
    "SQL basics": 0.22,
    "Python for analysts": 0.26,
    "Stewardship 101": 0.12,
}
_SEASON = [0.92, 0.98, 0.85, 1.08, 1.15, 1.02]
_COMPLETION = {
    "Data literacy": (0.672, 0.004),
    "SQL basics": (0.63, 0.017),
    "Python for analysts": (0.628, 0.004),
    "Stewardship 101": (0.585, 0.003),
}
_REGION_SHIFT = {
    "Gauteng": 0.02,
    "Western Cape": 0.035,
    "KwaZulu-Natal": -0.005,
    "Eastern Cape": -0.02,
    "Limpopo": -0.03,
    "Free State": -0.025,
    "Mpumalanga": -0.02,
}
_COURSE_FEE = {
    "Data literacy": 1080,
    "SQL basics": 1240,
    "Python for analysts": 1460,
    "Stewardship 101": 1020,
}
_REGION_FEE = {
    "Gauteng": 1.11,
    "Western Cape": 1.06,
    "KwaZulu-Natal": 0.95,
    "Eastern Cape": 0.86,
    "Limpopo": 0.84,
    "Free State": 0.81,
    "Mpumalanga": 0.83,
}

MEMBERS = [
    ("thandi", "Thandi Mokoena", "analyst", "Learning operations"),
    ("sipho", "Sipho Dlamini", "practitioner", "Learning operations"),
    ("naledi", "Naledi Khumalo", "steward", "Learning operations"),
    ("lerato", "Lerato Naidoo", "executive", None),
    ("amina", "Amina Patel", "admin", None),
    ("johan", "Johan Venter", "auditor", None),
    ("zola", "Zola Nkosi", "curator", None),
]


def mulberry32(seed: int) -> Callable[[], float]:
    """The frontend's tiny deterministic PRNG, bit for bit, so both sides generate the same numbers."""
    state = seed & 0xFFFFFFFF
    mask = 0xFFFFFFFF

    def next_value() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & mask
        t = ((state ^ (state >> 15)) * (1 | state)) & mask
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & mask)) & mask) ^ t
        return ((t ^ (t >> 14)) & mask) / 4294967296

    return next_value


def js_round(value: float) -> int:
    return math.floor(value + 0.5)


def fact_rows() -> list[dict[str, Any]]:
    """One row per region, course and month, as in the frontend's analysis/data/facts.ts."""
    random_value = mulberry32(20260314)

    def noise() -> float:
        return random_value() * 2 - 1

    rows = []
    for month_index, month in enumerate(MONTHS):
        for region in REGIONS:
            for course in COURSES:
                free_state_march = region == "Free State" and month == "2026-03"
                volume = (
                    _REGION_ENROLMENTS[region]
                    * _COURSE_SHARE[course]
                    * _SEASON[month_index]
                    * (1 + noise() * 0.06)
                    * (0.78 if free_state_march else 1)
                )
                enrolments = max(1, js_round(volume))
                base, per_month = _COMPLETION[course]
                rate = min(0.95, base + per_month * month_index + _REGION_SHIFT[region] + noise() * 0.015)
                completions = js_round(enrolments * rate)
                fee = (
                    _COURSE_FEE[course]
                    * _REGION_FEE[region]
                    * (0.94 if free_state_march else 1)
                    * (1 + noise() * 0.02)
                )
                revenue = js_round(enrolments * fee)
                sessions = js_round(enrolments * (1.5 + 2.2 * rate + noise() * 0.15))
                rows.append(
                    {
                        "region": region,
                        "course": course,
                        "month": month,
                        "enrolments": enrolments,
                        "completions": completions,
                        "revenue": revenue,
                        "sessions": sessions,
                    }
                )
    return rows


def completions_frame() -> pd.DataFrame:
    """Raw export as a warehouse would send it: numbers as text, a few formatting quirks to clean."""
    rows = [{k: v for k, v in r.items() if k != "revenue"} for r in fact_rows()]
    frame = pd.DataFrame(rows).astype(str).astype(object)
    return frame


def revenue_frame() -> pd.DataFrame:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in fact_rows():
        key = (row["region"], row["month"])
        entry = by_key.setdefault(
            key,
            {
                "region": row["region"],
                "month": f"{row['month']}-01",
                "enrolments": 0,
                "revenue": 0,
                "active_learners": 0,
            },
        )
        entry["enrolments"] += row["enrolments"]
        entry["revenue"] += row["revenue"]
    for entry in by_key.values():
        # Learners with a session in the month: most enrolled learners, plus returning ones.
        entry["active_learners"] = js_round(entry["enrolments"] * 1.54)
    return pd.DataFrame(list(by_key.values())).astype(str).astype(object)


def enrolments_frame(rows: int = 4800) -> pd.DataFrame:
    """A messy raw upload: mixed date formats, currency text, placeholders, variant labels, personal data."""
    rng = random.Random(20260314)
    first = [
        "Thandi",
        "Sipho",
        "Naledi",
        "Lerato",
        "Johan",
        "Zola",
        "Amina",
        "Kagiso",
        "Ayanda",
        "Pieter",
        "Nomvula",
        "Bongani",
    ]
    last = [
        "Mokoena",
        "Dlamini",
        "Khumalo",
        "Naidoo",
        "Venter",
        "Nkosi",
        "Patel",
        "Molefe",
        "Zulu",
        "Botha",
        "Mahlangu",
        "Ndlovu",
    ]
    region_labels = (
        ["Gauteng"] * 8
        + ["Western Cape"] * 4
        + ["KwaZulu-Natal"] * 3
        + ["Eastern Cape", "Limpopo", "Free State", "Mpumalanga", "GP", "gauteng", "W. Cape", ""]
    )
    records = []
    for i in range(rows):
        day = rng.randint(1, 28)
        month = rng.choice(["01", "02", "03"])
        style = rng.random()
        if style < 0.06:
            enrolled = ""
        elif style < 0.4:
            enrolled = f"2026/{month}/{day:02d}"
        elif style < 0.7:
            enrolled = f"{day:02d}-{month}-2026"
        else:
            enrolled = f"{['January', 'February', 'March'][int(month) - 1]} {day} 2026"
        course = rng.choice(
            ["Data literacy"] * 4 + ["SQL basics"] * 2 + ["Python for analysts"] * 3 + ["Stewardship 101"]
        )
        fee_base = {
            "Data literacy": 1250,
            "SQL basics": 980,
            "Python for analysts": 1400,
            "Stewardship 101": 1020,
        }[course]
        fee_style = rng.random()
        if fee_style < 0.04:
            fee = rng.choice(["N/A", "", "-"])
        elif fee_style < 0.3:
            fee = f"R {fee_base:,}.00".replace(",", " ")
        elif fee_style < 0.55:
            fee = f"R{fee_base:,}.00"
        elif fee_style < 0.7:
            fee = f"{fee_base:,},00".replace(",", " ", 1)
        else:
            fee = str(fee_base)
        if i % 211 == 0:
            fee = "98000"
        # South African ID numbers that pass the Luhn check, generated from a birth date.
        birth = date(1975 + rng.randint(0, 30), rng.randint(1, 12), rng.randint(1, 28))
        body = f"{birth:%y%m%d}{rng.randint(0, 9999):04d}08"
        digits = [int(c) for c in body]
        total = sum(
            d if idx % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2) for idx, d in enumerate(digits)
        )
        id_number = body + str((10 - total % 10) % 10)
        records.append(
            {
                "learner_id": f"LR-{100000 + i}",
                "full_name": f"{rng.choice(first)} {rng.choice(last)}",
                "id_number": id_number if rng.random() > 0.03 else "",
                "enrolled_on": enrolled,
                "course": course,
                "region": rng.choice(region_labels),
                "fee_paid": fee,
            }
        )
    frame = pd.DataFrame(records)
    duplicates = frame.sample(n=max(1, rows // 150), random_state=7)
    return pd.concat([frame, duplicates], ignore_index=True).astype(object)


def tickets_frame(rows: int = 1240) -> pd.DataFrame:
    rng = random.Random(8821)
    categories = [
        "Access",
        "Access",
        "Billing",
        "Course content",
        "Certificates",
        "Platform",
        "Access ",
        "access",
        "",
    ]
    records = []
    for i in range(rows):
        opened = f"2026-{rng.randint(1, 3):02d}-{rng.randint(1, 28):02d} {rng.randint(7, 18):02d}:{rng.randint(0, 59):02d}"
        still_open = rng.random() < 0.11
        records.append(
            {
                "ticket_id": f"TCK-{8000 + i}",
                "opened_at": opened,
                "category": rng.choice(categories),
                "resolution_hours": "" if still_open else f"{rng.lognormvariate(1.6, 0.8):.1f}",
                "channel": rng.choice(["email", "phone", "chat"]),
            }
        )
    return pd.DataFrame(records).astype(object)


# ------------------------------------------------------------------ seeding


async def _members(session: AsyncSession) -> dict[str, User]:
    out: dict[str, User] = {}
    for key, name, role, team in MEMBERS:
        email = f"{key}@dataforge.local"
        user = (await session.execute(select(User).where(User.email == email))).scalar_one_or_none()
        if user is None:
            user = User(
                id=key,
                email=email,
                name=name,
                role=role,
                team=team,
                password_hash=hash_password(settings.DEMO_PASSWORD),
                email_verified_at=utcnow(),
            )
            session.add(user)
        out[key] = user
    await session.commit()
    return out


async def _metrics(session: AsyncSession, owner: User) -> None:
    definitions = [
        (
            "active-learners",
            "active_learners",
            "Active learners",
            "count",
            "additive",
            "active_learners",
            None,
            None,
            "Learners with at least one session in the last 30 days. Used by the assistant, dashboards and monitors.",
            ["active", "learners active"],
            "region / month",
        ),
        (
            "completion-rate",
            "completion_rate",
            "Completion rate",
            "rate",
            "ratio",
            None,
            "completions",
            "enrolments",
            "Share of enrolments that produced a completion in the same period.",
            ["completion", "completing", "pass rate"],
            "course / month",
        ),
        (
            "average-fee",
            "avg_fee",
            "Average fee paid",
            "money",
            "ratio",
            None,
            "revenue",
            "enrolments",
            "Revenue divided by enrolments. Refunds are excluded.",
            ["average fee", "fee", "fees", "price"],
            "enrolment",
        ),
        (
            "enrolments",
            "enrolments",
            "Enrolments",
            "count",
            "additive",
            "enrolments",
            None,
            None,
            "Number of course enrolments.",
            ["enrolment", "enrollments", "sign-ups"],
            "course / month",
        ),
        (
            "completions",
            "completions",
            "Completions",
            "count",
            "additive",
            "completions",
            None,
            None,
            "Number of enrolments that ended in a completion.",
            ["completed"],
            "course / month",
        ),
        (
            "revenue",
            "revenue",
            "Revenue",
            "money",
            "additive",
            "revenue",
            None,
            None,
            "Approved fees paid, in rand.",
            ["income", "fees paid", "sales"],
            "region / month",
        ),
        (
            "sessions-per-learner",
            "sessions_per_learner",
            "Sessions per learner",
            "decimal",
            "ratio",
            None,
            "sessions",
            "enrolments",
            "Learning sessions logged, divided by enrolments.",
            ["sessions", "engagement"],
            "course / month",
        ),
    ]
    for (
        metric_id,
        key,
        name,
        unit,
        kind,
        column,
        numerator,
        denominator,
        description,
        synonyms,
        grain,
    ) in definitions:
        if await session.get(MetricDefinition, metric_id) is None:
            session.add(
                MetricDefinition(
                    id=metric_id,
                    key=key,
                    name=name,
                    unit=unit,
                    kind=kind,
                    column=column,
                    numerator=numerator,
                    denominator=denominator,
                    description=description,
                    synonyms=synonyms,
                    grain=grain,
                    status="published",
                    owner_id=owner.id,
                    owner_label="Data Stewardship",
                    published_by=owner.id,
                    published_at=utcnow(),
                )
            )
    if await session.get(MetricDefinition, "open-issues") is None:
        session.add(
            MetricDefinition(
                id="open-issues",
                key="open_issues",
                name="Open data issues",
                unit="count",
                kind="count",
                description="Quality flags still open on datasets the member can see.",
                owner_id=owner.id,
                owner_label=owner.name,
                status="draft",
                grain="dataset",
            )
        )
    await session.commit()


async def _clean_and_approve(
    session: AsyncSession,
    owner: Actor,
    steward: Actor,
    dataset_id: str,
    steps: list[tuple[str, dict[str, Any] | None]],
) -> None:
    for method_id, params in steps:
        await cleaning.add_step(session, owner, dataset_id, method_id, params, "accepted")
    await cleaning.run(session, owner, dataset_id)
    await datasets.approve(session, steward, dataset_id, "Quality profile reviewed")


async def seed_demo(session: AsyncSession) -> dict[str, Any]:
    """Create the demo organisation. Idempotent: existing records are left as they are."""
    users = await _members(session)
    actors = {k: Actor.from_user(u) for k, u in users.items()}
    await _metrics(session, users["naledi"])

    if await session.get(Connector, "wh-analytics") is None:
        session.add_all(
            [
                Connector(
                    id="pg-servicedesk",
                    name="Service desk",
                    kind="postgres",
                    status="connected",
                    owner_id="naledi",
                    owner_label="Naledi Khumalo",
                    config={
                        "host": "servicedesk.internal",
                        "port": 5432,
                        "database": "servicedesk",
                        "schema": "public",
                        "user": "dataforge_reader",
                    },
                    tables=14,
                    source_path="PostgreSQL · servicedesk.tickets",
                    approved_by="naledi",
                    last_sync_at=utcnow(),
                ),
                Connector(
                    id="wh-analytics",
                    name="Analytics warehouse",
                    kind="warehouse",
                    status="connected",
                    owner_id="naledi",
                    owner_label="Data Stewardship",
                    config={
                        "host": "warehouse.internal",
                        "port": 5439,
                        "database": "analytics",
                        "schema": "public",
                        "user": "dataforge_reader",
                    },
                    tables=38,
                    source_path="Warehouse · analytics.*",
                    approved_by="naledi",
                    last_sync_at=utcnow(),
                ),
                Connector(
                    id="sheet-fees",
                    name="Fee schedule",
                    kind="spreadsheet",
                    status="pending",
                    owner_id="sipho",
                    owner_label="Sipho Dlamini",
                    config={"filename": "fee_schedule.xlsx"},
                    tables=1,
                    source_path="Spreadsheet · fee_schedule.xlsx",
                ),
            ]
        )
        await session.commit()

    created: list[str] = []
    if await session.get(Dataset, "course-completions") is None:
        await datasets.ingest_frame(
            session,
            actors["sipho"],
            completions_frame(),
            name="Course completions",
            preferred_id="course-completions",
            table_name="course_completions",
            description="Completions by region, course and month, ready for analysis.",
            source="connector",
            source_name="Warehouse · analytics.completions",
            classification="internal-confidential",
            visibility="organisation",
            connector_id="wh-analytics",
        )
        await session.commit()
        await _clean_and_approve(
            session,
            actors["sipho"],
            actors["naledi"],
            "course-completions",
            [("type-coerce-sentinels", None)],
        )
        created.append("course-completions")

    if await session.get(Dataset, "regional-revenue") is None:
        await datasets.ingest_frame(
            session,
            actors["naledi"],
            revenue_frame(),
            name="Regional revenue",
            preferred_id="regional-revenue",
            table_name="revenue_by_region",
            description="Analytics-ready revenue by region and month, governed by the semantic layer.",
            source="connector",
            source_name="Warehouse · gold.revenue_by_region",
            classification="internal-general",
            visibility="organisation",
            connector_id="wh-analytics",
        )
        await session.commit()
        dataset = await session.get(Dataset, "regional-revenue")
        assert dataset is not None
        dataset.owner_label = "Data Stewardship"
        dataset.default_measure = "revenue"
        await session.commit()
        await _clean_and_approve(
            session, actors["naledi"], actors["naledi"], "regional-revenue", [("type-coerce-sentinels", None)]
        )
        await datasets.promote(session, actors["naledi"], "regional-revenue", None)
        created.append("regional-revenue")

    if await session.get(Dataset, "learner-enrolments") is None:
        frame = enrolments_frame()
        await datasets.ingest_frame(
            session,
            actors["thandi"],
            frame,
            name="Learner enrolments 2026",
            preferred_id="learner-enrolments",
            table_name="learner_enrolments",
            description="Raw enrolment export from the training portal, one row per enrolment.",
            source="upload",
            source_name="enrolments_q3.csv",
            classification="restricted-personal",
            visibility="team",
            landing_file=("enrolments_q3.csv", frame.to_csv(index=False).encode()),
        )
        await session.commit()
        created.append("learner-enrolments")

    if await session.get(Dataset, "support-tickets") is None:
        await datasets.ingest_frame(
            session,
            actors["naledi"],
            tickets_frame(),
            name="Support tickets",
            preferred_id="support-tickets",
            table_name="support_tickets",
            description="Ticket log from the service desk connector, awaiting steward approval.",
            source="connector",
            source_name="PostgreSQL · servicedesk.tickets",
            classification="internal-confidential",
            visibility="organisation",
            connector_id="pg-servicedesk",
        )
        await session.commit()
        created.append("support-tickets")

    await _dashboards(session)
    await _monitors(session)
    await _model(session, actors["zola"])
    await _notebook(session, actors["sipho"])
    if created:
        for question in (
            "What was the average fee by region last quarter?",
            "How has completion rate changed by course this year?",
            "Which learners are most likely to drop out next month?",
        ):
            await assistant.ask(session, actors["thandi"], question)
        await predictions.score(session, actors["thandi"], None, {"course": "SQL basics"})
        for monitor in (
            (await session.execute(select(Monitor).where(Monitor.status == "watching"))).scalars().all()
        ):
            await monitors.evaluate_monitor(session, monitor)
    return {"members": len(users), "datasetsCreated": created}


def _chart(
    dashboard_id: str,
    position: int,
    title: str,
    chart_type: str,
    metric: str,
    explanation: str,
    config: dict[str, Any],
) -> DashboardChart:
    return DashboardChart(
        dashboard_id=dashboard_id,
        position=position,
        title=title,
        type=chart_type,
        metric=metric,
        source="build",
        config=config,
        explanation=explanation,
    )


async def _dashboards(session: AsyncSession) -> None:
    if await session.get(Dashboard, "exec-overview") is not None:
        return
    session.add_all(
        [
            Dashboard(
                id="exec-overview",
                name="Executive overview",
                owner_id="naledi",
                owner_label="Data Stewardship",
                visibility="organisation",
                description="Governed view of active learners, completion and regional revenue. Shared read-only.",
                embed_enabled=True,
            ),
            Dashboard(
                id="enrolment-health",
                name="Enrolment health",
                owner_id="thandi",
                owner_label="Thandi Mokoena",
                visibility="team",
                team="Learning operations",
                description="Volume and completion of enrolments for the learning operations team.",
            ),
            Dashboard(
                id="my-scratch",
                name="My scratch board",
                owner_id="thandi",
                owner_label="Thandi Mokoena",
                visibility="private",
                description="Private working charts. Not visible to anyone else.",
            ),
        ]
    )
    await session.flush()
    session.add_all(
        [
            _chart(
                "exec-overview",
                0,
                "Active learners",
                "line",
                "Active learners",
                "Learners with at least one session in the last 30 days, from the semantic layer.",
                {"datasetId": "regional-revenue", "method": "trend", "measure": "active_learners"},
            ),
            _chart(
                "exec-overview",
                1,
                "Revenue by region",
                "bar",
                "Revenue",
                "Sum of approved fees, grouped by region, with row-level security applied.",
                {
                    "datasetId": "regional-revenue",
                    "method": "compare",
                    "measure": "revenue",
                    "dimension": "region",
                },
            ),
            _chart(
                "exec-overview",
                2,
                "Completion rate",
                "area",
                "Completion rate",
                "Completions divided by enrolments in the period.",
                {"datasetId": "course-completions", "method": "trend", "measure": "completion_rate"},
            ),
            _chart(
                "enrolment-health",
                0,
                "Enrolments by region",
                "bar",
                "Enrolments",
                "Enrolments grouped by region.",
                {
                    "datasetId": "course-completions",
                    "method": "compare",
                    "measure": "enrolments",
                    "dimension": "region",
                },
            ),
            _chart(
                "enrolment-health",
                1,
                "Enrolments by course",
                "bar",
                "Enrolments",
                "Enrolments grouped by course.",
                {
                    "datasetId": "course-completions",
                    "method": "compare",
                    "measure": "enrolments",
                    "dimension": "course",
                },
            ),
            _chart(
                "my-scratch",
                0,
                "Sessions per learner by course",
                "bar",
                "Sessions per learner",
                "A private cut of engagement by course.",
                {
                    "datasetId": "course-completions",
                    "method": "compare",
                    "measure": "sessions_per_learner",
                    "dimension": "course",
                },
            ),
        ]
    )
    await session.commit()


async def _monitors(session: AsyncSession) -> None:
    if (await session.execute(select(Monitor).limit(1))).first() is not None:
        return
    session.add_all(
        [
            Monitor(
                name="Free State revenue band",
                dataset_id="regional-revenue",
                measure="revenue",
                filters={"region": "Free State"},
                rule={"kind": "change", "threshold": 15, "window": 6},
                cadence="daily",
                owner_id="lerato",
                owner_label="Lerato Naidoo",
            ),
            Monitor(
                name="SQL basics completion",
                dataset_id="course-completions",
                measure="completion_rate",
                filters={"course": "SQL basics"},
                rule={"kind": "above", "threshold": 0.7, "window": 6},
                cadence="weekly",
                owner_id="thandi",
                owner_label="Thandi Mokoena",
            ),
            Monitor(
                name="Gauteng enrolments band",
                dataset_id="course-completions",
                measure="enrolments",
                filters={"region": "Gauteng"},
                rule={"kind": "band", "threshold": 2, "window": 5},
                cadence="daily",
                owner_id="naledi",
                owner_label="Naledi Khumalo",
                status="paused",
            ),
        ]
    )
    await session.commit()


async def _model(session: AsyncSession, curator: Actor) -> None:
    existing = (
        await session.execute(select(Dataset).where(Dataset.id == "course-completions"))
    ).scalar_one_or_none()
    if existing is None:
        return
    models = await predictions.list_models(session, curator)
    if any(m["name"] == "completion-heuristic" for m in models):
        return
    await predictions.create_version(
        session,
        curator,
        {
            "name": "completion-heuristic",
            "version": "1.4",
            "outcome": "Likelihood to complete within 60 days",
            "description": "A documented weighted heuristic. Each factor compares the cohort with every region-course group.",
            "datasetId": "course-completions",
            "factors": [
                {
                    "key": "recent_completion",
                    "name": "Recent completion rate",
                    "weight": 0.32,
                    "direction": "up",
                    "measure": "completion_rate",
                    "window": "recent",
                    "why": "Cohorts that have been completing recently keep completing.",
                },
                {
                    "key": "engagement",
                    "name": "Sessions per learner",
                    "weight": 0.27,
                    "direction": "up",
                    "measure": "sessions_per_learner",
                    "window": "recent",
                    "why": "Learners who log more sessions complete more often.",
                },
                {
                    "key": "completion_trend",
                    "name": "Completion trend",
                    "weight": 0.21,
                    "direction": "up",
                    "measure": "completion_rate",
                    "window": "trend",
                    "why": "An improving cohort is likelier to keep improving.",
                },
                {
                    "key": "cohort_size",
                    "name": "Cohort size",
                    "weight": 0.12,
                    "direction": "down",
                    "measure": "enrolments",
                    "window": "recent",
                    "why": "Larger cohorts get less facilitator time per learner.",
                },
                {
                    "key": "engagement_trend",
                    "name": "Engagement trend",
                    "weight": 0.08,
                    "direction": "up",
                    "measure": "sessions_per_learner",
                    "window": "trend",
                    "why": "Rising engagement precedes completion.",
                },
            ],
        },
    )


async def _notebook(session: AsyncSession, author: Actor) -> None:
    notebooks = await workbench.list_notebooks(session, author)
    if any(n["id"] == "nb-enrolment" for n in notebooks):
        return
    created = await workbench.create_notebook(
        session,
        author,
        "Enrolment quality notes",
        [
            {
                "kind": "markdown",
                "source": "A short notebook on enrolment quality. Each SQL cell runs in the sandbox against permitted tables only.",
            },
            {
                "kind": "sql",
                "source": (
                    "SELECT region,\n"
                    "       ROUND(SUM(revenue)::numeric / NULLIF(SUM(enrolments), 0), 2) AS avg_fee\n"
                    "FROM   gold.revenue_by_region\n"
                    "WHERE  month >= DATE '2026-01-01'\n"
                    "GROUP  BY region\n"
                    "ORDER  BY avg_fee DESC;"
                ),
            },
            {
                "kind": "markdown",
                "source": "Free State is the outlier. Next: open the alert in Monitors for a root-cause read.",
            },
        ],
        notebook_id="nb-enrolment",
        visibility="organisation",
    )
    sql_cell = next((c for c in created["cells"] if c["kind"] == "sql"), None)
    if sql_cell is not None:
        await workbench.run_cell(session, author, created["id"], sql_cell["id"])
