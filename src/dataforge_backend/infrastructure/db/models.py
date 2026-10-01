"""Metadata tables in the `app` schema. Member data itself lives in the bronze, silver and gold schemas."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ...core.database import Base
from ...core.security import utcnow


def new_id() -> str:
    return uuid.uuid4().hex


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow, onupdate=utcnow
    )


# ------------------------------------------------------------------ identity


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String(320), unique=True)
    name: Mapped[str] = mapped_column(String(200))
    password_hash: Mapped[str | None] = mapped_column(Text)
    role: Mapped[str] = mapped_column(String(32), default="analyst")
    team: Mapped[str | None] = mapped_column(String(120))
    status: Mapped[str] = mapped_column(String(16), default="active")  # active | disabled
    email_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    mfa_secret_enc: Mapped[str | None] = mapped_column(Text)
    mfa_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    failed_logins: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_active_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    quota_overrides: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    erased_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    family_id: Mapped[str] = mapped_column(String(32), index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    replaced_by: Mapped[str | None] = mapped_column(String(32))
    user_agent: Mapped[str | None] = mapped_column(String(300))
    ip: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow
    )


class EmailToken(Base):
    __tablename__ = "email_tokens"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    purpose: Mapped[str] = mapped_column(String(16))  # verify | reset
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow
    )


class Usage(Base):
    """Fair-use counters per member per month."""

    __tablename__ = "usage"
    __table_args__ = (UniqueConstraint("user_id", "period"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    period: Mapped[str] = mapped_column(String(7))  # YYYY-MM
    compute_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    ai_queries: Mapped[int] = mapped_column(Integer, default=0)


# ------------------------------------------------------------------ data


class Connector(TimestampMixin, Base):
    __tablename__ = "connectors"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    kind: Mapped[str] = mapped_column(String(16))  # postgres | warehouse | spreadsheet
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending | connected | error
    owner_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    owner_label: Mapped[str | None] = mapped_column(String(200))
    config: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict
    )  # host, port, database, schema (no secrets)
    secret_enc: Mapped[str | None] = mapped_column(Text)
    tables: Mapped[int] = mapped_column(Integer, default=0)
    source_path: Mapped[str] = mapped_column(String(300), default="")
    last_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
    approved_by: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))


class Dataset(TimestampMixin, Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    owner_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), index=True)
    owner_label: Mapped[str | None] = mapped_column(String(200))
    source: Mapped[str] = mapped_column(String(16))  # upload | connector
    source_name: Mapped[str] = mapped_column(String(300))
    connector_id: Mapped[str | None] = mapped_column(ForeignKey("connectors.id", ondelete="SET NULL"))
    layer: Mapped[str] = mapped_column(String(8), default="bronze")
    classification: Mapped[str] = mapped_column(String(32), default="internal-confidential")
    status: Mapped[str] = mapped_column(String(16), default="pending")
    visibility: Mapped[str] = mapped_column(String(16), default="private")  # private | team | organisation
    team: Mapped[str | None] = mapped_column(String(120))
    table_name: Mapped[str] = mapped_column(String(63), unique=True)
    current_version: Mapped[int] = mapped_column(Integer, default=1)
    rows: Mapped[int] = mapped_column(BigInteger, default=0)
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    quality: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    profile: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    columns: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    pii_columns: Mapped[dict[str, str]] = mapped_column(JSONB, default=dict)
    fields: Mapped[dict[str, Any] | None] = mapped_column(JSONB)  # analysis fields cache
    default_measure: Mapped[str | None] = mapped_column(String(63))
    decided_by: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    decision_reason: Mapped[str | None] = mapped_column(Text)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    erased_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    insights: Mapped[list["Insight"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")

    def qualified_table(self) -> str | None:
        """The table analyses read: gold if promoted, silver once cleaned."""
        if self.layer == "gold":
            return f"gold.{self.table_name}"
        if self.layer == "silver":
            return f"silver.{self.table_name}"
        return None


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    __table_args__ = (UniqueConstraint("dataset_id", "version", "layer"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    layer: Mapped[str] = mapped_column(String(8))
    table: Mapped[str] = mapped_column(String(140))
    rows: Mapped[int] = mapped_column(BigInteger, default=0)
    column_types: Mapped[dict[str, str]] = mapped_column(JSONB, default=dict)
    source_file: Mapped[str | None] = mapped_column(String(500))
    checksum: Mapped[str | None] = mapped_column(String(64))
    recipe_id: Mapped[str | None] = mapped_column(String(32))
    created_by: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow
    )


class Share(Base):
    """A dataset, dashboard, recipe or notebook shared with a named member."""

    __tablename__ = "shares"
    __table_args__ = (UniqueConstraint("resource_type", "resource_id", "user_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    resource_type: Mapped[str] = mapped_column(String(16))
    resource_id: Mapped[str] = mapped_column(String(80), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_by: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow
    )


class Insight(Base):
    __tablename__ = "insights"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(16))
    title: Mapped[str] = mapped_column(String(300))
    body: Mapped[str] = mapped_column(Text)
    suggestion: Mapped[str] = mapped_column(String(300))
    action: Mapped[str | None] = mapped_column(String(32))
    target: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(16), default="open")  # open | accepted | dismissed
    decided_by: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow
    )

    dataset: Mapped[Dataset] = relationship(back_populates="insights")


class LineageEdge(Base):
    __tablename__ = "lineage_edges"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    dataset_id: Mapped[str | None] = mapped_column(ForeignKey("datasets.id", ondelete="SET NULL"), index=True)
    source: Mapped[str] = mapped_column(String(300))
    target: Mapped[str] = mapped_column(String(300))
    transform: Mapped[str] = mapped_column(String(32))  # upload | connector | clean | promote | erase
    detail: Mapped[str] = mapped_column(Text, default="")
    recipe_id: Mapped[str | None] = mapped_column(String(32))
    actor_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow
    )


class RowPolicy(Base):
    """Row-level security: the rows of a dataset a role may read."""

    __tablename__ = "row_policies"
    __table_args__ = (UniqueConstraint("dataset_id", "role"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(32))
    predicate: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow
    )


# ------------------------------------------------------------------ cleaning


class CleaningRecipe(TimestampMixin, Base):
    __tablename__ = "cleaning_recipes"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    name: Mapped[str] = mapped_column(String(200), default="")
    status: Mapped[str] = mapped_column(String(16), default="draft")  # draft | applied
    visibility: Mapped[str] = mapped_column(String(16), default="private")
    created_by: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    output_table: Mapped[str | None] = mapped_column(String(140))
    output_rows: Mapped[int | None] = mapped_column(BigInteger)
    output_checksum: Mapped[str | None] = mapped_column(String(64))

    steps: Mapped[list["CleaningStep"]] = relationship(
        back_populates="recipe", cascade="all, delete-orphan", order_by="CleaningStep.position"
    )


class CleaningStep(Base):
    __tablename__ = "cleaning_steps"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    recipe_id: Mapped[str] = mapped_column(ForeignKey("cleaning_recipes.id", ondelete="CASCADE"), index=True)
    position: Mapped[int] = mapped_column(Integer)
    method_id: Mapped[str] = mapped_column(String(64))
    chapter_id: Mapped[str] = mapped_column(String(32))
    params: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="accepted")  # accepted | skipped
    summary: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow
    )

    recipe: Mapped[CleaningRecipe] = relationship(back_populates="steps")


# ------------------------------------------------------------------ semantic layer and analysis


class MetricDefinition(TimestampMixin, Base):
    __tablename__ = "metric_definitions"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    key: Mapped[str] = mapped_column(String(63), unique=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    owner_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    owner_label: Mapped[str] = mapped_column(String(200), default="")
    unit: Mapped[str] = mapped_column(String(16))
    kind: Mapped[str] = mapped_column(String(16))
    column: Mapped[str | None] = mapped_column(String(63))
    numerator: Mapped[str | None] = mapped_column(String(63))
    denominator: Mapped[str | None] = mapped_column(String(63))
    synonyms: Mapped[list[str]] = mapped_column(JSONB, default=list)
    grain: Mapped[str] = mapped_column(String(120), default="")
    status: Mapped[str] = mapped_column(String(16), default="draft")  # draft | published
    version: Mapped[int] = mapped_column(Integer, default=1)
    published_by: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Exploration(TimestampMixin, Base):
    """A saved point-and-click analysis and the SQL it generates."""

    __tablename__ = "explorations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    config: Mapped[dict[str, Any]] = mapped_column(JSONB)
    sql: Mapped[str] = mapped_column(Text, default="")


class Dashboard(TimestampMixin, Base):
    __tablename__ = "dashboards"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    owner_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), index=True)
    owner_label: Mapped[str] = mapped_column(String(200), default="")
    visibility: Mapped[str] = mapped_column(String(16), default="private")
    team: Mapped[str | None] = mapped_column(String(120))
    embed_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    superset_uuid: Mapped[str | None] = mapped_column(String(64))  # the linked Superset dashboard, if any

    charts: Mapped[list["DashboardChart"]] = relationship(
        back_populates="dashboard", cascade="all, delete-orphan", order_by="DashboardChart.position"
    )


class DashboardChart(Base):
    __tablename__ = "dashboard_charts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    dashboard_id: Mapped[str] = mapped_column(ForeignKey("dashboards.id", ondelete="CASCADE"), index=True)
    position: Mapped[int] = mapped_column(Integer, default=0)
    title: Mapped[str] = mapped_column(String(200))
    type: Mapped[str] = mapped_column(String(16), default="bar")  # bar | line | area
    metric: Mapped[str] = mapped_column(String(200), default="")
    source: Mapped[str] = mapped_column(String(16), default="build")  # build | sql
    config: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    sql: Mapped[str] = mapped_column(Text, default="")
    explanation: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow
    )

    dashboard: Mapped[Dashboard] = relationship(back_populates="charts")


# ------------------------------------------------------------------ monitors and predictions


class Monitor(TimestampMixin, Base):
    __tablename__ = "monitors"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(200))
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"))
    measure: Mapped[str] = mapped_column(String(63))
    filters: Mapped[dict[str, str]] = mapped_column(JSONB, default=dict)
    rule: Mapped[dict[str, Any]] = mapped_column(JSONB)
    cadence: Mapped[str] = mapped_column(String(16), default="daily")
    owner_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), index=True)
    owner_label: Mapped[str] = mapped_column(String(200), default="")
    status: Mapped[str] = mapped_column(String(16), default="watching")  # watching | paused
    notify_slack: Mapped[bool] = mapped_column(Boolean, default=False)
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_value: Mapped[float | None] = mapped_column(Float)
    last_month: Mapped[str | None] = mapped_column(String(7))


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    monitor_id: Mapped[str | None] = mapped_column(ForeignKey("monitors.id", ondelete="SET NULL"), index=True)
    metric: Mapped[str] = mapped_column(String(300))
    message: Mapped[str] = mapped_column(Text)
    severity: Mapped[str] = mapped_column(String(8))
    month: Mapped[str | None] = mapped_column(String(7))
    previous_month: Mapped[str | None] = mapped_column(String(7))
    value: Mapped[float | None] = mapped_column(Float)
    previous: Mapped[float | None] = mapped_column(Float)
    root_cause: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    explore: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    acknowledged_by: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("name", "version"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(120))
    version: Mapped[str] = mapped_column(String(32))
    outcome: Mapped[str] = mapped_column(String(300))
    description: Mapped[str] = mapped_column(Text, default="")
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"))
    factors: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(16), default="active")  # active | retired
    created_by: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    reviewed_by: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    model_version_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id", ondelete="RESTRICT"))
    requested_by: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), index=True)
    subject: Mapped[str] = mapped_column(String(300))
    cohort: Mapped[dict[str, str]] = mapped_column(JSONB, default=dict)
    score: Mapped[int] = mapped_column(Integer)
    band: Mapped[str] = mapped_column(String(32))
    factors: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow
    )


# ------------------------------------------------------------------ workbench and assistant


class Notebook(TimestampMixin, Base):
    __tablename__ = "notebooks"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    owner_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), index=True)
    owner_label: Mapped[str] = mapped_column(String(200), default="")
    visibility: Mapped[str] = mapped_column(String(16), default="private")
    jupyter_path: Mapped[str | None] = mapped_column(
        String(400)
    )  # the .ipynb in the Jupyter workspace, if linked

    cells: Mapped[list["NotebookCell"]] = relationship(
        back_populates="notebook", cascade="all, delete-orphan", order_by="NotebookCell.position"
    )


class NotebookCell(Base):
    __tablename__ = "notebook_cells"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    notebook_id: Mapped[str] = mapped_column(ForeignKey("notebooks.id", ondelete="CASCADE"), index=True)
    position: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(16))  # markdown | sql
    source: Mapped[str] = mapped_column(Text, default="")
    output: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    revision: Mapped[int] = mapped_column(Integer, default=1)  # optimistic concurrency for shared editing
    updated_by: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow, onupdate=utcnow
    )

    notebook: Mapped[Notebook] = relationship(back_populates="cells")


class AssistantQuery(Base):
    __tablename__ = "assistant_queries"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), index=True)
    question: Mapped[str] = mapped_column(Text)
    plan: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    answer: Mapped[str] = mapped_column(Text)
    explanation: Mapped[str] = mapped_column(Text)
    sql: Mapped[str] = mapped_column(Text, default="")
    refused: Mapped[bool] = mapped_column(Boolean, default=False)
    refusal_reason: Mapped[str | None] = mapped_column(String(32))
    grounded_in: Mapped[list[str]] = mapped_column(JSONB, default=list)
    result_table: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    build: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    planner: Mapped[str] = mapped_column(String(16), default="rules")
    channel: Mapped[str] = mapped_column(String(16), default="web")  # web | slack
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow
    )


class SandboxSession(Base):
    __tablename__ = "sandbox_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), index=True)
    kind: Mapped[str] = mapped_column(String(16))  # sql | analysis | cleaning | prediction | monitor
    status: Mapped[str] = mapped_column(
        String(16), default="running"
    )  # running | ok | refused | error | timeout
    caps: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    statement: Mapped[str] = mapped_column(Text, default="")
    rows_returned: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AuditEvent(Base):
    """Immutable: a database trigger rejects every UPDATE and DELETE."""

    __tablename__ = "audit_events"
    __table_args__ = (Index("ix_audit_events_occurred_at", "occurred_at"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow
    )
    actor_id: Mapped[str | None] = mapped_column(String(32), index=True)
    actor_label: Mapped[str] = mapped_column(String(200))
    action: Mapped[str] = mapped_column(String(80))
    resource_type: Mapped[str] = mapped_column(String(32))
    resource_id: Mapped[str | None] = mapped_column(String(120))
    resource_label: Mapped[str] = mapped_column(String(300), default="")
    detail: Mapped[str] = mapped_column(Text, default="")
    request_id: Mapped[str | None] = mapped_column(String(64))
    ip: Mapped[str | None] = mapped_column(String(64))
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class SlackLink(Base):
    __tablename__ = "slack_links"
    __table_args__ = (UniqueConstraint("slack_team_id", "slack_user_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    slack_team_id: Mapped[str | None] = mapped_column(String(32))
    slack_user_id: Mapped[str | None] = mapped_column(String(32))
    link_code_hash: Mapped[str | None] = mapped_column(String(64))
    link_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), default=utcnow
    )
