"""Trained models with review, scoring runs, anomaly alerts, answer traces and governed reports.

Every change is additive (new tables, or new columns that are nullable or have a default), so an API still
running the previous release keeps working against the upgraded schema.

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-19 10:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_JSON = postgresql.JSONB(astext_type=sa.Text())
_MODEL_JSON = (
    "target",
    "features",
    "protected",
    "metrics",
    "fairness",
    "card",
    "reference",
    "runtime",
    "drift",
)


def upgrade() -> None:
    # ---------------------------------------------------------------- model versions: trained models and review
    op.add_column(
        "model_versions",
        sa.Column("kind", sa.String(length=16), server_default="heuristic", nullable=False),
        schema="app",
    )
    op.add_column("model_versions", sa.Column("algorithm", sa.String(length=40), nullable=True), schema="app")
    for name in _MODEL_JSON:
        op.add_column("model_versions", sa.Column(name, _JSON, nullable=True), schema="app")
    op.add_column("model_versions", sa.Column("threshold", sa.Float(), nullable=True), schema="app")
    op.add_column(
        "model_versions",
        sa.Column("training_dataset_version_id", sa.String(length=32), nullable=True),
        schema="app",
    )
    op.add_column("model_versions", sa.Column("training_rows", sa.Integer(), nullable=True), schema="app")
    op.add_column(
        "model_versions", sa.Column("trained_by", sa.String(length=32), nullable=True), schema="app"
    )
    op.add_column(
        "model_versions", sa.Column("approved_by", sa.String(length=32), nullable=True), schema="app"
    )
    op.add_column(
        "model_versions", sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True), schema="app"
    )
    op.add_column("model_versions", sa.Column("decision_note", sa.Text(), nullable=True), schema="app")
    op.add_column("model_versions", sa.Column("artifact", sa.LargeBinary(), nullable=True), schema="app")
    op.add_column(
        "model_versions", sa.Column("artifact_sha256", sa.String(length=64), nullable=True), schema="app"
    )
    op.add_column(
        "model_versions",
        sa.Column("drift_checked_at", sa.DateTime(timezone=True), nullable=True),
        schema="app",
    )
    op.create_foreign_key(
        op.f("fk_model_versions_training_dataset_version_id_dataset_versions"),
        "model_versions",
        "dataset_versions",
        ["training_dataset_version_id"],
        ["id"],
        source_schema="app",
        referent_schema="app",
        ondelete="SET NULL",
    )
    for column in ("trained_by", "approved_by"):
        op.create_foreign_key(
            op.f(f"fk_model_versions_{column}_users"),
            "model_versions",
            "users",
            [column],
            ["id"],
            source_schema="app",
            referent_schema="app",
            ondelete="SET NULL",
        )

    # ---------------------------------------------------------------- scoring runs
    op.create_table(
        "scoring_runs",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("model_version_id", sa.String(length=32), nullable=False),
        sa.Column("dataset_id", sa.String(length=80), nullable=False),
        sa.Column("dataset_version", sa.Integer(), nullable=False),
        sa.Column("requested_by", sa.String(length=32), nullable=True),
        sa.Column("scores_table", sa.String(length=140), nullable=False),
        sa.Column("population", _JSON, nullable=True),
        sa.Column("rows_scored", sa.Integer(), server_default="0", nullable=False),
        sa.Column("flagged", sa.Integer(), server_default="0", nullable=False),
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("summary", _JSON, nullable=True),
        sa.Column("fairness", _JSON, nullable=True),
        sa.Column("drift", _JSON, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["app.model_versions.id"],
            name=op.f("fk_scoring_runs_model_version_id_model_versions"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["app.datasets.id"],
            name=op.f("fk_scoring_runs_dataset_id_datasets"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["requested_by"],
            ["app.users.id"],
            name=op.f("fk_scoring_runs_requested_by_users"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_scoring_runs")),
        schema="app",
    )
    op.create_index(
        "ix_scoring_runs_model_created",
        "scoring_runs",
        ["model_version_id", "created_at"],
        unique=False,
        schema="app",
    )

    # ---------------------------------------------------------------- alerts raised by the anomaly scan
    op.add_column("alerts", sa.Column("dataset_id", sa.String(length=80), nullable=True), schema="app")
    op.add_column(
        "alerts",
        sa.Column("source", sa.String(length=16), server_default="monitor", nullable=False),
        schema="app",
    )
    op.add_column("alerts", sa.Column("series", _JSON, nullable=True), schema="app")
    op.add_column("alerts", sa.Column("detector", _JSON, nullable=True), schema="app")
    op.create_foreign_key(
        op.f("fk_alerts_dataset_id_datasets"),
        "alerts",
        "datasets",
        ["dataset_id"],
        ["id"],
        source_schema="app",
        referent_schema="app",
        ondelete="CASCADE",
    )
    op.create_index("ix_alerts_dataset_month", "alerts", ["dataset_id", "month"], unique=False, schema="app")

    # ---------------------------------------------------------------- how each answer was governed
    op.add_column("assistant_queries", sa.Column("trace", _JSON, nullable=True), schema="app")
    op.add_column("assistant_queries", sa.Column("chart", _JSON, nullable=True), schema="app")

    # ---------------------------------------------------------------- governed reports
    op.create_table(
        "reports",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("template_id", sa.String(length=80), nullable=False),
        sa.Column("title", sa.String(length=300), nullable=False),
        sa.Column("period", _JSON, nullable=True),
        sa.Column("datasets", _JSON, nullable=True),
        sa.Column("figures", _JSON, nullable=True),
        sa.Column("narrative", _JSON, nullable=True),
        sa.Column("narrative_source", sa.String(length=16), server_default="template", nullable=False),
        sa.Column("model", sa.String(length=80), nullable=True),
        sa.Column("created_by", sa.String(length=32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["created_by"],
            ["app.users.id"],
            name=op.f("fk_reports_created_by_users"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_reports")),
        schema="app",
    )
    op.create_index(op.f("ix_app_reports_created_by"), "reports", ["created_by"], unique=False, schema="app")


def downgrade() -> None:
    op.drop_index(op.f("ix_app_reports_created_by"), table_name="reports", schema="app")
    op.drop_table("reports", schema="app")

    op.drop_column("assistant_queries", "chart", schema="app")
    op.drop_column("assistant_queries", "trace", schema="app")

    op.drop_index("ix_alerts_dataset_month", table_name="alerts", schema="app")
    op.drop_constraint(op.f("fk_alerts_dataset_id_datasets"), "alerts", schema="app", type_="foreignkey")
    for name in ("detector", "series", "source", "dataset_id"):
        op.drop_column("alerts", name, schema="app")

    op.drop_index("ix_scoring_runs_model_created", table_name="scoring_runs", schema="app")
    op.drop_table("scoring_runs", schema="app")

    for column in ("approved_by", "trained_by"):
        op.drop_constraint(
            op.f(f"fk_model_versions_{column}_users"), "model_versions", schema="app", type_="foreignkey"
        )
    op.drop_constraint(
        op.f("fk_model_versions_training_dataset_version_id_dataset_versions"),
        "model_versions",
        schema="app",
        type_="foreignkey",
    )
    for name in (
        "drift_checked_at",
        "artifact_sha256",
        "artifact",
        "decision_note",
        "approved_at",
        "approved_by",
        "trained_by",
        "training_rows",
        "training_dataset_version_id",
        "threshold",
        *reversed(_MODEL_JSON),
        "algorithm",
        "kind",
    ):
        op.drop_column("model_versions", name, schema="app")
