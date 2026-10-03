"""Notifications for members, scheduled connector refresh, and email for alerts.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-03 09:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "notifications",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("user_id", sa.String(length=32), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("title", sa.String(length=300), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("link", sa.String(length=300), nullable=True),
        sa.Column("read_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"], ["app.users.id"], name=op.f("fk_notifications_user_id_users"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_notifications")),
        schema="app",
    )
    op.create_index(
        "ix_notifications_user_created",
        "notifications",
        ["user_id", "created_at"],
        unique=False,
        schema="app",
    )
    # hourly | daily | weekly; empty means the connector is only read when someone asks.
    op.add_column(
        "connectors", sa.Column("refresh_cadence", sa.String(length=16), nullable=True), schema="app"
    )
    op.add_column(
        "monitors",
        sa.Column("notify_email", sa.Boolean(), server_default=sa.true(), nullable=False),
        schema="app",
    )


def downgrade() -> None:
    op.drop_column("monitors", "notify_email", schema="app")
    op.drop_column("connectors", "refresh_cadence", schema="app")
    op.drop_index("ix_notifications_user_created", table_name="notifications", schema="app")
    op.drop_table("notifications", schema="app")
