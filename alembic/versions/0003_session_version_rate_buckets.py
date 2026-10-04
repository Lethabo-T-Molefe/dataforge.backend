"""Session version for access-token revocation, and durable auth rate-limit buckets.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-04 09:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("session_version", sa.Integer(), server_default="0", nullable=False),
        schema="app",
    )
    op.create_table(
        "rate_buckets",
        sa.Column("key", sa.String(length=200), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("count", sa.Integer(), server_default="0", nullable=False),
        sa.PrimaryKeyConstraint("key", name=op.f("pk_rate_buckets")),
        schema="app",
    )


def downgrade() -> None:
    op.drop_table("rate_buckets", schema="app")
    op.drop_column("users", "session_version", schema="app")
