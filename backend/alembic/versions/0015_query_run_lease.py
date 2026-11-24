"""The lease of a running query run (RMD5-T01; RE2E-DEF-038, RE2E-DEF-041).

``query_runs`` gains two nullable columns: ``lease_expires_at``, when the run's lease ends unless
its job renews it, and ``owner_instance``, the process instance that claimed it. Existing rows keep
NULL in both, which the reconciler reads as "no lease" and judges by the run's deadline instead.

Additive. The downgrade drops exactly these two columns, so every row and every other column
survives it.

Revision ID: 0015
Revises: 0014
Create Date: 2026-10-27
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0015"
down_revision: str | None = "0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "query_runs", sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column("query_runs", sa.Column("owner_instance", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("query_runs", "owner_instance")
    op.drop_column("query_runs", "lease_expires_at")
