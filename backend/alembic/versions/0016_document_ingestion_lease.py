"""The lease of a document's ingestion (RMD5-T02; RMD-DEF-004).

``documents`` gains one nullable column, ``lease_expires_at``: when the ingestion holding the
document ends its claim unless its job renews it. Existing rows keep NULL, which the reconciler
reads as "no lease" and judges by the document's last update instead.

Additive. The downgrade drops exactly this column, so every row and every other column survives
it.

Revision ID: 0016
Revises: 0015
Create Date: 2026-10-27
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0016"
down_revision: str | None = "0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "documents", sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("documents", "lease_expires_at")
