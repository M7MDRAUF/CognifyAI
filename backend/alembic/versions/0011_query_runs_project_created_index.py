"""The composite index behind bounded run pages (SE-003, CT-11, SD-02).

A page of a project's runs is keyset-paged in ``(created_at DESC, id DESC)`` order. With only
``ix_query_runs_project_id`` the database fetches the project's whole history and sorts it for
every page: 2,003 buffers per page at 2,000 runs, at any depth. ``(project_id, created_at, id)``
lets it walk the index backwards from the cursor and stop after the page: 24 buffers, the same at
page 1 and page 51 (small-enhancements audit, section 6.3).

Additive and reversible. No row is read, rewritten or deleted, so the downgrade loses nothing.
``ix_query_runs_project_id`` stays: the admission count and the other per-project queries keep
their plans. ``CREATE INDEX`` runs inside Alembic's transaction and briefly blocks writes to
``query_runs``; ``CONCURRENTLY`` is not allowed there, and at this product's table sizes the lock
lasts well under a second.

Revision ID: 0011
Revises: 0010
Create Date: 2026-09-24
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_query_runs_project_id_created_at_id",
        "query_runs",
        ["project_id", "created_at", "id"],
    )


def downgrade() -> None:
    op.drop_index("ix_query_runs_project_id_created_at_id", table_name="query_runs")
