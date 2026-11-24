"""Per-query answer model selection on query_runs.

Adds ``query_runs.answer_model`` so a user can choose which configured text
model synthesises the answer. Added nullable and backfilled to ``powerful``
(the previous hard-coded behaviour) before being made NOT NULL, so existing
rows and in-flight queries are unaffected.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-17
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_DEFAULT = "powerful"


def _enum(*values: str, name: str) -> sa.Enum:
    return sa.Enum(*values, name=name, native_enum=False, length=30)


def upgrade() -> None:
    op.add_column(
        "query_runs",
        sa.Column(
            "answer_model",
            _enum("fast", "powerful", name="answer_model"),
            nullable=True,
        ),
    )
    op.execute(sa.text(f"UPDATE query_runs SET answer_model = '{_DEFAULT}'"))  # noqa: S608
    op.alter_column("query_runs", "answer_model", nullable=False)


def downgrade() -> None:
    op.drop_column("query_runs", "answer_model")
