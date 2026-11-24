"""Provider/model provenance on query_runs and agent_runs.

Adds four nullable ``sa.Text()`` columns so the audit trail records what
actually answered a query, not just the ``fast``/``powerful`` tier the user
picked. ``query_runs.answer_model`` is a tier; a provider set that has grown
to several vendors means the same tier can mean a different model week to
week, so the tier alone no longer identifies what ran.

Deliberately unlike ``0003``: there is no backfill and no ``NOT NULL``.
``0003`` could backfill ``'powerful'`` because that was provably the prior
behaviour for every existing row. Here the honest value for a pre-migration
row is *unknown*: filling in ``openai``/``gpt-4o`` would fabricate an audit
record for anyone who ran with a different ``LLM_TEXT_MODEL_STRONG``, or in
fake mode. NULL means "recorded before multi-provider support", and the UI
falls back to the tier badge.

``sa.Text()`` rather than an enum: the provider set is open by design, and
an enum column would mean a migration every time a provider is added.

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-17
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("query_runs", sa.Column("resolved_provider", sa.Text(), nullable=True))
    op.add_column("query_runs", sa.Column("resolved_model_name", sa.Text(), nullable=True))
    op.add_column("agent_runs", sa.Column("llm_provider", sa.Text(), nullable=True))
    op.add_column("agent_runs", sa.Column("llm_model", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("agent_runs", "llm_model")
    op.drop_column("agent_runs", "llm_provider")
    op.drop_column("query_runs", "resolved_model_name")
    op.drop_column("query_runs", "resolved_provider")
