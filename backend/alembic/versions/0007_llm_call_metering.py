"""Per-attempt provider metering (M-7, L-12, C-12).

Three defects, one root cause: nothing counted the billed unit.

**M-7 — the reranker was never metered.** ``RerankingService._score`` calls
``generate_json`` and discards ``result.usage`` entirely, so a call made once per retrieval round
-- twice on any run with an expansion -- contributed nothing to ``token_usage_estimate``.

**L-12 — a failed run recorded no spend at all.** ``token_usage_estimate`` had exactly one
writer, ``Orchestrator._finish``, which runs only on the success path. Every failed, cancelled and
deadline-stopped run recorded NULL, which read as "free".

**C-12 — provider requests were understated threefold.** Three logical completions per agent,
each retried up to three times: 7 to 60 HTTP requests per run against a documented figure of
seven.

``token_usage_estimate`` is a single scalar on ``query_runs`` and cannot express any of this, so
this table records **one row per provider HTTP attempt** -- the billed unit. ``rung`` and
``attempt`` together identify which of the up-to-nine requests an agent step made: ``rung=2,
attempt=3`` is the ninth and most expensive.

``outcome`` distinguishes ``charged_failure`` from ``failure``: a 2xx the vendor billed whose body
carried no usable answer is an invoice, and collapsing it into "failure" is how that spend came to
be discarded (N-002).

**Nullable token columns, deliberately.** ``NULL`` means the vendor reported no usage; ``0`` means
it reported zero. A metering row that cannot tell "unknown" from "free" cannot be reconciled
against a bill. Same rule for ``cost_usd``, which is ``NULL`` for a model absent from the rate
table and is **never** 0.0 (ADR-004).

**No backfill.** Historical attempts were never recorded and cannot be recovered. An empty table
for old runs is the truth; synthesising rows from ``token_usage_estimate`` would fabricate a
per-attempt breakdown that never existed.

``ondelete="CASCADE"`` matches every other child of ``query_runs``: deleting a run must not leave
orphaned metering rows behind.

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PGUUID

from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "llm_calls",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True),
        sa.Column(
            "query_run_id",
            PGUUID(as_uuid=True),
            sa.ForeignKey("query_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # The component that caused the attempt. `Text` and not the agent enum: the reranker and
        # the provider adapter are call sites and are not members of it (ADR-001).
        sa.Column("component", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("rung", sa.Integer(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.Column("failure_kind", sa.Text(), nullable=True),
        # NULL = the vendor reported no usage. 0 = it reported zero. Not the same thing.
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("backoff_ms", sa.Integer(), nullable=False, server_default="0"),
        # NULL = unpriced. Never 0.0 for a missing rate (ADR-004).
        sa.Column("cost_usd", sa.Numeric(12, 6), nullable=True),
        sa.Column("pricing_status", sa.Text(), nullable=False, server_default="unpriced"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    # Every read of this table is "the attempts for one run, in order", which is what the
    # budget ledger and the per-run report both do.
    op.create_index(
        "ix_llm_calls_query_run_id_created_at", "llm_calls", ["query_run_id", "created_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_llm_calls_query_run_id_created_at", table_name="llm_calls")
    op.drop_table("llm_calls")
