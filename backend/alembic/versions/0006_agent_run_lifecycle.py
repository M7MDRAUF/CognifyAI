"""AgentRun lifecycle and failure detail (M-6, M-9, L-6).

Three problems in one write path, so one revision.

**M-6 — the row was written once, after the step returned.** ``AgentRunStatus`` declares
``PENDING`` and ``RUNNING`` and neither was ever written: ``_record_agent_run`` inserted a
``COMPLETED`` row with ``started_at`` and ``completed_at`` both set to ``utc_now()`` at the same
instant. So a step in flight was invisible, and ``latency_ms`` was the only record that any time
had passed at all. ``attempt_count`` is added because a step that spent three repair rungs and a
step that succeeded first time were indistinguishable.

**M-9 / L-6 — a failure recorded nothing usable.** ``_step`` caught only ``CognifyError``, so any
other exception left no row at all, and the rows it did write carried ``output_json = NULL`` and
the fixed string ``"Agent step failed."``. The payload that failed validation was never stored or
logged, which is precisely the artefact needed to diagnose it. ``failure_excerpt`` and
``error_classification`` carry it, bounded and redacted at the application boundary.

**What is *not* done here, deliberately.** ``started_at``'s meaning changes from "when the row was
written" to "when the step began", and historical rows are **not** rewritten: for a row written
before this revision the true start time is not recoverable, and synthesising one from
``completed_at - latency_ms`` would manufacture an audit record. Existing rows keep
``started_at == completed_at``, which is a readable signal that they predate the change.

``attempt_count`` defaults to ``1`` rather than ``NULL``: every row that exists describes a step
that was attempted at least once, so ``1`` is the true value for history, not a guess. The two
failure columns are nullable with no backfill, because a historical failure genuinely has no
excerpt to recover.

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-22
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # `server_default` so the backfill of existing rows and the NOT NULL land in one statement;
    # dropped immediately afterwards so the application, not the database, owns the value going
    # forward -- a lingering server default would silently paper over a writer that forgot it.
    op.add_column(
        "agent_runs",
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="1"),
    )
    op.alter_column("agent_runs", "attempt_count", server_default=None)

    op.add_column("agent_runs", sa.Column("failure_excerpt", sa.Text(), nullable=True))
    op.add_column("agent_runs", sa.Column("error_classification", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("agent_runs", "error_classification")
    op.drop_column("agent_runs", "failure_excerpt")
    op.drop_column("agent_runs", "attempt_count")
