"""Named routing profile on query_runs (C-16, SDD 10.3, ADR-005).

``answer_model`` is a *tier*, and ``_synthesis_role`` mapped one tier onto four different agents
at once -- so ``answer_model=fast`` put the critic, the risk agent, the final-decision agent and
the evaluator all on the small model, while the SDD section the code's own comment cited put all
four on the stronger one. A tier also cannot express *which* of the seven call sites it affects,
so "this run used the fast model" was not answerable from the stored row.

``routing_profile`` records the named profile the run was created with. The per-site mapping
itself lives in ``app/integrations/llm/profiles.py`` and is deliberately **not** a column: it is
configuration, it is versioned with the code, and duplicating it per row would let a stored row
disagree with the profile it names.

**Nullable, no backfill.** A row written before profiles existed genuinely has no profile, and
``_routing_profile`` resolves those through the ``answer_model`` alias -- so a legacy run keeps
behaving exactly as it did. Backfilling ``'balanced'`` would be worse than wrong: it would claim
every historical ``answer_model=fast`` run had routed its synthesis agents to the strong model,
which is the opposite of what happened.

``answer_model`` is kept, not dropped. It is the deprecated alias (SDD 10.3.4), removing an
accepted request field is a breaking change, and there is no reason to spend one here.

Revision ID: 0008
Revises: 0007
Create Date: 2026-09-22
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("query_runs", sa.Column("routing_profile", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("query_runs", "routing_profile")
