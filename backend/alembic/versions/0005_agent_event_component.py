"""Internal component identity on agent_events (C-13, SDD 8.3.1, ADR-007).

``agent_events.agent_name`` is nullable and was **never populated**: all fourteen emit sites
omitted it, so every frame on the wire carried ``null``. The column existed, the frontend read
it, and nothing wrote it.

Populating it is only half an answer. Three of the twelve event types are produced by components
that are not agents at all -- the retriever, the reranker and the worker -- and for those there is
no valid ``AgentName`` to write. Widening that enum is not an option either: the client parses
``event_type`` and ``agent_name`` against *strict* Zod enums per frame, so an unrecognised member
fails the parse and the frame is silently **dropped** from the timeline.

So identity for non-agent producers gets its own column. ``component`` is additive, nullable and
outside the agent enumeration, and the client's ``agentEventSchema`` is a ``z.object``, which
strips unknown keys -- an old client receives the field, ignores it, and keeps working.

**No backfill, deliberately.** A historical row legitimately has no component: nothing recorded
one, and inferring it from ``event_type`` would fabricate provenance for rows that never carried
any. NULL means "emitted before this column existed", which is the truth.

``sa.Text()`` rather than an enum, for the same reason ``0004`` chose it: the component set is
internal and expected to grow (an evaluation runner and a provider adapter are already named in
the design), and an enum column would mean a migration for each.

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-22
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("agent_events", sa.Column("component", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("agent_events", "component")
