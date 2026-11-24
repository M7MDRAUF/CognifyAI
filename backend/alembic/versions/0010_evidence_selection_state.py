"""Evidence selection state and a uniqueness invariant (M-8, N-003).

**M-8 — every row claimed it reached the final context.** `_persist_evidence` hardcoded
`selected_for_context=True` for every row it wrote, so the column recorded nothing: a candidate
that the reranker cut and a chunk that actually reached the synthesis agents were indistinguishable
in the database. Any question of the form "what evidence did this answer actually see" was
unanswerable from the persisted record.

Four states replace the boolean, and they are ordered — each implies the ones before it:

* `candidate` — retrieved and hydrated, nothing more;
* `reranked` — scored by the reranker (including scored **zero**, which is a judgement);
* `selected` — survived the global top-K cut;
* `supplied` — passed to the synthesis agents as context.

`selected` and `supplied` differ on purpose: a run that is cancelled or fails between the cut and
the next agent call has rows that were selected and never supplied, and collapsing the two would
claim an agent saw evidence it never received.

**N-003 — the same chunk could be persisted twice.** `_expand` wrote all of round two's rows
without checking round one's, and nothing in the schema forbade it, so one chunk could appear
twice for one run with two different ranks. A unique constraint on `(query_run_id, chunk_id)` makes
that impossible rather than merely unlikely.

**The duplicate cleanup is reversible and non-destructive.** Duplicates are *moved* to a side
table before the constraint is added, keeping the lowest-`rank` row of each group -- the one
nearest the final context. The downgrade puts them back. Deleting them outright would make the
downgrade a lie, and these rows are real evidence records of real runs.

`selected_for_context` is **kept**, not dropped: it is the column existing readers use, and
removing it in the same revision that adds its replacement would break them for no gain. It is
now derived -- true exactly when the state is `selected` or `supplied`.

Revision ID: 0010
Revises: 0009
Create Date: 2026-09-22
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_UNIQUE = "uq_retrieved_evidence_query_run_chunk"

# Table names are written as literals in the statements below rather than interpolated from a
# constant: the value is fixed and under our control either way, but an f-string around SQL is a
# construct a reader (and a linter) has to stop and verify, and there is nothing here to gain by
# it.


def upgrade() -> None:
    op.add_column(
        "retrieved_evidence",
        sa.Column(
            "selection_state",
            sa.Text(),
            nullable=False,
            server_default="candidate",
        ),
    )
    op.add_column("retrieved_evidence", sa.Column("round_index", sa.Integer(), nullable=True))
    op.add_column("retrieved_evidence", sa.Column("round_rank", sa.Integer(), nullable=True))

    # Historical rows all claimed `selected_for_context = True`, which is exactly the defect.
    # `selected` is the honest reading of what that claim *meant*, and it is the only value that
    # does not invent information: it says the row was marked as reaching the context, without
    # asserting the stronger `supplied`, which nothing recorded.
    op.execute(
        sa.text(
            "UPDATE retrieved_evidence SET selection_state = 'selected' "
            "WHERE selected_for_context IS TRUE"
        )
    )
    op.alter_column("retrieved_evidence", "selection_state", server_default=None)

    # Quarantine before constraining. Same shape as the source table so a restore is a plain
    # INSERT ... SELECT, with no column mapping to get wrong.
    op.execute(
        sa.text(
            "CREATE TABLE retrieved_evidence_duplicates AS "
            "SELECT * FROM retrieved_evidence WHERE id IN ("
            "  SELECT id FROM ("
            "    SELECT id, ROW_NUMBER() OVER ("
            "      PARTITION BY query_run_id, chunk_id ORDER BY rank ASC, created_at ASC"
            "    ) AS position"
            "    FROM retrieved_evidence"
            "  ) ranked WHERE ranked.position > 1"
            ")"
        )
    )
    op.execute(
        sa.text(
            "DELETE FROM retrieved_evidence WHERE id IN "
            "(SELECT id FROM retrieved_evidence_duplicates)"
        )
    )
    op.create_unique_constraint(_UNIQUE, "retrieved_evidence", ["query_run_id", "chunk_id"])


def downgrade() -> None:
    op.drop_constraint(_UNIQUE, "retrieved_evidence", type_="unique")
    # Restore the quarantined duplicates before dropping the columns, so the rows come back with
    # the values they had. `IF EXISTS` because a database with no duplicates never created it.
    op.execute(
        sa.text(
            "INSERT INTO retrieved_evidence SELECT * FROM retrieved_evidence_duplicates "
            "WHERE EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_name = 'retrieved_evidence_duplicates')"
        )
    )
    op.execute(sa.text("DROP TABLE IF EXISTS retrieved_evidence_duplicates"))
    op.drop_column("retrieved_evidence", "round_rank")
    op.drop_column("retrieved_evidence", "round_index")
    op.drop_column("retrieved_evidence", "selection_state")
