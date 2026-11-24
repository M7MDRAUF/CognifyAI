"""Mark historical evaluation rows attached to ineligible runs (N-006).

`P2-T13` gave every quality aggregate one eligibility predicate: completed, has an answer, has an
evaluation. That fixes the *queries*. It does not fix the rows already in the table.

Two kinds of historical row are inconsistent with the policy:

* an `evaluation_results` row whose `query_runs` row is **not** completed -- written before
  `P2-T14` reordered the writes, so an interrupted run could leave an evaluation behind;
* an `evaluation_results` row whose run **has no answer**, the same window seen from the other
  side.

**Marked, never deleted.** The card forbids a destructive delete without a verified backup, and
deletion would be wrong regardless: the row is a real measurement of a real (if abandoned) run,
and the eligibility predicate already excludes it from every aggregate. What was missing was a
way to *tell* an operator that a row was excluded and why, without re-deriving the join.

The downgrade nulls the column rather than restoring per-row values, and that is exact: before
this revision the column did not exist, so `NULL` is precisely the prior state. The column is the
only thing this migration writes -- no existing column is touched, so nothing else needs restoring.

Reversibility is therefore total, and the test asserts it by counting rows before, after, and
after the downgrade.

Revision ID: 0009
Revises: 0008
Create Date: 2026-09-22
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Why a row is excluded from quality analytics. NULL means "eligible", which is the common case
# and costs nothing to store.
_RUN_NOT_COMPLETED = "run_not_completed"
_RUN_HAS_NO_ANSWER = "run_has_no_answer"


def upgrade() -> None:
    op.add_column(
        "evaluation_results",
        sa.Column("analytics_exclusion_reason", sa.Text(), nullable=True),
    )

    # Ordered so the more specific reason wins: a run that is both incomplete and answerless is
    # reported as incomplete, which is the cause rather than the symptom.
    op.execute(
        sa.text(
            """
            UPDATE evaluation_results
            SET analytics_exclusion_reason = :reason
            FROM query_runs
            WHERE query_runs.id = evaluation_results.query_run_id
              AND query_runs.answer_markdown IS NULL
            """
        ).bindparams(reason=_RUN_HAS_NO_ANSWER)
    )
    op.execute(
        sa.text(
            """
            UPDATE evaluation_results
            SET analytics_exclusion_reason = :reason
            FROM query_runs
            WHERE query_runs.id = evaluation_results.query_run_id
              AND query_runs.status <> 'completed'
            """
        ).bindparams(reason=_RUN_NOT_COMPLETED)
    )


def downgrade() -> None:
    # Dropping the column restores the prior state exactly: it did not exist before this
    # revision, and nothing else was modified.
    op.drop_column("evaluation_results", "analytics_exclusion_reason")
