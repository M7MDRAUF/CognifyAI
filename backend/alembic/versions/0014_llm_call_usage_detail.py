"""The call details of a metering row (PLV8-T02; PLV-D-027).

``llm_calls`` gains seven nullable columns, NULL meaning "not reported": the cached prompt tokens
and the reasoning tokens a vendor prices apart, the thinking mode and effort the request was sent
with, the provider's id of each response, and the model and system fingerprint the response
named. Existing rows keep NULL in each, which is what they are: unreported.

Additive. The downgrade drops exactly these seven columns, so every row and every other column
survives it.

Revision ID: 0014
Revises: 0013
Create Date: 2026-10-12
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0014"
down_revision: str | None = "0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_COLUMNS: tuple[tuple[str, type[sa.Integer] | type[sa.Text]], ...] = (
    ("cached_prompt_tokens", sa.Integer),
    ("reasoning_tokens", sa.Integer),
    ("thinking", sa.Text),
    ("reasoning_effort", sa.Text),
    ("provider_request_ids", sa.Text),
    ("response_model", sa.Text),
    ("system_fingerprint", sa.Text),
)


def upgrade() -> None:
    for name, kind in _COLUMNS:
        op.add_column("llm_calls", sa.Column(name, kind(), nullable=True))


def downgrade() -> None:
    for name, _ in reversed(_COLUMNS):
        op.drop_column("llm_calls", name)
