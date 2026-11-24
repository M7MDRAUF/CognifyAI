"""The pharmaceutical profile's storage: a project's domain profile and controlled documents.

``projects.domain_profile`` records which profile a project was created with (REQ-PRF-01).
It is not null with the server default ``general``, so every project written before this
revision reads ``general`` after it and the General code path, which never reads the column,
sees no difference. Stored as a constrained string, as the baseline stores its enumerations,
so a later profile is a one-line change to the check.

``controlled_documents`` holds one row per document of a pharmaceutical project: the controlled
metadata its header parsed to (REQ-DOC-08, REQ-DOC-09). The row's key is the document's id and
both foreign keys cascade, so deleting a document or its project deletes the row with it
(REQ-PER-07). Invalid or missing header values stay null; ``metadata_issues`` says why.

Additive. The downgrade drops the table and the column and nothing else, so every row of a
pre-existing table survives it; a pharmaceutical project read after a downgrade is a plain
project.

Revision ID: 0012
Revises: 0011
Create Date: 2026-10-09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _constrained(*values: str, name: str) -> sa.Enum:
    return sa.Enum(*values, name=name, native_enum=False, length=30)


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column(
            "domain_profile",
            sa.String(length=30),
            nullable=False,
            server_default="general",
        ),
    )
    op.create_check_constraint(
        "ck_projects_domain_profile",
        "projects",
        "domain_profile IN ('general', 'pharma_manufacturing')",
    )
    op.create_table(
        "controlled_documents",
        sa.Column(
            "document_id",
            sa.Uuid(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "project_id",
            sa.Uuid(),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("document_number", sa.Text(), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column(
            "document_type",
            _constrained(
                "sop",
                "work_instruction",
                "equipment_manual",
                "form",
                "regulation",
                "guidance",
                "other",
                name="controlled_document_type",
            ),
            nullable=True,
        ),
        sa.Column("revision", sa.Text(), nullable=True),
        sa.Column("effective_date", sa.Date(), nullable=True),
        sa.Column("lifecycle_status", sa.String(length=30), nullable=False),
        sa.Column("supersedes_number", sa.Text(), nullable=True),
        sa.Column("supersedes_revision", sa.Text(), nullable=True),
        sa.Column("site", sa.Text(), nullable=True),
        sa.Column("area", sa.Text(), nullable=True),
        sa.Column(
            "equipment",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("process", sa.Text(), nullable=True),
        sa.Column("product_family", sa.Text(), nullable=True),
        sa.Column(
            "source_authority",
            _constrained("site", "vendor", "regulator", "other", name="source_authority"),
            nullable=True,
        ),
        sa.Column(
            "control_status",
            _constrained("controlled", "uncontrolled", "reference", name="control_status"),
            nullable=True,
        ),
        sa.Column(
            "data_classification",
            _constrained("synthetic", "public", name="data_classification"),
            nullable=True,
        ),
        sa.Column("metadata_status", sa.String(length=30), nullable=False),
        sa.Column(
            "metadata_issues",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "lifecycle_status IN ('draft', 'effective', 'superseded', 'retired', 'unknown')",
            name="ck_controlled_documents_lifecycle_status",
        ),
        sa.CheckConstraint(
            "metadata_status IN ('valid', 'missing', 'invalid')",
            name="ck_controlled_documents_metadata_status",
        ),
    )
    op.create_index("ix_controlled_documents_project_id", "controlled_documents", ["project_id"])
    op.create_index(
        "ix_controlled_documents_project_id_document_number",
        "controlled_documents",
        ["project_id", "document_number"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_controlled_documents_project_id_document_number", table_name="controlled_documents"
    )
    op.drop_index("ix_controlled_documents_project_id", table_name="controlled_documents")
    op.drop_table("controlled_documents")
    op.drop_constraint("ck_projects_domain_profile", "projects", type_="check")
    op.drop_column("projects", "domain_profile")
