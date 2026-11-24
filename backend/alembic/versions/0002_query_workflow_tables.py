"""Query runs, agent runs, evidence, citations, evaluations, reports, events.

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-12
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _enum(*values: str, name: str) -> sa.Enum:
    return sa.Enum(*values, name=name, native_enum=False, length=30)


def upgrade() -> None:
    op.create_table(
        "query_runs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "project_id",
            sa.Uuid(),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("normalized_question", sa.Text(), nullable=False),
        sa.Column("retrieval_mode", _enum("hybrid", "semantic", "lexical", name="retrieval_mode")),
        sa.Column("response_style", _enum("concise", "detailed", name="response_style")),
        sa.Column("status", _enum("running", "completed", "failed", name="query_run_status")),
        sa.Column("answer_markdown", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("token_usage_estimate", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_query_runs_project_id", "query_runs", ["project_id"])

    op.create_table(
        "agent_runs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "query_run_id",
            sa.Uuid(),
            sa.ForeignKey("query_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "agent_name",
            _enum(
                "planner",
                "retriever",
                "researcher",
                "critic",
                "risk",
                "evaluator",
                "final_decision",
                name="agent_name",
            ),
        ),
        sa.Column(
            "status", _enum("pending", "running", "completed", "failed", name="agent_run_status")
        ),
        sa.Column("input_json", postgresql.JSONB(), nullable=False),
        sa.Column("output_json", postgresql.JSONB(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
    )
    op.create_index("ix_agent_runs_query_run_id", "agent_runs", ["query_run_id"])

    op.create_table(
        "retrieved_evidence",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "query_run_id",
            sa.Uuid(),
            sa.ForeignKey("query_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_id", sa.Uuid(), nullable=False),
        sa.Column("source", _enum("semantic", "lexical", "both", name="evidence_source")),
        sa.Column("semantic_score", sa.Float(), nullable=True),
        sa.Column("lexical_score", sa.Float(), nullable=True),
        sa.Column("hybrid_score", sa.Float(), nullable=False),
        sa.Column("rerank_score", sa.Float(), nullable=True),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("selected_for_context", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_retrieved_evidence_query_run_id", "retrieved_evidence", ["query_run_id"])

    op.create_table(
        "citations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "query_run_id",
            sa.Uuid(),
            sa.ForeignKey("query_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("citation_key", sa.Text(), nullable=False),
        sa.Column("snippet", sa.Text(), nullable=False),
        sa.Column("source_filename", sa.Text(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("relevance_score", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_citations_query_run_id", "citations", ["query_run_id"])

    op.create_table(
        "evaluation_results",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "query_run_id",
            sa.Uuid(),
            sa.ForeignKey("query_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("answer_relevance", sa.Float(), nullable=False),
        sa.Column("faithfulness", sa.Float(), nullable=False),
        sa.Column("citation_accuracy", sa.Float(), nullable=False),
        sa.Column("context_sufficiency", sa.Float(), nullable=False),
        sa.Column("hallucination_risk", sa.Float(), nullable=False),
        sa.Column("explanation", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("query_run_id", name="uq_evaluation_results_query_run_id"),
        sa.CheckConstraint(
            "answer_relevance BETWEEN 0 AND 1 AND faithfulness BETWEEN 0 AND 1 "
            "AND citation_accuracy BETWEEN 0 AND 1 AND context_sufficiency BETWEEN 0 AND 1 "
            "AND hallucination_risk BETWEEN 0 AND 1",
            name="ck_evaluation_scores_in_unit_interval",
        ),
    )
    op.create_index("ix_evaluation_results_query_run_id", "evaluation_results", ["query_run_id"])

    op.create_table(
        "reports",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "query_run_id",
            sa.Uuid(),
            sa.ForeignKey("query_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("markdown_content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_reports_query_run_id", "reports", ["query_run_id"])

    op.create_table(
        "agent_events",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("seq", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column(
            "query_run_id",
            sa.Uuid(),
            sa.ForeignKey("query_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "event_type",
            _enum(
                "planning_started",
                "retrieval_started",
                "retrieval_completed",
                "reranking_started",
                "research_started",
                "critic_started",
                "risk_started",
                "answer_started",
                "evaluation_started",
                "answer_completed",
                "workflow_failed",
                name="event_type",
            ),
        ),
        sa.Column(
            "agent_name",
            _enum(
                "planner",
                "retriever",
                "researcher",
                "critic",
                "risk",
                "evaluator",
                "final_decision",
                name="agent_name_event",
            ),
            nullable=True,
        ),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_agent_events_query_run_id_seq", "agent_events", ["query_run_id", "seq"])


def downgrade() -> None:
    op.drop_table("agent_events")
    op.drop_table("reports")
    op.drop_table("evaluation_results")
    op.drop_table("citations")
    op.drop_table("retrieved_evidence")
    op.drop_table("agent_runs")
    op.drop_table("query_runs")
