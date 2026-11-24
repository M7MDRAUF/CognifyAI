"""Evaluation dashboard schemas (PRD section 8.7)."""

import uuid
from datetime import datetime

from pydantic import BaseModel

from app.domain.enums import QueryRunStatus


class QualityPoint(BaseModel):
    created_at: datetime
    answer_quality: float


class StageLatency(BaseModel):
    agent_name: str
    average_latency_ms: float


class RiskCategoryCount(BaseModel):
    category: str
    count: int


class HallucinationBucket(BaseModel):
    label: str
    count: int


class RecentQuery(BaseModel):
    query_run_id: uuid.UUID
    question: str
    status: QueryRunStatus
    created_at: datetime
    answer_relevance: float | None
    faithfulness: float | None
    error_message: str | None


class EvaluationSummary(BaseModel):
    query_count: int
    completed_count: int
    failed_count: int
    cancelled_count: int
    running_count: int
    average_answer_relevance: float | None
    average_faithfulness: float | None
    average_citation_accuracy: float | None
    average_context_sufficiency: float | None
    average_latency_ms: float | None
    average_token_usage: float | None
    retrieval_hit_count: int
    agent_failure_count: int
    quality_over_time: list[QualityPoint]
    stage_latencies: list[StageLatency]
    risk_categories: list[RiskCategoryCount]
    hallucination_risk_distribution: list[HallucinationBucket]
    recent_queries: list[RecentQuery]
