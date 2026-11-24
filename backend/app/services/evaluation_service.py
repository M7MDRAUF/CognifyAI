"""Builds the dashboard EvaluationSummary from repository aggregates."""

import uuid

from app.core.errors import ProjectNotFoundError
from app.repositories.evaluation_repository import EvaluationRepository
from app.repositories.project_repository import ProjectRepository
from app.schemas.evaluations import (
    EvaluationSummary,
    HallucinationBucket,
    QualityPoint,
    RecentQuery,
    RiskCategoryCount,
    StageLatency,
)


class EvaluationService:
    def __init__(self, projects: ProjectRepository, evaluations: EvaluationRepository) -> None:
        self._projects = projects
        self._evaluations = evaluations

    async def summary(self, project_id: uuid.UUID) -> EvaluationSummary:
        if await self._projects.get(project_id) is None:
            raise ProjectNotFoundError(f"Project {project_id} does not exist.")
        data = await self._evaluations.summary(project_id)
        return EvaluationSummary(
            query_count=data.query_count,
            completed_count=data.completed_count,
            failed_count=data.failed_count,
            cancelled_count=data.cancelled_count,
            running_count=data.running_count,
            average_answer_relevance=data.avg_answer_relevance,
            average_faithfulness=data.avg_faithfulness,
            average_citation_accuracy=data.avg_citation_accuracy,
            average_context_sufficiency=data.avg_context_sufficiency,
            average_latency_ms=data.avg_latency_ms,
            average_token_usage=data.avg_token_usage,
            retrieval_hit_count=data.retrieval_hit_count,
            agent_failure_count=data.agent_failure_count,
            quality_over_time=[
                QualityPoint(created_at=created_at, answer_quality=score)
                for created_at, score in data.quality_over_time
            ],
            stage_latencies=[
                StageLatency(agent_name=name, average_latency_ms=latency)
                for name, latency in data.stage_latencies
            ],
            risk_categories=[
                RiskCategoryCount(category=category, count=count)
                for category, count in data.risk_categories
            ],
            hallucination_risk_distribution=[
                HallucinationBucket(label=label, count=count)
                for label, count in data.hallucination_buckets
            ],
            recent_queries=[
                RecentQuery(
                    query_run_id=row[0],
                    question=row[1],
                    status=row[2],
                    created_at=row[3],
                    answer_relevance=row[4],
                    faithfulness=row[5],
                    error_message=row[6],
                )
                for row in data.recent_queries
            ],
        )
