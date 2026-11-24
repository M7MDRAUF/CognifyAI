"""Evaluation routes (SDD section 7.6). Handlers stay logic-free."""

import uuid

from fastapi import APIRouter

from app.api.deps import EvaluationServiceDep, QueryServiceDep
from app.schemas.evaluations import EvaluationSummary
from app.schemas.queries import EvaluationRead

router = APIRouter(tags=["evaluations"])


@router.get("/queries/{query_run_id}/evaluation", response_model=EvaluationRead)
async def get_evaluation(query_run_id: uuid.UUID, service: QueryServiceDep) -> EvaluationRead:
    return await service.get_evaluation(query_run_id)


@router.get("/projects/{project_id}/evaluations/summary", response_model=EvaluationSummary)
async def get_evaluation_summary(
    project_id: uuid.UUID, service: EvaluationServiceDep
) -> EvaluationSummary:
    return await service.summary(project_id)
