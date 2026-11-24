"""Query routes (SDD section 7.4). Handlers stay logic-free."""

import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Query

from app.api.deps import QueryRunnerDep, QueryServiceDep
from app.domain.enums import QueryRunStatus
from app.schemas.queries import (
    QueryCreate,
    QueryRunCancelResponse,
    QueryRunDetail,
    QueryRunRead,
    QueryRunSummaryPage,
)
from app.services.pagination import MAX_CURSOR_LENGTH

router = APIRouter(tags=["queries"])


@router.post("/projects/{project_id}/queries", response_model=QueryRunRead, status_code=202)
async def create_query(
    project_id: uuid.UUID,
    payload: QueryCreate,
    background: BackgroundTasks,
    service: QueryServiceDep,
    runner: QueryRunnerDep,
) -> QueryRunRead:
    query_run = await service.create(project_id, payload)
    background.add_task(runner, query_run.id)
    return QueryRunRead.model_validate(query_run)


@router.get("/projects/{project_id}/queries", response_model=list[QueryRunRead], deprecated=True)
async def list_queries(project_id: uuid.UUID, service: QueryServiceDep) -> list[QueryRunRead]:
    runs = await service.list_for_project(project_id)
    return [QueryRunRead.model_validate(run) for run in runs]


@router.get("/projects/{project_id}/queries/summaries", response_model=QueryRunSummaryPage)
async def list_query_summaries(
    project_id: uuid.UUID,
    service: QueryServiceDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: Annotated[str | None, Query(max_length=MAX_CURSOR_LENGTH)] = None,
    status: QueryRunStatus | None = None,
) -> QueryRunSummaryPage:
    return await service.list_summaries_for_project(
        project_id, limit=limit, cursor=cursor, status=status
    )


@router.get("/queries/{query_run_id}", response_model=QueryRunDetail)
async def get_query(query_run_id: uuid.UUID, service: QueryServiceDep) -> QueryRunDetail:
    return await service.get_detail(query_run_id)


@router.post(
    "/queries/{query_run_id}/cancel",
    response_model=QueryRunCancelResponse,
    response_model_exclude_unset=True,
    status_code=202,
)
async def cancel_query(query_run_id: uuid.UUID, service: QueryServiceDep) -> QueryRunCancelResponse:
    """RMD5-T03 (RE2E-DEF-037): every 202 says ``cancel_requested: true``. When this process's
    task was cancelled but the database could not be read, the body names only the run, its
    ``running`` status and the pending cancel."""
    query_run = await service.cancel(query_run_id)
    if query_run is None:
        return QueryRunCancelResponse(
            id=query_run_id, status=QueryRunStatus.RUNNING, cancel_requested=True
        )
    read = QueryRunRead.model_validate(query_run).model_dump()
    return QueryRunCancelResponse.model_validate({**read, "cancel_requested": True})
