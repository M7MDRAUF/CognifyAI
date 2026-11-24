"""Report routes (SDD section 7.7). Handlers stay logic-free."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Query
from fastapi.responses import PlainTextResponse

from app.api.deps import ReportServiceDep
from app.schemas.reports import ReportPage, ReportRead
from app.services.pagination import MAX_CURSOR_LENGTH

router = APIRouter(tags=["reports"])


@router.post("/queries/{query_run_id}/reports", response_model=ReportRead, status_code=201)
async def create_report(query_run_id: uuid.UUID, service: ReportServiceDep) -> ReportRead:
    report = await service.generate(query_run_id)
    return ReportRead.model_validate(report)


@router.get("/projects/{project_id}/reports", response_model=ReportPage)
async def list_reports(
    project_id: uuid.UUID,
    service: ReportServiceDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: Annotated[str | None, Query(max_length=MAX_CURSOR_LENGTH)] = None,
) -> ReportPage:
    return await service.list_for_project(project_id, limit=limit, cursor=cursor)


@router.get("/reports/{report_id}", response_model=ReportRead)
async def get_report(report_id: uuid.UUID, service: ReportServiceDep) -> ReportRead:
    report = await service.get(report_id)
    return ReportRead.model_validate(report)


@router.get("/reports/{report_id}/download.md", response_class=PlainTextResponse)
async def download_report(report_id: uuid.UUID, service: ReportServiceDep) -> PlainTextResponse:
    report = await service.get(report_id)
    return PlainTextResponse(
        content=report.markdown_content,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="report-{report_id}.md"'},
    )
