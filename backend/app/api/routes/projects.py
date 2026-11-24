"""Project routes (SDD section 7.2). Handlers stay logic-free."""

import uuid

from fastapi import APIRouter

from app.api.deps import ProjectServiceDep
from app.schemas.projects import ProjectCreate, ProjectDetail, ProjectRead

router = APIRouter(prefix="/projects", tags=["projects"])


@router.post("", response_model=ProjectRead, status_code=201)
async def create_project(payload: ProjectCreate, service: ProjectServiceDep) -> ProjectRead:
    project = await service.create(payload)
    return ProjectRead.model_validate(project)


@router.get("", response_model=list[ProjectRead])
async def list_projects(service: ProjectServiceDep) -> list[ProjectRead]:
    projects = await service.list_projects()
    return [ProjectRead.model_validate(project) for project in projects]


@router.get("/{project_id}", response_model=ProjectDetail)
async def get_project(project_id: uuid.UUID, service: ProjectServiceDep) -> ProjectDetail:
    project, stats = await service.get_detail(project_id)
    return ProjectDetail(**ProjectRead.model_validate(project).model_dump(), stats=stats)


@router.delete("/{project_id}", status_code=204)
async def delete_project(project_id: uuid.UUID, service: ProjectServiceDep) -> None:
    await service.delete_project(project_id)
