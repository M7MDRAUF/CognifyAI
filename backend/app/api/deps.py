"""FastAPI dependency providers.

Routes receive services and probes through these functions only, so tests
can override a single provider instead of patching internals. Lifespan
stores a typed ``AppResources`` holder on ``app.state`` — the isinstance
checks below are the typed boundary around Starlette's untyped state.
"""

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from dataclasses import dataclass
from functools import partial
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.integrations.llm.router import LLMRouter
from app.integrations.postgres.database import check_postgres
from app.integrations.qdrant.client import QdrantService
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.evaluation_repository import EvaluationRepository
from app.repositories.event_repository import EventRepository
from app.repositories.pharma_repository import PharmaRepository
from app.repositories.project_repository import ProjectRepository
from app.repositories.query_repository import QueryRepository
from app.repositories.report_repository import ReportRepository
from app.services.document_service import DocumentService
from app.services.evaluation_service import EvaluationService
from app.services.event_bus import EventBus
from app.services.project_service import ProjectService
from app.services.query_service import QueryService
from app.services.report_service import ReportService
from app.services.run_registry import RunRegistry
from app.utils.files import FileStorage
from app.workers.jobs import run_ingestion_job, run_query_job

HealthProbe = Callable[[], Awaitable[bool]]
IngestionRunner = Callable[[uuid.UUID], Coroutine[None, None, None]]
QueryRunner = Callable[[uuid.UUID], Coroutine[None, None, None]]


@dataclass(frozen=True)
class AppResources:
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    qdrant: QdrantService
    llm: LLMRouter
    event_bus: EventBus
    run_registry: RunRegistry


def get_app_settings(request: Request) -> Settings:
    settings = request.app.state.settings
    if not isinstance(settings, Settings):
        msg = "Application settings are not initialized."
        raise RuntimeError(msg)
    return settings


def get_resources(request: Request) -> AppResources:
    resources = request.app.state.resources
    if not isinstance(resources, AppResources):
        msg = "Application resources are not initialized."
        raise RuntimeError(msg)
    return resources


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
ResourcesDep = Annotated[AppResources, Depends(get_resources)]


def get_postgres_probe(resources: ResourcesDep) -> HealthProbe:
    return partial(check_postgres, resources.engine)


def get_qdrant_probe(resources: ResourcesDep) -> HealthProbe:
    return resources.qdrant.check_ready


async def get_session(resources: ResourcesDep) -> AsyncIterator[AsyncSession]:
    async with resources.session_factory() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]


def get_file_storage(settings: SettingsDep) -> FileStorage:
    return FileStorage(settings.upload_dir)


FileStorageDep = Annotated[FileStorage, Depends(get_file_storage)]


def get_project_service(
    session: SessionDep, storage: FileStorageDep, settings: SettingsDep, resources: ResourcesDep
) -> ProjectService:
    return ProjectService(
        session=session,
        projects=ProjectRepository(session),
        storage=storage,
        qdrant=resources.qdrant,
        pharma_profile_enabled=settings.pharma_profile_enabled,
    )


def get_document_service(
    session: SessionDep, storage: FileStorageDep, settings: SettingsDep, resources: ResourcesDep
) -> DocumentService:
    return DocumentService(
        session=session,
        projects=ProjectRepository(session),
        documents=DocumentRepository(session),
        chunks=ChunkRepository(session),
        storage=storage,
        settings=settings,
        qdrant=resources.qdrant,
        controlled=PharmaRepository(session),
    )


def get_ingestion_runner(resources: ResourcesDep, settings: SettingsDep) -> IngestionRunner:
    return partial(
        run_ingestion_job,
        resources.session_factory,
        settings,
        resources.qdrant,
        resources.llm.embeddings(),
    )


def get_query_service(
    session: SessionDep, resources: ResourcesDep, settings: SettingsDep
) -> QueryService:
    """``settings`` carries the accept path's capacity limits, and every production
    ``QueryService`` is built here, so the admission cap in ``QueryService.create`` is always
    active in a running deployment."""
    return QueryService(
        session=session,
        projects=ProjectRepository(session),
        queries=QueryRepository(session),
        llm=resources.llm,
        registry=resources.run_registry,
        settings=settings,
        pharma=PharmaRepository(session),
        # RMD5-T01 (DV-RMD-31): an ownerless run a cancel closes ends a connected stream too.
        bus=resources.event_bus,
    )


def get_query_runner(resources: ResourcesDep, settings: SettingsDep) -> QueryRunner:
    return partial(
        run_query_job,
        resources.session_factory,
        settings,
        resources.qdrant,
        resources.llm,
        resources.event_bus,
        resources.run_registry,
    )


def get_event_repository(session: SessionDep) -> EventRepository:
    return EventRepository(session)


def get_evaluation_service(session: SessionDep) -> EvaluationService:
    return EvaluationService(ProjectRepository(session), EvaluationRepository(session))


def get_report_service(session: SessionDep) -> ReportService:
    return ReportService(
        session=session,
        queries=QueryRepository(session),
        reports=ReportRepository(session),
        projects=ProjectRepository(session),
        pharma=PharmaRepository(session),
    )


ProjectServiceDep = Annotated[ProjectService, Depends(get_project_service)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
IngestionRunnerDep = Annotated[IngestionRunner, Depends(get_ingestion_runner)]
QueryServiceDep = Annotated[QueryService, Depends(get_query_service)]
QueryRunnerDep = Annotated[QueryRunner, Depends(get_query_runner)]
EventRepositoryDep = Annotated[EventRepository, Depends(get_event_repository)]
EvaluationServiceDep = Annotated[EvaluationService, Depends(get_evaluation_service)]
ReportServiceDep = Annotated[ReportService, Depends(get_report_service)]
