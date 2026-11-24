"""SQL access for projects and their dashboard aggregates."""

import uuid

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import QueryRunStatus
from app.domain.models import Chunk, Document, EvaluationResult, Project, QueryRun
from app.schemas.projects import ProjectStats


class ProjectRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, project: Project) -> None:
        self._session.add(project)
        await self._session.flush()

    async def list_all(self) -> list[Project]:
        result = await self._session.scalars(select(Project).order_by(Project.created_at.desc()))
        return list(result.all())

    async def get(self, project_id: uuid.UUID) -> Project | None:
        return await self._session.get(Project, project_id)

    async def delete(self, project: Project) -> None:
        await self._session.delete(project)
        await self._session.flush()

    async def stats(self, project_id: uuid.UUID) -> ProjectStats:
        document_count = await self._count(
            select(func.count()).select_from(Document).where(Document.project_id == project_id)
        )
        chunk_count = await self._count(
            select(func.count()).select_from(Chunk).where(Chunk.project_id == project_id)
        )
        vector_count = await self._count(
            select(func.count())
            .select_from(Chunk)
            .where(Chunk.project_id == project_id, Chunk.qdrant_point_id.is_not(None))
        )
        last_query_at = await self._session.scalar(
            select(func.max(QueryRun.created_at)).where(QueryRun.project_id == project_id)
        )
        average_latency = await self._session.scalar(
            select(func.avg(QueryRun.latency_ms)).where(
                QueryRun.project_id == project_id,
                QueryRun.status == QueryRunStatus.COMPLETED,
            )
        )
        eval_averages = (
            await self._session.execute(
                select(
                    func.avg(EvaluationResult.answer_relevance),
                    func.avg(EvaluationResult.citation_accuracy),
                )
                .select_from(EvaluationResult)
                .join(QueryRun, EvaluationResult.query_run_id == QueryRun.id)
                .where(QueryRun.project_id == project_id)
            )
        ).one()
        return ProjectStats(
            document_count=document_count,
            chunk_count=chunk_count,
            vector_count=vector_count,
            last_query_at=last_query_at,
            average_answer_quality=_as_float(eval_averages[0]),
            average_citation_accuracy=_as_float(eval_averages[1]),
            average_latency_ms=_as_float(average_latency),
        )

    async def _count(self, statement: Select[tuple[int]]) -> int:
        return (await self._session.scalar(statement)) or 0


def _as_float(value: float | None) -> float | None:
    return float(value) if value is not None else None
