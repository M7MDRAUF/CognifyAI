"""SQL access for generated reports."""

import uuid
from datetime import datetime

from sqlalchemy import func, literal, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import load_only

from app.domain.models import QueryRun, Report


class ReportRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, report: Report) -> None:
        self._session.add(report)
        await self._session.flush()

    async def get(self, report_id: uuid.UUID) -> Report | None:
        return await self._session.get(Report, report_id)

    async def list_for_project(
        self,
        project_id: uuid.UUID,
        *,
        limit: int,
        before: tuple[datetime, uuid.UUID] | None = None,
    ) -> list[Report]:
        """Up to ``limit + 1`` of the project's reports strictly after ``before`` in
        ``(created_at DESC, id DESC)`` order, without their Markdown.

        ``reports`` has no ``project_id``: a report belongs to a project through its run, so the
        join on ``query_runs.project_id`` is what keeps another project's reports out. The extra
        row tells the caller whether another page exists.
        """
        statement = (
            select(Report)
            .options(load_only(Report.id, Report.query_run_id, Report.title, Report.created_at))
            .join(QueryRun, QueryRun.id == Report.query_run_id)
            .where(QueryRun.project_id == project_id)
            .order_by(Report.created_at.desc(), Report.id.desc())
            .limit(limit + 1)
        )
        if before is not None:
            created_at, row_id = before
            statement = statement.where(
                tuple_(Report.created_at, Report.id)
                < tuple_(
                    literal(created_at, Report.created_at.type),
                    literal(row_id, Report.id.type),
                )
            )
        result = await self._session.scalars(statement)
        return list(result.all())

    async def count_for_project(self, project_id: uuid.UUID) -> int:
        """How many reports the project's runs carry."""
        counted = await self._session.scalars(
            select(func.count())
            .select_from(Report)
            .join(QueryRun, QueryRun.id == Report.query_run_id)
            .where(QueryRun.project_id == project_id)
        )
        return counted.one()
