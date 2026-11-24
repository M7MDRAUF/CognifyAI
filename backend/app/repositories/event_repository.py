"""SQL access for persisted workflow events (enables SSE replay)."""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import AgentEvent


class EventRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, event: AgentEvent) -> None:
        self._session.add(event)
        await self._session.flush()

    async def list_for_query_run(self, query_run_id: uuid.UUID) -> list[AgentEvent]:
        result = await self._session.scalars(
            select(AgentEvent)
            .where(AgentEvent.query_run_id == query_run_id)
            .order_by(AgentEvent.seq)
        )
        return list(result.all())
