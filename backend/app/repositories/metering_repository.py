"""Embedding metering: one ``embedding_calls`` row per provider call (RMD9-T04, RE2E-DEF-029).

A meter belongs to one job, an ingestion or a run. It keeps a row for each call as the call
returns, and its job writes them in one session of their own when it ends (``flush``), after the
document's or the run's terminal write. Not in the job's own session: the ingestion pipeline
holds its transaction open until it ends, so a row there would be rolled back with a failed
ingestion although the vendor answered the call, and a commit inside it would split the
reprocess's single transaction. Not one session per call either: on this stack a write per call
added a commit, and at times a new connection, to every document's ingestion, measured at 10 to
31 percent of the median document's time (DV-RMD-67).

Rows that cannot be written are logged and counted, and the work goes on: metering records the
work, it never decides whether the work happens.
"""

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.models import EmbeddingCall
from app.utils.ids import new_uuid
from app.utils.time import utc_now

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingScope:
    """What a call served: its project and exactly one of a document or a run."""

    project_id: uuid.UUID
    document_id: uuid.UUID | None = None
    query_run_id: uuid.UUID | None = None


class EmbeddingMeter:
    def __init__(
        self, session_factory: async_sessionmaker[AsyncSession], scope: EmbeddingScope
    ) -> None:
        self._session_factory = session_factory
        self._scope = scope
        self._pending: list[EmbeddingCall] = []
        # The rows this meter could not write, also logged at each failed flush.
        self.failed_writes = 0

    def record(
        self,
        *,
        provider: str,
        model: str,
        input_count: int,
        prompt_tokens: int | None,
        latency_ms: int,
        outcome: str,
        failure_kind: str | None = None,
    ) -> None:
        """Keeps the call's row, timed when the call returned, for the next ``flush``."""
        self._pending.append(
            EmbeddingCall(
                id=new_uuid(),
                project_id=self._scope.project_id,
                document_id=self._scope.document_id,
                query_run_id=self._scope.query_run_id,
                provider=provider,
                model=model,
                input_count=input_count,
                prompt_tokens=prompt_tokens,
                latency_ms=latency_ms,
                outcome=outcome,
                failure_kind=failure_kind,
                created_at=utc_now(),
            )
        )

    async def flush(self) -> None:
        """Writes the rows kept since the last flush, in one session and one commit."""
        rows, self._pending = self._pending, []
        if not rows:
            return
        try:
            async with self._session_factory() as session:
                session.add_all(rows)
                await session.commit()
        except Exception as exc:
            self.failed_writes += len(rows)
            # The class only: the error's text can carry the statement and its parameters.
            logger.warning(
                "%d embedding metering rows were not written (%s; %d so far)",
                len(rows),
                type(exc).__name__,
                self.failed_writes,
                extra={"event_type": "embedding_metering_failed"},
            )
