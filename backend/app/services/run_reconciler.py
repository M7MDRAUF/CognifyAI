"""Run leases and the reconciler (RMD5-T01; RE2E-DEF-038, RE2E-DEF-041).

``RUNNING`` was a claim nothing checked. A run whose process died kept it for good, and so did a
run whose terminal write failed while the database was down; ``POST /cancel`` could not clear
either, because no process held a task for them. Two things now make the status a checked claim:

- **The lease.** The job that runs a query claims its row (``owner_instance`` names this process,
  ``lease_expires_at`` when the claim ends) and renews it every third of
  ``query_run_lease_seconds`` while it holds the run, its terminal write included
  (``app.workers.jobs``). A renewal that finds the run closed by another instance stops the
  workflow rather than let it write a second terminal state.
- **The reconciler.** At startup and every ``run_reconcile_interval_seconds``, ``reconcile_once``
  ends each ``RUNNING`` run whose lease has lapsed -- or, for a row with no lease (written before
  migration 0015, or not yet claimed), whose deadline has passed -- and for which this process
  holds no live task. The run ends ``failed`` with a reason naming the lapse, and its one terminal
  event (``workflow_failed``, payload code ``run_abandoned``) is stored and published, so a
  connected stream ends too. ``cancel_ownerless`` is the same close for a run a user stops: it
  ends ``cancelled`` at once instead of answering 409.

One terminal state per run: the reconciler claims its candidates with ``FOR UPDATE SKIP LOCKED``
and writes them in one conditional ``UPDATE``, and the job's terminal writers take the same row
lock before they check the status, so whichever writer is second finds a terminal row.

A lease proves liveness only between processes that can reach the database. An owner cut off from
it for longer than its lease, while another process reconciles, is closed; its next renewal sees
that and stops its workflow. The clocks of the processes sharing a database are assumed to agree
to well within a lease.

**Documents (RMD5-T02; RMD-DEF-004).** An ingestion holds a ``DocumentLease`` on its document
while it runs, renewed from a session of its own because the pipeline keeps one transaction open
until its end. The same passes run ``reconcile_documents``, which ends each ``PROCESSING``
document whose lease has lapsed -- or, with no lease, whose last update is older than
``ingestion_stale_after_seconds`` -- as ``failed`` with ``INGESTION_STOPPED_REASON``, after which
a reindex is accepted. Its chunks and its file are left as they are; the reindex replaces them.
"""

import asyncio
import contextlib
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final, Literal

from sqlalchemy import (
    ColumnElement,
    DateTime,
    Integer,
    and_,
    case,
    cast,
    func,
    literal,
    or_,
    select,
    update,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.errors import RUN_ABANDONED
from app.domain.enums import DocumentStatus, EventType, QueryRunStatus
from app.domain.models import Document, LLMCall, QueryRun
from app.repositories.event_repository import EventRepository
from app.repositories.query_repository import QueryRepository
from app.services.event_bus import EventBus
from app.services.event_emitter import EventEmitter
from app.services.run_registry import RunRegistry
from app.utils.time import utc_now

logger = logging.getLogger(__name__)

# This process's name as a run's owner. The pid tells processes on one host apart; the random part
# tells a restarted process from the one it replaced, which can reuse the pid.
INSTANCE_ID: Final = f"{os.getpid()}-{uuid.uuid4().hex[:12]}"

Lapse = Literal["lease", "deadline"]
# What a renewal found: the lease is still this process's; the run ended under this process (its
# own terminal write, or this process's reconciler); or another instance closed it.
Renewal = Literal["held", "ended", "taken"]

# Fixed text, never an exception string (the card's security rule).
LEASE_LAPSED_REASON: Final = (
    "This analysis was stopped because the process running it is gone and its lease lapsed. "
    "Ask the question again to run it."
)
DEADLINE_PASSED_REASON: Final = (
    "This analysis was stopped because it passed its maximum run time with no process running "
    "it. Ask the question again to run it."
)
OWNERLESS_CANCEL_MESSAGE: Final = "The run was cancelled. No process was running it any more."
# The card's fixed text for a document whose ingestion stopped (RMD5-T02).
INGESTION_STOPPED_REASON: Final = (
    "Processing stopped before it finished; reindex the document to try again"
)


def renew_interval_seconds(settings: Settings) -> float:
    """How often a job renews its lease: a third of the lease, so two renewals can fail in a row
    before the lease lapses."""
    return settings.query_run_lease_seconds / 3


def lapse_of(run: QueryRun, now: datetime, settings: Settings) -> Lapse | None:
    """Why no process owns ``run`` any more, or ``None`` while one may: a lapsed lease, or, for a
    row with no lease, a passed deadline."""
    if run.status is not QueryRunStatus.RUNNING:
        return None
    if run.lease_expires_at is not None:
        return "lease" if run.lease_expires_at < now else None
    deadline = run.created_at + timedelta(seconds=settings.query_run_deadline_seconds)
    return "deadline" if deadline < now else None


def _lapsed(now: datetime, settings: Settings) -> ColumnElement[bool]:
    """``lapse_of`` as a condition on the row."""
    cutoff = now - timedelta(seconds=settings.query_run_deadline_seconds)
    return or_(
        and_(QueryRun.lease_expires_at.is_not(None), QueryRun.lease_expires_at < now),
        and_(QueryRun.lease_expires_at.is_(None), QueryRun.created_at < cutoff),
    )


def _lease_end(now: datetime, settings: Settings) -> datetime:
    return now + timedelta(seconds=settings.query_run_lease_seconds)


async def claim_lease(
    session_factory: async_sessionmaker[AsyncSession],
    query_run_id: uuid.UUID,
    settings: Settings,
    *,
    instance: str = INSTANCE_ID,
) -> bool:
    """Names ``instance`` the run's owner and starts its lease. ``False`` when the run is no
    longer running (cancelled or closed before its job got here)."""
    async with session_factory() as session:
        claimed = await session.scalar(
            update(QueryRun)
            .where(QueryRun.id == query_run_id, QueryRun.status == QueryRunStatus.RUNNING)
            .values(owner_instance=instance, lease_expires_at=_lease_end(utc_now(), settings))
            .returning(QueryRun.id)
        )
        await session.commit()
        return claimed is not None


async def renew_lease(
    session_factory: async_sessionmaker[AsyncSession],
    query_run_id: uuid.UUID,
    settings: Settings,
    *,
    instance: str = INSTANCE_ID,
) -> Renewal:
    """Moves the lease forward while the run is still running and still ``instance``'s."""
    async with session_factory() as session:
        renewed = await session.scalar(
            update(QueryRun)
            .where(
                QueryRun.id == query_run_id,
                QueryRun.status == QueryRunStatus.RUNNING,
                QueryRun.owner_instance == instance,
            )
            .values(lease_expires_at=_lease_end(utc_now(), settings))
            .returning(QueryRun.id)
        )
        owner = (
            instance
            if renewed is not None
            else await session.scalar(
                select(QueryRun.owner_instance).where(QueryRun.id == query_run_id)
            )
        )
        await session.commit()
    if renewed is not None:
        return "held"
    return "ended" if owner == instance else "taken"


def _spent() -> ColumnElement[int]:
    """The run's metered tokens, as ``QueryRepository.total_tokens_for`` sums them (NULL when
    nothing was metered: unknown, never free)."""
    total = select(
        func.sum(
            func.coalesce(LLMCall.prompt_tokens, 0) + func.coalesce(LLMCall.completion_tokens, 0)
        )
    ).where(LLMCall.query_run_id == QueryRun.id)
    return total.scalar_subquery()


async def reconcile_once(
    session_factory: async_sessionmaker[AsyncSession],
    bus: EventBus,
    registry: RunRegistry,
    settings: Settings,
    *,
    now: datetime | None = None,
    instance: str = INSTANCE_ID,
) -> list[uuid.UUID]:
    """Closes every ``RUNNING`` run no process owns; returns the ids this pass closed.

    A run this process still holds a task for is left alone whatever its lease says: a renewal
    that failed during an outage leaves a lapsed lease on a live run. The closing instance is
    recorded as the run's last owner, which is how the original owner's renewal learns that its
    run was taken.
    """
    at = now if now is not None else utc_now()
    async with session_factory() as session:
        rows = (
            await session.execute(
                select(QueryRun.id, QueryRun.owner_instance)
                .where(QueryRun.status == QueryRunStatus.RUNNING, _lapsed(at, settings))
                .order_by(QueryRun.id)
                .with_for_update(skip_locked=True)
            )
        ).all()
        owners = {row.id: row.owner_instance for row in rows if not registry.is_running(row.id)}
        if not owners:
            await session.commit()
            return []
        closed = (
            await session.execute(
                update(QueryRun)
                .where(QueryRun.id.in_(owners), QueryRun.status == QueryRunStatus.RUNNING)
                .values(
                    status=QueryRunStatus.FAILED,
                    error_message=case(
                        (QueryRun.lease_expires_at.is_not(None), LEASE_LAPSED_REASON),
                        else_=DEADLINE_PASSED_REASON,
                    ),
                    latency_ms=func.greatest(
                        0,
                        cast(
                            func.extract(
                                "epoch", literal(at, DateTime(timezone=True)) - QueryRun.created_at
                            )
                            * 1000,
                            Integer,
                        ),
                    ),
                    token_usage_estimate=_spent(),
                    owner_instance=instance,
                )
                .returning(QueryRun.id, QueryRun.error_message)
                .execution_options(synchronize_session=False)
            )
        ).all()
        await session.commit()
        for run_id, message in closed:
            lapse: Lapse = "lease" if message == LEASE_LAPSED_REASON else "deadline"
            logger.warning(
                "Closed query run %s: no process owns it (its %s lapsed; last owner %s)",
                run_id,
                lapse,
                owners[run_id],
            )
            emitter = EventEmitter(session, EventRepository(session), bus, run_id)
            await emitter.emit(
                EventType.WORKFLOW_FAILED,
                message,
                payload={"code": RUN_ABANDONED, "lapse": lapse},
            )
    return [run_id for run_id, _ in closed]


async def cancel_ownerless(
    session: AsyncSession,
    bus: EventBus,
    query_run_id: uuid.UUID,
    settings: Settings,
    *,
    instance: str = INSTANCE_ID,
) -> QueryRun | None:
    """Ends ``cancelled`` a run no process owns, at once; ``None`` when a process may still own
    it (its lease is current) or it is no longer running. The row is locked before it is judged,
    as the reconciler and the job's terminal writers lock it."""
    run = await session.get(QueryRun, query_run_id, with_for_update=True, populate_existing=True)
    if run is None:
        return None
    now = utc_now()
    lapse = lapse_of(run, now, settings)
    if lapse is None:
        await session.commit()
        return None
    run.token_usage_estimate = await QueryRepository(session).total_tokens_for(query_run_id)
    run.status = QueryRunStatus.CANCELLED
    run.latency_ms = max(0, int((now - run.created_at).total_seconds() * 1000))
    run.owner_instance = instance
    await session.commit()
    logger.warning(
        "Cancelled query run %s, which no process owned (its %s lapsed)",
        query_run_id,
        lapse,
    )
    emitter = EventEmitter(session, EventRepository(session), bus, query_run_id)
    await emitter.emit(
        EventType.WORKFLOW_CANCELLED,
        OWNERLESS_CANCEL_MESSAGE,
        payload={"code": RUN_ABANDONED, "lapse": lapse},
    )
    return run


@dataclass(frozen=True)
class DocumentLease:
    """An ingestion's claim on its document (RMD5-T02): ``seconds`` long, renewed every third of
    that from ``session_factory``'s own sessions while the ingestion runs."""

    session_factory: async_sessionmaker[AsyncSession]
    seconds: int

    def end(self, now: datetime) -> datetime:
        return now + timedelta(seconds=self.seconds)

    async def renew(self, document_id: uuid.UUID) -> bool:
        """Moves the lease forward; ``False`` once the document is no longer ``PROCESSING``."""
        async with self.session_factory() as session:
            renewed = await session.scalar(
                update(Document)
                .where(Document.id == document_id, Document.status == DocumentStatus.PROCESSING)
                .values(lease_expires_at=self.end(utc_now()))
                .returning(Document.id)
            )
            await session.commit()
            return renewed is not None

    async def hold(self, document_id: uuid.UUID) -> None:
        """Renews until cancelled, or until the document is no longer ``PROCESSING``. A renewal
        that fails is logged and tried again at the next tick; the lease may lapse meanwhile."""
        interval = self.seconds / 3
        while True:
            await asyncio.sleep(interval)
            try:
                if not await self.renew(document_id):
                    return
            except Exception:
                logger.warning(
                    "Could not renew the ingestion lease of document %s; trying again in %.1fs",
                    document_id,
                    interval,
                    exc_info=True,
                )


async def reconcile_documents(
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    *,
    now: datetime | None = None,
) -> list[uuid.UUID]:
    """Ends ``failed`` every ``PROCESSING`` document no ingestion holds; returns their ids.

    Claimed ``FOR UPDATE SKIP LOCKED`` and written in one ``UPDATE``, as runs are, so two
    reconcilers never both write a document and never wait on each other.
    """
    at = now if now is not None else utc_now()
    stale = at - timedelta(seconds=settings.ingestion_stale_after_seconds)
    async with session_factory() as session:
        ids = list(
            await session.scalars(
                select(Document.id)
                .where(
                    Document.status == DocumentStatus.PROCESSING,
                    or_(
                        and_(
                            Document.lease_expires_at.is_not(None), Document.lease_expires_at < at
                        ),
                        and_(Document.lease_expires_at.is_(None), Document.updated_at < stale),
                    ),
                )
                .order_by(Document.id)
                .with_for_update(skip_locked=True)
            )
        )
        if ids:
            await session.execute(
                update(Document)
                .where(Document.id.in_(ids))
                .values(
                    status=DocumentStatus.FAILED,
                    error_message=INGESTION_STOPPED_REASON,
                    updated_at=at,
                )
                .execution_options(synchronize_session=False)
            )
        await session.commit()
    for document_id in ids:
        logger.warning("Closed document %s: its ingestion stopped before it finished", document_id)
    return ids


class RunReconciler:
    """The reconciler's place in the application's lifespan: one pass at startup, then one every
    ``run_reconcile_interval_seconds`` until ``stop``. Each pass closes runs, then documents
    (RMD5-T02). A part of a pass that fails is logged with its exception, the other part still
    runs, and the next pass runs on schedule; the loop itself never ends on an error."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        bus: EventBus,
        registry: RunRegistry,
        settings: Settings,
    ) -> None:
        self._session_factory = session_factory
        self._bus = bus
        self._registry = registry
        self._settings = settings
        self._task: asyncio.Task[None] | None = None

    async def reconcile(self) -> list[uuid.UUID]:
        return await reconcile_once(
            self._session_factory, self._bus, self._registry, self._settings
        )

    async def start(self) -> None:
        """The startup pass, then the periodic loop. A failed startup pass is logged like any
        other: the loop retries it one interval later."""
        await self._logged_pass()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self._settings.run_reconcile_interval_seconds)
            await self._logged_pass()

    async def reconcile_documents(self) -> list[uuid.UUID]:
        return await reconcile_documents(self._session_factory, self._settings)

    async def _logged_pass(self) -> None:
        interval = self._settings.run_reconcile_interval_seconds
        try:
            await self.reconcile()
        except Exception:
            logger.exception("Run reconciliation failed; the next pass runs in %ss", interval)
        try:
            await self.reconcile_documents()
        except Exception:
            logger.exception("Document reconciliation failed; the next pass runs in %ss", interval)
