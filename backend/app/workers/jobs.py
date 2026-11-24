"""Background jobs. Each job owns a fresh session so request lifecycles
and job lifecycles never share transactional state. The Qdrant service and
LLM provider are shared/stateless and passed in from application resources.

``run_query_job`` (Phase 5) runs the actual workflow as an *inner* ``asyncio.Task`` registered
with a ``RunRegistry``, rather than running it inline. That inner task is what a
``POST /queries/{id}/cancel`` request cancels; ``run_query_job`` itself keeps being scheduled the
same way it always was (as a Starlette ``BackgroundTasks`` entry -- see ``routes/queries.py``), so
callers and the integration-test scheduling semantics they depend on see no difference. See
``_execute_workflow`` and ``_mark_cancelled`` below for the split.
"""

import asyncio
import logging
import time
import uuid
from typing import cast

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.base import JsonAgentRunner
from app.agents.orchestrator import UNEXPECTED_FAILURE_MESSAGE, Orchestrator
from app.core.config import LLMProviderName, Settings
from app.core.errors import CognifyError
from app.core.logging import bound_context
from app.domain.enums import DomainProfile, EventType, QueryRunStatus
from app.domain.models import QueryRun
from app.integrations.llm.base import ChatProvider, EmbeddingProvider
from app.integrations.llm.router import LLMRouter
from app.integrations.qdrant.client import QdrantService
from app.pharma.action_classifier import CLASSIFIER_COMPONENT
from app.pharma.ingestion import ControlledIngestion
from app.pharma.recorder import DatabaseRecorder
from app.pharma.workflow import PharmaOrchestrator, PharmaRunInput, RepositoryMetadataLoader
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.event_repository import EventRepository
from app.repositories.metering_repository import EmbeddingMeter, EmbeddingScope
from app.repositories.pharma_repository import PharmaRepository
from app.repositories.project_repository import ProjectRepository
from app.repositories.query_repository import QueryRepository
from app.schemas.pharma import IncidentInput, PharmaFilters
from app.services import run_reconciler
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.event_bus import TERMINAL_QUERY_RUN_STATUSES, EventBus
from app.services.event_emitter import EventEmitter
from app.services.ingestion_service import IngestionService
from app.services.lexical_search_service import LexicalSearchService
from app.services.reranking_service import RerankingService
from app.services.retrieval_service import RetrievalService
from app.services.run_registry import RunRegistry
from app.utils.files import FileStorage

logger = logging.getLogger(__name__)

# How long the orphan guard below keeps waiting for a cancelled workflow to unwind. Bounded so
# a wedged provider call cannot hold up process teardown indefinitely.
_DRAIN_TIMEOUT_SECONDS = 5.0

# The failed outcome's write is tried this many times, with these waits between the tries
# (RE2E-DEF-038): about 7.5 s in all, so a short database outage no longer leaves the run
# `RUNNING`. Past the last try the job gives up and logs it; the run's lease then lapses and the
# reconciler closes it (`app.services.run_reconciler`).
_TERMINAL_WRITE_ATTEMPTS = 5
_TERMINAL_WRITE_DELAYS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)


class _RunDeadlineExceededError(Exception):
    """A workflow that outlived ``settings.query_run_deadline_seconds``.

    Private to this module and never raised through HTTP. It exists so ``run_query_job`` can tell
    its own wall clock firing apart from an arbitrary ``TimeoutError`` surfacing from, say, a
    database driver: the first is a policy decision with a caller-visible explanation, the second
    is an unexplained failure that must stay loud.
    """


def _deadline_message(seconds: int) -> str:
    """The user-visible ``error_message`` for a run stopped by its own deadline."""
    return f"This analysis exceeded its maximum run time of {seconds}s and was stopped."


def embedding_identity(settings: Settings) -> str:
    """Names the vector space this deployment's embeddings belong to.

    ``IngestionService`` reuses a stored vector only while this string is unchanged, so it has
    to name everything that decides what a vector *means*: which vendor produced it, with
    which model, at which dimension. ``LLMRouter.embeddings`` resolves ``FAKE`` ahead of
    ``LLM_EMBEDDING_PROVIDER``, and this mirrors that resolution rather than reading
    ``LLM_PROVIDER`` directly, so a deployment leaving fake mode re-embeds instead of keeping
    vectors no vendor ever produced.

    A base-URL override is deliberately not part of it: it can carry a credential, and the
    lever for "same model name, different host" is ``QDRANT_COLLECTION_NAME``, which is also
    part of the identity a point id is derived from.
    """
    provider = (
        LLMProviderName.FAKE
        if settings.llm_provider is LLMProviderName.FAKE
        else settings.llm_embedding_provider
    )
    return f"{provider.value}|{settings.llm_embedding_model}|{settings.llm_embedding_dimensions}"


def build_ingestion_service(
    session: AsyncSession,
    settings: Settings,
    qdrant: QdrantService,
    provider: EmbeddingProvider,
    controlled: ControlledIngestion | None = None,
    lease: run_reconciler.DocumentLease | None = None,
    meter: EmbeddingMeter | None = None,
) -> IngestionService:
    return IngestionService(
        session=session,
        documents=DocumentRepository(session),
        chunks=ChunkRepository(session),
        storage=FileStorage(settings.upload_dir),
        chunking=ChunkingService(),
        embeddings=EmbeddingService(provider, settings.llm_embedding_dimensions, meter=meter),
        qdrant=qdrant,
        embedding_identity=embedding_identity(settings),
        controlled=controlled,
        lease=lease,
    )


async def _controlled_ingestion(
    session: AsyncSession, document_id: uuid.UUID
) -> ControlledIngestion | None:
    """The pharmaceutical collaborator, for a pharmaceutical project's document only."""
    document = await DocumentRepository(session).get(document_id)
    if document is None:
        return None
    pharma = PharmaRepository(session)
    if await pharma.project_profile(document.project_id) is not DomainProfile.PHARMA_MANUFACTURING:
        return None
    return ControlledIngestion(pharma)


async def run_ingestion_job(
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    qdrant: QdrantService,
    provider: EmbeddingProvider,
    document_id: uuid.UUID,
) -> None:
    # RMD5-T02 (DV-RMD-32): the lease is renewed from sessions of its own, since the pipeline
    # holds this session's transaction open until it ends.
    lease = run_reconciler.DocumentLease(session_factory, settings.ingestion_lease_seconds)
    async with session_factory() as session:
        controlled = await _controlled_ingestion(session, document_id)
        # RE2E-DEF-029 (RMD9-T04): each embedding call is metered against this document.
        document = await DocumentRepository(session).get(document_id)
        meter = (
            EmbeddingMeter(
                session_factory, EmbeddingScope(document.project_id, document_id=document_id)
            )
            if document is not None
            else None
        )
        service = build_ingestion_service(
            session, settings, qdrant, provider, controlled, lease, meter
        )
        try:
            await service.ingest(document_id)
        finally:
            if meter is not None:
                await meter.flush()


def build_orchestrator(
    session: AsyncSession,
    settings: Settings,
    qdrant: QdrantService,
    provider: EmbeddingProvider,
    bus: EventBus,
    query_run_id: uuid.UUID,
    *,
    chat: ChatProvider | None = None,
    meter: EmbeddingMeter | None = None,
) -> Orchestrator:
    """``provider`` always drives retrieval's embeddings. ``chat`` — keyword-only, defaulting to
    ``provider`` — drives the agents' chat calls.

    Every existing (positional-only) caller passes one object — a ``FakeProvider`` in tests, an
    ``OpenAIProvider`` before the router existed — that is both an ``EmbeddingProvider`` and a
    ``ChatProvider``; the ``cast`` below documents that as the invariant those call sites rely on,
    rather than widening ``provider``'s declared type back to the full ``LLMProvider`` and losing
    the ability to pass a real embeddings-only client. ``run_query_job`` below is the one caller
    that does not rely on it: it always resolves and passes its own ``chat`` explicitly, so
    ``provider`` there can be — and is — the router's dedicated embedding client, while retrieval
    keeps using it and chat uses a different, independently-resolved provider.
    """
    chat_provider = chat if chat is not None else cast(ChatProvider, provider)
    chunks = ChunkRepository(session)
    retrieval = RetrievalService(
        qdrant=qdrant,
        embeddings=EmbeddingService(provider, settings.llm_embedding_dimensions, meter=meter),
        lexical=LexicalSearchService(chunks),
        chunks=chunks,
    )
    emitter = EventEmitter(session, EventRepository(session), bus, query_run_id)
    return Orchestrator(
        session=session,
        queries=QueryRepository(session),
        emitter=emitter,
        provider=chat_provider,
        retrieval=retrieval,
        reranking=RerankingService(chat_provider, settings.final_context_size),
        rerank_expanded_union=settings.rerank_expanded_union,
    )


async def build_pharma_orchestrator(
    session: AsyncSession,
    settings: Settings,
    qdrant: QdrantService,
    provider: EmbeddingProvider,
    bus: EventBus,
    query_run_id: uuid.UUID,
    *,
    chat: ChatProvider,
    meter: EmbeddingMeter | None = None,
) -> PharmaOrchestrator:
    """``PharmaOrchestrator`` for a pharmaceutical project's run (PDP8-T03, PH-D-012), built from
    the same services ``build_orchestrator`` builds, with the database recorder and the run's
    input read from the record ``QueryService.create`` opened. A run with no record still runs
    as a procedure question, and its recorder then refuses it (fail closed)."""
    chunks = ChunkRepository(session)
    records = PharmaRepository(session)
    record = await records.run_record(query_run_id)
    pharma_input = (
        PharmaRunInput(
            intent=record.intent,
            incident=(
                IncidentInput.model_validate(record.incident)
                if record.incident is not None
                else None
            ),
            filters=PharmaFilters.model_validate(record.filters),
        )
        if record is not None
        else PharmaRunInput()
    )
    return PharmaOrchestrator(
        session,
        QueryRepository(session),
        EventEmitter(session, EventRepository(session), bus, query_run_id),
        chat,
        RetrievalService(
            qdrant=qdrant,
            embeddings=EmbeddingService(provider, settings.llm_embedding_dimensions, meter=meter),
            lexical=LexicalSearchService(chunks),
            chunks=chunks,
        ),
        RerankingService(chat, settings.final_context_size),
        pharma_input=pharma_input,
        recorder=DatabaseRecorder(records),
        infos_loader=RepositoryMetadataLoader(records, chunks),
        rerank_expanded_union=settings.rerank_expanded_union,
        # PLV4-T04: every production run has the secondary detector of regulated acts.
        action_classifier=JsonAgentRunner(chat, CLASSIFIER_COMPONENT),
    )


async def run_query_job(
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    qdrant: QdrantService,
    router: LLMRouter,
    bus: EventBus,
    registry: RunRegistry,
    query_run_id: uuid.UUID,
) -> None:
    """Scheduled exactly as before (see the module docstring); the only change is that the
    workflow itself now runs as an inner, independently-cancellable task.

    ``asyncio.wait([task])`` -- not a bare ``await task`` -- is deliberate: a bare await raises
    ``CancelledError`` identically whether *this* coroutine was cancelled (e.g. application
    shutdown, while Starlette is tearing down background tasks) or the *inner* task was
    cancelled by a ``/cancel`` request, so wrapping it in ``try/except CancelledError`` could not
    tell those apart -- and would risk swallowing a genuine shutdown cancellation. ``asyncio.wait``
    never raises or re-propagates the inner task's outcome, so that outcome is interrogated
    explicitly below via ``task.cancelled()``/``task.exception()`` once it is known to be this
    task's own outcome, and a cancellation of *this* coroutine (which does not resolve
    ``asyncio.wait``'s awaited task) instead propagates straight out of the ``try`` block, past
    the code below, exactly like any other cancellation would -- except that ``asyncio.wait``
    itself does *not* cancel the futures it was waiting on when it is the one cancelled, so a bare
    ``try/finally`` around it would orphan ``task``: still running, still billing, and now
    unreachable by any future cancel since the ``finally`` below is about to discard its registry
    entry. The nested ``except BaseException`` exists solely to close that gap: on the way out, it
    cancels ``task`` itself and waits for it to actually finish unwinding before letting this
    coroutine's own cancellation continue propagating -- see ``_drain`` for why that wait has to
    absorb further cancellations to actually hold.
    """
    started = time.monotonic()
    # RE2E-DEF-013 (RMD9-T01): a run that completed used to log nothing at all, so the server log
    # could not show it; it now logs its start and its end with its id, under the request's
    # correlation id that scheduled it.
    with bound_context(query_run_id=query_run_id):
        logger.info("Query run %s started", query_run_id, extra={"event_type": "run_started"})
    task: asyncio.Task[None] = asyncio.create_task(
        _execute_workflow(session_factory, settings, qdrant, router, bus, query_run_id)
    )
    registry.register(query_run_id, task)
    # RMD5-T01: the run's lease is held from here until the terminal write below is done, so the
    # reconciler leaves the run alone for as long as this process still means to finish it.
    lease = asyncio.create_task(_hold_lease(session_factory, settings, registry, query_run_id))
    try:
        try:
            try:
                await asyncio.wait([task])
            except BaseException:
                task.cancel()
                await _drain(task, query_run_id)
                raise
        finally:
            registry.discard(query_run_id, task)

        # `consume_cancel_requested` must run (and its result be honoured) even when
        # `task.cancelled()` is `False`: a `CancelledError` that arrives while a DB statement is
        # in flight can be *replaced* by an ordinary exception raised while
        # `AsyncSession.__aexit__` rolls back (see `RunRegistry`'s docstring) -- at that point
        # `task.cancelled()` reports `False` even though a cancel genuinely was requested and
        # delivered. Without tracking that separately, the run would sit at `RUNNING` forever:
        # no terminal event, and the SSE stream never closes for a client this process already
        # answered 202.
        cancel_requested = registry.consume_cancel_requested(query_run_id)
        if task.cancelled():
            await _mark_cancelled(session_factory, bus, query_run_id, started)
        else:
            exc = task.exception()
            if cancel_requested:
                await _mark_cancelled(session_factory, bus, query_run_id, started)
            elif isinstance(exc, _RunDeadlineExceededError):
                # Recorded but *not* re-raised, unlike the branch below. This is the run's own
                # wall clock firing -- a deliberate policy outcome with a message the caller can
                # read off the row -- not an unexplained failure, and re-raising it would surface
                # every deadline as an uncaught background-task exception in the server log.
                logger.warning(
                    "Query run %s exceeded its %ss deadline and was stopped",
                    query_run_id,
                    settings.query_run_deadline_seconds,
                )
                await _mark_failed(
                    session_factory,
                    bus,
                    query_run_id,
                    started,
                    exc,
                    message=_deadline_message(settings.query_run_deadline_seconds),
                )
            elif exc is not None:
                # `Orchestrator.run` catches `CognifyError` and `Exception` internally and never
                # re-raises (it always resolves via `_fail`), so the only way `task` lands here,
                # not cancelled, with no cancel on record, and still holding an exception, is a
                # failure outside that try -- e.g. the session/repository calls at the top of
                # `_execute_workflow`, a provider client that fails to resolve credentials, or
                # `_fail` itself meeting a database outage. Re-raising (rather than swallowing)
                # keeps it visible the same way an uncaught background-task exception always was
                # before this refactor -- but on its own it left the run at `RUNNING` forever,
                # since no writer covers this path: no terminal event, the SSE stream never
                # closing for a client this process already answered 202, and `POST /cancel`
                # answering 409 so the user cannot even clear it. Record the terminal outcome
                # first, then re-raise.
                await _mark_failed(session_factory, bus, query_run_id, started, exc)
                raise exc
    finally:
        lease.cancel()
        await _drain(lease, query_run_id)
        with bound_context(query_run_id=query_run_id):
            logger.info(
                "Query run %s finished",
                query_run_id,
                extra={
                    "event_type": "run_finished",
                    "latency_ms": int((time.monotonic() - started) * 1000),
                },
            )


async def _hold_lease(
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    registry: RunRegistry,
    query_run_id: uuid.UUID,
) -> None:
    """Claims the run's lease for this process, then renews it every third of its length until
    ``run_query_job`` cancels this task (RMD5-T01).

    A claim or renewal that fails -- the database is down -- is logged and tried again at the next
    tick; the lease may lapse meanwhile, which is what lets the reconciler close a run whose job
    truly is gone. A renewal that finds the run closed by another instance cancels the workflow
    through the registry, so this job writes no second terminal state; one that finds the run
    ended under this process (its own terminal write) stops renewing.
    """
    interval = run_reconciler.renew_interval_seconds(settings)
    claimed = False
    while True:
        try:
            if not claimed:
                if not await run_reconciler.claim_lease(session_factory, query_run_id, settings):
                    return
                claimed = True
            else:
                renewal = await run_reconciler.renew_lease(session_factory, query_run_id, settings)
                if renewal == "taken":
                    logger.warning(
                        "Query run %s was closed by another process; stopping its workflow",
                        query_run_id,
                    )
                    registry.cancel(query_run_id)
                    return
                if renewal == "ended":
                    return
        except Exception:
            logger.warning(
                "Could not %s the lease of query run %s; trying again in %.1fs",
                "renew" if claimed else "claim",
                query_run_id,
                interval,
                exc_info=True,
            )
        await asyncio.sleep(interval)


async def _drain(task: asyncio.Task[None], query_run_id: uuid.UUID) -> None:
    """Waits for an already-cancelled ``task`` to finish unwinding, absorbing cancellations
    delivered to *this* coroutine while it waits.

    A plain ``await asyncio.wait([task])`` is not enough, because this runs precisely when the
    current task is being cancelled. Until RMD9-T01 ``app.main`` installed ``BaseHTTPMiddleware``,
    so every ``BackgroundTasks`` entry ran inside an anyio cancel scope, and anyio re-delivers its
    cancellation on every loop iteration until the task leaves the scope; a host that still runs
    background tasks in such a scope does the same. The *second* delivery would abort the wait
    and ``run_query_job`` would return with the workflow still mid-unwind:
    an ``AsyncSession.__aexit__`` rollback racing the lifespan's ``engine.dispose()`` and the
    router's closed HTTP pools, its registry entry already discarded. Absorbing the extra
    deliveries is safe because the caller re-raises its own cancellation the moment this
    returns; the deadline keeps a wedged provider call from blocking teardown forever.
    """
    deadline = time.monotonic() + _DRAIN_TIMEOUT_SECONDS
    while not task.done():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning(
                "A task of query run %s did not unwind within %ss; abandoning it",
                query_run_id,
                _DRAIN_TIMEOUT_SECONDS,
            )
            return
        try:
            await asyncio.wait([task], timeout=remaining)
        except asyncio.CancelledError:
            continue


async def _execute_workflow(
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    qdrant: QdrantService,
    router: LLMRouter,
    bus: EventBus,
    query_run_id: uuid.UUID,
) -> None:
    """The workflow body that used to be ``run_query_job`` itself, plus the run's wall clock.

    ``asyncio.timeout`` is the only deadline a run has. The per-HTTP-request provider timeouts
    (60-180s, ``openai_provider.py``) bound one call, and a run makes 7-20 of them, each retried
    up to three times -- so they bound nothing about the run as a whole, and a workflow whose
    provider simply never answers previously sat at ``RUNNING`` forever: no terminal event, an
    SSE stream that never closes for a client already answered 202, and a vendor bill still open.

    On expiry the scope cancels this task; ``CancelledError`` is a ``BaseException``, so it
    passes straight through ``Orchestrator.run``'s ``except Exception`` catch-all (the property
    ``test_cancellation`` already pins) and reaches ``__aexit__``, which converts it to
    ``TimeoutError``. That is re-raised as ``_RunDeadlineExceededError`` so ``run_query_job`` can
    record the right outcome; ``Timeout.expired()`` distinguishes it from a ``TimeoutError``
    raised by anything else inside the block.
    """
    deadline = asyncio.timeout(settings.query_run_deadline_seconds)
    try:
        async with deadline, session_factory() as session:
            queries = QueryRepository(session)
            query_run = await queries.get(query_run_id)
            if query_run is None:  # pragma: no cover - created just before scheduling
                logger.error("Query run %s vanished before the workflow started", query_run_id)
                return
            project = await ProjectRepository(session).get(query_run.project_id)
            domain = project.domain if project is not None else ""
            objective = project.objective if project is not None else ""
            # `QueryService.create` already resolved and persisted `resolved_provider` for a
            # caller-selected model option; for a legacy request (no `model_option_id`) it is
            # left `NULL`, and `chat_for_provider` maps both `None` and an unrecognised value to
            # the router's default provider, so every row -- old or new -- keeps running.
            chat = router.chat_for_provider(query_run.resolved_provider)
            embeddings = router.embeddings()
            # RE2E-DEF-029 (RMD9-T04): each retrieval embedding is metered against this run.
            meter = EmbeddingMeter(
                session_factory, EmbeddingScope(query_run.project_id, query_run_id=query_run_id)
            )
            # PH-D-012: the project's profile selects the workflow; a General project's run is
            # built exactly as before.
            orchestrator: Orchestrator
            if project is not None and project.domain_profile is DomainProfile.PHARMA_MANUFACTURING:
                orchestrator = await build_pharma_orchestrator(
                    session, settings, qdrant, embeddings, bus, query_run_id, chat=chat, meter=meter
                )
            else:
                orchestrator = build_orchestrator(
                    session, settings, qdrant, embeddings, bus, query_run_id, chat=chat, meter=meter
                )
            # Bound here rather than at the top of the function: `project_id` is only known once
            # the run row is loaded, and binding a half-populated context would put records on
            # the wire that claim a run id and omit the project. The `finally` inside
            # `bound_context` is what makes this safe under the deadline above -- expiry cancels
            # this task, and a `CancelledError` would skip any except-based cleanup.
            with bound_context(query_run_id=query_run_id, project_id=query_run.project_id):
                try:
                    await orchestrator.run(query_run, domain, objective)
                finally:
                    await meter.flush()
    except TimeoutError as exc:
        # The conversion happens in `Timeout.__aexit__`, so this `except` has to sit *outside*
        # the `async with` to see it at all.
        if not deadline.expired():
            raise
        raise _RunDeadlineExceededError(
            _deadline_message(settings.query_run_deadline_seconds)
        ) from exc


async def _mark_cancelled(
    session_factory: async_sessionmaker[AsyncSession],
    bus: EventBus,
    query_run_id: uuid.UUID,
    started: float,
) -> None:
    """Writes the cancelled outcome from a fresh session.

    The inner task's own session was interrupted by ``CancelledError`` at whatever ``await`` was
    in flight -- possibly mid-transaction -- so it may be left unusable; reusing it here would
    risk operating on a broken connection. A new session from the factory has none of that
    history.

    The re-read below applies the same terminal-state guard every other writer uses
    (``TERMINAL_QUERY_RUN_STATUSES``): ``Orchestrator._finish`` commits ``status=COMPLETED`` and
    only *then* awaits ``emit(ANSWER_COMPLETED)`` -- another real INSERT plus COMMIT -- and
    ``_fail`` has the same shape for ``FAILED``. For that whole window ``task.done()`` is still
    ``False``, so a ``/cancel`` request racing in right then still succeeds and a
    ``CancelledError`` can land on that trailing emit, *after* the terminal status is already
    durably committed. Without this guard, this function would overwrite that committed
    outcome -- answer, citations, evaluation, provenance all already persisted -- with
    ``CANCELLED``, clobbering ``latency_ms`` and leaving two terminal events in the run's
    history. (A run left terminal-but-eventless by that race still ends the SSE stream: the
    stream reads the run's status too -- see ``app.api.routes.events``.)

    The row is read under its lock (RMD5-T01): the reconciler takes the same lock, so the two
    cannot both find the run ``RUNNING`` and both write a terminal state.

    It is tried as ``_mark_failed``'s write is (RMD-DEF-036): a cancel that met a short database
    outage used to be lost, and the reconciler then closed the run as failed though it had been
    cancelled. Past the last try the run is left to the reconciler, as for a failure.
    """
    for attempt in range(1, _TERMINAL_WRITE_ATTEMPTS + 1):
        try:
            await _write_cancelled(session_factory, bus, query_run_id, started)
            return
        except Exception:
            if attempt == _TERMINAL_WRITE_ATTEMPTS:
                logger.exception(
                    "Could not record the cancelled outcome for query run %s in %s tries; the "
                    "reconciler closes it once its lease lapses",
                    query_run_id,
                    _TERMINAL_WRITE_ATTEMPTS,
                )
                return
            delay = _TERMINAL_WRITE_DELAYS[attempt - 1]
            logger.warning(
                "Could not record the cancelled outcome for query run %s (try %s of %s); trying "
                "again in %ss",
                query_run_id,
                attempt,
                _TERMINAL_WRITE_ATTEMPTS,
                delay,
                exc_info=True,
            )
            await asyncio.sleep(delay)


async def _write_cancelled(
    session_factory: async_sessionmaker[AsyncSession],
    bus: EventBus,
    query_run_id: uuid.UUID,
    started: float,
) -> None:
    """One try of ``_mark_cancelled``'s write."""
    async with session_factory() as session:
        queries = QueryRepository(session)
        query_run = await _locked(session, query_run_id)
        if query_run is None:  # pragma: no cover - vanished mid-flight
            return
        if query_run.status in TERMINAL_QUERY_RUN_STATUSES:
            return
        # L-12: spend on every terminal path, not only on success. The metering rows were
        # committed one per attempt as the run proceeded, so they survive the inner task's
        # broken session -- which is exactly why the total is read back from the table here
        # rather than carried in memory from an orchestrator that no longer exists.
        query_run.token_usage_estimate = await queries.total_tokens_for(query_run_id)
        query_run.status = QueryRunStatus.CANCELLED
        query_run.latency_ms = int((time.monotonic() - started) * 1000)
        await session.commit()
        emitter = EventEmitter(session, EventRepository(session), bus, query_run_id)
        await emitter.emit(EventType.WORKFLOW_CANCELLED, "The run was cancelled.")


async def _mark_failed(
    session_factory: async_sessionmaker[AsyncSession],
    bus: EventBus,
    query_run_id: uuid.UUID,
    started: float,
    exc: BaseException,
    *,
    message: str | None = None,
) -> None:
    """Writes the failed outcome for an exception raised *outside* ``Orchestrator.run``.

    ``message`` overrides the recorded ``error_message`` for a failure the caller deserves a
    specific explanation of -- today, a run stopped by its own deadline. Without it every such
    run would read ``UNEXPECTED_FAILURE_MESSAGE``, which is the one thing a deadline is not.

    Same shape as ``_mark_cancelled`` above, for the same reasons: a fresh session, because the
    inner task's own one may be unusable, and the shared terminal-state guard, because the row
    may already have been resolved by a writer that did get a turn.

    Never raises. The exception being recorded here is the one the caller is about to re-raise,
    and the likeliest cause of a failure *in* this function -- an unreachable database -- is
    also a likely cause of ``exc`` itself; losing the original to a secondary error would trade
    a visible failure for a confusing one.

    So it tries again instead (RE2E-DEF-038): ``_TERMINAL_WRITE_ATTEMPTS`` times, waiting
    ``_TERMINAL_WRITE_DELAYS`` between them, each try from a fresh session. The second failure
    used to be logged and dropped, leaving the run ``RUNNING`` for good. Past the last try it is
    logged as an error and left to the reconciler, which closes the run once its lease lapses.
    """
    if message is None:
        message = exc.message if isinstance(exc, CognifyError) else UNEXPECTED_FAILURE_MESSAGE
    for attempt in range(1, _TERMINAL_WRITE_ATTEMPTS + 1):
        try:
            await _write_failed(session_factory, bus, query_run_id, started, message)
            return
        except Exception:
            if attempt == _TERMINAL_WRITE_ATTEMPTS:
                logger.exception(
                    "Could not record the failed outcome for query run %s in %s tries; the "
                    "reconciler closes it once its lease lapses. Re-raising the original failure "
                    "regardless",
                    query_run_id,
                    _TERMINAL_WRITE_ATTEMPTS,
                )
                return
            delay = _TERMINAL_WRITE_DELAYS[attempt - 1]
            logger.warning(
                "Could not record the failed outcome for query run %s (try %s of %s); trying "
                "again in %ss",
                query_run_id,
                attempt,
                _TERMINAL_WRITE_ATTEMPTS,
                delay,
                exc_info=True,
            )
            await asyncio.sleep(delay)


async def _write_failed(
    session_factory: async_sessionmaker[AsyncSession],
    bus: EventBus,
    query_run_id: uuid.UUID,
    started: float,
    message: str,
) -> None:
    """One try of ``_mark_failed``'s write, under the row's lock like ``_mark_cancelled``."""
    async with session_factory() as session:
        queries = QueryRepository(session)
        query_run = await _locked(session, query_run_id)
        if query_run is None:
            return
        if query_run.status in TERMINAL_QUERY_RUN_STATUSES:
            return
        # L-12, same reasoning as `_mark_cancelled`: a run stopped by its deadline has usually
        # spent the most, and recorded NULL.
        query_run.token_usage_estimate = await queries.total_tokens_for(query_run_id)
        query_run.status = QueryRunStatus.FAILED
        query_run.error_message = message
        query_run.latency_ms = int((time.monotonic() - started) * 1000)
        await session.commit()
        emitter = EventEmitter(session, EventRepository(session), bus, query_run_id)
        await emitter.emit(EventType.WORKFLOW_FAILED, message)


async def _locked(session: AsyncSession, query_run_id: uuid.UUID) -> QueryRun | None:
    """The run's row, read fresh and locked until the session commits (RMD5-T01)."""
    return await session.get(QueryRun, query_run_id, with_for_update=True, populate_existing=True)
