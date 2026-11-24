"""Query lifecycle (FR-CHAT-001): create a QueryRun, expose results.

The orchestrator runs as a background task; this service only creates the
running QueryRun and reads results back. A query is rejected if the project
has no indexed content to retrieve from.
"""

import logging
import uuid
from collections.abc import Sequence
from datetime import timedelta
from typing import Protocol, cast

from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import (
    CognifyError,
    ConflictError,
    InvalidModelOptionError,
    InvalidRoutingProfileError,
    ModelOptionUnavailableError,
    PharmaInputNotAllowedError,
    ProjectNotFoundError,
    QueryRunNotCancellableError,
    QueryRunNotFoundError,
    RoutingProfileUnavailableError,
)
from app.core.logging import correlation_id_var
from app.domain.enums import AgentName, AgentRunStatus, DomainProfile, QueryRunStatus
from app.domain.models import (
    AgentRun,
    Citation,
    EvaluationResult,
    PharmaRunRecord,
    Project,
    QueryRun,
)
from app.integrations.llm.profiles import PROFILES, resolve
from app.integrations.llm.router import LLMRouter
from app.repositories.project_repository import ProjectRepository
from app.repositories.query_repository import QueryRepository
from app.schemas.agents import FinalDecisionOutput, RiskItem, RiskOutput
from app.schemas.pharma_api import PharmaQueryInput, PharmaRunRead
from app.schemas.queries import (
    EVALUATION_EXPLANATION_WITHHELD,
    AgentRunRead,
    CitationRead,
    EvaluationRead,
    QueryCreate,
    QueryRunDetail,
    QueryRunRead,
    QueryRunSummary,
    QueryRunSummaryPage,
    RiskItemRead,
)
from app.services import event_bus, run_reconciler
from app.services.answer_hygiene import (
    CONTEXT_NOTICE,
    removed_instruction_sentences,
    strip_context_lines,
    with_short_notice,
)
from app.services.citation_service import (
    citation_labels,
    conceal_identifiers,
    remove_unresolved_markers,
)
from app.services.event_bus import EventBus
from app.services.interfaces import SupportsCommit
from app.services.pagination import decode_cursor, encode_cursor
from app.services.run_registry import RunRegistry
from app.utils.ids import new_uuid
from app.utils.time import utc_now

logger = logging.getLogger(__name__)

# Re-exported, not redefined. `event_bus` owns the one statement of "this run is over", because
# it is the module that has to map each terminal status to its terminal event; `jobs.py`,
# `orchestrator.py` and the SSE route already read it from there. A second frozenset here was
# equal to that one, which is precisely the failure mode worth removing -- a duplicate stays equal
# right up until someone adds a status to one of them, and then the divergence is silent.
TERMINAL_QUERY_RUN_STATUSES = event_bus.TERMINAL_QUERY_RUN_STATUSES

# REQ-PER-06: the one statement of who initiated a pharmaceutical run. The API takes no
# credential, so no user identity exists; the request's correlation id is stored beside it.
UNAUTHENTICATED_INITIATOR = "unauthenticated local client"


class PharmaRunRecords(Protocol):
    async def create_run_record(self, record: PharmaRunRecord) -> None: ...

    async def run_record(self, query_run_id: uuid.UUID) -> PharmaRunRecord | None: ...


class QueryService:
    def __init__(
        self,
        session: SupportsCommit,
        projects: ProjectRepository,
        queries: QueryRepository,
        llm: LLMRouter,
        registry: RunRegistry,
        *,
        settings: Settings,
        pharma: PharmaRunRecords | None = None,
        bus: EventBus | None = None,
    ) -> None:
        """``settings`` carries the accept path's capacity limits.

        Required, not optional. It was briefly keyword-with-a-default because this
        constructor's five-argument shape predates the cap -- but a default of ``None``
        meaning "cap not enforced" is a spend control that fails open, which is the
        very shape this finding is about. Every call site passes it.

        ``bus`` is where ``cancel`` publishes the terminal event of an ownerless run it closes
        (RMD5-T01); without one the event is still stored, and a stream that connects later
        replays it.
        """
        self._session = session
        self._projects = projects
        self._queries = queries
        self._llm = llm
        self._registry = registry
        self._settings = settings
        # The pharmaceutical run records (PDP8-T02). None where no pharmaceutical project can
        # be queried; a pharmaceutical query is then refused rather than run unrecorded.
        self._pharma = pharma
        self._bus = bus if bus is not None else EventBus()

    async def create(self, project_id: uuid.UUID, data: QueryCreate) -> QueryRun:
        project = await self._projects.get(project_id)
        if project is None:
            raise ProjectNotFoundError(f"Project {project_id} does not exist.")
        # REQ-API-03: the pharmaceutical input asks for a workflow a General project does not run.
        # Refused before anything is counted, locked or created.
        if (
            data.pharma is not None
            and project.domain_profile is not DomainProfile.PHARMA_MANUFACTURING
        ):
            raise PharmaInputNotAllowedError(
                "The `pharma` object is accepted only for a pharmaceutical manufacturing project."
            )
        stats = await self._projects.stats(project_id)
        if stats.chunk_count == 0:
            raise ConflictError(
                "This project has no indexed documents yet. Upload and index documents "
                "before asking a question."
            )
        await self._reject_if_at_capacity(project_id)

        # `resolved_provider`/`resolved_model_name` hold the *selected* model from creation
        # (set here, from `model_option_id`) and the *actual* model from completion (set by
        # the orchestrator's `_finish`); for a completed run the two agree. A legacy request
        # with no `model_option_id` leaves both `None` here, exactly as before this feature.
        # F-009: the routing-profile field's rollout gate (plan section 26). Switched off, a
        # request that names a profile is refused -- never quietly run on other routing, for the
        # reason the check below gives -- and one that names none runs on its `answer_model`.
        if data.routing_profile is not None and not self._settings.routing_profile_requests_enabled:
            raise InvalidRoutingProfileError(
                "Routing profiles are switched off on this deployment. Omit `routing_profile`; "
                "the answer model then chooses the routing."
            )
        # Reject an unknown profile before anything is created or billed (SDD 10.3.3). Not
        # coerced to the default: a caller who asked for `powerful-synthesis` and silently got
        # `balanced` would be handed an answer written by a model they did not choose, on a run
        # whose recorded provenance contradicts its own request.
        try:
            resolve(data.routing_profile, data.answer_model)
        except KeyError as exc:
            msg = (
                f"Routing profile {data.routing_profile!r} is not recognised. "
                f"Known profiles: {', '.join(sorted(PROFILES))}."
            )
            raise InvalidRoutingProfileError(msg) from exc
        # F-011: SDD 10.3.4. Sending `answer_model` and `routing_profile` where they disagree "is
        # a caller bug, not something to resolve by precedence". Only an `answer_model` the caller
        # actually sent counts: the field's default is not a request.
        if data.routing_profile is not None and "answer_model" in data.model_fields_set:
            named = PROFILES[data.routing_profile]
            if named.tier is not data.answer_model:
                raise InvalidRoutingProfileError(
                    f"answer_model {data.answer_model.value!r} disagrees with routing_profile "
                    f"{named.name!r}, a {named.tier.value!r} profile. Send one or the other."
                )

        answer_model = data.answer_model
        resolved_provider: str | None = None
        resolved_model_name: str | None = None
        if data.model_option_id is not None:
            entry = self._llm.find_model(data.model_option_id)
            if entry is None:
                if self._provider_prefix_is_unavailable(data.model_option_id):
                    raise ModelOptionUnavailableError(
                        f"Model option {data.model_option_id!r} is not currently available."
                    )
                raise InvalidModelOptionError(f"Unknown model option {data.model_option_id!r}.")
            answer_model = entry.tier
            resolved_provider = entry.provider.value
            resolved_model_name = entry.model
        elif not self._llm.available_models():
            # A request that names no model runs on whatever the deployment provides. When the
            # catalogue is empty there is nothing to run it on, and accepting it anyway is how a
            # query used to be created, queued, dispatched to a background worker and only then
            # fail -- after the worker had resolved a live client and billed the vendor the API
            # was simultaneously reporting as unavailable. Reject it here, where there is still
            # a caller to tell.
            #
            # The test is `available_models()`, not `is_live()`: `is_live()` is deliberately
            # False in fake/offline mode (no *real* provider will be called), but fake mode is a
            # working configuration that answers queries, and its two synthetic entries keep the
            # catalogue non-empty. This rejects exactly the case where nothing at all can serve
            # the request.
            #
            # `ModelOptionUnavailableError` (409 `model_option_unavailable`) rather than a new
            # code: it is the same condition the branch above already returns for a *named*
            # option whose provider is unavailable, so a client that handles "the model you
            # asked for is not available right now" needs no new case, and the alternative --
            # 400 `invalid_model_option` -- would be a lie, since nothing about the request is
            # malformed.
            raise ModelOptionUnavailableError(
                "No language model is currently available to answer this question. The server "
                "has no provider configured, or every configured provider is disabled."
            )
        elif not self._llm.default_available():
            # F-008: other providers are available, but the one this option-less request would
            # run on is not. It used to be accepted and then fail in the worker. Refused here,
            # before anything is created: SDD 10.3.3 (4)'s 409 when a profile was named, and
            # otherwise the same 409 as the empty-catalogue branch above.
            if data.routing_profile is not None:
                raise RoutingProfileUnavailableError(
                    f"Routing profile {data.routing_profile!r} cannot run: the provider it would "
                    "use is not available on this deployment. Select an available model option."
                )
            raise ModelOptionUnavailableError(
                "The default model is not available on this deployment. Select an available "
                "model option."
            )

        # F-004: the profile is resolved from the *option's* tier, once the option is known. It
        # was resolved above, from the request's `answer_model` -- which defaults to `powerful` --
        # before the option's tier was written back, so a caller who selected the fast model was
        # answered by the strong one at every synthesis site, under `balanced`, while the run
        # recorded the model it had been asked for. The option wins outright (PRD FR-CHAT-001);
        # a named profile still wins over both (SDD 10.3.3), and the check above has already
        # refused an unknown one before anything was created.
        profile = resolve(data.routing_profile, answer_model)
        if data.routing_profile is not None:
            # F-012: SDD 10.3.4 -- the stored tier is written from the resolved profile, so a row
            # never says `powerful` about a run that ran `flash-all` because the field defaulted.
            answer_model = profile.tier
        query_run = QueryRun(
            id=new_uuid(),
            project_id=project_id,
            question=data.question,
            normalized_question="",
            retrieval_mode=data.retrieval_mode,
            response_style=data.response_style,
            answer_model=answer_model,
            routing_profile=profile.name,
            status=QueryRunStatus.RUNNING,
            latency_ms=0,
            resolved_provider=resolved_provider,
            resolved_model_name=resolved_model_name,
            created_at=utc_now(),
        )
        await self._queries.add(query_run)
        if project.domain_profile is DomainProfile.PHARMA_MANUFACTURING:
            await self._open_pharma_record(project, query_run, data.pharma or PharmaQueryInput())
        await self._session.commit()
        return query_run

    async def _open_pharma_record(
        self, project: Project, query_run: QueryRun, pharma: PharmaQueryInput
    ) -> None:
        """The run's pharmaceutical record with the request's input, its correlation id and the
        fixed initiator statement (REQ-PER-01, REQ-PER-06), in the transaction that creates the
        run, so a run never exists without its record."""
        if self._pharma is None:
            msg = "Pharmaceutical queries are not available on this deployment."
            raise CognifyError(msg)
        incident = pharma.incident.model_dump(mode="json") if pharma.incident is not None else None
        await self._pharma.create_run_record(
            PharmaRunRecord(
                query_run_id=query_run.id,
                project_id=project.id,
                intent=pharma.intent,
                incident=incident,
                filters=pharma.filters.model_dump(mode="json"),
                correlation_id=correlation_id_var.get(),
                initiated_by=UNAUTHENTICATED_INITIATOR,
                created_at=query_run.created_at,
                updated_at=query_run.created_at,
            )
        )

    async def _reject_if_at_capacity(self, project_id: uuid.UUID) -> None:
        """Refuses a query while the project already has its maximum number of runs in flight.

        The count below and the ``create``/``commit`` above are one decision, not two, and the
        lock taken first is what makes them one. Counting and *then* inserting is a check-then-act:
        every request in a simultaneous batch read the same pre-insert total, every one found room
        under the cap, and every one was admitted -- eight concurrent requests against a cap of
        three were all accepted, where the same eight sent one after another give three accepted
        and five refused. The property that held was "at most N as seen by a serialized reader",
        which is not the property worth anything here, because ``POST /projects/{id}/queries``
        takes no credential: an attacker picks the concurrency, and each admitted run commits
        7-20 billed vendor completions.

        ``lock_project_admission`` holds for the remainder of this transaction, which is the
        transaction that inserts and commits the new row, so the next admission for this project
        cannot begin counting until this one's row is visible. Releasing it any earlier -- between
        the count and the insert -- would restore the race exactly. See the repository for why
        this is an advisory lock rather than ``FOR UPDATE`` on the project row, why it cannot
        deadlock against ingestion, and why it needs no cleanup when a holder dies.

        Capacity rather than correctness, but it belongs beside the checks above for exactly the
        reason the comment in ``create`` records for the model-availability gate: this is the
        last point at which there is still a caller to tell, and everything past the commit is
        billed. One accepted run drives 7-20 vendor chat completions -- six agents at up to three
        repair rungs each (``agents/base.py``), plus a reranker completion per retrieval round --
        and nothing downstream counts them: ``workers/jobs.py`` spawns the workflow
        unconditionally, ``RunRegistry.register`` is a plain dict insert, and ``query_runs``
        carries no constraint on concurrent runs. Without this, one caller can hold an arbitrary
        number of those workflows open at once.

        Counted from ``query_runs`` rather than from ``RunRegistry`` deliberately: the registry
        is in-process only (see its docstring), so on a multi-worker deployment it would bound
        each worker separately and bound the deployment not at all.

        ``ConflictError`` (409 ``conflicting_state``) rather than a new code: it is the shape the
        chunk-count refusal above already uses for "this project cannot take this query right
        now", and a 429 would invite a client to retry immediately on a condition that clears
        only when a run finishes.

        The count is bounded to runs created within the last ``query_run_deadline_seconds``, and
        that window is load-bearing rather than an optimisation. ``RUNNING`` is not a liveness
        fact: a process that dies mid-run leaves its row at ``RUNNING`` permanently, and
        ``cancel`` cannot clear it either -- it answers 409 for a run this process holds no task
        for. Counting those would let a single crash cap a project forever, turning a cost
        control into a permanent refusal. A row older than the deadline *cannot* still be
        executing, because ``_execute_workflow`` now guarantees every run resolves within it
        (see ``app.workers.jobs``), so excluding it is sound rather than merely convenient.
        """
        await self._queries.lock_project_admission(project_id)
        in_flight_since = utc_now() - timedelta(seconds=self._settings.query_run_deadline_seconds)
        running = await self._queries.count_by_project_and_status(
            project_id, QueryRunStatus.RUNNING, created_after=in_flight_since
        )
        if running >= self._settings.max_running_queries_per_project:
            raise ConflictError(
                "This project already has the maximum number of analyses in progress. Wait "
                "for one to finish, or cancel it, before asking another question."
            )

    def _provider_prefix_is_unavailable(self, option_id: str) -> bool:
        """True when ``option_id`` names a real, currently-unavailable provider.

        Distinguishes "known but unavailable" (409) from "unknown" (400): only the provider
        prefix has to name a real ``LLMProviderName`` that is missing credentials or disabled
        -- the model suffix is not re-validated here, since an unavailable provider's model
        catalogue cannot be resolved without credentials anyway.
        """
        provider_prefix, _, _ = option_id.partition(":")
        return any(
            availability.provider.value == provider_prefix and not availability.available
            for availability in self._llm.provider_availability()
        )

    async def list_for_project(self, project_id: uuid.UUID) -> list[QueryRun]:
        if await self._projects.get(project_id) is None:
            raise ProjectNotFoundError(f"Project {project_id} does not exist.")
        return await self._queries.list_by_project(project_id)

    async def list_summaries_for_project(
        self,
        project_id: uuid.UUID,
        *,
        limit: int,
        cursor: str | None = None,
        status: QueryRunStatus | None = None,
    ) -> QueryRunSummaryPage:
        """One page of the project's runs as summaries (CT-02): three statements at most -- the
        project check, the page and the count.

        The page is fetched one row long; that extra row is how the next page is detected, and it
        is never returned. The next cursor names the last row returned.
        """
        if await self._projects.get(project_id) is None:
            raise ProjectNotFoundError(f"Project {project_id} does not exist.")
        before = None if cursor is None else decode_cursor(cursor)
        rows = await self._queries.list_summaries_by_project(
            project_id, limit=limit, status=status, before=before
        )
        page = rows[:limit]
        next_cursor = encode_cursor(page[-1].created_at, page[-1].id) if len(rows) > limit else None
        total = await self._queries.count_by_project(project_id, status=status)
        return QueryRunSummaryPage(
            items=[QueryRunSummary.model_validate(row) for row in page],
            next_cursor=next_cursor,
            total=total,
        )

    async def get_query_run(self, query_run_id: uuid.UUID) -> QueryRun:
        query_run = await self._queries.get(query_run_id)
        if query_run is None:
            raise QueryRunNotFoundError(f"Query run {query_run_id} does not exist.")
        return query_run

    async def cancel(self, query_run_id: uuid.UUID) -> QueryRun | None:
        """Requests cancellation of a running query; returns the run as read once the cancel was
        accepted, or ``None`` when this process's task was cancelled but the run could not be
        read (the database is unreachable).

        This process's ``RunRegistry`` is asked first (RMD5-T03, RE2E-DEF-040's wider cause): the
        run used to be read from the database before, so during an outage the cancel answered 500
        while the task it could have stopped was alive. The run is read afterwards for the body; a
        read that shows completed or failed means the cancel landed after the run's own end, and is
        answered 409 like any terminal run (``jobs._mark_cancelled`` relabels nothing), while a
        read that shows cancelled is this cancel's own outcome, already written, and is accepted.

        404s via ``get_query_run`` when the id is unknown. 409s (``QueryRunNotCancellableError``)
        when the run is already in a terminal state, or when it exists and is ``RUNNING`` but this
        process has no matching task in its ``RunRegistry`` and another live process owns it (see
        ``RunRegistry``'s docstring). The actual status transition to ``CANCELLED`` happens later,
        asynchronously, once the cancelled task unwinds -- see ``app.workers.jobs``.

        A run no process owns any more (RMD5-T01, RE2E-DEF-041) -- its lease lapsed, or, for a row
        with no lease, its deadline passed -- has no task anywhere to cancel and nothing that would
        ever close it, so it is closed ``CANCELLED`` here and now and answered 202. A run another
        live process owns (its lease is current) still answers 409.
        """
        if self._registry.cancel(query_run_id):
            try:
                accepted = await self._queries.get(query_run_id)
            except (SQLAlchemyError, OSError):
                logger.warning(
                    "Cancelled query run %s; its row could not be read for the response",
                    query_run_id,
                    exc_info=True,
                )
                return None
            # CANCELLED here is this cancel's own outcome (RMD-DEF-007): the task it stopped
            # unwound and ``jobs._mark_cancelled`` committed before this read. Any other terminal
            # status means the run ended on its own first.
            if (
                accepted is not None
                and accepted.status in TERMINAL_QUERY_RUN_STATUSES
                and accepted.status is not QueryRunStatus.CANCELLED
            ):
                raise QueryRunNotCancellableError(
                    f"Query run {query_run_id} is already {accepted.status.value} "
                    "and cannot be cancelled."
                )
            return accepted
        query_run = await self.get_query_run(query_run_id)
        if query_run.status in TERMINAL_QUERY_RUN_STATUSES:
            raise QueryRunNotCancellableError(
                f"Query run {query_run_id} is already {query_run.status.value} "
                "and cannot be cancelled."
            )
        if run_reconciler.lapse_of(query_run, utc_now(), self._settings) is not None:
            closed = await run_reconciler.cancel_ownerless(
                cast(AsyncSession, self._session), self._bus, query_run_id, self._settings
            )
            if closed is not None:
                return closed
        raise QueryRunNotCancellableError(
            f"Query run {query_run_id} is not currently running in this process."
        )

    async def get_detail(self, query_run_id: uuid.UUID) -> QueryRunDetail:
        query_run = await self.get_query_run(query_run_id)
        citations = await self._queries.list_citations(query_run_id)
        agent_runs = await self._queries.list_agent_runs(query_run_id)
        evaluation = await self._queries.get_evaluation(query_run_id)
        # PH-DEF-005: chosen by the project's profile, not by whether a record exists, so a
        # pharmaceutical run without one never falls through to the agent rows below.
        project = await self._projects.get(query_run.project_id)
        if project is not None and project.domain_profile is DomainProfile.PHARMA_MANUFACTURING:
            record = None if self._pharma is None else await self._pharma.run_record(query_run_id)
            return _pharma_detail_of(query_run, citations, agent_runs, evaluation, record)
        decision = _decision_of(agent_runs)
        labels = citation_labels((c.chunk_id, c.citation_key) for c in citations)
        kept = [citation.citation_key for citation in citations]
        drafted = [draft.citation_key for draft in decision.citations] if decision else []
        unresolved = 0
        hygiene = 0

        def decision_text(text: str) -> str:
            # The trace row is the decision as the model wrote it, before the orchestrator's
            # concealment, relabelling (DV-RX-06) and marker removal, so all of it happens here:
            # every marker naming no persisted citation is removed and counted (RMD8-T01,
            # RE2E-DEF-031; the count has no column, DV-RMD-52), and every line of the evidence
            # context is left out with a short notice (RMD8-T02; no passages here, DV-RMD-56).
            nonlocal unresolved, hygiene
            cleaned, removed = remove_unresolved_markers(
                conceal_identifiers(text, labels), kept, drafted
            )
            unresolved += removed
            cleaned, lines = with_short_notice(cleaned)
            hygiene += lines
            return cleaned

        # The stored answer lost its unresolved markers, context lines and planted copies before
        # it was persisted; one written before RMD8-T01 or RMD8-T02 loses the first two here.
        # Counted from the trace when there is one: its context lines, and its instruction-like
        # sentences the stored answer no longer holds.
        stored, removed_from_stored = remove_unresolved_markers(
            query_run.answer_markdown or "", kept, drafted
        )
        answer, lines_in_stored = strip_context_lines(stored)
        if lines_in_stored:
            answer = f"{answer}\n\n{CONTEXT_NOTICE}" if answer else CONTEXT_NOTICE
        if decision is not None:
            decision_text(decision.answer_markdown)
            drafted_answer, _ = strip_context_lines(
                conceal_identifiers(decision.answer_markdown, labels)
            )
            hygiene += removed_instruction_sentences(drafted_answer, answer)
        else:
            unresolved += removed_from_stored
            hygiene += lines_in_stored
        run = QueryRunRead.model_validate(query_run).model_dump()
        if query_run.answer_markdown is not None:
            run["answer_markdown"] = answer

        return QueryRunDetail(
            **run,
            citations=[CitationRead.model_validate(citation) for citation in citations],
            agent_runs=[AgentRunRead.model_validate(run) for run in agent_runs],
            evaluation=(
                EvaluationRead.model_validate(evaluation) if evaluation is not None else None
            ),
            # C-15, concealed like the answer they accompany (L-2): these are read from the
            # decision agent's trace row, which holds its output exactly as the model wrote it.
            uncertainty=(decision_text(decision.uncertainty) if decision is not None else None),
            next_steps=(
                [decision_text(step) for step in decision.next_steps]
                if decision is not None
                else []
            ),
            # FRP2-T02: the same trace rows and the same concealment. The Risk agent names chunks
            # by whole id in its prose on live runs (OBS-RX-07), so both prose fields are concealed.
            risks=[
                RiskItemRead(
                    category=risk.category,
                    description=decision_text(risk.description),
                    severity=risk.severity,
                    likelihood=risk.likelihood,
                    mitigation=decision_text(risk.mitigation),
                )
                for risk in _risks_of(agent_runs)
            ],
            contradictory_evidence=(
                [decision_text(item) for item in decision.contradictory_evidence]
                if decision is not None
                else []
            ),
            unresolved_citation_markers=unresolved,
            hygiene_removals=hygiene,
        )

    async def get_evaluation(self, query_run_id: uuid.UUID) -> EvaluationRead:
        query_run = await self.get_query_run(query_run_id)
        evaluation = await self._queries.get_evaluation(query_run_id)
        if evaluation is None:
            raise QueryRunNotFoundError(f"No evaluation is available for query run {query_run_id}.")
        read = EvaluationRead.model_validate(evaluation)
        # PH-D-030 on this route too (PH-DEF-025): a pharmaceutical run's evaluator explanation is
        # unchecked model text, chosen by the project's profile as the detail chooses.
        project = await self._projects.get(query_run.project_id)
        if project is not None and project.domain_profile is DomainProfile.PHARMA_MANUFACTURING:
            return read.model_copy(update={"explanation": EVALUATION_EXPLANATION_WITHHELD})
        return read


def _pharma_detail_of(
    query_run: QueryRun,
    citations: Sequence[Citation],
    agent_runs: Sequence[AgentRun],
    evaluation: EvaluationResult | None,
    record: PharmaRunRecord | None,
) -> QueryRunDetail:
    """A pharmaceutical run's detail, read from its policy-checked record (REQ-API-04, REQ-API-05).

    The agent rows hold the model's text before the regulatory-boundary check (PH-DEF-005), so no
    field is read from them. The four model-text fields take the checked response's values, as
    the workflow wrote the answer: the limitations joined, the next steps, and the conflicts the
    model reported. ``risks`` stays empty, because the response's flags have categories of their
    own and appear in ``pharma``. With no checked result yet, all four are empty and ``pharma`` is
    None. The evaluator's explanation is unchecked model text too and is withheld (PH-D-030).
    """
    checked = _pharma_run_read(record)
    response = None if checked is None else checked.response
    return QueryRunDetail(
        **QueryRunRead.model_validate(query_run).model_dump(),
        citations=[CitationRead.model_validate(citation) for citation in citations],
        agent_runs=[AgentRunRead.model_validate(run) for run in agent_runs],
        evaluation=(
            EvaluationRead.model_validate(evaluation).model_copy(
                update={"explanation": EVALUATION_EXPLANATION_WITHHELD}
            )
            if evaluation is not None
            else None
        ),
        uncertainty=(
            " ".join(response.limitations_and_uncertainty) if response is not None else None
        ),
        next_steps=(
            list(response.suggested_investigation_questions_or_next_steps)
            if response is not None
            else []
        ),
        risks=[],
        contradictory_evidence=(
            list(response.conflicts_or_ambiguities.model_reported) if response is not None else []
        ),
        pharma=checked,
    )


def _pharma_run_read(record: PharmaRunRecord | None) -> PharmaRunRead | None:
    """The record's ``pharma`` section, or None until the recorder has written its result.

    A record whose stored result does not validate is treated as having none: an unreadable
    result is not a checked one.
    """
    if record is None or record.response is None or record.sufficiency is None:
        return None
    try:
        return PharmaRunRead.model_validate(
            {
                "intent": record.intent,
                "incident": record.incident,
                "filters": record.filters,
                "sufficiency": record.sufficiency.get("sufficiency"),
                "abstained": record.response.get("abstained"),
                "disposition": record.response.get("disposition"),
                "disposition_gate": record.response.get("disposition_gate"),
                "rule_version": record.rule_version,
                "banner": record.response.get("banner"),
                "response": record.response.get("sections"),
                "excluded_evidence": [
                    {"anchor": entry, "reason": entry["exclusion_reason"]}
                    for entry in record.evidence_snapshot
                    if entry.get("exclusion_reason") is not None
                ],
                "policy_notices": (record.policy or {}).get("notices", []),
            }
        )
    except ValidationError:
        return None


def _decision_of(agent_runs: Sequence[AgentRun]) -> FinalDecisionOutput | None:
    """The decision agent's output, or ``None`` when the step never produced a readable one.

    ``None`` covers not-run, failed and unparseable alike: the detail response has one field to
    say "there is no statement", and an unreadable output is not a statement.
    """
    for run in agent_runs:
        if run.agent_name is AgentName.FINAL_DECISION and run.output_json is not None:
            try:
                return FinalDecisionOutput.model_validate(run.output_json)
            except ValidationError:
                return None
    return None


def _risks_of(agent_runs: Sequence[AgentRun]) -> list[RiskItem]:
    """The Risk agent's items from its completed step, or none when there is no readable output.

    An output that does not validate as ``RiskOutput`` is skipped, as the dashboard's
    ``_risk_categories`` skips it: an unreadable output states no risk, and it must not turn the
    whole detail into an error. A run that failed after its risk step still has that step's items.
    """
    for run in agent_runs:
        if (
            run.agent_name is AgentName.RISK
            and run.status is AgentRunStatus.COMPLETED
            and run.output_json is not None
        ):
            try:
                return RiskOutput.model_validate(run.output_json).risks
            except ValidationError:
                continue
    return []
