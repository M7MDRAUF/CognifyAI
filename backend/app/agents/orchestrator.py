"""Agent orchestrator (SDD section 8.1).

Runs the agent sequence for one QueryRun, persists each AgentRun and the
retrieved evidence/citations/evaluation, emits workflow events, and applies a
single critic-driven retrieval expansion. The evaluator is allowed to fail
without voiding the answer (graceful degradation).
"""

import asyncio
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import replace

from pydantic import BaseModel

from app.agents.base import AgentResult, JsonAgentRunner
from app.agents.critic import CriticAgent
from app.agents.evaluator import EvaluatorAgent
from app.agents.final_decision import FinalDecisionAgent
from app.agents.planner import PlannerAgent
from app.agents.premise_guard import claims_evidence_supply, claims_unseen_evidence
from app.agents.researcher import ResearchAgent
from app.agents.retriever import RetrieverAgent
from app.agents.risk import RiskAgent
from app.core.errors import AgentOutputValidationError, CognifyError, LLMProviderError
from app.core.logging import EXCERPT_CHAR_LIMIT, bound_context, bounded_excerpt
from app.core.pricing import RateTable, rate_table
from app.domain.enums import (
    AgentName,
    AgentRunStatus,
    AnswerModel,
    EventType,
    QueryRunStatus,
    ResponseStyle,
    RetrievalMode,
    SelectionState,
)
from app.domain.models import (
    AgentRun,
    Citation,
    EvaluationResult,
    LLMCall,
    QueryRun,
    RetrievedEvidence,
)
from app.integrations.llm.base import ChatProvider, ModelRole, TokenUsage
from app.integrations.llm.profiles import CallSite, RoutingProfile, resolve
from app.repositories.query_repository import QueryRepository
from app.schemas.agents import (
    CitationDraft,
    EvaluatorOutput,
    FinalDecisionOutput,
    PlannerOutput,
)
from app.services.answer_hygiene import clean_answer
from app.services.citation_service import (
    UNLABELLED_SOURCE,
    CitationReport,
    conceal_raw_identifiers,
    resolve_keyed_citations,
    validate_citations,
    without_unresolved_markers,
)
from app.services.event_bus import TERMINAL_QUERY_RUN_STATUSES
from app.services.event_emitter import EventEmitter
from app.services.interfaces import SupportsCommit
from app.services.reranking_service import RerankingService
from app.services.retrieval_service import HydrationReport, RetrievalService
from app.services.retrieval_types import RetrievedChunk
from app.utils.ids import new_uuid
from app.utils.time import utc_now

logger = logging.getLogger(__name__)

# Shared with `workers.jobs`, which writes the same outcome for a failure raised before
# `Orchestrator.run` is ever entered; one message, so the two paths are indistinguishable
# to a user who only ever sees the row.
UNEXPECTED_FAILURE_MESSAGE = "The analysis workflow failed unexpectedly."


class Orchestrator:
    def __init__(
        self,
        session: SupportsCommit,
        queries: QueryRepository,
        emitter: EventEmitter,
        provider: ChatProvider,
        retrieval: RetrievalService,
        reranking: RerankingService,
        *,
        rerank_expanded_union: bool = True,
        rates: RateTable | None = None,
    ) -> None:
        self._session = session
        # F-010: the union path's rollout gate (`Settings.rerank_expanded_union`, plan section 26).
        self._rerank_expanded_union = rerank_expanded_union
        # RE2E-DEF-028 (RMD9-T03): the table every metering row is priced from; the shipped one
        # unless a test gives another.
        self._rates = rates if rates is not None else rate_table()
        self._queries = queries
        self._emitter = emitter
        self._planner = PlannerAgent(JsonAgentRunner(provider, "planner"))
        self._retriever = RetrieverAgent(retrieval, reranking)
        # Held directly as well as through the retriever: the reranker's usage has to be read
        # after each retrieval round, and the retriever deliberately does not surface it.
        self._reranking = reranking
        self._retrieval = retrieval
        # Citation anchoring counts from the most recent run, for the trace (L-1).
        self._citation_report = CitationReport()
        self._researcher = ResearchAgent(JsonAgentRunner(provider, "researcher"))
        self._critic = CriticAgent(JsonAgentRunner(provider, "critic"))
        self._risk = RiskAgent(JsonAgentRunner(provider, "risk"))
        self._final = FinalDecisionAgent(JsonAgentRunner(provider, "final_decision"))
        self._evaluator = EvaluatorAgent(JsonAgentRunner(provider, "evaluator"))
        self._usage = TokenUsage()
        self._query_run_id = uuid.UUID(int=0)
        # Provider/model of the most recently completed `_step` rung. Snapshotted right
        # after the final-decision step (see `_run_workflow`) so `_finish` can copy the
        # decision agent's provenance onto `query_runs` without threading it through
        # `FinalDecisionOutput`.
        self._last_provider = ""
        self._last_model = ""
        # The deduplicated candidate pool from every retrieval round so far. An expansion unions
        # into it and reranks the whole thing once, which is what makes the final cut global.
        self._candidates: list[RetrievedChunk] = []

    async def run(self, query_run: QueryRun, domain: str, objective: str) -> None:
        self._query_run_id = query_run.id
        started = time.monotonic()
        try:
            await self._run_workflow(query_run, domain, objective, started)
        except CognifyError as exc:
            await self._fail(query_run, exc.message, started)
        except Exception:
            logger.exception("Workflow crashed")
            await self._fail(query_run, UNEXPECTED_FAILURE_MESSAGE, started)

    async def _run_workflow(
        self, query_run: QueryRun, domain: str, objective: str, started: float
    ) -> None:
        # Every call site is mapped by the profile; there is no site left to fall back to
        # whatever (SDD 10.3.2). The planner and the reranker stay small in all three profiles.
        profile = _routing_profile(query_run)

        await self._emitter.emit(EventType.PLANNING_STARTED, "Planning the analysis.")
        plan = await self._step(
            AgentName.PLANNER,
            {"question": query_run.question},
            self._planner.run(query_run.question, domain, objective),
        )
        query_run.normalized_question = _guarded_question(query_run, plan)
        # FR-002: a `risk_focus` item that states whether evidence was supplied is a claim the
        # planner cannot make, since it runs before retrieval; genuine risk topics are kept. The
        # planner's own row keeps every item, so the claim stays measurable.
        risk_focus = [item for item in plan.risk_focus if not claims_evidence_supply(item)]

        evidence = await self._retrieve(query_run, plan)

        await self._emitter.emit(EventType.RESEARCH_STARTED, "Synthesizing evidence.")
        await self._supply(query_run.id, evidence)
        research = await self._step(
            AgentName.RESEARCHER,
            {
                "normalized_question": query_run.normalized_question,
                # FR-016: the chunks the researcher was handed, recorded on its own row, because
                # an expansion's union rerank can later cut them from the final context and
                # `_persist_ranking` then marks them `reranked`.
                "supplied_chunk_ids": [str(item.chunk_id) for item in evidence],
            },
            self._researcher.run(
                query_run.normalized_question,
                evidence,
                role=profile.role_for(CallSite.RESEARCHER),
            ),
        )

        await self._emitter.emit(EventType.CRITIC_STARTED, "Checking claim support.")
        critic = await self._step(
            AgentName.CRITIC,
            {"findings": len(research.key_findings)},
            self._critic.run(
                query_run.normalized_question,
                research.key_findings,
                evidence,
                role=profile.role_for(CallSite.CRITIC),
            ),
        )

        if critic.needs_more_retrieval and critic.expansion_queries:
            evidence = await self._expand(query_run, critic.expansion_queries, evidence)

        await self._emitter.emit(EventType.RISK_STARTED, "Assessing risks.")
        risk = await self._step(
            AgentName.RISK,
            {"risk_focus": list(risk_focus)},
            self._risk.run(
                query_run.normalized_question,
                research.key_findings,
                risk_focus,
                role=profile.role_for(CallSite.RISK),
            ),
        )

        await self._emitter.emit(EventType.ANSWER_STARTED, "Composing the decision.")
        response_style = ResponseStyle(query_run.response_style)
        # After an expansion this is a different context from the one the researcher was handed,
        # and it has not been supplied to anyone yet; without an expansion it is the same rows.
        await self._supply(query_run.id, evidence)
        decision = await self._step(
            AgentName.FINAL_DECISION,
            {"response_style": response_style.value, "answer_model": query_run.answer_model.value},
            self._final.run(
                query_run.normalized_question,
                response_style,
                research.key_findings,
                critic,
                risk,
                evidence,
                role=profile.role_for(CallSite.FINAL_DECISION),
            ),
        )

        # Snapshot immediately: `_evaluate` below records its own AgentRun but does not
        # go through `_step`, so `self._last_provider`/`self._last_model` still hold the
        # final-decision agent's values here.
        decision_provider, decision_model = self._last_provider, self._last_model
        # RMD8-T01: the evidence keys decide which passage a citation names, before concealment
        # renames any raw-id key.
        decision = resolve_keyed_citations(decision, evidence)
        # L-2: the decision as the reader will see it, before anything is persisted or
        # validated -- anchoring then checks the answer the reader actually gets. The agent's
        # own output stays on its `agent_runs` row, which is the trace, not the answer.
        decision = conceal_raw_identifiers(decision)
        # FR-016: a key whose chunk the final context no longer holds would reach the reader as
        # a reference to nothing, since `validate_citations` drops its draft (C-10).
        decision, relabelled = relabel_keys_outside_context(decision, evidence)
        for key in relabelled:
            logger.info(
                "Relabelling citation key %s in run %s: its chunk is outside the final context",
                key,
                query_run.id,
                extra={"event_type": "citation_key_relabelled"},
            )
        # RE2E-DEF-043, RE2E-DEF-044 (RMD8-T02): the answer repeats no line of its evidence
        # context and no instruction planted in a passage, and says what it left out. Before
        # anchoring, so a citation counts only if the cleaned answer still holds its marker;
        # GET /queries/{id} counts the removals again from the trace (DV-RMD-56).
        answer, lines, copies = clean_answer(
            decision.answer_markdown, [item.text for item in evidence]
        )
        if lines or copies:
            decision = decision.model_copy(update={"answer_markdown": answer})
            logger.info(
                "Removed %d evidence context lines and %d quoted instructions from run %s",
                lines,
                copies,
                query_run.id,
                extra={"event_type": "answer_hygiene_applied"},
            )
        # RE2E-DEF-031: no marker that names no kept citation is shown; GET /queries/{id} counts
        # them again from the trace (DV-RMD-52).
        kept, _report = validate_citations(decision.citations, evidence, decision.answer_markdown)
        decision, unresolved = without_unresolved_markers(decision, [c.citation_key for c in kept])
        if unresolved:
            logger.info(
                "Removed %d citation markers naming no returned source from run %s",
                unresolved,
                query_run.id,
                extra={"event_type": "citation_markers_removed"},
            )

        # N-006 / P2-T14: the answer is persisted **before** the artefacts derived from it.
        #
        # The previous order wrote citations, then the evaluation, and only then the answer and
        # the terminal status. Every await in between is a point where the run can be cancelled
        # or hit its deadline, and each one left the same shape behind: citations and an
        # evaluation attached to a run whose `answer_markdown` is NULL. The detail endpoint then
        # returns citations for an answer that does not exist, and the analytics eligibility
        # predicate -- which tests `answer_markdown IS NOT NULL` -- sees an evaluation row it is
        # supposed to be able to trust.
        #
        # Writing the answer first inverts the failure: an interrupted run can now be missing
        # its citations, which reads correctly as "it did not get that far", instead of having
        # derivatives of an answer nobody has.
        await self._persist_answer(query_run, decision, decision_provider, decision_model)

        citations = await self._persist_citations(query_run, decision, evidence)

        await self._emitter.emit(EventType.EVALUATION_STARTED, "Scoring the answer.")
        await self._evaluate(
            query_run, decision, citations, evidence, profile.role_for(CallSite.EVALUATOR)
        )

        await self._finish(query_run, started)

    async def _retrieve(self, query_run: QueryRun, plan: PlannerOutput) -> list[RetrievedChunk]:
        # C-7: each event is emitted at the boundary it names. These two used to fire back to
        # back *before* retrieval ran, so the timestamps said retrieval and reranking began at
        # the same instant and neither told a reader when anything actually started.
        await self._emitter.emit(
            EventType.RETRIEVAL_STARTED, "Retrieving evidence.", payload={"round": 0}
        )
        started = time.monotonic()
        # M-6, M-9: the row is opened before the work, as `_step` opens an agent's. It used to be
        # written in one go after the round, so a failed retrieval left no row at all and a
        # successful one recorded a start time taken after the reranker had already answered.
        agent_run_id = await self._open_agent_run(
            AgentName.RETRIEVER, {"queries": list(plan.retrieval_plan.queries)}
        )
        mode = RetrievalMode(query_run.retrieval_mode)
        try:
            # The round-one *candidate pool* is kept, not just the reranked result: an expansion
            # unions this pool with round two's and reranks the union once (C-10, ADR-006).
            self._candidates = await self._retriever.gather(
                query_run.project_id, plan.retrieval_plan.queries, mode, round_index=0
            )
            await self._emitter.emit(
                EventType.RERANKING_STARTED,
                "Reranking candidates.",
                payload={"round": 0, "candidates": len(self._candidates)},
            )
            evidence = await self._retriever.rerank(query_run.normalized_question, self._candidates)
            # M-7: the reranker's call is made inside this step and its usage was discarded
            # outright, so a run's token estimate omitted one billed call per retrieval round.
            await self._meter_reranker(started)
            await self._persist_ranking(query_run.id, evidence)
        except Exception as exc:
            # `CancelledError` is not an `Exception` and passes straight through, leaving the
            # RUNNING row behind -- the same record `_step` leaves for a step that was stopped.
            await self._close_failed(AgentName.RETRIEVER, agent_run_id, started, exc)
            raise
        await self._close_agent_run(
            agent_run_id,
            started,
            status=AgentRunStatus.COMPLETED,
            output_json=self._retrieval_envelope(round_index=0, selected=evidence),
        )
        # C-6: the same shape as the expansion round's completion, round index included --
        # without it a client could not tell which round a completion closed.
        await self._emitter.emit(
            EventType.RETRIEVAL_COMPLETED,
            "Evidence selected.",
            payload={"round": 0, "count": len(evidence)},
        )
        return evidence

    async def _expand(
        self,
        query_run: QueryRun,
        expansion_queries: Sequence[str],
        evidence: Sequence[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        await self._emitter.emit(
            EventType.RETRIEVAL_STARTED,
            "Expanding retrieval per critic feedback.",
            payload={"round": 1},
        )
        mode = RetrievalMode(query_run.retrieval_mode)
        started = time.monotonic()
        # L-3: the expansion round records its own agent run. Seven rows were written for eight
        # billed completions, because this round produced none -- so the trace showed one
        # retrieval where two had happened and one reranker call where two were paid for. It is
        # opened before the work, for the reason `_retrieve` gives.
        agent_run_id = await self._open_agent_run(
            AgentName.RETRIEVER, {"queries": list(expansion_queries), "round": 1}
        )
        try:
            # C-10: gather round two's *candidates*, union them with round one's candidate pool,
            # and rerank the union **once**. The previous code reranked each round separately --
            # each cut to K -- and then pooled the two results without cutting again, so an
            # expanded run handed up to 2K chunks downstream and a round-two chunk had never been
            # scored on the same scale as a round-one chunk.
            extra_candidates = await self._retriever.gather(
                query_run.project_id, expansion_queries, mode, round_index=1
            )
            if self._rerank_expanded_union:
                reranked = await self._rerank_union(query_run, extra_candidates, started)
                context = reranked
            else:
                reranked = await self._rerank_round_alone(
                    query_run, extra_candidates, evidence, started
                )
                # Pooled with round one's context and not cut again. A chunk both rounds chose
                # keeps its round-one place, as `setdefault` keeps it in the union.
                pooled = {item.chunk_id: item for item in evidence}
                for item in reranked:
                    pooled.setdefault(item.chunk_id, item)
                context = list(pooled.values())
        except Exception as exc:
            await self._close_failed(AgentName.RETRIEVER, agent_run_id, started, exc)
            raise
        await self._close_agent_run(
            agent_run_id,
            started,
            status=AgentRunStatus.COMPLETED,
            output_json=self._retrieval_envelope(round_index=1, selected=reranked),
        )
        # C-6: the expansion round emitted only RETRIEVAL_STARTED, so a client watching the
        # stream saw a round begin and never end.
        await self._emitter.emit(
            EventType.RETRIEVAL_COMPLETED,
            "Expanded evidence selected.",
            payload={"round": 1, "count": len(context)},
        )
        return context

    async def _rerank_union(
        self,
        query_run: QueryRun,
        extra_candidates: Sequence[RetrievedChunk],
        started: float,
    ) -> list[RetrievedChunk]:
        """C-10: union round two's *candidates* with round one's pool and rerank the union once.

        The previous code reranked each round separately -- each cut to K -- and then pooled the
        two results without cutting again, so an expanded run handed up to 2K chunks downstream and
        a round-two chunk had never been scored on the same scale as a round-one chunk.
        """
        union: dict[uuid.UUID, RetrievedChunk] = {item.chunk_id: item for item in self._candidates}
        for item in extra_candidates:
            # `setdefault`: a chunk found in both rounds keeps its round-one identity, so
            # `round_index` records where it was *first* discovered.
            union.setdefault(item.chunk_id, item)
        self._candidates = list(union.values())

        await self._emitter.emit(
            EventType.RERANKING_STARTED,
            "Reranking the expanded union.",
            payload={"round": 1, "candidates": len(self._candidates)},
        )
        reranked = await self._retriever.rerank(query_run.normalized_question, self._candidates)
        # The expansion round is a second billed reranker call. Counting only the first round
        # is how "seven agent rows for eight billed completions" happened (L-3).
        await self._meter_reranker(started)
        # The whole union, not just the rows this round happened to touch: round one's cut
        # is re-cut here, and a chunk it drops must stop claiming the final context.
        await self._persist_ranking(query_run.id, reranked)
        return reranked

    async def _rerank_round_alone(
        self,
        query_run: QueryRun,
        extra_candidates: Sequence[RetrievedChunk],
        evidence: Sequence[RetrievedChunk],
        started: float,
    ) -> list[RetrievedChunk]:
        """F-010: the rollout gate is off, so this is the per-round path that C-10 replaced.

        Round two's own candidates are reranked and cut to K. The caller pools the cut with round
        one's context without cutting again, so an expanded run can hand up to 2K chunks on. Round
        one's context rows are left as they were, and round two's cut is written as selected. That
        is C-10's registered behaviour. Restoring it is what switching the union off means (plan
        section 26), and it is never the default.
        """
        round_one = {item.chunk_id for item in evidence}
        self._candidates = list(extra_candidates)
        await self._emitter.emit(
            EventType.RERANKING_STARTED,
            "Reranking the expansion round.",
            payload={"round": 1, "candidates": len(self._candidates)},
        )
        reranked = await self._retriever.rerank(query_run.normalized_question, self._candidates)
        await self._meter_reranker(started)
        await self._persist_evidence(
            query_run.id,
            [
                (item, SelectionState.SELECTED)
                for item in reranked
                if item.chunk_id not in round_one
            ],
        )
        return reranked

    async def _evaluate(
        self,
        query_run: QueryRun,
        decision: FinalDecisionOutput,
        citations: Sequence[CitationDraft],
        evidence: Sequence[RetrievedChunk],
        role: ModelRole,
    ) -> None:
        started = time.monotonic()
        agent_run_id = await self._open_agent_run(AgentName.EVALUATOR, {})
        try:
            # C-14: the evaluator records its own run rather than going through `_step`, so it
            # has to bind its own name -- otherwise every record its provider call makes is the
            # one step in the run with no agent to attribute it to.
            with bound_context(agent_name=AgentName.EVALUATOR.value):
                result = await self._evaluator.run(
                    query_run.normalized_question,
                    decision.answer_markdown,
                    citations,
                    evidence,
                    role=role,
                )
        except Exception as exc:
            # M-9: `CognifyError` alone was caught here, so any other exception left this row
            # RUNNING and failed a run whose answer was already persisted. PRD FR-AGENT-006 makes
            # no exception-type distinction: a run whose evaluator failed is `completed` with no
            # evaluation row. `CancelledError` is not an `Exception`, and still stops the run.
            logger.warning("Evaluator failed; persisting answer without evaluation scores")
            await self._close_failed(AgentName.EVALUATOR, agent_run_id, started, exc)
            return
        self._usage = _add_usage(self._usage, result.usage)
        # `_evaluate` records its own agent run rather than going through `_step`, so it has to
        # meter its own call too -- otherwise the one agent permitted to fail without voiding the
        # run is also the one whose spend goes unrecorded.
        await self._record_llm_call(
            AgentName.EVALUATOR.value,
            result.usage,
            started,
            provider=result.provider,
            model=result.model,
            rung=result.attempt_count,
        )
        await self._close_agent_run(
            agent_run_id,
            started,
            status=AgentRunStatus.COMPLETED,
            output_json=result.output.model_dump(mode="json"),
            provider=result.provider,
            model=result.model,
            attempt_count=result.attempt_count,
        )
        await self._persist_evaluation(query_run.id, result.output)

    async def _step[OutputT: BaseModel](
        self,
        agent_name: AgentName,
        input_json: dict[str, object],
        coro: Awaitable[AgentResult[OutputT]],
    ) -> OutputT:
        started = time.monotonic()
        agent_run_id = await self._open_agent_run(agent_name, input_json)
        # Bound around the await, not around the whole method: every log line the agent and the
        # provider adapter emit while this step runs -- including the ones from inside a retry --
        # carries the agent's name, and the binding is gone again before the next step starts.
        # `bound_context` resets in a `finally`, which is what keeps a cancelled step (deadline
        # expiry cancels the task, and `CancelledError` skips `except`) from leaking its name
        # onto whatever the worker logs next.
        with bound_context(agent_name=agent_name.value):
            try:
                result = await coro
            except asyncio.CancelledError:
                # Re-raised untouched, and **not** recorded as a failure. A cancelled step did
                # not fail: the run was stopped, and `jobs._mark_cancelled` owns that outcome.
                # Writing here would also mean awaiting a commit on a session that is already
                # unwinding. The RUNNING row committed above survives, which is the honest
                # record of a step that started and was stopped.
                raise
            except Exception as exc:
                # M-9: the handler used to catch `CognifyError` alone, so any other exception --
                # a `TypeError` from a vendor SDK, an `asyncpg` error, a bug in an agent --
                # left no row at all. The step is recorded, then the exception continues to
                # `run`, which is what decides the run's outcome.
                await self._close_failed(agent_name, agent_run_id, started, exc)
                raise
        self._usage = _add_usage(self._usage, result.usage)
        self._last_provider, self._last_model = result.provider, result.model
        await self._record_llm_call(
            agent_name.value,
            result.usage,
            started,
            provider=result.provider,
            model=result.model,
            rung=result.attempt_count,
        )
        await self._close_agent_run(
            agent_run_id,
            started,
            status=AgentRunStatus.COMPLETED,
            output_json=result.output.model_dump(mode="json"),
            provider=result.provider,
            model=result.model,
            attempt_count=result.attempt_count,
        )
        return result.output

    async def _open_agent_run(
        self, agent_name: AgentName, input_json: dict[str, object]
    ) -> uuid.UUID:
        """Insert the row **before** the step is awaited, in RUNNING, with a true start time.

        The previous write path inserted one COMPLETED row after the step returned, with
        `started_at` and `completed_at` both stamped at that same instant -- so a step in flight
        was invisible, `PENDING` and `RUNNING` were declared and never written, and the only
        evidence that time had passed was `latency_ms`. Committing the open row means a crash,
        a deadline or a cancellation mid-step leaves a RUNNING row behind, which is exactly the
        record needed to tell "it never started" from "it started and never finished".
        """
        agent_run = AgentRun(
            id=new_uuid(),
            query_run_id=self._query_run_id,
            agent_name=agent_name,
            status=AgentRunStatus.RUNNING,
            input_json=input_json,
            output_json=None,
            attempt_count=1,
            started_at=utc_now(),
            completed_at=None,
            latency_ms=0,
        )
        await self._queries.add_agent_run(agent_run)
        await self._session.commit()
        return agent_run.id

    async def _close_agent_run(
        self,
        agent_run_id: uuid.UUID,
        started: float,
        *,
        status: AgentRunStatus,
        output_json: dict[str, object] | None = None,
        provider: str = "",
        model: str = "",
        attempt_count: int = 1,
        error_message: str | None = None,
        failure_excerpt: str | None = None,
        error_classification: str | None = None,
    ) -> None:
        """Resolve the open row in place. One row per attempted component, always."""
        agent_run = await self._queries.get_agent_run(agent_run_id)
        if agent_run is None:  # pragma: no cover - the row was committed moments ago
            return
        agent_run.status = status
        agent_run.output_json = output_json
        agent_run.attempt_count = attempt_count
        agent_run.error_message = error_message
        agent_run.failure_excerpt = failure_excerpt
        agent_run.error_classification = error_classification
        # "" (fake mode / no LLM involved in this step) must persist as NULL,
        # never as an empty string pretending to be a provider name.
        agent_run.llm_provider = provider or None
        agent_run.llm_model = model or None
        agent_run.completed_at = utc_now()
        agent_run.latency_ms = _elapsed_ms(started)
        await self._session.commit()

    async def _close_failed(
        self, agent_name: AgentName, agent_run_id: uuid.UUID, started: float, exc: Exception
    ) -> None:
        """Resolve an open row as FAILED, whatever the exception type (M-9, L-6).

        A domain error carries a user-safe message and may carry usage the vendor billed, which
        is metered first. Anything else is logged with its traceback and recorded under the
        generic message; its own text reaches only the redacted, bounded excerpt.

        The attempt count is the rungs the step actually spent. An exhausted repair ladder paid
        for three, and the row recorded one -- understating both the step's retries and its bill.
        """
        if isinstance(exc, CognifyError):
            await self._record_charged_failure(agent_name.value, exc, started)
            message = exc.message
        else:
            logger.exception("Step %s raised", agent_name.value)
            message = UNEXPECTED_FAILURE_MESSAGE
        rungs = exc.rungs if isinstance(exc, AgentOutputValidationError) else 1
        await self._close_agent_run(
            agent_run_id,
            started,
            status=AgentRunStatus.FAILED,
            attempt_count=max(rungs, 1),
            error_message=message,
            failure_excerpt=_failure_excerpt(exc),
            error_classification=type(exc).__name__,
        )

    async def _record_llm_call(
        self,
        component: str,
        usage: TokenUsage,
        started: float,
        *,
        provider: str,
        model: str,
        outcome: str = "success",
        failure_kind: str | None = None,
        rung: int = 1,
        attempt: int = 1,
    ) -> None:
        """Write one metering row (M-7, C-12).

        Skipped entirely when no provider answered -- a fake-mode step names no vendor, and a row
        claiming `provider=""` would be a spend record for a call that never happened. That is
        the opposite failure to the one this table exists to fix, and just as misleading.
        """
        if not provider:
            return
        # RE2E-DEF-028 (RMD9-T03): priced from the rate table, or `unpriced` with no amount when
        # the table has no rate or the vendor reported no usage -- never 0 (ADR-004).
        prompt_tokens = usage.prompt_tokens if usage.reported else None
        completion_tokens = usage.completion_tokens if usage.reported else None
        cost = self._rates.estimate(
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        await self._queries.add_llm_call(
            LLMCall(
                id=new_uuid(),
                query_run_id=self._query_run_id,
                component=component,
                provider=provider,
                model=model,
                rung=rung,
                attempt=attempt,
                outcome=outcome,
                failure_kind=failure_kind,
                # F-003: NULL is "the vendor reported no usage", 0 is "it reported zero". A
                # missing usage block recorded as 0 is a billed attempt that prices as free.
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost.amount_usd,
                pricing_status=cost.status,
                # The call details (PLV8-T02, migration 0014), NULL when not reported.
                cached_prompt_tokens=usage.cached_prompt_tokens if usage.reported else None,
                reasoning_tokens=usage.reasoning_tokens if usage.reported else None,
                thinking=None if usage.mode is None else _THINKING[usage.mode.enabled],
                reasoning_effort=None if usage.mode is None else usage.mode.effort,
                provider_request_ids=" ".join(usage.response_ids) or None,
                response_model=usage.response_model or None,
                system_fingerprint=usage.system_fingerprint or None,
                latency_ms=_elapsed_ms(started),
                created_at=utc_now(),
            )
        )
        await self._session.commit()

    def _retrieval_envelope(
        self, *, round_index: int, selected: Sequence[RetrievedChunk]
    ) -> dict[str, object]:
        """The retriever step's record: an envelope, not a scalar (M-1, SDD 8.5).

        `{"retrieved": N}` answered one question and hid every other one. It could not say how
        many candidates were gathered, how many survived the cut, whether the rerank fell back,
        how many chunks were lost to hydration, or which model the reranker billed -- and with
        the expansion round recording nothing at all, it could not even say how many rounds ran.
        """
        hydration = self._retrieval_report()
        usage = self._reranking.last_usage
        return {
            "schema_version": 1,
            "round_index": round_index,
            "candidates": len(self._candidates),
            "selected": len(selected),
            "rerank_fallback": all(chunk.rerank_score is None for chunk in selected)
            and bool(selected),
            "hydration_requested": hydration.requested,
            "hydration_hydrated": hydration.hydrated,
            "hydration_dropped": hydration.dropped,
            "reranker_provider": self._reranking.last_provider or None,
            "reranker_model": self._reranking.last_model or None,
            # F-003: unknown when the vendor reported no usage, as on the metering row.
            "reranker_prompt_tokens": usage.prompt_tokens if usage.reported else None,
            "reranker_completion_tokens": usage.completion_tokens if usage.reported else None,
        }

    def _retrieval_report(self) -> HydrationReport:
        """Hydration counts from the retrieval service, via the retriever that owns it."""
        return self._retrieval.last_hydration

    async def _meter_reranker(self, started: float) -> None:
        """Accumulate and record the reranker's own call for the round just completed.

        A reply that was billed and could not be used is a charged failure, not a success (C-12,
        N-002): the reranker falls back, and the row must agree with the envelope that says so.
        """
        usage = self._reranking.last_usage
        failure_kind = self._reranking.last_failure_kind
        self._usage = _add_usage(self._usage, usage)
        await self._record_llm_call(
            "reranker",
            usage,
            started,
            provider=self._reranking.last_provider,
            model=self._reranking.last_model,
            outcome="success" if failure_kind is None else "charged_failure",
            failure_kind=failure_kind,
        )

    async def _record_charged_failure(
        self, component: str, exc: CognifyError, started: float
    ) -> None:
        """Meter a failure that the vendor still billed (L-12, N-002).

        Two errors carry charged-attempt usage: an ``LLMProviderError`` whose response reported
        some, and an ``AgentOutputValidationError`` from an exhausted repair ladder, every rung of
        which was billed. ``charged_tokens is None`` means nothing was reported, which is not the
        same as reporting zero and must not be recorded as a billed attempt.

        The tokens are added to the run's total as well as written as a row. Writing only the row
        left `token_usage_estimate` -- which `_fail` writes from that total -- disagreeing with
        the very rows it is supposed to summarise.
        """
        if isinstance(exc, LLMProviderError):
            failure_kind, rung = exc.kind.value, 1
        elif isinstance(exc, AgentOutputValidationError):
            failure_kind, rung = exc.code, max(exc.rungs, 1)
        else:
            return
        if exc.charged_tokens is None:
            return
        usage = TokenUsage(
            prompt_tokens=exc.prompt_tokens or 0,
            completion_tokens=exc.completion_tokens or 0,
        )
        if isinstance(exc, AgentOutputValidationError):
            # RE2E-DEF-010: every rung answered, so the row keeps their response ids and the cache
            # and reasoning tokens they reported, as a success row does (PLV8-T02).
            usage = replace(
                usage,
                cached_prompt_tokens=exc.cached_prompt_tokens,
                reasoning_tokens=exc.reasoning_tokens,
                response_ids=exc.response_ids,
            )
        self._usage = _add_usage(self._usage, usage)
        await self._record_llm_call(
            component,
            usage,
            started,
            provider=exc.provider or "",
            model=exc.model or "",
            outcome="charged_failure",
            failure_kind=failure_kind,
            rung=rung,
        )

    async def _persist_ranking(
        self, query_run_id: uuid.UUID, selected: Sequence[RetrievedChunk]
    ) -> None:
        """Persist the round's **whole** union in its one final order (M-8, N-004).

        Writing the gathered candidates and then the cut updated only the rows a round happened
        to touch. After an expansion, a round-one chunk the union rerank cut kept `selected` and
        its round-one rank, so the persisted final context could exceed `K` and two rows could
        share a rank. Every row is rewritten here, from the reranker's full ordering: the cut as
        `selected`, a chunk the rerank scored and cut as `reranked`, one it never scored as a
        `candidate`.
        """
        chosen = {item.chunk_id for item in selected}
        await self._persist_evidence(
            query_run_id,
            [
                (
                    item,
                    SelectionState.SELECTED
                    if item.chunk_id in chosen
                    else SelectionState.RERANKED
                    if item.rerank_score is not None
                    else SelectionState.CANDIDATE,
                )
                for item in self._reranking.last_ranking
            ],
        )

    async def _supply(self, query_run_id: uuid.UUID, evidence: Sequence[RetrievedChunk]) -> None:
        """Record the context as `supplied`, immediately before an agent is handed it (M-8).

        `selected` and `supplied` are distinct on purpose: a run that fails or is cancelled
        between the cut and the next agent call has a context nobody received, and recording it
        as supplied would claim an agent saw evidence it never did.
        """
        await self._persist_evidence(
            query_run_id, [(item, SelectionState.SUPPLIED) for item in evidence]
        )

    async def _persist_evidence(
        self,
        query_run_id: uuid.UUID,
        evidence: Sequence[tuple[RetrievedChunk, SelectionState]],
    ) -> None:
        """Upsert one row per chunk, recording how far it actually got (M-8, N-003).

        Every row used to be written `selected_for_context=True`, so the column recorded nothing
        and "what evidence did this answer see" was unanswerable from the database. Each state is
        the caller's honest claim about that chunk.
        """
        now = utc_now()
        rows = [
            RetrievedEvidence(
                id=new_uuid(),
                query_run_id=query_run_id,
                chunk_id=item.chunk_id,
                source=item.source,
                semantic_score=item.semantic_score,
                lexical_score=item.lexical_score,
                hybrid_score=item.hybrid_score,
                rerank_score=item.rerank_score,
                rank=item.rank,
                # Derived, so the legacy boolean can never disagree with the state beside it.
                selected_for_context=state in {SelectionState.SELECTED, SelectionState.SUPPLIED},
                selection_state=state,
                round_index=item.round_index,
                round_rank=item.round_rank,
                created_at=now,
            )
            for item, state in evidence
        ]
        await self._queries.upsert_evidence(rows)
        await self._session.commit()

    async def _persist_citations(
        self,
        query_run: QueryRun,
        decision: FinalDecisionOutput,
        evidence: Sequence[RetrievedChunk],
    ) -> list[CitationDraft]:
        # L-1: the answer text is the second rule. Without it, citations that the answer never
        # referred to were still rendered as supporting references.
        valid, report = validate_citations(decision.citations, evidence, decision.answer_markdown)
        self._citation_report = report
        now = utc_now()
        by_chunk = {item.chunk_id: item for item in evidence}
        rows = [
            Citation(
                id=new_uuid(),
                query_run_id=query_run.id,
                chunk_id=draft.chunk_id,
                document_id=by_chunk[draft.chunk_id].document_id,
                citation_key=draft.citation_key,
                snippet=draft.snippet,
                source_filename=by_chunk[draft.chunk_id].source_filename,
                page_number=by_chunk[draft.chunk_id].page_number,
                relevance_score=relevance_of(by_chunk[draft.chunk_id]),
                created_at=now,
            )
            for draft in valid
        ]
        await self._queries.add_citations(rows)
        await self._session.commit()
        return valid

    async def _persist_evaluation(
        self, query_run_id: uuid.UUID, evaluation: EvaluatorOutput
    ) -> None:
        await self._queries.add_evaluation(
            EvaluationResult(
                id=new_uuid(),
                query_run_id=query_run_id,
                answer_relevance=evaluation.answer_relevance,
                faithfulness=evaluation.faithfulness,
                citation_accuracy=evaluation.citation_accuracy,
                context_sufficiency=evaluation.context_sufficiency,
                hallucination_risk=evaluation.hallucination_risk,
                explanation=evaluation.explanation,
                created_at=utc_now(),
            )
        )
        await self._session.commit()

    async def _persist_answer(
        self,
        query_run: QueryRun,
        decision: FinalDecisionOutput,
        provider: str,
        model: str,
    ) -> None:
        """Durably write the answer and its provenance, while the run is still RUNNING.

        Split out of `_finish` so the answer lands before its derivatives (N-006). The status is
        deliberately **not** set here: the run is not complete until the evaluator has had its
        turn, and marking it complete early would make a cancelled run look successful.
        """
        query_run.answer_markdown = decision.answer_markdown
        # Denormalised headline for the dashboard/report service (SDD 5.3); the
        # per-agent rows on `agent_runs` remain the source of truth. "" -> NULL, same
        # rule as `_close_agent_run`.
        query_run.resolved_provider = provider or None
        query_run.resolved_model_name = model or None
        await self._session.commit()

    async def _finish(self, query_run: QueryRun, started: float) -> None:
        """Mark the run complete. The answer is already durable by this point. The row is read
        again under its lock first, as every other terminal writer reads it (RMD-DEF-035): a run
        the reconciler or a cancel closed meanwhile keeps that outcome, and no second terminal
        event is emitted."""
        run_id = query_run.id
        current = await self._queries.get_locked(run_id)
        if current is None or current.status in TERMINAL_QUERY_RUN_STATUSES:
            logger.warning(
                "Not marking query run %s completed: %s",
                run_id,
                "it no longer exists" if current is None else f"already {current.status.value}",
            )
            await self._session.commit()
            return
        current.status = QueryRunStatus.COMPLETED
        current.latency_ms = _elapsed_ms(started)
        current.token_usage_estimate = self._usage.total_tokens
        await self._session.commit()
        await self._emitter.emit(EventType.ANSWER_COMPLETED, "Analysis complete.")

    async def _fail(self, query_run: QueryRun, message: str, started: float) -> None:
        # F-001: the id is read **before** the rollback. A rollback with a transaction open
        # expires every instance the session holds, `query_run` included, and reading an expired
        # attribute lazy-loads outside the greenlet -- so `query_run.id` after the rollback raised
        # `MissingGreenlet`, the failure writer itself failed, and the run's own message was lost
        # to the job's generic one. Nothing but this rollback expires it (`expire_on_commit` is
        # off), so the attribute is still loaded here.
        run_id = query_run.id
        await self._session.rollback()
        target = await self._queries.get(run_id)
        if target is None:
            logger.warning("Not marking query run %s failed: it no longer exists", run_id)
            return
        if target.status in TERMINAL_QUERY_RUN_STATUSES:
            # A committed terminal status is final, for every writer -- not just for
            # `jobs._mark_cancelled`, which is where this guard first landed. `_finish` commits
            # `COMPLETED` and only *then* awaits `emit(ANSWER_COMPLETED)`; an ordinary exception
            # on that trailing INSERT (or a `CancelledError` replaced by a rollback error while
            # it unwinds) arrives here with the run already durably completed. Relabelling it
            # would falsify the audit trail and hide an answer that is already persisted behind
            # the UI's error alert.
            logger.warning(
                "Not marking query run %s failed: already %s", target.id, target.status.value
            )
            return
        target.status = QueryRunStatus.FAILED
        target.error_message = message
        target.latency_ms = _elapsed_ms(started)
        # L-12: `token_usage_estimate` had exactly one writer, `_finish`, which runs only on the
        # success path -- so every failed run recorded NULL, which reads as free. A run that
        # spent nothing still records 0 rather than NULL, because "no attempt was made" and
        # "we did not look" are different claims.
        target.token_usage_estimate = self._usage.total_tokens
        await self._session.commit()
        await self._emitter.emit(EventType.WORKFLOW_FAILED, message)


def relevance_of(chunk: RetrievedChunk) -> float:
    """The relevance the report shows for a citation, on the reranker's 0..1 scale when there is
    one and on the hybrid scale otherwise (C-11).

    `rerank_score or hybrid_score` treated a **legitimate zero** as absent, because 0.0 is falsy
    -- so a chunk the reranker judged outright irrelevant was silently promoted to its hybrid
    score, which is usually much higher. `_sort_key`, one file away in the reranking service,
    always used `is not None`; the two disagreed about what a zero means.

    The two scales are different and the column shows whichever applies: a `rerank_score` is the
    model's 0..1 relevance judgement, a `hybrid_score` is the normalised retrieval score. A
    citation showing the hybrid score is one the reranker never scored -- not one it scored zero.
    """
    return chunk.rerank_score if chunk.rerank_score is not None else chunk.hybrid_score


def _guarded_question(query_run: QueryRun, plan: PlannerOutput) -> str:
    """The question every later agent is given, with no claim the planner could not know (FR-002).

    The planner runs before retrieval, so it has seen no evidence; a normalization that says
    evidence was or was not supplied (the pre-registered L-13 detector) is replaced by the user's
    own question. The planner's row keeps what it wrote, so the claim stays measurable.
    """
    normalized = plan.normalized_question or query_run.question
    if claims_unseen_evidence(normalized):
        logger.info(
            "Planner premise guard fired for run %s: the normalized question claimed something "
            "about evidence, so the user's question is used instead",
            query_run.id,
            extra={"event_type": "planner_premise_guarded"},
        )
        return query_run.question
    return normalized


# One bracket on one line; its members are separated by commas or semicolons.
_BRACKET = re.compile(r"\[([^\[\]\n]+)\]")


def relabel_keys_outside_context(
    decision: FinalDecisionOutput, context: Sequence[RetrievedChunk]
) -> tuple[FinalDecisionOutput, list[str]]:
    """No key reaches the reader without a citation that can be kept (FR-016, L-1).

    After an expansion the union rerank can cut a round-one chunk that the researcher was handed
    and a finding cites; the decision then cites it too. ``validate_citations`` drops that draft,
    by design, because every citation comes from the final context (C-10), yet the key stayed in
    the prose as a reference to nothing. Each such key's bracketed references become
    ``UNLABELLED_SOURCE``, as L-2 does for an unnamed identifier, unless a draft whose chunk is in
    the final context uses the same key. Keys compare without case, as anchoring does. The drafts
    themselves are left alone: they are dropped downstream exactly as before.

    A key is a reference wherever it is a member of a bracket, not only when it fills one: a live
    answer wrote ``[strategy-report-baseline, second-hand]`` (DV-RX-05). There only that member
    becomes ``source``, and the bracket's other members and separators stay as written.

    Returns the decision and the keys it relabelled.
    """
    in_context = {item.chunk_id for item in context}
    kept = {c.citation_key.lower() for c in decision.citations if c.chunk_id in in_context}
    outside = sorted(
        {
            c.citation_key
            for c in decision.citations
            if c.chunk_id not in in_context and c.citation_key.lower() not in kept
        }
    )
    if not outside:
        return decision, []
    cut = {key.lower() for key in outside}

    def relabel_bracket(match: re.Match[str]) -> str:
        # Members alternate with their separators: parts[0], sep, parts[2], sep, ...
        parts = re.split(r"([,;])", match.group(1))
        members = parts[0::2]
        if not any(member.strip().lower() in cut for member in members):
            return match.group(0)
        if len(members) == 1:
            return UNLABELLED_SOURCE
        rebuilt = [
            part.replace(part.strip(), "source")
            if index % 2 == 0 and part.strip().lower() in cut
            else part
            for index, part in enumerate(parts)
        ]
        return f"[{''.join(rebuilt)}]"

    def relabel(text: str) -> str:
        return _BRACKET.sub(relabel_bracket, text)

    relabelled = decision.model_copy(
        update={
            "answer_markdown": relabel(decision.answer_markdown),
            "recommendation": relabel(decision.recommendation),
            "uncertainty": relabel(decision.uncertainty),
            "contradictory_evidence": [relabel(item) for item in decision.contradictory_evidence],
            "next_steps": [relabel(step) for step in decision.next_steps],
        }
    )
    return relabelled, outside


def _failure_excerpt(exc: BaseException) -> str:
    """A bounded, redacted excerpt of what actually went wrong (L-6).

    ``bounded_excerpt`` redacts first and truncates second, so a credential in a vendor body
    cannot be cut below the redactor's 20-character floor and survive. An ``LLMProviderError``
    carries a message already built to be user-safe; anything else contributes its own ``str``,
    which is the part a reader needs and the part that used to be discarded entirely.

    An exhausted ladder adds its failing fields (RE2E-DEF-010): the schema's field names and
    Pydantic's error types only, so they are appended after the redaction, which would otherwise
    mask a long field name, and the whole is still bounded.
    """
    excerpt = bounded_excerpt(f"{type(exc).__name__}: {exc}")
    if isinstance(exc, AgentOutputValidationError) and exc.validation_errors:
        listed = "; ".join(error.render() for error in exc.validation_errors)
        excerpt = f"{excerpt} Field errors: {listed}"
        if len(excerpt) > EXCERPT_CHAR_LIMIT:
            excerpt = excerpt[: EXCERPT_CHAR_LIMIT - len(_TRUNCATED)] + _TRUNCATED
    return excerpt


# The marker `bounded_excerpt` ends a cut excerpt with.
_TRUNCATED = "... [truncated]"


def _routing_profile(query_run: QueryRun) -> RoutingProfile:
    """The named profile this run routes by (C-16, SDD 10.3, ADR-005).

    Replaces the single ``_synthesis_role`` tier, which mapped one ``ModelRole`` onto the critic,
    the risk agent, the final-decision agent and the evaluator together -- so ``answer_model=fast``
    put all four on the small model while the SDD section its own comment cited put all four on
    the stronger one.

    ``query_runs.routing_profile`` is NULL for a row written before profiles existed, and
    ``answer_model`` is what such a row has, so the alias is what resolves it. A legacy run
    therefore keeps behaving exactly as it did, which is the point of keeping the alias.
    """
    return resolve(query_run.routing_profile, AnswerModel(query_run.answer_model))


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


# The metering row's `thinking` value for a mode's switch (PLV8-T02).
_THINKING: Mapping[bool, str] = {True: "enabled", False: "disabled"}


def _add_usage(total: TokenUsage, usage: TokenUsage) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=total.prompt_tokens + usage.prompt_tokens,
        completion_tokens=total.completion_tokens + usage.completion_tokens,
    )
