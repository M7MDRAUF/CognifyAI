"""SQL aggregates for the evaluation dashboard (PRD section 8.7)."""

import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

from pydantic import ValidationError
from sqlalchemy import ColumnElement, Integer, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import AgentName, AgentRunStatus, QueryRunStatus
from app.domain.models import AgentRun, EvaluationResult, QueryRun
from app.schemas.agents import RiskOutput

# Sum of the retriever agent's reported evidence count, stored in output_json.
# How many chunks a retriever step put into the final context.
#
# `P3-T06` replaced the scalar `{"retrieved": N}` with the retrieval envelope (M-1), whose
# equivalent field is `selected`. The legacy key is read as a fallback rather than dropped:
# rows written before that change carry only `retrieved`, and ignoring them would move a
# dashboard number for a reason that has nothing to do with the data. `coalesce` returns the
# first non-NULL, so a post-envelope row reads `selected` and a historical row reads `retrieved`.
_RETRIEVED = func.coalesce(
    cast(AgentRun.output_json["selected"].astext, Integer),
    cast(AgentRun.output_json["retrieved"].astext, Integer),
)

# The order the stages run in, which is the order the dashboard lists them in (CT-04). A retriever
# expansion round is a second row of the same stage, not a stage of its own.
STAGE_ORDER: tuple[AgentName, ...] = (
    AgentName.PLANNER,
    AgentName.RETRIEVER,
    AgentName.RESEARCHER,
    AgentName.CRITIC,
    AgentName.RISK,
    AgentName.FINAL_DECISION,
    AgentName.EVALUATOR,
)

_HALLUCINATION_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0.0-0.2", 0.0, 0.2),
    ("0.2-0.4", 0.2, 0.4),
    ("0.4-0.6", 0.4, 0.6),
    ("0.6-0.8", 0.6, 0.8),
    ("0.8-1.0", 0.8, 1.0001),
)


@dataclass(frozen=True)
class SummaryAggregates:
    query_count: int
    completed_count: int
    failed_count: int
    cancelled_count: int
    running_count: int
    avg_answer_relevance: float | None
    avg_faithfulness: float | None
    avg_citation_accuracy: float | None
    avg_context_sufficiency: float | None
    avg_latency_ms: float | None
    avg_token_usage: float | None
    retrieval_hit_count: int
    agent_failure_count: int
    quality_over_time: list[tuple[datetime, float]]
    stage_latencies: list[tuple[str, float]]
    risk_categories: list[tuple[str, int]]
    hallucination_buckets: list[tuple[str, int]]
    recent_queries: list[
        tuple[uuid.UUID, str, QueryRunStatus, datetime, float | None, float | None, str | None]
    ]


class EvaluationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def summary(self, project_id: uuid.UUID) -> SummaryAggregates:
        counts = await self._status_counts(project_id)
        averages = await self._evaluation_averages(project_id)
        latency_token = await self._latency_token_averages(project_id)
        return SummaryAggregates(
            query_count=sum(counts.values()),
            completed_count=counts.get(QueryRunStatus.COMPLETED, 0),
            failed_count=counts.get(QueryRunStatus.FAILED, 0),
            cancelled_count=counts.get(QueryRunStatus.CANCELLED, 0),
            running_count=counts.get(QueryRunStatus.RUNNING, 0),
            avg_answer_relevance=averages[0],
            avg_faithfulness=averages[1],
            avg_citation_accuracy=averages[2],
            avg_context_sufficiency=averages[3],
            avg_latency_ms=latency_token[0],
            avg_token_usage=latency_token[1],
            retrieval_hit_count=await self._retrieval_hit_count(project_id),
            agent_failure_count=await self._agent_failure_count(project_id),
            quality_over_time=await self._quality_over_time(project_id),
            stage_latencies=await self._stage_latencies(project_id),
            risk_categories=await self._risk_categories(project_id),
            hallucination_buckets=await self._hallucination_buckets(project_id),
            recent_queries=await self._recent_queries(project_id),
        )

    @staticmethod
    def _eligible(project_id: uuid.UUID) -> list[ColumnElement[bool]]:
        """The single analytics eligibility predicate (N-006, SDD 13.0.2).

        A run counts toward **quality** analytics when it is completed, produced an answer, and
        was evaluated. Three aggregates previously filtered on project alone while a fourth also
        filtered on completion, so the quality figures averaged failed and cancelled runs
        together with successful ones -- reporting an answer-relevance score for work that
        produced no answer.

        The evaluation join is what enforces the third clause: an evaluator-failed run has no
        `evaluation_results` row, and an inner join drops it. That is the intended behaviour --
        **excluded, not zero-filled**. A run with no evaluation contributes nothing to an
        average; contributing a zero would drag every mean toward the floor in proportion to how
        often the evaluator failed, which is a property of the evaluator and not of the answers.

        Returned as a list of clauses rather than a single `and_`, so each caller can add its own
        without nesting, and so a reader sees the clauses that apply at the call site.
        """
        return [
            QueryRun.project_id == project_id,
            QueryRun.status == QueryRunStatus.COMPLETED,
            QueryRun.answer_markdown.is_not(None),
        ]

    async def _status_counts(self, project_id: uuid.UUID) -> dict[QueryRunStatus, int]:
        """Every run in the project, by status. **Deliberately not eligibility-filtered.**

        SDD 13.0.2 permits this where the aggregate documents why. This is an activity figure, not
        a quality figure: the dashboard's query total and its detail line exist to account for
        every run the project has had, completed, failed, cancelled or still running (FR-004).
        Filtering it would make failures and cancellations disappear from the one count meant to
        show them.
        """
        rows = await self._session.execute(
            select(QueryRun.status, func.count())
            .where(QueryRun.project_id == project_id)
            .group_by(QueryRun.status)
        )
        return {row[0]: row[1] for row in rows.all()}

    async def _evaluation_averages(
        self, project_id: uuid.UUID
    ) -> tuple[float | None, float | None, float | None, float | None]:
        row = (
            await self._session.execute(
                select(
                    func.avg(EvaluationResult.answer_relevance),
                    func.avg(EvaluationResult.faithfulness),
                    func.avg(EvaluationResult.citation_accuracy),
                    func.avg(EvaluationResult.context_sufficiency),
                )
                .select_from(EvaluationResult)
                .join(QueryRun, EvaluationResult.query_run_id == QueryRun.id)
                .where(*self._eligible(project_id))
            )
        ).one()
        return (_f(row[0]), _f(row[1]), _f(row[2]), _f(row[3]))

    async def _latency_token_averages(
        self, project_id: uuid.UUID
    ) -> tuple[float | None, float | None]:
        # N-006: the evaluation clause is applied here too (SDD 13.0.2 applies the predicate to
        # every aggregate, latency and token averages included). `_eligible` enforces it only
        # through the inner join the quality aggregates make; this aggregate reads `query_runs`
        # alone, so an evaluator-failed run's latency and spend were being averaged in. An
        # `EXISTS` rather than a join, so the averages stay one row per run.
        evaluated = (
            select(EvaluationResult.id).where(EvaluationResult.query_run_id == QueryRun.id).exists()
        )
        row = (
            await self._session.execute(
                select(
                    func.avg(QueryRun.latency_ms), func.avg(QueryRun.token_usage_estimate)
                ).where(*self._eligible(project_id), evaluated)
            )
        ).one()
        return (_f(row[0]), _f(row[1]))

    async def _retrieval_hit_count(self, project_id: uuid.UUID) -> int:
        """Chunks the retriever steps of eligible runs put into their final context (FR-003).

        The SDD 13.0.2 predicate applies: a completed run with an answer and an evaluation
        (``_eligible`` plus the same ``EXISTS`` as ``_latency_token_averages``). A cancelled run's
        retriever step, or a failed run's, retrieved evidence no answer was judged on.
        """
        evaluated = (
            select(EvaluationResult.id).where(EvaluationResult.query_run_id == QueryRun.id).exists()
        )
        count = await self._session.scalar(
            select(func.coalesce(func.sum(_RETRIEVED), 0))
            .select_from(AgentRun)
            .join(QueryRun, AgentRun.query_run_id == QueryRun.id)
            .where(
                *self._eligible(project_id), evaluated, AgentRun.agent_name == AgentName.RETRIEVER
            )
        )
        return int(count or 0)

    async def _agent_failure_count(self, project_id: uuid.UUID) -> int:
        """Failed agent steps across every run. **Deliberately not eligibility-filtered.**

        SDD 13.0.2 permits this where the aggregate documents why. This is an activity figure, not
        a quality figure: a failed step belongs, almost by definition, to a run the predicate
        excludes (a failed evaluator leaves a run unevaluated), so filtering would report zero
        failures for exactly the runs that had them.
        """
        count = await self._session.scalar(
            select(func.count())
            .select_from(AgentRun)
            .join(QueryRun, AgentRun.query_run_id == QueryRun.id)
            .where(
                QueryRun.project_id == project_id,
                AgentRun.status == AgentRunStatus.FAILED,
            )
        )
        return int(count or 0)

    async def _quality_over_time(self, project_id: uuid.UUID) -> list[tuple[datetime, float]]:
        rows = await self._session.execute(
            select(QueryRun.created_at, EvaluationResult.answer_relevance)
            .select_from(EvaluationResult)
            .join(QueryRun, EvaluationResult.query_run_id == QueryRun.id)
            .where(*self._eligible(project_id))
            .order_by(QueryRun.created_at)
        )
        return [(created_at, float(score)) for created_at, score in rows]

    async def _stage_latencies(self, project_id: uuid.UUID) -> list[tuple[str, float]]:
        """The mean ``latency_ms`` per stage over COMPLETED steps of eligible runs (SD-01 A).

        Eligible is the one SDD 13.0.2 predicate every quality figure uses: a completed run with
        an answer and an evaluation (``_eligible`` plus the same ``EXISTS`` as
        ``_latency_token_averages``). Only COMPLETED steps count, because ``_open_agent_run``
        writes every step RUNNING with ``latency_ms = 0`` and a cancelled or deadline-stopped run
        leaves it that way for good (the harness keeps it as the honest record), while a FAILED
        attempt's latency is the cost of a failure, not of the stage. A plain mean over rows, so a
        retriever expansion round weighs as much as a first round. Stages come in pipeline order
        (``STAGE_ORDER``); a stage with no such step is omitted rather than reported as zero.
        """
        evaluated = (
            select(EvaluationResult.id).where(EvaluationResult.query_run_id == QueryRun.id).exists()
        )
        rows = await self._session.execute(
            select(AgentRun.agent_name, func.avg(AgentRun.latency_ms))
            .select_from(AgentRun)
            .join(QueryRun, AgentRun.query_run_id == QueryRun.id)
            .where(
                *self._eligible(project_id),
                evaluated,
                AgentRun.status == AgentRunStatus.COMPLETED,
                AgentRun.latency_ms.is_not(None),
            )
            .group_by(AgentRun.agent_name)
        )
        averages = {name: float(avg) for name, avg in rows if avg is not None}
        return [(stage.value, averages[stage]) for stage in STAGE_ORDER if stage in averages]

    async def _risk_categories(self, project_id: uuid.UUID) -> list[tuple[str, int]]:
        """Risk items by category over the risk steps of eligible runs (FR-003).

        The SDD 13.0.2 predicate applies (``_eligible`` plus the evaluation ``EXISTS``), so the
        chart's "after queries complete" is true: a run that failed after its risk step, or was
        cancelled, contributes no category.
        """
        evaluated = (
            select(EvaluationResult.id).where(EvaluationResult.query_run_id == QueryRun.id).exists()
        )
        rows = await self._session.scalars(
            select(AgentRun.output_json)
            .join(QueryRun, AgentRun.query_run_id == QueryRun.id)
            .where(*self._eligible(project_id), evaluated, AgentRun.agent_name == AgentName.RISK)
        )
        counter: Counter[str] = Counter()
        for output in rows:
            if output is None:
                continue
            try:
                parsed = RiskOutput.model_validate(output)
            except ValidationError:
                continue
            for risk in parsed.risks:
                counter[risk.category.value] += 1
        return sorted(counter.items())

    async def _hallucination_buckets(self, project_id: uuid.UUID) -> list[tuple[str, int]]:
        scores = await self._session.scalars(
            select(EvaluationResult.hallucination_risk)
            .join(QueryRun, EvaluationResult.query_run_id == QueryRun.id)
            .where(*self._eligible(project_id))
        )
        buckets = {label: 0 for label, _, _ in _HALLUCINATION_BUCKETS}
        for score in scores:
            for label, low, high in _HALLUCINATION_BUCKETS:
                if low <= float(score) < high:
                    buckets[label] += 1
                    break
        return [(label, buckets[label]) for label, _, _ in _HALLUCINATION_BUCKETS]

    async def _recent_queries(
        self, project_id: uuid.UUID
    ) -> list[
        tuple[uuid.UUID, str, QueryRunStatus, datetime, float | None, float | None, str | None]
    ]:
        rows = await self._session.execute(
            select(
                QueryRun.id,
                QueryRun.question,
                QueryRun.status,
                QueryRun.created_at,
                EvaluationResult.answer_relevance,
                EvaluationResult.faithfulness,
                QueryRun.error_message,
            )
            .select_from(QueryRun)
            .join(EvaluationResult, EvaluationResult.query_run_id == QueryRun.id, isouter=True)
            # **Deliberately not eligibility-filtered** (SDD 13.0.2 permits this where the
            # aggregate documents why). This is an activity feed, not a quality metric: it shows
            # the last ten runs *whatever happened to them*, and its `status` and `error_message`
            # columns exist precisely so a failed run is visible. Hiding failures here would be
            # the opposite of what an operator opens it for. The outer join is what lets an
            # unevaluated run appear with NULL scores rather than vanishing.
            .where(QueryRun.project_id == project_id)
            .order_by(QueryRun.created_at.desc())
            .limit(10)
        )
        return [
            (qid, question, status, created_at, _f(relevance), _f(faithfulness), error_message)
            for qid, question, status, created_at, relevance, faithfulness, error_message in rows
        ]


def _f(value: float | None) -> float | None:
    return float(value) if value is not None else None
