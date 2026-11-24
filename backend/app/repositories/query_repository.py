"""SQL access for query runs and their child rows (agent runs, evidence, citations)."""

import uuid
from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import func, literal, select, tuple_
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import load_only

from app.domain.enums import QueryRunStatus
from app.domain.models import (
    AgentRun,
    Citation,
    EvaluationResult,
    LLMCall,
    QueryRun,
    RetrievedEvidence,
)

# Namespace for query admission's advisory lock, occupying the ``classid`` half of PostgreSQL's
# two-key advisory lock space (the storage lock takes its own, ``document_repository``'s).
# Advisory locks are a single global namespace shared by everything connected to the database,
# and a collision between two unrelated uses of it is
# silent and close to undiagnosable: one feature simply waits on another for no visible reason.
# Reserving a ``classid`` costs nothing and makes that impossible by construction. The value is
# the ASCII of "QuAd" (query admission), chosen only to be memorable in ``pg_locks`` output.
ADVISORY_LOCK_NAMESPACE = 0x51754164


def _admission_lock_key(project_id: uuid.UUID) -> int:
    """A stable signed-int4 key for ``project_id``.

    Stable is the operative word: Python's ``hash`` is salted per process (``PYTHONHASHSEED``),
    so two workers would derive different keys for the same project and would not exclude one
    another at all. Folding the UUID's own bits is deterministic everywhere and needs no digest,
    since a v4 UUID is already uniform.

    The ``objid`` half of a two-key advisory lock is a *signed* 32-bit integer, hence the shift
    into range. Thirty-two bits means two distinct projects can share a key; the consequence is
    that their admissions serialise with each other, for the microseconds admission holds the
    lock. It is never incorrect -- the count each one then takes is still filtered by
    ``project_id`` -- so the collision costs a little ordering and nothing else.
    """
    folded = project_id.int ^ (project_id.int >> 64)
    folded = (folded ^ (folded >> 32)) & 0xFFFFFFFF
    return folded - 0x80000000


class QueryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def lock_project_admission(self, project_id: uuid.UUID) -> None:
        """Serialise query admission for one project, for the rest of this transaction.

        The cap in ``QueryService._reject_if_at_capacity`` is a check-then-act: it counts the
        project's in-flight runs and only afterwards does ``create`` insert and commit the new
        one. Nothing in between made that pair atomic -- ``query_runs`` carries a plain index and
        no constraint -- so N simultaneous requests for one project all counted the same
        pre-insert total, all found room under the cap, and all were admitted. The endpoint is
        anonymous, so N is the caller's to choose, and each admitted run commits 7-20 billed
        vendor completions.

        A transaction-scoped advisory lock keyed on the project, rather than the alternatives:

        * ``SELECT ... FROM projects WHERE id = :id FOR UPDATE`` would serialise admission
          correctly, but it does so by joining the lock graph that every other writer of that row
          is already in. Inserting a document, a chunk or a query run takes ``FOR KEY SHARE`` on
          the parent ``projects`` row to satisfy the foreign key, and ``FOR UPDATE`` conflicts
          with ``FOR KEY SHARE`` -- so admission would queue behind the whole of an ingestion
          transaction, and ingestion behind admission, which is both slow and one inconsistent
          lock ordering away from a genuine deadlock. An advisory lock names a number, not a row,
          so it cannot conflict with anything that is not this function.
        * a partial unique index or an exclusion constraint cannot express "at most N"; a unique
          index expresses "at most one". Encoding N would mean a slot column and a migration, and
          N is a runtime setting (``max_running_queries_per_project``) that a constraint would
          freeze into the schema.

        Deadlock: this lock has its own ``classid``, so it never waits on the application's other
        advisory lock (``DocumentRepository.lock_project_storage``, which takes another), it is
        taken once per transaction, and no transaction takes both -- so no cycle can form among
        advisory locks. Nor can one form with row locks: while holding this, admission
        only counts ``query_runs`` (no row locks) and inserts one row (``FOR KEY SHARE`` on the
        project). A transaction that blocks admission that way -- deleting the project -- never
        requests this lock, so there is no cycle to close.

        Lock-holder death: ``pg_advisory_xact_lock`` releases at COMMIT or ROLLBACK, and
        unconditionally when the backend exits, including a crashed client or a dropped
        connection. There is no residue and no way to wedge a project permanently -- which is the
        failure ``_reject_if_at_capacity``'s deadline window exists to prevent for ``RUNNING``
        rows, and which the session-scoped ``pg_advisory_lock`` would have reintroduced, since
        that one outlives the transaction and would leak across a pooled connection's next use.

        Unrelated projects: the key is derived from ``project_id``, so admissions for different
        projects do not wait on one another (see ``_admission_lock_key`` for the collision case).
        """
        key = _admission_lock_key(project_id)
        await self._session.execute(
            select(func.pg_advisory_xact_lock(ADVISORY_LOCK_NAMESPACE, key))
        )

    async def add(self, query_run: QueryRun) -> None:
        self._session.add(query_run)
        await self._session.flush()

    async def get(self, query_run_id: uuid.UUID) -> QueryRun | None:
        return await self._session.get(QueryRun, query_run_id)

    async def get_locked(self, query_run_id: uuid.UUID) -> QueryRun | None:
        """The run's row read fresh and locked until the session commits, as every terminal
        writer reads it (RMD5-T01, RMD-DEF-035)."""
        return await self._session.get(
            QueryRun, query_run_id, with_for_update=True, populate_existing=True
        )

    async def list_by_project(self, project_id: uuid.UUID) -> list[QueryRun]:
        result = await self._session.scalars(
            select(QueryRun)
            .where(QueryRun.project_id == project_id)
            .order_by(QueryRun.created_at.desc())
        )
        return list(result.all())

    async def list_summaries_by_project(
        self,
        project_id: uuid.UUID,
        *,
        limit: int,
        status: QueryRunStatus | None = None,
        before: tuple[datetime, uuid.UUID] | None = None,
    ) -> list[QueryRun]:
        """Up to ``limit + 1`` of the project's runs strictly after ``before`` in
        ``(created_at DESC, id DESC)`` order, loading only the summary columns.

        The extra row tells the caller whether another page exists without a second query. The
        row-value comparison matches ``ix_query_runs_project_id_created_at_id`` (migration 0011),
        so a page reads a bounded number of rows at any depth instead of sorting the whole history.
        The id breaks ties between runs created in the same microsecond.
        """
        statement = (
            select(QueryRun)
            .options(
                load_only(
                    QueryRun.id,
                    QueryRun.project_id,
                    QueryRun.question,
                    QueryRun.status,
                    QueryRun.answer_model,
                    QueryRun.resolved_provider,
                    QueryRun.resolved_model_name,
                    QueryRun.error_message,
                    QueryRun.created_at,
                )
            )
            .where(QueryRun.project_id == project_id)
            .order_by(QueryRun.created_at.desc(), QueryRun.id.desc())
            .limit(limit + 1)
        )
        if status is not None:
            statement = statement.where(QueryRun.status == status)
        if before is not None:
            created_at, row_id = before
            statement = statement.where(
                tuple_(QueryRun.created_at, QueryRun.id)
                < tuple_(
                    literal(created_at, QueryRun.created_at.type),
                    literal(row_id, QueryRun.id.type),
                )
            )
        result = await self._session.scalars(statement)
        return list(result.all())

    async def count_by_project(
        self, project_id: uuid.UUID, *, status: QueryRunStatus | None = None
    ) -> int:
        """How many of the project's runs there are, optionally in one ``status``."""
        statement = select(func.count()).select_from(QueryRun)
        statement = statement.where(QueryRun.project_id == project_id)
        if status is not None:
            statement = statement.where(QueryRun.status == status)
        counted = await self._session.scalars(statement)
        return counted.one()

    async def count_by_project_and_status(
        self,
        project_id: uuid.UUID,
        status: QueryRunStatus,
        *,
        created_after: datetime | None = None,
    ) -> int:
        """How many of the project's runs are in ``status``, optionally only recent ones.

        A COUNT rather than ``len(await list_by_project(...))``: the accept path calls this on
        every query creation to decide whether the project already has its maximum number of
        workflows in flight, and hydrating every historical row to count a handful of them would
        make that check grow with the project's history. ``ix_query_runs_project_id``
        (``domain/models.py``) already covers the predicate's leading column.

        ``created_after`` bounds the count to rows created at or after that instant. See
        ``QueryService._reject_if_at_capacity`` for why "in flight" has to mean *recent* and
        ``RUNNING`` rather than ``RUNNING`` alone.
        """
        statement = select(func.count()).select_from(QueryRun)
        statement = statement.where(QueryRun.project_id == project_id, QueryRun.status == status)
        if created_after is not None:
            statement = statement.where(QueryRun.created_at >= created_after)
        counted = await self._session.scalars(statement)
        return counted.one()

    async def add_agent_run(self, agent_run: AgentRun) -> None:
        self._session.add(agent_run)
        await self._session.flush()

    async def get_agent_run(self, agent_run_id: uuid.UUID) -> AgentRun | None:
        """For the two-phase write: the row is inserted RUNNING, then resolved in place.

        Resolving in place rather than inserting a second row is what keeps "exactly one row per
        attempted component" true -- a second row would double every step in the trace and make
        `list_agent_runs` unusable for counting attempts.
        """
        return await self._session.get(AgentRun, agent_run_id)

    async def list_agent_runs(self, query_run_id: uuid.UUID) -> list[AgentRun]:
        result = await self._session.scalars(
            select(AgentRun)
            .where(AgentRun.query_run_id == query_run_id)
            .order_by(AgentRun.started_at)
        )
        return list(result.all())

    async def add_llm_call(self, call: LLMCall) -> None:
        """One row per provider HTTP attempt (M-7, L-12, C-12)."""
        self._session.add(call)
        await self._session.flush()

    async def list_llm_calls(self, query_run_id: uuid.UUID) -> list[LLMCall]:
        result = await self._session.scalars(
            select(LLMCall)
            .where(LLMCall.query_run_id == query_run_id)
            .order_by(LLMCall.created_at, LLMCall.rung, LLMCall.attempt)
        )
        return list(result.all())

    async def total_tokens_for(self, query_run_id: uuid.UUID) -> int | None:
        """Summed tokens across every recorded attempt, or ``None`` when nothing was recorded.

        ``None`` rather than ``0``: a run with no metering rows spent nothing *that we know of*,
        and a run whose vendor reported no usage is not a free run. Returning ``0`` for either
        is the claim that made ``token_usage_estimate`` unusable for reconciliation.
        """
        result = await self._session.execute(
            select(
                func.sum(
                    func.coalesce(LLMCall.prompt_tokens, 0)
                    + func.coalesce(LLMCall.completion_tokens, 0)
                )
            ).where(LLMCall.query_run_id == query_run_id)
        )
        total = result.scalar_one_or_none()
        return None if total is None else int(total)

    async def upsert_evidence(self, evidence: Sequence[RetrievedEvidence]) -> None:
        """One row per chunk per run, whichever round found it (N-003).

        An upsert rather than an insert because a chunk retrieved in round one and again in
        round two is **one** piece of evidence. Writing both left the same chunk twice with two
        different ranks, so "how much evidence did this run gather" had no single answer.

        On conflict the later write wins for the ranking fields: an expansion reranks the union,
        so round two's scores are the ones computed over the full candidate set and supersede
        round one's. `round_index` is deliberately **not** updated -- it records where the chunk
        was *first* discovered, which the second write does not know better.
        """
        if not evidence:
            return
        rows = [
            {
                "id": row.id,
                "query_run_id": row.query_run_id,
                "chunk_id": row.chunk_id,
                "source": row.source,
                "semantic_score": row.semantic_score,
                "lexical_score": row.lexical_score,
                "hybrid_score": row.hybrid_score,
                "rerank_score": row.rerank_score,
                "rank": row.rank,
                "selected_for_context": row.selected_for_context,
                "selection_state": row.selection_state,
                "round_index": row.round_index,
                "round_rank": row.round_rank,
                "created_at": row.created_at,
            }
            for row in evidence
        ]
        statement = pg_insert(RetrievedEvidence).values(rows)
        await self._session.execute(
            statement.on_conflict_do_update(
                constraint="uq_retrieved_evidence_query_run_chunk",
                set_={
                    "source": statement.excluded.source,
                    "semantic_score": statement.excluded.semantic_score,
                    "lexical_score": statement.excluded.lexical_score,
                    "hybrid_score": statement.excluded.hybrid_score,
                    "rerank_score": statement.excluded.rerank_score,
                    "rank": statement.excluded.rank,
                    "selected_for_context": statement.excluded.selected_for_context,
                    "selection_state": statement.excluded.selection_state,
                    "round_rank": statement.excluded.round_rank,
                },
            )
        )
        await self._session.flush()

    async def add_citations(self, citations: Sequence[Citation]) -> None:
        self._session.add_all(list(citations))
        await self._session.flush()

    async def list_citations(self, query_run_id: uuid.UUID) -> list[Citation]:
        result = await self._session.scalars(
            select(Citation)
            .where(Citation.query_run_id == query_run_id)
            .order_by(Citation.citation_key)
        )
        return list(result.all())

    async def add_evaluation(self, evaluation: EvaluationResult) -> None:
        self._session.add(evaluation)
        await self._session.flush()

    async def get_evaluation(self, query_run_id: uuid.UUID) -> EvaluationResult | None:
        result = await self._session.scalars(
            select(EvaluationResult).where(EvaluationResult.query_run_id == query_run_id)
        )
        return result.first()
