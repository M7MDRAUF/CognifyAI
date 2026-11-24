"""Ingestion pipeline: extract → chunk → persist (FR-DOC-002/003).

Phase 3 extends this pipeline with embedding generation and Qdrant
indexing before the document is marked as indexed. Reprocessing is
idempotent: existing chunks are replaced, never duplicated, and a chunk
whose text is unchanged keeps the vector that was already paid for
instead of buying it again.

Reprocessing is also the operator's repair tool, and that is the stronger
requirement: whatever state the index is in when a reindex starts --
complete, half-written, emptied, or belonging to a different host -- the
document ends with one live vector per stored chunk and nothing of its own
left over. Reuse is therefore conditional on the vector still being present
in the collection, never on the chunk row alone, and everything this
pipeline removes it removes only once the rows that replace it are durable.
"""

import asyncio
import contextlib
import hashlib
import logging
import time
import uuid
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from app.core.errors import (
    CognifyError,
    DocumentNotFoundError,
    ExtractionFailedError,
    QdrantOperationError,
)
from app.domain.enums import DocumentStatus
from app.domain.models import Chunk, Document
from app.integrations.parsers import extract_blocks
from app.integrations.qdrant.client import QdrantService
from app.integrations.qdrant.collections import ChunkPoint
from app.pharma.ingestion import ControlledIngestion
from app.pharma.metadata import ParsedHeader
from app.services.chunking_service import ChunkDraft, ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.interfaces import (
    ChunkRepositoryLike,
    DocumentRepositoryLike,
    SupportsCommit,
)
from app.services.run_reconciler import DocumentLease
from app.utils.files import FileStorage
from app.utils.ids import new_uuid
from app.utils.time import utc_now

logger = logging.getLogger(__name__)

# Namespace for the derived ids below. Fixed and arbitrary: uuid5 needs one, and using our own
# keeps these ids from colliding with anything else derived from the same names.
_INGESTION_NAMESPACE = uuid.UUID("2f8a1f3c-5b0e-4f7a-9c3d-6d1b0a4e77c5")

# How long a cleanup step gets to finish while this task is being cancelled. Bounded for the
# reason ``workers.jobs._DRAIN_TIMEOUT_SECONDS`` is: the cancellation that triggered the
# cleanup is usually process teardown, and a wedged database must not hold that up forever.
_CLEANUP_TIMEOUT_SECONDS = 5.0

# What an interrupted document says for itself. It has to read as retryable, because it is:
# the row is left at FAILED precisely so ``prepare_reindex`` will accept it.
INTERRUPTED_MESSAGE = (
    "Processing was interrupted before it finished. Reindex the document to try again."
)

# Appended to a failure's reason when the document still holds an earlier version's chunks
# (RMD6-T02, RE2E-DEF-035): retrieval leaves a failed document out, so the reader learns that the
# version they may remember is not what answers are using.
PREVIOUS_VERSION_NOTE = "Its previous version is not used for answers until it is processed again."


def _derive_id(name: str) -> uuid.UUID:
    return uuid.uuid5(_INGESTION_NAMESPACE, name)


def _text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _point_ids(rows: Sequence[Chunk]) -> set[uuid.UUID]:
    return {row.qdrant_point_id for row in rows if row.qdrant_point_id is not None}


class IngestionChunkRepository(ChunkRepositoryLike, Protocol):
    """``ChunkRepositoryLike`` plus the one read only this pipeline performs.

    Kept here instead of widening ``ChunkRepositoryLike``: no other service needs a
    document's stored point ids, and the narrower protocol is what every other consumer of
    the chunk repository should keep being able to satisfy.
    """

    async def point_ids_by_document(self, document_id: uuid.UUID) -> set[uuid.UUID]: ...


@dataclass(frozen=True)
class _ChunkPlan:
    """What this run will store, split by what it still has to buy.

    ``rows`` is every chunk of the new document text; ``fresh`` is the subset with no usable
    vector yet, and is the only part that reaches the provider. ``stale_point_ids`` is
    everything either store still associates with this document that ``rows`` does not refer
    to -- the previous version's points, and any orphan a failed run left in the collection
    without a row to name it.
    """

    rows: list[Chunk]
    fresh: list[Chunk]
    reused: int
    stale_point_ids: list[uuid.UUID]


class IngestionService:
    def __init__(
        self,
        session: SupportsCommit,
        documents: DocumentRepositoryLike,
        chunks: IngestionChunkRepository,
        storage: FileStorage,
        chunking: ChunkingService,
        embeddings: EmbeddingService,
        qdrant: QdrantService,
        embedding_identity: str | None = None,
        controlled: ControlledIngestion | None = None,
        lease: DocumentLease | None = None,
    ) -> None:
        """``embedding_identity`` names the vector space ``embeddings`` produces into.

        It is what makes reuse safe rather than merely cheap: a stored vector may only be
        reused for text that is byte-identical *and* was embedded by the same model, at the
        same dimension, by the same vendor. That cannot be read back off a vector -- two
        models can share a dimension -- so the caller that configured the provider states it,
        and every derived point id is bound to it. ``None`` means the caller did not state
        one, and nothing is reused: an unknown vector space is treated as a different one.

        ``controlled`` is set only for a pharmaceutical project's document (``workers.jobs``):
        it strips and stores the controlled header and chunks the body by section. ``None``,
        the case for every General document, leaves this pipeline exactly as it was.

        ``lease`` (RMD5-T02) is the document's ingestion lease, which ``ingest`` takes with the
        ``PROCESSING`` status and renews while the pipeline runs, so the reconciler can tell a
        live ingestion from one whose process died. ``workers.jobs`` always passes one; without
        it the document is judged by its last update instead.
        """
        self._session = session
        self._documents = documents
        self._chunks = chunks
        self._storage = storage
        self._chunking = chunking
        self._embeddings = embeddings
        self._qdrant = qdrant
        self._embedding_identity = embedding_identity
        self._controlled = controlled
        self._lease = lease

    async def ingest(self, document_id: uuid.UUID) -> None:
        """Process one document; failures are persisted, never raised past here.

        Cancellation is the one thing that does leave: it is recorded like any other failure
        and then re-raised. Swallowing it would report a clean shutdown for work that did not
        happen, and letting it pass unrecorded is worse still -- ``CancelledError`` is a
        ``BaseException``, so before this handler existed a redeploy or a process shutdown
        mid-pipeline left the row at PROCESSING, which is the single status
        ``DocumentService.prepare_reindex`` refuses. The document that most needed reindexing
        was the one document that could not be.

        A process that dies gets no turn at that handler either. That case is the lease's
        (RMD5-T02, RMD-DEF-004): taken with ``PROCESSING`` and renewed until the pipeline ends,
        it lapses once the process is gone, and the reconciler then ends the document ``failed``
        (``app.services.run_reconciler.reconcile_documents``) so that it can be reindexed.
        """
        document = await self._documents.get(document_id)
        if document is None:
            raise DocumentNotFoundError(f"Document {document_id} does not exist.")
        document.status = DocumentStatus.PROCESSING
        document.error_message = None
        document.updated_at = utc_now()
        if self._lease is not None:
            document.lease_expires_at = self._lease.end(document.updated_at)
        await self._session.commit()
        holder = None if self._lease is None else asyncio.create_task(self._lease.hold(document_id))
        try:
            try:
                await self._run_pipeline(document)
            except Exception as exc:
                await self._session.rollback()
                await self._mark_failed(document_id, exc)
            except BaseException as exc:
                await self._to_completion(self._session.rollback(), "roll back the interrupted run")
                await self._to_completion(
                    self._mark_failed(document_id, exc, message=INTERRUPTED_MESSAGE),
                    "record the interrupted document",
                )
                raise
        finally:
            if holder is not None:
                holder.cancel()
                await self._to_completion(holder, "stop renewing the ingestion lease")

    async def _run_pipeline(self, document: Document) -> None:
        # perf_counter, not monotonic: Windows' monotonic clock ticks every 15.6 ms or so, which
        # reads a short reindex as 0 ms.
        started = time.perf_counter()
        content = self._storage.read(document.project_id, document.filename)
        extension = document.filename.rsplit(".", maxsplit=1)[-1].lower()

        # Parsing and chunking are synchronous and CPU-bound. Called inline they hold the
        # single uvicorn event loop for their whole duration -- ingestion runs as a
        # Starlette BackgroundTask on that loop -- so no route, SSE stream or healthcheck
        # is served meanwhile. The offload gives the loop a scheduling point either way.
        def parse_and_chunk() -> list[ChunkDraft]:
            blocks = extract_blocks(content, extension)
            return self._chunking.build_chunks(blocks)

        header: ParsedHeader | None = None
        if self._controlled is None:
            drafts = await asyncio.to_thread(parse_and_chunk)
        else:
            prepared = await asyncio.to_thread(self._controlled.prepare, content, extension)
            drafts, header = prepared.drafts, prepared.header
        if not drafts:
            raise ExtractionFailedError("Document produced no usable text chunks.")

        now = utc_now()
        plan = await self._plan_chunks(document, drafts, now)

        # Idempotent reprocessing: the rows are replaced, and nothing is removed from the
        # index until the replacements are committed. Deleting first -- document-wide or by
        # id -- is what made a partial failure dangerous: ``delete_points`` batches, so an
        # interrupted delete has already removed some of the points that the rows a rollback
        # restores still name, leaving rows nothing can hydrate and no error anywhere.
        await self._chunks.delete_by_document(document.id)
        await self._chunks.add_all(plan.rows)
        if self._controlled is not None and header is not None:
            # In the same transaction as the chunk rows: a failed run rolls both back.
            await self._controlled.persist(document, header, now)

        await self._index_chunks(document, plan.fresh, now)

        document.status = DocumentStatus.INDEXED
        document.updated_at = utc_now()
        await self._session.commit()

        # Only now, and only best-effort. With the new rows durable, every id below is one no
        # row refers to any more, so the worst a failed delete can leave is a vector nobody
        # points at -- invisible to retrieval, which hydrates from the rows, and removed by
        # the next reindex, which reads the collection rather than trusting the rows. Raising
        # here instead would mark a document FAILED that is, in fact, correctly indexed.
        await self._drop_superseded_points(plan.stale_point_ids)
        # RE2E-DEF-026 (RMD9-T02): the latency this run took, read to the point the document is
        # indexed, where the line used to carry the literal 0.
        logger.info(
            "Document ingested",
            extra={
                "event_type": "document_indexed",
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "chunks_embedded": len(plan.fresh),
                "chunks_reused": plan.reused,
            },
        )

    async def _plan_chunks(
        self, document: Document, drafts: Sequence[ChunkDraft], created_at: datetime
    ) -> _ChunkPlan:
        """Work out which drafts already have a vector that is still valid for them.

        The identity of a chunk's vector is the whole of what produced it: the vector space
        (``embedding_identity`` plus the collection the point lives in), the document, the
        exact text, and which occurrence of that text this is. Both derived ids below are
        ``uuid5`` over that tuple, so ``qdrant_point_id`` equality *is* the reuse test -- a
        changed model, a changed collection or a changed character all produce a different
        id and therefore a fresh embedding, with no extra column to keep in step. Rows
        written before ids were derived carry random ones, so they match nothing and are
        re-embedded exactly once.

        The occurrence ordinal is load-bearing, not a detail: a document can legitimately
        produce two identical chunks, and without it they would collapse onto one point and
        one chunk row, losing a chunk the un-derived pipeline stored.

        A matching id is necessary for reuse and not sufficient: it says what the vector would
        be *called*, not that the collection still has it. The two claims come apart every
        time the vector store loses data the database keeps -- a collection emptied, dropped
        and recreated, restored from an empty snapshot, or a deployment pointed at another
        host holding the same collection name -- and in each of those the chunk row is the
        stale witness. Deciding reuse from the row alone turned reindex, the operator's repair
        for exactly that situation, into a silent no-op that reported success while writing
        nothing. So the collection is asked what it holds, and reuse needs both: the row that
        says this deployment wrote the point for this document, and the collection that says
        it is still there.
        """
        stored = await self._chunks.point_ids_by_document(document.id)
        live = await self._qdrant.point_ids_for_document(document.project_id, document.id)
        # One scroll per ingest, and it answers both questions this run has: what may be
        # reused, and what is left over. The leftovers are the union rather than the rows'
        # ids, because a point the collection holds and no row names is an orphan -- the kind
        # a failed run leaves -- and a reindex is the only thing that ever looks for it.
        leftovers = stored | live

        if self._embedding_identity is None:
            unkeyed = [
                self._build_chunk(document, draft, created_at, new_uuid(), new_uuid())
                for draft in drafts
            ]
            return _ChunkPlan(
                rows=unkeyed,
                fresh=unkeyed,
                reused=0,
                stale_point_ids=sorted(leftovers - _point_ids(unkeyed)),
            )

        reusable = stored & live
        vector_space = f"{self._embedding_identity}|{self._qdrant.collection_name}"
        rows: list[Chunk] = []
        fresh: list[Chunk] = []
        reused: set[uuid.UUID] = set()
        occurrences: dict[str, int] = {}
        for draft in drafts:
            digest = _text_digest(draft.text)
            ordinal = occurrences.get(digest, 0)
            occurrences[digest] = ordinal + 1
            suffix = f"{document.id}|{digest}|{ordinal}"
            chunk_id = _derive_id(f"chunk|{suffix}")
            point_id = _derive_id(f"point|{vector_space}|{suffix}")
            row = self._build_chunk(document, draft, created_at, chunk_id, point_id)
            rows.append(row)
            if point_id in reusable:
                reused.add(point_id)
            else:
                fresh.append(row)
        return _ChunkPlan(
            rows=rows,
            fresh=fresh,
            reused=len(reused),
            # Subtracted from the rows this run stores, not from what it reused: a point that
            # is being re-bought because its row was missing is in ``live`` and not in
            # ``reused``, and deleting it after the write would delete the run's own work.
            stale_point_ids=sorted(leftovers - _point_ids(rows)),
        )

    async def _index_chunks(
        self,
        document: Document,
        rows: Sequence[Chunk],
        created_at: datetime,
    ) -> None:
        """Embed one batch, write it, drop it, then take the next.

        Embedding the whole document first kept a vector for every chunk resident at once
        purely to build one proportional write. Streaming keeps one batch of vectors alive
        instead; the points, their ids and their payloads are unchanged, so what is stored
        is the same either way.

        ``rows`` is only the chunks that still need buying, so a reprocess of unchanged text
        runs this loop zero times and issues no provider request and no write at all.
        """
        batch_size = self._embeddings.batch_size
        written: list[uuid.UUID] = []
        try:
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                vectors = await self._embeddings.embed([chunk.text for chunk in batch])
                points = [
                    self._build_point(document, chunk, vector, created_at)
                    for chunk, vector in zip(batch, vectors, strict=True)
                ]
                # Recorded before the write, not after: ``upsert_points`` batches internally
                # and can fail with part of this batch already stored.
                written.extend(point.point_id for point in points)
                await self._qdrant.upsert_points(points)
        except BaseException:
            # A batched write is not atomic, and the caller rolls the chunk rows back on
            # failure. Without this, a failure part-way would leave vectors behind that no
            # chunk row can hydrate -- state the single un-batched write never produced.
            # Only what this run wrote is removed: the points it reused still belong to the
            # rows the rollback restores.
            #
            # ``BaseException`` because the realistic interruption of a long write loop is a
            # ``CancelledError`` from process teardown, which an ``except Exception`` lets
            # past without compensating at all -- the orphans this clause exists to prevent,
            # produced by the one event most likely to produce them.
            await self._to_completion(
                self._delete_quietly(written), "remove an interrupted run's vectors"
            )
            raise

    @staticmethod
    def _build_chunk(
        document: Document,
        draft: ChunkDraft,
        created_at: datetime,
        chunk_id: uuid.UUID,
        point_id: uuid.UUID,
    ) -> Chunk:
        return Chunk(
            id=chunk_id,
            project_id=document.project_id,
            document_id=document.id,
            chunk_index=draft.chunk_index,
            text=draft.text,
            char_start=draft.char_start,
            char_end=draft.char_end,
            page_number=draft.page_number,
            section_title=draft.section_title,
            token_estimate=draft.token_estimate,
            qdrant_point_id=point_id,
            created_at=created_at,
        )

    @staticmethod
    def _build_point(
        document: Document, chunk: Chunk, vector: list[float], created_at: datetime
    ) -> ChunkPoint:
        point_id = chunk.qdrant_point_id
        if point_id is None:  # pragma: no cover - set in _build_chunk
            point_id = new_uuid()
        return ChunkPoint(
            point_id=point_id,
            vector=vector,
            project_id=document.project_id,
            document_id=document.id,
            chunk_id=chunk.id,
            chunk_index=chunk.chunk_index,
            source_filename=document.original_filename,
            page_number=chunk.page_number,
            section_title=chunk.section_title,
            created_at=created_at.isoformat(),
        )

    async def _drop_superseded_points(self, point_ids: Sequence[uuid.UUID]) -> None:
        """Remove what the committed rows no longer refer to. Never fails the ingest."""
        if not point_ids:
            return
        try:
            await self._qdrant.delete_points(point_ids)
        except QdrantOperationError:
            logger.warning(
                "Could not remove %d superseded vectors; the next reindex will",
                len(point_ids),
                exc_info=True,
            )

    async def _delete_quietly(self, point_ids: Sequence[uuid.UUID]) -> None:
        with contextlib.suppress(QdrantOperationError):
            await self._qdrant.delete_points(point_ids)

    async def _to_completion(self, step: Awaitable[None], description: str) -> None:
        """Run one cleanup step to the end, even while this task is being cancelled.

        ``asyncio.wait`` rather than a bare ``await``, for the reason ``workers.jobs._drain``
        documents at length: ingestion runs as a Starlette ``BackgroundTask`` inside an anyio
        cancel scope, and anyio re-delivers its cancellation on every loop iteration until the
        task leaves the scope. A bare await would surrender at the *second* delivery with the
        cleanup half-applied -- a rollback that never ran, or a delete that removed some of a
        failed run's points and not the rest. ``asyncio.wait`` does not cancel what it waits
        on, so absorbing the re-delivery and looping lets the step finish. The deadline is the
        counterweight: teardown must not be held up by a database that has already gone away.

        Never raises. Every caller is already unwinding something more important.
        """
        task = asyncio.ensure_future(step)
        deadline = time.monotonic() + _CLEANUP_TIMEOUT_SECONDS
        while not task.done():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                task.cancel()
                logger.warning(
                    "Gave up trying to %s within %ss", description, _CLEANUP_TIMEOUT_SECONDS
                )
                return
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait([task], timeout=remaining)
        if task.cancelled():
            return
        failure = task.exception()
        if failure is not None:
            logger.warning("Could not %s: %s", description, failure)

    async def _mark_failed(
        self, document_id: uuid.UUID, exc: BaseException, *, message: str | None = None
    ) -> None:
        # ``exc_info=exc`` rather than ``logger.exception``: on the cancellation path this
        # runs inside its own task, where the ambient exception state is empty. The record names
        # the document because the reason stored for the user is fixed text (RMD6-T04): the
        # parser library's own exception, chained as the cause, is read here.
        logger.error("Ingestion of document %s failed", document_id, exc_info=exc)
        if message is None:
            message = (
                exc.message
                if isinstance(exc, CognifyError)
                else ("Unexpected error while processing the document.")
            )
        document = await self._documents.get(document_id)
        if document is None:  # pragma: no cover - deleted mid-flight
            return
        if await self._chunks.count_by_document(document_id) > 0:
            message = f"{message} {PREVIOUS_VERSION_NOTE}"
        document.status = DocumentStatus.FAILED
        document.error_message = message
        document.updated_at = utc_now()
        await self._session.commit()
