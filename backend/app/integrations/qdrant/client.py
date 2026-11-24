"""Typed wrapper around the Qdrant client.

Application code never touches the raw client: every operation goes
through ``QdrantService`` so failures surface as ``QdrantOperationError``
and every search is forced to carry a project filter.
"""

import contextlib
import logging
import uuid
from collections.abc import Sequence
from typing import Protocol, cast

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import QueryResponse, ScoredPoint
from qdrant_client.models import (
    CollectionInfo,
    Distance,
    ExtendedPointId,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    Record,
    UpdateResult,
    VectorParams,
)

from app.core.errors import QdrantOperationError
from app.core.logging import bounded_excerpt
from app.integrations.qdrant.collections import (
    PAYLOAD_INDEXES,
    VECTOR_NAME,
    ChunkPoint,
    SemanticHit,
)

# The request body must never grow with the size of the uploaded document. qdrant-client
# serialises an upsert into one JSON ``str`` and httpx then encodes that into one ``bytes``,
# both live at the same time, and neither library caps a request body -- so an un-batched
# write turned a document into a proportional body plus two full copies of it. 64 matches
# the embedding batch size, so the streaming pipeline issues one write per embedded batch,
# and it is what the client's own ``upload_points`` helper defaults to.
UPSERT_BATCH_SIZE = 64

# How many point ids one page of ``point_ids_for_document`` asks for. Larger than the write
# batch on purpose: that page carries ids and nothing else -- no vectors, no payload -- so a
# full page is tens of kilobytes rather than megabytes, and the ingestion pipeline needs the
# *whole* list before it can decide anything, so fewer round trips is strictly better here.
# The body still does not grow with the document: a bigger document costs more pages, not a
# bigger page.
SCROLL_PAGE_SIZE = 1024

logger = logging.getLogger(__name__)


class QdrantClientLike(Protocol):
    """Subset of the async Qdrant client used by the application."""

    async def get_collections(self) -> object: ...

    async def collection_exists(self, collection_name: str) -> bool: ...

    async def get_collection(self, collection_name: str) -> CollectionInfo: ...

    async def create_collection(
        self, collection_name: str, vectors_config: dict[str, VectorParams]
    ) -> bool: ...

    async def create_payload_index(
        self, collection_name: str, field_name: str, field_schema: PayloadSchemaType
    ) -> object: ...

    async def upsert(self, collection_name: str, points: Sequence[PointStruct]) -> UpdateResult: ...

    async def query_points(
        self,
        collection_name: str,
        query: list[float],
        using: str,
        query_filter: Filter,
        limit: int,
        with_payload: bool,
    ) -> QueryResponse: ...

    async def delete(
        self, collection_name: str, points_selector: Filter | PointIdsList
    ) -> UpdateResult: ...

    async def scroll(
        self,
        collection_name: str,
        scroll_filter: Filter,
        limit: int,
        offset: ExtendedPointId | None,
        with_payload: bool,
        with_vectors: bool,
    ) -> tuple[list[Record], ExtendedPointId | None]: ...

    async def close(self) -> None: ...


def build_qdrant_client(url: str) -> QdrantClientLike:
    """Build the Qdrant client from ``QDRANT_URL``.

    Supports an embedded in-process store for local development without a Qdrant
    server (no Docker required): ``:memory:`` for an ephemeral store, or
    ``local:<path>`` for a persistent on-disk store. Any other value is treated
    as an HTTP URL. The cast is the single typed boundary around the SDK, whose
    signatures are broader than the narrow surface the app uses.
    """
    stripped = url.strip()
    if stripped in {":memory:", "memory", "embedded"}:
        return cast(QdrantClientLike, AsyncQdrantClient(location=":memory:"))
    if stripped.startswith("local:"):
        return cast(QdrantClientLike, AsyncQdrantClient(path=stripped.removeprefix("local:")))
    return cast(QdrantClientLike, AsyncQdrantClient(url=stripped))


def _project_filter(project_id: uuid.UUID, document_id: uuid.UUID | None = None) -> Filter:
    conditions = [
        FieldCondition(key="project_id", match=MatchValue(value=str(project_id))),
    ]
    if document_id is not None:
        conditions.append(
            FieldCondition(key="document_id", match=MatchValue(value=str(document_id)))
        )
    return Filter(must=list(conditions))


def _operation_failed(operation: str, exc: Exception) -> QdrantOperationError:
    """The error a caller sees names the operation, never the client's own text (F-002).

    ``QdrantOperationError.message`` is user-facing: the orchestrator writes it to
    ``query_runs.error_message`` and the ``workflow_failed`` event, and the API returns it as a
    503 body. The client's exception text is whatever the transport or the server put there -- a
    URL, an echoed header, a response body -- and it used to be interpolated into that message
    verbatim. The detail is for the operator, so it goes to the log, redacted and bounded.
    """
    logger.warning(
        "Qdrant %s failed: %s",
        operation,
        bounded_excerpt(f"{type(exc).__name__}: {exc}"),
        extra={"event_type": "qdrant_operation_failed"},
    )
    return QdrantOperationError(f"Qdrant {operation} failed.")


class QdrantService:
    def __init__(self, client: QdrantClientLike, collection_name: str) -> None:
        self._client = client
        self.collection_name = collection_name

    async def check_ready(self) -> bool:
        try:
            await self._client.get_collections()
        except Exception:
            return False
        return True

    async def ensure_collection(self, dimension: int) -> None:
        """Create the collection if missing; fail fast on dimension mismatch."""
        try:
            exists = await self._client.collection_exists(self.collection_name)
            if not exists:
                await self._client.create_collection(
                    self.collection_name,
                    vectors_config={
                        VECTOR_NAME: VectorParams(size=dimension, distance=Distance.COSINE)
                    },
                )
                for field_name, field_schema in PAYLOAD_INDEXES:
                    await self._client.create_payload_index(
                        self.collection_name, field_name, field_schema
                    )
                return
            existing = await self._existing_dimension()
        except QdrantOperationError:
            raise
        except Exception as exc:
            raise _operation_failed("collection setup", exc) from exc
        if existing != dimension:
            raise QdrantOperationError(
                f"Qdrant collection '{self.collection_name}' has vector dimension "
                f"{existing}, but LLM_EMBEDDING_DIMENSIONS is {dimension}. Refusing to "
                "start; reindex into a fresh collection or fix the configuration."
            )

    async def upsert_points(self, points: Sequence[ChunkPoint]) -> None:
        """Write points in fixed-size batches, in order.

        Batching is what keeps the request bounded; it is not a performance tweak. Points
        are written in their given order and nothing about a point changes, so the stored
        result is identical to a single write of the same sequence.
        """
        for start in range(0, len(points), UPSERT_BATCH_SIZE):
            structs = [
                PointStruct(
                    id=str(point.point_id),
                    vector={VECTOR_NAME: point.vector},
                    payload=point.payload(),
                )
                for point in points[start : start + UPSERT_BATCH_SIZE]
            ]
            try:
                await self._client.upsert(self.collection_name, structs)
            except Exception as exc:
                raise _operation_failed("upsert", exc) from exc

    async def search(
        self,
        project_id: uuid.UUID,
        query_vector: list[float],
        limit: int,
        document_id: uuid.UUID | None = None,
    ) -> list[SemanticHit]:
        try:
            response = await self._client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using=VECTOR_NAME,
                query_filter=_project_filter(project_id, document_id),
                limit=limit,
                with_payload=True,
            )
        except Exception as exc:
            raise _operation_failed("search", exc) from exc
        return [_to_hit(point) for point in response.points]

    async def point_ids_for_document(
        self, project_id: uuid.UUID, document_id: uuid.UUID
    ) -> set[uuid.UUID]:
        """Which points this collection *actually* holds for one document, right now.

        The ingestion pipeline derives a point id from the text and the vector space, so a
        matching id on a chunk row says what the vector would be called -- not that it is
        still there. Those are different claims whenever the collection has been emptied,
        dropped and recreated, restored from an empty snapshot, or swapped for another host's
        collection of the same name, and in every one of those cases the row is the stale
        witness. So reuse is decided against this, and the leftovers to clean up are computed
        from it too: a point returned here that no current row refers to is an orphan no other
        read would find.

        Scrolled by payload filter rather than probed by candidate id, which costs the same
        order of round trips and answers the larger question: the id probe can only confirm
        what the rows already predicted, while this also returns the points the rows have
        forgotten. ``document_id`` and ``project_id`` are both payload-indexed
        (``PAYLOAD_INDEXES``), and ids come back without payloads or vectors.
        """
        found: set[uuid.UUID] = set()
        offset: ExtendedPointId | None = None
        while True:
            try:
                records, next_offset = await self._client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=_project_filter(project_id, document_id),
                    limit=SCROLL_PAGE_SIZE,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
            except Exception as exc:
                raise _operation_failed("scroll", exc) from exc
            for record in records:
                point_id = _as_uuid(record.id)
                if point_id is not None:
                    found.add(point_id)
            # ``None`` is the documented end of the scroll. The other two conditions are
            # termination guards, not expected outcomes: a page that returns nothing, or one
            # that hands back the cursor it was given, would otherwise loop forever.
            if next_offset is None or not records or next_offset == offset:
                return found
            offset = next_offset

    async def delete_by_project(self, project_id: uuid.UUID) -> None:
        await self._delete(_project_filter(project_id))

    async def delete_by_document(self, project_id: uuid.UUID, document_id: uuid.UUID) -> None:
        await self._delete(_project_filter(project_id, document_id))

    async def delete_points(self, point_ids: Sequence[uuid.UUID]) -> None:
        """Delete exactly these points, in bounded batches.

        The filter deletes above remove everything under a project or a document. This is the
        targeted counterpart the ingestion pipeline needs, because a reprocess that keeps a
        document's unchanged vectors must remove only the points whose text is gone -- a
        document-wide delete would throw away precisely what was not re-bought. Batched at
        ``UPSERT_BATCH_SIZE`` for the same reason writes are: the request body must not grow
        with the size of the document. Ids that no longer exist are ignored by Qdrant, which
        is what makes this safe to use as a compensating delete after a partial write.
        """
        for start in range(0, len(point_ids), UPSERT_BATCH_SIZE):
            batch = point_ids[start : start + UPSERT_BATCH_SIZE]
            selector = PointIdsList(points=[str(point_id) for point_id in batch])
            try:
                await self._client.delete(self.collection_name, selector)
            except Exception as exc:
                raise _operation_failed("delete", exc) from exc

    async def _delete(self, points_selector: Filter) -> None:
        try:
            await self._client.delete(self.collection_name, points_selector)
        except Exception as exc:
            raise _operation_failed("delete", exc) from exc

    async def _existing_dimension(self) -> int:
        info = await self._client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        if not isinstance(vectors, dict) or VECTOR_NAME not in vectors:
            raise QdrantOperationError(
                f"Qdrant collection '{self.collection_name}' is missing the named vector "
                f"'{VECTOR_NAME}'."
            )
        return vectors[VECTOR_NAME].size

    async def close(self) -> None:
        await self._client.close()


def _as_uuid(point_id: ExtendedPointId) -> uuid.UUID | None:
    """Point ids this application writes are uuids; anything else is not ours to reason about.

    The collection is shared, and Qdrant also allows integer ids, so a point written by
    something other than this pipeline can in principle come back from a scroll. Returning
    ``None`` for it keeps such a point out of both the reuse set and the delete set, which is
    the only safe thing to do with a point whose ownership is not established.
    """
    try:
        return uuid.UUID(str(point_id))
    except ValueError:
        return None


def _to_hit(point: ScoredPoint) -> SemanticHit:
    payload = point.payload or {}
    chunk_id_raw = payload.get("chunk_id")
    if not isinstance(chunk_id_raw, str):
        raise QdrantOperationError("Qdrant search result is missing a chunk_id payload field.")
    document_id_raw = payload.get("document_id")
    document_id = None
    if isinstance(document_id_raw, str):
        # A malformed id reads as unknown: retrieval then judges the point by its row alone.
        with contextlib.suppress(ValueError):
            document_id = uuid.UUID(document_id_raw)
    return SemanticHit(chunk_id=uuid.UUID(chunk_id_raw), score=point.score, document_id=document_id)
