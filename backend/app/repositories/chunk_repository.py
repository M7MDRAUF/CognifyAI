"""SQL access for chunks, including PostgreSQL full-text lexical search."""

import uuid
from collections.abc import Sequence

from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import DocumentStatus
from app.domain.models import Chunk, Document

# The statuses whose stored chunks are not the document's current version (RMD6-T02,
# RE2E-DEF-035, DV-RMD-36): failed, its last processing failed and the chunks are the previous
# version; processing, being replaced. Retrieval leaves both out.
NOT_READY_STATUSES = (DocumentStatus.FAILED, DocumentStatus.PROCESSING)


class ChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_all(self, chunks: Sequence[Chunk]) -> None:
        self._session.add_all(list(chunks))
        await self._session.flush()

    async def delete_by_document(self, document_id: uuid.UUID) -> None:
        await self._session.execute(delete(Chunk).where(Chunk.document_id == document_id))

    async def count_by_document(self, document_id: uuid.UUID) -> int:
        count = await self._session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
        )
        return count or 0

    async def list_by_document(self, document_id: uuid.UUID) -> list[Chunk]:
        result = await self._session.scalars(
            select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index)
        )
        return list(result.all())

    async def point_ids_by_document(self, document_id: uuid.UUID) -> set[uuid.UUID]:
        """The Qdrant point ids this document's chunks currently hold.

        Deliberately a scalar select rather than ``list_by_document``: the ingestion pipeline
        derives a chunk's primary key from its text, so a reprocess deletes and re-inserts the
        *same* ids inside one transaction. Loading the old rows as ORM entities would put those
        identities in the session's identity map and make the re-insert collide with them, so
        this read must not produce entities.
        """
        result = await self._session.scalars(
            select(Chunk.qdrant_point_id)
            .where(Chunk.document_id == document_id)
            .where(Chunk.qdrant_point_id.is_not(None))
        )
        return {point_id for point_id in result.all() if point_id is not None}

    async def get_many(self, chunk_ids: Sequence[uuid.UUID]) -> dict[uuid.UUID, Chunk]:
        if not chunk_ids:
            return {}
        result = await self._session.scalars(select(Chunk).where(Chunk.id.in_(list(chunk_ids))))
        return {chunk.id: chunk for chunk in result.all()}

    async def get_many_with_filename(
        self, chunk_ids: Sequence[uuid.UUID]
    ) -> dict[uuid.UUID, tuple[Chunk, str]]:
        """Return chunks joined to their document's original filename (for citations), for
        documents that are ready only (RMD6-T02): a failed or processing document's chunks are
        not its current version and are never cited."""
        if not chunk_ids:
            return {}
        result = await self._session.execute(
            select(Chunk, Document.original_filename)
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.id.in_(list(chunk_ids)), Document.status.not_in(NOT_READY_STATUSES))
        )
        return {chunk.id: (chunk, filename) for chunk, filename in result.all()}

    async def not_ready_candidates(
        self,
        chunk_ids: Sequence[uuid.UUID],
        hit_documents: dict[uuid.UUID, uuid.UUID],
    ) -> set[uuid.UUID]:
        """Of ``chunk_ids``, those whose document is failed or processing: by the chunk's own row,
        or, for a point with no row, by the document its payload names (``hit_documents``, chunk
        id to document id). Retrieval counts these as excluded, not lost (RMD6-T02, OBS-RMD-10)."""
        if not chunk_ids:
            return set()
        wanted = list(chunk_ids)
        by_row = await self._session.scalars(
            select(Chunk.id)
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.id.in_(wanted), Document.status.in_(NOT_READY_STATUSES))
        )
        excluded = set(by_row)
        named = {
            chunk_id: hit_documents[chunk_id] for chunk_id in wanted if chunk_id in hit_documents
        }
        if named:
            unready = set(
                await self._session.scalars(
                    select(Document.id).where(
                        Document.id.in_(set(named.values())),
                        Document.status.in_(NOT_READY_STATUSES),
                    )
                )
            )
            excluded |= {
                chunk_id for chunk_id, document_id in named.items() if document_id in unready
            }
        return excluded

    async def set_qdrant_point_ids(self, point_ids: dict[uuid.UUID, uuid.UUID]) -> None:
        for chunk_id, point_id in point_ids.items():
            chunk = await self._session.get(Chunk, chunk_id)
            if chunk is not None:
                chunk.qdrant_point_id = point_id
        await self._session.flush()

    async def lexical_search(
        self, project_id: uuid.UUID, query: str, limit: int
    ) -> list[tuple[uuid.UUID, float]]:
        """Rank project chunks by PostgreSQL full-text relevance (FR-RET-002).

        The tsquery is chosen in SQL from the bound text. A query using web-search syntax -- a
        double quote, a standalone ``or``, or a term starting with ``-`` -- keeps
        ``websearch_to_tsquery``'s meaning. Any other query matches **any** of its terms: the
        disjunction of ``plainto_tsquery``'s lexemes, rewritten from PostgreSQL's own output, so
        the text never leaves its bound parameter. ``websearch_to_tsquery`` joins plain terms with
        ``&``, so the planner's seven-to-ten-term queries had to find every term in one chunk and
        almost never did: the lexical half of hybrid retrieval matched nothing for 451 of 504 live
        planned queries (FR-001). ``ts_rank_cd`` still ranks, so a chunk holding more of the terms
        ranks higher.

        A failed or processing document's chunks are left out (RMD6-T02): they are not its
        current version.
        """
        statement = text(
            "SELECT chunks.id, ts_rank_cd(chunks.text_tsv, q.tsq) AS rank "
            "FROM chunks JOIN documents ON documents.id = chunks.document_id, "
            "(SELECT CASE WHEN :query ~* '(\"|(^|\\s)or(\\s|$)|(^|\\s)-\\S)' "
            "THEN websearch_to_tsquery('english', :query) "
            "ELSE replace(plainto_tsquery('english', :query)::text, ' & ', ' | ')::tsquery "
            "END AS tsq) AS q "
            "WHERE chunks.project_id = :project_id "
            "AND documents.status NOT IN (:failed, :processing) "
            "AND chunks.text_tsv @@ q.tsq "
            "ORDER BY rank DESC LIMIT :limit"
        )
        result = await self._session.execute(
            statement,
            {
                "query": query,
                "project_id": project_id,
                "limit": limit,
                "failed": DocumentStatus.FAILED.value,
                "processing": DocumentStatus.PROCESSING.value,
            },
        )
        return [(row.id, float(row.rank)) for row in result]
