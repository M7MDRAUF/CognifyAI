"""SQL access for documents."""

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import Document

# The ``classid`` half of the storage lock's two-key advisory lock (RMD6-T03), its own rather than
# query admission's (``query_repository.ADVISORY_LOCK_NAMESPACE``), so an upload never waits on an
# admission of the same project or the reverse. The ASCII of "StoR" (storage), for ``pg_locks``.
STORAGE_LOCK_NAMESPACE = 0x53746F52


def _storage_lock_key(project_id: uuid.UUID) -> int:
    """A stable signed-int4 key for ``project_id``, folded as ``_admission_lock_key`` folds it
    (``query_repository``, which explains why ``hash`` will not do and what a collision costs:
    two projects' uploads serialise with each other, never wrongly)."""
    folded = project_id.int ^ (project_id.int >> 64)
    folded = (folded ^ (folded >> 32)) & 0xFFFFFFFF
    return folded - 0x80000000


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, document: Document) -> None:
        self._session.add(document)
        await self._session.flush()

    async def get(self, document_id: uuid.UUID) -> Document | None:
        return await self._session.get(Document, document_id)

    async def list_by_project(self, project_id: uuid.UUID) -> list[Document]:
        result = await self._session.scalars(
            select(Document)
            .where(Document.project_id == project_id)
            .order_by(Document.created_at.desc())
        )
        return list(result.all())

    async def delete(self, document: Document) -> None:
        await self._session.delete(document)
        await self._session.flush()

    async def lock_project_storage(self, project_id: uuid.UUID) -> None:
        """Serialise one project's uploads from the storage check to the row's commit, for the
        rest of this transaction (RMD6-T03, RE2E-DEF-023).

        The upload reads the project's stored total, compares it with the limit, saves the file
        and inserts its row; without this, two uploads that together cross the limit both read
        the total before either committed, and both were accepted. Held here, the second waits
        and then reads the total the first left. ``pg_advisory_xact_lock`` releases at COMMIT or
        ROLLBACK and when the connection drops, so a failed or abandoned upload leaves nothing
        held; uploads of different projects take different keys and do not wait on one another.
        """
        await self._session.execute(
            select(
                func.pg_advisory_xact_lock(STORAGE_LOCK_NAMESPACE, _storage_lock_key(project_id))
            )
        )

    async def total_size_bytes(self, project_id: uuid.UUID) -> int:
        total = await self._session.scalar(
            select(func.coalesce(func.sum(Document.file_size_bytes), 0)).where(
                Document.project_id == project_id
            )
        )
        return total or 0
