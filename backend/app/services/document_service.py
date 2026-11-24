"""Document use cases: upload, listing, deletion, reindex (FR-DOC-001)."""

import logging
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Protocol

from app.core.config import Settings
from app.core.errors import (
    ConflictError,
    DocumentNotFoundError,
    FileTooLargeError,
    ProjectNotFoundError,
)
from app.core.security import sanitize_display_filename, validate_upload
from app.domain.enums import DocumentStatus
from app.domain.models import ControlledDocument, Document
from app.integrations.qdrant.client import QdrantService
from app.schemas.pharma import ControlledDocumentRead
from app.services.interfaces import (
    ChunkRepositoryLike,
    DocumentRepositoryLike,
    ProjectRepositoryLike,
    SupportsCommit,
)
from app.utils.files import FileStorage
from app.utils.ids import new_uuid
from app.utils.time import utc_now

_MAX_ORIGINAL_FILENAME_LENGTH = 255

logger = logging.getLogger(__name__)


def _ingestion_alive(document: Document, now: datetime, settings: Settings) -> bool:
    """Whether an ingestion may still be running ``document``: it is ``PROCESSING`` and its lease
    is current, or, with no lease, its last update is newer than ``ingestion_stale_after_seconds``
    -- the rule by which the reconciler leaves it alone (RMD5-T02)."""
    if document.status is not DocumentStatus.PROCESSING:
        return False
    if document.lease_expires_at is not None:
        return document.lease_expires_at >= now
    return document.updated_at >= now - timedelta(seconds=settings.ingestion_stale_after_seconds)


class ControlledMetadataReader(Protocol):
    """Reads stored controlled-document rows (``PharmaRepository`` satisfies it)."""

    async def controlled_for_documents(
        self, document_ids: Sequence[uuid.UUID]
    ) -> Mapping[uuid.UUID, ControlledDocument]: ...


class DocumentService:
    def __init__(
        self,
        session: SupportsCommit,
        projects: ProjectRepositoryLike,
        documents: DocumentRepositoryLike,
        chunks: ChunkRepositoryLike,
        storage: FileStorage,
        settings: Settings,
        qdrant: QdrantService,
        controlled: ControlledMetadataReader | None = None,
    ) -> None:
        """``controlled`` reads pharmaceutical documents' controlled metadata; ``None`` (the
        default, and every construction that predates it) reads none, so every document's
        ``controlled_metadata`` is null."""
        self._session = session
        self._projects = projects
        self._documents = documents
        self._chunks = chunks
        self._storage = storage
        self._settings = settings
        self._qdrant = qdrant
        self._controlled = controlled

    async def upload(
        self,
        project_id: uuid.UUID,
        original_filename: str,
        declared_mime_type: str,
        content: bytes,
    ) -> Document:
        await self._require_project(project_id)
        validated = validate_upload(
            original_filename,
            declared_mime_type,
            content,
            max_file_size_bytes=self._settings.max_upload_size_bytes,
        )
        # RMD6-T03 (RE2E-DEF-023): held from the total's read to the commit below, so a parallel
        # upload to this project reads the total this one leaves. Taken after validation, so a
        # request refused for its own content never waits for it.
        await self._documents.lock_project_storage(project_id)
        existing_bytes = await self._documents.total_size_bytes(project_id)
        if existing_bytes + len(content) > self._settings.max_project_size_bytes:
            raise FileTooLargeError(
                f"Project storage limit of {self._settings.max_project_size_mb} MB exceeded."
            )
        document_id = new_uuid()
        stored_filename = self._storage.build_stored_filename(
            document_id, validated.sanitized_filename
        )
        self._storage.save(project_id, stored_filename, content)
        now = utc_now()
        document = Document(
            id=document_id,
            project_id=project_id,
            filename=stored_filename,
            # The single write point for the copy every downstream consumer reads: the agent
            # prompts, the Qdrant payload, the report and the API schema. `validated`
            # above carries the sanitized *on-disk* name only, so this is where the
            # documented filename control has to hold for the displayed one.
            original_filename=sanitize_display_filename(
                original_filename, limit=_MAX_ORIGINAL_FILENAME_LENGTH
            ),
            mime_type=declared_mime_type.split(";", maxsplit=1)[0].strip().lower(),
            file_size_bytes=len(content),
            status=DocumentStatus.UPLOADED,
            created_at=now,
            updated_at=now,
        )
        await self._documents.add(document)
        await self._session.commit()
        return document

    async def list_for_project(self, project_id: uuid.UUID) -> list[Document]:
        await self._require_project(project_id)
        return await self._documents.list_by_project(project_id)

    async def get_document(self, document_id: uuid.UUID) -> Document:
        document = await self._documents.get(document_id)
        if document is None:
            raise DocumentNotFoundError(f"Document {document_id} does not exist.")
        return document

    async def get_detail(self, document_id: uuid.UUID) -> tuple[Document, int]:
        document = await self.get_document(document_id)
        chunk_count = await self._chunks.count_by_document(document_id)
        return document, chunk_count

    async def controlled_metadata(
        self, document_ids: Sequence[uuid.UUID]
    ) -> dict[uuid.UUID, ControlledDocumentRead]:
        """The controlled metadata of those documents that have it, read-only (REQ-DOC-08);
        attached by the list and detail reads. A document without a row is absent."""
        if self._controlled is None or not document_ids:
            return {}
        rows = await self._controlled.controlled_for_documents(document_ids)
        return {key: ControlledDocumentRead.model_validate(row) for key, row in rows.items()}

    async def delete(self, document_id: uuid.UUID) -> None:
        """Remove the Qdrant points, then the row (chunks cascade), then the stored file
        (RMD6-T01; RE2E-DEF-025, RE2E-DEF-027).

        The row used to be committed gone first, so a Qdrant failure answered 503 for a
        document that no longer existed, its points and file left with nothing naming them. The
        points go first now: a Qdrant failure raises before anything is deleted, and the 503 is
        true. A database failure after them leaves the row without points, which a reindex
        restores. The file goes last; one that cannot be removed is logged under the document's
        id, and the orphan check removes it.

        A document whose ingestion still holds a live lease is refused with 409, as a reindex is:
        the delete used to wait on the running job's locks until the job ended.
        """
        document = await self.get_document(document_id)
        if _ingestion_alive(document, utc_now(), self._settings):
            raise ConflictError(
                "Document is being processed; wait for it to finish before deleting it."
            )
        stored_filename = document.filename
        project_id = document.project_id
        # RMD-DEF-121: the read's transaction ends before the vector store is called, so no
        # pooled connection waits idle in transaction on it.
        await self._session.commit()
        await self._qdrant.delete_by_document(project_id, document_id)
        await self._documents.delete(document)
        await self._session.commit()
        try:
            self._storage.delete_file(project_id, stored_filename)
        except OSError:
            logger.warning(
                "Deleted document %s, but its file could not be removed; the orphan check "
                "removes it",
                document_id,
                exc_info=True,
            )

    async def prepare_reindex(self, document_id: uuid.UUID) -> Document:
        """Reset a document so the ingestion job can reprocess it idempotently."""
        document = await self.get_document(document_id)
        if document.status is DocumentStatus.PROCESSING:
            raise ConflictError(
                "Document is already being processed; wait for it to finish before reindexing."
            )
        document.status = DocumentStatus.UPLOADED
        document.error_message = None
        document.updated_at = utc_now()
        await self._session.commit()
        return document

    async def _require_project(self, project_id: uuid.UUID) -> None:
        if await self._projects.get(project_id) is None:
            raise ProjectNotFoundError(f"Project {project_id} does not exist.")
