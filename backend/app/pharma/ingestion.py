"""Ingestion of a pharmaceutical project's documents (REQ-DOC-01, REQ-DOC-04, REQ-DOC-07).

``IngestionService`` takes this as an optional collaborator, built only for a pharmaceutical
project's document (``workers.jobs``); for every General document it is ``None`` and the
baseline pipeline runs unchanged.

For a text document the controlled header is parsed and removed before extraction, so its
values are stored as metadata and never reach a chunk, a vector or a prompt; the body is chunked
by section (``SectionChunker``). A file type that carries no header (PDF, DOCX, CSV, XLSX) is
chunked by section the same way and stored with metadata status ``missing``. A missing or
invalid header gives lifecycle ``unknown``, which version-aware retrieval treats as not current.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from app.core.errors import ExtractionFailedError
from app.domain.models import ControlledDocument, Document
from app.integrations.parsers import extract_blocks
from app.pharma.chunking import SectionChunker
from app.pharma.enums import MetadataStatus
from app.pharma.metadata import ControlledMetadataDraft, ParsedHeader, parse_controlled_header
from app.services.chunking_service import ChunkDraft

TEXT_EXTENSIONS = frozenset({"md", "txt"})
NO_HEADER_FOR_TYPE = "this file type carries no controlled header"


class ControlledRowWriter(Protocol):
    async def upsert_controlled_document(self, row: ControlledDocument) -> None: ...


@dataclass(frozen=True)
class PreparedDocument:
    drafts: list[ChunkDraft]
    header: ParsedHeader


class ControlledIngestion:
    def __init__(
        self, repository: ControlledRowWriter, chunker: SectionChunker | None = None
    ) -> None:
        self._repository = repository
        self._chunker = chunker if chunker is not None else SectionChunker()

    def prepare(self, content: bytes, extension: str) -> PreparedDocument:
        """Parse the header and chunk the body. Synchronous and CPU-bound: the pipeline runs
        it in its worker thread, as it does the baseline parse and chunk."""
        if extension in TEXT_EXTENSIONS:
            try:
                text = content.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise ExtractionFailedError("Text file is not valid UTF-8.") from exc
            header = parse_controlled_header(text)
            body = header.body.encode("utf-8")
        else:
            header = ParsedHeader(
                ControlledMetadataDraft(), "", MetadataStatus.MISSING, (NO_HEADER_FOR_TYPE,)
            )
            body = content
        blocks = extract_blocks(body, extension)
        return PreparedDocument(self._chunker.build_chunks(blocks), header)

    async def persist(self, document: Document, header: ParsedHeader, created_at: datetime) -> None:
        await self._repository.upsert_controlled_document(row_for(document, header, created_at))


def row_for(document: Document, header: ParsedHeader, created_at: datetime) -> ControlledDocument:
    meta = header.metadata
    return ControlledDocument(
        document_id=document.id,
        project_id=document.project_id,
        document_number=meta.document_number,
        title=meta.title,
        document_type=meta.document_type,
        revision=meta.revision,
        effective_date=meta.effective_date,
        lifecycle_status=meta.lifecycle_status,
        supersedes_number=meta.supersedes_number,
        supersedes_revision=meta.supersedes_revision,
        site=meta.site,
        area=meta.area,
        equipment=list(meta.equipment),
        process=meta.process,
        product_family=meta.product_family,
        source_authority=meta.source_authority,
        control_status=meta.control_status,
        data_classification=meta.data_classification,
        metadata_status=header.status,
        metadata_issues=list(header.issues),
        created_at=created_at,
    )
