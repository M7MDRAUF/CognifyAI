"""Document API schemas (FR-DOC-001)."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.domain.enums import DocumentStatus
from app.schemas.pharma import ControlledDocumentRead


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID
    filename: str
    original_filename: str
    mime_type: str
    file_size_bytes: int
    status: DocumentStatus
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    # REQ-DOC-08: a pharmaceutical project's document's controlled metadata, read-only; null for
    # every other document. Returned by the list and detail reads; the upload and reindex
    # routes exclude it (`WITHOUT_CONTROLLED_METADATA`), so their responses stay the baseline's.
    controlled_metadata: ControlledDocumentRead | None = None


# For `response_model_exclude` on the routes whose responses predate the field.
WITHOUT_CONTROLLED_METADATA = frozenset({"controlled_metadata"})


class DocumentDetail(DocumentRead):
    chunk_count: int
