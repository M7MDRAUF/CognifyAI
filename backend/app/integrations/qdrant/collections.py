"""Qdrant collection schema for ``cognify_chunks`` (SDD section 6.2)."""

import uuid
from dataclasses import dataclass

from qdrant_client.models import PayloadSchemaType

VECTOR_NAME = "text_embedding"

PAYLOAD_INDEXES: tuple[tuple[str, PayloadSchemaType], ...] = (
    ("project_id", PayloadSchemaType.KEYWORD),
    ("document_id", PayloadSchemaType.KEYWORD),
    ("chunk_id", PayloadSchemaType.KEYWORD),
    ("chunk_index", PayloadSchemaType.INTEGER),
    ("page_number", PayloadSchemaType.INTEGER),
)


@dataclass(frozen=True)
class ChunkPoint:
    """A chunk embedding plus the payload stored alongside it in Qdrant."""

    point_id: uuid.UUID
    vector: list[float]
    project_id: uuid.UUID
    document_id: uuid.UUID
    chunk_id: uuid.UUID
    chunk_index: int
    source_filename: str
    page_number: int | None
    section_title: str | None
    created_at: str

    def payload(self) -> dict[str, str | int | None]:
        return {
            "project_id": str(self.project_id),
            "document_id": str(self.document_id),
            "chunk_id": str(self.chunk_id),
            "chunk_index": self.chunk_index,
            "source_filename": self.source_filename,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class SemanticHit:
    chunk_id: uuid.UUID
    score: float
    # The point's document, from its payload (RMD6-T02, DV-RMD-36): it lets retrieval leave out
    # a point of a document that is not ready even when the point has no row to read it from.
    document_id: uuid.UUID | None = None
