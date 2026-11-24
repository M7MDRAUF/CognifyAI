"""Citation anchors built from stored rows only (REQ-CHK-04, REQ-SEC-03; threat T-08).

An anchor says where a cited passage comes from: document number, revision, title, section (or
page), chunk index and lifecycle status. Every field comes from the stored chunk row and the
stored controlled-document row; nothing here accepts model output, so a model cannot invent a
revision, promote a superseded document or relabel a section. A value the rows do not hold is
``unknown``, never guessed.
"""

import uuid
from dataclasses import dataclass
from datetime import date

from app.domain.models import Chunk, ControlledDocument
from app.pharma.enums import (
    ControlledDocumentType,
    ControlStatus,
    LifecycleStatus,
    MetadataStatus,
    SourceAuthority,
)
from app.schemas.pharma import UNKNOWN, EvidenceAnchor


@dataclass(frozen=True)
class ControlledDocumentInfo:
    """The stored controlled metadata an anchor and version-aware retrieval read."""

    document_id: uuid.UUID
    document_number: str | None
    revision: str | None
    title: str | None
    lifecycle_status: LifecycleStatus
    metadata_status: MetadataStatus
    effective_date: date | None = None
    document_type: ControlledDocumentType | None = None
    source_authority: SourceAuthority | None = None
    control_status: ControlStatus | None = None
    supersedes_number: str | None = None
    supersedes_revision: str | None = None
    area: str | None = None
    process: str | None = None
    equipment: tuple[str, ...] = ()
    # RMD3-T03 (DV-RMD-22): the site the document applies to, which each passage's label gives.
    site: str | None = None


def info_from_row(row: ControlledDocument) -> ControlledDocumentInfo:
    return ControlledDocumentInfo(
        document_id=row.document_id,
        document_number=row.document_number,
        revision=row.revision,
        title=row.title,
        lifecycle_status=row.lifecycle_status,
        metadata_status=row.metadata_status,
        effective_date=row.effective_date,
        document_type=row.document_type,
        source_authority=row.source_authority,
        control_status=row.control_status,
        supersedes_number=row.supersedes_number,
        supersedes_revision=row.supersedes_revision,
        area=row.area,
        process=row.process,
        equipment=tuple(row.equipment),
        site=row.site,
    )


def anchor_for(chunk: Chunk, info: ControlledDocumentInfo | None) -> EvidenceAnchor:
    """The anchor of the stored ``chunk`` row. ``info`` is ``None`` for a document with no
    controlled row; a row that belongs to another document is refused rather than mixed in."""
    if info is not None and info.document_id != chunk.document_id:
        msg = "controlled metadata of another document cannot anchor this chunk"
        raise ValueError(msg)
    section = chunk.section_title.strip() if chunk.section_title else ""
    return EvidenceAnchor(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        chunk_index=chunk.chunk_index,
        document_number=_known(info.document_number if info else None),
        revision=_known(info.revision if info else None),
        title=_known(info.title if info else None),
        section=section or UNKNOWN,
        page=chunk.page_number,
        lifecycle_status=info.lifecycle_status if info else LifecycleStatus.UNKNOWN,
    )


def _known(value: str | None) -> str:
    return value if value else UNKNOWN
