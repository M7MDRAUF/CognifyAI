"""SQL access for the pharmaceutical profile: controlled documents (REQ-DOC-08, REQ-PER-07) and
run records (REQ-PER-01 to REQ-PER-04)."""

import uuid
from collections.abc import Sequence

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import CognifyError
from app.domain.enums import DomainProfile
from app.domain.models import ControlledDocument, PharmaRunRecord, Project
from app.utils.time import utc_now


class PharmaRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def project_profile(self, project_id: uuid.UUID) -> DomainProfile | None:
        """The project's profile, or ``None`` when the project does not exist."""
        profile: DomainProfile | None = await self._session.scalar(
            select(Project.domain_profile).where(Project.id == project_id)
        )
        return profile

    async def upsert_controlled_document(self, row: ControlledDocument) -> None:
        """Store ``row`` as the document's only controlled row (a reindex replaces it)."""
        await self._session.execute(
            delete(ControlledDocument).where(ControlledDocument.document_id == row.document_id)
        )
        self._session.add(row)
        await self._session.flush()

    async def get_controlled(self, document_id: uuid.UUID) -> ControlledDocument | None:
        return await self._session.get(ControlledDocument, document_id)

    async def controlled_for_documents(
        self, document_ids: Sequence[uuid.UUID]
    ) -> dict[uuid.UUID, ControlledDocument]:
        if not document_ids:
            return {}
        rows = await self._session.scalars(
            select(ControlledDocument).where(ControlledDocument.document_id.in_(document_ids))
        )
        return {row.document_id: row for row in rows.all()}

    async def controlled_for_project(self, project_id: uuid.UUID) -> list[ControlledDocument]:
        rows = await self._session.scalars(
            select(ControlledDocument)
            .where(ControlledDocument.project_id == project_id)
            .order_by(ControlledDocument.document_number, ControlledDocument.revision)
        )
        return list(rows.all())

    async def create_run_record(self, record: PharmaRunRecord) -> None:
        """Open a run's record with its input; committed with the run's own row."""
        self._session.add(record)
        await self._session.flush()

    async def run_record(self, query_run_id: uuid.UUID) -> PharmaRunRecord | None:
        return await self._session.get(PharmaRunRecord, query_run_id)

    async def save_outcome(
        self,
        query_run_id: uuid.UUID,
        *,
        rule_version: str,
        evidence_snapshot: list[dict[str, object]],
        sufficiency: dict[str, object],
        conflicts: list[dict[str, object]],
        policy: dict[str, object],
        escalation: dict[str, object],
        response: dict[str, object],
    ) -> None:
        """Write a finished run's results into its record (PDP8-T03). Flushed, not committed:
        the workflow's next commit stores the record and the answer together."""
        record = await self.run_record(query_run_id)
        if record is None:
            msg = "The pharmaceutical run record is missing; no answer is shown."
            raise CognifyError(msg)
        record.rule_version = rule_version
        record.evidence_snapshot = evidence_snapshot
        record.sufficiency = sufficiency
        record.conflicts = conflicts
        record.policy = policy
        record.escalation = escalation
        record.response = response
        record.updated_at = utc_now()
        await self._session.flush()
