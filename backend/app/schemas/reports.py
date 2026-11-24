"""Report API schemas (PRD section 8.8)."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ReportRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    query_run_id: uuid.UUID
    title: str
    markdown_content: str
    created_at: datetime


class ReportSummary(BaseModel):
    """One row of the reports list (CT-01): enough to find and open a report, without its body.

    ``GET /reports/{id}`` serves the Markdown; a list never renders it.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    query_run_id: uuid.UUID
    title: str
    created_at: datetime


class ReportPage(BaseModel):
    """A keyset page of a project's reports, newest first, with the next cursor (``None`` on the
    last page) and the project's total."""

    items: list[ReportSummary]
    next_cursor: str | None
    total: int
