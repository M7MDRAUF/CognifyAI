"""SSE agent-event schema (SDD section 8.3).

No event payload may contain raw API keys or secrets; payloads carry only
workflow metadata (stage names, counts, messages).
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.domain.enums import AgentName, EventType


class AgentEventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    query_run_id: uuid.UUID
    event_type: EventType
    agent_name: AgentName | None
    # Wire v2, additive and optional. The client's `agentEventSchema` is a `z.object`, which
    # strips unknown keys, so a deployed client that predates this field ignores it and keeps
    # working -- which is exactly why identity for non-agent producers went here rather than
    # into the `agent_name` enum (ADR-007).
    component: str | None = None
    message: str
    payload: dict[str, object]
    created_at: datetime
