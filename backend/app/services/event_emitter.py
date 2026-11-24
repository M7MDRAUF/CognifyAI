"""Persists a workflow event and publishes it to live SSE subscribers.

Identity is assigned here, from the normative table in SDD 8.3.1, rather than at each call site.
See ``EVENT_IDENTITY`` below and ADR-007 for why non-agent producers get their own field instead
of a widened ``AgentName``.
"""

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from app.domain.enums import AgentName, EventType
from app.domain.models import AgentEvent
from app.repositories.event_repository import EventRepository
from app.schemas.events import AgentEventRead
from app.services.event_bus import EventBus
from app.services.interfaces import SupportsCommit
from app.utils.ids import new_uuid
from app.utils.time import utc_now

# The closed internal component set (SDD 8.3.1). Not the agent enumeration, and never merged
# into it.
COMPONENT_ORCHESTRATOR: Final = "orchestrator"
COMPONENT_RETRIEVER: Final = "retriever"
COMPONENT_RERANKER: Final = "reranker"
COMPONENT_PROVIDER_ADAPTER: Final = "provider_adapter"
COMPONENT_EVALUATION_RUNNER: Final = "evaluation_runner"
COMPONENT_WORKER: Final = "worker"


@dataclass(frozen=True)
class EventIdentity:
    agent_name: AgentName | None
    component: str


# SDD 8.3.1, normative. Every one of the twelve `EventType` members appears exactly once, and the
# completeness of that is asserted by a contract test rather than trusted -- a member added
# without a policy would otherwise silently emit a frame with no identity at all, which is the
# defect this table exists to close.
#
# `agent_name` is `None` for the three workflow-level producers that are not agents. They are not
# anonymous: `component` names them.
EVENT_IDENTITY: Final[Mapping[EventType, EventIdentity]] = {
    EventType.PLANNING_STARTED: EventIdentity(AgentName.PLANNER, COMPONENT_ORCHESTRATOR),
    EventType.RETRIEVAL_STARTED: EventIdentity(None, COMPONENT_RETRIEVER),
    EventType.RETRIEVAL_COMPLETED: EventIdentity(None, COMPONENT_RETRIEVER),
    EventType.RERANKING_STARTED: EventIdentity(None, COMPONENT_RERANKER),
    EventType.RESEARCH_STARTED: EventIdentity(AgentName.RESEARCHER, COMPONENT_ORCHESTRATOR),
    EventType.CRITIC_STARTED: EventIdentity(AgentName.CRITIC, COMPONENT_ORCHESTRATOR),
    EventType.RISK_STARTED: EventIdentity(AgentName.RISK, COMPONENT_ORCHESTRATOR),
    EventType.ANSWER_STARTED: EventIdentity(AgentName.FINAL_DECISION, COMPONENT_ORCHESTRATOR),
    EventType.EVALUATION_STARTED: EventIdentity(AgentName.EVALUATOR, COMPONENT_ORCHESTRATOR),
    EventType.ANSWER_COMPLETED: EventIdentity(None, COMPONENT_ORCHESTRATOR),
    EventType.WORKFLOW_FAILED: EventIdentity(None, COMPONENT_ORCHESTRATOR),
    EventType.WORKFLOW_CANCELLED: EventIdentity(None, COMPONENT_WORKER),
}


class EventEmitter:
    def __init__(
        self,
        session: SupportsCommit,
        events: EventRepository,
        bus: EventBus,
        query_run_id: uuid.UUID,
    ) -> None:
        self._session = session
        self._events = events
        self._bus = bus
        self._query_run_id = query_run_id

    async def emit(
        self,
        event_type: EventType,
        message: str,
        *,
        agent_name: AgentName | None = None,
        component: str | None = None,
        payload: dict[str, object] | None = None,
    ) -> None:
        """Persist and publish one event.

        ``component`` and ``agent_name`` are both optional at this signature but not in the
        contract: SDD 8.3.1 assigns every one of the twelve event types exactly one policy, and
        ``EVENT_IDENTITY`` below is that table. The default is applied here rather than left to
        fourteen call sites, because leaving it to the call sites is precisely how all fourteen
        came to omit ``agent_name`` and every frame shipped with a null identity.

        An explicit argument still wins, so a caller with better information than the table --
        the worker naming itself on a cancellation, say -- can say so.
        """
        policy = EVENT_IDENTITY.get(event_type)
        if policy is not None:
            agent_name = agent_name if agent_name is not None else policy.agent_name
            component = component if component is not None else policy.component
        event = AgentEvent(
            id=new_uuid(),
            query_run_id=self._query_run_id,
            event_type=event_type,
            agent_name=agent_name,
            component=component,
            message=message,
            payload=payload or {},
            created_at=utc_now(),
        )
        await self._events.add(event)
        await self._session.commit()
        self._bus.publish(AgentEventRead.model_validate(event))
