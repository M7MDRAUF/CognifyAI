"""In-process pub/sub for workflow events (SSE delivery).

The orchestrator runs as a background task; the SSE endpoint runs in a
request. They share one ``EventBus`` (held in application resources) so live
events reach connected clients. Events are also persisted, so a client that
connects late or reconnects replays history first, then follows live.
"""

import asyncio
from collections import defaultdict

from app.domain.enums import EventType, QueryRunStatus
from app.schemas.events import AgentEventRead

# The one definition of "this run is over", for both halves of the contract: the event every
# terminal status owes the stream, and the statuses no writer may relabel. Kept here rather
# than beside either half because both the writers (`Orchestrator._fail`, `jobs._mark_*`) and
# the reader (`api.routes.events`) already depend on this module, and a second copy is how the
# guard ended up on one writer and not the others.
_TERMINAL_EVENT_FOR_STATUS = {
    QueryRunStatus.COMPLETED: EventType.ANSWER_COMPLETED,
    QueryRunStatus.FAILED: EventType.WORKFLOW_FAILED,
    QueryRunStatus.CANCELLED: EventType.WORKFLOW_CANCELLED,
}

TERMINAL_EVENTS = frozenset(_TERMINAL_EVENT_FOR_STATUS.values())
TERMINAL_QUERY_RUN_STATUSES = frozenset(_TERMINAL_EVENT_FOR_STATUS)


def terminal_event_for_status(status: QueryRunStatus) -> EventType | None:
    """The terminal event a run in ``status`` owes its stream, or ``None`` while it runs."""
    return _TERMINAL_EVENT_FOR_STATUS.get(status)


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue[AgentEventRead]]] = defaultdict(set)

    def publish(self, event: AgentEventRead) -> None:
        for queue in list(self._subscribers.get(str(event.query_run_id), set())):
            queue.put_nowait(event)

    def subscribe(self, query_run_id: str) -> asyncio.Queue[AgentEventRead]:
        queue: asyncio.Queue[AgentEventRead] = asyncio.Queue()
        self._subscribers[query_run_id].add(queue)
        return queue

    def unsubscribe(self, query_run_id: str, queue: asyncio.Queue[AgentEventRead]) -> None:
        subscribers = self._subscribers.get(query_run_id)
        if subscribers is None:
            return
        subscribers.discard(queue)
        if not subscribers:
            self._subscribers.pop(query_run_id, None)
