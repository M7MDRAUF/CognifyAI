"""Server-Sent Events stream for a query run (SDD section 7.5).

On connect the client replays persisted events (so late joins and reconnects
see history), then follows live events until a terminal event. Subscribing to
the live bus happens before the replay read so no event is missed in between;
duplicates are removed by event id. A run whose terminal *status* is committed
without a terminal *event* -- the two are separate transactions -- still ends
the stream, from the status the 404 guard already loaded.
"""

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from app.api.deps import EventRepositoryDep, QueryServiceDep, ResourcesDep, SessionDep
from app.domain.enums import QueryRunStatus
from app.domain.models import QueryRun
from app.schemas.events import AgentEventRead
from app.services.event_bus import TERMINAL_EVENTS, terminal_event_for_status
from app.services.event_emitter import EVENT_IDENTITY
from app.utils.time import utc_now

router = APIRouter(tags=["events"])

_HEARTBEAT_SECONDS = 15

# What a reconstructed frame says when the row carries no message of its own. A failed run
# always does (every writer sets `error_message`), so this is only a fallback for rows that
# predate that; the other two match the message their writer emits.
_RECONSTRUCTED_MESSAGES = {
    QueryRunStatus.COMPLETED: "Analysis complete.",
    QueryRunStatus.FAILED: "The analysis workflow failed.",
    QueryRunStatus.CANCELLED: "The run was cancelled.",
}


class _DocumentedEventStream(EventSourceResponse):
    """The media type FastAPI documents for this route, and nothing else (FR-005).

    FastAPI reads a route's media type from its response class's *class* attribute, which
    `EventSourceResponse` sets only per instance, so without this the document says
    `application/json`. The handler builds and returns its own `EventSourceResponse`, so this class
    is never instantiated and the stream is byte-identical.
    """

    media_type = "text/event-stream"


# FR-005: what the served OpenAPI document says this route returns. With no declaration, FastAPI
# documents an `application/json` body with an empty schema, which the route never serves. The
# response class sets the documented media type, and the model makes `AgentEventRead` the frame's
# component schema.
_STREAM_RESPONSE = {
    "model": AgentEventRead,
    "description": (
        "A stream of Server-Sent Events (`text/event-stream`). Each frame is a named event: an "
        "`event:` line with the event type, then a `data:` line holding one `AgentEventRead` as "
        "JSON. The stream replays the run's stored events, then follows live ones, and closes "
        "after a terminal event (`answer_completed`, `workflow_failed` or `workflow_cancelled`). "
        "A comment line is sent as a heartbeat while nothing happens."
    ),
}


@router.get(
    "/queries/{query_run_id}/events",
    response_class=_DocumentedEventStream,
    responses={200: _STREAM_RESPONSE},
)
async def stream_events(
    query_run_id: uuid.UUID,
    service: QueryServiceDep,
    events: EventRepositoryDep,
    resources: ResourcesDep,
    session: SessionDep,
) -> EventSourceResponse:
    bus = resources.event_bus
    # Subscribe before either read. `EventBus.publish` only reaches queues that exist when it
    # runs and nothing re-publishes a missed event, so anything published between a read and
    # the subscription is lost outright -- and when that is the terminal event, the stream
    # never closes. This cannot be deferred into the generator (where it used to live): an
    # async generator body does not run until its first `__anext__`, which happens long after
    # this handler returns, and the request-scoped session behind `events`/`service` is gone by
    # then. The `seen` dedupe below is what makes the resulting history/queue overlap harmless.
    queue = bus.subscribe(str(query_run_id))
    # RMD-DEF-021: the stream reads the database no more after these reads, and FastAPI closes a
    # request-scoped session only after the response has been sent, which for a stream is its
    # whole life. Closed here, the connection goes back to the pool instead of sitting idle in
    # transaction for as long as the client stays connected (``events``, ``service`` and
    # ``session`` are one request's one session). The rows read stay loaded.
    try:
        history = await events.list_for_query_run(query_run_id)
        # Read last so it reflects the latest possible state. Doubles as the 404 guard, and is
        # the only terminator available for a run whose terminal status committed while its
        # terminal event never did -- `Orchestrator._finish` commits the status and only then
        # emits, so a crash or a cancellation in between leaves exactly that.
        query_run = await service.get_query_run(query_run_id)
    except BaseException:
        bus.unsubscribe(str(query_run_id), queue)
        # RMD-DEF-118: a close that fails here must not hide why the reads failed (a missing run
        # stays a 404); FastAPI's own teardown closes the session again.
        with contextlib.suppress(Exception):
            await session.close()
        raise
    try:
        await session.close()
    except BaseException:
        # RMD-DEF-118: no stream will run to release the subscription.
        bus.unsubscribe(str(query_run_id), queue)
        raise

    async def event_stream() -> AsyncIterator[dict[str, str]]:
        seen: set[uuid.UUID] = set()

        def unseen(event: AgentEventRead) -> bool:
            if event.id in seen:
                return False
            seen.add(event.id)
            return True

        try:
            for row in history:
                event = AgentEventRead.model_validate(row)
                seen.add(event.id)
                yield _format(event)
                if event.event_type in TERMINAL_EVENTS:
                    return
            # Drain what the bus already delivered before consulting the status: those events
            # were published after `subscribe`, so one of them may be the very event the
            # status is about to be read as reflecting. Preferring the real event keeps the
            # reconstruction below to the cases where there genuinely is none.
            while True:
                try:
                    event = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if not unseen(event):
                    continue
                yield _format(event)
                if event.event_type in TERMINAL_EVENTS:
                    return
            if terminal_event_for_status(query_run.status) is not None:
                yield _format(_reconstruct_terminal_event(query_run))
                return
            while True:
                event = await queue.get()
                if not unseen(event):
                    continue
                yield _format(event)
                if event.event_type in TERMINAL_EVENTS:
                    return
        finally:
            bus.unsubscribe(str(query_run_id), queue)

    return EventSourceResponse(event_stream(), ping=_HEARTBEAT_SECONDS)


def _reconstruct_terminal_event(query_run: QueryRun) -> AgentEventRead:
    """The terminal frame for a run whose terminal event was never persisted.

    Marked ``reconstructed_from_status`` in the payload rather than passed off as a real
    ``agent_events`` row: the outcome it reports is genuine (it comes from the committed
    status), but the event itself never happened, and an audit reader must be able to tell.
    The id is derived from the run and its status so reconnects reconstruct the *same* id and
    a client deduping by id does not accumulate duplicates.
    """
    status = query_run.status
    event_type = terminal_event_for_status(status)
    if event_type is None:  # pragma: no cover - callers check first
        msg = f"Query run {query_run.id} is not in a terminal status."
        raise ValueError(msg)
    message = _RECONSTRUCTED_MESSAGES[status]
    if status is QueryRunStatus.FAILED and query_run.error_message:
        message = query_run.error_message
    # C-13: the same identity a persisted frame of this type carries. The reconstruction used to
    # set none, so the one frame a client is most likely to act on was the one that named no
    # producer -- `component` defaulted to NULL.
    identity = EVENT_IDENTITY[event_type]
    return AgentEventRead(
        id=uuid.uuid5(query_run.id, status.value),
        query_run_id=query_run.id,
        event_type=event_type,
        agent_name=identity.agent_name,
        component=identity.component,
        message=message,
        payload={"reconstructed_from_status": True},
        created_at=utc_now(),
    )


def _format(event: AgentEventRead) -> dict[str, str]:
    return {"event": event.event_type.value, "data": event.model_dump_json()}
