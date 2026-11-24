"""In-process registry mapping a running query's id to its ``asyncio.Task`` (Phase 5).

Exactly like ``EventBus`` (see that module's docstring), this registry is **in-process
only**: it holds live ``asyncio.Task`` objects in memory, so a cancel request can only reach
a run scheduled on the *same* worker process. With more than one worker process, a cancel
issued against a process that isn't running the target task simply reports "not running" --
it never reaches the other process. That is the same constraint ``EventBus`` already has for
event delivery; it is not a new limitation introduced here, but it is worth stating plainly
since a naive reader could assume cancellation reaches every worker.
"""

import asyncio
import uuid


class RunRegistry:
    def __init__(self) -> None:
        self._tasks: dict[uuid.UUID, asyncio.Task[None]] = {}
        # Ids for which `cancel()` actually delivered a cancellation to a still-running task.
        # Tracked separately from `_tasks` (and never touched by `discard`) because a
        # `CancelledError` that lands on an in-flight DB await can be *replaced* by an ordinary
        # exception raised while `AsyncSession.__aexit__` unwinds (see `run_query_job`) -- at
        # that point `task.cancelled()` reports `False` even though a cancel genuinely was
        # requested, so `run_query_job` needs this independent record to still treat the run as
        # cancelled rather than as an unexplained failure.
        self._cancel_requested: set[uuid.UUID] = set()

    def register(self, query_run_id: uuid.UUID, task: asyncio.Task[None]) -> None:
        self._tasks[query_run_id] = task

    def cancel(self, query_run_id: uuid.UUID) -> bool:
        """Requests cancellation of the run's task. ``False`` when the id is unknown (never
        registered, already discarded) or when the task is already done -- ``asyncio.Task.cancel``
        itself returns ``False`` for a task that has already finished, so that case is not
        distinguished from "unknown" here; either way there is nothing left to interrupt.
        """
        task = self._tasks.get(query_run_id)
        if task is None:
            return False
        cancelled = task.cancel()
        if cancelled:
            self._cancel_requested.add(query_run_id)
        return cancelled

    def discard(self, query_run_id: uuid.UUID, task: asyncio.Task[None]) -> None:
        """Idempotent, and identity-checked: only removes the mapping when it still points at
        this exact ``task``. Without the identity check, if ``query_run_id`` were ever
        (re)registered by a second job before the first job's own ``finally`` ran, the first
        job's unconditional pop would evict the *second* job's registration, leaving it
        permanently uncancellable. Never raises, so callers can call this unconditionally from a
        ``finally`` block."""
        if self._tasks.get(query_run_id) is task:
            del self._tasks[query_run_id]

    def is_running(self, query_run_id: uuid.UUID) -> bool:
        task = self._tasks.get(query_run_id)
        return task is not None and not task.done()

    def consume_cancel_requested(self, query_run_id: uuid.UUID) -> bool:
        """One-shot check: ``True`` when ``cancel()`` ever delivered a cancellation for this id.
        Consumes the flag on read -- each query run id is scheduled at most once, so nothing
        needs it kept around afterwards, and leaving it in place would grow the set for the
        lifetime of the process."""
        try:
            self._cancel_requested.remove(query_run_id)
        except KeyError:
            return False
        return True
