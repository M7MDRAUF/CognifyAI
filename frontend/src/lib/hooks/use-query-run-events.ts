"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { type InvalidFrameKind, subscribeToQueryEvents } from "@/lib/api/events";
import { getQueryDetail } from "@/lib/api/queries";
import { type AgentEvent, TERMINAL_EVENT_TYPES } from "@/lib/types/events";
import type { QueryRunDetail, QueryRunStatus } from "@/lib/types/query";

/**
 * Fallback polling of `GET /queries/{id}` while the stream is down (FE-002). One request at a time,
 * never more often than every two seconds, and for a bounded time: the plan's hard stop is more
 * than one request per two seconds or any request after a terminal status.
 */
export const STREAM_POLL_INTERVAL_MS = 2000;
/** How long the fallback polls a run that is still `running` before it reports a failure. */
export const STREAM_POLL_CEILING_MS = 30_000;
/**
 * A live stream that delivers nothing for this long gets one status check. The backend's heartbeat
 * is an SSE comment the browser never surfaces, and single agent steps take up to about half a
 * minute on the live stack, so this is well above a normal gap. The check ends the run only if its
 * terminal status already committed (a missing terminal frame); otherwise nothing changes.
 */
export const STREAM_STALL_MS = 60_000;

const TERMINAL_STATUSES: ReadonlySet<QueryRunStatus> = new Set<QueryRunStatus>([
  "completed",
  "failed",
  "cancelled",
]);

/**
 * `live`: the stream is connected, or the run has finished. `recovering`: the stream failed and the
 * run's detail is being polled. `failed`: neither the stream nor polling produced a final state
 * within the bound; the run may still be running, and `retry` resubscribes.
 */
export type StreamStatus = "live" | "recovering" | "failed";

interface StreamTiming {
  pollIntervalMs?: number;
  pollCeilingMs?: number;
  stallMs?: number;
}

/** Frames the client could not use, by kind (FE-013). */
export type InvalidFrameCounts = Readonly<Record<InvalidFrameKind, number>>;

const NO_INVALID_FRAMES: InvalidFrameCounts = {
  unknown_event: 0,
  invalid_json: 0,
  invalid_schema: 0,
};

interface StreamState {
  runId: string | null;
  events: AgentEvent[];
  isComplete: boolean;
  streamStatus: StreamStatus;
  invalidFrames: InvalidFrameCounts;
  detailError: unknown;
}

interface UseQueryRunEvents {
  events: AgentEvent[];
  isComplete: boolean;
  streamStatus: StreamStatus;
  invalidFrames: InvalidFrameCounts;
  /**
   * The error of the last detail read that failed while the stream was down, or `null` once a read
   * answers (RMD5-T04, RE2E-DEF-039). It used to be swallowed, so a run whose stream and detail
   * both failed showed nothing of why.
   */
  detailError: unknown;
  /** Resubscribe after `failed`: a fresh stream replays the run's history. */
  retry: () => void;
}

/**
 * Append-only subscription to a query run's event stream (SDD 11.3). State is never mutated in
 * place; a terminal event marks the stream complete and triggers the optional onComplete callback
 * (used to refetch the final result and the history list). When a detail read ended the run, the
 * callback receives that detail, so the caller need not read it again (UDP6-T02, RV-01-02).
 *
 * The visible state always converges to the truth: when the stream fails, the run's detail is
 * polled within a fixed bound, a terminal status there ends the run exactly as a terminal event
 * would, and a run still unresolved at the bound is reported as a failure the user can retry.
 *
 * State for a new run is reset during render (the React-recommended pattern for resetting on a
 * changed prop) rather than synchronously inside the effect.
 */
export function useQueryRunEvents(
  queryRunId: string | null,
  onComplete?: (detail?: QueryRunDetail) => void,
  timing?: StreamTiming,
): UseQueryRunEvents {
  const [state, setState] = useState<StreamState>({
    runId: queryRunId,
    events: [],
    isComplete: false,
    streamStatus: "live",
    invalidFrames: NO_INVALID_FRAMES,
    detailError: null,
  });
  const [attempt, setAttempt] = useState(0);
  const completeRef = useRef(onComplete);
  // The event ids already delivered for this run, kept across subscriptions: every subscribe
  // replays the run's history, so a set per subscription let a Reconnect append it again
  // (UDF-03). It is replaced only when the run changes.
  const seenRef = useRef<{ runId: string | null; ids: Set<string> }>({
    runId: null,
    ids: new Set(),
  });
  const pollIntervalMs = timing?.pollIntervalMs ?? STREAM_POLL_INTERVAL_MS;
  const pollCeilingMs = timing?.pollCeilingMs ?? STREAM_POLL_CEILING_MS;
  const stallMs = timing?.stallMs ?? STREAM_STALL_MS;

  useEffect(() => {
    completeRef.current = onComplete;
  }, [onComplete]);

  if (state.runId !== queryRunId) {
    setState({
      runId: queryRunId,
      events: [],
      isComplete: false,
      streamStatus: "live",
      invalidFrames: NO_INVALID_FRAMES,
      detailError: null,
    });
  }

  useEffect(() => {
    if (queryRunId === null) {
      // The page dropped its run, and the reset above emptied the events: forget their ids too,
      // or a return to the same run would discard its whole replay (UDP6-T02, RV-01-01).
      seenRef.current = { runId: null, ids: new Set() };
      return;
    }
    let disposed = false;
    let finished = false;
    let pollTimer: ReturnType<typeof setTimeout> | null = null;
    let pollStartedAt: number | null = null;
    let stallTimer: ReturnType<typeof setTimeout> | null = null;
    if (seenRef.current.runId !== queryRunId) {
      seenRef.current = { runId: queryRunId, ids: new Set() };
    }
    const seen = seenRef.current.ids;

    const setStatus = (streamStatus: StreamStatus) => {
      setState((current) =>
        current.runId === queryRunId && current.streamStatus !== streamStatus
          ? { ...current, streamStatus }
          : current,
      );
    };
    const clearStall = () => {
      if (stallTimer !== null) {
        clearTimeout(stallTimer);
        stallTimer = null;
      }
    };
    const stopPolling = () => {
      if (pollTimer !== null) {
        clearTimeout(pollTimer);
        pollTimer = null;
      }
      pollStartedAt = null;
    };

    const complete = (detail?: QueryRunDetail) => {
      if (finished || disposed) {
        return;
      }
      finished = true;
      stopPolling();
      clearStall();
      setState((current) =>
        current.runId === queryRunId
          ? { ...current, isComplete: true, streamStatus: "live" }
          : current,
      );
      completeRef.current?.(detail);
      subscription.close();
    };

    const setDetailError = (detailError: unknown) => {
      setState((current) =>
        current.runId === queryRunId && current.detailError !== detailError
          ? { ...current, detailError }
          : current,
      );
    };

    /** One detail read: true when the run has a terminal status (and so is now complete). */
    const readTerminal = async (): Promise<boolean> => {
      try {
        const detail = await getQueryDetail(queryRunId);
        setDetailError(null);
        if (TERMINAL_STATUSES.has(detail.status)) {
          complete(detail);
          return true;
        }
      } catch (cause: unknown) {
        // A failed read counts toward the ceiling like a `running` one, and is kept for the view.
        if (!disposed) {
          setDetailError(cause);
        }
      }
      return false;
    };

    const pollOnce = async () => {
      pollTimer = null;
      if (await readTerminal()) {
        return;
      }
      if (finished || disposed || pollStartedAt === null) {
        return;
      }
      if (Date.now() - pollStartedAt >= pollCeilingMs) {
        pollStartedAt = null;
        clearStall();
        setStatus("failed");
        // Stop the browser's own reconnect loop too; `retry` opens a fresh stream.
        subscription.close();
        return;
      }
      pollTimer = setTimeout(() => void pollOnce(), pollIntervalMs);
    };

    const startPolling = () => {
      if (finished || disposed || pollStartedAt !== null) {
        return;
      }
      pollStartedAt = Date.now();
      void pollOnce();
    };

    const armStall = () => {
      clearStall();
      stallTimer = setTimeout(() => {
        stallTimer = null;
        if (finished || disposed || pollStartedAt !== null) {
          return;
        }
        void readTerminal().then((terminal) => {
          if (!terminal && !finished && !disposed) {
            armStall();
          }
        });
      }, stallMs);
    };

    const subscription = subscribeToQueryEvents(queryRunId, {
      onOpen: () => {
        stopPolling();
        setStatus("live");
        armStall();
      },
      onEvent: (event) => {
        if (seen.has(event.id)) {
          return;
        }
        seen.add(event.id);
        stopPolling();
        const terminal = TERMINAL_EVENT_TYPES.has(event.event_type);
        setState((current) =>
          current.runId === queryRunId
            ? {
                ...current,
                events: [...current.events, event],
                isComplete: current.isComplete || terminal,
                streamStatus: "live",
              }
            : current,
        );
        if (terminal) {
          complete();
        } else {
          armStall();
        }
      },
      onInvalidFrame: (frame) => {
        setState((current) =>
          current.runId === queryRunId
            ? {
                ...current,
                invalidFrames: {
                  ...current.invalidFrames,
                  [frame.kind]: current.invalidFrames[frame.kind] + 1,
                },
              }
            : current,
        );
        // A terminal frame the client cannot read may be the run's last word: switch to the run's
        // detail rather than wait for a frame that will never come (FE-013 into FE-002's fallback).
        if (frame.terminal && !finished && !disposed) {
          setStatus("recovering");
          startPolling();
        }
      },
      onError: () => {
        if (finished || disposed) {
          return;
        }
        // Closed for good or reconnecting, the run's truth is one request away; poll it until the
        // stream delivers again or the run reaches a terminal status.
        setStatus("recovering");
        startPolling();
      },
    });
    armStall();

    return () => {
      disposed = true;
      stopPolling();
      clearStall();
      subscription.close();
    };
  }, [queryRunId, attempt, pollIntervalMs, pollCeilingMs, stallMs]);

  const retry = useCallback(() => {
    setState((current) => ({ ...current, streamStatus: "live" }));
    setAttempt((current) => current + 1);
  }, []);

  return {
    events: state.events,
    isComplete: state.isComplete,
    streamStatus: state.streamStatus,
    invalidFrames: state.invalidFrames,
    detailError: state.detailError,
    retry,
  };
}
