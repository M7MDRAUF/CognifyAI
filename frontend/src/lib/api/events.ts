import {
  type AgentEvent,
  TERMINAL_EVENT_TYPES,
  agentEventSchema,
  eventTypeSchema,
} from "@/lib/types/events";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api/v1";

export interface EventSubscription {
  close: () => void;
}

/**
 * How a stream error left the connection. `closed`: the browser gave up for good (the endpoint
 * answered non-200 or a non-SSE type, or the source was closed), so nothing more will arrive.
 * `reconnecting`: a live connection dropped and the browser is retrying; the backend replays the
 * run's history when it reconnects.
 */
export type StreamErrorState = "closed" | "reconnecting";

/**
 * Why a frame could not be used (FE-013). `unknown_event`: its event type is not one of the twelve
 * the client knows, which ADR-007 makes a major wire change. `invalid_json`: the data is not JSON.
 * `invalid_schema`: a known event type whose payload fails `agentEventSchema`.
 */
export type InvalidFrameKind = "unknown_event" | "invalid_json" | "invalid_schema";

export interface InvalidFrame {
  kind: InvalidFrameKind;
  /** The SSE event name, or the payload's `event_type` for an unnamed frame. */
  eventName: string;
  /** The frame named a terminal event, so the run may have ended without the client knowing. */
  terminal: boolean;
}

interface SubscribeHandlers {
  onEvent: (event: AgentEvent) => void;
  onOpen?: () => void;
  onError?: (state: StreamErrorState) => void;
  onInvalidFrame?: (frame: InvalidFrame) => void;
}

function isTerminalName(name: string): boolean {
  const parsed = eventTypeSchema.safeParse(name);
  return parsed.success && TERMINAL_EVENT_TYPES.has(parsed.data);
}

/**
 * Subscribe to a query run's SSE stream. Each message is validated against the AgentEvent schema
 * before being delivered; a frame that cannot be used is reported through `onInvalidFrame` rather
 * than dropped silently. A named event of a type the client does not know never reaches any
 * listener (EventSource has no wildcard), so only unnamed frames can reveal an unknown type.
 */
export function subscribeToQueryEvents(
  queryRunId: string,
  handlers: SubscribeHandlers,
): EventSubscription {
  const source = new EventSource(`${API_BASE_URL}/queries/${queryRunId}/events`);

  const handleMessage = (event: MessageEvent<string>) => {
    let parsed: unknown;
    try {
      parsed = JSON.parse(event.data);
    } catch {
      handlers.onInvalidFrame?.({
        kind: "invalid_json",
        eventName: event.type,
        terminal: isTerminalName(event.type),
      });
      return;
    }
    const result = agentEventSchema.safeParse(parsed);
    if (result.success) {
      handlers.onEvent(result.data);
      return;
    }
    const declared =
      typeof parsed === "object" && parsed !== null && "event_type" in parsed
        ? String(parsed.event_type)
        : "";
    const eventName = event.type !== "message" ? event.type : declared || "message";
    handlers.onInvalidFrame?.({
      kind: eventTypeSchema.safeParse(eventName).success ? "invalid_schema" : "unknown_event",
      eventName,
      terminal: isTerminalName(eventName),
    });
  };

  // The backend emits *named* SSE events (e.g. `event: planning_started`), so a
  // generic `onmessage` handler never fires for them — `onmessage` only receives
  // unnamed `message` events. Register the handler for every known event type, and
  // keep `onmessage` as a fallback in case an unnamed event is ever sent.
  for (const eventType of eventTypeSchema.options) {
    source.addEventListener(eventType, handleMessage);
  }
  source.onmessage = handleMessage;
  source.onopen = () => {
    handlers.onOpen?.();
  };
  source.onerror = () => {
    handlers.onError?.(source.readyState === EventSource.CLOSED ? "closed" : "reconnecting");
  };

  return {
    close: () => {
      source.close();
    },
  };
}
