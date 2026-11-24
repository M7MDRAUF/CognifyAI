import { AGENT_LABELS } from "@/components/chat/agent-runs-panel";
import type { AgentEvent, EventType } from "@/lib/types/events";

const LABELS: Record<EventType, string> = {
  planning_started: "Planning the analysis",
  retrieval_started: "Retrieving evidence",
  retrieval_completed: "Evidence selected",
  reranking_started: "Reranking candidates",
  research_started: "Synthesizing evidence",
  critic_started: "Checking claim support",
  risk_started: "Assessing risks",
  answer_started: "Composing the decision",
  evaluation_started: "Scoring the answer",
  answer_completed: "Analysis complete",
  workflow_failed: "Workflow failed",
  workflow_cancelled: "Analysis cancelled",
};

/** SSE v2 `component` values (ADR-007) for events no agent produced. */
const COMPONENT_LABELS: Record<string, string> = {
  orchestrator: "Orchestrator",
  retriever: "Retriever",
  reranker: "Reranker",
  worker: "Worker",
};

/** Who produced the event: its agent, else its component (FE-037). */
function producer(event: AgentEvent): string | null {
  if (event.agent_name !== null) {
    return AGENT_LABELS[event.agent_name];
  }
  if (event.component === null || event.component === undefined) {
    return null;
  }
  return COMPONENT_LABELS[event.component] ?? event.component;
}

/** The retrieval round: 0 is the initial retrieval, 1 and up expand it on the critic's feedback. */
function roundText(event: AgentEvent): string | null {
  const round = event.payload.round;
  if (typeof round !== "number") {
    return null;
  }
  return round === 0 ? "initial round" : `expansion round ${String(round)}`;
}

function dotClass(eventType: EventType): string {
  if (eventType === "workflow_failed") {
    return "mt-1.5 h-2 w-2 shrink-0 rounded-full bg-destructive";
  }
  if (eventType === "workflow_cancelled") {
    return "mt-1.5 h-2 w-2 shrink-0 rounded-full bg-muted-foreground";
  }
  return "mt-1.5 h-2 w-2 shrink-0 rounded-full bg-primary";
}

/** The latest event as one sentence for assistive technology: stage, round, or outcome. */
function announcement(event: AgentEvent): string {
  const label = LABELS[event.event_type];
  if (event.event_type === "workflow_failed") {
    return `${label}: ${event.message}`;
  }
  const round = roundText(event);
  return round === null ? `${label}.` : `${label}, ${round}.`;
}

/**
 * FE-018: one polite status line, present before the first event so the first change is heard,
 * holding only the latest event. A re-render with no new event leaves its text unchanged, so
 * nothing is announced twice.
 */
function LiveProgress({ events }: { events: readonly AgentEvent[] }) {
  const latest = events.at(-1);
  return (
    <p role="status" aria-live="polite" className="sr-only">
      {latest === undefined ? "" : announcement(latest)}
    </p>
  );
}

interface AgentTimelineProps {
  events: readonly AgentEvent[];
  /**
   * Stream frames the client could not read (FE-013). Shown as a non-blocking note, so a contract
   * drift is visible instead of silently missing from the timeline.
   */
  invalidFrameCount?: number;
  /** The activity log's disclosure: the list is hidden, and the polite line stays (UXR-17). */
  collapsed?: boolean;
  /** The list's id, which the disclosure's `aria-controls` names. */
  listId?: string;
}

function InvalidFrameNote({ count }: { count: number }) {
  if (count === 0) {
    return null;
  }
  return (
    <p role="status" className="text-xs text-muted-foreground">
      Some progress updates could not be read ({count}). The final result is still loaded from the
      server.
    </p>
  );
}

export function AgentTimeline({
  events,
  invalidFrameCount = 0,
  collapsed = false,
  listId,
}: AgentTimelineProps) {
  if (events.length === 0) {
    return (
      <div className="space-y-1">
        <LiveProgress events={events} />
        {!collapsed && (
          <p className="text-sm text-muted-foreground">Waiting for the workflow to start…</p>
        )}
        <InvalidFrameNote count={invalidFrameCount} />
      </div>
    );
  }
  return (
    <div className="space-y-2">
      <LiveProgress events={events} />
      <ol id={listId} hidden={collapsed} className="space-y-2" aria-label="Agent timeline">
        {events.map((event) => {
          const round = roundText(event);
          const source = producer(event);
          return (
            <li key={event.id} className="flex items-start gap-2 text-sm">
              <span className={dotClass(event.event_type)} aria-hidden />
              <span className="min-w-0">
                <span>{LABELS[event.event_type]}</span>
                {round !== null && (
                  <span className="text-xs text-muted-foreground"> · {round}</span>
                )}
                {source !== null && (
                  <span className="ml-2 text-xs text-muted-foreground">
                    {/* UDF-07: read apart from the label, not run into it. */}
                    <span className="sr-only">, </span>
                    {source}
                  </span>
                )}
                {event.event_type === "workflow_failed" && (
                  <span className="block text-xs text-destructive">{event.message}</span>
                )}
              </span>
            </li>
          );
        })}
      </ol>
      <InvalidFrameNote count={invalidFrameCount} />
    </div>
  );
}
