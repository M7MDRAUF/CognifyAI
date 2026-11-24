import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { providerLabel } from "@/lib/types/llm";
import type { AgentName, AgentRun, QueryRunStatus } from "@/lib/types/query";
import { formatLatency } from "@/lib/utils/format";

/** Reader-facing names of the agent steps, shared with the workflow graph and the timeline. */
export const AGENT_LABELS: Record<AgentName, string> = {
  planner: "Planner",
  retriever: "Retriever",
  researcher: "Researcher",
  critic: "Critic",
  risk: "Risk",
  final_decision: "Decision",
  evaluator: "Evaluator",
};

/**
 * A step's status as the reader should understand it. A cancel or a deadline leaves the step that
 * was in flight RUNNING (`backend/app/workers/jobs.py`); only the run's own status says why it
 * stopped, so a finished run never shows a step as still running.
 */
function statusText(status: AgentRun["status"], runStatus: QueryRunStatus): string {
  switch (status) {
    case "completed":
      return "Completed";
    case "failed":
      return "Failed";
    case "pending":
    case "running":
      if (runStatus === "running") {
        return status === "running" ? "Running" : "Pending";
      }
      return runStatus === "cancelled" ? "Cancelled" : "Did not finish";
  }
}

function modelText(run: AgentRun): string {
  return run.llm_provider !== null &&
    run.llm_provider !== undefined &&
    run.llm_model !== null &&
    run.llm_model !== undefined
    ? `${providerLabel(run.llm_provider)} · ${run.llm_model}`
    : "—";
}

function latencyText(run: AgentRun): string {
  if (run.status !== "completed" && run.status !== "failed") {
    return "—";
  }
  return formatLatency(run.latency_ms);
}

interface AgentRunsPanelProps {
  runStatus: QueryRunStatus;
  /** The run's `agent_runs`, in the order the backend recorded them (a step can repeat). */
  agentRuns: readonly AgentRun[];
}

/** Every agent step of a run with its status, attempts, model and latency (FE-012). */
export function AgentRunsPanel({ runStatus, agentRuns }: AgentRunsPanelProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Agent steps</CardTitle>
      </CardHeader>
      <CardContent>
        {agentRuns.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No agent steps were recorded for this analysis.
          </p>
        ) : (
          // FE-016: five columns do not fit 320 px; the table scrolls inside its card instead, and
          // the scroller takes focus so it can be scrolled from the keyboard (WCAG 2.1.1).
          <div
            role="region"
            aria-label="Agent steps"
            tabIndex={0}
            className="overflow-x-auto rounded-md focus-visible:outline-2 focus-visible:outline-offset-2"
          >
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-left text-muted-foreground">
                  <th scope="col" className="py-2 pr-4 font-medium">
                    Step
                  </th>
                  <th scope="col" className="py-2 pr-4 font-medium">
                    Status
                  </th>
                  <th scope="col" className="py-2 pr-4 font-medium">
                    Attempts
                  </th>
                  <th scope="col" className="py-2 pr-4 font-medium">
                    Model
                  </th>
                  <th scope="col" className="py-2 font-medium">
                    Latency
                  </th>
                </tr>
              </thead>
              <tbody>
                {agentRuns.map((run, index) => (
                  <tr key={`${run.agent_name}-${String(index)}`} className="border-b">
                    <th scope="row" className="py-2 pr-4 text-left font-normal">
                      {AGENT_LABELS[run.agent_name]}
                    </th>
                    <td
                      className={
                        run.status === "failed" ? "py-2 pr-4 text-destructive" : "py-2 pr-4"
                      }
                    >
                      {statusText(run.status, runStatus)}
                    </td>
                    <td className="py-2 pr-4">{String(run.attempt_count)}</td>
                    <td className="py-2 pr-4">{modelText(run)}</td>
                    <td className="py-2">{latencyText(run)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
