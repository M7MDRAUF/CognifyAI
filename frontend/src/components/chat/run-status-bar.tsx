import {
  CircleCheck,
  CircleSlash,
  CircleX,
  Loader,
  type LucideIcon,
  RefreshCw,
  TriangleAlert,
} from "lucide-react";

import { CopyReferenceButton } from "@/components/chat/copy-reference-button";
import { Button } from "@/components/ui/button";
import type { PipelineModel } from "@/lib/chat/pipeline";
import type { StreamStatus } from "@/lib/hooks/use-query-run-events";
import { cn } from "@/lib/utils";

/** The run's phase as the pill shows it: words, an icon and a plan section 7 tone. */
export interface RunPhaseSummary {
  label: string;
  icon: LucideIcon;
  tone: string;
}

/**
 * The run's phase in words (AST-26 to AST-29, AST-32, AST-33). While the run is not known to have
 * ended, a stream that stopped delivering says so; once it ended, its outcome decides.
 */
export function runPhaseSummary(model: PipelineModel, streamStatus: StreamStatus): RunPhaseSummary {
  if (model.outcome === null && streamStatus === "recovering") {
    return { label: "Reconnecting…", icon: RefreshCw, tone: "text-muted-foreground" };
  }
  if (model.outcome === null && streamStatus === "failed") {
    return { label: "Live progress unavailable", icon: TriangleAlert, tone: "text-amber-700" };
  }
  switch (model.outcome) {
    case null:
      return { label: "Running", icon: Loader, tone: "text-foreground" };
    case "completed":
      return { label: "Completed", icon: CircleCheck, tone: "text-emerald-700" };
    case "completed_degraded":
      return {
        label: "Completed — evaluation unavailable",
        icon: TriangleAlert,
        tone: "text-amber-700",
      };
    case "completed_without_answer":
      return { label: "Completed without an answer", icon: TriangleAlert, tone: "text-amber-700" };
    case "completed_with_failure":
      return { label: "Completed with a failure", icon: TriangleAlert, tone: "text-amber-700" };
    case "failed":
      return { label: "Failed", icon: CircleX, tone: "text-red-700" };
    case "cancelled":
      return { label: "Cancelled", icon: CircleSlash, tone: "text-muted-foreground" };
  }
}

/** Why a Stop failed, as the chat words it, and the reference a 5xx carries. */
export interface StopFailure {
  message: string;
  reference: string | null;
}

interface RunStatusBarProps {
  summary: RunPhaseSummary;
  /** The run can be stopped: it runs and its stream has not failed. */
  showStop: boolean;
  /** A Stop is held until the run's terminal event (UDP1-T03). */
  stopRequested: boolean;
  onStop: () => void;
  /** The last Stop of this run failed (RMD5-T04, RE2E-DEF-040). */
  stopError?: StopFailure | null;
}

/**
 * The top of the selected run (UXR-25): the run-phase pill, in plain text rather than a live
 * region (the timeline's polite line stays the one announcer), and a Stop slot whose width is
 * reserved, so the bar does not move when the control comes or goes (UXR-21).
 *
 * A Stop that failed is reported here, beside the control that was pressed, in an alert that says
 * the stop failed, why, and the reference to quote (RMD5-T04): it used to land under the question
 * form, out of view of the run, as a message that did not say what had failed.
 */
export function RunStatusBar({
  summary,
  showStop,
  stopRequested,
  onStop,
  stopError = null,
}: RunStatusBarProps) {
  const Icon = summary.icon;
  return (
    <div
      role="group"
      aria-label="Run status"
      className="flex min-h-9 flex-wrap items-center justify-between gap-2"
    >
      <p
        className={cn(
          "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium",
          summary.tone,
        )}
      >
        <Icon className="size-3.5 shrink-0" aria-hidden />
        {summary.label}
      </p>
      <div data-stop-slot className="flex w-28 shrink-0 justify-end">
        {showStop && (
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={stopRequested}
            onClick={onStop}
          >
            {stopRequested ? "Stopping…" : "Stop"}
          </Button>
        )}
      </div>
      {stopError !== null && (
        <div role="alert" className="basis-full space-y-2 text-sm text-destructive">
          <p>The analysis could not be stopped. {stopError.message}</p>
          {stopError.reference !== null && (
            <CopyReferenceButton value={stopError.reference} label="Copy reference" />
          )}
        </div>
      )}
    </div>
  );
}
