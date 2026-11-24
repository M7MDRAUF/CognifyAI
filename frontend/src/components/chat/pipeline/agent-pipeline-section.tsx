"use client";

import { ChevronDown } from "lucide-react";
import { type FocusEvent, useId, useMemo, useState } from "react";

import { AgentTimeline } from "@/components/agents/agent-timeline";
import { AgentPipeline, stageTabId } from "@/components/chat/pipeline/agent-pipeline";
import { StageFocusCard } from "@/components/chat/pipeline/stage-focus-card";
import { RunStatusBar, type StopFailure, runPhaseSummary } from "@/components/chat/run-status-bar";
import {
  type PipelineModel,
  type PipelineRunInput,
  type PipelineStageId,
  derivePipeline,
} from "@/lib/chat/pipeline";
import type { StreamStatus } from "@/lib/hooks/use-query-run-events";
import type { AgentEvent } from "@/lib/types/events";
import { cn } from "@/lib/utils";

/**
 * The stage the selection follows until the person chooses one: the running stage, then the
 * failed one, then the last stage the run reached.
 */
function followedStage(model: PipelineModel): PipelineStageId {
  if (model.runningStage !== null) return model.runningStage;
  const failed = model.stages.find(
    (stage) => stage.status === "failed" || stage.status === "interrupted",
  );
  if (failed !== undefined) return failed.id;
  const reached = model.stages.filter(
    (stage) => stage.status !== "pending" && stage.status !== "not_run",
  );
  return reached.at(-1)?.id ?? "planner";
}

interface AgentPipelineSectionProps {
  /** The selected run's id, which a failed stage's card offers to copy (UXR-18). */
  runId: string;
  /** The stream's events, de-duplicated across subscriptions (UDP1-T02). */
  events: readonly AgentEvent[];
  /** The run's detail, once read. */
  run: PipelineRunInput | undefined;
  /** The view's held Stop request for this run (UDP1-T03). */
  stopRequested: boolean;
  streamStatus: StreamStatus;
  invalidFrameCount: number;
  /** The run can be stopped now. */
  showStop: boolean;
  onStop: () => void;
  /** The last Stop of this run failed; shown in the status bar (RMD5-T04, DV-RMD-33). */
  stopError?: StopFailure | null;
}

/**
 * The selected run's pipeline area, keyed by run id so no selection outlives its run. It derives
 * the stage model once per change of its inputs (plan section 17) and renders the status bar, the
 * eight stage tabs, the selected stage's card and the chronological timeline, whose polite line
 * stays the page's one announcer.
 *
 * The selection follows the running stage until the person chooses one, and holds still while
 * focus is inside the tablist or the selected stage's card, so neither the tab nor the card under
 * focus changes beneath it; following never moves focus (UXR-16, A11Y-09; UDP6-T02, RV-02-04).
 * The activity log is a disclosure, open at first and never stored (UXR-17).
 */
export function AgentPipelineSection({
  runId,
  events,
  run,
  stopRequested,
  streamStatus,
  invalidFrameCount,
  showStop,
  onStop,
  stopError = null,
}: AgentPipelineSectionProps) {
  const panelId = useId();
  const [chosen, setChosen] = useState<PipelineStageId | null>(null);
  // The selection when focus entered the tablist; null while focus is elsewhere.
  const [frozen, setFrozen] = useState<PipelineStageId | null>(null);
  const [activityOpen, setActivityOpen] = useState(true);
  const model = useMemo(
    () => derivePipeline(events, run, stopRequested),
    [events, run, stopRequested],
  );
  const selectedId = chosen ?? frozen ?? followedStage(model);
  const timelineId = `${panelId}-timeline`;

  function handleFocusOut(event: FocusEvent<HTMLDivElement>) {
    // Switching windows blurs with no related target while focus stays here in the page; the
    // freeze holds through it (UDP6-T02, RV-01-03).
    if (event.relatedTarget === null && !document.hasFocus()) {
      return;
    }
    if (!(
      event.relatedTarget instanceof Node && event.currentTarget.contains(event.relatedTarget)
    )) {
      setFrozen(null);
    }
  }
  return (
    <div className="space-y-4">
      <RunStatusBar
        summary={runPhaseSummary(model, streamStatus)}
        showStop={showStop}
        stopRequested={stopRequested}
        onStop={onStop}
        stopError={stopError}
      />
      <div
        className="space-y-4"
        onFocus={() => {
          setFrozen((current) => current ?? selectedId);
        }}
        onBlur={handleFocusOut}
      >
        <AgentPipeline
          model={model}
          selectedId={selectedId}
          onSelect={setChosen}
          panelId={panelId}
        />
        {/* The model always holds all eight stages, so exactly one card renders. */}
        {model.stages
          .filter((stage) => stage.id === selectedId)
          .map((stage) => (
            <StageFocusCard
              key={stage.id}
              runId={runId}
              stage={stage}
              model={model}
              events={events}
              panelId={panelId}
              tabId={stageTabId(panelId, stage.id)}
            />
          ))}
      </div>
      <div className="space-y-2">
        <button
          type="button"
          aria-expanded={activityOpen}
          aria-controls={events.length > 0 ? timelineId : undefined}
          onClick={() => {
            setActivityOpen((open) => !open);
          }}
          className="inline-flex min-h-6 cursor-pointer items-center gap-1 rounded-sm text-sm font-medium text-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
        >
          <ChevronDown
            className={cn("size-4 text-muted-foreground", !activityOpen && "-rotate-90")}
            aria-hidden
          />
          Activity log
        </button>
        <AgentTimeline
          events={events}
          invalidFrameCount={invalidFrameCount}
          collapsed={!activityOpen}
          listId={timelineId}
        />
      </div>
    </div>
  );
}
