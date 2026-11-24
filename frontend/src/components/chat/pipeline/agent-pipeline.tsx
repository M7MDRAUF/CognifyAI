import { type KeyboardEvent, useRef } from "react";

import styles from "@/components/chat/pipeline/agent-pipeline.module.css";
import { StageGlyph } from "@/components/chat/pipeline/stage-glyph";
import {
  STAGE_ICONS,
  STATUS_VISUALS,
  StatusIcon,
  stageStatusText,
} from "@/components/chat/pipeline/stage-visuals";
import { useOffscreenPause } from "@/components/chat/pipeline/use-offscreen-pause";
import {
  useRevealOnChange,
  useSettleEmphasis,
} from "@/components/chat/pipeline/use-settle-emphasis";
import {
  type PipelineModel,
  type PipelineStage,
  type PipelineStageId,
  STAGE_LABELS,
  STAGE_TECHNICAL_LABELS,
} from "@/lib/chat/pipeline";
import { cn } from "@/lib/utils";

/** The id of a stage's tab, which the panel names as its label (`aria-labelledby`). */
export function stageTabId(panelId: string, stage: PipelineStageId): string {
  return `${panelId}-tab-${stage}`;
}

/**
 * The line from the previous stage: decorative, and shown only in the one-row layout. It comes in
 * with a 150 ms opacity change as the stage leaves `pending` (MOT-12).
 */
function PipelineConnector({ active }: { active: boolean }) {
  return (
    <span
      aria-hidden
      data-connector
      data-active={active ? "true" : "false"}
      className={cn(
        styles.connector,
        "pointer-events-none absolute top-1/2 -left-2 hidden h-px w-2 bg-primary lg:block",
      )}
    />
  );
}

/**
 * The recorded attempts a tab shows as its amber badge (AST-23): above 1, except on a failed
 * stage, whose status already reads "Failed after N attempts", and on a running one. 0: no badge.
 */
function badgeAttempts(stage: PipelineStage): number {
  return stage.attempts > 1 && stage.status !== "failed" && stage.status !== "running"
    ? stage.attempts
    : 0;
}

interface StageTabProps {
  stage: PipelineStage;
  selected: boolean;
  panelId: string;
  onSelect: (id: PipelineStageId) => void;
  /** The first stage has no connector before it. */
  first: boolean;
  /** The run is live: only a run watched live reveals a late attempts badge (RV-01-04). */
  live: boolean;
}

/**
 * One stage: its agent's icon and name, and one status line with the status's icon. The name and
 * the status are also the tab's accessible name; the backend's step name is its `title`. Every
 * status keeps the same box, so a status change moves nothing (UXR-21).
 *
 * While the stage runs, the tab carries a 2 px trace and its agent's glyph in the status icon's
 * box (MOT-01 to MOT-09). The CSS module animates both from the tab's `data-status` and
 * `data-agent`, so the motion leaves on the commit that changes the status. A completion this tab
 * watched happen settles its status icon once (MOT-10), and an attempts badge that appears while
 * it is mounted fades in once (MOT-11).
 */
function StageTab({ stage, selected, panelId, onSelect, first, live }: StageTabProps) {
  const visual = STATUS_VISUALS[stage.status];
  const AgentIcon = STAGE_ICONS[stage.id];
  const label = STAGE_LABELS[stage.id];
  const status = stageStatusText(stage);
  const attempts = badgeAttempts(stage);
  const iconRef = useRef<HTMLSpanElement>(null);
  const badgeRef = useRef<HTMLSpanElement>(null);
  useSettleEmphasis(iconRef, stage.status, stage.provisional);
  useRevealOnChange(badgeRef, attempts, live);
  // A held stop request is a warning on the stage still running (plan section 7).
  const tone = stage.status === "running" && stage.stopRequested ? "text-amber-700" : visual.tone;
  // In the one-row layout (from 1024 px) a tab is about 115 px wide, so its padding and name are
  // a little smaller there: the name keeps its whole width, and the longest status ("Unavailable
  // — the answer stands") gets three lines.
  return (
    <button
      type="button"
      role="tab"
      id={stageTabId(panelId, stage.id)}
      aria-selected={selected}
      aria-controls={panelId}
      aria-label={`${label}: ${status}`}
      tabIndex={selected ? 0 : -1}
      title={STAGE_TECHNICAL_LABELS[stage.id]}
      data-status={stage.status}
      data-agent={stage.id}
      onClick={() => {
        onSelect(stage.id);
      }}
      className={cn(
        styles.tab,
        "relative flex h-18 w-full min-w-0 cursor-pointer flex-col justify-center gap-1 rounded-md border px-2.5 text-left transition-colors hover:bg-accent focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring lg:h-20 lg:px-2",
        visual.border,
        selected &&
          "after:absolute after:inset-x-2.5 after:bottom-0 after:h-0.5 after:rounded-full after:bg-primary lg:after:inset-x-2",
      )}
    >
      {!first && (
        <PipelineConnector active={stage.status !== "pending" && stage.status !== "not_run"} />
      )}
      {stage.status === "running" && (
        <span aria-hidden className={cn(styles.trace, "after:bg-primary")} />
      )}
      <span className="flex min-w-0 items-center gap-1.5 text-sm font-medium text-foreground lg:gap-1">
        <AgentIcon className="size-4 shrink-0 text-muted-foreground" aria-hidden />
        <span className="truncate lg:text-[13px]">{label}</span>
      </span>
      <span className={cn("flex min-w-0 items-start gap-1.5 text-xs lg:gap-1", tone)}>
        <span ref={iconRef} className="inline-flex shrink-0">
          {stage.status === "running" ? (
            <StageGlyph agent={stage.id} />
          ) : (
            <StatusIcon stage={stage} />
          )}
        </span>
        {/* With a badge the text reads exactly as `stageStatusText`: "<status> · N attempts". */}
        <span className="line-clamp-2 lg:line-clamp-3">
          {attempts > 0 ? visual.label : status}
          {attempts > 0 && (
            <>
              {" "}
              <span ref={badgeRef} className="inline-block text-amber-700">
                · {attempts} attempts
              </span>
            </>
          )}
        </span>
      </span>
    </button>
  );
}

/**
 * The stage a key moves the selection to (the WAI tabs pattern): Right and Down forward, Left
 * and Up back, both wrapping; Home and End the first and last. `null` for any other key.
 */
function stepFor(key: string, current: number, count: number): number | null {
  switch (key) {
    case "ArrowRight":
    case "ArrowDown":
      return (current + 1) % count;
    case "ArrowLeft":
    case "ArrowUp":
      return (current - 1 + count) % count;
    case "Home":
      return 0;
    case "End":
      return count - 1;
    default:
      return null;
  }
}

interface AgentPipelineProps {
  model: PipelineModel;
  selectedId: PipelineStageId;
  onSelect: (id: PipelineStageId) => void;
  /** The id of the panel the tabs control (the stage focus card). */
  panelId: string;
}

/**
 * The run's eight stages as an ARIA tablist (plan section 21.5), in workflow order, replacing the
 * React Flow chain (DEC-UDP-03). Every status is written, with an icon and a section 7 tone
 * (UXR-20), and only the selected tab is a tab stop. One column below 640 px, a 4x2 grid from
 * 640 px, one row from 1024 px (UXR-24). Outside the viewport it is marked `data-offscreen`, and
 * its running motion pauses (MOT-17).
 *
 * The keys select and move focus together (UXR-16): the arrows (both pairs at every width), Home
 * and End. Focus moves only in answer to a key; a selection from outside never takes it.
 */
export function AgentPipeline({ model, selectedId, onSelect, panelId }: AgentPipelineProps) {
  const rootRef = useRef<HTMLDivElement>(null);
  useOffscreenPause(rootRef);

  function handlePipelineKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const order = model.stages.map((stage) => stage.id);
    const next = stepFor(event.key, Math.max(0, order.indexOf(selectedId)), order.length);
    const id = next === null ? undefined : order[next];
    if (id === undefined) {
      return;
    }
    event.preventDefault();
    onSelect(id);
    document.getElementById(stageTabId(panelId, id))?.focus();
  }

  return (
    <div
      ref={rootRef}
      onKeyDown={handlePipelineKeyDown}
      role="tablist"
      aria-label="Agent pipeline"
      className={cn(styles.pipeline, "grid grid-cols-1 gap-2 sm:grid-cols-4 lg:grid-cols-8")}
    >
      {model.stages.map((stage, index) => (
        <StageTab
          key={stage.id}
          stage={stage}
          selected={stage.id === selectedId}
          panelId={panelId}
          onSelect={onSelect}
          first={index === 0}
          live={model.phase === "live"}
        />
      ))}
    </div>
  );
}
