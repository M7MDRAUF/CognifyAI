import { ChevronDown, RotateCcw } from "lucide-react";
import { useState } from "react";

import { CopyReferenceButton } from "@/components/chat/copy-reference-button";
import {
  STATUS_VISUALS,
  StatusIcon,
  stageStatusText,
} from "@/components/chat/pipeline/stage-visuals";
import { Badge } from "@/components/ui/badge";
import {
  type PipelineModel,
  type PipelineStage,
  type PipelineStageId,
  STAGE_LABELS,
  STAGE_TECHNICAL_LABELS,
  type StageRecord,
  type StageRound,
} from "@/lib/chat/pipeline";
import type { AgentEvent, EventType } from "@/lib/types/events";
import { providerLabel } from "@/lib/types/llm";
import { cn } from "@/lib/utils";
import { EVALUATOR_SCORE_CAPTION, formatLatency } from "@/lib/utils/format";

/** What each agent does: one fixed, safe sentence per stage (plan UDP2-T02). */
export const STAGE_DESCRIPTIONS: Readonly<Record<PipelineStageId, string>> = {
  planner: "Turns the question into a retrieval plan.",
  retriever: "Finds candidate evidence with semantic and keyword search.",
  reranker: "Orders the candidates by relevance to the question.",
  researcher: "Synthesises the selected evidence into findings.",
  critic: "Checks each claim against the evidence and can ask for more.",
  risk: "Assesses the risks the findings raise.",
  decision: "Writes the recommendation with its citations.",
  evaluator: "Scores the answer; its scores are a model's judgement.",
};

/**
 * A failure class in words (UXR-06, AST-24), keyed by the backend's exception names
 * (`backend/app/core/errors.py`). The raw class is still shown, as secondary text.
 */
export const FAILURE_CLASS_TEXT: Readonly<Record<string, string>> = {
  AgentOutputValidationError: "Its output could not be validated or repaired",
  LLMProviderError: "The model provider returned an error",
  RetrievalFailedError: "Evidence retrieval failed",
  QdrantOperationError: "The vector store could not be reached",
  EmbeddingFailedError: "The embedding step failed",
};
const UNKNOWN_FAILURE_CLASS = "An unexpected error stopped this step";

function failureClassText(classification: string): string {
  return FAILURE_CLASS_TEXT[classification] ?? UNKNOWN_FAILURE_CLASS;
}

/** A recorded step's status, as the attempt history words it. */
const RECORD_STATUS: Readonly<Record<StageRecord["status"], string>> = {
  pending: "Waiting",
  running: "Not closed",
  completed: "Completed",
  failed: "Failed",
};

/** A stage's events, worded as the timeline words them (`agent-timeline.tsx`). */
const ACTIVITY_LABELS: Partial<Record<EventType, string>> = {
  planning_started: "Planning the analysis",
  retrieval_started: "Retrieving evidence",
  reranking_started: "Reranking candidates",
  retrieval_completed: "Evidence selected",
  research_started: "Synthesizing evidence",
  critic_started: "Checking claim support",
  risk_started: "Assessing risks",
  answer_started: "Composing the decision",
  evaluation_started: "Scoring the answer",
};

/** The stage of an event's producer: its agent, else the retrieval component that sent it. */
const STAGE_OF_PRODUCER: Readonly<Record<string, PipelineStageId>> = {
  planner: "planner",
  retriever: "retriever",
  reranker: "reranker",
  researcher: "researcher",
  critic: "critic",
  risk: "risk",
  final_decision: "decision",
  evaluator: "evaluator",
};

function producedBy(event: AgentEvent): PipelineStageId | null {
  const producer = event.agent_name ?? event.component;
  return producer === null || producer === undefined ? null : (STAGE_OF_PRODUCER[producer] ?? null);
}

function roundText(round: unknown): string | null {
  if (typeof round !== "number") return null;
  return round === 0 ? "initial round" : `expansion round ${String(round)}`;
}

/** Round 0 is the initial retrieval; each later round expands it on the critic's feedback. */
function roundName(round: number): string {
  return round === 0 ? "Round 1" : `Round ${String(round + 1)} (expansion)`;
}

/** A round's facts: its status, and what the stream and the records say about it (AST-22). */
function RoundFacts({ round }: { round: StageRound }) {
  const facts = [STATUS_VISUALS[round.status].label];
  if (round.candidates !== null) facts.push(`${String(round.candidates)} candidates`);
  if (round.selected !== null) facts.push(`${String(round.selected)} selected`);
  // A latency is recorded when the round's step closes, as the steps table shows it.
  if (round.latencyMs !== null && (round.status === "completed" || round.status === "failed")) {
    facts.push(formatLatency(round.latencyMs));
  }
  return (
    <li>
      <span className="font-medium">{roundName(round.round)}</span>
      <span className="text-muted-foreground"> · {facts.join(" · ")}</span>
    </li>
  );
}

/** The friendly class, with the raw class beside it as secondary text (AST-24). */
function FailureClass({ classification }: { classification: string }) {
  return (
    <p className="text-muted-foreground">
      {failureClassText(classification)} ·{" "}
      <span className="font-mono text-muted-foreground">{classification}</span>
    </p>
  );
}

/**
 * A failed or interrupted stage (UXR-06): its safe message, its class in words, and the run's ID
 * with a copy action. A run failure carries no correlation id, so only the run ID is offered, and
 * it is labelled as such (BH-02).
 */
function FailureDetails({ stage, runId }: { stage: PipelineStage; runId: string | null }) {
  return (
    <div className="space-y-1 rounded-md border border-destructive/40 p-3 text-xs">
      {stage.errorMessage !== null && <p className="text-foreground">{stage.errorMessage}</p>}
      {stage.classification !== null && <FailureClass classification={stage.classification} />}
      {runId !== null && (
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1 pt-1">
          <p className="min-w-0 break-all">
            <span className="text-muted-foreground">Run ID</span> <code>{runId}</code>
          </p>
          <CopyReferenceButton value={runId} label="Copy run ID" />
        </div>
      )}
    </div>
  );
}

/**
 * A degraded Evaluator (UXR-09, AST-15): what is unavailable and what stands, in amber, beside
 * the answer the page keeps.
 */
function DegradedDetails({ stage }: { stage: PipelineStage }) {
  return (
    <div className="space-y-1 rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-xs">
      <p className="font-medium text-amber-700">The evaluator could not score this answer.</p>
      <p className="text-foreground">
        The answer, its citations and the other stages are complete; only the scores are
        unavailable.
      </p>
      {stage.errorMessage !== null && <p className="text-foreground">{stage.errorMessage}</p>}
      {stage.classification !== null && (
        <p className="text-foreground">
          {failureClassText(stage.classification)} ·{" "}
          <span className="font-mono">{stage.classification}</span>
        </p>
      )}
    </div>
  );
}

/**
 * Every recorded attempt, per recorded step (UXR-05): the records are written as each step
 * closes, so this is history, never a live "Retrying" (BH-01).
 */
function AttemptHistory({
  records,
  id,
  hidden,
}: {
  records: readonly StageRecord[];
  id: string;
  hidden: boolean;
}) {
  return (
    <ul id={id} hidden={hidden} aria-label="Attempts" className="space-y-0.5 text-xs">
      {records.map((record, index) => (
        <li key={index}>
          <span className="font-medium">
            {records.length > 1 ? roundName(index) : "Recorded step"}
          </span>
          <span className="text-muted-foreground">
            {" "}
            · {record.attempts === 1 ? "1 attempt" : `${String(record.attempts)} attempts`} ·{" "}
            {RECORD_STATUS[record.status]}
            {record.classification !== null && ` · ${failureClassText(record.classification)}`}
          </span>
        </li>
      ))}
    </ul>
  );
}

/**
 * The stage a terminal frame closed, as the section follows a finished run: the failed or
 * interrupted stage, else the last stage reached (`followedStage` in `agent-pipeline-section.tsx`).
 */
function closingStage(model: PipelineModel): PipelineStageId | null {
  const failed = model.stages.find(
    (stage) => stage.status === "failed" || stage.status === "interrupted",
  );
  if (failed !== undefined) return failed.id;
  const reached = model.stages.filter(
    (stage) => stage.status !== "pending" && stage.status !== "not_run",
  );
  return reached.at(-1)?.id ?? null;
}

/**
 * A terminal frame the backend rebuilt from the run's saved status (AST-31), said calmly in the
 * card of the stage it closed: plain text, not a live region.
 */
function ReconstructedNote() {
  return (
    <p className="text-xs text-muted-foreground">
      The final state was read from the saved run record.
    </p>
  );
}

interface StageFocusCardProps {
  /** The run's id, which a failed or interrupted stage offers to copy (UXR-18). */
  runId: string;
  stage: PipelineStage;
  /** The run's model, for the run-level notes a stage's card carries. */
  model: PipelineModel;
  /** The stream's events, for the stage's own activity lines. */
  events: readonly AgentEvent[];
  panelId: string;
  /** The selected tab, which names this panel. */
  tabId: string;
}

/**
 * The selected stage's safe details (UXR-15): what the agent does, its status in words, its rounds
 * with the stream's own facts, its recorded latency and model, and the events it produced. It
 * shows no prompt, input, output, normalised question or reasoning, which the detail API does not
 * carry either. The title is a paragraph, not a heading, so the page's heading order is kept, and
 * a reserved height keeps the layout still when the selection changes.
 *
 * A failed or interrupted stage adds its failure (UXR-06), a degraded one what stands (UXR-09),
 * and recorded attempts above 1 an amber badge and their history (UXR-05).
 */
export function StageFocusCard({
  runId,
  stage,
  model,
  events,
  panelId,
  tabId,
}: StageFocusCardProps) {
  // One recorded step: its latency and model. The retriever's rounds carry their own instead.
  const record = stage.records.length === 1 ? stage.records[0] : undefined;
  const latency =
    record !== undefined && (record.status === "completed" || record.status === "failed")
      ? formatLatency(record.latencyMs)
      : null;
  const provider = record?.provider ?? null;
  const recordedModel = record?.model ?? null;
  const modelText =
    provider !== null && recordedModel !== null
      ? `${providerLabel(provider)} · ${recordedModel}`
      : null;
  // A held stop request is a warning on the stage still running (plan section 7).
  const tone =
    stage.status === "running" && stage.stopRequested
      ? "text-amber-700"
      : STATUS_VISUALS[stage.status].tone;
  const activity = events.filter(
    (event) => producedBy(event) === stage.id && ACTIVITY_LABELS[event.event_type] !== undefined,
  );
  // The badge carries the attempts, so the status line gives the status alone beside it.
  const attempts = stage.attempts > 1 && stage.status !== "running" ? stage.attempts : 0;
  const failed = stage.status === "failed" || stage.status === "interrupted";
  // The attempt history is a disclosure, open at first (UXR-17).
  const [historyOpen, setHistoryOpen] = useState(true);
  const historyId = `${panelId}-attempts`;
  return (
    <div
      role="tabpanel"
      id={panelId}
      aria-labelledby={tabId}
      tabIndex={0}
      className="min-h-36 space-y-3 rounded-lg border p-4 text-sm focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
    >
      <div className="space-y-1">
        <p className="font-medium text-foreground">
          {STAGE_LABELS[stage.id]}{" "}
          <span className="text-xs font-normal text-muted-foreground">
            {STAGE_TECHNICAL_LABELS[stage.id]}
          </span>
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <p className={cn("flex items-center gap-1.5 text-xs", tone)}>
            <StatusIcon stage={stage} />
            {attempts > 0 ? STATUS_VISUALS[stage.status].label : stageStatusText(stage)}
          </p>
          {attempts > 0 && (
            <Badge variant="warning" className="gap-1">
              <RotateCcw className="size-3" aria-hidden />
              {attempts} attempts
            </Badge>
          )}
        </div>
      </div>
      <p className="text-muted-foreground">{STAGE_DESCRIPTIONS[stage.id]}</p>
      {stage.id === "evaluator" && (
        <p className="text-xs text-muted-foreground">{EVALUATOR_SCORE_CAPTION}</p>
      )}
      {model.reconstructed && closingStage(model) === stage.id && <ReconstructedNote />}
      {failed && <FailureDetails stage={stage} runId={runId} />}
      {stage.status === "degraded" && <DegradedDetails stage={stage} />}
      {(latency !== null || modelText !== null) && (
        <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
          {latency !== null && (
            <>
              <dt className="text-muted-foreground">Latency</dt>
              <dd>{latency}</dd>
            </>
          )}
          {modelText !== null && (
            <>
              <dt className="text-muted-foreground">Model</dt>
              <dd>{modelText}</dd>
            </>
          )}
        </dl>
      )}
      {stage.rounds.length > 0 && (
        <ul aria-label="Rounds" className="space-y-0.5 text-xs">
          {stage.rounds.map((round) => (
            <RoundFacts key={round.round} round={round} />
          ))}
        </ul>
      )}
      {stage.records.some((record) => record.attempts > 1) && (
        <div className="space-y-1">
          <button
            type="button"
            aria-expanded={historyOpen}
            aria-controls={historyId}
            onClick={() => {
              setHistoryOpen((open) => !open);
            }}
            className="inline-flex min-h-6 cursor-pointer items-center gap-1 rounded-sm text-xs font-medium text-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
          >
            <ChevronDown
              className={cn("size-3.5 text-muted-foreground", !historyOpen && "-rotate-90")}
              aria-hidden
            />
            Attempt history
          </button>
          <AttemptHistory records={stage.records} id={historyId} hidden={!historyOpen} />
        </div>
      )}
      {activity.length > 0 && (
        <ul aria-label="Activity" className="space-y-0.5 text-xs text-muted-foreground">
          {activity.map((event) => {
            const round = roundText(event.payload.round);
            return (
              <li key={event.id}>
                {ACTIVITY_LABELS[event.event_type]}
                {round !== null && ` · ${round}`}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
