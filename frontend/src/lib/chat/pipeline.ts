import type { AgentEvent, EventType } from "@/lib/types/events";
import type { AgentName, AgentRun, QueryRunStatus } from "@/lib/types/query";

/**
 * The Chat pipeline's state model (UI-D-PRO plan section 9, AST-01 to AST-34). One pure function
 * turns the run's event stream and its recorded steps into eight stages with truthful statuses.
 * It lives outside `src/lib/types/`, whose declarations the backend's SSE mirror pins (APR-01).
 *
 * Rules, in order:
 * 1. Events are reduced in arrival order, never by `created_at`, and de-duplicated by `id`.
 * 2. The first terminal event fixes the run phase; later frames are ignored.
 * 3. Records merge: while the run is live the more advanced status wins; once it is terminal the
 *    records are authoritative for every stage that has one.
 * 4. The reranker has no record: its status comes from its events and, once the run has ended,
 *    from the retriever's record, which covers a round's gather and its rerank and closes only
 *    after the rerank (AST-25; UDP6-T02, RV-02-02 and RV-02-05).
 * 5. The stage closed by `answer_completed` is provisional until the detail's rows confirm it
 *    (BH-04, AST-30).
 * 6. After a failure or a cancel, stages that never started are `not_run`.
 * 7. A terminal frame rebuilt from the run's status carries `reconstructed_from_status` (AST-31).
 */

export const PIPELINE_STAGES = [
  "planner",
  "retriever",
  "reranker",
  "researcher",
  "critic",
  "risk",
  "decision",
  "evaluator",
] as const;

export type PipelineStageId = (typeof PIPELINE_STAGES)[number];

export type StageStatus =
  | "pending"
  | "running"
  | "completed"
  | "degraded"
  | "failed"
  | "interrupted"
  | "cancelled"
  | "not_run";

export type RunPhase = "waiting" | "live" | "completed" | "failed" | "cancelled";

/** One retrieval round of the retriever or the reranker; round 0 is initial, 1 and up expand it. */
export interface StageRound {
  round: number;
  status: StageStatus;
  latencyMs: number | null;
  attempts: number;
  /** Candidates handed to the reranker (its `reranking_started` payload). */
  candidates: number | null;
  /** Evidence selected by the reranker (its `retrieval_completed` payload). */
  selected: number | null;
}

/** One recorded step of the stage, in the order the run recorded it. */
export interface StageRecord {
  status: AgentRun["status"];
  latencyMs: number | null;
  attempts: number;
  errorMessage: string | null;
  classification: string | null;
  provider: string | null;
  model: string | null;
}

export interface PipelineStage {
  id: PipelineStageId;
  status: StageStatus;
  /** The last event that touched the stage: the safe activity summary. */
  activity: EventType | null;
  /** The latest recorded step's attempts; 0 when the stage has no record. */
  attempts: number;
  latencyMs: number | null;
  errorMessage: string | null;
  classification: string | null;
  rounds: StageRound[];
  records: StageRecord[];
  /** Closed by `answer_completed` and not yet confirmed by the detail's rows (BH-04). */
  provisional: boolean;
  /** A Stop was requested while this stage ran (AST-34). */
  stopRequested: boolean;
}

export interface PipelineModel {
  stages: PipelineStage[];
  phase: RunPhase;
  outcome:
    | "completed"
    | "completed_without_answer"
    | "completed_degraded"
    | "completed_with_failure"
    | "failed"
    | "cancelled"
    | null;
  runningStage: PipelineStageId | null;
  /** The terminal frame was rebuilt from the run's status (AST-31). */
  reconstructed: boolean;
  /** The `workflow_failed` event's message, when the stream carried one. */
  terminalMessage: string | null;
  /** A Stop is held until a terminal event (AST-34). */
  stopRequested: boolean;
}

/** What the model reads from the run's detail. */
export interface PipelineRunInput {
  status: QueryRunStatus;
  agent_runs: readonly AgentRun[];
  answer_markdown: string | null;
}

export const STAGE_LABELS: Readonly<Record<PipelineStageId, string>> = {
  planner: "Planner",
  retriever: "Retriever",
  reranker: "Reranker",
  researcher: "Researcher",
  critic: "Critic",
  risk: "Risk",
  decision: "Decision",
  evaluator: "Evaluator",
};

/** The backend's names for the steps, shown as secondary labels. */
export const STAGE_TECHNICAL_LABELS: Readonly<Record<PipelineStageId, string>> = {
  planner: "planner",
  retriever: "retriever",
  reranker: "reranker",
  researcher: "researcher",
  critic: "critic",
  risk: "risk",
  decision: "final_decision",
  evaluator: "evaluator",
};

const STAGE_OF_AGENT: Readonly<Record<AgentName, PipelineStageId>> = {
  planner: "planner",
  retriever: "retriever",
  researcher: "researcher",
  critic: "critic",
  risk: "risk",
  final_decision: "decision",
  evaluator: "evaluator",
};

/** The stage each start event opens. */
const OPENS: Partial<Record<EventType, PipelineStageId>> = {
  planning_started: "planner",
  retrieval_started: "retriever",
  reranking_started: "reranker",
  research_started: "researcher",
  critic_started: "critic",
  risk_started: "risk",
  answer_started: "decision",
  evaluation_started: "evaluator",
};

const TERMINAL: ReadonlySet<EventType> = new Set<EventType>([
  "answer_completed",
  "workflow_failed",
  "workflow_cancelled",
]);

/** pending, then running, then any closed status. */
function rank(status: StageStatus): number {
  if (status === "pending") return 0;
  if (status === "running") return 1;
  return 2;
}

function numberIn(payload: Record<string, unknown>, key: string): number | null {
  const value = payload[key];
  return typeof value === "number" ? value : null;
}

interface Draft {
  status: StageStatus;
  activity: EventType | null;
  errorMessage: string | null;
  /** A failure class the stage takes from another stage's record (the reranker's, RV-02-08). */
  classification: string | null;
  rounds: Map<number, StageRound>;
  provisional: boolean;
}

type Drafts = Record<PipelineStageId, Draft>;
type Records = Record<PipelineStageId, StageRecord[]>;

interface Reduced {
  drafts: Drafts;
  terminal: EventType | null;
  terminalMessage: string | null;
  reconstructed: boolean;
  /** The stage open when the terminal event arrived. */
  openAtTerminal: PipelineStageId | null;
  /** The round of the last `reranking_started`, for a failure closed during reranking. */
  rerankRound: number;
  touched: boolean;
}

function perStage<T>(make: () => T): Record<PipelineStageId, T> {
  return {
    planner: make(),
    retriever: make(),
    reranker: make(),
    researcher: make(),
    critic: make(),
    risk: make(),
    decision: make(),
    evaluator: make(),
  };
}

function roundOf(draft: Draft, round: number): StageRound {
  const existing = draft.rounds.get(round);
  if (existing !== undefined) return existing;
  const created: StageRound = {
    round,
    status: "pending",
    latencyMs: null,
    attempts: 0,
    candidates: null,
    selected: null,
  };
  draft.rounds.set(round, created);
  return created;
}

/** Rule 1 and 2: the events in arrival order, each id once, nothing after the terminal event. */
function reduceEvents(events: readonly AgentEvent[]): Reduced {
  const drafts: Drafts = perStage(() => ({
    status: "pending",
    activity: null,
    errorMessage: null,
    classification: null,
    rounds: new Map<number, StageRound>(),
    provisional: false,
  }));
  const reduced: Reduced = {
    drafts,
    terminal: null,
    terminalMessage: null,
    reconstructed: false,
    openAtTerminal: null,
    rerankRound: 0,
    touched: false,
  };
  const seen = new Set<string>();
  let open: PipelineStageId | null = null;

  for (const event of events) {
    if (seen.has(event.id) || reduced.terminal !== null) continue;
    seen.add(event.id);
    reduced.touched = true;
    const round = numberIn(event.payload, "round") ?? 0;
    const opened = OPENS[event.event_type];
    if (opened !== undefined) {
      // The orchestrator starts a stage only after the previous one closed.
      if (open !== null && open !== opened) drafts[open].status = "completed";
      const stage = drafts[opened];
      stage.status = "running";
      stage.activity = event.event_type;
      if (opened === "retriever") {
        roundOf(stage, round).status = "running";
      } else if (opened === "reranker") {
        // Reranking closes the retrieval round it ranks (AST-03).
        roundOf(drafts.retriever, round).status = "completed";
        const ranking = roundOf(stage, round);
        ranking.status = "running";
        ranking.candidates = numberIn(event.payload, "candidates");
        reduced.rerankRound = round;
      }
      open = opened;
      continue;
    }
    if (event.event_type === "retrieval_completed") {
      // The reranked round's evidence is selected (AST-04). A stream with no `reranking_started`
      // for the round never showed the reranker working, so it gains no round of its own.
      const reranker = drafts.reranker;
      const ranked = reranker.rounds.get(round);
      if (ranked !== undefined) {
        ranked.status = "completed";
        ranked.selected = numberIn(event.payload, "count");
        reranker.activity = event.event_type;
        reranker.status = "completed";
      }
      roundOf(drafts.retriever, round).status = "completed";
      if (drafts.retriever.status === "running") drafts.retriever.status = "completed";
      open = null;
      continue;
    }
    if (TERMINAL.has(event.event_type)) {
      reduced.terminal = event.event_type;
      reduced.openAtTerminal = open;
      reduced.reconstructed = event.payload.reconstructed_from_status === true;
      if (event.event_type === "workflow_failed") reduced.terminalMessage = event.message;
      if (open !== null) {
        const stage = drafts[open];
        if (event.event_type === "answer_completed") {
          stage.status = "completed";
          stage.provisional = true;
        } else if (event.event_type === "workflow_failed") {
          stage.status = "failed";
          stage.errorMessage = event.message;
        } else {
          stage.status = "cancelled";
        }
        // A round still open when the run ended ends as its stage did.
        for (const item of stage.rounds.values()) {
          if (item.status === "running") item.status = stage.status;
        }
      }
      open = null;
    }
  }
  return reduced;
}

function recordOf(run: AgentRun): StageRecord {
  return {
    status: run.status,
    latencyMs: run.latency_ms,
    attempts: run.attempt_count,
    errorMessage: run.error_message,
    classification: run.error_classification ?? null,
    provider: run.llm_provider ?? null,
    model: run.llm_model ?? null,
  };
}

/** A recorded step's status in a finished run (AST-13 to AST-19). */
function terminalStatus(
  record: StageRecord,
  runStatus: QueryRunStatus,
  stage: PipelineStageId,
): StageStatus {
  if (record.status === "completed") return "completed";
  if (record.status === "failed") {
    // The evaluator never fails the run; a completed run with a failed evaluator is degraded.
    return stage === "evaluator" && runStatus === "completed" ? "degraded" : "failed";
  }
  if (record.status === "pending") return "not_run";
  // RUNNING in a finished run: left by a cancel or by the run's own failure (AST-17, AST-18).
  if (runStatus === "cancelled") return "cancelled";
  if (runStatus === "failed") return "interrupted";
  // A completed run leaves no step running: an unclosed record of it reads as completed.
  return "completed";
}

/** Rule 3: the recorded steps, merged into the event-derived drafts. */
function mergeRecords(drafts: Drafts, run: PipelineRunInput | undefined): Records {
  const records: Records = perStage((): StageRecord[] => []);
  if (run === undefined) return records;
  for (const row of run.agent_runs) {
    records[STAGE_OF_AGENT[row.agent_name]].push(recordOf(row));
  }
  const live = run.status === "running";
  // While the run is live a record can only move a stage forward; once it ended, it decides.
  const merged = (current: StageStatus, record: StageRecord, id: PipelineStageId): StageStatus => {
    if (!live) return terminalStatus(record, run.status, id);
    return rank(record.status) > rank(current) ? record.status : current;
  };
  for (const id of PIPELINE_STAGES) {
    const rows = records[id];
    const latest = rows.at(-1);
    if (latest === undefined) continue;
    const stage = drafts[id];
    stage.status = merged(stage.status, latest, id);
    if (!live) stage.provisional = false;
  }
  // Each retriever record is one retrieval round, in order (AST-22).
  records.retriever.forEach((row, index) => {
    const item = roundOf(drafts.retriever, index);
    item.latencyMs = row.latencyMs;
    item.attempts = row.attempts;
    item.status = merged(item.status, row, "retriever");
  });
  if (live && records.retriever.length > 0) {
    // A live retriever is where its latest round is: a record of an earlier round, read before an
    // expansion began, cannot close the round the events show running (UDP6-T02, RV-02-03).
    const latestRound = [...drafts.retriever.rounds.values()].reduce((last, item) =>
      item.round > last.round ? item : last,
    );
    drafts.retriever.status = latestRound.status;
  }
  return records;
}

/**
 * Rule 4, once the run has ended. The retriever's record covers a round's gather and its rerank,
 * and closes only after the rerank; a cancel leaves it RUNNING (AST-25). So a round whose record
 * completed was reranked, and a run that ended while the reranker was open ended the rerank, not
 * the gather: the record's failure, class included, is the reranker's, and a record the end left
 * running leaves the reranker cancelled or unfinished (UDP6-T02, RV-02-02, RV-02-05, RV-02-08).
 */
function attributeRerank(
  reduced: Reduced,
  records: Records,
  run: PipelineRunInput | undefined,
): void {
  if (run === undefined || run.status === "running") return;
  const row = records.retriever.at(-1);
  if (row === undefined) return;
  const { reranker, retriever } = reduced.drafts;
  if (row.status === "completed") {
    records.retriever.forEach((item, index) => {
      if (item.status === "completed") roundOf(reranker, index).status = "completed";
    });
    reranker.status = "completed";
    return;
  }
  if (reduced.openAtTerminal !== "reranker") return;
  let ended: StageStatus;
  if (row.status === "failed") ended = "failed";
  else if (row.status === "running")
    ended = run.status === "cancelled" ? "cancelled" : "interrupted";
  else return;
  // The retrieval round itself completed before the reranker ran.
  retriever.status = "completed";
  roundOf(retriever, reduced.rerankRound).status = "completed";
  reranker.status = ended;
  if (ended === "failed") {
    reranker.errorMessage = row.errorMessage ?? reduced.terminalMessage;
    reranker.classification = row.classification;
  }
  const ranking = reranker.rounds.get(reduced.rerankRound);
  if (ranking !== undefined) ranking.status = ended;
}

function phaseOf(reduced: Reduced, run: PipelineRunInput | undefined): RunPhase {
  if (run !== undefined && run.status !== "running") return run.status;
  if (reduced.terminal === "answer_completed") return "completed";
  if (reduced.terminal === "workflow_failed") return "failed";
  if (reduced.terminal === "workflow_cancelled") return "cancelled";
  return reduced.touched ? "live" : "waiting";
}

/** A stage still running when the run's recorded status ended it (the reranker has no record). */
function endedWith(phase: RunPhase): StageStatus {
  if (phase === "cancelled") return "cancelled";
  if (phase === "failed") return "failed";
  return "completed";
}

/**
 * The pipeline for one run: `events` as the stream delivered them, `run` the detail when it has
 * been read, and `stopRequested` the view's held Stop request.
 */
export function derivePipeline(
  events: readonly AgentEvent[],
  run: PipelineRunInput | undefined,
  stopRequested: boolean,
): PipelineModel {
  const reduced = reduceEvents(events);
  const phase = phaseOf(reduced, run);
  const terminal = phase !== "waiting" && phase !== "live";
  const records = mergeRecords(reduced.drafts, run);
  attributeRerank(reduced, records, run);
  const confirmed = run !== undefined && run.status !== "running";

  const stages: PipelineStage[] = PIPELINE_STAGES.map((id) => {
    const draft = reduced.drafts[id];
    const rows = records[id];
    const latest = rows.at(-1);
    let status = draft.status;
    // Rule 6: after the run ended, a stage nothing started did not run, and one the events
    // left running ended as the run did.
    if (terminal && latest === undefined) {
      if (status === "pending") status = "not_run";
      else if (status === "running") status = endedWith(phase);
    }
    return {
      id,
      status,
      activity: draft.activity,
      attempts: latest?.attempts ?? 0,
      latencyMs: latest?.latencyMs ?? null,
      errorMessage: latest?.errorMessage ?? draft.errorMessage,
      classification: latest?.classification ?? draft.classification,
      rounds: [...draft.rounds.values()].sort((a, b) => a.round - b.round),
      records: rows,
      provisional: draft.provisional && !confirmed,
      stopRequested: stopRequested && !terminal && status === "running",
    };
  });

  let outcome: PipelineModel["outcome"] = null;
  if (phase === "completed") {
    if (stages.some((stage) => stage.status === "degraded")) outcome = "completed_degraded";
    else if (confirmed && run.answer_markdown === null) outcome = "completed_without_answer";
    else outcome = "completed";
  } else if (phase === "failed") {
    // D-3, decided as the page decides it: any stored answer, even an empty one (RV-02-07).
    outcome =
      run !== undefined && run.answer_markdown !== null ? "completed_with_failure" : "failed";
  } else if (phase === "cancelled") {
    outcome = "cancelled";
  }

  return {
    stages,
    phase,
    outcome,
    runningStage: stages.find((stage) => stage.status === "running")?.id ?? null,
    reconstructed: reduced.reconstructed,
    terminalMessage: reduced.terminalMessage,
    stopRequested: stopRequested && !terminal,
  };
}
