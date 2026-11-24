"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type ReadonlyURLSearchParams, useSearchParams } from "next/navigation";
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { AgentRunsPanel } from "@/components/chat/agent-runs-panel";
import { CopyReferenceButton } from "@/components/chat/copy-reference-button";
import { AgentPipelineSection } from "@/components/chat/pipeline/agent-pipeline-section";
import { QueryHistoryPanel } from "@/components/chat/query-history-panel";
import { QuestionForm } from "@/components/chat/question-form";
import { RunResult } from "@/components/chat/run-result";
import { PharmaInquiryPanel } from "@/components/pharma/pharma-inquiry-panel";
import { ChatPrivacyNotice } from "@/components/privacy/data-processing-notice";
import { Button } from "@/components/ui/button";
import { LocalTime } from "@/components/ui/local-time";
import { ApiError } from "@/lib/api/client";
import { getProject } from "@/lib/api/projects";
import { cancelQuery, getQueryDetail, submitQuery } from "@/lib/api/queries";
import { useLlmConfig } from "@/lib/hooks/use-llm-config";
import { useQueryRunEvents } from "@/lib/hooks/use-query-run-events";
import { runSummariesKey, useRunSummaries } from "@/lib/hooks/use-run-summaries";
import { useSubmitLock } from "@/lib/hooks/use-submit-lock";
import {
  type DraftErrors,
  EMPTY_PHARMA_DRAFT,
  type PharmaDraft,
  buildPharmaInput,
} from "@/lib/pharma/incident-form";
import { providerLabel } from "@/lib/types/llm";
import { profileOf } from "@/lib/types/pharma";
import type {
  QueryCreateInput,
  QueryRun,
  QueryRunDetail,
  QueryRunSummary,
} from "@/lib/types/query";
import { isUuid } from "@/lib/utils/validation";

/**
 * The `run` search parameter, validated before it can reach any request: a UUID, nothing (no run
 * selected), or invalid (shown as an error rather than sent to the backend).
 */
export function parseRunParam(value: string | null | undefined): {
  runId: string | null;
  invalid: boolean;
} {
  if (value === null || value === undefined || value === "") {
    return { runId: null, invalid: false };
  }
  return isUuid(value)
    ? { runId: value.toLowerCase(), invalid: false }
    : { runId: null, invalid: true };
}

/**
 * What happened to a run, from its detail (FE-003). A failed run can still carry the answer it
 * produced before a later step failed (D-3); that is its own outcome, not "failed", because
 * hiding a paid-for answer behind an error is the misreport the certified harness forbids.
 */
type RunOutcome =
  | "running"
  | "completed"
  | "completed_without_answer"
  | "completed_with_failure"
  | "failed"
  | "cancelled";

function runOutcome(run: QueryRunDetail): RunOutcome {
  switch (run.status) {
    case "running":
      return "running";
    case "cancelled":
      return "cancelled";
    case "completed":
      return run.answer_markdown === null ? "completed_without_answer" : "completed";
    case "failed":
      return run.answer_markdown === null ? "failed" : "completed_with_failure";
  }
}

function modelLabel(
  run: Pick<QueryRun, "resolved_provider" | "resolved_model_name" | "answer_model">,
): string {
  return run.resolved_provider !== null &&
    run.resolved_provider !== undefined &&
    run.resolved_model_name !== null &&
    run.resolved_model_name !== undefined
    ? `${providerLabel(run.resolved_provider)} · ${run.resolved_model_name}`
    : `${run.answer_model} model tier`;
}

/** Puts the viewed run in the address bar without a history entry (Next.js native History API). */
function writeRunToUrl(runId: string): void {
  window.history.replaceState(null, "", `?run=${runId}`);
}

// Render isolation (UDF-01): every stream event re-renders the view, and these children do not
// depend on the events. With stable props they are skipped, so an event re-renders only what
// shows the run's progress.
const StableQuestionForm = memo(QuestionForm);
const StableHistoryPanel = memo(QueryHistoryPanel);
const StablePrivacyNotice = memo(ChatPrivacyNotice);
const StableAgentRunsPanel = memo(AgentRunsPanel);

/** One empty error set, so the panel's errors prop is stable while nothing is shown. */
const NO_ERRORS: DraftErrors = {};

export const PROFILE_UNKNOWN_MESSAGE =
  "This project could not be read, so the chat cannot tell which profile it uses. Try again to " +
  "ask a question.";

interface ChatViewProps {
  projectId: string;
  /** The raw `?run=` value of the page URL, if any (FE-011). */
  initialRunParam?: string | null;
}

/** A Chat error as shown: its text, and the reference to copy when the text shows one. */
interface ChatError {
  message: string;
  reference: string | null;
}

/**
 * The correlation id an error's text shows: at status 500 and above, as the client words it
 * (`withReference` in `frontend/src/lib/api/error-messages.ts`). Below 500 there is none to copy.
 */
function referenceOf(error: ApiError): string | null {
  return error.status >= 500 && error.correlationId !== null && error.correlationId !== ""
    ? error.correlationId
    : null;
}

function chatError(cause: unknown, fallback: string): ChatError {
  return cause instanceof ApiError
    ? { message: cause.message, reference: referenceOf(cause) }
    : { message: fallback, reference: null };
}

export function ChatView({ projectId, initialRunParam = null }: ChatViewProps) {
  const queryClient = useQueryClient();
  const [runParam, setRunParam] = useState(initialRunParam);
  const [queryRunId, setQueryRunId] = useState<string | null>(
    () => parseRunParam(initialRunParam).runId,
  );
  const [invalidRunParam, setInvalidRunParam] = useState(
    () => parseRunParam(initialRunParam).invalid,
  );
  // A run this view started, or picked from this project's own history, is known to belong here.
  // A run named only by the URL is confirmed against its detail first (FE-011, isolation).
  const [trustedRunId, setTrustedRunId] = useState<string | null>(null);
  // The submit failure: its text, and the reference it shows at status 500 and above.
  const [submitError, setSubmitError] = useState<ChatError | null>(null);
  // The failed Stop of the viewed run, shown in its status bar (RMD5-T04, RE2E-DEF-040).
  const [stopError, setStopError] = useState<ChatError | null>(null);
  // The run a Stop was requested for (UDF-04). The request holds from the click until that run's
  // terminal event, so the control cannot send a second cancel; it is read only for the viewed run
  // while that run still runs, so a run change or a terminal event ends it without an effect.
  const [stopRequestedFor, setStopRequestedFor] = useState<string | null>(null);
  const runHeadingRef = useRef<HTMLHeadingElement>(null);
  // Set when a history pick should move focus to the run it opened (not for a new submission).
  const focusRunRef = useRef(false);

  // The address bar's `?run=` as the router holds it. The page's prop is only the value the server
  // rendered: the Chat tab navigates back to that URL and leaves the prop as it was, so a view that
  // followed the prop alone kept a run on screen the address bar no longer named (RE2E-DEF-012).
  // Next syncs our own `replaceState` into these search params. Outside the app router (unit
  // tests) there are none, and the prop is the URL. Widened to nullable because the hook returns
  // null there (next/dist/client/components/navigation.js), though its declared type says not.
  const searchParams = useSearchParams() as ReadonlyURLSearchParams | null;
  const urlRunParam = searchParams === null ? initialRunParam : searchParams.get("run");

  // A navigation to another `?run=`, or to none (a link, back/forward, the Chat tab); follow it.
  if (urlRunParam !== runParam) {
    setRunParam(urlRunParam);
    const parsed = parseRunParam(urlRunParam);
    setQueryRunId(parsed.runId);
    setInvalidRunParam(parsed.invalid);
  }

  const showRun = useCallback((id: string, options: { focus: boolean }) => {
    setQueryRunId(id);
    setTrustedRunId(id);
    setInvalidRunParam(false);
    setStopError(null);
    focusRunRef.current = options.focus;
    writeRunToUrl(id);
  }, []);

  const handleSelectQuery = useCallback(
    (id: string) => {
      showRun(id, { focus: true });
      setSubmitError(null);
    },
    [showRun],
  );

  useEffect(() => {
    if (focusRunRef.current) {
      runHeadingRef.current?.focus();
      focusRunRef.current = false;
    }
  }, [queryRunId]);

  const runTrusted = queryRunId !== null && queryRunId === trustedRunId;
  const [streamIsComplete, setStreamIsComplete] = useState(false);
  const detail = useQuery({
    queryKey: ["query", queryRunId],
    queryFn: () => getQueryDetail(queryRunId ?? ""),
    // A URL run is read at once, to confirm it exists in this project before anything streams.
    enabled: queryRunId !== null && (streamIsComplete || !runTrusted),
  });
  const runNotFound =
    queryRunId !== null &&
    !runTrusted &&
    ((detail.error instanceof ApiError && detail.error.status === 404) ||
      (detail.data !== undefined && detail.data.project_id !== projectId));
  const runConfirmed = queryRunId !== null && (runTrusted || detail.data?.project_id === projectId);
  const streamRunId = runConfirmed ? queryRunId : null;
  // Run creation is not idempotent (known limitation 4): each accepted POST is a billed run.
  const submitLock = useSubmitLock();

  const { events, isComplete, streamStatus, invalidFrames, detailError, retry } = useQueryRunEvents(
    streamRunId,
    (terminalDetail) => {
      if (queryRunId !== null) {
        if (terminalDetail !== undefined) {
          // A detail read ended the run: use it as the result, so the page shows the outcome at
          // once and does not depend on reading it again (UDP6-T02, RV-01-02).
          queryClient.setQueryData(["query", queryRunId], terminalDetail);
        } else {
          void queryClient.invalidateQueries({ queryKey: ["query", queryRunId] });
        }
      }
      // The history lists this run too; without a refetch it stays "running" (FE-029).
      void queryClient.invalidateQueries({ queryKey: runSummariesKey(projectId) });
    },
  );

  if (isComplete !== streamIsComplete) {
    setStreamIsComplete(isComplete);
  }

  // Capability lookup is static for the life of the process. A failure leaves the config
  // undefined with status "error", which locks the model selector — and, crucially, is told
  // apart from "the backend reports nothing configured" everywhere it is rendered.
  const capability = useLlmConfig();

  const mutation = useMutation({
    mutationFn: (input: QueryCreateInput) => submitQuery(projectId, input),
    onSuccess: (run) => {
      setSubmitError(null);
      // RE2E-DEF-046: the inquiry details belong to the question just accepted; left filled, they
      // went with the next one. A refused question keeps them, to be sent again.
      setPharmaDraft(EMPTY_PHARMA_DRAFT);
      showRun(run.id, { focus: false });
      void queryClient.invalidateQueries({ queryKey: runSummariesKey(projectId) });
    },
    onError: (cause: unknown) => {
      // Released at once, so the 409 capacity message renders and the user can ask again.
      submitLock.release();
      setSubmitError(chatError(cause, "Could not start the analysis."));
    },
  });

  // REQ-UX-04: a pharmaceutical project's inquiry details. The project is read from the query the
  // project's guard has already loaded (`ProjectGuard`), never fetched here, so a General chat
  // makes exactly the requests it made before; until it is known, the chat is General.
  const project = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => getProject(projectId),
    enabled: false,
  });
  const pharmaceutical =
    project.data !== undefined && profileOf(project.data) === "pharma_manufacturing";
  // The guard renders the page even when its read failed for a reason other than "not found"; the
  // profile is then unknown, so the chat says so in place of the question form rather than
  // silently running a pharmaceutical project's question as a General one (PH-DEF-040).
  const profileUnknown = project.isError;
  const [pharmaDraft, setPharmaDraft] = useState<PharmaDraft>(EMPTY_PHARMA_DRAFT);
  const [showPharmaErrors, setShowPharmaErrors] = useState(false);
  const [pharmaFocusRequest, setPharmaFocusRequest] = useState(0);
  // Memoised, so a stream event does not hand the question form a new submit handler (UDF-01).
  const pharmaInput = useMemo(
    () => (pharmaceutical ? buildPharmaInput(pharmaDraft) : null),
    [pharmaceutical, pharmaDraft],
  );
  const pharmaErrors =
    showPharmaErrors && pharmaInput !== null && !pharmaInput.ok ? pharmaInput.errors : NO_ERRORS;

  const { acquire: acquireSubmitLock } = submitLock;
  const { mutate: submitRun } = mutation;
  const handleSubmit = useCallback(
    (input: QueryCreateInput) => {
      let request = input;
      if (pharmaInput !== null) {
        if (!pharmaInput.ok) {
          // Refused before any request; the panel focuses the first field in error.
          setShowPharmaErrors(true);
          setPharmaFocusRequest((count) => count + 1);
          return;
        }
        request = { ...input, pharma: pharmaInput.input };
      }
      if (!acquireSubmitLock()) {
        return;
      }
      setShowPharmaErrors(false);
      submitRun(request);
    },
    [acquireSubmitLock, submitRun, pharmaInput],
  );

  const cancelMutation = useMutation({
    mutationFn: (id: string) => cancelQuery(id),
    onError: (cause: unknown, id) => {
      // A 409 query_run_not_cancellable used to be swallowed as the race with the run's own end,
      // holding "Stopping…" for a terminal event that, for a run no process owned any more, never
      // came (RE2E-DEF-041). The backend now closes such a run itself and answers 202, so a 409
      // means the run has already ended, or another server process owns it. The message below
      // says so, and a fresh stream replays the run's stored end, or reconstructs it from the
      // run's status (RMD5-T01).
      if (cause instanceof ApiError && cause.code === "query_run_not_cancellable") {
        retry();
      }
      // A refused stop leaves the run as it was: the control is given back, and the failure is
      // said at the run's status bar, where the Stop was pressed (RMD5-T04, RE2E-DEF-040).
      setStopRequestedFor((current) => (current === id ? null : current));
      setStopError(chatError(cause, "Check your connection, then try again."));
    },
  });

  // A stream that failed past the polling bound no longer counts as running: the form is free
  // again and the failure below says what is known (FE-002).
  const running = streamRunId !== null && !isComplete && streamStatus !== "failed";
  const stopRequested = running && stopRequestedFor === streamRunId;
  // Nothing of a run is shown until it is confirmed to belong to this project.
  const result = runConfirmed ? detail.data : undefined;
  const detailReference = detail.error instanceof ApiError ? referenceOf(detail.error) : null;
  // RMD5-T04 (RE2E-DEF-039): neither the stream nor the detail answers, and nothing of the run has
  // been seen. A finished run picked from the history used to read "Waiting for the workflow to
  // start" here and "It may still be running"; its load error, and its reference, are shown instead.
  const loadError =
    streamStatus !== "live" && detailError !== null && events.length === 0
      ? chatError(detailError, "Check your connection, then reconnect.")
      : null;
  const outcome = result === undefined ? null : runOutcome(result);
  // The run's question and provenance head the result (FE-030): from its detail once loaded, and
  // before that from the pages of this project's run summaries already loaded (the history
  // panel's infinite query). A URL run older than those pages shows its heading with its detail.
  const runs = useRunSummaries(projectId, "all");
  const runSummary: QueryRunDetail | QueryRunSummary | undefined =
    result ?? runs.data?.pages.flatMap((page) => page.items).find((run) => run.id === streamRunId);

  // Held from the click until the run it started stops running; released here rather than in
  // `onSettled`, which fires before the re-render that disables the form.
  const { release: releaseSubmitLock } = submitLock;
  useEffect(() => {
    if (!running && !mutation.isPending) {
      releaseSubmitLock();
    }
  }, [running, mutation.isPending, releaseSubmitLock]);

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold tracking-tight">Ask a question</h1>
      <StablePrivacyNotice llmConfig={capability.config} capabilityStatus={capability.status} />
      {profileUnknown && (
        <div className="space-y-2">
          <p role="alert" className="text-sm text-destructive">
            {PROFILE_UNKNOWN_MESSAGE}
          </p>
          {/* RE2E-DEF-016: the project is read again here, so a reload is not the only way back. */}
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={project.isFetching}
            onClick={() => {
              void project.refetch();
            }}
          >
            {project.isFetching ? "Retrying…" : "Try again"}
          </Button>
        </div>
      )}
      {pharmaceutical && (
        <PharmaInquiryPanel
          draft={pharmaDraft}
          onChange={setPharmaDraft}
          errors={pharmaErrors}
          focusRequest={pharmaFocusRequest}
          disabled={mutation.isPending || running}
        />
      )}
      {/* Not merely disabled: a disabled form reads "Analyzing…", and nothing is running. */}
      {!profileUnknown && (
        <StableQuestionForm
          disabled={mutation.isPending || running}
          llmConfig={capability.config}
          capabilityStatus={capability.status}
          onSubmit={handleSubmit}
        />
      )}
      {submitError !== null && (
        <div role="alert" className="space-y-2 text-sm text-destructive">
          <p>{submitError.message}</p>
          {submitError.reference !== null && (
            <CopyReferenceButton value={submitError.reference} label="Copy reference" />
          )}
        </div>
      )}

      <StableHistoryPanel
        projectId={projectId}
        currentQueryRunId={queryRunId}
        onSelectQuery={handleSelectQuery}
      />

      {invalidRunParam && (
        <p role="alert" className="text-sm text-destructive">
          This link does not point to a valid analysis. Check the address, or pick a run from the
          history.
        </p>
      )}
      {runNotFound && (
        <p role="alert" className="text-sm text-destructive">
          This analysis could not be found in this project. It may have been deleted, or the link is
          wrong.
        </p>
      )}

      {streamRunId !== null && (
        // `data-stream` pauses the pipeline's running motion once live progress is unavailable
        // (AST-33); a Reconnect sets it back to `live`.
        <section
          className="space-y-4"
          aria-labelledby="selected-analysis-heading"
          data-stream={streamStatus}
        >
          <h2
            id="selected-analysis-heading"
            ref={runHeadingRef}
            tabIndex={-1}
            className="text-lg font-semibold tracking-tight focus-visible:outline-2 focus-visible:outline-offset-2"
          >
            Selected analysis
          </h2>
          {runSummary !== undefined && (
            <div className="space-y-1">
              <p className="font-medium">{runSummary.question}</p>
              <p className="text-xs text-muted-foreground">
                Asked <LocalTime iso={runSummary.created_at} /> · {modelLabel(runSummary)}
              </p>
            </div>
          )}
          {loadError !== null ? (
            <div role="alert" className="space-y-2 text-sm text-destructive">
              <p>The analysis could not be loaded. {loadError.message}</p>
              <div className="flex flex-wrap gap-2">
                {loadError.reference !== null && (
                  <CopyReferenceButton value={loadError.reference} label="Copy reference" />
                )}
                <Button type="button" variant="outline" size="sm" onClick={retry}>
                  Reconnect
                </Button>
              </div>
            </div>
          ) : (
            /* Keyed by run, so no stage selection outlives the run it was made in. */
            <AgentPipelineSection
              key={streamRunId}
              runId={streamRunId}
              events={events}
              run={result}
              stopRequested={stopRequested}
              streamStatus={streamStatus}
              invalidFrameCount={
                invalidFrames.unknown_event +
                invalidFrames.invalid_json +
                invalidFrames.invalid_schema
              }
              showStop={running}
              onStop={() => {
                setStopError(null);
                setStopRequestedFor(streamRunId);
                cancelMutation.mutate(streamRunId);
              }}
              stopError={stopError}
            />
          )}
          {result !== undefined && (
            <StableAgentRunsPanel runStatus={result.status} agentRuns={result.agent_runs} />
          )}
          {streamStatus === "recovering" && !isComplete && loadError === null && (
            <p role="status" className="text-sm text-muted-foreground">
              Live progress was interrupted. Checking the analysis status…
            </p>
          )}
          {streamStatus === "failed" && !isComplete && loadError === null && (
            <div role="alert" className="space-y-2 text-sm text-destructive">
              <p>
                Live progress for this analysis is unavailable, and it has not reported a final
                state yet. It may still be running.
              </p>
              <Button type="button" variant="outline" size="sm" onClick={retry}>
                Reconnect
              </Button>
            </div>
          )}
        </section>
      )}

      {/* The detail is the run's result; its own loading and failure are states too (FE-007). */}
      {runConfirmed && streamIsComplete && detail.isPending && (
        <div aria-busy="true" className="space-y-2">
          <p className="text-sm text-muted-foreground">Loading the result…</p>
          <div className="h-24 animate-pulse rounded-md bg-muted" />
        </div>
      )}
      {detail.isError && !runNotFound && (
        <div role="alert" className="space-y-2 text-sm text-destructive">
          <p>The page could not load this analysis&apos;s result. {detail.error.message}</p>
          <div className="flex flex-wrap gap-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={detail.isFetching}
              onClick={() => {
                void detail.refetch();
              }}
            >
              {detail.isFetching ? "Loading…" : "Load the result again"}
            </Button>
            {detailReference !== null && (
              <CopyReferenceButton value={detailReference} label="Copy reference" />
            )}
          </div>
        </div>
      )}

      {outcome === "failed" && (
        <p role="alert" className="text-sm text-destructive">
          {result?.error_message ?? "The analysis failed."}
        </p>
      )}

      {outcome === "cancelled" && (
        <p className="text-sm text-muted-foreground">Analysis cancelled.</p>
      )}

      {outcome === "completed_without_answer" && (
        <p className="text-sm text-muted-foreground">
          The analysis completed but produced no answer.
        </p>
      )}

      {outcome === "completed_with_failure" && (
        <div role="alert" className="space-y-1 rounded-md border border-destructive/40 p-3 text-sm">
          <p className="font-semibold text-destructive">Completed with a failure</p>
          <p>
            {result?.error_message ?? "A step after the answer failed."} The answer below was
            produced before that failure. It is shown as it was stored, and it cannot be exported as
            a report.
          </p>
        </div>
      )}

      {result !== undefined &&
        result.answer_markdown !== null &&
        (outcome === "completed" || outcome === "completed_with_failure") && (
          <RunResult result={result} markdown={result.answer_markdown} />
        )}
    </div>
  );
}
