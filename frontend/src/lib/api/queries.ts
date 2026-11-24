import { jsonInit, requestJson } from "@/lib/api/client";
import {
  type QueryCreateInput,
  type QueryRun,
  type QueryRunCancel,
  type QueryRunDetail,
  type QueryRunStatus,
  type QueryRunSummaryPage,
  queryRunCancelSchema,
  queryRunDetailSchema,
  queryRunSchema,
  queryRunSummaryPageSchema,
} from "@/lib/types/query";

/**
 * Starts a run. `pharma`, a pharmaceutical project's inquiry details (REQ-API-03), is sent only
 * when the caller sets it: JSON leaves an undefined key out, so a General question is the request
 * it always was.
 */
export function submitQuery(projectId: string, input: QueryCreateInput): Promise<QueryRun> {
  return requestJson(`/projects/${projectId}/queries`, queryRunSchema, jsonInit("POST", input));
}

export interface QuerySummariesRequest {
  /** The previous page's `next_cursor`; omitted (or null) for the first page. */
  cursor?: string | null;
  status?: QueryRunStatus;
  limit?: number;
}

/**
 * One page of the project's run summaries, newest first (CT-02). Each option is sent only when
 * given, so the server's defaults apply otherwise.
 */
export function listQuerySummaries(
  projectId: string,
  request: QuerySummariesRequest = {},
): Promise<QueryRunSummaryPage> {
  const search = new URLSearchParams();
  if (request.limit !== undefined) {
    search.set("limit", String(request.limit));
  }
  if (request.cursor !== undefined && request.cursor !== null) {
    search.set("cursor", request.cursor);
  }
  if (request.status !== undefined) {
    search.set("status", request.status);
  }
  const query = search.toString();
  return requestJson(
    `/projects/${projectId}/queries/summaries${query === "" ? "" : `?${query}`}`,
    queryRunSummaryPageSchema,
  );
}

export function getQueryDetail(queryRunId: string): Promise<QueryRunDetail> {
  return requestJson(`/queries/${queryRunId}`, queryRunDetailSchema);
}

/**
 * Requests cancellation of a running query. The backend answers 202 with the run row read
 * before the cancellation finishes unwinding, so the response may still say `status: "running"`;
 * `cancel_requested: true` says the cancel is pending (RMD5-T03). Treat it as an acknowledgement,
 * not the new state: the authoritative transition arrives over SSE as `workflow_cancelled`.
 */
export function cancelQuery(queryRunId: string): Promise<QueryRunCancel> {
  return requestJson(`/queries/${queryRunId}/cancel`, queryRunCancelSchema, { method: "POST" });
}
