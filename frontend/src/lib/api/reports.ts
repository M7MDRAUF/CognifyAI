import { requestJson } from "@/lib/api/client";
import { type Report, type ReportPage, reportPageSchema, reportSchema } from "@/lib/types/report";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api/v1";

export function generateReport(queryRunId: string): Promise<Report> {
  return requestJson(`/queries/${queryRunId}/reports`, reportSchema, { method: "POST" });
}

export interface ReportsRequest {
  /** The previous page's `next_cursor`; omitted (or null) for the first page. */
  cursor?: string | null;
  limit?: number;
}

/**
 * One page of the project's generated reports, newest first, without their Markdown (CT-01). Each
 * option is sent only when given, so the server's defaults apply otherwise.
 */
export function listReports(projectId: string, request: ReportsRequest = {}): Promise<ReportPage> {
  const search = new URLSearchParams();
  if (request.limit !== undefined) {
    search.set("limit", String(request.limit));
  }
  if (request.cursor !== undefined && request.cursor !== null) {
    search.set("cursor", request.cursor);
  }
  const query = search.toString();
  return requestJson(
    `/projects/${projectId}/reports${query === "" ? "" : `?${query}`}`,
    reportPageSchema,
  );
}

/** One report with its Markdown, for the reports page's preview (SD-04). */
export function getReport(reportId: string): Promise<Report> {
  return requestJson(`/reports/${reportId}`, reportSchema);
}

export function reportDownloadUrl(reportId: string): string {
  return `${API_BASE_URL}/reports/${reportId}/download.md`;
}
