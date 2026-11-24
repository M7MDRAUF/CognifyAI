"use client";

import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { Markdown } from "@/components/ui/markdown";
import { useEffect, useId, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ApiError } from "@/lib/api/client";
import { generateReport, getReport, listReports, reportDownloadUrl } from "@/lib/api/reports";
import { useRunSummaries } from "@/lib/hooks/use-run-summaries";
import { useSubmitLock } from "@/lib/hooks/use-submit-lock";
import { LocalTime } from "@/components/ui/local-time";

// Both lists read bounded server pages, newest first (SE-001, SE-003): the completed runs come from
// the summaries endpoint filtered by the server, the reports from the reports list, which never
// carries a report's Markdown. "Show more" asks the server for the next page (FE-042).
const COMPLETED_PAGE_SIZE = 20;
const REPORTS_PAGE_SIZE = 20;

/** The paging state both lists share (a `useInfiniteQuery` result). */
interface PagedList {
  hasNextPage: boolean;
  isFetchingNextPage: boolean;
  isFetchNextPageError: boolean;
  error: Error | null;
  fetchNextPage: () => Promise<unknown>;
}

/**
 * "Showing N of M …" with "Show more", or the failed next page with "Try again"; the loaded rows
 * stay either way. "Show more" is disabled while its page loads, which can move focus away, so
 * focus is handed back once the page is in: to "Show more", or, when that page was the last and the
 * paging row is gone, to the last loaded item's control (RE2E-DEF-036), never to the page body.
 */
function PagingFooter({
  list,
  listRef,
  shown,
  total,
  noun,
  failure,
}: {
  list: PagedList;
  listRef: React.RefObject<HTMLUListElement | null>;
  shown: number;
  total: number;
  noun: string;
  failure: string;
}) {
  const rowRef = useRef<HTMLDivElement>(null);
  const refocus = useRef(false);
  const { isFetchingNextPage } = list;
  // Keyed on the row count too: a fast page can turn isFetchingNextPage on and off in one render.
  useEffect(() => {
    if (!isFetchingNextPage && refocus.current) {
      refocus.current = false;
      const target =
        rowRef.current?.querySelector("button") ??
        listRef.current?.querySelector<HTMLElement>("li:last-child button, li:last-child a[href]");
      target?.focus();
    }
  }, [isFetchingNextPage, shown, listRef]);

  if (!list.hasNextPage) {
    return null;
  }
  return (
    <div className="mt-3 space-y-2 text-xs text-muted-foreground">
      <div ref={rowRef} className="flex items-center justify-between gap-2">
        <p>
          Showing {shown} of {total} {noun}.
        </p>
        {!list.isFetchNextPageError && (
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={isFetchingNextPage}
            onClick={() => {
              refocus.current = true;
              void list.fetchNextPage();
            }}
          >
            Show more
          </Button>
        )}
      </div>
      {list.isFetchNextPageError && (
        <div role="alert" className="space-y-2 text-sm text-destructive">
          <p>
            {failure} {list.error?.message}
          </p>
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={isFetchingNextPage}
            onClick={() => {
              void list.fetchNextPage();
            }}
          >
            {isFetchingNextPage ? "Loading…" : "Try again"}
          </Button>
        </div>
      )}
    </div>
  );
}

export function ReportsView({ projectId }: { projectId: string }) {
  const queryClient = useQueryClient();
  const [selectedReportId, setSelectedReportId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const submitLock = useSubmitLock();
  const ids = useId();
  const reportsTitleId = `${ids}-generated-reports`;
  const previewTitleId = `${ids}-preview`;
  const completedListRef = useRef<HTMLUListElement>(null);
  const reportsListRef = useRef<HTMLUListElement>(null);
  // RE2E-DEF-014: Open and Generate hand focus to the preview's heading once it shows that report,
  // so a keyboard or screen-reader user lands on what opened instead of staying on the button.
  const focusPreview = useRef(false);
  const [previewRequest, setPreviewRequest] = useState(0);
  const showPreview = (reportId: string) => {
    focusPreview.current = true;
    setPreviewRequest((n) => n + 1);
    setSelectedReportId(reportId);
  };

  const completed = useRunSummaries(projectId, "completed", COMPLETED_PAGE_SIZE);
  const reports = useInfiniteQuery({
    queryKey: ["reports", projectId],
    queryFn: ({ pageParam }) =>
      listReports(projectId, { cursor: pageParam, limit: REPORTS_PAGE_SIZE }),
    initialPageParam: null as string | null,
    getNextPageParam: (page) => page.next_cursor,
  });
  // SD-04: a report's Markdown is read only to preview it, under its own key. A report never
  // changes once generated, so what was read (or returned by generating it) is never refetched.
  const preview = useQuery({
    queryKey: ["report", selectedReportId],
    queryFn: () => getReport(selectedReportId ?? ""),
    enabled: selectedReportId !== null,
    staleTime: Infinity,
  });
  const shownReportId = preview.data?.id;
  useEffect(() => {
    if (focusPreview.current && shownReportId !== undefined && shownReportId === selectedReportId) {
      focusPreview.current = false;
      document.getElementById(previewTitleId)?.focus();
    }
  }, [previewRequest, shownReportId, selectedReportId, previewTitleId]);

  const mutation = useMutation({
    mutationFn: generateReport,
    onSuccess: (generated) => {
      setError(null);
      // The response is the whole report: it previews without a second request, and the list
      // refetches so the report is found again after a reload (SE-001).
      queryClient.setQueryData(["report", generated.id], generated);
      showPreview(generated.id);
      void queryClient.invalidateQueries({ queryKey: ["reports", projectId] });
    },
    onError: (cause: unknown) => {
      setError(cause instanceof ApiError ? cause.message : "Could not generate the report.");
    },
    onSettled: submitLock.release,
  });

  // Loading, failure and "nothing completed yet" are three different facts (FE-010, Probe P-19).
  function completedQueries() {
    if (completed.isPending) {
      return (
        <div aria-busy="true" className="space-y-2">
          <p className="text-sm text-muted-foreground">Loading completed queries…</p>
          <div className="h-10 animate-pulse rounded-md bg-muted" />
        </div>
      );
    }
    if (completed.data === undefined || (completed.isError && !completed.isFetchNextPageError)) {
      return (
        <div role="alert" className="space-y-2 text-sm text-destructive">
          <p>The completed queries could not be loaded. {completed.error.message}</p>
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={completed.isFetching}
            onClick={() => {
              void completed.refetch();
            }}
          >
            {completed.isFetching ? "Loading…" : "Try again"}
          </Button>
        </div>
      );
    }
    const pages = completed.data.pages;
    const runs = pages.flatMap((page) => page.items);
    if (runs.length === 0) {
      return (
        <p className="text-sm text-muted-foreground">
          No completed queries yet. Go to the{" "}
          <Link
            href={`/projects/${projectId}/chat`}
            className="font-semibold underline hover:no-underline"
          >
            Chat tab
          </Link>{" "}
          to run your first analysis, then return here to generate a downloadable report.
        </p>
      );
    }
    return (
      <>
        <ul ref={completedListRef} className="space-y-2">
          {runs.map((run) => {
            // RE2E-DEF-015: the pressed button says the report is being made.
            const generating = mutation.isPending && mutation.variables === run.id;
            return (
              <li key={run.id} className="flex items-center justify-between gap-4 text-sm">
                <span>
                  {run.question}{" "}
                  <span className="text-muted-foreground">
                    (<LocalTime iso={run.created_at} />)
                  </span>
                </span>
                {/* RE2E-DEF-007: named for its question; the visible text stays. */}
                <Button
                  size="sm"
                  aria-label={
                    generating ? "Generating report…" : `Generate report for ${run.question}`
                  }
                  disabled={mutation.isPending}
                  onClick={() => {
                    if (!submitLock.acquire()) {
                      return;
                    }
                    mutation.mutate(run.id);
                  }}
                >
                  {generating ? "Generating report…" : "Generate report"}
                </Button>
              </li>
            );
          })}
        </ul>
        <PagingFooter
          list={completed}
          listRef={completedListRef}
          shown={runs.length}
          total={pages[pages.length - 1]?.total ?? runs.length}
          noun="completed queries"
          failure="More completed queries could not be loaded."
        />
      </>
    );
  }

  function generatedReports() {
    if (reports.isPending) {
      return (
        <p aria-busy="true" className="text-sm text-muted-foreground">
          Loading generated reports…
        </p>
      );
    }
    if (reports.data === undefined || (reports.isError && !reports.isFetchNextPageError)) {
      return (
        <div role="alert" className="space-y-2 text-sm text-destructive">
          <p>Generated reports could not be loaded. {reports.error.message}</p>
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={reports.isFetching}
            onClick={() => {
              void reports.refetch();
            }}
          >
            {reports.isFetching ? "Loading…" : "Try again"}
          </Button>
        </div>
      );
    }
    const pages = reports.data.pages;
    const items = pages.flatMap((page) => page.items);
    if (items.length === 0) {
      return (
        <p className="text-sm text-muted-foreground">
          No reports yet. Generate one from a completed query above.
        </p>
      );
    }
    return (
      <>
        <ul ref={reportsListRef} aria-labelledby={reportsTitleId} className="space-y-2">
          {items.map((item) => {
            const titleId = `${ids}-report-${item.id}`;
            return (
              <li
                key={item.id}
                className="flex flex-wrap items-center justify-between gap-2 text-sm"
              >
                <span>
                  <span id={titleId}>{item.title}</span>{" "}
                  <span className="text-muted-foreground">
                    (<LocalTime iso={item.created_at} />)
                  </span>
                </span>
                <span className="flex items-center gap-2">
                  {/* RE2E-DEF-007: named for its report; the visible text stays. */}
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    aria-label={`Open ${item.title}`}
                    onClick={() => {
                      showPreview(item.id);
                    }}
                  >
                    Open
                  </Button>
                  <a
                    className="rounded-md text-sm font-medium underline hover:no-underline focus-visible:outline-2 focus-visible:outline-offset-2"
                    href={reportDownloadUrl(item.id)}
                    download
                    aria-describedby={titleId}
                  >
                    Download Markdown
                  </a>
                </span>
              </li>
            );
          })}
        </ul>
        <PagingFooter
          list={reports}
          listRef={reportsListRef}
          shown={items.length}
          total={pages[pages.length - 1]?.total ?? items.length}
          noun="reports"
          failure="More reports could not be loaded."
        />
      </>
    );
  }

  function reportPreview() {
    if (selectedReportId === null) {
      return null;
    }
    const report = preview.data;
    if (report === undefined && !preview.isError) {
      return (
        <p aria-busy="true" className="text-sm text-muted-foreground">
          Loading the report…
        </p>
      );
    }
    if (report === undefined) {
      return (
        <div role="alert" className="space-y-2 text-sm text-destructive">
          <p>The report could not be loaded. {preview.error.message}</p>
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={preview.isFetching}
            onClick={() => {
              void preview.refetch();
            }}
          >
            {preview.isFetching ? "Loading…" : "Try again"}
          </Button>
        </div>
      );
    }
    return (
      <section aria-labelledby={previewTitleId}>
        <Card>
          <CardHeader>
            {/* Focused by script after Open or Generate (RE2E-DEF-014); not a control, so no ring. */}
            <CardTitle id={previewTitleId} tabIndex={-1} className="focus:outline-none">
              {report.title}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <a
              className="inline-block rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 focus-visible:outline-2 focus-visible:outline-offset-2"
              href={reportDownloadUrl(report.id)}
              download
            >
              Download Markdown
            </a>
            {/* The report's own "#" headings sit below this card's title (FE-033). */}
            <Markdown headingOffset={3}>{report.markdown_content}</Markdown>
          </CardContent>
        </Card>
      </section>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Reports</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Generate a Markdown decision report for any completed query. Reports include an executive
          summary, recommendation, evidence table, risk matrix, evaluation scores, and a citation
          appendix.
        </p>
        {/* SD-04 (SE-001): the project's reports are listed below, so a reload loses none. */}
        <p className="mt-1 text-sm text-muted-foreground">
          Reports you generate are listed under Generated reports, newest first. Open one to preview
          it, or download its Markdown.
        </p>
      </div>
      {error !== null && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
      <h2 className="sr-only">Generate a report</h2>
      <Card>
        <CardHeader>
          <CardTitle>Completed queries</CardTitle>
        </CardHeader>
        <CardContent>{completedQueries()}</CardContent>
      </Card>

      <h2 className="sr-only">Past reports</h2>
      <Card>
        <CardHeader>
          <CardTitle id={reportsTitleId}>Generated reports</CardTitle>
        </CardHeader>
        <CardContent>{generatedReports()}</CardContent>
      </Card>

      {reportPreview()}
    </div>
  );
}
