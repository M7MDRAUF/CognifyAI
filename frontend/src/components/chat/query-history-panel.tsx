"use client";

import { useEffect, useRef } from "react";

import { Button } from "@/components/ui/button";
import { useRunSummaries } from "@/lib/hooks/use-run-summaries";
import { providerLabel } from "@/lib/types/llm";
import { LocalTime } from "@/components/ui/local-time";

interface QueryHistoryPanelProps {
  projectId: string;
  currentQueryRunId: string | null;
  onSelectQuery: (queryRunId: string) => void;
}

// The history reads bounded server pages of summary rows, newest first (SE-003, CT-02): each page
// holds only the fields a row shows, and "Show more" asks the server for the next one (FE-042).
const HISTORY_PAGE_SIZE = 20;

export function QueryHistoryPanel({
  projectId,
  currentQueryRunId,
  onSelectQuery,
}: QueryHistoryPanelProps) {
  const history = useRunSummaries(projectId, "all", HISTORY_PAGE_SIZE);
  const pagingRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLUListElement>(null);
  // Set when "Show more" is pressed: the button is disabled while its page loads, which can move
  // focus away, so it is handed back once the page is in: to "Show more", or, when that page was
  // the last and the paging row is gone, to the last loaded row (RE2E-DEF-036), never the body.
  const refocusShowMore = useRef(false);
  const { isFetchingNextPage } = history;
  // Keyed on the row count too: a fast page can turn isFetchingNextPage on and off in one render.
  const loadedRows = history.data?.pages.reduce((count, page) => count + page.items.length, 0);
  useEffect(() => {
    if (!isFetchingNextPage && refocusShowMore.current) {
      refocusShowMore.current = false;
      const target =
        pagingRef.current?.querySelector("button") ??
        listRef.current?.querySelector<HTMLElement>("li:last-child button");
      target?.focus();
    }
  }, [isFetchingNextPage, loadedRows]);

  const heading = <h2 className="text-sm font-medium text-muted-foreground">Previous queries</h2>;

  // Loading, failure and an empty history are each told apart; a failure never hides the panel
  // (FE-028, Probe P-30 (b)).
  if (history.isPending) {
    return (
      <section aria-label="Query history" aria-busy="true" className="space-y-2">
        {heading}
        <p className="text-sm text-muted-foreground">Loading previous queries…</p>
      </section>
    );
  }

  // A failed first load, or a failed refresh of the loaded pages (as the single-list query showed
  // it). A failed next page is told apart: it keeps the loaded rows (below).
  if (history.data === undefined || (history.isError && !history.isFetchNextPageError)) {
    return (
      <section aria-label="Query history" className="space-y-2">
        {heading}
        <div role="alert" className="space-y-2 text-sm text-destructive">
          <p>Previous queries could not be loaded. {history.error.message}</p>
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={history.isFetching}
            onClick={() => {
              void history.refetch();
            }}
          >
            {history.isFetching ? "Loading…" : "Try again"}
          </Button>
        </div>
      </section>
    );
  }

  // The run being viewed stays in the list, marked as the current one, so the history reads as the
  // project's complete record rather than losing an entry whenever it is opened (FE-030).
  const pages = history.data.pages;
  const pastQueries = pages.flatMap((page) => page.items);
  // The newest page's count: runs started while paging are counted, never listed twice (keyset).
  const total = pages[pages.length - 1]?.total ?? pastQueries.length;

  if (pastQueries.length === 0) {
    return (
      <section aria-label="Query history" className="space-y-2">
        {heading}
        <p className="text-sm text-muted-foreground">No previous queries yet.</p>
      </section>
    );
  }

  return (
    <section aria-label="Query history" className="space-y-2">
      {heading}
      <ul ref={listRef} className="space-y-1">
        {pastQueries.map((query) => (
          <li key={query.id}>
            <button
              type="button"
              aria-current={query.id === currentQueryRunId ? "true" : undefined}
              onClick={() => {
                onSelectQuery(query.id);
              }}
              className="flex w-full items-start justify-between gap-2 rounded-md px-3 py-2 text-left text-sm hover:bg-muted focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:rounded-md aria-[current=true]:bg-muted aria-[current=true]:font-medium"
            >
              <span className="min-w-0 flex-1">
                <span className="block truncate">{query.question}</span>
                <span className="text-xs text-muted-foreground">
                  {query.status} · <LocalTime iso={query.created_at} />
                </span>
                <span className="block text-xs text-muted-foreground">
                  {query.resolved_provider !== null &&
                  query.resolved_provider !== undefined &&
                  query.resolved_model_name !== null &&
                  query.resolved_model_name !== undefined
                    ? `${providerLabel(query.resolved_provider)} · ${query.resolved_model_name}`
                    : query.answer_model}
                </span>
                {query.status === "failed" && query.error_message !== null && (
                  // FR-014: `text-destructive` measured 4.37:1 on the selected and hover
                  // background (`bg-muted`); `text-red-700` holds 4.5:1 there and on white.
                  <span className="block text-xs text-red-700 truncate">{query.error_message}</span>
                )}
              </span>
              <span className="mt-0.5 shrink-0 text-muted-foreground" aria-hidden>
                ›
              </span>
            </button>
          </li>
        ))}
      </ul>
      {history.hasNextPage && (
        <div className="space-y-2 px-3 text-xs text-muted-foreground">
          <div ref={pagingRef} className="flex items-center justify-between gap-2">
            <p>
              Showing {pastQueries.length} of {total}.
            </p>
            {!history.isFetchNextPageError && (
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={isFetchingNextPage}
                onClick={() => {
                  refocusShowMore.current = true;
                  void history.fetchNextPage();
                }}
              >
                Show more
              </Button>
            )}
          </div>
          {history.isFetchNextPageError && (
            <div role="alert" className="space-y-2 text-sm text-destructive">
              <p>More queries could not be loaded. {history.error.message}</p>
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={isFetchingNextPage}
                onClick={() => {
                  void history.fetchNextPage();
                }}
              >
                {isFetchingNextPage ? "Loading…" : "Try again"}
              </Button>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
