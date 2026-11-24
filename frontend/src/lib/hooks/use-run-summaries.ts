"use client";

import { useInfiniteQuery } from "@tanstack/react-query";

import { listQuerySummaries } from "@/lib/api/queries";

/** The page size every run-summaries list asks the server for (CT-02's default). */
export const RUN_SUMMARIES_PAGE_SIZE = 20;

/** `"all"` lists every run; `"completed"` asks the server for completed runs only. */
export type RunSummariesStatus = "all" | "completed";

/** Every run-summaries key of a project starts with this: invalidating it refreshes both lists. */
export function runSummariesKey(projectId: string): readonly ["query-summaries", string] {
  return ["query-summaries", projectId];
}

/**
 * The project's run summaries as server pages, newest first (CT-09). This is the one definition of
 * the key and the fetcher, so the history panel and the chat view share one infinite query. Every
 * caller of the same key must ask for the same `limit`, because the key does not include it.
 */
export function useRunSummaries(
  projectId: string,
  status: RunSummariesStatus,
  limit: number = RUN_SUMMARIES_PAGE_SIZE,
) {
  return useInfiniteQuery({
    queryKey: [...runSummariesKey(projectId), status],
    queryFn: ({ pageParam }) =>
      listQuerySummaries(projectId, {
        cursor: pageParam,
        status: status === "all" ? undefined : status,
        limit,
      }),
    initialPageParam: null as string | null,
    getNextPageParam: (page) => page.next_cursor,
  });
}
