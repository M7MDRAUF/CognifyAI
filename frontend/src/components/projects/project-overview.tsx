"use client";

import { useQuery } from "@tanstack/react-query";
import type { ReactNode } from "react";

import ProjectNotFound from "@/app/projects/[projectId]/not-found";
import { HumanReviewBanner } from "@/components/pharma/human-review-banner";
import { ProfileBadge } from "@/components/pharma/profile-badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ApiError } from "@/lib/api/client";
import { getProject } from "@/lib/api/projects";
import { LocalTime } from "@/components/ui/local-time";
import { profileOf } from "@/lib/types/pharma";

/** Retries after the first failure: at most three attempts in all, then the error state (FE-008). */
const PROJECT_LOOKUP_RETRIES = 2;

/**
 * Bounded retry. A 4xx is never retried (the project is missing or the id is malformed, and asking
 * again cannot help); anything else is retried at most `PROJECT_LOOKUP_RETRIES` times with
 * TanStack's backoff (about 1 s, then 2 s), so an outage reaches the error state instead of an
 * endless skeleton.
 */
function retryProjectLookup(failureCount: number, error: Error): boolean {
  if (error instanceof ApiError && error.status >= 400 && error.status < 500) {
    return false;
  }
  return failureCount < PROJECT_LOOKUP_RETRIES;
}

function useProject(projectId: string) {
  return useQuery({
    queryKey: ["project", projectId],
    queryFn: () => getProject(projectId),
    retry: retryProjectLookup,
    // The guard and the overview observe this one query. Without this, the overview mounting on
    // an errored query would start a fresh retry cycle, the guard would hide it again while that
    // cycle runs, and the bound above would never end in an error state. "Try again" refetches.
    retryOnMount: false,
  });
}

/**
 * Resolves the project once for every project route (FE-009). The layout has already sent a
 * malformed id to the not-found page on the server; this guard handles a well-formed id the
 * backend does not know, which only the API can answer. No page, and so no upload or ask control,
 * renders until the project is confirmed. A failure that is not a 404 renders the page, whose own
 * requests then show their errors. The query is shared with `ProjectOverview` (same key).
 */
export function ProjectGuard({ projectId, children }: { projectId: string; children: ReactNode }) {
  const { isPending, error } = useProject(projectId);
  if (error instanceof ApiError && (error.status === 404 || error.status === 422)) {
    return <ProjectNotFound />;
  }
  if (isPending) {
    return (
      <p className="text-sm text-muted-foreground" aria-busy="true">
        Loading project…
      </p>
    );
  }
  return <>{children}</>;
}

export function ProjectOverview({ projectId }: { projectId: string }) {
  const { data, isLoading, isError, error, refetch, isFetching } = useProject(projectId);

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="animate-pulse space-y-2">
          <div className="h-8 w-48 rounded bg-muted" />
          <div className="h-4 w-64 rounded bg-muted" />
        </div>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="animate-pulse rounded-lg border bg-muted h-24" />
          ))}
        </div>
      </div>
    );
  }
  if (isError || data === undefined) {
    // A missing project is `ProjectGuard`'s to render; this alert is for every other failure.
    return (
      <div role="alert" className="space-y-3">
        <p className="text-destructive">
          {error instanceof Error ? error.message : "Failed to load project."}
        </p>
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={isFetching}
          onClick={() => {
            void refetch();
          }}
        >
          {isFetching ? "Retrying…" : "Try again"}
        </Button>
      </div>
    );
  }

  const metrics: { label: string; value: ReactNode }[] = [
    { label: "Documents", value: String(data.stats.document_count) },
    { label: "Chunks", value: String(data.stats.chunk_count) },
    { label: "Indexed vectors", value: String(data.stats.vector_count) },
    { label: "Last query", value: <LocalTime iso={data.stats.last_query_at} /> },
  ];

  const pharmaceutical = profileOf(data) === "pharma_manufacturing";

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">{data.name}</h1>
        {data.objective !== "" && (
          <p className="text-muted-foreground">Objective: {data.objective}</p>
        )}
        {pharmaceutical && (
          <div className="mt-2">
            <ProfileBadge profile="pharma_manufacturing" />
          </div>
        )}
      </div>
      {pharmaceutical && <HumanReviewBanner />}
      <h2 className="sr-only">Project summary</h2>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {metrics.map((metric) => (
          <Card key={metric.label}>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-muted-foreground">{metric.label}</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-semibold">{metric.value}</p>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
