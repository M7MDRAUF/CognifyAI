"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";

import { MetricCards } from "@/components/dashboard/metric-cards";
import { QualityLineChart } from "@/components/dashboard/quality-line-chart";
import { RecentQueriesTable } from "@/components/dashboard/recent-queries-table";
import { RiskDonutChart } from "@/components/dashboard/risk-donut-chart";
import { StageLatencyBarChart } from "@/components/dashboard/stage-latency-bar-chart";
import { Button } from "@/components/ui/button";
import { getEvaluationSummary } from "@/lib/api/evaluations";

export function DashboardView({ projectId }: { projectId: string }) {
  const { data, isLoading, isError, error, isFetching, refetch } = useQuery({
    queryKey: ["evaluation-summary", projectId],
    queryFn: () => getEvaluationSummary(projectId),
  });

  if (isLoading) {
    return <p className="text-muted-foreground">Loading dashboard…</p>;
  }
  // RE2E-DEF-016: the message says to try again, so the page offers the means.
  if (isError || data === undefined) {
    return (
      <div role="alert" className="space-y-3">
        <p className="text-destructive">
          {error instanceof Error ? error.message : "Failed to load the dashboard."}
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

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold tracking-tight">Evaluation dashboard</h1>
      {data.completed_count === 0 && (
        <div className="rounded-md border border-dashed p-6 text-center">
          <p className="text-sm text-muted-foreground">
            No completed queries yet. Go to the{" "}
            <Link
              href={`/projects/${projectId}/chat`}
              className="font-semibold underline hover:no-underline"
            >
              Chat tab
            </Link>{" "}
            to run your first analysis.
          </p>
        </div>
      )}
      <h2 className="sr-only">Summary figures</h2>
      <MetricCards summary={data} />
      <div className="grid gap-4 lg:grid-cols-2">
        <QualityLineChart points={data.quality_over_time} />
        <StageLatencyBarChart stages={data.stage_latencies} />
        <RiskDonutChart categories={data.risk_categories} />
        <RecentQueriesTable queries={data.recent_queries} />
      </div>
    </div>
  );
}
