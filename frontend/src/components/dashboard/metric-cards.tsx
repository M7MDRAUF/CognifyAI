import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { EvaluationSummary } from "@/lib/types/evaluation";
import { EVALUATOR_SCORE_CAPTION, formatLatency, formatScore } from "@/lib/utils/format";

interface Metric {
  label: string;
  value: string;
  /** A second line under the value. */
  detail?: string;
  /** An average of evaluator scores, which carries the OD-02 caption (FE-006). */
  evaluator?: true;
}

/**
 * FR-004: the detail line accounts for every run in the total. Each status is named only when it
 * has a run, so a zero never adds noise and a cancelled or running run never goes unexplained.
 */
function statusDetail(summary: EvaluationSummary): string | undefined {
  const statuses: [number, string][] = [
    [summary.completed_count, "completed"],
    [summary.failed_count, "failed"],
    [summary.cancelled_count, "cancelled"],
    [summary.running_count, "running"],
  ];
  const named = statuses
    .filter(([count]) => count > 0)
    .map(([count, status]) => `${String(count)} ${status}`);
  return named.length > 0 ? named.join(" · ") : undefined;
}

function metricList(summary: EvaluationSummary): Metric[] {
  const tokens =
    summary.average_token_usage === null
      ? "—"
      : Math.round(summary.average_token_usage).toLocaleString("en-US");
  return [
    {
      label: "Queries",
      value: String(summary.query_count),
      detail: statusDetail(summary),
    },
    {
      label: "Avg. relevance",
      value: formatScore(summary.average_answer_relevance),
      evaluator: true,
    },
    {
      label: "Avg. faithfulness",
      value: formatScore(summary.average_faithfulness),
      evaluator: true,
    },
    {
      label: "Avg. citation accuracy",
      value: formatScore(summary.average_citation_accuracy),
      evaluator: true,
    },
    {
      label: "Avg. context sufficiency",
      value: formatScore(summary.average_context_sufficiency),
      evaluator: true,
    },
    { label: "Avg. latency", value: formatLatency(summary.average_latency_ms) },
    { label: "Avg. tokens", value: tokens },
    { label: "Retrieval hits", value: String(summary.retrieval_hit_count) },
    { label: "Agent failures", value: String(summary.agent_failure_count) },
  ];
}

/**
 * N-006: every average covers only the runs that completed with an answer and were evaluated.
 * The summary carries one quality point per such run, so its length is the denominator.
 */
function denominatorSentence(summary: EvaluationSummary): string {
  const eligible = summary.quality_over_time.length;
  const covered =
    eligible === 1
      ? "the 1 analysis that completed with an answer and was evaluated"
      : `the ${String(eligible)} analyses that completed with an answer and were evaluated`;
  return (
    `Averages cover ${covered}, out of ${String(summary.query_count)} in total. ` +
    "Failed, cancelled and unevaluated analyses are not averaged."
  );
}

export function MetricCards({ summary }: { summary: EvaluationSummary }) {
  return (
    <div className="space-y-3">
      <p className="text-sm text-muted-foreground">{denominatorSentence(summary)}</p>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {metricList(summary).map((metric) => (
          <Card key={metric.label}>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-muted-foreground">{metric.label}</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-semibold">{metric.value}</p>
              {metric.detail !== undefined && (
                <p className="mt-1 text-xs text-muted-foreground">{metric.detail}</p>
              )}
              {metric.evaluator && (
                <p className="mt-1 text-xs text-muted-foreground">{EVALUATOR_SCORE_CAPTION}</p>
              )}
            </CardContent>
          </Card>
        ))}
        <Card className="sm:col-span-2 lg:col-span-3">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Hallucination risk</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground">
              Evaluated answers by hallucination-risk score.
            </p>
            <ul className="mt-2 flex flex-wrap gap-x-6 gap-y-1 text-sm">
              {summary.hallucination_risk_distribution.map((bucket) => (
                <li key={bucket.label}>
                  <span className="text-muted-foreground">{bucket.label}:</span>{" "}
                  <span className="font-medium">{bucket.count}</span>
                </li>
              ))}
            </ul>
            <p className="mt-2 text-xs text-muted-foreground">{EVALUATOR_SCORE_CAPTION}</p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
