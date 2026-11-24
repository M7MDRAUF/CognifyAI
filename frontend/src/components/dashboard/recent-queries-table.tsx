import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { RecentQuery } from "@/lib/types/evaluation";
import { LocalTime } from "@/components/ui/local-time";
import { EVALUATOR_SCORE_CAPTION, formatScore } from "@/lib/utils/format";

const STATUS_LABEL: Record<RecentQuery["status"], string> = {
  running: "Running",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
};

export function RecentQueriesTable({ queries }: { queries: RecentQuery[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Recent queries</CardTitle>
      </CardHeader>
      <CardContent>
        {queries.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No queries yet. Go to the Chat tab to ask your first question.
          </p>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left text-muted-foreground">
                <th className="py-2 pr-4 font-medium">Question</th>
                <th className="py-2 pr-4 font-medium">Status</th>
                <th className="py-2 pr-4 font-medium hidden sm:table-cell">Relevance</th>
                <th className="py-2 font-medium hidden sm:table-cell">When</th>
              </tr>
            </thead>
            <tbody>
              {queries.map((query) => (
                <tr key={query.query_run_id} className="border-b">
                  <td className="py-2 pr-4">
                    <span className="block">{query.question}</span>
                    {query.status === "failed" && query.error_message && (
                      <span className="block text-xs text-destructive">{query.error_message}</span>
                    )}
                  </td>
                  <td className="py-2 pr-4">
                    <span className={query.status === "failed" ? "text-destructive" : undefined}>
                      {STATUS_LABEL[query.status]}
                    </span>
                  </td>
                  <td className="py-2 pr-4 hidden sm:table-cell">
                    {formatScore(query.answer_relevance)}
                  </td>
                  <td className="py-2 hidden sm:table-cell">
                    <LocalTime iso={query.created_at} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {queries.length > 0 && (
          // The relevance column is hidden on narrow screens, and its caption with it.
          <p className="mt-2 hidden text-xs text-muted-foreground sm:block">
            Relevance: {EVALUATOR_SCORE_CAPTION}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
