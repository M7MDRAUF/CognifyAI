import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { Citation, Evaluation } from "@/lib/types/query";
import { citationLabels, concealIdentifiers } from "@/lib/utils/conceal-identifiers";
import { EVALUATOR_SCORE_CAPTION, formatScore } from "@/lib/utils/format";

const METRICS: { key: keyof Omit<Evaluation, "explanation">; label: string }[] = [
  { key: "answer_relevance", label: "Answer relevance" },
  { key: "faithfulness", label: "Faithfulness" },
  { key: "citation_accuracy", label: "Citation accuracy" },
  { key: "context_sufficiency", label: "Context sufficiency" },
  { key: "hallucination_risk", label: "Hallucination risk" },
];

interface EvaluationPanelProps {
  evaluation: Evaluation | null;
  /** The run's citations: the explanation's raw chunk ids are relabelled with their keys (L-2). */
  citations: readonly Citation[];
}

export function EvaluationPanel({ evaluation, citations }: EvaluationPanelProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Evaluation</CardTitle>
        {evaluation !== null && <CardDescription>{EVALUATOR_SCORE_CAPTION}</CardDescription>}
      </CardHeader>
      <CardContent>
        {evaluation === null ? (
          <p className="text-sm text-muted-foreground">Evaluation is not available for this run.</p>
        ) : (
          <div className="space-y-3">
            <dl className="grid grid-cols-2 gap-2 text-sm">
              {METRICS.map((metric) => (
                <div key={metric.key} className="flex justify-between gap-2">
                  <dt className="text-muted-foreground">{metric.label}</dt>
                  <dd className="font-medium">{formatScore(evaluation[metric.key])}</dd>
                </div>
              ))}
            </dl>
            <p className="text-sm text-muted-foreground">
              {concealIdentifiers(evaluation.explanation, citationLabels(citations))}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
