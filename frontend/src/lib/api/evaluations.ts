import { requestJson } from "@/lib/api/client";
import { type EvaluationSummary, evaluationSummarySchema } from "@/lib/types/evaluation";

export function getEvaluationSummary(projectId: string): Promise<EvaluationSummary> {
  return requestJson(`/projects/${projectId}/evaluations/summary`, evaluationSummarySchema);
}
