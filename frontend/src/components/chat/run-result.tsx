import { memo } from "react";

import { AnswerView } from "@/components/chat/answer-view";
import { CitationsPanel } from "@/components/chat/citations-panel";
import { EvaluationPanel } from "@/components/chat/evaluation-panel";
import { ContradictionsPanel, RisksPanel } from "@/components/chat/risks-panel";
import { PharmaResultView } from "@/components/pharma/pharma-result-view";
import type { QueryRunDetail } from "@/lib/types/query";

interface RunResultProps {
  result: QueryRunDetail;
  /** The result's answer, present whenever this section is shown. */
  markdown: string;
}

/**
 * A finished run's answer, risks, contradictory evidence, citations and evaluation. Memoised on
 * the run's detail, so the stream events a reload or a history pick replays, which re-render the
 * view, do not re-render the Markdown or the panels (UDF-01); the citation keys are derived once
 * per result.
 */
export const RunResult = memo(function RunResult({ result, markdown }: RunResultProps) {
  // A pharmaceutical run is shown from its policy-checked record (REQ-UX-05); its answer
  // Markdown is that record rendered, so the structured view replaces it. Every other run is
  // shown exactly as before.
  if (result.pharma != null) {
    return <PharmaResultView result={result} pharma={result.pharma} />;
  }
  const citationKeys = result.citations.map((citation) => citation.citation_key);
  return (
    <section className="space-y-4">
      <AnswerView
        markdown={markdown}
        uncertainty={result.uncertainty}
        nextSteps={result.next_steps}
        citationKeys={citationKeys}
        unresolvedCitationMarkers={result.unresolved_citation_markers}
      />
      {/* FR-010: the risks and contradictory evidence the run recorded, beside its answer. */}
      <RisksPanel risks={result.risks} citationKeys={citationKeys} />
      <ContradictionsPanel items={result.contradictory_evidence} citationKeys={citationKeys} />
      <div className="grid gap-4 md:grid-cols-2">
        <CitationsPanel citations={result.citations} />
        <EvaluationPanel evaluation={result.evaluation} citations={result.citations} />
      </div>
    </section>
  );
});
