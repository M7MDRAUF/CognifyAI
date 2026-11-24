import type { ReactNode } from "react";

import { EvaluationPanel } from "@/components/chat/evaluation-panel";
import { AbstentionState } from "@/components/pharma/abstention-state";
import { ConflictComparison } from "@/components/pharma/conflict-comparison";
import { EscalationCallout } from "@/components/pharma/escalation-callout";
import { CitedText, EvidenceCitationCard } from "@/components/pharma/evidence-citation-card";
import { LifecycleBadge } from "@/components/pharma/lifecycle-badge";
import { PolicyNotices } from "@/components/pharma/policy-notices";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { INTENT_LABELS } from "@/lib/pharma/incident-form";
import { excludedVersions } from "@/lib/pharma/versions";
import type { DispositionGate, PharmaDisposition, PharmaRun } from "@/lib/types/pharma";
import type { QueryRunDetail } from "@/lib/types/query";

/** The server's empty-section sentence (`NONE_IDENTIFIED`, `backend/app/pharma/catalogue.py`). */
const NONE_IDENTIFIED = "None identified.";

function words(value: string): string {
  const spaced = value.replace(/_/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

/** One of the nine sections, under the server's own heading (REQ-RSP-01). */
function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="space-y-2">
      <h4 className="text-sm font-semibold">{title}</h4>
      {children}
    </section>
  );
}

function TextList({ items, keys }: { items: readonly string[]; keys: ReadonlySet<string> }) {
  const shown = items.filter((item) => item.trim() !== "");
  if (shown.length === 0) {
    return <p className="text-sm text-muted-foreground">{NONE_IDENTIFIED}</p>;
  }
  return (
    <ul className="list-disc space-y-1 pl-5 text-sm">
      {shown.map((item, index) => (
        <li key={`${String(index)}-${item.slice(0, 24)}`}>
          <CitedText text={item} keys={keys} />
        </li>
      ))}
    </ul>
  );
}

function keysOf(keys: readonly string[]): string {
  return keys.length > 0 ? ` (${keys.join(", ")})` : "";
}

/** The header's words for each disposition and the gate that decided it (RMD3-T01). */
const DISPOSITION_LABELS: Record<PharmaDisposition, string> = {
  answer: "Answered",
  partial_answer: "Partly answered",
  abstain_insufficient_evidence: "Abstained, insufficient evidence",
  refuse_prohibited_decision: "Refused, a regulated decision",
  out_of_scope: "Out of scope",
  conflict_requires_review: "Conflicting evidence, review required",
  system_failure: "System failure",
};
const GATE_LABELS: Record<DispositionGate, string> = {
  sufficiency: "the retrieval's sufficiency rule",
  decision: "the decision step",
  grounding: "the grounding check",
  policy: "the regulatory-boundary check",
};

/**
 * The result's disposition and its gate (RE2E-DEF-034), never the retrieval's sufficiency, which
 * stood above results that gave no answer; a run recorded before the disposition was stored keeps
 * the retrieval's classification, the only reading it has.
 */
function resultLine(pharma: PharmaRun): string {
  if (!pharma.disposition) {
    return `Evidence: ${pharma.sufficiency}`;
  }
  const gate = pharma.disposition_gate
    ? `; decided by ${GATE_LABELS[pharma.disposition_gate]}`
    : "";
  return `Result: ${DISPOSITION_LABELS[pharma.disposition]}${gate}`;
}

/**
 * A pharmaceutical run's result (REQ-UX-05 to REQ-UX-07), read from `detail.pharma`, the
 * policy-checked record: the human-review banner, an abstention's own state, the policy notices,
 * and the nine sections in the server's order, with version-aware citation cards, the conflict
 * comparison and the escalation callout. Every model-written field is rendered as text. The
 * evaluation follows, its explanation withheld by the server (PH-D-030).
 */
export function PharmaResultView({
  result,
  pharma,
}: {
  result: QueryRunDetail;
  pharma: PharmaRun;
}) {
  const response = pharma.response;
  const keys = new Set(response.evidence_and_citations.map((item) => item.citation_key));
  const excluded = excludedVersions(pharma);
  return (
    <section className="space-y-4">
      {/* A labelled region, as on the overview, so moving by landmarks reaches it (PH-DEF-040). */}
      <section aria-label="Human review" className="rounded-lg border bg-muted/40 p-4 text-sm">
        <p>
          <strong>Human review required.</strong> {pharma.banner}
        </p>
      </section>
      {pharma.abstained && (
        <AbstentionState
          summary={response.incident_or_question_summary}
          disposition={pharma.disposition}
        />
      )}
      <PolicyNotices notices={pharma.policy_notices} />
      <Card>
        <CardHeader>
          <CardTitle>Pharmaceutical analysis</CardTitle>
          <CardDescription>{INTENT_LABELS[pharma.intent]}</CardDescription>
          <p className="text-sm">{resultLine(pharma)}</p>
        </CardHeader>
        <CardContent className="space-y-6">
          <Section title="Incident or Question Summary">
            <p className="text-sm">
              {response.incident_or_question_summary.trim() === "" ? (
                NONE_IDENTIFIED
              ) : (
                <CitedText text={response.incident_or_question_summary} keys={keys} />
              )}
            </p>
          </Section>
          <Section title="Relevant Procedures">
            {response.relevant_procedures.length === 0 ? (
              <p className="text-sm text-muted-foreground">{NONE_IDENTIFIED}</p>
            ) : (
              <ul className="space-y-2 text-sm">
                {response.relevant_procedures.map((procedure) => (
                  <li
                    key={`${procedure.document_number}-${procedure.revision}`}
                    className="space-y-1"
                  >
                    <p className="flex flex-wrap items-center gap-x-2 gap-y-1">
                      <span className="font-medium">{procedure.document_number}</span>
                      <span>revision {procedure.revision}</span>
                      <LifecycleBadge status={procedure.lifecycle_status} />
                      <span>{procedure.title}</span>
                    </p>
                    <p className="text-muted-foreground">
                      Sections: {procedure.sections.join(", ") || "unknown"}
                      {keysOf(procedure.citation_keys)}
                    </p>
                    {procedure.summary.trim() !== "" && (
                      <p>
                        <CitedText text={procedure.summary} keys={keys} />
                      </p>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </Section>
          <Section title="Evidence and Citations">
            {response.evidence_and_citations.length === 0 ? (
              <p className="text-sm text-muted-foreground">{NONE_IDENTIFIED}</p>
            ) : (
              <div className="grid gap-3">
                {response.evidence_and_citations.map((item) => (
                  <EvidenceCitationCard key={item.citation_key} item={item} />
                ))}
              </div>
            )}
            {excluded.length > 0 && (
              <div className="space-y-1 text-sm">
                <h5 className="font-medium">Versions excluded from this answer</h5>
                {/* One line per version, its sections merged, at every width (PH-DEF-039). */}
                <ul aria-label="Excluded versions" className="space-y-1">
                  {excluded.map((version) => (
                    <li key={version.key} className="flex flex-wrap items-center gap-x-2 gap-y-1">
                      <span>
                        {version.documentNumber} revision {version.revision}
                      </span>
                      <LifecycleBadge status={version.lifecycle} />
                      <span className="text-muted-foreground">
                        {version.use}; sections: {version.sections.join(", ") || "unknown"}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </Section>
          <Section title="Quality and Operational Risks">
            {response.quality_and_operational_risks.length === 0 ? (
              <p className="text-sm text-muted-foreground">{NONE_IDENTIFIED}</p>
            ) : (
              <ul className="list-disc space-y-1 pl-5 text-sm">
                {response.quality_and_operational_risks.map((flag, index) => (
                  <li key={`${String(index)}-${flag.category}`}>
                    <span className="font-medium">
                      {words(flag.severity)} severity, {words(flag.category).toLowerCase()}
                    </span>
                    : <CitedText text={flag.description} keys={keys} />
                    {keysOf(flag.citation_keys)}
                  </li>
                ))}
              </ul>
            )}
          </Section>
          <Section title="Conflicts or Ambiguities">
            <ConflictComparison run={pharma} citationKeys={keys} />
          </Section>
          <Section title="Missing Information">
            <TextList items={response.missing_information} keys={keys} />
          </Section>
          <Section title="Required Human Escalation">
            <EscalationCallout escalation={response.required_human_escalation} />
          </Section>
          <Section title="Suggested Investigation Questions or Next Steps">
            <TextList
              items={response.suggested_investigation_questions_or_next_steps}
              keys={keys}
            />
          </Section>
          <Section title="Limitations and Uncertainty">
            <TextList items={response.limitations_and_uncertainty} keys={keys} />
          </Section>
        </CardContent>
      </Card>
      <EvaluationPanel evaluation={result.evaluation} citations={result.citations} />
    </section>
  );
}
