import { ShieldAlert } from "lucide-react";

import type { EscalationRole, HumanEscalation } from "@/lib/types/pharma";

export const ROLE_LABELS: Readonly<Record<EscalationRole, string>> = {
  quality_assurance: "Quality assurance",
  process_owner: "Process owner",
  subject_matter_expert: "Subject matter expert",
  document_control: "Document control",
};

/**
 * The response's escalation (REQ-UX-07): a required escalation is a callout naming the roles and
 * every reason, each marked as the rules' or as the analysis model's suggestion, as the server
 * words them. The icon repeats the heading; the meaning is in the text.
 */
export function EscalationCallout({ escalation }: { escalation: HumanEscalation }) {
  if (!escalation.required) {
    return (
      <p className="text-sm text-muted-foreground">
        No escalation is required by the rules; a qualified person still reviews the output.
      </p>
    );
  }
  const roles = escalation.roles.map((role) => ROLE_LABELS[role]).join(", ");
  return (
    <div
      role="note"
      aria-label="Escalation required"
      className="space-y-2 rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-sm"
    >
      <p className="flex items-center gap-2 font-semibold">
        <ShieldAlert aria-hidden="true" className="size-4" />
        Human escalation required
      </p>
      <p>Roles: {roles === "" ? "none named" : roles}.</p>
      {escalation.reasons.length > 0 && (
        <ul className="list-disc space-y-1 pl-5">
          {/* Keyed by position too: every reason the model suggests carries REQ-ESC-07 (PH-DEF-040). */}
          {escalation.reasons.map((reason, index) => (
            <li key={`${String(index)}-${reason.code}`}>
              {reason.code} ({reason.source === "rule" ? "rule" : "suggested by the analysis model"}
              ): {reason.text}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
