import { CircleSlash } from "lucide-react";

import type { PharmaDisposition } from "@/lib/types/pharma";

/** The state's name for each disposition shown as an abstention (RMD3-T01). */
const ABSTENTION_NAMES: Partial<Record<PharmaDisposition, string>> = {
  refuse_prohibited_decision: "Regulated decision",
  out_of_scope: "Out of scope",
};
const INSUFFICIENT_EVIDENCE = "Insufficient evidence";

/**
 * A result without a procedure answer (REQ-UX-06), distinct from an answer: the server abstained
 * and its summary says why. Named by the disposition (RMD3-T01): insufficient evidence, whichever
 * gate found it, a regulated decision the regulatory-boundary check refused, or a question out of
 * scope; a run recorded before the disposition was stored abstained on insufficient evidence.
 * Shown above the nine sections, which still list what is missing and who reviews it.
 */
export function AbstentionState({
  summary,
  disposition,
}: {
  summary: string;
  disposition?: PharmaDisposition | null;
}) {
  const name = (disposition ? ABSTENTION_NAMES[disposition] : undefined) ?? INSUFFICIENT_EVIDENCE;
  return (
    <div
      role="status"
      aria-label={name}
      className="space-y-1 rounded-md border-2 border-dashed p-4 text-sm"
    >
      {/* A heading, so moving by headings reaches it before the analysis (PH-DEF-040). */}
      <h3 className="flex items-center gap-2 font-semibold">
        <CircleSlash aria-hidden="true" className="size-4" />
        {`${name}: no procedure answer`}
      </h3>
      <p>{summary}</p>
    </div>
  );
}
