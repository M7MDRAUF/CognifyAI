import { TriangleAlert } from "lucide-react";
import { useId } from "react";

import type { ControlledMetadata } from "@/lib/types/pharma";

const STATUS_TEXT: Readonly<Record<ControlledMetadata["metadata_status"], string>> = {
  valid: "Metadata valid",
  missing: "Metadata missing",
  invalid: "Metadata invalid",
};

/**
 * Why a document's controlled header could not be fully read (REQ-UX-03): the status in words and
 * each issue the server recorded, as text. Nothing renders for valid metadata with no issue.
 */
export function MetadataIssues({
  status,
  issues,
}: {
  status: ControlledMetadata["metadata_status"];
  issues: readonly string[];
}) {
  const id = useId();
  if (status === "valid" && issues.length === 0) {
    return null;
  }
  return (
    <div className="mt-1 text-xs text-muted-foreground">
      <p id={id} className="flex items-center gap-1 font-medium text-foreground">
        <TriangleAlert aria-hidden="true" className="size-3.5" />
        {STATUS_TEXT[status]}
      </p>
      {issues.length > 0 && (
        <ul aria-label="Metadata issues" aria-describedby={id} className="ml-4 list-disc">
          {issues.map((issue, index) => (
            <li key={`${String(index)}-${issue}`}>{issue}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
