import { Badge } from "@/components/ui/badge";

/**
 * The server's fixed texts (`backend/app/pharma/catalogue.py`), copied byte for byte;
 * `pharma-project.test.tsx` reads the catalogue and fails if either drifts (REQ-UX-02).
 */
export const HUMAN_REVIEW_BANNER =
  "Decision support for qualified personnel. This research prototype does not approve " +
  "procedures, release batches, determine root causes or replace the site's quality process; " +
  "a qualified person reviews every output.";

export const PROTOTYPE_LIMITATION =
  "This is a research prototype evaluated only on synthetic and public documents. It is not " +
  "validated for regulated use, and its output is not a controlled record.";

/**
 * A pharmaceutical project's standing notice (REQ-UX-02): the human-review banner, worded as the
 * server words it above every answer, and the data indicator. Text carries the meaning; the
 * border and the badge only group it.
 */
export function HumanReviewBanner() {
  return (
    <section
      aria-label="Human review"
      className="space-y-2 rounded-lg border bg-muted/40 p-4 text-sm"
    >
      <p>
        <strong>Human review required.</strong> {HUMAN_REVIEW_BANNER}
      </p>
      <p className="text-muted-foreground">
        <Badge variant="outline" className="mr-2">
          Synthetic and public data only
        </Badge>
        {PROTOTYPE_LIMITATION}
      </p>
    </section>
  );
}
