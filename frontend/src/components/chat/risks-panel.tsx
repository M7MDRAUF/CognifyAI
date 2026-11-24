import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { citationAnchorId, citationMarkerSegments } from "@/components/ui/markdown";
import type { RiskItem } from "@/lib/types/query";

/** The dashboard's words for each `RiskCategory` (`risk-donut-chart.tsx`), so both read alike. */
const CATEGORY_LABELS: Readonly<Record<string, string>> = {
  academic: "Academic",
  data_quality: "Data quality",
  ethical: "Ethical",
  implementation: "Implementation",
  operational: "Operational",
  privacy: "Privacy",
};

/** A category's readable label; an unknown value reads as its words, first letter capitalised. */
function categoryLabel(category: string): string {
  const known = CATEGORY_LABELS[category];
  if (known !== undefined) {
    return known;
  }
  const words = category.replaceAll("_", " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/**
 * Model text rendered as text, never as Markdown (FR-010): only a `[key]` marker naming one of
 * the run's citations becomes a link, to that citation's entry on the page, as the answer's
 * uncertainty and next steps do. Nothing else in the text can load or link anywhere.
 */
function LinkedText({ text, keys }: { text: string; keys: ReadonlySet<string> }) {
  return (
    <>
      {citationMarkerSegments(text, keys).map((segment, index) =>
        "citationKey" in segment ? (
          <a
            key={`${String(index)}-${segment.citationKey}`}
            href={`#${citationAnchorId(segment.citationKey)}`}
            className="underline underline-offset-2"
          >
            {segment.text}
          </a>
        ) : (
          segment.text
        ),
      )}
    </>
  );
}

/** The Risk agent's items for one run: category, severity and likelihood in words, never colour. */
export function RisksPanel({
  risks,
  citationKeys,
}: {
  risks: readonly RiskItem[];
  citationKeys: readonly string[];
}) {
  const keys = new Set(citationKeys);
  return (
    <Card>
      <CardHeader>
        <CardTitle>Risks</CardTitle>
      </CardHeader>
      <CardContent>
        {risks.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            The Risk agent recorded no risks for this analysis.
          </p>
        ) : (
          <ul className="space-y-4">
            {risks.map((risk, index) => (
              <li key={`${String(index)}-${risk.category}`} className="space-y-1 text-sm">
                <p>
                  <span className="font-medium">{categoryLabel(risk.category)}</span>
                  <span className="text-muted-foreground">
                    {" "}
                    · Severity: {risk.severity} · Likelihood: {risk.likelihood}
                  </span>
                </p>
                <p className="whitespace-pre-wrap">
                  <LinkedText text={risk.description} keys={keys} />
                </p>
                {risk.mitigation !== "" && (
                  <p className="whitespace-pre-wrap text-muted-foreground">
                    Mitigation: <LinkedText text={risk.mitigation} keys={keys} />
                  </p>
                )}
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

/** The decision's contradictory evidence for one run, as text. */
export function ContradictionsPanel({
  items,
  citationKeys,
}: {
  items: readonly string[];
  citationKeys: readonly string[];
}) {
  const keys = new Set(citationKeys);
  return (
    <Card>
      <CardHeader>
        <CardTitle>Contradictory evidence</CardTitle>
      </CardHeader>
      <CardContent>
        {items.length === 0 ? (
          <p className="text-sm text-muted-foreground">No contradictory evidence was recorded.</p>
        ) : (
          <ul className="list-disc space-y-2 pl-5 text-sm">
            {items.map((item, index) => (
              <li key={`${String(index)}-${item.slice(0, 24)}`} className="whitespace-pre-wrap">
                <LinkedText text={item} keys={keys} />
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}
