import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Markdown, citationAnchorId, citationMarkerSegments } from "@/components/ui/markdown";

interface AnswerViewProps {
  markdown: string;
  /**
   * C-15 / D-7. `null`: the decision step produced no readable output. `""`: the agent stated no
   * uncertainty. `undefined`: a backend that predates the field, so nothing is claimed either way.
   */
  uncertainty?: string | null;
  nextSteps?: readonly string[];
  /** The run's citation keys: `[key]` markers in the answer link to their entries (FE-036). */
  citationKeys?: readonly string[];
  /**
   * RMD8-T01 (RE2E-DEF-031): how many references in the answer named no returned source and were
   * removed by the backend, which used to show each as "[source]".
   */
  unresolvedCitationMarkers?: number;
}

function unresolvedText(count: number): string {
  return count === 1
    ? "1 reference in this answer named no returned source and was removed."
    : `${String(count)} references in this answer named no returned source and were removed.`;
}

/** Plain text whose citation markers link to their entries, as the answer's do (FE-036). */
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

function uncertaintyText(uncertainty: string | null): string {
  if (uncertainty === null) {
    return "No readable decision output: the decision step returned no statement of uncertainty.";
  }
  if (uncertainty === "") {
    return "The agent stated no uncertainty.";
  }
  return uncertainty;
}

export function AnswerView({
  markdown,
  uncertainty,
  nextSteps = [],
  citationKeys,
  unresolvedCitationMarkers = 0,
}: AnswerViewProps) {
  const keys = new Set(citationKeys);
  return (
    <Card>
      <CardHeader>
        <CardTitle>Recommendation</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* The answer's own "#" headings sit below the card title, an h3 (FE-033). */}
        <Markdown citationKeys={citationKeys} headingOffset={3}>
          {markdown}
        </Markdown>
        {unresolvedCitationMarkers > 0 && (
          <p className="text-sm text-muted-foreground">
            {unresolvedText(unresolvedCitationMarkers)}
          </p>
        )}
        {/* Model output, rendered as text rather than Markdown: these two fields are never
            formatted by the agent, and text keeps them inert. */}
        {uncertainty !== undefined && (
          <section className="space-y-1">
            <h4 className="text-sm font-semibold">Uncertainty</h4>
            <p className="whitespace-pre-wrap text-sm text-muted-foreground">
              <LinkedText text={uncertaintyText(uncertainty)} keys={keys} />
            </p>
          </section>
        )}
        <section className="space-y-1">
          <h4 className="text-sm font-semibold">Next steps</h4>
          {nextSteps.length === 0 ? (
            <p className="text-sm text-muted-foreground">The agent stated no next steps.</p>
          ) : (
            <ol className="list-decimal space-y-1 pl-5 text-sm">
              {nextSteps.map((step, index) => (
                <li key={`${String(index)}-${step.slice(0, 24)}`}>
                  <LinkedText text={step} keys={keys} />
                </li>
              ))}
            </ol>
          )}
        </section>
      </CardContent>
    </Card>
  );
}
