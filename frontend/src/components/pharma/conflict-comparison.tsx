import { CitedText } from "@/components/pharma/evidence-citation-card";
import { LifecycleBadge } from "@/components/pharma/lifecycle-badge";
import {
  UNNUMBERED,
  type VersionRow,
  citedVersions,
  excludedVersions,
} from "@/lib/pharma/versions";
import type { PharmaRun } from "@/lib/types/pharma";

/** The server's label for model-reported items, never merged into the rules' list (REQ-CON-02). */
export const MODEL_REPORTED_LABEL = "Reported by the analysis model:";

/**
 * What a comparison lists, stated under it (PH-DEF-039): the response names the versions the
 * answer cited and the ones the run excluded, not every eligible version it retrieved.
 */
export const COMPARISON_SCOPE =
  "Lists the versions this answer cited or excluded. A version retrieved and eligible but not " +
  "cited is not listed.";

/** Said instead of a table for an unnumbered document (PH-DEF-040). */
export const UNNUMBERED_NOT_COMPARED =
  "Documents without a document number are not compared: nothing identifies them as versions " +
  "of one document.";

function words(value: string): string {
  const spaced = value.replace(/_/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

function sectionsOf(version: VersionRow): string {
  return version.sections.join(", ") || "unknown";
}

/**
 * The versions of one document a conflict involves, one row per version (PH-DEF-039): revision,
 * lifecycle, every section the run used and whether the run cited or excluded it. Each is shown
 * at every width: in its own column from sm, beside the revision or the use below it.
 */
function Comparison({ documentNumber, run }: { documentNumber: string; run: PharmaRun }) {
  if (documentNumber === UNNUMBERED) {
    return <p className="text-muted-foreground">{UNNUMBERED_NOT_COMPARED}</p>;
  }
  const versions = [
    ...citedVersions(run, documentNumber),
    ...excludedVersions(run, documentNumber),
  ];
  if (versions.length === 0) {
    return (
      <p className="text-muted-foreground">
        No version of {documentNumber} was cited or excluded in this answer, so none is listed.
      </p>
    );
  }
  return (
    <div className="overflow-x-auto">
      <table aria-label={`Versions of ${documentNumber}`} className="w-full text-left">
        <caption className="caption-bottom pt-1 text-left text-xs text-muted-foreground">
          {COMPARISON_SCOPE}
        </caption>
        <thead>
          <tr className="border-b text-muted-foreground">
            <th className="py-1 pr-3 font-medium">Revision</th>
            <th className="hidden py-1 pr-3 font-medium sm:table-cell">Lifecycle</th>
            <th className="hidden py-1 pr-3 font-medium sm:table-cell">Sections</th>
            <th className="py-1 font-medium">In this answer</th>
          </tr>
        </thead>
        <tbody>
          {/* Below sm the lifecycle moves under the revision and the sections under the use, so
              the table fits its box at 320 px and loses nothing (PH-DEF-039). */}
          {versions.map((version) => (
            <tr key={version.key} className="border-b">
              <td className="py-1 pr-3 align-top sm:align-middle">
                {version.revision}
                <span className="mt-1 block sm:hidden">
                  <LifecycleBadge status={version.lifecycle} />
                </span>
              </td>
              <td className="hidden py-1 pr-3 sm:table-cell">
                <LifecycleBadge status={version.lifecycle} />
              </td>
              <td className="hidden py-1 pr-3 sm:table-cell">{sectionsOf(version)}</td>
              <td className="py-1 align-top sm:align-middle">
                {version.use}
                <span className="block text-muted-foreground sm:hidden">
                  Sections: {sectionsOf(version)}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * The conflicts the rules found, each with the versions it involves side by side (REQ-UX-06).
 * The conflicts the analysis model reported follow under their own label.
 */
export function ConflictComparison({
  run,
  citationKeys,
}: {
  run: PharmaRun;
  citationKeys: ReadonlySet<string>;
}) {
  const { deterministic, model_reported: reported } = run.response.conflicts_or_ambiguities;
  const shown = reported.filter((text) => text.trim() !== "");
  if (deterministic.length === 0 && shown.length === 0) {
    return <p className="text-sm text-muted-foreground">None identified.</p>;
  }
  return (
    <div className="space-y-4 text-sm">
      {deterministic.map((conflict, index) => (
        <div key={`${String(index)}-${conflict.kind}`} className="space-y-2">
          <p>
            <span className="font-medium">{words(conflict.kind)}</span>:{" "}
            <CitedText text={conflict.description} keys={citationKeys} />
          </p>
          {conflict.document_numbers.map((number) => (
            <Comparison key={number} documentNumber={number} run={run} />
          ))}
        </div>
      ))}
      {shown.length > 0 && (
        <div>
          <h5 className="font-medium">{MODEL_REPORTED_LABEL}</h5>{" "}
          <ul className="list-disc space-y-1 pl-5">
            {shown.map((text, index) => (
              <li key={`${String(index)}-${text.slice(0, 24)}`}>
                <CitedText text={text} keys={citationKeys} />
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
