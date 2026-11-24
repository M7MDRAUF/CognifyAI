import type { EvidenceAnchor, LifecycleStatus, PharmaRun } from "@/lib/types/pharma";

export type ExclusionReason = PharmaRun["excluded_evidence"][number]["reason"];

/** How the result states each exclusion reason, the same words wherever it appears. */
export const EXCLUSION_WORDS: Readonly<Record<ExclusionReason, string>> = {
  superseded: "superseded",
  retired: "retired",
  draft: "draft",
  lower_revision: "a lower revision",
  superseded_by_effective: "superseded by an effective revision",
  filter_mismatch: "outside the filters",
};

/** The server's number for a document whose header states none (`UNKNOWN`, schemas/pharma.py). */
export const UNNUMBERED = "unknown";

/**
 * One version of a controlled document as a result shows it: its number, revision and lifecycle,
 * every section of it the run used, and what the run did with it. The run's evidence and its
 * exclusions are per chunk; a version is one row however many of its chunks were retrieved
 * (PH-DEF-039).
 */
export interface VersionRow {
  key: string;
  documentNumber: string;
  revision: string;
  lifecycle: LifecycleStatus;
  sections: string[];
  use: string;
}

interface Entry {
  anchor: EvidenceAnchor;
  group: string;
  use: (labels: string[]) => string;
  label: string;
}

function rows(entries: Entry[]): VersionRow[] {
  const byKey = new Map<string, VersionRow & { labels: string[]; describe: Entry["use"] }>();
  for (const { anchor, group, use, label } of entries) {
    // No number, revision, status or reason holds "|".
    const key = [anchor.document_number, anchor.revision, anchor.lifecycle_status, group].join("|");
    let row = byKey.get(key);
    if (row === undefined) {
      row = {
        key,
        documentNumber: anchor.document_number,
        revision: anchor.revision,
        lifecycle: anchor.lifecycle_status,
        sections: [],
        use: "",
        labels: [],
        describe: use,
      };
      byKey.set(key, row);
    }
    if (anchor.section.trim() !== "" && !row.sections.includes(anchor.section)) {
      row.sections.push(anchor.section);
    }
    if (label !== "" && !row.labels.includes(label)) {
      row.labels.push(label);
    }
  }
  return [...byKey.values()].map(({ labels, describe, ...row }) => ({
    ...row,
    use: describe(labels),
  }));
}

/** The versions the answer cited, in citation order, each with every key that cites it. */
export function citedVersions(run: PharmaRun, documentNumber?: string): VersionRow[] {
  return rows(
    run.response.evidence_and_citations
      .filter(
        (item) => documentNumber === undefined || item.anchor.document_number === documentNumber,
      )
      .map((item) => ({
        anchor: item.anchor,
        group: "cited",
        label: item.citation_key,
        use: (keys) => `Cited as ${keys.join(", ")}`,
      })),
  );
}

/** The versions the run excluded, in the server's order, one row per version and reason. */
export function excludedVersions(run: PharmaRun, documentNumber?: string): VersionRow[] {
  return rows(
    run.excluded_evidence
      .filter(
        (item) => documentNumber === undefined || item.anchor.document_number === documentNumber,
      )
      .map((item) => ({
        anchor: item.anchor,
        group: item.reason,
        label: "",
        use: () => `Excluded: ${EXCLUSION_WORDS[item.reason]}`,
      })),
  );
}
