/**
 * L-2 on the chat surface: the relabelling the certified report applies to reader-facing text
 * (`backend/app/services/citation_service.py::conceal_identifiers`, used by `report_service.py`).
 * A raw identifier, bare or in the brackets a model usually wraps a citation in, becomes
 * `[<citation key>]` when a citation of the run names that chunk. One no citation names is
 * removed with the space before it (RMD8-T01, RE2E-DEF-031): it used to become `[source]`, which
 * read as a reference and led nowhere, as the backend's unresolved markers are now removed too.
 * Whitespace is absorbed only inside brackets, so a label is never welded onto the word before it.
 */
const ID = "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}";
const RAW_IDENTIFIER = new RegExp(`([ \\t]?)(?:\\[\\s*(${ID})\\s*\\]|(${ID}))`, "g");
const WHOLE_IDENTIFIER = new RegExp(`^(?:\\[\\s*${ID}\\s*\\]|${ID})$`);

/**
 * `chunk id -> citation key` for the run's citations; the first citation of a chunk wins. A key that
 * is itself a raw identifier is never a label, since it would re-expose what is being concealed.
 */
export function citationLabels(
  citations: Iterable<{ chunk_id: string; citation_key: string }>,
): ReadonlyMap<string, string> {
  const labels = new Map<string, string>();
  for (const citation of citations) {
    const chunkId = citation.chunk_id.toLowerCase();
    if (!WHOLE_IDENTIFIER.test(citation.citation_key.trim()) && !labels.has(chunkId)) {
      labels.set(chunkId, citation.citation_key);
    }
  }
  return labels;
}

/** Replace every raw identifier in `text` with the citation that names it, or remove it. */
export function concealIdentifiers(text: string, labels: ReadonlyMap<string, string>): string {
  return text.replace(
    RAW_IDENTIFIER,
    (_match, space: string, bracketed?: string, bare?: string) => {
      const key = labels.get((bracketed ?? bare ?? "").toLowerCase());
      return key ? `${space}[${key}]` : "";
    },
  );
}
