import { LifecycleBadge } from "@/components/pharma/lifecycle-badge";
import { citationAnchorId, citationMarkerSegments } from "@/components/ui/markdown";
import type { PharmaResponse } from "@/lib/types/pharma";

type EvidenceItem = PharmaResponse["evidence_and_citations"][number];

/**
 * Model text whose `[key]` markers link to their evidence cards (FE-036). Text, never parsed as
 * Markdown or HTML: a checked field is plain prose, and rendering it as text keeps it inert.
 */
export function CitedText({ text, keys }: { text: string; keys: ReadonlySet<string> }) {
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

/**
 * One cited passage with the version it came from (REQ-UX-05): document number, revision,
 * lifecycle status in text, section, page and chunk, and the bounded excerpt. Its id is the
 * citation's anchor, so every `[key]` marker in the result links here.
 */
export function EvidenceCitationCard({ item }: { item: EvidenceItem }) {
  const { anchor } = item;
  const page = anchor.page === null ? "" : `, page ${String(anchor.page)}`;
  // The stored index counts from 0; the card counts the document's chunks from 1 (PH-DEF-040).
  const chunk = ` · Chunk ${String(anchor.chunk_index + 1)}`;
  return (
    <article
      id={citationAnchorId(item.citation_key)}
      aria-label={`Evidence ${item.citation_key}`}
      className="scroll-mt-4 space-y-1 rounded-md border p-3 text-sm"
    >
      <p className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <span className="font-medium">[{item.citation_key}]</span>
        <span className="font-medium">{anchor.document_number}</span>
        <span>Revision {anchor.revision}</span>
        <LifecycleBadge status={anchor.lifecycle_status} />
      </p>
      <p className="text-muted-foreground">
        {anchor.title} · Section {anchor.section}
        {page}
        {chunk}
      </p>
      {item.excerpt.trim() !== "" && (
        <blockquote className="border-l-2 pl-3 whitespace-pre-wrap text-muted-foreground">
          {item.excerpt}
        </blockquote>
      )}
    </article>
  );
}
