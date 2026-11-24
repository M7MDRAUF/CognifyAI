import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { citationAnchorId } from "@/components/ui/markdown";
import type { Citation } from "@/lib/types/query";

/**
 * `citation_service._MAX_SNIPPET`: a snippet that fell back to the chunk's opening is
 * `chunk.text[:300]`, so one this long that ends inside a word was cut there.
 */
const BACKEND_SNIPPET_LIMIT = 300;
const TABLE_DELIMITER = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;

/** One snippet line with its Markdown syntax removed, as text (never parsed into HTML). */
function plainLine(line: string): string {
  let text = line;
  if (text.includes("|")) {
    // A table row, whole or cut: its cells, joined.
    text = text
      .split("|")
      .map((cell) => cell.trim())
      .filter((cell) => cell !== "")
      .join(" · ");
  }
  return (
    text
      .replace(/^\s{0,3}#{1,6}\s+/, "")
      .replace(/^\s{0,3}>\s?/, "")
      .replace(/^\s*[-*+]\s+/, "• ")
      // A link destination may hold one level of parentheses, as `javascript:alert(1)` does.
      .replace(/!\[([^\]]*)\]\((?:[^()]|\([^()]*\))*\)/g, "$1")
      .replace(/\[([^\]]+)\]\((?:[^()]|\([^()]*\))*\)/g, "$1")
      .replace(/`+/g, "")
      .replace(/(\*\*|__)(?=\S)(.+?)(?<=\S)\1/g, "$2")
      .replace(/(^|[^\w*])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])/g, "$1$2")
      .replace(/(^|\W)_(?=\S)(.+?)(?<=\S)_(?!\w)/g, "$1$2")
      .replace(/~~(.+?)~~/g, "$1")
      .replace(/\*\*|__/g, "")
  );
}

/**
 * A snippet as readable text (FE-036): Markdown syntax removed, rows kept on their own lines,
 * and a snippet the backend cut inside a word ended at its last whole word with an ellipsis.
 */
export function readableSnippet(snippet: string): string {
  let text = snippet
    .split("\n")
    .filter((line) => !TABLE_DELIMITER.test(line))
    .map(plainLine)
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
  if (snippet.length >= BACKEND_SNIPPET_LIMIT && /[\p{L}\p{N}]$/u.test(snippet)) {
    text = `${text.replace(/\s*\S+$/, "")}…`;
  }
  return text;
}

export function CitationsPanel({ citations }: { citations: readonly Citation[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Citations</CardTitle>
      </CardHeader>
      <CardContent>
        {citations.length === 0 ? (
          <p className="text-sm text-muted-foreground">No citations were produced.</p>
        ) : (
          <ul className="space-y-3">
            {citations.map((citation) => (
              <li
                key={citation.citation_key}
                id={citationAnchorId(citation.citation_key)}
                className="scroll-mt-4 text-sm"
              >
                <span className="font-medium">[{citation.citation_key}]</span>{" "}
                {citation.source_filename}
                {citation.page_number !== null && ` p.${String(citation.page_number)}`}{" "}
                {/* The report's evidence-table scale: rerank 0..1, else hybrid (C-11). */}
                <span className="text-xs text-muted-foreground">
                  Relevance {citation.relevance_score.toFixed(2)}
                </span>
                {/*
                  Text, not a Markdown render. A snippet is an arbitrary slice of a chunk, so its
                  Markdown is usually malformed -- table fragments start below their header row,
                  with no delimiter line -- and would render as bare pipes. Its syntax is removed
                  as text and its newlines kept, so rows stay rows; the words are unchanged. The
                  verbatim excerpt stays in the generated report.
                */}
                <blockquote className="mt-1 border-l-2 pl-3 text-muted-foreground whitespace-pre-wrap">
                  {readableSnippet(citation.snippet)}
                </blockquote>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}
