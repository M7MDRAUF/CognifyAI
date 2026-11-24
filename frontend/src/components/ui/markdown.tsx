import type { ComponentPropsWithoutRef } from "react";
import ReactMarkdown, { defaultUrlTransform } from "react-markdown";
import remarkGfm from "remark-gfm";

import { cn } from "@/lib/utils";

/** The element id of a citation's entry, which an in-answer `[key]` marker links to (FE-036). */
export function citationAnchorId(citationKey: string): string {
  return `citation-${citationKey.toLowerCase().replace(/[^a-z0-9_-]+/g, "-")}`;
}

/** The parts of an mdast node this file reads or builds. */
interface MdNode {
  type: string;
  value?: string;
  url?: string;
  depth?: number;
  children?: MdNode[];
}

const MARKER = /\[([^[\]\n]+)\]/g;

export type MarkerSegment = { text: string } | { text: string; citationKey: string };

/** `value` split into plain text and the `[key]` markers that name one of `keys`. */
export function citationMarkerSegments(value: string, keys: ReadonlySet<string>): MarkerSegment[] {
  const segments: MarkerSegment[] = [];
  let last = 0;
  for (const match of value.matchAll(MARKER)) {
    const key = match[1];
    if (key === undefined || !keys.has(key)) {
      continue;
    }
    if (match.index > last) {
      segments.push({ text: value.slice(last, match.index) });
    }
    segments.push({ text: match[0], citationKey: key });
    last = match.index + match[0].length;
  }
  if (last < value.length) {
    segments.push({ text: value.slice(last) });
  }
  return segments;
}

/** Split a text node so every `[key]` naming a citation becomes a link to that citation. */
function splitMarkers(value: string, keys: ReadonlySet<string>): MdNode[] {
  return citationMarkerSegments(value, keys).map((segment) =>
    "citationKey" in segment
      ? {
          type: "link",
          url: `#${citationAnchorId(segment.citationKey)}`,
          children: [{ type: "text", value: segment.text }],
        }
      : { type: "text", value: segment.text },
  );
}

function linkMarkers(node: MdNode, keys: ReadonlySet<string>): void {
  // Existing links and code keep their text as written.
  if (node.children === undefined || node.type === "link" || node.type === "linkReference") {
    return;
  }
  node.children = node.children.flatMap((child) => {
    if (child.type === "text" && child.value !== undefined) {
      return splitMarkers(child.value, keys);
    }
    linkMarkers(child, keys);
    return [child];
  });
}

/**
 * A remark step linking in-answer citation markers to their entries. It builds mdast link nodes,
 * never raw HTML, so the renderer's sanitising stays in force (the card's hard stop).
 */
function linkCitationMarkers(keys: ReadonlySet<string>) {
  return () => (tree: MdNode) => {
    linkMarkers(tree, keys);
  };
}

/**
 * The single markdown renderer for agent output.
 *
 * `remark-gfm` is not optional here. `react-markdown` parses CommonMark, which has no table
 * syntax, so before this every `| Field | Value |` row an agent produced reached the page as
 * literal pipe characters — and the synthesis agents emit tables constantly, because the prompt
 * asks them to lay out figures and per-row evidence.
 *
 * The `table` override exists because a `prose` table has no width constraint: a wide one from a
 * CSV row pushes the whole card sideways and the page scrolls horizontally on a phone. Wrapping
 * it keeps the overflow inside the card. `not-prose` on the wrapper stops the typography plugin
 * styling the scroll container itself; the table inside is styled explicitly, so the two do not
 * fight over borders.
 */
/**
 * FE-020 (N-012): model output is untrusted, so a destination survives only as an in-page
 * fragment or an absolute http(s) or mailto URL. react-markdown's own transform already blanks
 * unsafe protocols such as `javascript:`; protocol-relative (`//host`) and relative paths are
 * blanked here too, and a blank destination renders as plain text.
 */
function safeUrl(url: string): string {
  if (url.startsWith("#")) {
    return url;
  }
  if (url.startsWith("//")) {
    return "";
  }
  const transformed = defaultUrlTransform(url);
  return /^(https?:|mailto:)/i.test(transformed) ? transformed : "";
}

function hostOf(href: string): string {
  try {
    const url = new URL(href);
    return url.host === "" ? url.pathname : url.host;
  } catch {
    return href;
  }
}

/**
 * No image is ever loaded: a remote image in an answer is a request the reader never chose (an
 * exfiltration channel), so only its alt text is shown. Every outbound link opens in a new tab
 * with `rel` set, and says where it goes.
 */
const SAFE_COMPONENTS = {
  img: ({ alt }: ComponentPropsWithoutRef<"img">) => (
    <span className="rounded border border-dashed px-1 text-xs text-muted-foreground">
      Image not shown: {alt !== undefined && alt !== "" ? alt : "no description"}
    </span>
  ),
  a: ({ href, children: content }: ComponentPropsWithoutRef<"a">) => {
    if (href === undefined || href === "") {
      return <>{content}</>;
    }
    if (href.startsWith("#")) {
      return <a href={href}>{content}</a>;
    }
    return (
      <a href={href} target="_blank" rel="noopener noreferrer nofollow">
        {content}
        <span className="ml-1 text-xs text-muted-foreground">({hostOf(href)})</span>
        <span aria-hidden> ↗</span>
        <span className="sr-only"> (opens in a new tab)</span>
      </a>
    );
  },
};

function headingDepths(node: MdNode, depths: Set<number>): void {
  if (node.type === "heading" && node.depth !== undefined) {
    depths.add(node.depth);
  }
  node.children?.forEach((child) => {
    headingDepths(child, depths);
  });
}

function setHeadingDepths(node: MdNode, target: ReadonlyMap<number, number>): void {
  if (node.type === "heading" && node.depth !== undefined) {
    node.depth = target.get(node.depth) ?? node.depth;
  }
  node.children?.forEach((child) => {
    setHeadingDepths(child, target);
  });
}

/**
 * A remark step placing a document's headings under the card title it renders in (FE-033): the
 * levels the document uses, ranked, become consecutive levels from `offset + 1`, capped at h6. So
 * a document's own "#" never becomes a page's second h1, and one that opens at "##" or skips a
 * level still leaves no gap under the card (axe `heading-order`).
 */
function nestHeadings(offset: number) {
  return () => (tree: MdNode) => {
    const depths = new Set<number>();
    headingDepths(tree, depths);
    const ranked = [...depths].sort((a, b) => a - b);
    setHeadingDepths(
      tree,
      new Map(ranked.map((depth, rank) => [depth, Math.min(offset + rank + 1, 6)])),
    );
  };
}

interface MarkdownProps {
  children: string;
  className?: string;
  /** Levels to shift the document's headings down by, when it renders under a heading. */
  headingOffset?: number;
  /** Citation keys whose `[key]` markers link to their entries on the same page. */
  citationKeys?: readonly string[];
}

export function Markdown({ children, className, citationKeys, headingOffset = 0 }: MarkdownProps) {
  const remarkPlugins = [
    remarkGfm,
    ...(citationKeys === undefined || citationKeys.length === 0
      ? []
      : [linkCitationMarkers(new Set(citationKeys))]),
    ...(headingOffset > 0 ? [nestHeadings(headingOffset)] : []),
  ];
  return (
    <div className={cn("prose prose-sm max-w-none", className)}>
      <ReactMarkdown
        remarkPlugins={remarkPlugins}
        urlTransform={safeUrl}
        components={{
          ...SAFE_COMPONENTS,
          // A wide table scrolls inside its own box; the box takes focus so the keyboard can
          // scroll it (WCAG 2.1.1). It stays unnamed: an answer may hold several tables.
          table: ({ children: rows }: ComponentPropsWithoutRef<"table">) => (
            <div
              tabIndex={0}
              className="not-prose my-4 w-full overflow-x-auto rounded-lg border border-border focus-visible:outline-2 focus-visible:outline-offset-2"
            >
              <table className="w-full border-collapse text-sm">{rows}</table>
            </div>
          ),
          thead: ({ children: head }: ComponentPropsWithoutRef<"thead">) => (
            <thead className="bg-muted/60">{head}</thead>
          ),
          // `style` carries the column's GFM alignment (`|--:|`) as `text-align` (FE-038).
          th: ({ children: cell, style }: ComponentPropsWithoutRef<"th">) => (
            <th
              style={style}
              className="border-b border-border px-3 py-2 text-left font-semibold whitespace-nowrap"
            >
              {cell}
            </th>
          ),
          td: ({ children: cell, style }: ComponentPropsWithoutRef<"td">) => (
            <td
              style={style}
              className="border-b border-border px-3 py-2 align-top last:border-b-0"
            >
              {cell}
            </td>
          ),
          tr: ({ children: cells }: ComponentPropsWithoutRef<"tr">) => (
            <tr className="last:[&>td]:border-b-0 even:bg-muted/25">{cells}</tr>
          ),
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
