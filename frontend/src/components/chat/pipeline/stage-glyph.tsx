import type { CSSProperties } from "react";

import styles from "@/components/chat/pipeline/agent-pipeline.module.css";
import type { PipelineStageId } from "@/lib/chat/pipeline";
import { cn } from "@/lib/utils";

/** A part's box in the 16 px glyph, with the motion variables the CSS module reads. */
type PartStyle = CSSProperties & Record<`--udp-${string}`, string>;

interface GlyphPart {
  shape: "bar" | "dot" | "tick";
  style: PartStyle;
}

function box(left: number, top: number, width: number, height: number): PartStyle {
  return { left, top, width, height };
}

/** A researcher dot that draws in to the glyph's centre as it shrinks. */
function corner(left: number, top: number): GlyphPart {
  return {
    shape: "dot",
    style: {
      ...box(left, top, 3, 3),
      transformOrigin: `${String(8 - left)}px ${String(8 - top)}px`,
      "--udp-scale": "0.4",
      "--udp-fade": "0.35",
    },
  };
}

/** An evaluator bar standing on the baseline, growing from its bottom edge. */
function rising(left: number, height: number, delay: string): GlyphPart {
  return {
    shape: "bar",
    style: {
      ...box(left, 14 - height, 3, height),
      transformOrigin: "bottom",
      "--udp-delay": delay,
    },
  };
}

/**
 * Each agent's parts (plan section 18's metaphors). Standing still, they are the glyph as it
 * reads under reduced motion; the CSS module moves them only while the stage runs.
 */
const GLYPH_PARTS: Readonly<Record<PipelineStageId, readonly GlyphPart[]>> = {
  // Three bars settle into a column, 150 ms apart (MOT-02).
  planner: [
    { shape: "bar", style: box(2, 3, 12, 2) },
    { shape: "bar", style: { ...box(2, 7, 9, 2), "--udp-delay": "150ms" } },
    { shape: "bar", style: { ...box(2, 11, 11, 2), "--udp-delay": "300ms" } },
  ],
  // Three dots converge on the centre (MOT-03).
  retriever: [
    { shape: "dot", style: { ...box(0, 6, 4, 4), "--udp-shift": "6px" } },
    { shape: "dot", style: box(6, 6, 4, 4) },
    { shape: "dot", style: { ...box(12, 6, 4, 4), "--udp-shift": "-6px" } },
  ],
  // Two bars exchange places (MOT-04).
  reranker: [
    { shape: "bar", style: { ...box(2, 4, 12, 2), "--udp-shift": "6px" } },
    { shape: "bar", style: { ...box(2, 10, 7, 2), "--udp-shift": "-6px" } },
  ],
  // Four dots consolidate into the one at the centre, which grows (MOT-05).
  researcher: [
    corner(2, 2),
    corner(11, 2),
    corner(2, 11),
    corner(11, 11),
    { shape: "dot", style: { ...box(6, 6, 4, 4), "--udp-scale": "1.5" } },
  ],
  // Check marks appear in turn, 200 ms apart (MOT-06).
  critic: [
    { shape: "tick", style: box(0, 4, 5, 6) },
    { shape: "tick", style: { ...box(5.5, 4, 5, 6), "--udp-delay": "200ms" } },
    { shape: "tick", style: { ...box(11, 4, 5, 6), "--udp-delay": "400ms" } },
  ],
  // Markers fade in calmly, never flashing (MOT-07).
  risk: [
    { shape: "dot", style: box(6, 2, 4, 4) },
    { shape: "dot", style: { ...box(2, 10, 4, 4), "--udp-delay": "400ms" } },
    { shape: "dot", style: { ...box(10, 10, 4, 4), "--udp-delay": "800ms" } },
  ],
  // Two strokes converge into one (MOT-08).
  decision: [
    { shape: "bar", style: { ...box(2, 4, 12, 2), "--udp-shift": "3px" } },
    { shape: "bar", style: { ...box(2, 10, 12, 2), "--udp-shift": "-3px" } },
  ],
  // Three dimension bars rise, 150 ms apart (MOT-09).
  evaluator: [rising(2, 6, "0ms"), rising(6.5, 10, "150ms"), rising(11, 8, "300ms")],
};

/**
 * The running agent's glyph, in the status icon's 16 px box (MOT-02 to MOT-09): abstract, drawn
 * in `currentColor`, and hidden from assistive technology, because the status text says it all.
 */
export function StageGlyph({ agent }: { agent: PipelineStageId }) {
  return (
    <span data-glyph={agent} aria-hidden="true" className={styles.glyph}>
      {GLYPH_PARTS[agent].map((part, index) => (
        <span key={index} className={cn(styles.part, styles[part.shape])} style={part.style} />
      ))}
    </span>
  );
}
