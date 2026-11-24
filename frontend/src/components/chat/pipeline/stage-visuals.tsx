import {
  ArrowDownUp,
  BookOpenText,
  Circle,
  CircleCheck,
  CircleDashed,
  CircleSlash,
  CircleX,
  Gauge,
  ListChecks,
  type LucideIcon,
  OctagonX,
  Scale,
  Search,
  SearchCheck,
  ShieldAlert,
  TriangleAlert,
} from "lucide-react";

import type { PipelineStage, PipelineStageId, StageStatus } from "@/lib/chat/pipeline";
import { cn } from "@/lib/utils";

/** Each agent's icon (plan section 9): its identity in the tab, never its status. */
export const STAGE_ICONS: Readonly<Record<PipelineStageId, LucideIcon>> = {
  planner: ListChecks,
  retriever: Search,
  reranker: ArrowDownUp,
  researcher: BookOpenText,
  critic: SearchCheck,
  risk: ShieldAlert,
  decision: Scale,
  evaluator: Gauge,
};

interface StatusVisual {
  /** The status in words; it is always written, never carried by colour or motion alone. */
  label: string;
  /** The status icon; `null` for a running stage, which shows its agent's icon instead. */
  icon: LucideIcon | null;
  /** The status line's colour, from plan section 7's status roles only. */
  tone: string;
  /** The tab's border and surface. */
  border: string;
}

/**
 * Every normalised status's label, icon and tone (plan sections 7 and 9). The colours are the
 * product's own: the tokens, and the badge variants' `emerald-700` and `amber-*` (UDP0-T02's
 * contract test holds this file to them).
 */
export const STATUS_VISUALS: Readonly<Record<StageStatus, StatusVisual>> = {
  pending: {
    label: "Waiting",
    icon: Circle,
    tone: "text-muted-foreground",
    border: "border-border",
  },
  running: {
    label: "Running",
    icon: null,
    tone: "text-foreground",
    border: "border-primary",
  },
  completed: {
    label: "Completed",
    icon: CircleCheck,
    tone: "text-emerald-700",
    border: "border-border",
  },
  degraded: {
    label: "Unavailable — the answer stands",
    icon: TriangleAlert,
    tone: "text-amber-700",
    border: "border-amber-500/40 bg-amber-500/10",
  },
  failed: {
    label: "Failed",
    icon: CircleX,
    tone: "text-red-700",
    border: "border-destructive",
  },
  interrupted: {
    label: "Did not finish",
    icon: OctagonX,
    tone: "text-red-700",
    border: "border-destructive",
  },
  cancelled: {
    label: "Cancelled",
    icon: CircleSlash,
    tone: "text-muted-foreground",
    border: "border-dashed border-border",
  },
  not_run: {
    label: "Not run",
    icon: CircleDashed,
    tone: "text-muted-foreground",
    border: "border-border",
  },
};

/**
 * The stage's status line: the label, with the recorded attempts when there was more than one
 * (AST-23) and a held stop request on the running stage (AST-34).
 */
export function stageStatusText(stage: PipelineStage): string {
  const { label } = STATUS_VISUALS[stage.status];
  if (stage.status === "running") {
    return stage.stopRequested ? `${label} · Stop requested` : label;
  }
  if (stage.attempts > 1) {
    const attempts = String(stage.attempts);
    return stage.status === "failed"
      ? `Failed after ${attempts} attempts`
      : `${label} · ${attempts} attempts`;
  }
  return label;
}

/** The status icon in its 16 px box, hidden from assistive technology: the text says it all. */
export function StatusIcon({ stage, className }: { stage: PipelineStage; className?: string }) {
  const Icon = STATUS_VISUALS[stage.status].icon ?? STAGE_ICONS[stage.id];
  return <Icon className={cn("size-4 shrink-0", className)} aria-hidden />;
}
