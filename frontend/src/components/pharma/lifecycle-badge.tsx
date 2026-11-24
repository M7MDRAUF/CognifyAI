import {
  Archive,
  CircleCheck,
  CircleQuestionMark,
  FileClock,
  type LucideIcon,
  PencilLine,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import type { LifecycleStatus } from "@/lib/types/pharma";

type BadgeVariant = "secondary" | "success" | "warning" | "outline";

export const LIFECYCLE_LABELS: Readonly<Record<LifecycleStatus, string>> = {
  draft: "Draft",
  effective: "Effective",
  superseded: "Superseded",
  retired: "Retired",
  unknown: "Unknown",
};

const STYLE: Readonly<Record<LifecycleStatus, { variant: BadgeVariant; icon: LucideIcon }>> = {
  draft: { variant: "warning", icon: PencilLine },
  effective: { variant: "success", icon: CircleCheck },
  superseded: { variant: "secondary", icon: FileClock },
  retired: { variant: "outline", icon: Archive },
  unknown: { variant: "outline", icon: CircleQuestionMark },
};

/**
 * A controlled document's lifecycle status (REQ-UX-03). The word carries the meaning and the icon
 * repeats it; the existing badge colours only reinforce both, so no status is told by colour alone.
 */
export function LifecycleBadge({ status }: { status: LifecycleStatus }) {
  const { variant, icon: Icon } = STYLE[status];
  return (
    <Badge variant={variant} data-lifecycle={status} className="gap-1 whitespace-nowrap">
      <Icon aria-hidden="true" className="size-3.5" />
      {LIFECYCLE_LABELS[status]}
    </Badge>
  );
}
