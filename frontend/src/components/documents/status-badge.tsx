import { Badge } from "@/components/ui/badge";
import type { DocumentStatus } from "@/lib/types/document";

type BadgeVariant = "secondary" | "default" | "success" | "destructive" | "warning";

const VARIANT: Record<DocumentStatus, { variant: BadgeVariant; label: string }> = {
  uploaded: { variant: "secondary", label: "Uploaded" },
  processing: { variant: "warning", label: "Processing" },
  indexed: { variant: "success", label: "Indexed" },
  failed: { variant: "destructive", label: "Failed" },
};

export function StatusBadge({ status }: { status: DocumentStatus }) {
  const { variant, label } = VARIANT[status];
  return <Badge variant={variant}>{label}</Badge>;
}
