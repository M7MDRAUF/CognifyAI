import { Badge } from "@/components/ui/badge";
import type { DomainProfile } from "@/lib/types/pharma";

export const PROFILE_LABELS: Readonly<Record<DomainProfile, string>> = {
  general: "General",
  pharma_manufacturing: "Pharmaceutical manufacturing",
};

/**
 * A pharmaceutical project's profile, as text (REQ-UX-02). A General project shows no badge, so
 * every General screen renders exactly as before the profile existed.
 */
export function ProfileBadge({ profile }: { profile: DomainProfile }) {
  if (profile === "general") {
    return null;
  }
  return <Badge variant="secondary">{PROFILE_LABELS[profile]}</Badge>;
}
