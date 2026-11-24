import type { Metadata } from "next";

import { ReportsView } from "@/components/reports/reports-view";

export const metadata: Metadata = { title: "Reports" };

export default async function ReportsPage({ params }: { params: Promise<{ projectId: string }> }) {
  const { projectId } = await params;
  return <ReportsView projectId={projectId} />;
}
