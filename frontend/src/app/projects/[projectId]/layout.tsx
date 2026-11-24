import Link from "next/link";
import { notFound } from "next/navigation";

import { ProjectNav } from "@/components/layout/project-nav";
import { ProjectGuard } from "@/components/projects/project-overview";
import { isUuid } from "@/lib/utils/validation";

export default async function ProjectLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: Promise<{ projectId: string }>;
}) {
  const { projectId } = await params;
  // A malformed id can never name a project: answer it on the server, with a real 404, before
  // any view mounts or any request carries it (FE-009). `notFound()` thrown in this layout is
  // caught by the parent segment's boundary, `app/projects/not-found.tsx`.
  if (!isUuid(projectId)) {
    notFound();
  }
  return (
    <div className="mx-auto w-full max-w-5xl flex-1 px-6 py-8">
      <Link href="/projects" className="text-sm text-muted-foreground hover:underline">
        ← All projects
      </Link>
      <div className="mt-4">
        <ProjectNav projectId={projectId} />
      </div>
      <div className="mt-6">
        <ProjectGuard projectId={projectId}>{children}</ProjectGuard>
      </div>
    </div>
  );
}
