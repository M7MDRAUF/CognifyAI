import Link from "next/link";

export default function ProjectNotFound() {
  return (
    <div className="flex flex-col items-center justify-center gap-6 py-24 text-center">
      {/* Rendered by the not-found boundary and by the client guard alike (FE-034). */}
      <title>Project not found · CognifyAI</title>
      <h1 className="text-3xl font-bold tracking-tight">Project not found</h1>
      <p className="text-muted-foreground">This project does not exist or may have been deleted.</p>
      <Link
        href="/projects"
        className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 focus-visible:outline-2 focus-visible:outline-offset-2"
      >
        Back to projects
      </Link>
    </div>
  );
}
