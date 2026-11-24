"use client";

import { useQuery } from "@tanstack/react-query";

import { CreateProjectDialog } from "@/components/projects/create-project-dialog";
import { ProjectCard } from "@/components/projects/project-card";
import { Button } from "@/components/ui/button";
import { StatusAnnouncer } from "@/components/ui/status-announcer";
import { listProjects } from "@/lib/api/projects";

export function ProjectsView() {
  const { data, isLoading, isError, error, isFetching, refetch } = useQuery({
    queryKey: ["projects"],
    queryFn: listProjects,
  });

  return (
    <div className="mx-auto w-full max-w-5xl flex-1 px-6 py-12">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Projects</h1>
          <p className="text-muted-foreground">Each project holds its own documents and queries.</p>
        </div>
        <CreateProjectDialog />
      </div>
      {isLoading && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <div key={i} className="animate-pulse rounded-lg border bg-muted h-40" />
          ))}
        </div>
      )}
      {/* RE2E-DEF-016: the message says to try again, so the page offers the means. */}
      {isError && (
        <div role="alert" className="space-y-3">
          <p className="text-destructive">
            {error instanceof Error ? error.message : "Failed to load projects."}
          </p>
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={isFetching}
            onClick={() => {
              void refetch();
            }}
          >
            {isFetching ? "Retrying…" : "Try again"}
          </Button>
        </div>
      )}
      {data?.length === 0 && (
        <p className="text-muted-foreground">No projects yet. Create your first one.</p>
      )}
      {data !== undefined && data.length > 0 && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {data.map((project) => (
            <ProjectCard key={project.id} project={project} />
          ))}
        </div>
      )}
      {/* Where "Project created." and "Project deleted." are said (RE2E-DEF-017). Mounted by the view
          whose actions speak, not the layout, so the chat page keeps its one status (DV-RMD-77). */}
      <StatusAnnouncer />
    </div>
  );
}
