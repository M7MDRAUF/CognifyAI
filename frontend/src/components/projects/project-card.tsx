import Link from "next/link";

import { ProfileBadge } from "@/components/pharma/profile-badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { LocalTime } from "@/components/ui/local-time";
import { profileOf } from "@/lib/types/pharma";
import type { Project } from "@/lib/types/project";

import { DeleteProjectDialog } from "./delete-project-dialog";

export function ProjectCard({ project }: { project: Project }) {
  return (
    <Card className="relative flex flex-col transition-shadow hover:shadow-sm">
      {/*
       * Stretched link: the absolute overlay makes the entire card clickable
       * while keeping the delete button interactive above it via z-10.
       */}
      <Link
        href={`/projects/${project.id}`}
        aria-label={project.name}
        className="absolute inset-0 rounded-[inherit] focus-visible:outline-2 focus-visible:outline-offset-2"
      />
      <CardHeader>
        <CardTitle aria-hidden>{project.name}</CardTitle>
        <CardDescription>
          {project.domain === "" ? "No domain set" : project.domain}
        </CardDescription>
        {project.objective !== "" && (
          <p className="text-xs text-muted-foreground line-clamp-1">{project.objective}</p>
        )}
        {profileOf(project) !== "general" && (
          <div>
            <ProfileBadge profile={profileOf(project)} />
          </div>
        )}
      </CardHeader>
      <CardContent className="flex flex-1 flex-col justify-between gap-4">
        <p className="text-sm text-muted-foreground line-clamp-3">
          {project.description === "" ? "No description provided." : project.description}
        </p>
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">
            Created <LocalTime iso={project.created_at} />
          </span>
          <span className="relative z-10">
            <DeleteProjectDialog projectId={project.id} projectName={project.name} />
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
