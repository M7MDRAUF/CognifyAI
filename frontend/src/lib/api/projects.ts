import { jsonInit, requestJson, requestVoid } from "@/lib/api/client";
import {
  type Project,
  type ProjectCreateInput,
  type ProjectDetail,
  projectDetailSchema,
  projectListSchema,
  projectSchema,
} from "@/lib/types/project";

export function listProjects(): Promise<Project[]> {
  return requestJson("/projects", projectListSchema);
}

/**
 * Sends `domain_profile` only for a non-General profile (REQ-UX-01): a General project is created
 * with exactly the request a client sent before the profile existed, and the server's default
 * makes it General.
 */
export function createProject(input: ProjectCreateInput): Promise<Project> {
  const { domain_profile, ...general } = input;
  const body = domain_profile === undefined || domain_profile === "general" ? general : input;
  return requestJson("/projects", projectSchema, jsonInit("POST", body));
}

export function getProject(projectId: string): Promise<ProjectDetail> {
  return requestJson(`/projects/${projectId}`, projectDetailSchema);
}

export function deleteProject(projectId: string): Promise<void> {
  return requestVoid(`/projects/${projectId}`, { method: "DELETE" });
}
