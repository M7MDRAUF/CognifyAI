import { type DomainProfile, domainProfileSchema } from "@/lib/types/pharma";
import { z } from "@/lib/zod";

export const projectSchema = z.object({
  id: z.uuid(),
  name: z.string(),
  description: z.string(),
  domain: z.string(),
  objective: z.string(),
  created_at: z.string(),
  updated_at: z.string(),
  // Optional: a backend that predates the profile sends none, which reads as General (`profileOf`).
  domain_profile: domainProfileSchema.optional(),
});

export type Project = z.infer<typeof projectSchema>;

export const projectListSchema = z.array(projectSchema);

export const projectStatsSchema = z.object({
  document_count: z.number(),
  chunk_count: z.number(),
  vector_count: z.number(),
  last_query_at: z.string().nullable(),
  average_answer_quality: z.number().nullable(),
  average_citation_accuracy: z.number().nullable(),
  average_latency_ms: z.number().nullable(),
});

export const projectDetailSchema = projectSchema.extend({
  stats: projectStatsSchema,
});

export type ProjectDetail = z.infer<typeof projectDetailSchema>;

export interface ProjectCreateInput {
  name: string;
  description: string;
  domain: string;
  objective: string;
  /** Fixed at creation. Omitted means General, the server's default. */
  domain_profile?: DomainProfile;
}
