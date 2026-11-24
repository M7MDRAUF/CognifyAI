import { controlledMetadataSchema } from "@/lib/types/pharma";
import { z } from "@/lib/zod";

export const documentStatusSchema = z.enum(["uploaded", "processing", "indexed", "failed"]);

export type DocumentStatus = z.infer<typeof documentStatusSchema>;

export const documentSchema = z.object({
  id: z.uuid(),
  project_id: z.uuid(),
  filename: z.string(),
  original_filename: z.string(),
  mime_type: z.string(),
  file_size_bytes: z.number(),
  status: documentStatusSchema,
  error_message: z.string().nullable(),
  created_at: z.string(),
  updated_at: z.string(),
  // Pharmaceutical projects only (REQ-DOC-08); General documents send null or nothing.
  controlled_metadata: controlledMetadataSchema.nullish(),
});

export type ProjectDocument = z.infer<typeof documentSchema>;

export const documentListSchema = z.array(documentSchema);

export const documentDetailSchema = documentSchema.extend({
  chunk_count: z.number(),
});

export type ProjectDocumentDetail = z.infer<typeof documentDetailSchema>;
