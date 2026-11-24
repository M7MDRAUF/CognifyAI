import { z } from "@/lib/zod";

export const reportSchema = z.object({
  id: z.uuid(),
  query_run_id: z.uuid(),
  title: z.string(),
  markdown_content: z.string(),
  created_at: z.string(),
});

export type Report = z.infer<typeof reportSchema>;

/** One row of `GET /projects/{id}/reports` (CT-01): the list never carries the Markdown body. */
export const reportSummarySchema = z.object({
  id: z.uuid(),
  query_run_id: z.uuid(),
  title: z.string(),
  created_at: z.string(),
});
export type ReportSummary = z.infer<typeof reportSummarySchema>;

/** A page of report summaries, newest first; `next_cursor` is null on the last page. */
export const reportPageSchema = z.object({
  items: z.array(reportSummarySchema),
  next_cursor: z.string().nullable(),
  total: z.number(),
});
export type ReportPage = z.infer<typeof reportPageSchema>;
