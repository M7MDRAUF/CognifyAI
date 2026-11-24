import { z } from "@/lib/zod";

import { queryRunStatusSchema } from "@/lib/types/query";

export const qualityPointSchema = z.object({
  created_at: z.string(),
  answer_quality: z.number(),
});

export const stageLatencySchema = z.object({
  agent_name: z.string(),
  average_latency_ms: z.number(),
});

export const riskCategoryCountSchema = z.object({
  category: z.string(),
  count: z.number(),
});

export const hallucinationBucketSchema = z.object({
  label: z.string(),
  count: z.number(),
});

export const recentQuerySchema = z.object({
  query_run_id: z.uuid(),
  question: z.string(),
  status: queryRunStatusSchema,
  created_at: z.string(),
  answer_relevance: z.number().nullable(),
  faithfulness: z.number().nullable(),
  error_message: z.string().nullish(),
});

export const evaluationSummarySchema = z.object({
  query_count: z.number(),
  completed_count: z.number(),
  failed_count: z.number(),
  cancelled_count: z.number(),
  running_count: z.number(),
  average_answer_relevance: z.number().nullable(),
  average_faithfulness: z.number().nullable(),
  average_citation_accuracy: z.number().nullable(),
  average_context_sufficiency: z.number().nullable(),
  average_latency_ms: z.number().nullable(),
  average_token_usage: z.number().nullable(),
  retrieval_hit_count: z.number(),
  agent_failure_count: z.number(),
  quality_over_time: z.array(qualityPointSchema),
  stage_latencies: z.array(stageLatencySchema),
  risk_categories: z.array(riskCategoryCountSchema),
  hallucination_risk_distribution: z.array(hallucinationBucketSchema),
  recent_queries: z.array(recentQuerySchema),
});

export type EvaluationSummary = z.infer<typeof evaluationSummarySchema>;
export type QualityPoint = z.infer<typeof qualityPointSchema>;
export type StageLatency = z.infer<typeof stageLatencySchema>;
export type RiskCategoryCount = z.infer<typeof riskCategoryCountSchema>;
export type RecentQuery = z.infer<typeof recentQuerySchema>;
