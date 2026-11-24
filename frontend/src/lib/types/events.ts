import { z } from "@/lib/zod";

import { agentNameSchema } from "@/lib/types/query";

export const eventTypeSchema = z.enum([
  "planning_started",
  "retrieval_started",
  "retrieval_completed",
  "reranking_started",
  "research_started",
  "critic_started",
  "risk_started",
  "answer_started",
  "evaluation_started",
  "answer_completed",
  "workflow_failed",
  "workflow_cancelled",
]);
export type EventType = z.infer<typeof eventTypeSchema>;

export const TERMINAL_EVENT_TYPES: ReadonlySet<EventType> = new Set<EventType>([
  "answer_completed",
  "workflow_failed",
  "workflow_cancelled",
]);

export const agentEventSchema = z.object({
  id: z.uuid(),
  query_run_id: z.uuid(),
  event_type: eventTypeSchema,
  agent_name: agentNameSchema.nullable(),
  // SSE payload v2 (C-13, ADR-007): the producer of events that have no agent. A closed set on
  // the backend, but kept a plain string here so a new producer is additive, never a parse error.
  component: z.string().nullish(),
  message: z.string(),
  payload: z.record(z.string(), z.unknown()),
  created_at: z.string(),
});
export type AgentEvent = z.infer<typeof agentEventSchema>;
