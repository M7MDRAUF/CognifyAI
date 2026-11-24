import { type PharmaQueryInput, pharmaRunSchema } from "@/lib/types/pharma";
import { z } from "@/lib/zod";

export const retrievalModeSchema = z.enum(["hybrid", "semantic", "lexical"]);
export type RetrievalMode = z.infer<typeof retrievalModeSchema>;

export const responseStyleSchema = z.enum(["concise", "detailed"]);
export type ResponseStyle = z.infer<typeof responseStyleSchema>;

/** Which configured text model synthesises the answer. */
export const answerModelSchema = z.enum(["fast", "powerful"]);
export type AnswerModel = z.infer<typeof answerModelSchema>;

/**
 * The certified routing profiles (C-16, SDD 10.3.2, `backend/app/integrations/llm/profiles.py`).
 * Each maps all seven model call sites to one tier.
 */
export const routingProfileSchema = z.enum(["flash-all", "balanced", "powerful-synthesis"]);
export type RoutingProfile = z.infer<typeof routingProfileSchema>;

/**
 * SDD 10.3.4: with no profile named, the server routes by the `answer_model` alias, and a model
 * option's tier is written back into `answer_model` first. So the profile a request that names
 * none will run on is this function of the effective tier.
 */
export const ROUTING_PROFILE_FOR_TIER: Readonly<Record<AnswerModel, RoutingProfile>> = {
  fast: "flash-all",
  powerful: "balanced",
};

export const queryRunStatusSchema = z.enum(["running", "completed", "failed", "cancelled"]);
export type QueryRunStatus = z.infer<typeof queryRunStatusSchema>;

export const agentNameSchema = z.enum([
  "planner",
  "retriever",
  "researcher",
  "critic",
  "risk",
  "evaluator",
  "final_decision",
]);
export type AgentName = z.infer<typeof agentNameSchema>;

export const queryRunSchema = z.object({
  id: z.uuid(),
  project_id: z.uuid(),
  question: z.string(),
  normalized_question: z.string(),
  retrieval_mode: retrievalModeSchema,
  response_style: responseStyleSchema,
  answer_model: answerModelSchema,
  status: queryRunStatusSchema,
  answer_markdown: z.string().nullable(),
  error_message: z.string().nullable(),
  latency_ms: z.number(),
  created_at: z.string(),
  /** Which provider actually answered. Null for runs recorded before multi-provider support. */
  resolved_provider: z.string().nullish(),
  resolved_model_name: z.string().nullish(),
});
export type QueryRun = z.infer<typeof queryRunSchema>;

/**
 * The 202 of `POST /queries/{id}/cancel` (RMD5-T03, RE2E-DEF-037): the run as read when the cancel
 * was accepted, and `cancel_requested: true` — a cancel is pending even while `status` still reads
 * `running`. The backend cancels this process's task before it reads the database, so during an
 * outage the body names only `id`, `status` and `cancel_requested`; every other field may be absent.
 * `cancel_requested` itself is absent from a backend that predates it, which still parses.
 */
export const queryRunCancelSchema = queryRunSchema.partial().extend({
  id: z.uuid(),
  status: queryRunStatusSchema,
  cancel_requested: z.boolean().optional(),
});
export type QueryRunCancel = z.infer<typeof queryRunCancelSchema>;

/**
 * One row of `GET /projects/{id}/queries/summaries` (CT-02): what a run list shows, without the
 * answer or the other detail fields the legacy list carried.
 */
export const queryRunSummarySchema = z.object({
  id: z.uuid(),
  project_id: z.uuid(),
  question: z.string(),
  status: queryRunStatusSchema,
  answer_model: answerModelSchema,
  resolved_provider: z.string().nullish(),
  resolved_model_name: z.string().nullish(),
  error_message: z.string().nullable(),
  created_at: z.string(),
});
export type QueryRunSummary = z.infer<typeof queryRunSummarySchema>;

/** A page of run summaries, newest first; `next_cursor` is null on the last page. */
export const queryRunSummaryPageSchema = z.object({
  items: z.array(queryRunSummarySchema),
  next_cursor: z.string().nullable(),
  total: z.number(),
});
export type QueryRunSummaryPage = z.infer<typeof queryRunSummaryPageSchema>;

export const citationSchema = z.object({
  citation_key: z.string(),
  chunk_id: z.uuid(),
  document_id: z.uuid(),
  source_filename: z.string(),
  snippet: z.string(),
  page_number: z.number().nullable(),
  relevance_score: z.number(),
});
export type Citation = z.infer<typeof citationSchema>;

export const agentRunSchema = z.object({
  agent_name: agentNameSchema,
  status: z.enum(["pending", "running", "completed", "failed"]),
  latency_ms: z.number().nullable(),
  error_message: z.string().nullable(),
  // AgentRunRead v2 (M-6, M-9, L-6). Optional so a backend that predates them still parses.
  llm_provider: z.string().nullish(),
  llm_model: z.string().nullish(),
  attempt_count: z.number().default(1),
  error_classification: z.string().nullish(),
});
export type AgentRun = z.infer<typeof agentRunSchema>;

export const evaluationSchema = z.object({
  answer_relevance: z.number(),
  faithfulness: z.number(),
  citation_accuracy: z.number(),
  context_sufficiency: z.number(),
  hallucination_risk: z.number(),
  explanation: z.string(),
});
export type Evaluation = z.infer<typeof evaluationSchema>;

/**
 * One Risk agent item (FR-010, PRD FR-CHAT-003), its prose concealed by the backend. The enums
 * are read as strings, so a category or level the backend adds later still renders, as words.
 */
export const riskItemSchema = z.object({
  category: z.string(),
  description: z.string(),
  severity: z.string(),
  likelihood: z.string(),
  mitigation: z.string(),
});
export type RiskItem = z.infer<typeof riskItemSchema>;

export const queryRunDetailSchema = queryRunSchema.extend({
  citations: z.array(citationSchema),
  agent_runs: z.array(agentRunSchema),
  evaluation: evaluationSchema.nullable(),
  // C-15 / D-7. `null` means the decision step produced no readable output, which is not the
  // same as an agent that stated no uncertainty (""); keep the two apart.
  uncertainty: z.string().nullish(),
  next_steps: z.array(z.string()).default([]),
  // FR-010: shown beside the answer. A backend that predates them sends neither, and an empty
  // list is what the answer screen then says it has.
  risks: z.array(riskItemSchema).default([]),
  contradictory_evidence: z.array(z.string()).default([]),
  // RMD8-T01 (RE2E-DEF-031): the citation markers removed because they named no returned source.
  // A backend that predates the field sends nothing, which the answer view reads as none.
  unresolved_citation_markers: z.number().int().nonnegative().optional(),
  // RMD8-T02 (RE2E-DEF-043, RE2E-DEF-044): the evidence context lines and quoted planted
  // instructions left out of the answer, whose own notice says so. Optional, like the field above.
  hygiene_removals: z.number().int().nonnegative().optional(),
  // Pharmaceutical runs only (REQ-API-04): the policy-checked nine-section response. General runs
  // send null, and a backend that predates the profile sends nothing.
  pharma: pharmaRunSchema.nullish(),
});
export type QueryRunDetail = z.infer<typeof queryRunDetailSchema>;

export interface QueryCreateInput {
  question: string;
  retrieval_mode: RetrievalMode;
  response_style: ResponseStyle;
  answer_model: AnswerModel;
  /** Explicit "<provider>:<model_id>" selection. Omitted entirely when the caller has no preference. */
  model_option_id?: string;
  /**
   * A named routing profile. When sent, `answer_model` must be omitted or equal the profile's tier
   * (400 `invalid_routing_profile` otherwise, F-011), and a deployment with the rollout gate off
   * refuses it (F-009). Under OD-01 option B the form never sends it: the server derives the
   * profile from the model's tier (`ROUTING_PROFILE_FOR_TIER`) and the form shows that profile.
   */
  routing_profile?: RoutingProfile;
  /** Intent, incident and filters of a question on a pharmaceutical project; never sent otherwise. */
  pharma?: PharmaQueryInput;
}
