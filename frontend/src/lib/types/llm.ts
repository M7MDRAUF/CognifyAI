import { z } from "@/lib/zod";

import { answerModelSchema } from "@/lib/types/query";

export const llmProviderSchema = z.string().min(1);

const PROVIDER_LABELS: Record<string, string> = {
  openai: "OpenAI",
  deepseek: "DeepSeek",
  nvidia: "NVIDIA NIM",
  glm: "GLM (Z.ai)",
  kimi: "Kimi (Moonshot)",
  fake: "Offline (fake provider)",
};

/** Falls back to the raw id so an unknown provider still renders. */
export function providerLabel(id: string): string {
  return PROVIDER_LABELS[id] ?? id;
}

export const unavailableReasonSchema = z
  .enum(["missing_api_key", "provider_disabled", "offline_mode", "unknown"])
  .catch("unknown");
export type UnavailableReason = z.infer<typeof unavailableReasonSchema>;

export const llmProviderInfoSchema = z.object({
  id: llmProviderSchema,
  label: z.string(),
  available: z.boolean(),
  unavailable_reason: unavailableReasonSchema.nullish(),
  is_default: z.boolean().default(false),
});
export type LlmProviderInfo = z.infer<typeof llmProviderInfoSchema>;

export const llmModelOptionSchema = z.object({
  tier: answerModelSchema,
  model_name: z.string(),
  description: z.string(),
});

/** One concrete, selectable provider+model pair. `id` is "<provider>:<model_id>". */
export const llmModelOptionEntrySchema = z.object({
  id: z.string().min(1),
  provider: llmProviderSchema,
  tier: answerModelSchema,
  model_name: z.string(),
  label: z.string(),
  description: z.string(),
  context_window_tokens: z.number().nullish(),
  available: z.boolean(),
  unavailable_reason: unavailableReasonSchema.nullish(),
});
export type LlmModelOptionEntry = z.infer<typeof llmModelOptionEntrySchema>;

export const llmConfigSchema = z.object({
  // --- v1: required, semantics unchanged ---
  provider: llmProviderSchema,
  /** False when no real external LLM API will be called (fake provider or no key). */
  configured: z.boolean(),
  default_tier: answerModelSchema,
  models: z.array(llmModelOptionSchema),
  // --- v2: defaulted, so a v1 backend response still parses ---
  contract_version: z.number().default(1),
  default_model_option_id: z.string().nullable().default(null),
  providers: z.array(llmProviderInfoSchema).default([]),
  model_options: z.array(llmModelOptionEntrySchema).default([]),
  // --- embedding destination: nullish, never defaulted ---
  //
  // Document text goes to a *second*, independent egress channel that `provider` says nothing
  // about, so no honest disclosure can be derived without these two fields.
  //
  // Three states, deliberately distinguishable — defaulting either key would invent a claim
  // the backend never made:
  //   undefined  the key is absent: an older backend that cannot tell us either way
  //   null       nothing is embedded remotely (the built-in offline provider)
  //   "openai"   that provider's API dialect receives the text
  /** Which provider embeds document text, or null when nothing is embedded remotely. */
  embedding_provider: llmProviderSchema.nullish(),
  /**
   * False when a base URL override is configured. The provider id is the *dialect*; a
   * self-hosted endpoint speaking that dialect is a different destination entirely, so the
   * vendor may only be named when this is explicitly `true`.
   */
  embedding_endpoint_is_default: z.boolean().nullish(),
});
export type LlmConfig = z.infer<typeof llmConfigSchema>;
