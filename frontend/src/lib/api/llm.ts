import { requestJson } from "@/lib/api/client";
import { type LlmConfig, llmConfigSchema } from "@/lib/types/llm";

/**
 * Reports which answer models are selectable and whether selection is live.
 *
 * Deliberately not `/health`: that endpoint returns 503 when a dependency is
 * degraded, which `requestJson` turns into a throw. This endpoint always
 * returns 200 so the UI can render a disabled control instead of an error.
 */
export function getLlmConfig(): Promise<LlmConfig> {
  return requestJson("/llm/config", llmConfigSchema);
}
