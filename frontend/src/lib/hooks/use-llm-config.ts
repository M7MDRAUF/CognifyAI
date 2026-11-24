"use client";

import { useQuery } from "@tanstack/react-query";

import { getLlmConfig } from "@/lib/api/llm";
import type { LlmConfig } from "@/lib/types/llm";

/**
 * Three states, deliberately kept apart instead of collapsing onto `config === undefined`.
 *
 * "pending" and "error" both mean *we do not know how this deployment is configured* — they must
 * never be rendered as "nothing is configured", because the backend still decides what runs and
 * where the data goes. `retry: false` means a single failed lookup pins "error" for the life of
 * the page, so the copy chosen for it has to stay true indefinitely.
 */
export type CapabilityStatus = "pending" | "error" | "ready";

export interface LlmCapability {
  /** Only defined once the lookup has actually succeeded. */
  config: LlmConfig | undefined;
  status: CapabilityStatus;
}

/**
 * The shared `/llm/config` lookup. Chat and Documents both need it (each has copy that would
 * otherwise have to guess where data goes), and sharing one query key means one request.
 */
export function useLlmConfig(): LlmCapability {
  const query = useQuery({
    queryKey: ["llm-config"],
    queryFn: getLlmConfig,
    staleTime: Infinity,
    retry: false,
  });
  if (query.data !== undefined) {
    return { config: query.data, status: "ready" };
  }
  return { config: undefined, status: query.isError ? "error" : "pending" };
}
