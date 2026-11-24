"use client";

import { useRef, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import type { CapabilityStatus } from "@/lib/hooks/use-llm-config";
import { usePersistedModelOption } from "@/lib/hooks/use-persisted-model-option";
import {
  type LlmConfig,
  type LlmModelOptionEntry,
  type LlmProviderInfo,
  type UnavailableReason,
  providerLabel,
} from "@/lib/types/llm";
import {
  type AnswerModel,
  type QueryCreateInput,
  ROUTING_PROFILE_FOR_TIER,
  type ResponseStyle,
  type RetrievalMode,
  type RoutingProfile,
} from "@/lib/types/query";

/**
 * The backend's `MAX_QUESTION_LENGTH` (`backend/app/schemas/queries.py`). Enforced here too, so
 * the limit is met while typing instead of coming back as a validation error after a round trip.
 */
const MAX_QUESTION_LENGTH = 4000;

const RETRIEVAL_MODES: RetrievalMode[] = ["hybrid", "semantic", "lexical"];
const RESPONSE_STYLES: ResponseStyle[] = ["detailed", "concise"];
// Used only while llmConfig has not resolved at all (still loading, or the capability lookup
// failed). The select is disabled in that state regardless, so these values are never actually
// choosable — they just keep the disabled control non-empty. Once llmConfig resolves, Mode B's
// options always come from llmConfig.models so a v1 backend publishing a single tier cannot
// offer a phantom one the user could select and submit.
const FALLBACK_MODE_B_TIERS: AnswerModel[] = ["powerful", "fast"];

/**
 * The provider id of the built-in offline provider (`LLM_PROVIDER=fake`) — the one configuration
 * in which answers really are produced locally, because the router returns the fake provider for
 * chat and for embeddings alike. Anywhere else, promising an offline provider would be a claim
 * nothing enforces (ORCH-1).
 */
const OFFLINE_PROVIDER_ID = "fake";

const OFFLINE_PROVIDER_NOTICE =
  "No external LLM API is configured, so every answer uses the built-in offline provider. " +
  "Configure a provider API key to choose a model.";

const NO_PROVIDER_WARNING =
  "No LLM provider is currently available, so questions are disabled. Ask an administrator to " +
  "configure a provider API key.";

const NO_AVAILABLE_MODELS_WARNING =
  "The configured providers have no available models right now (missing API keys or disabled), " +
  "so questions are disabled until one becomes available.";

const CAPABILITY_PENDING_NOTICE = "Checking which models are available…";

/**
 * Every clause here is hedged on purpose. The commonest cause of this state is the backend
 * being unreachable, in which case *nothing* answers — so the round-1 copy's promise that "the
 * backend's default provider will answer" was an assertion that is false in precisely the
 * situation it was written for. The lookup is never retried (`use-llm-config.ts`), so whatever
 * this says has to stay true for the life of the page.
 */
const CAPABILITY_UNKNOWN_WARNING =
  "Could not determine which models are configured. Asking may still work — the backend " +
  "decides, and it may use an external API — or it may fail.";

const RETRIEVAL_DESCRIPTIONS: Record<RetrievalMode, string> = {
  hybrid: "Combines vector similarity and keyword search (recommended)",
  semantic: "Vector similarity only — best for conceptual questions",
  lexical: "Keyword search only — best for exact terms",
};

/**
 * What each certified profile means for cost and quality (SDD 10.3.2). OD-01 option B: the profile
 * is derived from the chosen model's tier and shown, not chosen separately.
 */
const ROUTING_PROFILE_DESCRIPTIONS: Record<RoutingProfile, string> = {
  "flash-all": "every step uses the fast model, the cheapest and quickest routing",
  balanced:
    "evidence synthesis, critique, risk and the decision use the stronger model; planning, reranking and scoring use the fast one",
  "powerful-synthesis":
    "as balanced, but the answer is also scored by the stronger model; the highest cost",
};

const STYLE_DESCRIPTIONS: Record<ResponseStyle, string> = {
  detailed: "Full analysis with supporting evidence",
  concise: "Brief recommendation with key points only",
};

/** The API sends machine-readable codes; this component owns the human wording. */
function unavailableReasonLabel(reason: UnavailableReason | null | undefined): string {
  switch (reason) {
    case "missing_api_key":
      return "needs an API key";
    case "provider_disabled":
      return "disabled";
    case "offline_mode":
      return "offline mode only";
    default:
      return "unavailable";
  }
}

/**
 * Prefers the label the backend published for this provider (`providers[]`), so a sixth
 * provider's optgroup heading is correct the moment the backend ships it, with no frontend
 * deploy required to teach it a new id. Falls back to the local map (`providerLabel`) for a
 * v1 backend that sends no `providers[]` at all, and finally to the raw id.
 */
function resolveProviderLabel(providerId: string, providers: readonly LlmProviderInfo[]): string {
  const published = providers.find((provider) => provider.id === providerId)?.label;
  return published ?? providerLabel(providerId);
}

/**
 * Picks the entry to preselect, in order: the persisted choice (if still available), the
 * backend's `default_model_option_id` (if it names an available entry), the entry matching
 * `default_tier`, then simply the first available entry.
 */
function resolveDefaultOptionId(
  availableEntries: readonly LlmModelOptionEntry[],
  persistedOptionId: string | null,
  defaultModelOptionId: string | null,
  defaultTier: AnswerModel,
): string | null {
  if (
    persistedOptionId !== null &&
    availableEntries.some((entry) => entry.id === persistedOptionId)
  ) {
    return persistedOptionId;
  }
  if (
    defaultModelOptionId !== null &&
    availableEntries.some((entry) => entry.id === defaultModelOptionId)
  ) {
    return defaultModelOptionId;
  }
  const tierMatch = availableEntries.find((entry) => entry.tier === defaultTier);
  if (tierMatch !== undefined) {
    return tierMatch.id;
  }
  return availableEntries[0]?.id ?? null;
}

/**
 * The copy under a locked Model control. Every distinguishable state gets its own sentence:
 * collapsing them is how the form came to tell users their answers were offline whenever the
 * capability lookup merely hiccuped (A16-3).
 */
function lockedModelMessage(
  llmConfig: LlmConfig | undefined,
  capabilityStatus: CapabilityStatus,
  noAvailableModels: boolean,
): string {
  if (llmConfig === undefined) {
    return capabilityStatus === "error" ? CAPABILITY_UNKNOWN_WARNING : CAPABILITY_PENDING_NOTICE;
  }
  if (llmConfig.provider === OFFLINE_PROVIDER_ID) {
    return OFFLINE_PROVIDER_NOTICE;
  }
  return noAvailableModels ? NO_AVAILABLE_MODELS_WARNING : NO_PROVIDER_WARNING;
}

function groupByProvider(
  entries: readonly LlmModelOptionEntry[],
): Map<string, LlmModelOptionEntry[]> {
  const groups = new Map<string, LlmModelOptionEntry[]>();
  for (const entry of entries) {
    const group = groups.get(entry.provider);
    if (group === undefined) {
      groups.set(entry.provider, [entry]);
    } else {
      group.push(entry);
    }
  }
  return groups;
}

interface QuestionFormProps {
  disabled: boolean;
  onSubmit: (input: QueryCreateInput) => void;
  /** Undefined while the capability request is in flight or has failed. */
  llmConfig?: LlmConfig;
  /**
   * Tells "still loading" apart from "the lookup failed"; both leave `llmConfig` undefined, and
   * neither means "nothing is configured". Defaults to "pending" so a caller that has not
   * resolved the lookup at all cannot accidentally assert anything.
   */
  capabilityStatus?: CapabilityStatus;
}

export function QuestionForm({
  disabled,
  onSubmit,
  llmConfig,
  capabilityStatus = "pending",
}: QuestionFormProps) {
  const [question, setQuestion] = useState("");
  const [retrievalMode, setRetrievalMode] = useState<RetrievalMode>("hybrid");
  const [responseStyle, setResponseStyle] = useState<ResponseStyle>("detailed");
  // Mode B only (no model_options): the user's explicit tier choice, if any.
  const [selectedTier, setSelectedTier] = useState<AnswerModel | null>(null);
  const [error, setError] = useState<string | null>(null);
  const formRef = useRef<HTMLFormElement>(null);

  // Mode A when the backend sends concrete provider+model entries; Mode B (the v1 shape) when
  // it doesn't — both must keep working, see task-6-lane-b.md "Two rendering modes".
  const modelOptions = llmConfig?.model_options ?? [];
  const hasModelOptions = modelOptions.length > 0;
  const availableEntries = modelOptions.filter((entry) => entry.available);
  const availableOptionIds = availableEntries.map((entry) => entry.id);

  // Called unconditionally regardless of mode (rules of hooks); Mode B passes an empty id
  // list, so the persisted value it returns is always null there and never written to.
  const [persistedOptionId, setPersistedOptionId] = usePersistedModelOption(availableOptionIds);

  // Fail closed: if we cannot confirm a live LLM API, the control is shown but locked to the
  // backend's default rather than offering a choice that has no effect.
  const fallbackTier = llmConfig?.default_tier ?? "powerful";
  const configuredOk = llmConfig?.configured === true;
  const modelSelectable = hasModelOptions
    ? configuredOk && availableEntries.length > 0
    : configuredOk;
  // Distinguishes "no provider is configured at all" from "providers are configured but none
  // currently have an available model" — the select renders blank in both cases, so the copy
  // must not claim nothing is configured when something is, just unavailable.
  const noAvailableModels = hasModelOptions && configuredOk && availableEntries.length === 0;
  const capabilityPending = llmConfig === undefined && capabilityStatus !== "error";
  // The backend published something, and none of it can answer. Submitting anyway would send a
  // question the UI cannot describe the destination of, so the form stops here instead. The
  // built-in offline provider is excluded: it is reported as unconfigured yet does answer.
  const nothingCanAnswer =
    llmConfig !== undefined &&
    llmConfig.provider !== OFFLINE_PROVIDER_ID &&
    (!configuredOk || noAvailableModels);
  // RE2E-DEF-011: until the lookup answers, the privacy notice above has said nothing, so a
  // question asked now would leave before the user is told where it goes. A failed lookup
  // releases Ask: the notice hedges in that state and the lookup is never retried.
  const askBlocked = disabled || nothingCanAnswer || capabilityPending;

  const modeBTiers: AnswerModel[] =
    llmConfig !== undefined ? llmConfig.models.map((option) => option.tier) : FALLBACK_MODE_B_TIERS;

  const selectedOptionId = modelSelectable
    ? resolveDefaultOptionId(
        availableEntries,
        persistedOptionId,
        llmConfig?.default_model_option_id ?? null,
        fallbackTier,
      )
    : null;
  const selectedEntry = availableEntries.find((entry) => entry.id === selectedOptionId);
  // In the built-in offline mode the control is locked, and it must still show the model that
  // will answer: the backend's default, not whichever option happens to be listed first (FE-041).
  // Display only; the request is unchanged and the backend still decides.
  const defaultOptionId = llmConfig?.default_model_option_id ?? null;
  const lockedOptionId =
    !modelSelectable &&
    llmConfig?.provider === OFFLINE_PROVIDER_ID &&
    defaultOptionId !== null &&
    modelOptions.some((entry) => entry.id === defaultOptionId)
      ? defaultOptionId
      : null;

  const effectiveAnswerModel: AnswerModel = hasModelOptions
    ? (selectedEntry?.tier ?? fallbackTier)
    : modelSelectable
      ? (selectedTier ?? fallbackTier)
      : fallbackTier;

  const effectiveProfile = ROUTING_PROFILE_FOR_TIER[effectiveAnswerModel];

  const modelDescriptions = new Map(
    (llmConfig?.models ?? []).map((option) => [
      option.tier,
      `${option.model_name} — ${option.description}`,
    ]),
  );
  const selectedDescription = hasModelOptions
    ? selectedEntry === undefined
      ? undefined
      : `${selectedEntry.model_name} — ${selectedEntry.description}`
    : modelDescriptions.get(effectiveAnswerModel);

  const providerGroups = groupByProvider(modelOptions);

  const handleSubmit = (event: React.SyntheticEvent) => {
    event.preventDefault();
    // The gate the Ask button advertises, enforced on every path that reaches a submit.
    // A disabled `<button type="submit">` blocks the click and nothing else: Ctrl+Enter called
    // this function directly, so the form submitted a question while rendering "questions are
    // disabled" (V6 §4) and could queue a second run mid-flight. Enforcing it here rather than
    // in the key handler leaves one copy of the rule, and covers any future submit path.
    if (askBlocked) {
      return;
    }
    if (question.trim() === "") {
      setError("Enter a question to analyze.");
      // FE-035: the empty field takes focus, so the error is read where it can be fixed.
      const field = formRef.current?.elements.namedItem("question");
      if (field instanceof HTMLTextAreaElement) {
        field.focus();
      }
      return;
    }
    setError(null);
    const payload: QueryCreateInput = {
      question: question.trim(),
      retrieval_mode: retrievalMode,
      response_style: responseStyle,
      answer_model: effectiveAnswerModel,
    };
    // Mode A sends the concrete choice alongside its tier; Mode B omits the key entirely
    // rather than setting it to undefined, so callers that assert the object exactly see it
    // absent, not present-with-no-value.
    if (hasModelOptions && selectedEntry !== undefined) {
      payload.model_option_id = selectedEntry.id;
    }
    onSubmit(payload);
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      handleSubmit(event);
    }
  };

  return (
    <form ref={formRef} onSubmit={handleSubmit} noValidate className="space-y-3">
      <Textarea
        name="question"
        required
        aria-label="Question"
        aria-invalid={error !== null ? true : undefined}
        aria-describedby={error !== null ? "question-length question-error" : "question-length"}
        placeholder="Ask a decision question grounded in this project's documents…"
        value={question}
        rows={3}
        maxLength={MAX_QUESTION_LENGTH}
        // RMD5-T04 (RE2E-DEF-047): the question and its options are locked with the button, so
        // the form never shows a question or settings the running analysis does not use.
        disabled={disabled}
        onChange={(event) => {
          setQuestion(event.target.value);
        }}
        onKeyDown={handleKeyDown}
      />
      <p id="question-length" className="text-right text-xs text-muted-foreground">
        {question.length.toLocaleString("en-US")} / {MAX_QUESTION_LENGTH.toLocaleString("en-US")}{" "}
        characters
      </p>
      <div className="flex flex-wrap items-start gap-4">
        <div className="flex flex-col gap-1">
          <label className="flex items-center gap-2 text-sm">
            Retrieval
            <select
              aria-label="Retrieval mode"
              className="rounded-md border bg-transparent px-2 py-1 disabled:cursor-not-allowed disabled:opacity-50"
              value={retrievalMode}
              disabled={disabled}
              onChange={(event) => {
                setRetrievalMode(event.target.value as RetrievalMode);
              }}
            >
              {RETRIEVAL_MODES.map((mode) => (
                <option key={mode} value={mode}>
                  {mode}
                </option>
              ))}
            </select>
          </label>
          <p className="text-xs text-muted-foreground">{RETRIEVAL_DESCRIPTIONS[retrievalMode]}</p>
        </div>
        <div className="flex flex-col gap-1">
          <label className="flex items-center gap-2 text-sm">
            Style
            <select
              aria-label="Response style"
              className="rounded-md border bg-transparent px-2 py-1 disabled:cursor-not-allowed disabled:opacity-50"
              value={responseStyle}
              disabled={disabled}
              onChange={(event) => {
                setResponseStyle(event.target.value as ResponseStyle);
              }}
            >
              {RESPONSE_STYLES.map((style) => (
                <option key={style} value={style}>
                  {style}
                </option>
              ))}
            </select>
          </label>
          <p className="text-xs text-muted-foreground">{STYLE_DESCRIPTIONS[responseStyle]}</p>
        </div>
        {/* FE-016: at 320 px the model block wraps, its select shrinks and the badge drops below.
            UDF-11: from 640 px it always starts its own row, so the row does not re-wrap when the
            capability lookup widens the select and adds the badge. */}
        <div className="flex min-w-0 max-w-full flex-col gap-1 sm:basis-md">
          <label className="flex flex-wrap items-center gap-2 text-sm">
            Model
            {hasModelOptions ? (
              <select
                aria-label="Model"
                aria-describedby="answer-model-description routing-profile-description"
                className="min-w-0 max-w-full rounded-md border bg-transparent px-2 py-1 disabled:cursor-not-allowed disabled:opacity-50"
                value={selectedOptionId ?? lockedOptionId ?? ""}
                disabled={disabled || !modelSelectable}
                onChange={(event) => {
                  setPersistedOptionId(event.target.value);
                }}
              >
                {[...providerGroups.entries()].map(([provider, entries]) => (
                  <optgroup
                    key={provider}
                    label={resolveProviderLabel(provider, llmConfig?.providers ?? [])}
                  >
                    {entries.map((entry) => (
                      <option key={entry.id} value={entry.id} disabled={!entry.available}>
                        {entry.available
                          ? entry.label
                          : `${entry.label} (${unavailableReasonLabel(entry.unavailable_reason)})`}
                      </option>
                    ))}
                  </optgroup>
                ))}
              </select>
            ) : (
              <select
                aria-label="Model"
                aria-describedby="answer-model-description routing-profile-description"
                className="min-w-0 max-w-full rounded-md border bg-transparent px-2 py-1 disabled:cursor-not-allowed disabled:opacity-50"
                value={effectiveAnswerModel}
                disabled={disabled || !modelSelectable}
                onChange={(event) => {
                  setSelectedTier(event.target.value as AnswerModel);
                }}
              >
                {modeBTiers.map((tier) => (
                  <option key={tier} value={tier}>
                    {tier}
                  </option>
                ))}
              </select>
            )}
            {!modelSelectable && !capabilityPending && (
              <Badge variant="warning" aria-hidden>
                unavailable
              </Badge>
            )}
          </label>
          {/* Associated with the select via aria-describedby rather than given
              its own landmark role, which the privacy notice already owns. */}
          {/* UDF-11: four lines, the longest message's height, stay reserved in every state, so
              nothing below moves when the capability lookup answers. */}
          <p
            id="answer-model-description"
            className={
              modelSelectable || capabilityPending
                ? "min-h-[4lh] text-xs text-muted-foreground"
                : "min-h-[4lh] max-w-xs text-xs text-amber-700"
            }
          >
            {modelSelectable
              ? (selectedDescription ?? "Chooses which configured model writes the answer")
              : lockedModelMessage(llmConfig, capabilityStatus, noAvailableModels)}
          </p>
          <p id="routing-profile-description" className="max-w-xs text-xs text-muted-foreground">
            Routing profile: {effectiveProfile} — {ROUTING_PROFILE_DESCRIPTIONS[effectiveProfile]}.
          </p>
        </div>
        <div className="ml-auto">
          <Button
            type="submit"
            disabled={askBlocked}
            aria-describedby={
              nothingCanAnswer || capabilityPending ? "answer-model-description" : undefined
            }
          >
            {disabled ? (
              "Analyzing…"
            ) : (
              <>
                Ask{" "}
                <kbd
                  className="ml-1 rounded border border-primary-foreground/40 px-1 text-[10px] font-sans"
                  aria-hidden
                >
                  Ctrl+↵
                </kbd>
              </>
            )}
          </Button>
        </div>
      </div>
      {error !== null && (
        <p id="question-error" role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
    </form>
  );
}
