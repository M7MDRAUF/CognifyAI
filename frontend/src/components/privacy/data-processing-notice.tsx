"use client";

import { useState, useSyncExternalStore } from "react";

import type { CapabilityStatus } from "@/lib/hooks/use-llm-config";
import { type LlmConfig, providerLabel } from "@/lib/types/llm";
import { cn } from "@/lib/utils";

const STORAGE_KEY = "cognify_privacy_dismissed";

/**
 * The one provider id for which nothing leaves the machine: with `LLM_PROVIDER=fake` the router
 * returns the built-in offline provider for chat *and* for embeddings
 * (`app/integrations/llm/router.py`, `embeddings()`), so no request is made at all. Every other
 * value means document text is embedded through a real API, whether or not a chat model is
 * currently selectable — which is why these notices key on the provider id and not on
 * `configured` alone.
 */
const OFFLINE_PROVIDER_ID = "fake";

/**
 * What the deployment will be told to call the place document text is sent — or `null` when
 * nothing the backend published justifies naming a vendor.
 *
 * Two facts are required before a name may be printed, and both come from `/llm/config`:
 *
 *  - `embedding_provider`, because the chat `provider` says nothing about it: embeddings are a
 *    second, independent egress channel.
 *  - `embedding_endpoint_is_default === true`, because the provider id fixes only the *API
 *    dialect*. `LLM_OPENAI_BASE_URL`, and the documented `LLM_BASE_URL` self-host hatch, point
 *    an "openai"-dialect client at any host it likes (`registry.resolve_credentials`), and then
 *    the chunks never reach OpenAI. Round 1 hardcoded "OpenAI" from the dialect alone and was
 *    wrong in every one of those deployments.
 *
 * Absent keys (an older backend), a null provider, and an overridden endpoint all land in the
 * same place: describe the destination without naming it. Saying less is always available;
 * naming the wrong company is not.
 */
function embeddingDestination(llmConfig: LlmConfig): string | null {
  const provider = llmConfig.embedding_provider;
  if (provider === null || provider === undefined) {
    return null;
  }
  if (llmConfig.embedding_endpoint_is_default !== true) {
    return null;
  }
  return providerLabel(provider);
}

const UNNAMED_DESTINATION = "the embedding service this deployment is configured with";

const CHAT_EGRESS =
  "Your question and the retrieved document excerpts are sent to the configured external LLM " +
  "API to produce the answer.";

/**
 * The question is embedded only when retrieval actually runs a vector search:
 * `RetrievalService._semantic` returns `[]` for `RetrievalMode.LEXICAL` and `embed_query` is
 * never called. Lexical is offered by a select in the same form, so the flat "and your
 * question" this replaces was false in a state the user can pick for themselves.
 */
function embeddingEgress(destination: string | null): string {
  return (
    "Every chunk of every uploaded document is sent to " +
    `${destination ?? UNNAMED_DESTINATION} to generate embeddings for indexing and search — ` +
    "whichever chat provider you select. In hybrid or semantic retrieval mode your question is " +
    "embedded there too."
  );
}

function ingestionEgress(destination: string | null): string {
  return (
    "Indexing a document sends the text of every one of its chunks to " +
    `${destination ?? UNNAMED_DESTINATION} to generate embeddings. This happens on every ` +
    "upload, whichever chat provider is selected."
  );
}

const CHAT_UNKNOWN =
  "Could not confirm how this deployment is configured, so your question, the retrieved " +
  "document excerpts and your document text may be sent to external APIs.";

const INGESTION_UNKNOWN =
  "Could not confirm how this deployment is configured, so indexing an uploaded document may " +
  "send its text to an external embedding API.";

/**
 * Said while the lookup is still in flight — the one surface where silence is not safe,
 * because the upload control sits directly below this notice and is the first thing on the
 * page. The chat surface can stay quiet: its action is behind a button the user has to reach.
 */
const INGESTION_PENDING =
  "Checking where this deployment sends document text to be embedded. Uploads are paused " +
  "until that is known.";

interface Notice {
  text: string;
  /** Wording for the collapsed badge; must be no stronger a claim than `text`. */
  badge: string;
}

/**
 * What holds the chat notice's place while the lookup is in flight (RE2E-DEF-021), so the form
 * below does not jump when it answers. It is the longest notice `chatNotice` can return, and the
 * one the captured live deployment shows: a configured chat model, an unnamed embedding
 * destination. A shorter answer moves the form up by the difference; this one moves nothing.
 * It is rendered invisible and hidden from assistive technology, so it claims nothing.
 */
const CHAT_HELD: Notice = {
  text: `${CHAT_EGRESS} ${embeddingEgress(null)}`,
  badge: "External LLM active",
};

export interface DataProcessingNoticeProps {
  /** Undefined until the capability lookup succeeds. */
  llmConfig?: LlmConfig;
  capabilityStatus: CapabilityStatus;
}

/**
 * The notice to show on the chat surface, or `null` when no honest claim can be made yet.
 *
 * While the lookup is in flight we say nothing rather than flash a warning we may have to
 * retract. A failed lookup is the one case where saying nothing would be worse than hedging:
 * the query is never retried, so the user would otherwise act on silence.
 */
function chatNotice(llmConfig: LlmConfig | undefined, status: CapabilityStatus): Notice | null {
  if (status === "error") {
    return { text: CHAT_UNKNOWN, badge: "External processing unconfirmed" };
  }
  if (llmConfig === undefined || llmConfig.provider === OFFLINE_PROVIDER_ID) {
    return null;
  }
  const embedding = embeddingEgress(embeddingDestination(llmConfig));
  if (!llmConfig.configured) {
    // No chat model is selectable, so only the ingestion/search half is claimed here.
    return { text: embedding, badge: "External embedding active" };
  }
  return { text: `${CHAT_EGRESS} ${embedding}`, badge: "External LLM active" };
}

function ingestionNotice(
  llmConfig: LlmConfig | undefined,
  status: CapabilityStatus,
): Notice | null {
  if (status === "error") {
    return { text: INGESTION_UNKNOWN, badge: "External processing unconfirmed" };
  }
  if (llmConfig === undefined) {
    return { text: INGESTION_PENDING, badge: "Checking where document text goes" };
  }
  if (llmConfig.provider === OFFLINE_PROVIDER_ID) {
    return null;
  }
  return {
    text: ingestionEgress(embeddingDestination(llmConfig)),
    badge: "External embedding active",
  };
}

function noopUnsubscribe() {
  return undefined;
}

// localStorage has no same-tab event subscription; we combine this with
// local React state so the badge updates immediately on dismiss.
function subscribeToStorage() {
  return noopUnsubscribe;
}

function getStorageSnapshot() {
  try {
    return localStorage.getItem(STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

function getStorageServerSnapshot() {
  return false;
}

/** `held`: the element only keeps its place; it is invisible and hidden from assistive technology. */
function NoticeBadge({ label, held = false }: { label: string; held?: boolean }) {
  return (
    <span
      aria-hidden={held ? true : undefined}
      className={cn(
        "inline-flex items-center rounded-md border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-xs text-amber-700",
        held && "invisible",
      )}
    >
      {label}
    </span>
  );
}

function NoticeBox({
  text,
  onDismiss,
  held = false,
}: {
  text: string;
  onDismiss?: () => void;
  held?: boolean;
}) {
  return (
    <div
      role={held ? undefined : "note"}
      aria-hidden={held ? true : undefined}
      className={cn(
        "flex items-start justify-between gap-4 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-700",
        held && "invisible",
      )}
    >
      <p>{text}</p>
      {/* The dismiss control's place, not the control: nothing in a held box can be reached. */}
      {held && <span className="shrink-0">✕</span>}
      {!held && onDismiss !== undefined && (
        <button
          type="button"
          onClick={onDismiss}
          aria-label="Dismiss privacy notice"
          className="shrink-0 rounded opacity-70 hover:opacity-100 focus-visible:outline-2 focus-visible:outline-offset-2"
        >
          ✕
        </button>
      )}
    </div>
  );
}

/**
 * Dismissible disclosure for the chat surface (PRD 9.4).
 *
 * Shown exactly when data will leave the machine, derived from `/llm/config`: only the backend
 * knows which provider a question will reach.
 */
export function ChatPrivacyNotice({ llmConfig, capabilityStatus }: DataProcessingNoticeProps) {
  const storedDismissed = useSyncExternalStore(
    subscribeToStorage,
    getStorageSnapshot,
    getStorageServerSnapshot,
  );
  const [localDismissed, setLocalDismissed] = useState(false);
  const dismissed = storedDismissed || localDismissed;

  const notice = chatNotice(llmConfig, capabilityStatus);
  if (notice === null) {
    if (llmConfig === undefined && capabilityStatus === "pending") {
      return dismissed ? (
        <NoticeBadge label={CHAT_HELD.badge} held />
      ) : (
        <NoticeBox text={CHAT_HELD.text} held />
      );
    }
    return null;
  }

  const handleDismiss = () => {
    try {
      localStorage.setItem(STORAGE_KEY, "1");
    } catch {
      /* localStorage unavailable in some restricted contexts */
    }
    setLocalDismissed(true);
  };

  if (dismissed) {
    return <NoticeBadge label={notice.badge} />;
  }
  return <NoticeBox text={notice.text} onDismiss={handleDismiss} />;
}

/**
 * The same disclosure at the point of upload, where the decision to send a document is actually
 * made. Not dismissible: it is the only notice on that surface, and it is short.
 */
export function IngestionPrivacyNotice({ llmConfig, capabilityStatus }: DataProcessingNoticeProps) {
  const notice = ingestionNotice(llmConfig, capabilityStatus);
  if (notice === null) {
    return null;
  }
  return <NoticeBox text={notice.text} />;
}
