"use client";

import { useState, useSyncExternalStore } from "react";

const STORAGE_KEY = "cognify_model_option";

function noopUnsubscribe() {
  return undefined;
}

// localStorage has no same-tab event subscription; we combine this with local
// React state (set on a successful write) so a selection made in this render
// is reflected immediately without waiting on a storage event.
function subscribeToStorage() {
  return noopUnsubscribe;
}

function getStorageSnapshot(): string | null {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

function getStorageServerSnapshot(): string | null {
  return null;
}

/**
 * Remembers the user's model choice across sessions.
 *
 * Returns the stored option id, or null when nothing is stored *or* the stored id is not in
 * `availableOptionIds` — a revoked key or an environment change must never strand the user on a
 * selection that no longer exists.
 */
export function usePersistedModelOption(
  availableOptionIds: readonly string[],
): readonly [
  selectedOptionId: string | null,
  setSelectedOptionId: (optionId: string | null) => void,
] {
  const storedOptionId = useSyncExternalStore(
    subscribeToStorage,
    getStorageSnapshot,
    getStorageServerSnapshot,
  );
  // undefined means "no local override yet"; until the setter succeeds once,
  // this render defers entirely to the storage snapshot above.
  const [localOverride, setLocalOverride] = useState<string | null | undefined>(undefined);

  const rawOptionId = localOverride === undefined ? storedOptionId : localOverride;
  const selectedOptionId =
    rawOptionId !== null && availableOptionIds.includes(rawOptionId) ? rawOptionId : null;

  const setSelectedOptionId = (optionId: string | null) => {
    try {
      if (optionId === null) {
        localStorage.removeItem(STORAGE_KEY);
      } else {
        localStorage.setItem(STORAGE_KEY, optionId);
      }
    } catch {
      /* localStorage unavailable in some restricted contexts; nothing to remember */
    }
    // Outside the try: a selection must take effect in this session even when persistence
    // fails, or a blocked localStorage (private browsing, blocked site data) makes the
    // control inert — the user picks a value and watches it silently revert.
    setLocalOverride(optionId);
  };

  return [selectedOptionId, setSelectedOptionId] as const;
}
