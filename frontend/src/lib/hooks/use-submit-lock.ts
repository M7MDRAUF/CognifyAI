"use client";

import { useCallback, useRef, useState } from "react";

interface SubmitLock {
  /**
   * Take the lock synchronously. Returns false when it is already held, in which case the caller
   * must send nothing: this is the check a `disabled` prop cannot make, because a second click or
   * key press can be dispatched before React re-renders with the pending state (FE-001, FE-021).
   */
  acquire: () => boolean;
  /** Give the lock back; call it from the mutation's `onSettled` so an error never leaves it held. */
  release: () => void;
  /** Mirrors the lock for rendering (disabled states, labels). */
  isLocked: boolean;
}

/** One submission per intent for create actions whose endpoints are not idempotent. */
export function useSubmitLock(): SubmitLock {
  const held = useRef(false);
  const [isLocked, setIsLocked] = useState(false);

  const acquire = useCallback(() => {
    if (held.current) {
      return false;
    }
    held.current = true;
    setIsLocked(true);
    return true;
  }, []);

  const release = useCallback(() => {
    held.current = false;
    setIsLocked(false);
  }, []);

  return { acquire, release, isLocked };
}
