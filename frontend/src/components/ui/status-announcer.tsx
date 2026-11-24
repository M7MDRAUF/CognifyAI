"use client";

import { useEffect, useState } from "react";

/**
 * The outcome of an action, said to assistive technology (RE2E-DEF-017, RMD10-T02). A create, delete or reindex used
 * to change the page silently: the dialog closed or a row went, and a screen-reader user heard nothing. The views
 * whose actions announce (projects, documents) render this polite live region; `announce` puts a sentence in it.
 * A page that announces nothing has none, so the chat page keeps its one status (DEC-UDP-03, DV-RMD-77). Only what
 * is announced after the region mounts is shown, so nothing stale is read out on a new page.
 */
type Listener = (message: string) => void;

const listeners = new Set<Listener>();

export function announce(message: string): void {
  for (const listener of listeners) listener(message);
}

export function StatusAnnouncer() {
  const [said, setSaid] = useState({ message: "", count: 0 });

  useEffect(() => {
    const listener: Listener = (message) => {
      setSaid((previous) => ({ message, count: previous.count + 1 }));
    };
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  }, []);

  return (
    <div role="status" aria-live="polite" aria-atomic="true" className="sr-only">
      {/* A new node for each sentence, so the same words said twice are read twice. */}
      <span key={said.count}>{said.message}</span>
    </div>
  );
}
