"use client";

import { useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";

/** How long "Copied" or "Copy failed" stays before the label returns. */
const FEEDBACK_MS = 2000;

type CopyState = "idle" | "copied" | "failed";

interface CopyReferenceButtonProps {
  /** Exactly the text shown beside the button: an error's reference, or a run's id. */
  value: string;
  /** The action's name, such as "Copy reference" or "Copy run ID". */
  label: string;
}

/**
 * Copy a reference to the clipboard (UXR-18). The feedback is the button's own name, "Copied" or
 * "Copy failed", with no live region, and focus stays on the button (A11Y-12). One timeout gives
 * the label back; it is cleared on unmount.
 */
export function CopyReferenceButton({ value, label }: CopyReferenceButtonProps) {
  const [state, setState] = useState<CopyState>("idle");
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mounted = useRef(false);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      if (timer.current !== null) {
        clearTimeout(timer.current);
        timer.current = null;
      }
    };
  }, []);

  async function copy() {
    let next: CopyState = "copied";
    try {
      await navigator.clipboard.writeText(value);
    } catch {
      // A refused permission, or no clipboard at all outside a secure context.
      next = "failed";
    }
    if (!mounted.current) {
      return;
    }
    setState(next);
    if (timer.current !== null) {
      clearTimeout(timer.current);
    }
    timer.current = setTimeout(() => {
      timer.current = null;
      setState("idle");
    }, FEEDBACK_MS);
  }

  return (
    <Button
      type="button"
      variant="outline"
      size="sm"
      onClick={() => {
        void copy();
      }}
    >
      {state === "copied" ? "Copied" : state === "failed" ? "Copy failed" : label}
    </Button>
  );
}
