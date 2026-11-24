import { type RefObject, useLayoutEffect, useRef } from "react";

import type { StageStatus } from "@/lib/chat/pipeline";

/** A stage must be seen running this long before its completion settles (MOT-10). */
export const MIN_OBSERVED_RUNNING_MS = 300;
/** The settle's length (MOT-10). */
export const SETTLE_MS = 200;

const SETTLE_KEYFRAMES: Keyframe[] = [
  { transform: "scale(0.85)", opacity: 0.4 },
  { transform: "scale(1)", opacity: 1 },
];
const SETTLE_TIMING: KeyframeAnimationOptions = {
  duration: SETTLE_MS,
  easing: "cubic-bezier(0.2, 0, 0, 1)",
};
// MOT-11: the attempts badge fades in and rises 2 px, once.
const REVEAL_KEYFRAMES: Keyframe[] = [
  { opacity: 0, transform: "translateY(2px)" },
  { opacity: 1, transform: "translateY(0)" },
];
const REVEAL_TIMING: KeyframeAnimationOptions = { duration: 180, easing: "ease-out" };

/** Whether an emphasis may play: the element can animate and no reduced motion is asked for. */
function canAnimate(element: Element | null): element is Element {
  if (element === null || typeof element.animate !== "function") {
    return false;
  }
  return !(
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

/**
 * Settle a stage's status icon once when this mounted component watched the stage run for at
 * least `MIN_OBSERVED_RUNNING_MS` and saw it complete (MOT-10). A provisional completion waits
 * for its record to confirm it (AST-30); a failure, a cancel or a degraded close never settles. A
 * replay burst, a reload, a history pick or a duplicate frame shows no stage running for that
 * long, so it never settles. The animation starts in a layout effect on the commit that set the
 * status: no state, no timer, and nothing waits for its end; a change or an unmount cancels it
 * (R07).
 */
export function useSettleEmphasis(
  ref: RefObject<Element | null>,
  status: StageStatus,
  provisional: boolean,
): void {
  // When this component first saw the stage running, or null.
  const runningSince = useRef<number | null>(null);
  // How long it was seen running when it closed, held until its completion is confirmed.
  const observedMs = useRef<number | null>(null);
  useLayoutEffect(() => {
    if (status === "running") {
      runningSince.current ??= performance.now();
      return undefined;
    }
    if (runningSince.current !== null) {
      observedMs.current = performance.now() - runningSince.current;
      runningSince.current = null;
    }
    if (status !== "completed") {
      observedMs.current = null;
      return undefined;
    }
    if (provisional) {
      return undefined;
    }
    const observed = observedMs.current;
    observedMs.current = null;
    const element = ref.current;
    if (observed === null || observed < MIN_OBSERVED_RUNNING_MS || !canAnimate(element)) {
      return undefined;
    }
    const animation = element.animate(SETTLE_KEYFRAMES, SETTLE_TIMING);
    return () => {
      animation.cancel();
    };
  }, [ref, status, provisional]);
}

/**
 * Reveal an element once when `value` first turns non-zero after mount (MOT-11, the attempts
 * badge), in a run this mounted component watched live for at least `MIN_OBSERVED_RUNNING_MS`. A
 * value present from the first render is shown as it is, and so is one that arrives after a replay
 * nobody watched: a history pick reads the run's record after its replay (AST-23; UDP6-T02,
 * RV-01-04). A change or an unmount cancels the reveal in flight; nothing waits for its end.
 */
export function useRevealOnChange(
  ref: RefObject<Element | null>,
  value: number,
  live: boolean,
): void {
  // Whether the value has been non-zero: at mount, a badge already there needs no reveal.
  const shown = useRef(value !== 0);
  // When this component first saw the run live, and whether it has watched it long enough.
  const liveSince = useRef<number | null>(null);
  const watched = useRef(false);
  useLayoutEffect(() => {
    if (live) {
      liveSince.current ??= performance.now();
      return;
    }
    if (liveSince.current !== null) {
      watched.current ||= performance.now() - liveSince.current >= MIN_OBSERVED_RUNNING_MS;
      liveSince.current = null;
    }
  }, [live]);
  useLayoutEffect(() => {
    if (value === 0 || shown.current) {
      return undefined;
    }
    shown.current = true;
    const liveFor = liveSince.current === null ? 0 : performance.now() - liveSince.current;
    if (!watched.current && liveFor < MIN_OBSERVED_RUNNING_MS) {
      return undefined;
    }
    const element = ref.current;
    if (!canAnimate(element)) {
      return undefined;
    }
    const animation = element.animate(REVEAL_KEYFRAMES, REVEAL_TIMING);
    return () => {
      animation.cancel();
    };
  }, [ref, value]);
}
