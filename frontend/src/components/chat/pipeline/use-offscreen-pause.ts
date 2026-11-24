import { type RefObject, useEffect } from "react";

/**
 * Mark the element `data-offscreen` while it is outside the viewport, so the pipeline's CSS
 * pauses its running motion there (MOT-17). One IntersectionObserver writes the attribute
 * directly: no React state, so leaving or entering the viewport renders nothing. Without
 * IntersectionObserver it does nothing, and the motion still stops at the terminal event.
 */
export function useOffscreenPause(ref: RefObject<HTMLElement | null>): void {
  useEffect(() => {
    const element = ref.current;
    if (element === null || typeof IntersectionObserver === "undefined") {
      return;
    }
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          entry.target.removeAttribute("data-offscreen");
        } else {
          entry.target.setAttribute("data-offscreen", "");
        }
      }
    });
    observer.observe(element);
    return () => {
      observer.disconnect();
    };
  }, [ref]);
}
