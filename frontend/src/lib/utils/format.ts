const UNITS = ["B", "KB", "MB", "GB"] as const;

export function formatBytes(bytes: number): string {
  if (bytes <= 0) {
    return "0 B";
  }
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < UNITS.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  const isWhole = Number.isInteger(value);
  const rounded = value >= 10 || unitIndex === 0 || isWhole ? Math.round(value) : value.toFixed(1);
  return `${String(rounded)} ${UNITS[unitIndex] ?? "B"}`;
}

/**
 * OD-02 (FE-006): the caption on every surface that shows an evaluator score. The evaluator is an
 * agent under test and never a gate (D-2, ADR-003), and its agreement with independent graders is
 * not computed (known limitation 31).
 */
export const EVALUATOR_SCORE_CAPTION =
  "Scored by the evaluator agent — a model's judgement, not a verified measurement.";

/** A latency in human units (FE-039, Probe P-35): "120 ms", "7.9 s", "2 min 46 s". */
export function formatLatency(milliseconds: number | null): string {
  if (milliseconds === null) {
    return "—";
  }
  if (milliseconds < 1000) {
    return `${String(Math.round(milliseconds))} ms`;
  }
  const seconds = milliseconds / 1000;
  if (seconds < 60) {
    return `${seconds.toFixed(1)} s`;
  }
  const whole = Math.round(seconds);
  return `${String(Math.floor(whole / 60))} min ${String(whole % 60)} s`;
}

export function formatScore(score: number | null): string {
  if (score === null) {
    return "—";
  }
  return `${String(Math.round(score * 100))}%`;
}
