/**
 * OD-10 (FE-040): times are shown in the viewer's own zone with the zone named, and the UTC time
 * on hover. The backend's timestamps are ISO-8601 UTC; shown as UTC without saying so, a time
 * read as local was off by the viewer's offset.
 */
const PARTS: Intl.DateTimeFormatOptions = {
  year: "numeric",
  month: "short",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
  timeZoneName: "short",
};

const utcFormatter = new Intl.DateTimeFormat("en-US", { ...PARTS, timeZone: "UTC" });
// Created on first use, so it takes the zone of the browser that renders it.
let localFormatter: Intl.DateTimeFormat | undefined;
let localDateFormatter: Intl.DateTimeFormat | undefined;

function parse(isoTimestamp: string | null): Date | null {
  if (isoTimestamp === null) {
    return null;
  }
  const parsed = new Date(isoTimestamp);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

/**
 * The viewer's local date and time with its zone's abbreviation, e.g. "Sep 23, 2026, 5:20 PM PDT".
 * `timeZone` overrides the viewer's zone (tests).
 */
export function formatDateTime(isoTimestamp: string | null, timeZone?: string): string {
  const parsed = parse(isoTimestamp);
  if (parsed === null) {
    return "—";
  }
  const formatter =
    timeZone === undefined
      ? (localFormatter ??= new Intl.DateTimeFormat("en-US", PARTS))
      : new Intl.DateTimeFormat("en-US", { ...PARTS, timeZone });
  return formatter.format(parsed);
}

/** The same instant in UTC, for the hover title; empty when there is no valid time. */
export function formatUtcDateTime(isoTimestamp: string | null): string {
  const parsed = parse(isoTimestamp);
  return parsed === null ? "" : utcFormatter.format(parsed);
}

/** The viewer's local date alone, e.g. "Sep 23, 2026" (chart axes and summaries). */
export function formatDate(isoTimestamp: string | null): string {
  const parsed = parse(isoTimestamp);
  if (parsed === null) {
    return "—";
  }
  localDateFormatter ??= new Intl.DateTimeFormat("en-US", { dateStyle: "medium" });
  return localDateFormatter.format(parsed);
}
