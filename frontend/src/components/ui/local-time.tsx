import { formatDateTime, formatUtcDateTime } from "@/lib/utils/dates";

/** A timestamp in the viewer's zone with the zone named, and the UTC time on hover (OD-10). */
export function LocalTime({ iso }: { iso: string | null }) {
  const local = formatDateTime(iso);
  if (iso === null || local === "—") {
    return <>{local}</>;
  }
  return (
    <time dateTime={iso} title={formatUtcDateTime(iso)}>
      {local}
    </time>
  );
}
