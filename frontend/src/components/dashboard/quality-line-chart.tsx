"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { QualityPoint } from "@/lib/types/evaluation";
import { formatDate } from "@/lib/utils/dates";
import { EVALUATOR_SCORE_CAPTION } from "@/lib/utils/format";

export function QualityLineChart({ points }: { points: QualityPoint[] }) {
  // Each point on its run's date, not its position in the list (FE-039).
  const data = points.map((point) => ({
    date: formatDate(point.created_at),
    quality: Number((point.answer_quality * 100).toFixed(1)),
  }));
  const first = data[0];
  const last = data.at(-1);
  const span =
    first !== undefined && last !== undefined ? `, from ${first.date} to ${last.date}` : "";
  const average =
    data.length === 0
      ? 0
      : Math.round(data.reduce((sum, point) => sum + point.quality, 0) / data.length);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Answer quality over time</CardTitle>
        {data.length > 0 && <CardDescription>{EVALUATOR_SCORE_CAPTION}</CardDescription>}
      </CardHeader>
      <CardContent>
        <p className="sr-only">
          Answer relevance across {data.length} {data.length === 1 ? "query" : "queries"}, averaging{" "}
          {average}%{span}.
        </p>
        {data.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No quality data yet. Run a query from the Chat tab to see answer quality trend over
            time.
          </p>
        ) : (
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" interval="preserveStartEnd" tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Line type="monotone" dataKey="quality" stroke="var(--chart-1)" />
            </LineChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}
