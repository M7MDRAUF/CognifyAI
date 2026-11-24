"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { AGENT_LABELS } from "@/components/chat/agent-runs-panel";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { StageLatency } from "@/lib/types/evaluation";
import { agentNameSchema } from "@/lib/types/query";
import { formatLatency } from "@/lib/utils/format";

function stageLabel(agentName: string): string {
  const parsed = agentNameSchema.safeParse(agentName);
  return parsed.success ? AGENT_LABELS[parsed.data] : agentName;
}

/**
 * The chart's height for `stages` rows: 32 px a stage plus room for the axis, never under 160 px
 * (7 stages: 272 px). Horizontal bars give each stage name a row of its own (SE-007), so the labels
 * stay readable at 320 px, where the vertical axis overlapped them.
 */
export function chartHeight(stages: number): number {
  return Math.max(160, stages * 32 + 48);
}

export function StageLatencyBarChart({ stages }: { stages: StageLatency[] }) {
  // Named stages in seconds, in the order received (the backend's pipeline order, CT-04); every
  // label shown at 12 px (FE-039, SE-007).
  const data = stages.map((stage) => ({
    agent: stageLabel(stage.agent_name),
    seconds: Number((stage.average_latency_ms / 1000).toFixed(1)),
    text: formatLatency(stage.average_latency_ms),
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Agent latency by stage</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="sr-only">
          {data.map((stage) => `${stage.agent}: ${stage.text}`).join("; ") ||
            "No agent latency recorded."}
        </p>
        {data.length === 0 ? (
          <p className="text-sm text-muted-foreground">No agent activity yet.</p>
        ) : (
          <ResponsiveContainer width="100%" height={chartHeight(data.length)}>
            <BarChart data={data} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" unit=" s" tick={{ fontSize: 12 }} />
              <YAxis
                type="category"
                dataKey="agent"
                width={96}
                interval={0}
                tick={{ fontSize: 12 }}
              />
              <Tooltip />
              <Bar dataKey="seconds" name="Average latency (s)" fill="var(--chart-2)" />
            </BarChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}
