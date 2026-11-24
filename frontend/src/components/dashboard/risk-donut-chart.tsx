"use client";

import { Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { RiskCategoryCount } from "@/lib/types/evaluation";

/**
 * One fixed colour per `RiskCategory` value (SE-006, SD-05), so a category looks the same on every
 * project whatever else is present. A value the backend adds later falls back to a neutral colour.
 */
export const CATEGORY_COLORS: Readonly<Record<string, string>> = {
  academic: "var(--chart-1)",
  data_quality: "var(--chart-2)",
  ethical: "var(--chart-3)",
  implementation: "var(--chart-4)",
  operational: "var(--chart-5)",
  privacy: "var(--chart-6)",
};
const FALLBACK_COLOR = "var(--muted-foreground)";

const CATEGORY_LABELS: Readonly<Record<string, string>> = {
  academic: "Academic",
  data_quality: "Data quality",
  ethical: "Ethical",
  implementation: "Implementation",
  operational: "Operational",
  privacy: "Privacy",
};

/** A category's readable label; an unknown value reads as its words, first letter capitalised. */
function categoryLabel(category: string): string {
  const known = CATEGORY_LABELS[category];
  if (known !== undefined) {
    return known;
  }
  const words = category.replaceAll("_", " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

export function RiskDonutChart({ categories }: { categories: RiskCategoryCount[] }) {
  // Slices keep the backend's order; the colour follows the category, never the position.
  const data = categories.map((category) => ({
    name: category.category,
    label: categoryLabel(category.category),
    value: category.count,
    fill: CATEGORY_COLORS[category.category] ?? FALLBACK_COLOR,
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Risk categories</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="sr-only">
          {data.map((slice) => `${slice.label}: ${String(slice.value)}`).join("; ") ||
            "No risks identified."}
        </p>
        {data.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No risk data yet. Risk categories (ethical, privacy, operational, and others) appear
            here after queries complete.
          </p>
        ) : (
          <>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie
                  data={data}
                  dataKey="value"
                  nameKey="label"
                  innerRadius={50}
                  outerRadius={80}
                />
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            {/* Visible text for sighted users (WCAG 1.4.1). Hidden from screen readers, which
                already get the same labels and counts once from the summary above. */}
            <ul
              aria-hidden="true"
              className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-foreground"
            >
              {data.map((slice) => (
                <li key={slice.name} className="flex items-center gap-1.5">
                  <span
                    className="inline-block size-2.5 shrink-0 rounded-sm"
                    style={{ background: slice.fill }}
                  />
                  {`${slice.label} (${String(slice.value)})`}
                </li>
              ))}
            </ul>
          </>
        )}
      </CardContent>
    </Card>
  );
}
