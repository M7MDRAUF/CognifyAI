import type { Metadata } from "next";

import { HomeHero } from "@/components/layout/home-hero";

export const metadata: Metadata = {
  title: { absolute: "CognifyAI: decision intelligence over your own documents" },
};

export default function HomePage() {
  return <HomeHero />;
}
