"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";

import { cn } from "@/lib/utils";

const TABS = [
  { segment: "", label: "Overview" },
  { segment: "documents", label: "Documents" },
  { segment: "chat", label: "Chat" },
  { segment: "dashboard", label: "Dashboard" },
  { segment: "reports", label: "Reports" },
] as const;

export function ProjectNav({ projectId }: { projectId: string }) {
  const pathname = usePathname();
  const base = `/projects/${projectId}`;
  const listRef = useRef<HTMLElement>(null);
  // FE-016: on a narrow screen the tab list scrolls; an edge fade marks the side with hidden tabs.
  const [hidden, setHidden] = useState({ left: false, right: false });

  const measure = useCallback(() => {
    const list = listRef.current;
    if (list === null) {
      return;
    }
    setHidden({
      left: list.scrollLeft > 1,
      right: list.scrollLeft + list.clientWidth < list.scrollWidth - 1,
    });
  }, []);

  // Keep the current tab in view, without scrolling the page vertically.
  const reveal = useCallback(() => {
    const list = listRef.current;
    const current = list?.querySelector<HTMLElement>('[aria-current="page"]');
    if (list !== null && current !== null && current !== undefined) {
      const start = current.offsetLeft - list.offsetLeft;
      const end = start + current.offsetWidth;
      if (start < list.scrollLeft) {
        list.scrollLeft = start;
      } else if (end > list.scrollLeft + list.clientWidth) {
        list.scrollLeft = end - list.clientWidth;
      }
    }
    measure();
  }, [measure]);

  useEffect(() => {
    reveal();
  }, [pathname, reveal]);

  // Tabs change width after mount (the web font loads, the viewport turns): reveal again then.
  useEffect(() => {
    const list = listRef.current;
    if (list === null) {
      return;
    }
    const observer = new ResizeObserver(reveal);
    observer.observe(list);
    for (const tab of list.children) {
      observer.observe(tab);
    }
    return () => {
      observer.disconnect();
    };
  }, [reveal]);

  return (
    <div className="relative">
      <nav
        ref={listRef}
        onScroll={measure}
        className="flex gap-1 overflow-x-auto border-b"
        aria-label="Project sections"
      >
        {TABS.map((tab) => {
          const href = tab.segment === "" ? base : `${base}/${tab.segment}`;
          const active = pathname === href;
          return (
            <Link
              key={tab.label}
              href={href}
              aria-current={active ? "page" : undefined}
              className={cn(
                "shrink-0 border-b-2 px-4 py-2 text-sm font-medium transition-colors",
                active
                  ? "border-primary text-foreground font-semibold"
                  : "border-transparent text-muted-foreground hover:text-foreground",
              )}
            >
              {tab.label}
            </Link>
          );
        })}
      </nav>
      {hidden.left && (
        <div
          data-tab-fade="left"
          aria-hidden
          className="pointer-events-none absolute inset-y-0 left-0 w-8 bg-gradient-to-r from-background"
        />
      )}
      {hidden.right && (
        <div
          data-tab-fade="right"
          aria-hidden
          className="pointer-events-none absolute inset-y-0 right-0 w-8 bg-gradient-to-l from-background"
        />
      )}
    </div>
  );
}
