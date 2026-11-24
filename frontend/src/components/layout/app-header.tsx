import Link from "next/link";

export function AppHeader() {
  return (
    <header className="border-b bg-background">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4 sm:px-6">
        <Link
          href="/"
          className="text-base font-bold tracking-tight focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:rounded-sm"
        >
          CognifyAI
        </Link>
        <nav aria-label="Main navigation">
          <Link
            href="/projects"
            className="text-sm font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:rounded-sm"
          >
            Projects
          </Link>
        </nav>
      </div>
    </header>
  );
}
