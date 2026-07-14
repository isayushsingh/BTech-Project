import Link from "next/link";
import Badge from "./ui/Badge";

export default function SiteHeader() {
  return (
    <header className="mx-auto flex w-full max-w-5xl flex-wrap items-center justify-between gap-x-4 gap-y-3 px-6 py-6">
      <Link href="/" className="shrink-0">
        <Badge variant="neutral">B.Tech Thesis, rebuilt</Badge>
      </Link>
      <nav className="flex shrink-0 items-center gap-4 font-mono text-xs whitespace-nowrap uppercase tracking-wide text-muted sm:gap-5">
        <Link href="/how-it-works" className="transition-colors hover:text-foreground">
          How it works
        </Link>
        <Link href="/benchmarks" className="transition-colors hover:text-foreground">
          Benchmarks
        </Link>
      </nav>
    </header>
  );
}
