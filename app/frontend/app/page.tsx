import Link from "next/link";

export default function Home() {
  return (
    <main className="mx-auto flex min-h-screen max-w-3xl flex-col items-center justify-center gap-8 px-6 py-24 text-center">
      <div className="flex flex-col gap-3">
        <p className="text-sm font-medium uppercase tracking-wide text-neutral-500">
          B.Tech Thesis, rebuilt
        </p>
        <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">
          A Hybrid Movie Recommender
        </h1>
        <p className="mx-auto max-w-xl text-neutral-500">
          Content-based filtering, collaborative filtering via SVD, and a live
          cold-start fold-in — combined into one blended recommendation engine.
          Try it yourself, or see exactly how it works.
        </p>
      </div>

      <div className="flex flex-wrap justify-center gap-4">
        <Link
          href="/try"
          className="rounded-md bg-neutral-900 px-5 py-3 text-sm font-medium text-white dark:bg-white dark:text-neutral-900"
        >
          Try the recommender →
        </Link>
        <Link
          href="/how-it-works"
          className="rounded-md border border-black/15 px-5 py-3 text-sm font-medium dark:border-white/20"
        >
          See how it works
        </Link>
      </div>
    </main>
  );
}
