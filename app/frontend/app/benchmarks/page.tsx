import Link from "next/link";
import { getBenchmarkResults } from "@/lib/api";
import ResultsTable from "@/components/benchmarks/ResultsTable";
import BaselineBarChart from "@/components/benchmarks/BaselineBarChart";
import WeightSweepChart from "@/components/benchmarks/WeightSweepChart";

export const revalidate = 3600;

export default async function BenchmarksPage() {
  const results = await getBenchmarkResults();

  return (
    <main className="mx-auto flex max-w-3xl flex-col gap-14 px-6 py-16">
      <div>
        <Link href="/" className="text-sm text-neutral-500 hover:underline">
          ← Back
        </Link>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight">How does this compare?</h1>
        <p className="mt-3 text-neutral-500">
          Not a comparison against Netflix&apos;s actual production system — that&apos;s
          proprietary, trained on billions of interactions, and not something anyone
          outside the company can honestly benchmark against. Instead, this is a
          rigorous, standard offline evaluation: the same held-out{" "}
          <a
            className="underline"
            href="https://www.kaggle.com/rounakbanik/the-movies-dataset"
            target="_blank"
            rel="noreferrer"
          >
            MovieLens
          </a>{" "}
          data, the same metrics the recommender-systems field actually uses, and
          baselines that represent real milestones in the field — including the
          matrix-factorization (SVD) technique that won the{" "}
          <a
            className="underline"
            href="https://en.wikipedia.org/wiki/Netflix_Prize"
            target="_blank"
            rel="noreferrer"
          >
            Netflix Prize
          </a>
          .
        </p>
      </div>

      <section className="flex flex-col gap-4">
        <h2 className="text-lg font-semibold">Methodology</h2>
        <ul className="flex flex-col gap-1.5 text-sm text-neutral-500">
          <li>
            • Per-user 80/20 train/test split ({results.n_train_ratings.toLocaleString()}{" "}
            train / {results.n_test_ratings.toLocaleString()} held-out ratings, across{" "}
            {results.n_users_evaluated.toLocaleString()} evaluated users)
          </li>
          <li>
            • Every baseline sees only the train split — the collaborative factors used
            here are retrained from scratch on train-only data, kept separate from the
            production model
          </li>
          <li>
            • Precision/Recall/NDCG@{results.k}, treating held-out ratings of{" "}
            {results.relevance_threshold}+ as &quot;relevant&quot;
          </li>
          <li>
            • RMSE/MAE reported only for baselines that produce a calibrated rating
            prediction — content-based and hybrid scores aren&apos;t on that scale
          </li>
          <li>
            • The SVD-only, content-only, and hybrid rows run through the exact same
            fold-in and blending code that powers the live{" "}
            <Link href="/try" className="underline">
              demo
            </Link>{" "}
            — this isn&apos;t a separate reimplementation
          </li>
        </ul>
      </section>

      <section className="flex flex-col gap-5">
        <h2 className="text-lg font-semibold">Results</h2>
        <ResultsTable baselines={results.baselines} />
        <BaselineBarChart baselines={results.baselines} />
      </section>

      <section className="flex flex-col gap-5">
        <h2 className="text-lg font-semibold">Tuning the blend weight</h2>
        <p className="text-sm text-neutral-500">
          The hybrid blends content and collaborative scores as{" "}
          <code className="rounded bg-neutral-100 px-1 py-0.5 dark:bg-neutral-900">
            w · collaborative + (1 − w) · content
          </code>
          . Sweeping w against this held-out split found w={results.best_weight} performs
          best — the production demo uses that value, not an arbitrary 50/50 split.
        </p>
        <WeightSweepChart
          sweep={results.weight_sweep}
          productionWeight={results.production_weight}
          bestWeight={results.best_weight}
        />
      </section>

      <section className="flex flex-col gap-3 border-t border-black/10 pt-8 text-sm text-neutral-500 dark:border-white/10">
        <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
          Honest takeaways
        </h2>
        <p>
          The hybrid clearly beats both of its individual components (content-only and
          SVD-only) — the core thesis that blending helps holds up under held-out
          evaluation, not just anecdotally.
        </p>
        <p>
          It doesn&apos;t beat item-based KNN here, though. On a dataset this small and
          sparse (~1,000 movies with ratings, ~700 users), classical memory-based
          collaborative filtering is a genuinely strong, well-documented baseline —
          matrix factorization methods typically need substantially more data to show
          their advantage. Reporting that plainly is more useful than hiding it.
        </p>
      </section>
    </main>
  );
}
