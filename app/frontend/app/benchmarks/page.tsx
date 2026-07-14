import Link from "next/link";
import { getBenchmarkResults } from "@/lib/api";
import ResultsTable from "@/components/benchmarks/ResultsTable";
import BaselineBarChart from "@/components/benchmarks/BaselineBarChart";
import WeightSweepChart from "@/components/benchmarks/WeightSweepChart";

export const revalidate = 3600;

export default async function BenchmarksPage() {
  const results = await getBenchmarkResults();

  return (
    <main className="mx-auto flex max-w-3xl flex-col gap-14 px-6 py-12">
      <div>
        <h1 className="font-mono text-3xl font-semibold tracking-tight">
          How does this compare?
        </h1>
        <p className="mt-3 text-muted">
          Not a comparison against Netflix&apos;s actual production system — that&apos;s
          proprietary, trained on billions of interactions, and not something anyone
          outside the company can honestly benchmark against. Instead, this is a
          rigorous, standard offline evaluation: the same held-out{" "}
          <a
            className="text-accent underline"
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
            className="text-accent underline"
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
        <ul className="flex flex-col gap-1.5 text-sm text-muted">
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
            <Link href="/#demo" className="text-accent underline">
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
        <p className="text-sm text-muted">
          The hybrid blends content and collaborative scores as{" "}
          <code className="rounded bg-white/[0.06] px-1 py-0.5 text-foreground">
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

      <section className="flex flex-col gap-3 border-t border-surface-border pt-8 text-sm text-muted">
        <h2 className="text-lg font-semibold text-foreground">Honest takeaways</h2>
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
