import Link from "next/link";
import RevealSection from "@/components/RevealSection";
import SoupDiagram from "@/components/explainer/SoupDiagram";
import BatmanRobinProblem from "@/components/explainer/BatmanRobinProblem";
import UtilityMatrix from "@/components/explainer/UtilityMatrix";
import FactorizationDiagram from "@/components/explainer/FactorizationDiagram";
import FoldInFlow from "@/components/explainer/FoldInFlow";
import HybridBlendSlider from "@/components/explainer/HybridBlendSlider";

function Step({ number, title }: { number: string; title: string }) {
  return (
    <div className="flex items-center gap-3">
      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-neutral-900 text-xs font-medium text-white dark:bg-white dark:text-neutral-900">
        {number}
      </span>
      <h2 className="text-xl font-semibold">{title}</h2>
    </div>
  );
}

export default function HowItWorksPage() {
  return (
    <main className="mx-auto flex max-w-3xl flex-col gap-20 px-6 py-16">
      <div>
        <Link href="/" className="text-sm text-neutral-500 hover:underline">
          ← Back
        </Link>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight">How this recommender works</h1>
        <p className="mt-3 text-neutral-500">
          This started as a B.Tech thesis on hybrid recommendation engines. Here&apos;s
          the actual pipeline behind the{" "}
          <Link href="/try" className="underline">
            live demo
          </Link>
          , from raw movie metadata to a blended recommendation.
        </p>
      </div>

      <RevealSection className="flex flex-col gap-6">
        <Step number="1" title="Build a text &quot;soup&quot; per movie" />
        <p className="text-neutral-500">
          Content-based filtering treats each movie as a document. We pull its cast,
          director, keywords, and genres from the metadata and concatenate them into
          one string per movie — the vocabulary that describes it.
        </p>
        <SoupDiagram />
      </RevealSection>

      <RevealSection className="flex flex-col gap-6">
        <Step number="2" title="Vectorize and compare with cosine similarity" />
        <p className="text-neutral-500">
          A <code className="rounded bg-neutral-100 px-1 py-0.5 text-sm dark:bg-neutral-900">
            CountVectorizer
          </code>{" "}
          turns every movie&apos;s soup into a vector of token counts. Movies with
          overlapping cast, crew, and themes end up close together in that vector
          space — measured with cosine similarity. This is what powers the
          &quot;quick similar-movie search&quot; in the demo, no rating required.
        </p>
      </RevealSection>

      <RevealSection className="flex flex-col gap-6">
        <Step number="2.1" title="The Batman &amp; Robin problem" />
        <p className="text-neutral-500">
          Pure text similarity has a blind spot: it can&apos;t tell a beloved movie
          from a bad one made by the same people. <em>Batman &amp; Robin</em> scores
          nearly as high on cast/crew overlap with <em>The Dark Knight</em> as{" "}
          <em>Batman Begins</em> does — even though critics widely consider it one of
          the worst entries in the franchise.
        </p>
        <BatmanRobinProblem />
        <p className="text-neutral-500">
          The fix: gate the ranking by IMDB&apos;s weighted-rating formula, so
          similarity alone can&apos;t carry a poorly reviewed movie to the top.
        </p>
      </RevealSection>

      <RevealSection className="flex flex-col gap-6">
        <Step number="3" title="Collaborative filtering: fill in the utility matrix" />
        <p className="text-neutral-500">
          Content similarity ignores something important: what people actually
          thought. Collaborative filtering treats ratings as a giant, mostly-empty
          user × movie matrix, and tries to predict the missing cells from patterns in
          the ones that are filled in.
        </p>
        <UtilityMatrix />
      </RevealSection>

      <RevealSection className="flex flex-col gap-6">
        <Step number="4" title="Factorize it with SVD" />
        <p className="text-neutral-500">
          Singular value decomposition approximates that sparse ratings matrix as the
          product of two much smaller matrices — a latent vector per user, and a
          latent vector per movie. Trained with regularized gradient descent on the
          MovieLens ratings, this gives every movie a small &quot;taste
          fingerprint&quot; (<code className="rounded bg-neutral-100 px-1 py-0.5 text-sm dark:bg-neutral-900">qᵢ</code>).
        </p>
        <FactorizationDiagram />
      </RevealSection>

      <RevealSection className="flex flex-col gap-6">
        <Step number="5" title="Cold-start fold-in for you, live" />
        <p className="text-neutral-500">
          The SVD above was trained once, offline, on existing MovieLens users. You
          aren&apos;t one of them — so when you rate movies in the demo, there&apos;s
          no retraining. Instead we solve a small regularized least-squares problem
          against the fixed, already-trained movie vectors to approximate{" "}
          <em>your</em> latent vector on the spot.
        </p>
        <FoldInFlow />
      </RevealSection>

      <RevealSection className="flex flex-col gap-6">
        <Step number="6" title="Blend both signals" />
        <p className="text-neutral-500">
          The final recommendation is a weighted average of the content-based score
          (what&apos;s textually similar to what you rated highly) and the
          collaborative score (what people with similar taste vectors liked) — the
          same &quot;hybrid&quot; idea the original thesis set out to build, now
          running against your own ratings in real time.
        </p>
        <HybridBlendSlider />
      </RevealSection>

      <RevealSection className="flex flex-col items-center gap-4 border-t border-black/10 pt-12 text-center dark:border-white/10">
        <p className="text-neutral-500">That&apos;s the whole pipeline. See it work on real data.</p>
        <Link
          href="/try"
          className="rounded-md bg-neutral-900 px-5 py-3 text-sm font-medium text-white dark:bg-white dark:text-neutral-900"
        >
          Try the recommender →
        </Link>
      </RevealSection>
    </main>
  );
}
