"use client";

import { useState } from "react";
import Link from "next/link";
import MovieCard from "@/components/MovieCard";
import MovieSearchBox from "@/components/MovieSearchBox";
import RatingPicker from "@/components/RatingPicker";
import ScoreBreakdown from "@/components/ScoreBreakdown";
import Badge from "@/components/ui/Badge";
import Button from "@/components/ui/Button";
import Card from "@/components/ui/Card";
import { getRecommendations, getSimilarMovies } from "@/lib/api";
import type { RatedMovie, RecommendResult, SearchResult, SimilarResult } from "@/lib/types";

const MIN_RATINGS = 3;
const MAX_RATINGS = 10;

export default function Home() {
  return (
    <main className="mx-auto flex w-full max-w-5xl flex-col gap-6 px-6 pb-24">
      <Hero />
      <DemoCard />
      <TeaserCards />
      <Footer />
    </main>
  );
}

function Hero() {
  return (
    <Card className="flex flex-col gap-8 p-8 sm:p-12">
      <div className="flex flex-wrap justify-end gap-2">
        <Badge>Content-based</Badge>
        <Badge>Collaborative (SVD)</Badge>
        <Badge>Cold-start fold-in</Badge>
      </div>
      <div className="flex flex-col gap-4">
        <h1 className="font-mono text-4xl font-semibold leading-tight tracking-tight sm:text-5xl">
          A Hybrid Movie
          <br />
          Recommender
        </h1>
        <p className="max-w-xl text-muted">
          A 2020 B.Tech thesis, rebuilt as something you can actually use. Rate a
          few movies below and get live, personalized recommendations — blended
          from content similarity and a collaborative model, folded in on the
          spot for a visitor the system has never seen before.
        </p>
      </div>
      <div className="flex flex-wrap gap-3">
        <Button href="#demo" variant="accent">
          Try it below ↓
        </Button>
        <Button href="/how-it-works" variant="outline">
          How it works
        </Button>
        <Button href="/benchmarks" variant="outline">
          Benchmarks
        </Button>
      </div>
    </Card>
  );
}

function DemoCard() {
  const [mode, setMode] = useState<"recommend" | "similar">("recommend");

  return (
    <Card id="demo" className="scroll-mt-6 p-6 sm:p-8">
      <div className="flex flex-col gap-6">
        <div className="flex flex-wrap gap-2 font-mono text-sm">
          <button
            onClick={() => setMode("recommend")}
            className={`rounded-full px-4 py-2 transition-colors ${
              mode === "recommend"
                ? "bg-accent text-accent-foreground"
                : "border border-surface-border text-muted hover:text-foreground"
            }`}
          >
            Rate &amp; get recommendations
          </button>
          <button
            onClick={() => setMode("similar")}
            className={`rounded-full px-4 py-2 transition-colors ${
              mode === "similar"
                ? "bg-accent text-accent-foreground"
                : "border border-surface-border text-muted hover:text-foreground"
            }`}
          >
            Quick similar-movie search
          </button>
        </div>

        {mode === "recommend" ? <RecommendFlow /> : <SimilarFlow />}
      </div>
    </Card>
  );
}

function RecommendFlow() {
  const [rated, setRated] = useState<RatedMovie[]>([]);
  const [results, setResults] = useState<RecommendResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function addMovie(movie: SearchResult) {
    if (rated.some((r) => r.movie_id === movie.id) || rated.length >= MAX_RATINGS) return;
    setRated((prev) => [
      ...prev,
      { movie_id: movie.id, title: movie.title, poster_path: movie.poster_path, rating: 4 },
    ]);
    setResults(null);
  }

  function setRating(movieId: number, rating: number) {
    setRated((prev) => prev.map((r) => (r.movie_id === movieId ? { ...r, rating } : r)));
  }

  function removeMovie(movieId: number) {
    setRated((prev) => prev.filter((r) => r.movie_id !== movieId));
    setResults(null);
  }

  async function submit() {
    setLoading(true);
    setError(null);
    try {
      const recs = await getRecommendations(
        rated.map((r) => ({ movie_id: r.movie_id, rating: r.rating }))
      );
      setResults(recs);
    } catch {
      setError("Something went wrong getting recommendations. Try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-3">
        <MovieSearchBox
          placeholder="Search a movie you like and rate it…"
          onSelect={addMovie}
        />
        <p className="text-xs text-muted">
          Rate at least {MIN_RATINGS} movies ({rated.length}/{MAX_RATINGS} added).
        </p>
      </div>

      {rated.length > 0 && (
        <ul className="flex flex-col gap-2">
          {rated.map((r) => (
            <li
              key={r.movie_id}
              className="flex items-center justify-between gap-3 rounded-xl border border-surface-border px-4 py-2.5"
            >
              <span className="text-sm">{r.title}</span>
              <div className="flex items-center gap-3">
                <RatingPicker value={r.rating} onChange={(rating) => setRating(r.movie_id, rating)} />
                <button
                  onClick={() => removeMovie(r.movie_id)}
                  className="text-xs text-muted hover:text-red-400"
                  aria-label="Remove"
                >
                  ✕
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}

      <Button
        onClick={submit}
        disabled={rated.length < MIN_RATINGS || loading}
        variant="solid"
        className="self-start"
      >
        {loading ? "Finding recommendations…" : "Get my recommendations"}
      </Button>

      {error && <p className="text-sm text-red-400">{error}</p>}

      {results && results.length > 0 && (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4">
          {results.map((r) => (
            <MovieCard
              key={r.movie.id}
              movie={r.movie}
              footer={
                <ScoreBreakdown
                  contentScore={r.content_score}
                  collabScore={r.collab_score}
                  blendedScore={r.blended_score}
                />
              }
            />
          ))}
        </div>
      )}

      {results && results.length === 0 && (
        <p className="text-sm text-muted">
          No recommendations found for that combination — try rating a few more movies.
        </p>
      )}
    </div>
  );
}

function SimilarFlow() {
  const [seed, setSeed] = useState<SearchResult | null>(null);
  const [results, setResults] = useState<SimilarResult[]>([]);
  const [loading, setLoading] = useState(false);

  async function pick(movie: SearchResult) {
    setSeed(movie);
    setLoading(true);
    try {
      const data = await getSimilarMovies(movie.id);
      setResults(data.results);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <MovieSearchBox placeholder="Type any movie…" onSelect={pick} />

      {loading && <p className="text-sm text-muted">Loading…</p>}

      {seed && !loading && (
        <>
          <p className="text-sm text-muted">
            Movies most similar to <span className="font-medium text-foreground">{seed.title}</span>{" "}
            by cast, director, keywords and genre:
          </p>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4">
            {results.map((r) => (
              <MovieCard key={r.movie.id} movie={r.movie} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function TeaserCards() {
  return (
    <div className="grid gap-4 sm:grid-cols-2">
      <Link href="/how-it-works">
        <Card className="flex h-full flex-col gap-3 p-6 transition-colors hover:border-white/20">
          <span className="font-mono text-xs uppercase tracking-wide text-muted">
            The pipeline
          </span>
          <h2 className="text-xl font-semibold">How this actually works</h2>
          <p className="text-sm text-muted">
            Soup construction → vector space → cosine similarity → SVD → cold-start
            fold-in → hybrid blend, walked through with the real diagrams and math.
          </p>
          <span className="mt-auto font-mono text-sm text-accent">Read it →</span>
        </Card>
      </Link>
      <Link href="/benchmarks">
        <Card className="flex h-full flex-col gap-3 p-6 transition-colors hover:border-white/20">
          <span className="font-mono text-xs uppercase tracking-wide text-muted">
            Evaluated, not assumed
          </span>
          <div className="font-mono text-4xl font-semibold text-accent">0.082</div>
          <p className="text-sm text-muted">
            NDCG@10 for the hybrid on a held-out MovieLens split — beating both its
            content-only and SVD-only components individually, benchmarked against
            popularity and item-KNN baselines too.
          </p>
          <span className="mt-auto font-mono text-sm text-accent">See the numbers →</span>
        </Card>
      </Link>
    </div>
  );
}

function Footer() {
  return (
    <footer className="flex flex-col gap-1 py-8 text-xs text-muted">
      <p>Hybrid movie recommender — B.Tech thesis, rebuilt as a live demo.</p>
      <a
        href="https://github.com/isayushsingh/BTech-Project"
        target="_blank"
        rel="noreferrer"
        className="text-accent hover:underline"
      >
        View source on GitHub →
      </a>
    </footer>
  );
}
