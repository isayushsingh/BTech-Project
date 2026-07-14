"use client";

import { useState } from "react";
import Link from "next/link";
import MovieCard from "@/components/MovieCard";
import MovieSearchBox from "@/components/MovieSearchBox";
import RatingPicker from "@/components/RatingPicker";
import ScoreBreakdown from "@/components/ScoreBreakdown";
import { getRecommendations, getSimilarMovies } from "@/lib/api";
import type { RatedMovie, RecommendResult, SearchResult, SimilarResult } from "@/lib/types";

const MIN_RATINGS = 3;
const MAX_RATINGS = 10;

export default function TryPage() {
  const [mode, setMode] = useState<"recommend" | "similar">("recommend");

  return (
    <main className="mx-auto flex max-w-4xl flex-col gap-8 px-6 py-12">
      <div>
        <Link href="/" className="text-sm text-neutral-500 hover:underline">
          ← Back
        </Link>
        <h1 className="mt-2 text-2xl font-semibold">Try the recommender</h1>
        <p className="mt-1 text-sm text-neutral-500">
          Rate a few movies you know for personalized hybrid recommendations, or just
          search a movie for instant content-based matches — no rating required.{" "}
          <Link href="/how-it-works" className="underline">
            See how this works →
          </Link>
        </p>
      </div>

      <div className="flex gap-2 text-sm">
        <button
          onClick={() => setMode("recommend")}
          className={`rounded-full px-3 py-1.5 ${
            mode === "recommend"
              ? "bg-neutral-900 text-white dark:bg-white dark:text-neutral-900"
              : "border border-black/10 dark:border-white/15"
          }`}
        >
          Rate & get recommendations
        </button>
        <button
          onClick={() => setMode("similar")}
          className={`rounded-full px-3 py-1.5 ${
            mode === "similar"
              ? "bg-neutral-900 text-white dark:bg-white dark:text-neutral-900"
              : "border border-black/10 dark:border-white/15"
          }`}
        >
          Quick similar-movie search
        </button>
      </div>

      {mode === "recommend" ? <RecommendFlow /> : <SimilarFlow />}
    </main>
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
        <p className="text-xs text-neutral-500">
          Rate at least {MIN_RATINGS} movies ({rated.length}/{MAX_RATINGS} added).
        </p>
      </div>

      {rated.length > 0 && (
        <ul className="flex flex-col gap-2">
          {rated.map((r) => (
            <li
              key={r.movie_id}
              className="flex items-center justify-between gap-3 rounded-md border border-black/10 dark:border-white/10 px-3 py-2"
            >
              <span className="text-sm">{r.title}</span>
              <div className="flex items-center gap-3">
                <RatingPicker value={r.rating} onChange={(rating) => setRating(r.movie_id, rating)} />
                <button
                  onClick={() => removeMovie(r.movie_id)}
                  className="text-xs text-neutral-400 hover:text-red-500"
                  aria-label="Remove"
                >
                  ✕
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}

      <button
        onClick={submit}
        disabled={rated.length < MIN_RATINGS || loading}
        className="self-start rounded-md bg-neutral-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-40 dark:bg-white dark:text-neutral-900"
      >
        {loading ? "Finding recommendations…" : "Get my recommendations"}
      </button>

      {error && <p className="text-sm text-red-500">{error}</p>}

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
        <p className="text-sm text-neutral-500">
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

      {loading && <p className="text-sm text-neutral-500">Loading…</p>}

      {seed && !loading && (
        <>
          <p className="text-sm text-neutral-500">
            Movies most similar to <span className="font-medium">{seed.title}</span> by
            cast, director, keywords and genre:
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
