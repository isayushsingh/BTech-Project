import type { RecommendResult, SearchResult, SimilarResult } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function searchMovies(query: string): Promise<SearchResult[]> {
  if (!query.trim()) return [];
  const res = await fetch(
    `${API_URL}/api/movies/search?q=${encodeURIComponent(query)}`
  );
  if (!res.ok) throw new Error("search failed");
  const data = await res.json();
  return data.results;
}

export async function getSimilarMovies(
  movieId: number
): Promise<{ seed: SearchResult; results: SimilarResult[] }> {
  const res = await fetch(`${API_URL}/api/movies/${movieId}/similar`);
  if (!res.ok) throw new Error("similar lookup failed");
  return res.json();
}

export async function getRecommendations(
  ratings: { movie_id: number; rating: number }[]
): Promise<RecommendResult[]> {
  const res = await fetch(`${API_URL}/api/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ratings }),
  });
  if (!res.ok) throw new Error("recommend failed");
  const data = await res.json();
  return data.results;
}

export function posterUrl(path: string | null, size: "w200" | "w342" = "w200") {
  if (!path) return null;
  return `https://image.tmdb.org/t/p/${size}${path}`;
}
