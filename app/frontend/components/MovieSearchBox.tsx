"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { posterUrl, searchMovies } from "@/lib/api";
import type { SearchResult } from "@/lib/types";

export default function MovieSearchBox({
  placeholder = "Search for a movie…",
  onSelect,
}: {
  placeholder?: string;
  onSelect: (movie: SearchResult) => void;
}) {
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!query.trim()) {
      setSuggestions([]);
      return;
    }
    setLoading(true);
    const handle = setTimeout(async () => {
      try {
        const results = await searchMovies(query);
        setSuggestions(results);
      } catch {
        setSuggestions([]);
      } finally {
        setLoading(false);
      }
    }, 250);
    return () => clearTimeout(handle);
  }, [query]);

  return (
    <div className="relative">
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder={placeholder}
        className="w-full rounded-full border border-surface-border bg-surface px-4 py-3 font-mono text-sm text-foreground outline-none placeholder:text-muted focus:border-accent"
      />
      {loading && (
        <div className="absolute right-4 top-3.5 text-xs text-muted">…</div>
      )}
      {suggestions.length > 0 && (
        <ul className="absolute z-10 mt-2 max-h-72 w-full overflow-y-auto rounded-2xl border border-surface-border bg-surface shadow-xl">
          {suggestions.map((movie) => {
            const src = posterUrl(movie.poster_path);
            return (
              <li key={movie.id}>
                <button
                  type="button"
                  onClick={() => {
                    onSelect(movie);
                    setQuery("");
                    setSuggestions([]);
                  }}
                  className="flex w-full items-center gap-3 px-4 py-2.5 text-left text-sm hover:bg-white/[0.05]"
                >
                  <div className="relative h-10 w-7 shrink-0 overflow-hidden rounded bg-white/[0.06]">
                    {src && (
                      <Image src={src} alt={movie.title} fill sizes="28px" className="object-cover" />
                    )}
                  </div>
                  <span>
                    {movie.title}
                    {movie.year ? (
                      <span className="text-muted"> ({movie.year})</span>
                    ) : null}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
