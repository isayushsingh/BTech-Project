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
        className="w-full rounded-md border border-black/15 dark:border-white/15 bg-white dark:bg-neutral-900 px-3 py-2 text-sm outline-none focus:border-neutral-500"
      />
      {loading && (
        <div className="absolute right-3 top-2.5 text-xs text-neutral-400">
          …
        </div>
      )}
      {suggestions.length > 0 && (
        <ul className="absolute z-10 mt-1 max-h-72 w-full overflow-y-auto rounded-md border border-black/10 dark:border-white/10 bg-white dark:bg-neutral-900 shadow-lg">
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
                  className="flex w-full items-center gap-3 px-3 py-2 text-left text-sm hover:bg-neutral-100 dark:hover:bg-neutral-800"
                >
                  <div className="relative h-10 w-7 shrink-0 overflow-hidden rounded bg-neutral-200 dark:bg-neutral-800">
                    {src && (
                      <Image src={src} alt={movie.title} fill sizes="28px" className="object-cover" />
                    )}
                  </div>
                  <span>
                    {movie.title}
                    {movie.year ? (
                      <span className="text-neutral-400"> ({movie.year})</span>
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
