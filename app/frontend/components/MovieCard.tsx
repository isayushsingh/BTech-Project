import { posterUrl } from "@/lib/api";
import type { Movie } from "@/lib/types";
import PosterImage from "./PosterImage";

export default function MovieCard({
  movie,
  footer,
}: {
  movie: Movie;
  footer?: React.ReactNode;
}) {
  const src = posterUrl(movie.poster_path);
  let genres: string[] = [];
  try {
    genres = JSON.parse(movie.genres);
  } catch {
    genres = [];
  }

  return (
    <div className="flex flex-col overflow-hidden rounded-lg border border-black/10 dark:border-white/10 bg-white dark:bg-neutral-900">
      <div className="relative aspect-[2/3] w-full bg-neutral-200 dark:bg-neutral-800">
        {src ? (
          <PosterImage src={src} alt={movie.title} />
        ) : (
          <div className="flex h-full items-center justify-center p-2 text-center text-xs text-neutral-500">
            {movie.title}
          </div>
        )}
      </div>
      <div className="flex flex-1 flex-col gap-1 p-3">
        <h3 className="line-clamp-2 text-sm font-medium">{movie.title}</h3>
        <p className="text-xs text-neutral-500">
          {movie.year ?? "—"} · {genres.slice(0, 2).join(", ")}
        </p>
        {footer}
      </div>
    </div>
  );
}
