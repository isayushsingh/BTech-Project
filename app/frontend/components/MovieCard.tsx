import { posterUrl } from "@/lib/api";
import type { Movie } from "@/lib/types";
import PosterImage from "./PosterImage";
import Card from "./ui/Card";

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
    <Card className="flex flex-col overflow-hidden">
      <div className="relative aspect-[2/3] w-full bg-white/[0.04]">
        {src ? (
          <PosterImage src={src} alt={movie.title} />
        ) : (
          <div className="flex h-full items-center justify-center p-2 text-center text-xs text-muted">
            {movie.title}
          </div>
        )}
      </div>
      <div className="flex flex-1 flex-col gap-1 p-3">
        <h3 className="line-clamp-2 font-mono text-sm font-medium">{movie.title}</h3>
        <p className="text-xs text-muted">
          {movie.year ?? "—"} · {genres.slice(0, 2).join(", ")}
        </p>
        {footer}
      </div>
    </Card>
  );
}
