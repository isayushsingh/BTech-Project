export type Movie = {
  id: number;
  title: string;
  genres: string; // JSON-encoded string array, decode with JSON.parse
  poster_path: string | null;
  year: number | null;
  vote_average: number;
  vote_count: number;
  weighted_rating: number;
};

export type SearchResult = {
  id: number;
  title: string;
  poster_path: string | null;
  year: number | null;
};

export type SimilarResult = {
  movie: Movie;
  content_score: number;
};

export type RecommendResult = {
  movie: Movie;
  content_score: number;
  collab_score: number;
  blended_score: number;
};

export type RatedMovie = {
  movie_id: number;
  title: string;
  poster_path: string | null;
  rating: number;
};
