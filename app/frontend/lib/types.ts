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

export type BaselineResult = {
  label: string;
  precision_at_k: number | null;
  recall_at_k: number | null;
  ndcg_at_k: number | null;
  rmse: number | null;
  mae: number | null;
};

export type WeightSweepPoint = {
  w: number;
  precision_at_k: number;
  ndcg_at_k: number | null;
};

export type BenchmarkResults = {
  k: number;
  n_users_evaluated: number;
  n_train_ratings: number;
  n_test_ratings: number;
  relevance_threshold: number;
  production_weight: number;
  best_weight: number;
  weight_sweep: WeightSweepPoint[];
  baselines: Record<string, BaselineResult>;
};
