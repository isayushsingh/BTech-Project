"""Benchmark the hybrid recommender against standard baselines on a held-out
split of the same MovieLens data everything else in this repo trains on.

Reuses the *exact* production fold-in/hybrid code (api/foldin.py,
api/hybrid.py) for the SVD-only, content-only, and hybrid baselines, so this
measures what's actually deployed -- not a reimplementation. Only the
popularity and item-KNN baselines are new code, since nothing in api/
implements them.

Split: per-user 80/20 holdout, so every test user still has train-set
ratings to fold in with -- mirrors exactly what a real cold-start visitor
does, just against a user we happen to have held-out ground truth for.

Output: {ARTIFACTS_DIR}/benchmark_results.json
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))  # so `api.*` imports work from pipeline/

from api.foldin import fold_in_user, predict_collab_scores  # noqa: E402
from api.hybrid import get_recommendations  # noqa: E402
from common import ARTIFACTS_DIR, DATA_DIR  # noqa: E402
from train_collab_model import train_svd  # noqa: E402

K = 10
MIN_RATINGS_PER_USER = 10
TEST_FRACTION = 0.2
RELEVANCE_THRESHOLD = 4.0
SEED = 42
HYBRID_WEIGHT = 0.7  # current production default (api/hybrid.py) -- swept below
WEIGHT_SWEEP = [round(w, 1) for w in np.arange(0.0, 1.01, 0.1)]


class EvalStore:
    """Same read interface as api.store.Store, but backed by train-split-only
    collaborative factors so api.hybrid.get_recommendations() can run
    unmodified against it without leaking test ratings into training."""

    def __init__(self, qi, bi, mu, item_ids):
        self.conn = sqlite3.connect(ARTIFACTS_DIR / "movies.db")
        self.conn.row_factory = sqlite3.Row
        self.qi = qi
        self.bi = bi
        self.mu = mu
        self.item_ids = item_ids
        self.item_id_to_idx = {int(mid): i for i, mid in enumerate(item_ids)}

        rows = self.conn.execute("SELECT id, weighted_rating FROM movies").fetchall()
        self.weighted_rating_by_id = {row["id"]: row["weighted_rating"] for row in rows}
        wrs = list(self.weighted_rating_by_id.values())
        self.wr_min, self.wr_max = min(wrs), max(wrs)

    def get_movies(self, ids):
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        rows = self.conn.execute(f"SELECT * FROM movies WHERE id IN ({placeholders})", ids).fetchall()
        return {row["id"]: dict(row) for row in rows}

    def get_neighbors(self, movie_id, limit=20):
        rows = self.conn.execute(
            "SELECT neighbor_id, score FROM content_neighbors WHERE movie_id=? ORDER BY score DESC LIMIT ?",
            (movie_id, limit),
        ).fetchall()
        return [(row["neighbor_id"], row["score"]) for row in rows]

    def normalized_quality(self, movie_id):
        wr = self.weighted_rating_by_id.get(movie_id)
        if wr is None or self.wr_max == self.wr_min:
            return 0.5
        return (wr - self.wr_min) / (self.wr_max - self.wr_min)


def split_train_test(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    train_rows, test_rows = [], []
    for _, group in ratings.groupby("userId"):
        if len(group) < MIN_RATINGS_PER_USER:
            train_rows.append(group)
            continue
        shuffled = group.sample(frac=1, random_state=rng.integers(1 << 30))
        n_test = max(1, int(len(shuffled) * TEST_FRACTION))
        test_rows.append(shuffled.iloc[:n_test])
        train_rows.append(shuffled.iloc[n_test:])
    return pd.concat(train_rows, ignore_index=True), pd.concat(test_rows, ignore_index=True)


SIGNIFICANCE_SHRINKAGE = 10  # standard co-rating shrinkage constant


def train_item_knn(train: pd.DataFrame, item_ids: np.ndarray):
    """Classic item-based collaborative filtering: dense item-item cosine
    similarity over the train ratings matrix. Small enough at this scale
    (~1-2k items) that a dense matrix is simpler and fast enough.

    Raw cosine similarity is unreliable for items with very few ratings --
    an item rated by exactly one user is trivially "maximally similar" to
    whatever else that user rated, which floods naive item-KNN with
    spurious 1-rating matches. Standard fix: significance-weight similarity
    by co-rating support, so pairs with few shared raters get shrunk
    toward zero instead of dominating the ranking.
    """
    item_id_to_idx = {int(mid): i for i, mid in enumerate(item_ids)}
    user_ids = train["userId"].unique()
    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}

    matrix = np.zeros((len(item_ids), len(user_ids)))
    for row in train.itertuples():
        matrix[item_id_to_idx[row.movieId], user_id_to_idx[row.userId]] = row.rating

    similarity = cosine_similarity(matrix)
    np.fill_diagonal(similarity, 0.0)

    rated_mask = (matrix > 0).astype(float)
    co_counts = rated_mask @ rated_mask.T
    shrinkage = co_counts / (co_counts + SIGNIFICANCE_SHRINKAGE)
    similarity *= shrinkage

    return similarity, item_id_to_idx, matrix


def item_knn_scores(
    rated: list[tuple[int, float]], similarity, item_id_to_idx, n_items, global_mean: float
) -> np.ndarray:
    scores = np.zeros(n_items)
    weights = np.zeros(n_items)
    for mid, rating in rated:
        idx = item_id_to_idx.get(mid)
        if idx is None:
            continue
        sim_row = similarity[idx]
        scores += sim_row * rating
        weights += np.abs(sim_row)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.divide(scores, weights, out=np.full_like(scores, global_mean), where=weights > 0)
    # A weighted average alone is invariant to uniform similarity shrinkage --
    # one neighbor at any similarity still averages to exactly that neighbor's
    # rating. Damp toward the global mean based on total confidence (summed
    # similarity weight), so a prediction backed by only one low-support
    # neighbor doesn't get to look as confident as one backed by many.
    predicted = (weights * raw + SIGNIFICANCE_SHRINKAGE * global_mean) / (weights + SIGNIFICANCE_SHRINKAGE)
    return predicted


def popularity_scores(train: pd.DataFrame, item_ids: np.ndarray) -> dict[int, float]:
    stats = train.groupby("movieId")["rating"].agg(["mean", "count"])
    c = train["rating"].mean()
    m = stats["count"].median()
    weighted = (stats["count"] / (stats["count"] + m)) * stats["mean"] + (m / (stats["count"] + m)) * c
    return {int(mid): float(weighted.get(mid, c)) for mid in item_ids}


def ranking_metrics(ranked_ids: list[int], relevant: set[int], k: int) -> tuple[float, float, float]:
    top_k = ranked_ids[:k]
    hits = [1.0 if mid in relevant else 0.0 for mid in top_k]
    precision = sum(hits) / k
    recall = sum(hits) / len(relevant) if relevant else None

    dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
    ideal_hits = [1.0] * min(len(relevant), k)
    idcg = sum(h / np.log2(i + 2) for i, h in enumerate(ideal_hits))
    ndcg = dcg / idcg if idcg > 0 else None

    return precision, recall, ndcg


def rmse_mae(pairs: list[tuple[float, float]]) -> tuple[float, float]:
    if not pairs:
        return None, None
    errors = np.array([pred - actual for pred, actual in pairs])
    return float(np.sqrt(np.mean(errors**2))), float(np.mean(np.abs(errors)))


def main():
    movies = pd.read_parquet(ARTIFACTS_DIR / "movies_trimmed.parquet")
    known_ids = set(movies["id"].tolist())

    ratings = pd.read_csv(DATA_DIR / "ratings_small.csv", low_memory=False)
    ratings = ratings[ratings["movieId"].isin(known_ids)].copy()

    train, test = split_train_test(ratings)
    print(f"train: {len(train)} ratings, test: {len(test)} ratings")

    item_ids = np.sort(train["movieId"].unique())
    item_id_to_idx = {int(mid): i for i, mid in enumerate(item_ids)}
    user_id_to_idx = {uid: i for i, uid in enumerate(train["userId"].unique())}
    user_index = train["userId"].map(user_id_to_idx).to_numpy()
    item_index = train["movieId"].map(item_id_to_idx).to_numpy()

    print("training benchmark SVD on train split only...")
    mu, bi, qi = train_svd(train, n_items=len(item_ids), user_index=user_index, item_index=item_index)
    eval_store = EvalStore(qi, bi, mu, item_ids)

    print("training item-KNN baseline...")
    similarity, knn_item_to_idx, _ = train_item_knn(train, item_ids)

    pop_scores = popularity_scores(train, item_ids)
    pop_ranked_all = [mid for mid, _ in sorted(pop_scores.items(), key=lambda kv: kv[1], reverse=True)]

    train_by_user = {uid: list(zip(g["movieId"], g["rating"])) for uid, g in train.groupby("userId")}
    test_by_user = {uid: list(zip(g["movieId"], g["rating"])) for uid, g in test.groupby("userId")}

    results = {
        name: {"precision": [], "recall": [], "ndcg": [], "rmse_pairs": []}
        for name in ["popularity", "item_knn", "svd_only", "content_only", "hybrid"]
    }

    n_evaluated = 0
    for uid, test_ratings in test_by_user.items():
        rated = train_by_user.get(uid)
        if not rated:
            continue
        rated_ids = {mid for mid, _ in rated}
        relevant = {mid for mid, r in test_ratings if r >= RELEVANCE_THRESHOLD}
        n_evaluated += 1

        # -- popularity --
        ranked = [mid for mid in pop_ranked_all if mid not in rated_ids]
        p, r, n = ranking_metrics(ranked, relevant, K)
        results["popularity"]["precision"].append(p)
        if r is not None:
            results["popularity"]["recall"].append(r)
        if n is not None:
            results["popularity"]["ndcg"].append(n)
        for mid, actual in test_ratings:
            if mid in pop_scores:
                results["popularity"]["rmse_pairs"].append((pop_scores[mid], actual))

        # -- item-KNN --
        knn_scores = item_knn_scores(rated, similarity, knn_item_to_idx, len(item_ids), global_mean=mu)
        order = np.argsort(-knn_scores)
        ranked = [int(item_ids[i]) for i in order if int(item_ids[i]) not in rated_ids][:K]
        p, r, n = ranking_metrics(ranked, relevant, K)
        results["item_knn"]["precision"].append(p)
        if r is not None:
            results["item_knn"]["recall"].append(r)
        if n is not None:
            results["item_knn"]["ndcg"].append(n)
        for mid, actual in test_ratings:
            idx = knn_item_to_idx.get(mid)
            if idx is not None:
                results["item_knn"]["rmse_pairs"].append((knn_scores[idx], actual))

        # -- SVD only (direct fold-in, no content blending) --
        p_u = fold_in_user(rated, qi, bi, mu, item_id_to_idx)
        if p_u is not None:
            all_scores = predict_collab_scores(p_u, qi, bi, mu)
            order = np.argsort(-all_scores)
            ranked = [int(item_ids[i]) for i in order if int(item_ids[i]) not in rated_ids][:K]
            p, r, n = ranking_metrics(ranked, relevant, K)
            results["svd_only"]["precision"].append(p)
            if r is not None:
                results["svd_only"]["recall"].append(r)
            if n is not None:
                results["svd_only"]["ndcg"].append(n)
            for mid, actual in test_ratings:
                idx = item_id_to_idx.get(mid)
                if idx is not None:
                    results["svd_only"]["rmse_pairs"].append((float(all_scores[idx]), actual))

        # -- content only (hybrid w=0) and hybrid (production default w) via production code --
        for name, w in [("content_only", 0.0), ("hybrid", HYBRID_WEIGHT)]:
            recs = get_recommendations(eval_store, rated, top_n=K, w=w)
            ranked = [rec["movie"]["id"] for rec in recs]
            p, r, n = ranking_metrics(ranked, relevant, K)
            results[name]["precision"].append(p)
            if r is not None:
                results[name]["recall"].append(r)
            if n is not None:
                results[name]["ndcg"].append(n)

    print(f"evaluated {n_evaluated} users")

    print("sweeping hybrid blend weight...")
    weight_sweep = []
    for w in WEIGHT_SWEEP:
        precisions, ndcgs = [], []
        for uid, test_ratings in test_by_user.items():
            rated = train_by_user.get(uid)
            if not rated:
                continue
            relevant = {mid for mid, r in test_ratings if r >= RELEVANCE_THRESHOLD}
            recs = get_recommendations(eval_store, rated, top_n=K, w=w)
            ranked = [rec["movie"]["id"] for rec in recs]
            p, _, n = ranking_metrics(ranked, relevant, K)
            precisions.append(p)
            if n is not None:
                ndcgs.append(n)
        weight_sweep.append(
            {
                "w": w,
                "precision_at_k": round(float(np.mean(precisions)), 4),
                "ndcg_at_k": round(float(np.mean(ndcgs)), 4) if ndcgs else None,
            }
        )
        print(f"  w={w:.1f}  precision={weight_sweep[-1]['precision_at_k']}  ndcg={weight_sweep[-1]['ndcg_at_k']}")
    best = max(weight_sweep, key=lambda row: row["ndcg_at_k"] or 0)
    print(f"best weight by ndcg: w={best['w']}")

    output = {
        "k": K,
        "n_users_evaluated": n_evaluated,
        "n_train_ratings": len(train),
        "n_test_ratings": len(test),
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "production_weight": HYBRID_WEIGHT,
        "weight_sweep": weight_sweep,
        "best_weight": best["w"],
        "baselines": {},
    }
    labels = {
        "popularity": "Popularity",
        "item_knn": "Item-based KNN",
        "svd_only": "SVD only (Netflix Prize-style)",
        "content_only": "Content-based only",
        "hybrid": "Hybrid (content + collaborative)",
    }
    for name, m in results.items():
        rmse, mae = rmse_mae(m["rmse_pairs"])
        output["baselines"][name] = {
            "label": labels[name],
            "precision_at_k": round(float(np.mean(m["precision"])), 4) if m["precision"] else None,
            "recall_at_k": round(float(np.mean(m["recall"])), 4) if m["recall"] else None,
            "ndcg_at_k": round(float(np.mean(m["ndcg"])), 4) if m["ndcg"] else None,
            "rmse": round(rmse, 4) if rmse is not None else None,
            "mae": round(mae, 4) if mae is not None else None,
        }
        print(name, output["baselines"][name])

    with open(ARTIFACTS_DIR / "benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"wrote {ARTIFACTS_DIR / 'benchmark_results.json'}")


if __name__ == "__main__":
    main()
