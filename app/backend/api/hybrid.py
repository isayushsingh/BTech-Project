"""Combine content-based and cold-start collaborative signals into one
ranked list -- the live equivalent of the thesis's "weighted average of
content recommendation and collaborative recommendation" (see
presentations/btp_final.pdf, Project Aim slide).
"""
import numpy as np

from .foldin import fold_in_user, predict_collab_scores
from .store import Store

COLLAB_CANDIDATE_POOL = 100


def _normalize(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def get_recommendations(
    store: Store,
    rated: list[tuple[int, float]],
    top_n: int = 12,
    w: float = 0.5,
) -> list[dict]:
    rated_ids = {mid for mid, _ in rated}
    liked = [mid for mid, r in rated if r >= 4]

    # Content candidates: union of each liked movie's precomputed neighbors,
    # scored by the best (max) similarity across all liked movies.
    content_scores: dict[int, float] = {}
    for lid in liked:
        for neighbor_id, score in store.get_neighbors(lid, limit=20):
            if neighbor_id in rated_ids:
                continue
            if score > content_scores.get(neighbor_id, -1.0):
                content_scores[neighbor_id] = score

    # Collaborative candidates: fold the visitor's ratings into the trained
    # SVD's latent space, then take their highest-predicted unseen movies.
    collab_scores: dict[int, float] = {}
    p_u = fold_in_user(rated, store.qi, store.bi, store.mu, store.item_id_to_idx)
    if p_u is not None:
        all_scores = predict_collab_scores(p_u, store.qi, store.bi, store.mu)
        for i in np.argsort(-all_scores)[: COLLAB_CANDIDATE_POOL + len(rated_ids)]:
            movie_id = int(store.item_ids[i])
            if movie_id in rated_ids:
                continue
            collab_scores[movie_id] = float(all_scores[i])
            if len(collab_scores) >= COLLAB_CANDIDATE_POOL:
                break

    candidates = set(content_scores) | set(collab_scores)
    if not candidates:
        return []

    norm_content = _normalize(content_scores)
    norm_collab = _normalize(collab_scores)

    ranked = []
    for movie_id in candidates:
        c = norm_content.get(movie_id, 0.0)
        cf = norm_collab.get(movie_id, 0.0)
        quality = store.normalized_quality(movie_id)
        # Quality gate mirrors the thesis's fix for the "Batman and Robin"
        # problem: don't let a text-similar-but-poorly-reviewed movie win
        # purely on cast/crew overlap.
        blended = (w * cf + (1 - w) * c) * (0.7 + 0.3 * quality)
        ranked.append((movie_id, blended, c, cf))

    ranked.sort(key=lambda row: row[1], reverse=True)
    top = ranked[:top_n]

    movies = store.get_movies([movie_id for movie_id, *_ in top])
    results = []
    for movie_id, blended, c, cf in top:
        movie = movies.get(movie_id)
        if not movie:
            continue
        results.append(
            {
                "movie": movie,
                "content_score": round(c, 4),
                "collab_score": round(cf, 4),
                "blended_score": round(blended, 4),
            }
        )
    return results
