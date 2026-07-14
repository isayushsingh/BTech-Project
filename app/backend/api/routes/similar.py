"""Pure content-based lookup -- always available, no rating required. This is
the always-works fallback demo path described in the plan."""
from fastapi import APIRouter, HTTPException

from ..store import store

router = APIRouter()


@router.get("/movies/{movie_id}/similar")
def similar_movies(movie_id: int, limit: int = 12):
    seed = store.get_movies([movie_id]).get(movie_id)
    if seed is None:
        raise HTTPException(status_code=404, detail="movie not found")

    neighbors = store.get_neighbors(movie_id, limit=limit)
    neighbor_movies = store.get_movies([nid for nid, _ in neighbors])

    results = []
    for neighbor_id, score in neighbors:
        movie = neighbor_movies.get(neighbor_id)
        if movie:
            results.append({"movie": movie, "content_score": round(score, 4)})

    return {"seed": seed, "results": results}
