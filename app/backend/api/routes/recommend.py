from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, conlist

from ..hybrid import get_recommendations
from ..store import store

router = APIRouter()


class RatedMovie(BaseModel):
    movie_id: int
    rating: float = Field(ge=0.5, le=5)


class RecommendRequest(BaseModel):
    ratings: conlist(RatedMovie, min_length=1, max_length=30)


@router.post("/recommend")
def recommend(body: RecommendRequest):
    rated = [(r.movie_id, r.rating) for r in body.ratings]
    known = store.get_movies([mid for mid, _ in rated])
    if not known:
        raise HTTPException(status_code=400, detail="none of the rated movies are in the catalog")

    results = get_recommendations(store, rated, top_n=12)
    return {"results": results}
