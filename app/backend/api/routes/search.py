from fastapi import APIRouter, Query

from ..store import store

router = APIRouter()


@router.get("/movies/search")
def search_movies(q: str = Query(min_length=1), limit: int = 10):
    return {"results": store.search(q, limit=limit)}
