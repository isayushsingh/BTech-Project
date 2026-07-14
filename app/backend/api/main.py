import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import benchmark, recommend, search, similar

app = FastAPI(title="Hybrid Movie Recommender API")

allowed_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(search.router, prefix="/api")
app.include_router(similar.router, prefix="/api")
app.include_router(recommend.router, prefix="/api")
app.include_router(benchmark.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok"}
