"""Loads Phase 0 artifacts once at process startup and serves them to routes.

Everything here is read-only precomputed data (SQLite for movie
metadata/content neighbors, a .npz for the collaborative item factors) --
no training or heavy computation happens at request time.
"""
import sqlite3
from pathlib import Path

import numpy as np

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


class Store:
    def __init__(self):
        self.conn = sqlite3.connect(ARTIFACTS_DIR / "movies.db", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        factors = np.load(ARTIFACTS_DIR / "item_factors.npz")
        self.qi = factors["qi"]
        self.bi = factors["bi"]
        self.mu = float(factors["mu"])
        self.item_ids = factors["item_ids"]
        self.item_id_to_idx = {int(mid): i for i, mid in enumerate(self.item_ids)}

        rows = self.conn.execute("SELECT id, weighted_rating FROM movies").fetchall()
        self.weighted_rating_by_id = {row["id"]: row["weighted_rating"] for row in rows}
        wrs = list(self.weighted_rating_by_id.values())
        self.wr_min, self.wr_max = min(wrs), max(wrs)

    def search(self, query: str, limit: int = 10) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, title, poster_path, year FROM movies WHERE title LIKE ? "
            "ORDER BY vote_count DESC LIMIT ?",
            (f"%{query}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_movies(self, ids: list[int]) -> dict[int, dict]:
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT * FROM movies WHERE id IN ({placeholders})", ids
        ).fetchall()
        return {row["id"]: dict(row) for row in rows}

    def get_neighbors(self, movie_id: int, limit: int = 20) -> list[tuple[int, float]]:
        rows = self.conn.execute(
            "SELECT neighbor_id, score FROM content_neighbors WHERE movie_id=? "
            "ORDER BY score DESC LIMIT ?",
            (movie_id, limit),
        ).fetchall()
        return [(row["neighbor_id"], row["score"]) for row in rows]

    def normalized_quality(self, movie_id: int) -> float:
        wr = self.weighted_rating_by_id.get(movie_id)
        if wr is None or self.wr_max == self.wr_min:
            return 0.5
        return (wr - self.wr_min) / (self.wr_max - self.wr_min)


store = Store()
