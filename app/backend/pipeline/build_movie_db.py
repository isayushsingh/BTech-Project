"""Phase 0, step 3: pack everything into a single SQLite file the API serves from.

Input:  {ARTIFACTS_DIR}/movies_trimmed.parquet
        {ARTIFACTS_DIR}/content_neighbors.json
Output: {ARTIFACTS_DIR}/movies.db
"""
import json
import sqlite3

import pandas as pd

from common import ARTIFACTS_DIR

DB_PATH = ARTIFACTS_DIR / "movies.db"


def main():
    movies = pd.read_parquet(ARTIFACTS_DIR / "movies_trimmed.parquet")
    with open(ARTIFACTS_DIR / "content_neighbors.json") as f:
        neighbors = json.load(f)

    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)

    movies_out = movies.copy()
    movies_out["genres"] = movies_out["genres"].apply(lambda g: json.dumps(list(g)))
    movies_out.to_sql("movies", conn, index=False)
    conn.execute("CREATE INDEX idx_movies_title ON movies (title COLLATE NOCASE)")

    conn.execute(
        """
        CREATE TABLE content_neighbors (
            movie_id INTEGER NOT NULL,
            neighbor_id INTEGER NOT NULL,
            score REAL NOT NULL
        )
        """
    )
    rows = [
        (int(movie_id), neighbor_id, score)
        for movie_id, pairs in neighbors.items()
        for neighbor_id, score in pairs
    ]
    conn.executemany("INSERT INTO content_neighbors VALUES (?, ?, ?)", rows)
    conn.execute("CREATE INDEX idx_neighbors_movie ON content_neighbors (movie_id)")

    conn.commit()
    conn.close()
    print(f"Wrote {len(movies_out)} movies and {len(rows)} neighbor edges to {DB_PATH}")


if __name__ == "__main__":
    main()
