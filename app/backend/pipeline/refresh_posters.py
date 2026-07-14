"""Phase 0 (data quality) addendum: refresh poster_path from TMDB's API.

The Kaggle "Movies Dataset" is a ~2017 snapshot. TMDB periodically garbage
collects old image file hashes from its CDN, so a meaningful fraction of the
poster_path values in movies_trimmed.parquet now 404. Since we already have
each movie's TMDB id, we can just ask TMDB for the current poster_path.

Requires the TMDB_API_KEY env var (v3 auth key from
themoviedb.org/settings/api). Never hardcode the key in this file.

Input/Output: {ARTIFACTS_DIR}/movies_trimmed.parquet (updated in place)
Run this before build_movie_db.py (or rerun build_movie_db.py after).
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

import dns_fix  # noqa: F401 -- patches socket.getaddrinfo before any requests are made
from common import ARTIFACTS_DIR

TMDB_API_KEY = os.environ["TMDB_API_KEY"]
TMDB_MOVIE_URL = "https://api.themoviedb.org/3/movie/{id}"
MAX_WORKERS = 8
TIMEOUT = 10


def fetch_poster_path(movie_id: int) -> tuple[int, str | None]:
    for attempt in range(3):
        try:
            resp = requests.get(
                TMDB_MOVIE_URL.format(id=movie_id),
                params={"api_key": TMDB_API_KEY},
                timeout=TIMEOUT,
            )
        except requests.RequestException:
            time.sleep(1 + attempt)
            continue

        if resp.status_code == 429:
            time.sleep(int(resp.headers.get("Retry-After", "2")))
            continue
        if resp.status_code != 200:
            return movie_id, None

        return movie_id, resp.json().get("poster_path")

    return movie_id, None


def main():
    movies = pd.read_parquet(ARTIFACTS_DIR / "movies_trimmed.parquet")
    ids = movies["id"].tolist()

    updated = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_poster_path, mid): mid for mid in ids}
        done = 0
        for future in as_completed(futures):
            movie_id, poster_path = future.result()
            if poster_path:
                updated[movie_id] = poster_path
            done += 1
            if done % 500 == 0:
                print(f"{done}/{len(ids)} movies checked")

    movies["poster_path"] = movies.apply(
        lambda row: updated.get(row["id"], row["poster_path"]), axis=1
    )
    movies.to_parquet(ARTIFACTS_DIR / "movies_trimmed.parquet", index=False)
    print(f"Refreshed poster paths for {len(updated)}/{len(ids)} movies")


if __name__ == "__main__":
    main()
