"""Phase 0, step 1: build content-based similarity artifacts.

Adapts the soup-building + CountVectorizer + cosine-similarity approach from
recommendation/improved_content.py, but instead of one full NxN similarity
matrix (too large to ship), stores only each movie's top-20 nearest
neighbors, and instead of printing recommendations, writes them to disk for
the API to load at startup.

Input:  {DATA_DIR}/movies_metadata.csv, credits.csv, keywords.csv
Output: {ARTIFACTS_DIR}/movies_trimmed.parquet
        {ARTIFACTS_DIR}/content_neighbors.json   {movie_id: [[neighbor_id, score], ...]}
"""
import json
from ast import literal_eval

import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

from common import ARTIFACTS_DIR, DATA_DIR, MAX_MOVIES, MIN_VOTE_COUNT_QUANTILE, weighted_rating

N_NEIGHBORS = 20


def load_and_trim_movies() -> pd.DataFrame:
    md = pd.read_csv(DATA_DIR / "movies_metadata.csv", low_memory=False)
    md = md[pd.to_numeric(md["id"], errors="coerce").notna()].copy()
    md["id"] = md["id"].astype(int)
    md = md.drop_duplicates(subset=["id"])

    md["vote_count"] = pd.to_numeric(md["vote_count"], errors="coerce").fillna(0)
    md["vote_average"] = pd.to_numeric(md["vote_average"], errors="coerce").fillna(0)
    md["genres"] = (
        md["genres"].fillna("[]").apply(literal_eval).apply(lambda x: [i["name"] for i in x] if isinstance(x, list) else [])
    )
    md["year"] = pd.to_datetime(md["release_date"], errors="coerce").dt.year

    c = md["vote_average"].mean()
    m = md["vote_count"].quantile(MIN_VOTE_COUNT_QUANTILE)
    md = md[md["vote_count"] >= m].copy()
    md["weighted_rating"] = md.apply(lambda x: weighted_rating(x["vote_count"], x["vote_average"], m, c), axis=1)

    md = md.sort_values("vote_count", ascending=False).head(MAX_MOVIES).reset_index(drop=True)
    return md


def build_soup(md: pd.DataFrame) -> pd.Series:
    credits = pd.read_csv(DATA_DIR / "credits.csv", low_memory=False).drop_duplicates(subset=["id"])
    keywords = pd.read_csv(DATA_DIR / "keywords.csv", low_memory=False).drop_duplicates(subset=["id"])
    credits["id"] = credits["id"].astype(int)
    keywords["id"] = keywords["id"].astype(int)

    md = md.merge(credits, on="id", how="left").merge(keywords, on="id", how="left")
    md["cast"] = md["cast"].fillna("[]").apply(literal_eval)
    md["crew"] = md["crew"].fillna("[]").apply(literal_eval)
    md["keywords"] = md["keywords"].fillna("[]").apply(literal_eval)

    def get_director(crew):
        for member in crew:
            if member.get("job") == "Director":
                return member.get("name", "")
        return ""

    def clean(tokens):
        return [str(t).lower().replace(" ", "") for t in tokens]

    md["director"] = md["crew"].apply(get_director)
    md["cast"] = md["cast"].apply(lambda x: [i["name"] for i in x][:3])

    keyword_counts = pd.Series([kw["name"] for kws in md["keywords"] for kw in kws]).value_counts()
    frequent_keywords = set(keyword_counts[keyword_counts > 1].index)
    stemmer = SnowballStemmer("english")

    md["keywords"] = md["keywords"].apply(
        lambda kws: [stemmer.stem(kw["name"]) for kw in kws if kw["name"] in frequent_keywords]
    )

    soup = (
        md["keywords"].apply(clean)
        + md["cast"].apply(clean)
        + md["director"].apply(lambda d: [d.lower().replace(" ", "")] * 3)
        + md["genres"].apply(clean)
    )
    return md, soup.apply(lambda tokens: " ".join(tokens))


def main():
    md = load_and_trim_movies()
    md, soup = build_soup(md)

    count = CountVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, stop_words="english")
    count_matrix = count.fit_transform(soup)

    # Sparse, cosine-metric nearest neighbors -- avoids ever materializing a
    # full NxN similarity matrix, which would be too large to fit in memory
    # or ship as an artifact at this scale.
    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=N_NEIGHBORS + 1)
    nn.fit(count_matrix)
    distances, indices = nn.kneighbors(count_matrix)

    ids = md["id"].to_numpy()
    neighbors = {}
    for row_idx, movie_id in enumerate(ids):
        neighbor_ids = ids[indices[row_idx]]
        scores = 1 - distances[row_idx]
        pairs = [
            [int(nid), round(float(score), 4)]
            for nid, score in zip(neighbor_ids, scores)
            if nid != movie_id
        ][:N_NEIGHBORS]
        neighbors[int(movie_id)] = pairs

    out_cols = ["id", "title", "genres", "poster_path", "year", "vote_average", "vote_count", "weighted_rating"]
    md[out_cols].to_parquet(ARTIFACTS_DIR / "movies_trimmed.parquet", index=False)
    with open(ARTIFACTS_DIR / "content_neighbors.json", "w") as f:
        json.dump(neighbors, f)

    print(f"Wrote {len(md)} movies and neighbor lists to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
