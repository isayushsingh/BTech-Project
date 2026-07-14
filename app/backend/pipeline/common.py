"""Shared paths and helpers for the offline artifact-building pipeline.

Run order: build_content_artifacts.py -> train_collab_model.py -> build_movie_db.py
Each script reads/writes small intermediate files under ARTIFACTS_DIR so they
can be rerun independently once the raw CSVs are in DATA_DIR.
"""
import os
import re
import json
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("MOVIE_DATA_DIR", BACKEND_DIR / "data"))
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Keep only movies with at least this many votes -- trims 45k movies down to a
# deployable subset and matches the "qualified" threshold from the original
# recommendation/popularity_rec.py.
MIN_VOTE_COUNT_QUANTILE = 0.90
MAX_MOVIES = 20000


def convert_ids(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def to_json_list(cell) -> list:
    """Parse the stringified-python-list columns (genres, cast, crew, keywords)."""
    if not isinstance(cell, str):
        return []
    try:
        return json.loads(re.sub(r"'", '"', cell))
    except (json.JSONDecodeError, TypeError):
        try:
            from ast import literal_eval

            return literal_eval(cell)
        except (ValueError, SyntaxError):
            return []


def weighted_rating(v: float, r: float, m: float, c: float) -> float:
    """IMDB weighted rating formula, same as recommendation/popularity_rec.py."""
    if v + m == 0:
        return 0.0
    return (v / (v + m) * r) + (m / (v + m) * c)
