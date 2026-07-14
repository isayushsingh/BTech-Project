"""Serves the precomputed offline evaluation from pipeline/evaluate.py --
nothing is computed at request time, same pattern as the other routes."""
import json
from pathlib import Path

from fastapi import APIRouter

router = APIRouter()

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"


@router.get("/benchmark")
def get_benchmark():
    with open(ARTIFACTS_DIR / "benchmark_results.json") as f:
        return json.load(f)
