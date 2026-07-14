"""Cold-start fold-in: turn a handful of a visitor's ratings into an
approximate latent vector in the trained SVD's item-factor space, without
retraining. See the plan's Phase 1 section for the derivation.

p_u = (Q^T Q + lambda*I)^-1 Q^T (r - mu - b_i)

where Q is the matrix of item factor rows for the movies the visitor rated.
This solves the same regularized objective the model was trained with
(see pipeline/train_collab_model.py), so the fold-in stays internally
consistent with the fixed qi/bi/mu it's built on top of.
"""
from __future__ import annotations

import numpy as np

REG_LAMBDA = 0.1


def fold_in_user(
    rated: list[tuple[int, float]],
    qi: np.ndarray,
    bi: np.ndarray,
    mu: float,
    item_id_to_idx: dict[int, int],
) -> np.ndarray | None:
    idx = [item_id_to_idx[mid] for mid, _ in rated if mid in item_id_to_idx]
    if not idx:
        return None

    ratings = np.array([r for mid, r in rated if mid in item_id_to_idx])
    Q = qi[idx]
    residual = ratings - mu - bi[idx]
    n_factors = qi.shape[1]
    return np.linalg.solve(Q.T @ Q + REG_LAMBDA * np.eye(n_factors), Q.T @ residual)


def predict_collab_scores(p_u: np.ndarray, qi: np.ndarray, bi: np.ndarray, mu: float) -> np.ndarray:
    """Predicted rating for every item in the catalog, given a (possibly folded-in) user vector."""
    return mu + bi + qi @ p_u
