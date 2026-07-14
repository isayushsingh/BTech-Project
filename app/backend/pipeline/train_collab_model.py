"""Phase 0, step 2: train the collaborative-filtering model.

Trains a regularized matrix factorization (Simon-Funk-style SGD, the same
objective scikit-surprise's default SVD uses) directly in NumPy instead of
depending on scikit-surprise, which needs a C build step that's brittle on
hosted containers (see plan's Risks section). The math -- and the fold-in
approach the live API uses on top of it -- is unaffected by this choice.

Only the pieces needed for *cold-start fold-in* are persisted: the item
factor matrix qi, item biases bi, and the global mean mu. Per-user factors
(pu, bu) from the training set are intentionally not needed at serving time
-- new visitors always go through fold-in (see api/foldin.py).

Input:  {DATA_DIR}/ratings_small.csv
        {ARTIFACTS_DIR}/movies_trimmed.parquet  (to restrict to known movies)
Output: {ARTIFACTS_DIR}/item_factors.npz  (qi, bi, mu, item_ids)
"""
import numpy as np
import pandas as pd

from common import ARTIFACTS_DIR, DATA_DIR

# 20 factors / 50 epochs chosen via pipeline/evaluate.py's held-out
# benchmark: this catalog's ratings_small intersection is small (~1k items,
# ~700 users, ~20k train ratings after an eval split), and the original
# 50-factor/20-epoch setting was measurably overfitting at that scale --
# held-out NDCG@10 roughly doubled after dropping to 20 factors and training
# longer. Re-run evaluate.py if the underlying dataset size changes
# meaningfully, since the right factor count is data-size dependent.
N_FACTORS = 20
N_EPOCHS = 50
LEARNING_RATE = 0.005
REG = 0.02
SEED = 42


def train_svd(ratings: pd.DataFrame, n_items: int, user_index: np.ndarray, item_index: np.ndarray):
    rng = np.random.default_rng(SEED)
    n_users = user_index.max() + 1

    mu = ratings["rating"].mean()
    bu = np.zeros(n_users)
    bi = np.zeros(n_items)
    pu = rng.normal(scale=0.1, size=(n_users, N_FACTORS))
    qi = rng.normal(scale=0.1, size=(n_items, N_FACTORS))

    u = user_index
    i = item_index
    r = ratings["rating"].to_numpy()
    order = np.arange(len(r))

    for epoch in range(N_EPOCHS):
        rng.shuffle(order)
        sq_err = 0.0
        for idx in order:
            uu, ii, rr = u[idx], i[idx], r[idx]
            pred = mu + bu[uu] + bi[ii] + pu[uu] @ qi[ii]
            err = rr - pred
            sq_err += err ** 2

            bu[uu] += LEARNING_RATE * (err - REG * bu[uu])
            bi[ii] += LEARNING_RATE * (err - REG * bi[ii])
            pu_uu = pu[uu].copy()
            pu[uu] += LEARNING_RATE * (err * qi[ii] - REG * pu[uu])
            qi[ii] += LEARNING_RATE * (err * pu_uu - REG * qi[ii])

        rmse = np.sqrt(sq_err / len(r))
        print(f"epoch {epoch + 1}/{N_EPOCHS}  train RMSE={rmse:.4f}")

    return mu, bi, qi


def main():
    movies = pd.read_parquet(ARTIFACTS_DIR / "movies_trimmed.parquet")
    known_ids = set(movies["id"].tolist())

    ratings = pd.read_csv(DATA_DIR / "ratings_small.csv", low_memory=False)
    ratings = ratings[ratings["movieId"].isin(known_ids)].copy()

    item_ids = np.sort(ratings["movieId"].unique())
    item_id_to_index = {mid: idx for idx, mid in enumerate(item_ids)}
    user_ids = ratings["userId"].unique()
    user_id_to_index = {uid: idx for idx, uid in enumerate(user_ids)}

    user_index = ratings["userId"].map(user_id_to_index).to_numpy()
    item_index = ratings["movieId"].map(item_id_to_index).to_numpy()

    mu, bi, qi = train_svd(ratings, n_items=len(item_ids), user_index=user_index, item_index=item_index)

    np.savez(
        ARTIFACTS_DIR / "item_factors.npz",
        qi=qi,
        bi=bi,
        mu=np.array(mu),
        item_ids=item_ids,
    )
    print(f"Wrote item factor artifacts for {len(item_ids)} movies to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
