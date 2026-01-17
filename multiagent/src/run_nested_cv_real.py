import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from validation import nested_cv_classification  # uses your module

DATA_DIR = "data/logs"

def main():
    epi = pd.read_parquet(os.path.join(DATA_DIR, "episodes.parquet"))
    feat = pd.read_parquet(os.path.join(DATA_DIR, "features.parquet"))

    # Merge episode predictors + engineered features
    df = epi.merge(feat, on="episode_id", how="left")

    # Target
    y = df["success"].astype(int).to_numpy()

    # Predictors (exclude IDs + target)
    drop_cols = {"success", "episode_id", "map_id", "run_id", "seed"}
    X_df = df.drop(columns=[c for c in df.columns if c in drop_cols])

    # One-hot encode categoricals
    X_df = pd.get_dummies(X_df, columns=["communication", "disturbance", "heuristic"], drop_first=False)

    X = X_df.to_numpy(dtype=float)

    # Model + hyperparameter space
    model = LogisticRegression(max_iter=5000, solver="liblinear")
    param_grid = {"C": [0.01, 0.1, 1, 10, 100]}

    res = nested_cv_classification(
        model, X, y, param_grid,
        search="grid",
        outer_splits=5,
        inner_splits=3,
        scoring="average_precision"
    )

    # ✅ Fold-level PR-AUC values
    fold_scores = [r.metric_value for r in res]
    print("Fold PR-AUC:", [round(v, 4) for v in fold_scores])

    # ✅ Mean ± SD PR-AUC
    print("Mean PR-AUC:", round(float(np.mean(fold_scores)), 4))
    print("SD PR-AUC:", round(float(np.std(fold_scores, ddof=1)), 4))

    # ✅ Best parameter sets
    print("Best params per fold:")
    for r in res:
        print(f"  Fold {r.outer_fold}: PR-AUC={r.metric_value:.4f}, best_params={r.best_params}")

if __name__ == "__main__":
    main()
