import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score

from tuning import random_search_cv

DATA_DIR = "data/logs"

def main():
    epi = pd.read_parquet(os.path.join(DATA_DIR, "episodes.parquet"))
    feat = pd.read_parquet(os.path.join(DATA_DIR, "features.parquet"))
    df = epi.merge(feat, on="episode_id", how="left")

    y = df["success"].astype(int).to_numpy()

    drop_cols = {"success", "episode_id", "map_id", "run_id", "seed"}
    X_df = df.drop(columns=[c for c in df.columns if c in drop_cols])
    X_df = pd.get_dummies(X_df, columns=["communication", "disturbance", "heuristic"], drop_first=False)
    X = X_df.to_numpy(dtype=float)

    # 70/30 split (you can change to 80/20 if desired)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # --- Baseline RF
    baseline = RandomForestClassifier(random_state=42)
    baseline.fit(X_tr, y_tr)
    p_base = baseline.predict_proba(X_te)[:, 1]
    pr_base = average_precision_score(y_te, p_base)
    roc_base = roc_auc_score(y_te, p_base)

    # --- Tuned RF via randomized search CV on training set
    param_space = {
        "n_estimators": [100, 200, 400, 800],
        "max_depth": [None, 6, 10, 14, 20],
        "min_samples_split": [2, 4, 8, 16],
        "min_samples_leaf": [1, 2, 4, 8]
    }

    best_model, cv_results = random_search_cv(
        RandomForestClassifier(random_state=42),
        param_space,
        X_tr, y_tr,
        n_splits=3,
        n_iter=25,
        scoring="average_precision"
    )

    best_model.fit(X_tr, y_tr)
    p_tuned = best_model.predict_proba(X_te)[:, 1]
    pr_tuned = average_precision_score(y_te, p_tuned)
    roc_tuned = roc_auc_score(y_te, p_tuned)

    print("\n--- Hyperparameter Optimization (Random Forest) ---")
    print("Best hyperparameters:", best_model.get_params())

    print("\n--- Holdout (70/30) Performance ---")
    print(f"Baseline PR-AUC: {pr_base:.4f} | Tuned PR-AUC: {pr_tuned:.4f} | Delta: {(pr_tuned - pr_base):+.4f}")
    print(f"Baseline ROC-AUC: {roc_base:.4f} | Tuned ROC-AUC: {roc_tuned:.4f} | Delta: {(roc_tuned - roc_base):+.4f}")

if __name__ == "__main__":
    main()
