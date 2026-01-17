import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from validation import bootstrap_ci, classification_report_from_scores

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

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=5000, solver="liblinear")
    model.fit(X_tr, y_tr)
    y_score = model.predict_proba(X_te)[:, 1]

    report = classification_report_from_scores(y_te, y_score, threshold=0.5)

    roc_ci = bootstrap_ci(lambda yt, ys: roc_auc_score(yt, ys), y_te, y_score, n_boot=2000)
    pr_ci  = bootstrap_ci(lambda yt, ys: average_precision_score(yt, ys), y_te, y_score, n_boot=2000)

    print("Confusion-matrix-derived report:", report)
    print("ROC-AUC 95% CI:", roc_ci)
    print("PR-AUC 95% CI:", pr_ci)

if __name__ == "__main__":
    main()
