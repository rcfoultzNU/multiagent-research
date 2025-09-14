
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

try:
    from sklearn.metrics import (
        confusion_matrix, matthews_corrcoef, roc_auc_score, average_precision_score,
        accuracy_score, precision_score, recall_score, f1_score,
        mean_squared_error, mean_absolute_error
    )
    from sklearn.model_selection import StratifiedKFold, GroupKFold, GridSearchCV, RandomizedSearchCV
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    import matplotlib.pyplot as plt
    MPL_OK = True
except Exception:
    MPL_OK = False

def classification_report_from_scores(y_true: np.ndarray,
                                      y_pred_proba: np.ndarray,
                                      threshold: float = 0.5) -> Dict[str, float]:
    if not SKLEARN_OK:
        raise RuntimeError("scikit-learn is required for classification metrics")
    y_pred = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
        "pr_auc": float(average_precision_score(y_true, y_pred_proba))
    }

def regression_report(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      log_sensitivity: bool = False) -> Dict[str, float]:
    if not SKLEARN_OK:
        raise RuntimeError("scikit-learn is required for regression metrics")
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    out = {"rmse": rmse, "mae": mae}
    if log_sensitivity:
        eps = 1e-9
        y_true_l = np.log(y_true + eps)
        y_pred_l = np.log(y_pred + eps)
        from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae_fn
        out["rmse_log"] = float(np.sqrt(mse(y_true_l, y_pred_l)))
        out["mae_log"] = float(mae_fn(y_true_l, y_pred_l))
    return out

def bootstrap_ci(metric_fn: Callable[[np.ndarray, np.ndarray], float],
                 y_true: np.ndarray,
                 y_score: np.ndarray,
                 n_boot: int = 1000,
                 alpha: float = 0.05,
                 random_state: Optional[int] = 1234) -> Tuple[float, float]:
    if not SKLEARN_OK:
        raise RuntimeError("scikit-learn is required for bootstrap metrics")
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = metric_fn(y_true[idx], y_score[idx])
    lo = float(np.percentile(vals, 100*alpha/2))
    hi = float(np.percentile(vals, 100*(1 - alpha/2)))
    return lo, hi

@dataclass
class CVResult:
    outer_fold: int
    metric_name: str
    metric_value: float
    best_params: Optional[Dict] = None

def nested_cv_classification(estimator,
                             X: np.ndarray,
                             y: np.ndarray,
                             param_grid: Dict,
                             groups: Optional[np.ndarray] = None,
                             outer_splits: int = 5,
                             inner_splits: int = 3,
                             search: str = "random",
                             n_iter: int = 25,
                             random_state: int = 42,
                             scoring: str = "average_precision") -> List[CVResult]:
    if not SKLEARN_OK:
        raise RuntimeError("scikit-learn is required for nested CV")
    if groups is not None:
        outer = GroupKFold(n_splits=outer_splits)
        outer_iter = outer.split(X, y, groups=groups)
    else:
        outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
        outer_iter = outer.split(X, y)
    results: List[CVResult] = []
    fold = 0
    for train_idx, test_idx in outer_iter:
        fold += 1
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        g_tr = groups[train_idx] if groups is not None else None
        if groups is not None:
            inner = {"cv": GroupKFold(n_splits=inner_splits).split(X_tr, y_tr, groups=g_tr)}
        else:
            inner = {"cv": StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)}
        if search == "grid":
            searcher = GridSearchCV(estimator, param_grid=param_grid, scoring=scoring, **inner, n_jobs=-1)
        else:
            searcher = RandomizedSearchCV(estimator, param_distributions=param_grid, n_iter=n_iter,
                                          scoring=scoring, **inner, n_jobs=-1, random_state=random_state)
        searcher.fit(X_tr, y_tr, groups=g_tr)
        best_model = searcher.best_estimator_
        from sklearn.metrics import average_precision_score
        y_score = best_model.predict_proba(X_te)[:, 1] if hasattr(best_model, "predict_proba") else best_model.decision_function(X_te)
        pr_auc = float(average_precision_score(y_te, y_score))
        results.append(CVResult(outer_fold=fold, metric_name="pr_auc", metric_value=pr_auc, best_params=searcher.best_params_))
    return results
