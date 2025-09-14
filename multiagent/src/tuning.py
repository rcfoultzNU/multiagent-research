
from __future__ import annotations
from typing import Dict, Optional, Tuple, Any
import numpy as np

try:
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, GroupKFold
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

def grid_search_cv(estimator,
                   param_grid: Dict,
                   X: np.ndarray,
                   y: np.ndarray,
                   groups: Optional[np.ndarray] = None,
                   n_splits: int = 3,
                   scoring: str = "average_precision",
                   n_jobs: int = -1,
                   refit: bool = True,
                   random_state: int = 42):
    if not SKLEARN_OK:
        raise RuntimeError("scikit-learn is required for tuning")
    if groups is not None:
        cv = GroupKFold(n_splits=n_splits).split(X, y, groups)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    gs = GridSearchCV(estimator, param_grid=param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs, refit=refit)
    gs.fit(X, y, groups=groups)
    return gs.best_estimator_, gs.cv_results_

def random_search_cv(estimator,
                     param_distributions: Dict,
                     X: np.ndarray,
                     y: np.ndarray,
                     groups: Optional[np.ndarray] = None,
                     n_splits: int = 3,
                     n_iter: int = 50,
                     scoring: str = "average_precision",
                     n_jobs: int = -1,
                     refit: bool = True,
                     random_state: int = 42):
    if not SKLEARN_OK:
        raise RuntimeError("scikit-learn is required for tuning")
    if groups is not None:
        cv = GroupKFold(n_splits=n_splits).split(X, y, groups)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rs = RandomizedSearchCV(estimator,
                            param_distributions=param_distributions,
                            n_iter=n_iter,
                            scoring=scoring,
                            cv=cv,
                            n_jobs=n_jobs,
                            random_state=random_state,
                            refit=refit)
    rs.fit(X, y, groups=groups)
    return rs.best_estimator_, rs.cv_results_
