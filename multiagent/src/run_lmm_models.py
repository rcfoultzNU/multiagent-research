# run_lmm_models.py
# Linear Mixed-Effects Models (LMM) + diagnostics + fit statistics
# Robust to statsmodels summary-table formats AND convergence issues
#
# Outputs:
#   model_outputs/lmm_makespan_coefficients.csv
#   model_outputs/lmm_log_node_expansions_coefficients.csv
#   model_outputs/lmm_fit_statistics.csv
#   model_outputs/lmm_makespan_qqplot.png
#   model_outputs/lmm_makespan_resid_vs_fitted.png
#   model_outputs/lmm_log_node_expansions_qqplot.png
#   model_outputs/lmm_log_node_expansions_resid_vs_fitted.png
#   model_outputs/lmm_makespan_model_summary.txt
#   model_outputs/lmm_log_node_expansions_model_summary.txt
#   model_outputs/lmm_fit_attempts.csv  (which optimizers were tried, converged or not)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf

DATA_DIR = "data/logs"
OUT_DIR = "model_outputs"


def summary_table_to_df(table) -> pd.DataFrame:
    """
    statsmodels summary tables vary by version:
      - sometimes pandas DataFrame
      - sometimes statsmodels SimpleTable with .data
    This helper makes extraction robust.
    """
    if isinstance(table, pd.DataFrame):
        return table.copy()
    return pd.DataFrame(table.data[1:], columns=table.data[0])


def qqplot_resid(resid: np.ndarray, outpath: str, title: str):
    plt.figure()
    stats.probplot(resid, dist="norm", plot=plt)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def resid_vs_fitted(fitted: np.ndarray, resid: np.ndarray, outpath: str, title: str):
    plt.figure()
    plt.scatter(fitted, resid)
    plt.axhline(0)
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def fit_mixedlm_with_fallbacks(model, *,
                              reml: bool = False,
                              maxiter: int = 2000,
                              optimizers=None,
                              tag: str = "model"):

    """
    Try multiple optimizers; return (best_result, attempts_df).
    This helps avoid common MixedLM convergence failures (small N, boundary solutions).
    """
    if optimizers is None:
        optimizers = ["lbfgs", "bfgs", "cg", "powell", "nm"]

    attempts = []
    best_res = None

    fit_kws = dict(
        reml=reml,
        maxiter=maxiter,
        full_output=True,
        disp=False
    )

    for meth in optimizers:
        try:
            res = model.fit(method=meth, **fit_kws)
            converged = bool(getattr(res, "converged", True))
            llf = float(getattr(res, "llf", np.nan))
            aic = float(getattr(res, "aic", np.nan))
            bic = float(getattr(res, "bic", np.nan))
            grad = np.nan
            # Some versions expose mle_retvals with gradient / iterations
            try:
                mr = getattr(res, "mle_retvals", {}) or {}
                grad = mr.get("grad", np.nan)
            except Exception:
                pass

            attempts.append({
                "tag": tag,
                "optimizer": meth,
                "converged": converged,
                "llf": llf,
                "aic": aic,
                "bic": bic,
                "grad": grad
            })

            # Prefer the first converged solution
            if converged:
                best_res = res
                break

            # If not converged, keep the best by log-likelihood so far
            if best_res is None or (not np.isnan(llf) and llf > float(getattr(best_res, "llf", -np.inf))):
                best_res = res

        except Exception as e:
            attempts.append({
                "tag": tag,
                "optimizer": meth,
                "converged": False,
                "llf": np.nan,
                "aic": np.nan,
                "bic": np.nan,
                "grad": np.nan,
                "error": repr(e)
            })
            continue

    attempts_df = pd.DataFrame(attempts)
    return best_res, attempts_df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    episodes_path = os.path.join(DATA_DIR, "episodes.parquet")
    features_path = os.path.join(DATA_DIR, "features.parquet")

    if not os.path.exists(episodes_path) or not os.path.exists(features_path):
        raise FileNotFoundError(
            f"Missing parquet files. Expected:\n  {episodes_path}\n  {features_path}\n"
            "Generate them first (e.g., run main.py)."
        )

    epi = pd.read_parquet(episodes_path)
    feat = pd.read_parquet(features_path)

    # Merge episode predictors + engineered features
    df = epi.merge(feat, on="episode_id", how="left")

    # Log transforms for heavy-tailed metrics
    if "node_expansions" not in df.columns:
        raise KeyError("Expected column 'node_expansions' not found in merged dataframe.")
    df["log_node_expansions"] = np.log1p(df["node_expansions"])

    if "runtime_ms" in df.columns:
        df["log_runtime_ms"] = np.log1p(df["runtime_ms"])

    # Ensure categorical treatment
    for col in ["heuristic", "communication", "disturbance"]:
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' not found in merged dataframe.")
        df[col] = df[col].astype("category")

    if "map_id" not in df.columns:
        raise KeyError("Expected grouping column 'map_id' not found in episodes dataframe.")

    # -----------------------------
    # LMM 1: Makespan (raw scale)
    # -----------------------------
    formula_m = (
        "makespan ~ agent_count + obstacle_density + "
        "C(communication) + C(disturbance) + C(heuristic) + "
        "obstacle_density:C(heuristic)"
    )

    m1 = smf.mixedlm(formula_m, df, groups=df["map_id"])
    r1, attempts1 = fit_mixedlm_with_fallbacks(m1, reml=False, maxiter=2000, tag="LMM_makespan")

    # Save attempts log (helps you justify convergence decisions)
    attempts_path = os.path.join(OUT_DIR, "lmm_fit_attempts.csv")
    attempts1.to_csv(attempts_path, index=False)

    # Save coefficients table
    coef1_df = summary_table_to_df(r1.summary().tables[1])
    coef1_df.to_csv(os.path.join(OUT_DIR, "lmm_makespan_coefficients.csv"), index=False)

    with open(os.path.join(OUT_DIR, "lmm_makespan_model_summary.txt"), "w") as f:
        f.write(str(r1.summary()))

    resid1 = np.asarray(r1.resid)
    fitted1 = np.asarray(r1.fittedvalues)

    qqplot_resid(
        resid1,
        os.path.join(OUT_DIR, "lmm_makespan_qqplot.png"),
        "LMM Residual Q–Q Plot (Makespan)"
    )
    resid_vs_fitted(
        fitted1,
        resid1,
        os.path.join(OUT_DIR, "lmm_makespan_resid_vs_fitted.png"),
        "Residuals vs Fitted (LMM Makespan)"
    )

    # ---------------------------------------
    # LMM 2: log(node_expansions) (stabilized)
    # ---------------------------------------
    formula_e = (
        "log_node_expansions ~ agent_count + obstacle_density + "
        "C(communication) + C(disturbance) + C(heuristic) + "
        "obstacle_density:C(heuristic)"
    )

    m2 = smf.mixedlm(formula_e, df, groups=df["map_id"])
    r2, attempts2 = fit_mixedlm_with_fallbacks(m2, reml=False, maxiter=2000, tag="LMM_log_node_expansions")

    # Append attempts
    attempts_all = pd.concat([attempts1, attempts2], ignore_index=True)
    attempts_all.to_csv(attempts_path, index=False)

    coef2_df = summary_table_to_df(r2.summary().tables[1])
    coef2_df.to_csv(os.path.join(OUT_DIR, "lmm_log_node_expansions_coefficients.csv"), index=False)

    with open(os.path.join(OUT_DIR, "lmm_log_node_expansions_model_summary.txt"), "w") as f:
        f.write(str(r2.summary()))

    resid2 = np.asarray(r2.resid)
    fitted2 = np.asarray(r2.fittedvalues)

    qqplot_resid(
        resid2,
        os.path.join(OUT_DIR, "lmm_log_node_expansions_qqplot.png"),
        "LMM Residual Q–Q Plot (log Node Expansions)"
    )
    resid_vs_fitted(
        fitted2,
        resid2,
        os.path.join(OUT_DIR, "lmm_log_node_expansions_resid_vs_fitted.png"),
        "Residuals vs Fitted (LMM log Node Expansions)"
    )

    # -----------------------------
    # Fit statistics summary table
    # -----------------------------
    fit_stats = pd.DataFrame(
        [
            {
                "model": "LMM makespan",
                "converged": bool(getattr(r1, "converged", True)),
                "AIC": float(getattr(r1, "aic", np.nan)),
                "BIC": float(getattr(r1, "bic", np.nan)),
                "logLik": float(getattr(r1, "llf", np.nan))
            },
            {
                "model": "LMM log(node_expansions)",
                "converged": bool(getattr(r2, "converged", True)),
                "AIC": float(getattr(r2, "aic", np.nan)),
                "BIC": float(getattr(r2, "bic", np.nan)),
                "logLik": float(getattr(r2, "llf", np.nan))
            },
        ]
    )
    fit_stats.to_csv(os.path.join(OUT_DIR, "lmm_fit_statistics.csv"), index=False)

    print("[OK] LMM outputs saved to:", OUT_DIR)
    print("  - lmm_makespan_coefficients.csv")
    print("  - lmm_log_node_expansions_coefficients.csv")
    print("  - lmm_fit_statistics.csv")
    print("  - lmm_fit_attempts.csv")
    print("  - lmm_makespan_qqplot.png")
    print("  - lmm_makespan_resid_vs_fitted.png")
    print("  - lmm_log_node_expansions_qqplot.png")
    print("  - lmm_log_node_expansions_resid_vs_fitted.png")
    print("  - lmm_makespan_model_summary.txt")
    print("  - lmm_log_node_expansions_model_summary.txt")


if __name__ == "__main__":
    main()
