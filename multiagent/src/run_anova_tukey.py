import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

DATA_DIR = "data/logs"
OUT_DIR = "model_outputs"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    epi = pd.read_parquet(os.path.join(DATA_DIR, "episodes.parquet"))
    feat = pd.read_parquet(os.path.join(DATA_DIR, "features.parquet"))
    df = epi.merge(feat, on="episode_id", how="left")

    # Log transforms for heavy tails (optional but common)
    df["log_node_expansions"] = np.log1p(df["node_expansions"])

    # Categoricals for factorial ANOVA
    for col in ["heuristic", "communication", "disturbance", "agent_count", "obstacle_density"]:
        df[col] = df[col].astype("category")

    # ---------------- ANOVA: Makespan ----------------
    # Full factorial-ish, but keep it reasonable for N=96:
    # heuristic * obstacle_density + communication + disturbance + agent_count
    formula = "makespan ~ C(agent_count) + C(obstacle_density) * C(heuristic) + C(communication) + C(disturbance)"

    ols_model = smf.ols(formula, data=df).fit()
    anova_tbl = anova_lm(ols_model, typ=2)  # Type II ANOVA (good default)
    anova_tbl.to_csv(os.path.join(OUT_DIR, "anova_makespan_type2.csv"))

    # ---------------- Tukey HSD ----------------
    # Tukey requires a single grouping factor.
    # A common choice: compare heuristics (collapsed across other factors),
    # or compare heuristic within a specific obstacle_density level.
    # Here: heuristic only (global comparison)
    tukey = pairwise_tukeyhsd(endog=df["makespan"], groups=df["heuristic"], alpha=0.05)

    tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
    tukey_df.to_csv(os.path.join(OUT_DIR, "tukey_makespan_by_heuristic.csv"), index=False)

    # Optional: save Tukey plot
    fig = tukey.plot_simultaneous()
    plt.title("Tukey HSD: Makespan by Heuristic")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "tukey_makespan_by_heuristic.png"), dpi=300)
    plt.close()

    # ---------------- ANOVA: log(node_expansions) ----------------
    formula2 = "log_node_expansions ~ C(agent_count) + C(obstacle_density) * C(heuristic) + C(communication) + C(disturbance)"
    ols_model2 = smf.ols(formula2, data=df).fit()
    anova_tbl2 = anova_lm(ols_model2, typ=2)
    anova_tbl2.to_csv(os.path.join(OUT_DIR, "anova_log_node_expansions_type2.csv"))

    print("[OK] ANOVA + Tukey outputs saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
