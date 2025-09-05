
import argparse, os, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/logs")
    args = ap.parse_args()

    epi = pd.read_parquet(os.path.join(args.path, "episodes.parquet"))
    feat = pd.read_parquet(os.path.join(args.path, "features.parquet"))

    print("Episodes:", epi.shape, "Features:", feat.shape)
    print("\nBy heuristic, success rate and mean makespan:")
    print(epi.groupby("heuristic").agg(success_rate=("success","mean"),
                                       makespan_mean=("makespan","mean")).round(3))

    print("\nHeuristic × density success rate:")
    print(epi.groupby(["heuristic","obstacle_density"]).agg(success_rate=("success","mean")).round(3).unstack())

if __name__ == "__main__":
    main()
