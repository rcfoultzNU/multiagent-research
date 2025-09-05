
import json, os, argparse, random, hashlib
from datetime import datetime
import pandas as pd

from mapgen import generate_map, free_ratio, clustering_index
from simulator import simulate_episode

def factorial_levels(factors):
    for N in factors["agent_count"]:
        for rho in factors["obstacle_density"]:
            for comm in factors["communication"]:
                for disturb in factors["disturbance"]:
                    for heuristic in factors["heuristic"]:
                        yield (N, rho, comm, disturb, heuristic)

def episode_id_of(seed, N, rho, comm, disturb, heuristic):
    key = f"{seed}-{N}-{rho}-{comm}-{disturb}-{heuristic}".encode()
    return hashlib.sha1(key).hexdigest()[:12]

def main():
    import pyarrow as pa
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--replications", type=int, default=3)
    ap.add_argument("--out", default="data/logs")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--chunksize", type=int, default=10000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(args.manifest) as f:
        factors = json.load(f)

    rnd = random.Random(args.seed)
    map_w = factors.get("map_width", 32)
    map_h = factors.get("map_height", 32)

    manifest_path = os.path.join(os.path.dirname(args.manifest), "run_manifest.jsonl")
    with open(manifest_path, "w") as mf:
        for (N, rho, comm, disturb, heuristic) in factorial_levels(factors):
            mf.write(json.dumps({
                "agent_count": N, "obstacle_density": rho,
                "communication": comm, "disturbance": disturb,
                "heuristic": heuristic
            }) + "\n")

    episodes_path = os.path.join(args.out, "episodes.parquet")
    features_path = os.path.join(args.out, "features.parquet")

    epi_rows, feat_rows = [], []
    run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    total = 0
    writer_epi = None
    writer_feat = None

    try:
        for (N, rho, comm, disturb, heuristic) in factorial_levels(factors):
            map_seed = rnd.randrange(1,1_000_000_000)
            grid = generate_map(map_w, map_h, rho, map_seed)
            map_id = hashlib.md5(f"{map_seed}-{rho}-{map_w}x{map_h}".encode()).hexdigest()[:10]
            fr = free_ratio(grid); ci = clustering_index(grid)

            for r in range(args.replications):
                seed = rnd.randrange(1,1_000_000_000)
                episode_id = episode_id_of(seed,N,rho,comm,disturb,heuristic)
                metrics, features = simulate_episode(grid,N,rho,comm,disturb,heuristic,seed)

                epi_rows.append({
                    "run_id": run_id, "episode_id": episode_id, "map_id": map_id, "seed": seed,
                    "agent_count": N, "obstacle_density": rho, "communication": comm,
                    "disturbance": disturb, "heuristic": heuristic, **metrics
                })
                feat_rows.append({
                    "episode_id": episode_id, "map_width": map_w, "map_height": map_h,
                    "free_ratio": round(fr,4), "clustering_index": round(ci,3),
                    "start_goal_dispersion": int(features["start_goal_dispersion"]),
                    "open_list_pressure_est": int(features["open_list_pressure_est"])
                })
                total += 1

                if len(epi_rows) >= args.chunksize:
                    df_e = pd.DataFrame(epi_rows); df_f = pd.DataFrame(feat_rows)
                    table_e = pa.Table.from_pandas(df_e, preserve_index=False)
                    table_f = pa.Table.from_pandas(df_f, preserve_index=False)
                    if writer_epi is None:
                        writer_epi = pq.ParquetWriter(episodes_path, table_e.schema)
                        writer_feat = pq.ParquetWriter(features_path, table_f.schema)
                    writer_epi.write_table(table_e)
                    writer_feat.write_table(table_f)
                    epi_rows.clear(); feat_rows.clear()

        if epi_rows:
            df_e = pd.DataFrame(epi_rows); df_f = pd.DataFrame(feat_rows)
            table_e = pa.Table.from_pandas(df_e, preserve_index=False)
            table_f = pa.Table.from_pandas(df_f, preserve_index=False)
            if writer_epi is None:
                writer_epi = pq.ParquetWriter(episodes_path, table_e.schema)
                writer_feat = pq.ParquetWriter(features_path, table_f.schema)
            writer_epi.write_table(table_e)
            writer_feat.write_table(table_f)

    finally:
        if writer_epi: writer_epi.close()
        if writer_feat: writer_feat.close()

    print(f"[OK] Wrote {total} episodes to {episodes_path} and features to {features_path}")

if __name__ == "__main__":
    main()
