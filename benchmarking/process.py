from pathlib import Path
import pandas as pd

# ========= EDIT THESE =========
PAIR_DIR = Path("/home/zixi/orion_bu/artifact_evaluation/FICKS/critical_schedule_profile/pair_durations")
OUT_CSV  = Path("A4000_pair_profiles_flat.csv")
# ==============================

def build_pair_profiles():
    pair_files = sorted(PAIR_DIR.glob("pair_durations_*.csv"))
    if not pair_files:
        raise RuntimeError(f"No pair_durations_*.csv found in {PAIR_DIR}")

    print("Found pair_durations files:")
    for f in pair_files:
        print("  -", f.name)

    records = []
    per_file_counts = {}

    for path in pair_files:
        df = pd.read_csv(path)

        required = {
            "model0", "model1",
            "kernel0_id", "kernel1_id",
            "tpc0", "tpc1",
            # "client0_wall_ns",
            "makespan_us",
        }
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(f"{path} missing columns: {missing}")

        # group by kernel pair (UT/CO)
        grouped = df.groupby(["model0", "model1", "kernel0_id", "kernel1_id"])
        n_groups = len(grouped)
        per_file_counts[path.name] = n_groups
        print(f"{path.name}: {n_groups} aggregated kernel-pair rows")

        for (m0, m1, k0, k1), sub in grouped:
            rec = {
                "model_ut": m0,
                "kernel_ut_id": k0,
                "model_co": m1,
                "kernel_co_id": k1,
            }

            # aggregate UT performance per TPC pair (median over iterations)
            # tpc_group = sub.groupby(["tpc0", "tpc1"])["client0_wall_ns"].median()
            tpc_group = sub.groupby(["tpc0", "tpc1"])["makespan_us"].median()

            for (t0, t1), val in tpc_group.items():
                col_name = f"({int(t0)},{int(t1)})"
                rec[col_name] = val

            records.append(rec)

    df_out = pd.DataFrame(records)

    # === NEW: collapse duplicate (model_ut, kernel_ut_id, model_co, kernel_co_id) by average ===
    meta_cols = ["model_ut", "kernel_ut_id", "model_co", "kernel_co_id"]
    tpc_cols = [c for c in df_out.columns if c not in meta_cols]

    before_rows = len(df_out)
    # group and average numeric columns (TPC columns); non-numeric ignored
    df_out = df_out.groupby(meta_cols, as_index=False).mean(numeric_only=True)
    after_rows = len(df_out)
    print(f"\nCollapsed duplicates: {before_rows} -> {after_rows} unique kernel pairs")

    # re-detect TPC cols after groupby
    tpc_cols = [c for c in df_out.columns if c not in meta_cols]

    # order TPC columns: sorted by (tpc0, tpc1)
    def _tpc_key(name):
        # "(1,63)" -> (1, 63)
        inside = name.strip("()")
        a, b = inside.split(",")
        return int(a), int(b)

    tpc_cols_sorted = sorted(tpc_cols, key=_tpc_key)
    df_out = df_out[meta_cols + tpc_cols_sorted]

    print("\nSummary (aggregated kernel-pair rows per file):")
    for fname, cnt in per_file_counts.items():
        print(f"  {fname}: {cnt}")

    print(f"\nTotal rows in {OUT_CSV.name}: {len(df_out)}")
    df_out.to_csv(OUT_CSV, index=False)


if __name__ == "__main__":
    build_pair_profiles()
