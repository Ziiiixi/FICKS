#!/usr/bin/env python3
import pandas as pd
import sys

# ─── CONFIG ───────────────────────────────────────────────────────────────────
model1       = "resnet152"
model2       = "resnet152"
PAIRS_CSV    = "critical_pairs.csv"
TPC1_CSV     = f"/home/zixi/orion_bu/artifact_evaluation/fig7/new_A4000_profile/{model1}_bz8.csv"
TPC2_CSV     = f"/home/zixi/orion_bu/artifact_evaluation/fig7/new_A4000_profile/{model2}_bz8.csv"
OUTPUT_CSV   = "all_partitions.csv"
TOTAL_TPCS   = 24
# ───────────────────────────────────────────────────────────────────────────────

def load_and_tag(path):
    """
    Read a TPC‐profile CSV, tag each row with an 'id' (0..N-1), and return the DataFrame.
    """
    df = pd.read_csv(path)
    df['id'] = range(len(df))
    return df

def main():
    # 1) load data
    try:
        tpc1 = load_and_tag(TPC1_CSV)
        tpc2 = load_and_tag(TPC2_CSV)
        pairs = pd.read_csv(PAIRS_CSV)
    except Exception as e:
        sys.exit(f"ERROR loading files: {e}")

    # 2) for each pair, try every partition n1 / (TOTAL_TPCS - n1)
    results = []
    for _, p in pairs.iterrows():
        id1, id2 = int(p['id1']), int(p['id2'])
        r1 = tpc1.loc[tpc1['id'] == id1]
        r2 = tpc2.loc[tpc2['id'] == id2]
        if r1.empty or r2.empty:
            print(f"WARNING: id {id1} or {id2} not found; skipping")
            continue

        row1 = r1.iloc[0]
        row2 = r2.iloc[0]

        for n1 in range(1, TOTAL_TPCS):
            n2 = TOTAL_TPCS - n1
            perf1 = row1[str(n1)]
            perf2 = row2[str(n2)]
            duration_nano = max(perf1, perf2)

            results.append({
                'id1':           id1,
                'id2':           id2,
                'tpc1':          n1,
                'tpc2':          n2,
                'duration_nano': duration_nano
            })

    # 3) write out sorted by “config” (tpc1 ascending)
    df_out = pd.DataFrame(results)
    df_out.sort_values(by='tpc1', inplace=True)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df_out)} rows (all 1–{TOTAL_TPCS-1} partitions per pair) to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
