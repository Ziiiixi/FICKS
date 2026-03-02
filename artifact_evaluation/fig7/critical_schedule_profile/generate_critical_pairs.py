#!/usr/bin/env python3
import pandas as pd
import random
import sys

# ─── CONFIGURE ────────────────────────────────────────────────────────────────
model1       = "resnet152"
model2       = "resnet152"
CSV1         = f"/home/zixi/orion_bu/benchmarking/model_kernels/ours/{model1}_8_fwd"
CSV2         = f"/home/zixi/orion_bu/benchmarking/model_kernels/ours/{model2}_8_fwd"
OUTPUT_CSV   = "critical_pairs.csv"
NUM_PAIRS    = 1  # max number of pairs (from the pool of second-IDs)
SEED         = None   # or None for no fixed seed

FIXED_ID1    = 445   # ← the kernel ID you want to lock in as the first element
# ───────────────────────────────────────────────────────────────────────────────

def load_and_tag_ids(path):
    df = pd.read_csv(path)
    df['id'] = range(len(df))
    return df

def main():
    # 1) Read & tag each CSV
    df1 = load_and_tag_ids(CSV1)
    df2 = load_and_tag_ids(CSV2)

    # 2) Filter only critical kernels in each
    crit1 = df1[df1['kernel type'] == 'k1']
    crit2 = df2

    if crit1.empty or crit2.empty:
        sys.exit("ERROR: Need at least one critical kernel in each model to form pairs.")

    # 3) Make sure FIXED_ID1 is actually in crit1
    # if FIXED_ID1 not in crit1['id'].values:
    #     sys.exit(f"ERROR: Fixed ID {FIXED_ID1} is not among critical kernels in model1.")

    # 4) Build all pairs with that fixed first ID
    all_pairs = [(FIXED_ID1, id2) for id2 in crit2['id']]
    # 5) Sample down if too many
    if SEED is not None:
        random.seed(SEED)
    if len(all_pairs) > NUM_PAIRS:
        pairs = random.sample(all_pairs, NUM_PAIRS)
    # else:
    #     pairs = all_pairs

    # 6) Write out
    out_df = pd.DataFrame(pairs, columns=['id1', 'id2'])
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(pairs)} pairs (fixed {FIXED_ID1}→model2) to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
