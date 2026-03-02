#!/usr/bin/env python3
import os
import re
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from kneed import KneeLocator

PERF_DIR = "/home/zixi/orion_bu/artifact_evaluation/FICKS/new_A4000_profiles"

def find_knee_tpc(x, y_raw):
    # 1) lightly smooth the curve
    x_arr = np.array(x, dtype=int)
    mask  = x_arr >= 6
    x     = x_arr[mask]
    y_raw = np.array(y_raw, dtype=float)[mask]

    if len(y_raw) < 3:
        # too short, just pick max TPC
        return int(x[-1])

    # make window length odd and <= len(y_raw)
    wl = min(7, len(y_raw) if len(y_raw) % 2 == 1 else len(y_raw) - 1)
    if wl < 3:  # still too small, fallback
        return int(x[-1])

    y_smooth = savgol_filter(y_raw, window_length=wl, polyorder=2)

    # 2) estimate noise & set S = (peak-to-peak) / (std dev)
    noise  = np.std(y_smooth - y_raw)
    signal = max(y_smooth) - min(y_smooth)
    S      = max(1.0, signal / (noise + 1e-6))

    # 3) try Kneedle first
    kl = KneeLocator(
        x, y_smooth,
        curve="convex", direction="decreasing",
        interp_method="polynomial",
        polynomial_degree=2,
        S=S
    )
    if kl.knee is not None:
        return int(kl.knee)

    # 4) fallback: first point where marginal speedup < 5%
    diffs = y_smooth[:-1] - y_smooth[1:]
    rel   = diffs / y_smooth[:-1]
    small = np.where(rel < 0.05)[0]
    if len(small) > 0:
        return int(x[small[0] + 1])

    # 5) last resort: max-distance heuristic
    xn = (x - x.min()) / (x.max() - x.min() + 1e-12)
    yn = (y_smooth - y_smooth.min()) / (y_smooth.max() - y_smooth.min() + 1e-12)
    dist = np.abs(xn + yn - 1)
    return int(x[np.argmax(dist)])

def process_fwd_file(fwd_filename):
    """
    Input file examples:
      - bert_8_fwd_KRISP
      - mobilenetv2_32_fwd_KRISP

    We:
      - parse model, batch
      - read perf from PERF_DIR/<model>_bz<batch>.csv
      - compute Knee_TPC per row
      - read original fwd file, keep only first 7 columns
      - append new Knee_TPC column
      - write to new file WITHOUT '_KRISP', e.g. mobilenetv2_32_fwd
    """
    # match: <model>_<batch>_fwd or <model>_<batch>_fwd_KRISP
    m = re.match(r"(.+?)_(\d+)_fwd(?:_KRISP)?$", fwd_filename)
    if not m:
        return
    model, batch = m.groups()

    perf_path = os.path.join(PERF_DIR, f"{model}_bz{batch}.csv")
    if not os.path.isfile(perf_path):
        print("Perf missing:", perf_path)
        return

    # ---- load perf CSV and compute knee ----
    perf_df = pd.read_csv(perf_path)
    perf_df.columns = perf_df.columns.str.strip()

    # TPC columns are numeric column names like "1","2","4",...
    tpc_cols = sorted([c for c in perf_df.columns if str(c).isdigit()],
                      key=lambda c: int(c))
    if not tpc_cols:
        print("No TPC columns in perf file:", perf_path)
        return

    x = np.array([int(c) for c in tpc_cols])

    perf_df["Knee_TPC"] = perf_df[tpc_cols].apply(
        lambda row: find_knee_tpc(x, row.values.astype(float)),
        axis=1
    )

    # ---- load fwd CSV, drop old Knee_TPC/is_critical, keep first 7 cols ----
    fwd_df = pd.read_csv(fwd_filename)
    if len(fwd_df) != len(perf_df):
        print("Length mismatch:", fwd_filename, "vs", perf_path)
        return

    # keep only first 7 columns from original file
    base_cols = list(fwd_df.columns[:7])
    new_df = fwd_df[base_cols].copy()

    # add new Knee_TPC column from perf_df
    new_df["Knee_TPC"] = perf_df["Knee_TPC"].values

    # ---- output filename WITHOUT _KRISP ----
    # e.g. mobilenetv2_32_fwd_KRISP -> mobilenetv2_32_fwd
    out_filename = re.sub(r"_KRISP$", "", fwd_filename)
    new_df.to_csv(out_filename, index=False)
    print("Updated", out_filename)

if __name__ == "__main__":
    for fn in os.listdir("."):
        if not os.path.isfile(fn):
            continue
        # process only *_fwd_KRISP (or *_fwd if you ever have them)
        if re.match(r".+_\d+_fwd(?:_KRISP)?$", fn):
            process_fwd_file(fn)
