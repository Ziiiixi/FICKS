#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ONE-FILE PIPELINE (CSV-based, no Excel):

Step 1 → Build A4000_ratio_by_plan.csv from a unified 'profile plan WITH clusters' CSV.
          - Input: A4000_profile_plan_all_cluster_pairs_up_to3.csv
          - For each plan row and each config column '(n,m)',
            compute ratio = (L_coloc(UT,n) - L_excl(UT,n)) / L_excl(UT,n),
            where L_excl(UT,n) comes from the exclusive CSV of the *UT model*.

IMPORTANT UNIT NOTE:
  - PLAN CSV config columns '(n,m)' are in nanoseconds (ns)
  - Exclusive profile CSVs are in microseconds (µs)
  => We convert coloc(ns) -> coloc(µs) by dividing by 1000 before ratio.

Step 2 → Compute geomean ratios per (Model_UT, cluster_UT, cluster_CO) from A4000_ratio_by_plan.csv.

Outputs:
  - A4000_ratio_by_plan.csv
  - A4000_geomean_by_cluster_pair.csv
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

# ──────────────────────────────── CONFIG ────────────────────────────────
PLAN_CSV = Path("A4000_profile_plan_all_cluster_pairs_up_to3.csv")

# Exclusive profile directory (per-model CSVs)
EXCL_DIRS = [Path("/home/zixi/orion_bu/artifact_evaluation/FICKS/A4000_TPC_profiles_cudnn_based")]

OUT_RATIO_PLAN = Path("A4000_ratio_by_plan.csv")
OUT_GEOMEAN    = Path("A4000_geomean_by_cluster_pair.csv")

CONFIG_RE  = re.compile(r"^\(\s*(\d+)\s*,\s*(\d+)\s*\)$")   # "(n,m)"
TPC_COL_RE = re.compile(r"^\s*(\d{1,3})\s*$")               # "1","2",...

# ---- UNIT CONVERSION ----
# Plan config columns: ns
# Exclusive profiles:  µs
NS_TO_US = 1.0 / 1000.0

# ──────────────────────────────── HELPERS ───────────────────────────────
def _to_int64(s):
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def best_id_col(df: pd.DataFrame):
    """
    Try to find a kernel ID column.
    Recognizes 'Kernel Index', 'id', 'kernel_id', etc.
    """
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")

    norm_to_orig = {norm(c): c for c in df.columns}

    kernel_id_candidates = [
        "kernel_index",
        "kernel_id",
        "kernelid",
        "kernel",
    ]
    generic_id_candidates = ["id", "idx", "index"]

    for cand in kernel_id_candidates + generic_id_candidates:
        if cand in norm_to_orig:
            return norm_to_orig[cand]

    # Fallback: unnamed integer column (from Excel/CSV)
    for n, o in norm_to_orig.items():
        if n.startswith("unnamed") and df[o].dtype.kind in "iu":
            return o

    return None


def find_config_columns(df):
    """
    Return (config_cols, ut_n_list) for columns like '(n,m)'.
    Order doesn't matter for correctness; we keep them in the original
    column order for readability.
    """
    cols, n_ut = [], []
    for c in df.columns:
        if not isinstance(c, str):
            continue
        m = CONFIG_RE.match(c)
        if m:
            cols.append(c)
            n_ut.append(int(m.group(1)))
    return cols, n_ut

# ─────────────────────────────── STEP 1 ───────────────────────────────
def load_exclusive_csvs():
    """
    Load all exclusive CSVs and build:
        raw[stem] = DataFrame
    where stem is the filename without extension, e.g. 'resnet152_bz8'.
    """
    found = {}
    print("[INFO] Searching for exclusive CSVs in:")
    for root in EXCL_DIRS:
        print(f"   - {root.resolve()}")
        if not root.exists():
            print("      (directory does not exist)")
            continue
        for p in root.glob("*.csv"):
            model = p.stem   # e.g., 'resnet152_bz8'
            try:
                found[model] = pd.read_csv(p)
                print(f"      ✓ Loaded {p.name}")
            except Exception as e:
                print(f"      ✗ Failed to read {p.name}: {e}")
    print(f"[INFO] Total exclusive CSVs loaded: {len(found)}\n")
    return found


def exclusive_lookup(excl_df):
    """
    Build a LUT:
        lut[kernel_id][TPC_n] = latency_us   (µs)

    Works with ID columns named 'Kernel Index', 'id', 'kernel_id', etc.

    NOTE: exclusive values are assumed already in microseconds (µs).
    """
    if excl_df is None or excl_df.empty:
        return {}

    id_col = best_id_col(excl_df)
    if id_col is None:
        print("[WARN] Exclusive CSV has no ID-like column (expected 'Kernel Index', 'id', etc.).")
        return {}

    # Find numeric TPC columns like "1","2",..., "24"
    tpc_cols = []
    for c in excl_df.columns:
        if not isinstance(c, str):
            continue
        m = TPC_COL_RE.match(c)
        if m:
            tpc_cols.append((int(m.group(1)), c))

    if not tpc_cols:
        print("[WARN] Exclusive CSV has no TPC columns matching '1','2',...,'24'.")
        return {}

    df = excl_df.copy()
    df[id_col] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    df = df.dropna(subset=[id_col])

    names = [c for _, c in tpc_cols]
    df[names] = df[names].apply(pd.to_numeric, errors="coerce")

    lut = {}
    for _, row in df.iterrows():
        kid = row[id_col]
        if pd.isna(kid):
            continue
        kid = int(kid)
        lut[kid] = {}
        for n, col in tpc_cols:
            val = row[col]
            if pd.notna(val):
                # Exclusive values are in µs
                lut[kid][n] = float(val)
    return lut


def choose_exclusive_lut_for_model(model_name: str, luts_by_file: dict):
    """
    Choose which exclusive LUT to use for a given UT model name.

    Model_UT might be 'resnet152', 'resnet152_bz8', etc.
    We try:
      - exact stem match
      - case-insensitive match
      - prefix match (stem startswith(model_name))
      - prefix match (model_name startswith(stem)) as last resort.
    """
    if not luts_by_file:
        return {}

    if model_name in luts_by_file:
        return luts_by_file[model_name]

    low_map = {k.lower(): v for k, v in luts_by_file.items()}
    if model_name.lower() in low_map:
        return low_map[model_name.lower()]

    for stem, lut in luts_by_file.items():
        if stem.lower().startswith(model_name.lower()):
            return lut

    for stem, lut in luts_by_file.items():
        if model_name.lower().startswith(stem.lower()):
            return lut

    return {}


def compute_ratios_for_all_rows(plan_df, model_to_lut):
    """
    For the whole plan_df (no reordering):
      For each row:
        - model = row['Model_UT']
        - lut   = model_to_lut[model]
        - kid   = row[Kernel_ID_UT]
        - for each '(n,m)' column (coloc latency is in ns):
              coloc_us = coloc_ns / 1000
              ratio = (coloc_us - excl_us) / excl_us

    Returns a new DataFrame with the same row order and same columns,
    but '(n,m)' columns replaced by ratios.
    """
    out = plan_df.copy()

    cfg_cols, n_ut_list = find_config_columns(out)
    if not cfg_cols:
        print("[WARN] No '(n,m)' config columns found in plan CSV.")
        return out

    # UT kernel id column
    ut_col = None
    for cand in ["Kernel_ID_UT", "Kernel Id_UT", "Kernel Index_UT"]:
        if cand in out.columns:
            ut_col = cand
            break
    if ut_col is None:
        print("[WARN] No UT kernel ID column found; all config ratios will be NaN.")
        for c in cfg_cols:
            out[c] = np.nan
        return out

    # Plan values: config columns are ns
    out[cfg_cols] = out[cfg_cols].apply(pd.to_numeric, errors="coerce")
    ut_ids = _to_int64(out[ut_col]).astype(float)
    ut_models = out["Model_UT"].astype(str)

    for col, n_ut in zip(cfg_cols, n_ut_list):
        vals = []
        for i in range(len(out)):
            kid_float = ut_ids.iat[i]
            model = ut_models.iat[i]

            if np.isnan(kid_float):
                vals.append(np.nan)
                continue
            kid = int(kid_float)

            lut = model_to_lut.get(model, {})
            excl_us = lut.get(kid, {}).get(n_ut, np.nan)  # µs
            coloc_ns = out.iat[i, out.columns.get_loc(col)]  # ns

            if pd.isna(coloc_ns) or pd.isna(excl_us) or excl_us <= 0:
                vals.append(np.nan)
                continue

            coloc_us = float(coloc_ns) * NS_TO_US  # ns -> µs
            vals.append((coloc_us - excl_us) / excl_us)

        out[col] = vals

    return out


def step1_build_ratio_csv():
    print("\n=== STEP 1: Building A4000_ratio_by_plan.csv (from CSV plan) ===")

    if not PLAN_CSV.exists():
        raise FileNotFoundError(f"Plan CSV not found: {PLAN_CSV.resolve()}")

    plan_df = pd.read_csv(PLAN_CSV)
    if "Model_UT" not in plan_df.columns:
        raise KeyError("Plan CSV must contain a 'Model_UT' column.")

    excl_raw_by_file = load_exclusive_csvs()
    excl_luts_by_file = {stem: exclusive_lookup(df) for stem, df in excl_raw_by_file.items()}

    models = sorted(plan_df["Model_UT"].dropna().astype(str).unique().tolist())
    model_to_lut = {}
    for model in models:
        lut = choose_exclusive_lut_for_model(model, excl_luts_by_file)
        model_to_lut[model] = lut
        if not lut:
            print(f"[WARN] Model_UT '{model}': no matching exclusive file found; ratios will be NaN for its rows.")

    ratio_df = compute_ratios_for_all_rows(plan_df, model_to_lut)
    ratio_df.to_csv(OUT_RATIO_PLAN, index=False)
    print(f"[OK] Wrote ratio CSV: {OUT_RATIO_PLAN.resolve()}\n")

# ─────────────────────────────── STEP 2 ───────────────────────────────
def geomean_ratio_signed(series):
    """
    series contains ratios r, we compute geomean over (1+r) and return (gm - 1).
    """
    x = pd.to_numeric(series, errors="coerce").to_numpy(float)
    x = x[np.isfinite(x) & (x > -1 + 1e-12)]
    if x.size == 0:
        return np.nan
    gm = np.exp(np.mean(np.log(1 + x)))
    return gm - 1.0


def step2_build_geomean_by_cluster_pair():
    print("\n=== STEP 2: Building A4000_geomean_by_cluster_pair.csv ===")
    if not OUT_RATIO_PLAN.exists():
        raise FileNotFoundError("Missing ratio CSV from step 1.")

    df = pd.read_csv(OUT_RATIO_PLAN)

    required = ["Model_UT", "cluster_UT", "cluster_CO"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Ratio CSV must contain column '{c}'.")

    cfg_cols, _ = find_config_columns(df)
    if not cfg_cols:
        raise RuntimeError("No '(n,m)' config columns found in ratio CSV.")

    df_cfg = df.dropna(subset=["Model_UT", "cluster_UT", "cluster_CO"]).copy()
    df_cfg[cfg_cols] = df_cfg[cfg_cols].apply(pd.to_numeric, errors="coerce")

    grp = df_cfg.groupby(["Model_UT", "cluster_UT", "cluster_CO"])
    agg = grp[cfg_cols].agg(geomean_ratio_signed).reset_index()

    agg.to_csv(OUT_GEOMEAN, index=False)
    print(f"[OK] Wrote geomean CSV: {OUT_GEOMEAN.resolve()}\n")

# ─────────────────────────────── MAIN ───────────────────────────────
if __name__ == "__main__":
    step1_build_ratio_csv()
    step2_build_geomean_by_cluster_pair()
    print("Pipeline complete ✅")
