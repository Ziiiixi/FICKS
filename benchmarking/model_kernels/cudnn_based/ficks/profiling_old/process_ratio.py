#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Latency estimator tables from cluster-pair geomean ratios (CSV version).

Input:
  - A4000_geomean_by_cluster_pair.csv
    Columns:
      Model_UT, cluster_UT, cluster_CO, and per-config columns "(n,m)" with geomean ratios.

We IGNORE Model_UT here and only care about cluster pairs.

Outputs (global, across all models):
  - A4000_ratios_pair.csv
      Columns: cluster_UT, cluster_CO, n, m, ratio_gm, mult
  - A4000_ratios_ut.csv
      Columns: cluster_UT, n, m, ratio_ut_gm, mult_ut

Usage in scheduler:
  - Known CO cluster:
        L_coloc_hat = L_excl(n) * mult              (from ratios_pair)
  - Unknown CO cluster:
        L_coloc_hat = L_excl(n) * mult_ut           (from ratios_ut)
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

# --------- Paths ---------
SRC_GEOMEAN = Path("A4000_geomean_by_cluster_pair.csv")

OUT_PAIR = Path("A4000_ratios_pair.csv")
OUT_UT   = Path("A4000_ratios_ut.csv")

# --------- Regex ----------
CONFIG_RE = re.compile(r"^\(\s*(\d+)\s*,\s*(\d+)\s*\)$")  # "(n,m)"

# --------- Helpers ---------
def find_cfg_triples(df: pd.DataFrame):
    """Return [(col, n, m), ...] for all '(n,m)' columns (order not important)."""
    triples = []
    for c in df.columns:
        if isinstance(c, str):
            m = CONFIG_RE.match(c)
            if m:
                triples.append((c, int(m.group(1)), int(m.group(2))))
    return triples

def geomean_of_mult(series_1plus_ratio: pd.Series) -> float:
    """Geometric mean of multipliers (>0). NaN-safe; epsilon-clip non-positives."""
    x = pd.to_numeric(series_1plus_ratio, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    eps = 1e-12
    bad = x <= 0.0
    if bad.any():
        x = np.where(bad, eps, x)
    return float(np.exp(np.mean(np.log(x))))

# --------- Main build ---------
def main():
    if not SRC_GEOMEAN.exists():
        raise FileNotFoundError(f"Missing geomean CSV: {SRC_GEOMEAN}")

    print(f"[INFO] Using geomean ratio CSV: {SRC_GEOMEAN}")
    df = pd.read_csv(SRC_GEOMEAN)

    required_cols = ["cluster_UT", "cluster_CO"]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"Input geomean CSV must contain column '{c}'.")

    triples = find_cfg_triples(df)
    if not triples:
        raise RuntimeError("No '(n,m)' config columns found in geomean CSV.")

    cfg_cols = [c for c, _, _ in triples]
    df[cfg_cols] = df[cfg_cols].apply(pd.to_numeric, errors="coerce")

    # --- Build long-format rows across ALL models (we ignore Model_UT) ---
    long_rows = []
    for _, row in df.iterrows():
        c_ut = row["cluster_UT"]
        c_co = row["cluster_CO"]
        try:
            c_ut = int(c_ut)
        except Exception:
            pass
        try:
            c_co = int(c_co)
        except Exception:
            pass

        for col, n, m in triples:
            r = pd.to_numeric(row[col], errors="coerce")
            if pd.isna(r):
                continue
            r = float(r)
            mult = 1.0 + r
            long_rows.append({
                "cluster_UT": c_ut,
                "cluster_CO": c_co,
                "n": int(n),
                "m": int(m),
                "mult": mult,
            })

    if not long_rows:
        raise RuntimeError("No valid (cluster, n, m) rows produced from geomean CSV.")

    long_df = pd.DataFrame(long_rows)

    # --- Pair-specific ratios (known co-runner) ---
    # geomean over ALL models for each (cluster_UT, cluster_CO, n, m)
    grp_pair = long_df.groupby(["cluster_UT", "cluster_CO", "n", "m"], as_index=False)
    df_pairs = grp_pair["mult"].agg(geomean_of_mult)
    df_pairs = df_pairs.rename(columns={"mult": "mult"})
    df_pairs["ratio_gm"] = df_pairs["mult"] - 1.0
    df_pairs = df_pairs[["cluster_UT", "cluster_CO", "n", "m", "ratio_gm", "mult"]]

    # sort by cluster_UT then cluster_CO (and then n,m for stability)
    df_pairs = df_pairs.sort_values(
        by=["cluster_UT", "cluster_CO", "n", "m"]
    ).reset_index(drop=True)

    # --- UT-only fallback (unknown co-runner) ---
    # geomean over ALL CO clusters and ALL models for each (cluster_UT, n, m)
    grp_ut = df_pairs.groupby(["cluster_UT", "n", "m"], as_index=False)
    df_ut = grp_ut["mult"].agg(geomean_of_mult)
    df_ut = df_ut.rename(columns={"mult": "mult_ut"})
    df_ut["ratio_ut_gm"] = df_ut["mult_ut"] - 1.0
    df_ut = df_ut[["cluster_UT", "n", "m", "ratio_ut_gm", "mult_ut"]]

    # sort UT table by cluster_UT, then n, then m
    df_ut = df_ut.sort_values(
        by=["cluster_UT", "n", "m"]
    ).reset_index(drop=True)

    # --- Write global CSVs ---
    df_pairs.to_csv(OUT_PAIR, index=False, float_format="%.6f")
    df_ut.to_csv(OUT_UT, index=False, float_format="%.6f")

    print(f"[OK] Wrote pair ratios: {OUT_PAIR.resolve()}  (rows={len(df_pairs)})")
    print(f"[OK] Wrote UT-only ratios: {OUT_UT.resolve()}  (rows={len(df_ut)})")
    print("[DONE]")

if __name__ == "__main__":
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 200)
    main()
