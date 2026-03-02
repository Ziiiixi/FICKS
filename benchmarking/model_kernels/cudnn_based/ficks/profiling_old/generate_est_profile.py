#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate estimated colocate latency curves from *existing fwd kernel tables*.

Your fwd files already contain per-kernel exclusive curves:
  Kernel_ID, Name, cluster, ..., excl_1, excl_2, ..., excl_24

So we DO NOT read EXCL_ROOT at all.

We only load:
  - Geo multiplier table: A4000_ratios_ut.csv
      cluster_UT, n, m, ratio_ut_gm, mult_ut
    (robust to BOM / header name variants)

Estimation:
  est_coloc_<n> = excl_<n> * mult_ut(cluster, n)

Output:
  MODEL_ROOT/estimated_coloc_curves/<fwd_filename>_est_coloc.csv
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================
MODEL_ROOT = Path("/home/zixi/orion_bu/benchmarking/model_kernels/cudnn_based/ficks")

OUT_DIR = MODEL_ROOT / "estimated_coloc_curves"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GEO_MULT_CSV = MODEL_ROOT / "profiling" / "A4000_ratios_ut.csv"


# ============================================================
# SETTINGS
# ============================================================
MIN_TPC_KEEP = 1

# Output verbosity
WRITE_RATIO_COLS = True     # write ratio_ut_gm_<n>
WRITE_MULT_COLS  = True     # write mult_ut_<n>

# Debug prints
DEBUG = True
DEBUG_MODEL = "resnet152"
DEBUG_KERNEL_ID = 34
DEBUG_PRINT_FIRST_N = 12


# ============================================================
# REGEX
# ============================================================
EXCL_COL_RE = re.compile(r"^excl_(\d{1,3})$")
FWD_NAME_RE = re.compile(r".+_\d+_fwd(\.csv)?$", re.IGNORECASE)


# ============================================================
# HELPERS: check if a file looks like your fwd kernel table
# ============================================================
def looks_like_fwd_kernel_table(path: Path) -> bool:
    """
    A valid fwd kernel table file must have:
      - Kernel_ID
      - cluster
      - at least one excl_<n>
    """
    if not path.exists() or not path.is_file():
        return False
    try:
        df = pd.read_csv(path, nrows=5)
    except Exception:
        return False
    if df is None or df.empty:
        return False

    cols = set(df.columns)
    if "Kernel_ID" not in cols:
        return False
    if "cluster" not in cols:
        return False

    has_excl = any(isinstance(c, str) and c.startswith("excl_") for c in df.columns)
    return has_excl


# ============================================================
# DISCOVERY: your _fwd are FILES not DIRS
# ============================================================
def discover_fwd_table_files(root: Path) -> List[Path]:
    """
    Discover fwd kernel table files under MODEL_ROOT.

    Priority:
      (1) directly under MODEL_ROOT:  resnet152_8_fwd  or resnet152_8_fwd.csv
      (2) fallback recursive search:  root.rglob("*_fwd") / "*_fwd.csv"
    """
    files: List[Path] = []

    if not root.exists():
        return files

    # (1) direct children first
    for p in sorted(root.iterdir()):
        if p.name in {"profiling", "estimated_coloc_curves"}:
            continue
        if p.is_file():
            if FWD_NAME_RE.match(p.name) and looks_like_fwd_kernel_table(p):
                files.append(p)

    if files:
        return files

    # (2) fallback recursive search (depth any)
    #     This is robust if your fwd tables got moved somewhere else.
    cands = []
    for pat in ["*_fwd", "*_fwd.csv"]:
        cands.extend(list(root.rglob(pat)))

    # filter + stable ordering
    uniq = []
    seen = set()
    for p in sorted(cands):
        if not p.is_file():
            continue
        # skip output/profiling artifacts
        if "estimated_coloc_curves" in str(p):
            continue
        if p.name.startswith("."):
            continue
        if p in seen:
            continue
        seen.add(p)
        if looks_like_fwd_kernel_table(p):
            uniq.append(p)

    return uniq


def parse_model_key_from_fwd_name(filename: str) -> Tuple[str, str]:
    """
    From filename:
      resnet152_8_fwd      -> sheet=resnet152_bz8, model_key=resnet152
      mobilenet_v2_32_fwd  -> sheet=mobilenet_v2_bz32, model_key=mobilenet_v2

    Also works if filename has .csv suffix.
    """
    name = filename
    if name.lower().endswith(".csv"):
        name = name[:-4]

    m = re.match(r"^(?P<base>.+)_(?P<bz>\d+)_fwd$", name)
    if m:
        base = m.group("base")
        bz = m.group("bz")
        return f"{base}_bz{bz}", base

    # fallback
    if name.endswith("_fwd"):
        base = name[:-4]
        return base, base

    return name, name


# ============================================================
# GEO MULT TABLE LOADING (ROBUST)
# ============================================================
def load_geo_mult_table(path: Path) -> Dict[int, Dict[int, Dict[str, float]]]:
    """
    Return:
      geo[cluster_UT][n] = {"m": m, "ratio": ratio_ut_gm, "mult": mult_ut}

    Robust to BOM / spaces / case / missing mult_ut (derived from ratio_ut_gm).
    """
    if not path.exists():
        raise FileNotFoundError(f"GEO mult CSV not found: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    def norm_col(c: str) -> str:
        c = str(c).replace("\ufeff", "")
        c = c.strip().lower()
        return c

    df = df.rename(columns={c: norm_col(c) for c in df.columns})

    def pick_col(*cands: str) -> Optional[str]:
        for c in cands:
            if c in df.columns:
                return c
        return None

    c_cluster = pick_col("cluster_ut", "cluster", "clusterut")
    c_n = pick_col("n", "tpc_n", "ut_n")
    c_m = pick_col("m", "tpc_m", "ut_m")
    c_ratio = pick_col("ratio_ut_gm", "ratio_gm", "ratio")
    c_mult = pick_col("mult_ut", "mult", "multiplier")

    # derive mult_ut if missing
    if c_mult is None and c_ratio is not None:
        df["mult_ut"] = pd.to_numeric(df[c_ratio], errors="coerce") + 1.0
        c_mult = "mult_ut"

    ok = (c_cluster is not None) and (c_n is not None) and (c_m is not None) and (c_mult is not None)
    if not ok:
        raise KeyError(
            f"[ERR] GEO mult CSV missing required logical columns.\n"
            f"Columns seen: {list(df.columns)}\n"
            f"Resolved: cluster={c_cluster}, n={c_n}, m={c_m}, ratio={c_ratio}, mult={c_mult}"
        )

    # ensure ratio exists
    if c_ratio is None:
        df["ratio_ut_gm"] = pd.to_numeric(df[c_mult], errors="coerce") - 1.0
        c_ratio = "ratio_ut_gm"

    df[c_cluster] = pd.to_numeric(df[c_cluster], errors="coerce").astype("Int64")
    df[c_n] = pd.to_numeric(df[c_n], errors="coerce").astype("Int64")
    df[c_m] = pd.to_numeric(df[c_m], errors="coerce").astype("Int64")
    df[c_ratio] = pd.to_numeric(df[c_ratio], errors="coerce")
    df[c_mult] = pd.to_numeric(df[c_mult], errors="coerce")

    df = df.dropna(subset=[c_cluster, c_n, c_m, c_ratio, c_mult])

    geo: Dict[int, Dict[int, Dict[str, float]]] = {}
    for _, r in df.iterrows():
        cluster = int(r[c_cluster])
        n = int(r[c_n])
        m = int(r[c_m])
        ratio = float(r[c_ratio])
        mult = float(r[c_mult])

        geo.setdefault(cluster, {})
        geo[cluster][n] = {"m": float(m), "ratio": float(ratio), "mult": float(mult)}

    return geo


# ============================================================
# EXCL COLUMN PARSING
# ============================================================
def get_excl_ns_from_columns(cols: List[str]) -> List[int]:
    ns = []
    for c in cols:
        if not isinstance(c, str):
            continue
        m = EXCL_COL_RE.match(c.strip())
        if m:
            ns.append(int(m.group(1)))
    ns = sorted(set([n for n in ns if n >= int(MIN_TPC_KEEP)]))
    return ns


# ============================================================
# CORE: build estimated coloc curve from fwd table
# ============================================================
def build_estimated_coloc_from_fwd(
    fwd_df: pd.DataFrame,
    geo: Dict[int, Dict[int, Dict[str, float]]]
) -> pd.DataFrame:
    """
    Input fwd_df must contain:
      Kernel_ID, Name, cluster, excl_<n>...

    Output adds:
      est_coloc_<n>
      (optional) ratio_ut_gm_<n>, mult_ut_<n>
    """
    if fwd_df is None or fwd_df.empty:
        return pd.DataFrame()

    need = {"Kernel_ID", "cluster"}
    if not need.issubset(set(fwd_df.columns)):
        raise KeyError(f"FWD df missing required columns {need}. Got={list(fwd_df.columns)}")

    out = fwd_df.copy()

    # normalize types
    out["Kernel_ID"] = pd.to_numeric(out["Kernel_ID"], errors="coerce").astype("Int64")
    out["cluster"] = pd.to_numeric(out["cluster"], errors="coerce").fillna(-1).astype(int)

    excl_ns = get_excl_ns_from_columns(list(out.columns))
    if not excl_ns:
        raise RuntimeError("No excl_<n> columns found in fwd file.")

    # add output cols
    for n in excl_ns:
        est_col = f"est_coloc_{n}"
        out[est_col] = np.nan

        if WRITE_RATIO_COLS:
            out[f"ratio_ut_gm_{n}"] = np.nan
        if WRITE_MULT_COLS:
            out[f"mult_ut_{n}"] = np.nan

    # compute
    for idx, row in out.iterrows():
        c = int(row["cluster"])
        if c < 0:
            continue

        geo_c = geo.get(c, None)
        if geo_c is None:
            continue

        for n in excl_ns:
            excl_col = f"excl_{n}"
            est_col = f"est_coloc_{n}"

            excl_us = pd.to_numeric(row.get(excl_col, np.nan), errors="coerce")
            if not np.isfinite(excl_us) or float(excl_us) <= 0:
                continue

            info = geo_c.get(int(n), None)
            if info is None:
                continue

            mult = float(info["mult"])
            ratio = float(info["ratio"])

            out.at[idx, est_col] = float(excl_us) * float(mult)

            if WRITE_RATIO_COLS:
                out.at[idx, f"ratio_ut_gm_{n}"] = ratio
            if WRITE_MULT_COLS:
                out.at[idx, f"mult_ut_{n}"] = mult

    return out


# ============================================================
# DEBUG PRINT
# ============================================================
def debug_print_kernel(out_df: pd.DataFrame, model_key: str, kernel_id: int, k: int):
    hit = out_df[out_df["Kernel_ID"] == int(kernel_id)]
    if hit.empty:
        print(f"[DEBUG] model={model_key}: Kernel_ID={kernel_id} not found.")
        return

    row = hit.iloc[0].to_dict()
    name = str(row.get("Name", ""))
    cluster = row.get("cluster", None)

    print(f"\n[DEBUG] ===== EST COLOC CHECK =====")
    print(f"  model={model_key} Kernel_ID={kernel_id} cluster={cluster}")
    print(f"  Name: {name[:140]}")
    print(f"  est_coloc_<n> = excl_<n> * mult_ut(cluster,n)")
    print(f"  Showing first {k} n values found in file")

    est_cols = [c for c in out_df.columns if isinstance(c, str) and c.startswith("est_coloc_")]
    est_cols = sorted(est_cols, key=lambda x: int(x.split("_")[-1]))

    shown = 0
    for ec in est_cols:
        n = int(ec.split("_")[-1])
        excl = row.get(f"excl_{n}", np.nan)
        mult = row.get(f"mult_ut_{n}", np.nan)
        ratio = row.get(f"ratio_ut_gm_{n}", np.nan)
        est = row.get(ec, np.nan)

        if not np.isfinite(pd.to_numeric(excl, errors="coerce")):
            continue

        print(f"    n={n:2d} | excl_us={float(excl):10.4f} | ratio_gm={float(ratio):8.4f} | mult={float(mult):8.4f} | est_coloc_us={float(est):10.4f}")
        shown += 1
        if shown >= int(k):
            break

    print(f"[DEBUG] ============================\n")


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"[INFO] MODEL_ROOT = {MODEL_ROOT}")
    print(f"[INFO] OUT_DIR    = {OUT_DIR}")
    print(f"[INFO] GEO_MULT   = {GEO_MULT_CSV}")

    # load geo mult table
    geo = load_geo_mult_table(GEO_MULT_CSV)
    print(f"[INFO] Loaded geo_mult for {len(geo)} clusters")

    # discover fwd table files
    fwd_files = discover_fwd_table_files(MODEL_ROOT)

    if not fwd_files:
        print(f"[ERR] No fwd TABLE FILES found under {MODEL_ROOT}")
        print(f"[ERR] Items under MODEL_ROOT are:")
        for p in sorted(MODEL_ROOT.iterdir()):
            t = "DIR" if p.is_dir() else "FILE"
            print(f"   [{t}] {p.name}")
        raise RuntimeError("No fwd kernel tables discovered (expected '*_<bz>_fwd' files)")

    print(f"[INFO] Discovered {len(fwd_files)} fwd kernel table files:")
    for f in fwd_files:
        print("   ", str(f))

    total_ok = 0

    for fwd_path in fwd_files:
        sheet, model_key = parse_model_key_from_fwd_name(fwd_path.name)

        print(f"\n[INFO] Processing fwd table: {fwd_path.name}")
        print(f"       sheet={sheet}, model={model_key}")

        try:
            fwd_df = pd.read_csv(fwd_path)
            out_df = build_estimated_coloc_from_fwd(fwd_df, geo)
        except Exception as e:
            print(f"[ERR] Failed on {fwd_path}: {e}")
            continue

        # output filename
        base_name = fwd_path.name
        if base_name.lower().endswith(".csv"):
            base_name = base_name[:-4]

        out_path = OUT_DIR / f"{base_name}_est_coloc.csv"
        out_df.to_csv(out_path, index=False)

        print(f"[OK] wrote {out_path} | rows={len(out_df)} cols={len(out_df.columns)}")
        total_ok += 1

        if DEBUG and str(model_key).lower() == str(DEBUG_MODEL).lower():
            debug_print_kernel(out_df, model_key=model_key, kernel_id=int(DEBUG_KERNEL_ID), k=int(DEBUG_PRINT_FIRST_N))

    print("\n[DONE]")
    print(f"Processed fwd tables: {total_ok}")
    print(f"Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    pd.set_option("display.max_columns", 300)
    pd.set_option("display.width", 240)
    main()
