#!/usr/bin/env python3
# fix_knee_insert_after_block.py
#
# For each CSV-like file under ROOT_DIR matching PATTERN:
#   - find its matching TPC profile CSV under PROFILE_ROOT
#   - compute knee_tpc per kernel using row_knee_derivative_raw (same method as your script)
#   - remove existing knee_tpc if present
#   - INSERT knee_tpc right AFTER 'Block' column (or 'BlockXYZ' if 'Block' missing)
#
# Python 3.7+; no argparse.

from pathlib import Path
import re
import json
import shutil
import numpy as np
import pandas as pd

# =========================
# EDIT THESE SETTINGS
# =========================
ROOT_DIR = Path("/home/zixi/orion_bu/benchmarking/model_kernels/cudnn_based/test")
PATTERN  = "*_fwd"        # e.g., "*_fwd", "*" , "resnet152_8_fwd"
INPLACE  = True
OUT_DIR  = Path("/tmp/fwd_with_knee")  # used only if INPLACE=False
BACKUP   = False         # only used if INPLACE=True; creates .bak once

# Where the per-model TPC curves live (you gave this)
PROFILE_ROOT = Path("/home/zixi/orion_bu/artifact_evaluation/FICKS/A4000_TPC_profiles_cudnn_based")

# Optional: if present, use same param convention as your long script (defaults/models/exclusive)
KNEE_PARAMS_JSON = Path("/home/zixi/orion_bu/benchmarking/model_kernels/ficks/profiling/knee_params.json")

# Knee defaults if JSON missing or model not listed
DEFAULT_TAU_ABS  = 0.01
DEFAULT_TAU_FRAC = 0.10
DEFAULT_WIN      = 2
# =========================

MEM_RE = re.compile(r"memcpy|memset", re.IGNORECASE)
TPC_COL_RE = re.compile(r"^\s*(\d{1,3})\s*$")  # "1","2",...

def is_probably_csv(path: Path) -> bool:
    if (not path.is_file()) or path.name.startswith("."):
        return False
    if path.suffix.lower() == ".csv":
        return True
    try:
        with path.open("r", errors="ignore") as f:
            first = f.readline()
        return ("Name" in first) and ("," in first)
    except Exception:
        return False

def best_id_col(df: pd.DataFrame):
    """
    Find an ID column if it exists, else None.
    (Same spirit as your long script, but minimal.)
    """
    def norm(s):
        return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")

    norm_to_orig = {norm(c): c for c in df.columns}
    for cand in ["kernel_id", "kernelid", "kernel_index", "kernel_index_ut", "id", "idx", "index"]:
        if cand in norm_to_orig:
            return norm_to_orig[cand]

    # unnamed integer column sometimes appears
    for n, o in norm_to_orig.items():
        if n.startswith("unnamed"):
            try:
                if df[o].dtype.kind in "iu":
                    return o
            except Exception:
                pass
    return None

def detect_tpc_columns(df: pd.DataFrame):
    cols = []
    for c in df.columns:
        if isinstance(c, str):
            m = TPC_COL_RE.match(c)
            if m:
                cols.append((int(m.group(1)), c))
    cols.sort(key=lambda x: x[0])
    return cols  # list of (tpc_int, colname_str)

def row_knee_derivative_raw(n_list, l_list, tau_abs, tau_frac, win):
    """
    Exactly the method from your long script (logic preserved).
    Returns an int TPC knee, or np.nan if cannot compute.
    """
    n = np.asarray(n_list, dtype=float)
    Lr = np.asarray(l_list, dtype=float)
    mask = np.isfinite(Lr) & (Lr > 0)
    n, L = n[mask], Lr[mask]
    if n.size < 2:
        return np.nan

    denom = np.maximum(L[:-1], 1e-12)
    gains = (L[:-1] - L[1:]) / denom
    gains = np.maximum(gains, 0.0)
    if gains.size == 0:
        return int(n[-1])

    k0 = min(3, gains.size)
    baseline = float(np.median(gains[:k0])) if k0 > 0 else float(np.median(gains))
    tau = max(float(tau_abs), float(tau_frac) * baseline)

    if gains.size >= int(win):
        win = int(win)
        win_means = np.convolve(gains, np.ones(win) / win, mode="valid")
        for i, gm in enumerate(win_means):
            if np.isfinite(gm) and gm <= tau:
                idx = i + win - 1
                return int(n[idx])

    total = L[0] - np.nanmin(L)
    if total > 0:
        for i in range(len(L)):
            if (L[0] - L[i]) / total >= 0.90:
                return int(n[i])
    return int(n[-1])

def parse_sheet_from_filename(stem: str) -> str:
    """
    Convert e.g.
      resnet152_8_fwd  -> resnet152_bz8
      mobilenet_v2_32_fwd -> mobilenet_v2_bz32
    """
    m = re.match(r"^(?P<base>.+)_(?P<bz>\d+)_fwd$", stem)
    if m:
        return f"{m.group('base')}_bz{m.group('bz')}"
    # fallback: if it already contains bz
    if re.search(r"_bz\d+$", stem):
        return stem
    return stem

def load_knee_params():
    """
    Load defaults/models from knee_params.json if exists.
    Only use the 'exclusive' section (since we're selecting knee on TPC curves).
    """
    if not KNEE_PARAMS_JSON.exists():
        return None
    try:
        with open(KNEE_PARAMS_JSON, "r") as f:
            data = json.load(f)
        return data
    except Exception:
        return None

def params_for_sheet(sheet: str, data):
    if data is None:
        return DEFAULT_TAU_ABS, DEFAULT_TAU_FRAC, DEFAULT_WIN
    defaults = data.get("defaults", {}).get("exclusive", {})
    models   = data.get("models", {}).get(sheet, {}).get("exclusive", {})

    tau_abs  = float(models.get("tau_abs",  defaults.get("tau_abs",  DEFAULT_TAU_ABS)))
    tau_frac = float(models.get("tau_frac", defaults.get("tau_frac", DEFAULT_TAU_FRAC)))
    win      = int(  models.get("win",      defaults.get("win",      DEFAULT_WIN)))
    return tau_abs, tau_frac, win

def insert_knee_after_block(df: pd.DataFrame, knee_vals):
    # drop existing
    if "knee_tpc" in df.columns:
        df = df.drop(columns=["knee_tpc"])

    # find insert position
    insert_after = None
    if "Block" in df.columns:
        insert_after = "Block"
    elif "BlockXYZ" in df.columns:
        insert_after = "BlockXYZ"

    if insert_after is None:
        df["knee_tpc"] = knee_vals
        return df

    idx = list(df.columns).index(insert_after) + 1
    df.insert(idx, "knee_tpc", knee_vals)
    return df

def build_knee_map_from_profile(profile_csv: Path, tau_abs, tau_frac, win):
    """
    Returns dict: kernel_id(int) -> knee_tpc(int)
    Supports profile CSVs that have an ID column, else row-order.
    """
    prof = pd.read_csv(profile_csv)
    if prof.empty:
        return {}

    id_col = best_id_col(prof)
    tpc_cols = detect_tpc_columns(prof)
    if not tpc_cols:
        return {}

    ns = [n for n, _ in tpc_cols]
    cols = [c for _, c in tpc_cols]

    # numeric conversion
    work = prof.copy()
    work[cols] = work[cols].apply(pd.to_numeric, errors="coerce")

    knee_map = {}

    if id_col is not None:
        work[id_col] = pd.to_numeric(work[id_col], errors="coerce")
        for _, row in work.iterrows():
            kid = row[id_col]
            if pd.isna(kid):
                continue
            kid = int(kid)
            L = [row[c] for c in cols]
            k = row_knee_derivative_raw(ns, L, tau_abs=tau_abs, tau_frac=tau_frac, win=win)
            if pd.notna(k):
                knee_map[kid] = int(k)
    else:
        # row-order fallback
        for ridx in range(len(work)):
            row = work.iloc[ridx]
            L = [row[c] for c in cols]
            k = row_knee_derivative_raw(ns, L, tau_abs=tau_abs, tau_frac=tau_frac, win=win)
            if pd.notna(k):
                knee_map[int(ridx)] = int(k)

    return knee_map

def fix_one(path: Path, inplace: bool, out_dir: Path, backup: bool, params_data, profile_cache, knee_map_cache):
    if backup and inplace:
        bak = path.with_name(path.name + ".bak")
        if not bak.exists():
            shutil.copy2(path, bak)

    df = pd.read_csv(path)
    if df.empty:
        # still enforce knee_tpc placement
        df = insert_knee_after_block(df, [])
        if inplace:
            df.to_csv(path, index=False)
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_dir / path.name, index=False)
        return

    # infer sheet/model key from file name
    sheet = parse_sheet_from_filename(path.stem)   # e.g., resnet152_bz8
    profile_path = PROFILE_ROOT / f"{sheet}.csv"

    # select knee params
    tau_abs, tau_frac, win = params_for_sheet(sheet, params_data)

    # load knee_map for this sheet/profile
    knee_map = {}
    if profile_path.exists():
        cache_key = (str(profile_path), tau_abs, tau_frac, win)
        if cache_key in knee_map_cache:
            knee_map = knee_map_cache[cache_key]
        else:
            knee_map = build_knee_map_from_profile(profile_path, tau_abs, tau_frac, win)
            knee_map_cache[cache_key] = knee_map
    else:
        # no profile: fill -1
        knee_map = {}

    # map each row to kernel id
    id_col = best_id_col(df)
    knee_vals = []

    if id_col is not None:
        kids = pd.to_numeric(df[id_col], errors="coerce")
        for x in kids:
            if pd.isna(x):
                knee_vals.append(-1)
            else:
                knee_vals.append(int(knee_map.get(int(x), -1)))
    else:
        # row-order fallback
        for ridx in range(len(df)):
            knee_vals.append(int(knee_map.get(int(ridx), -1)))

    # insert right after Block / BlockXYZ
    df2 = insert_knee_after_block(df, knee_vals)

    if inplace:
        df2.to_csv(path, index=False)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        df2.to_csv(out_dir / path.name, index=False)

def main():
    if not ROOT_DIR.exists():
        raise SystemExit(f"ROOT_DIR does not exist: {ROOT_DIR}")
    if not PROFILE_ROOT.exists():
        raise SystemExit(f"PROFILE_ROOT does not exist: {PROFILE_ROOT}")

    params_data = load_knee_params()

    candidates = sorted(ROOT_DIR.rglob(PATTERN))
    targets = [p for p in candidates if is_probably_csv(p)]
    if not targets:
        raise SystemExit(f"No CSV-like files found under {ROOT_DIR} with pattern {PATTERN}")

    profile_cache = {}
    knee_map_cache = {}

    for p in targets:
        fix_one(p, inplace=INPLACE, out_dir=OUT_DIR, backup=BACKUP,
                params_data=params_data, profile_cache=profile_cache, knee_map_cache=knee_map_cache)

    print(f"Processed {len(targets)} files:")
    for p in targets:
        print("  ", p)

if __name__ == "__main__":
    main()
