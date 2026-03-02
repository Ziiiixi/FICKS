#!/usr/bin/env python3
# fix_knee_add_real_knee.py
#
# For each CSV-like *_fwd file under ROOT_DIR:
#   - remove existing "knee_tpc" (case-insensitive) column if present
#   - compute a REAL knee point per row from a perf CSV:
#         PERF_DIR/<model>_bz<batch>.csv
#     where perf CSV has numeric TPC columns like: "1","2","4","6",...
#   - append "knee_tpc" as the LAST column (int), default -1 if cannot compute
#
# Differences vs your old "fill -1" script:
#   - actually reads perf curves and runs knee detection
#   - normalizes mobilenet_v2 -> mobilenetv2 when locating perf file
#   - robust to missing kneed/scipy: will fallback to derivative heuristic
#
# Python 3.7+

from __future__ import annotations

from pathlib import Path
import shutil
import re
import os
import math
import warnings

import numpy as np
import pandas as pd

# =========================
# EDIT THESE SETTINGS
# =========================
ROOT_DIR = Path("/home/zixi/orion_bu/benchmarking/model_kernels/cudnn_based/others")
PATTERN  = "*_fwd"                 # e.g., "*_fwd", "*" , "resnet152_8_fwd"
INPLACE  = True                    # True: modify files in place; False: write to OUT_DIR
OUT_DIR  = Path("/tmp/fwd_with_knee")  # used only if INPLACE=False
BACKUP   = False                   # only used if INPLACE=True; creates .bak once

# Where perf curves live:
# PERF CSV naming must be: <model>_bz<batch>.csv
# numeric TPC columns must be digit strings: "1","2","4",...
PERF_DIR = Path("/home/zixi/orion_bu/artifact_evaluation/FICKS/A4000_TPC_profiles_cudnn_based")

# Optional: ignore very small TPCs when finding knee (as in your reference code)
MIN_TPC_FOR_KNEE = 6

# Fallback threshold for marginal improvement (if kneed not available)
FALLBACK_REL_IMPROVE = 0.05  # 5%
# =========================


# ------------------------- Optional deps (scipy, kneed) -------------------------
_HAVE_SAVGOL = False
_HAVE_KNEED = False

try:
    from scipy.signal import savgol_filter  # type: ignore
    _HAVE_SAVGOL = True
except Exception:
    _HAVE_SAVGOL = False

try:
    from kneed import KneeLocator  # type: ignore
    _HAVE_KNEED = True
except Exception:
    _HAVE_KNEED = False


# ------------------------- Helpers -------------------------
def is_probably_csv(path: Path) -> bool:
    if not path.is_file() or path.name.startswith("."):
        return False
    if path.suffix.lower() == ".csv":
        return True
    # heuristic for your fwd-like "csv without extension"
    try:
        with path.open("r", errors="ignore") as f:
            first = f.readline()
        return ("Name" in first) and ("SM_usage" in first) and ("Duration" in first) and ("," in first)
    except Exception:
        return False


def normalize_model_name(model: str) -> str:
    # you explicitly want this mapping for perf file lookup
    if model == "mobilenet_v2":
        return "mobilenet_v2"
    return model


def parse_model_batch_from_fwd_filename(stem: str) -> tuple[str, int] | None:
    """
    Accepts:
      <model>_<batch>_fwd
      <model>_<batch>_fwd_KRISP   (if you have these)
    Returns (model, batch) or None.
    """
    m = re.match(r"(.+?)_(\d+)_fwd(?:_KRISP)?$", stem)
    if not m:
        return None
    model, batch_s = m.groups()
    try:
        batch = int(batch_s)
    except ValueError:
        return None
    return model, batch


def numeric_tpc_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        cs = str(c).strip()
        if cs.isdigit():
            cols.append(cs)
    # keep unique, sorted by int
    cols = sorted(set(cols), key=lambda x: int(x))
    return cols


def _smooth_curve(y_raw: np.ndarray) -> np.ndarray:
    """
    Light smoothing. If scipy is not available or too short, returns original.
    """
    y = np.array(y_raw, dtype=float)
    if y.size < 3:
        return y
    if not _HAVE_SAVGOL:
        return y

    # window must be odd and <= len(y)
    wl = min(7, y.size)
    if wl % 2 == 0:
        wl -= 1
    if wl < 3:
        return y

    try:
        return savgol_filter(y, window_length=wl, polyorder=2)
    except Exception:
        return y


def find_knee_tpc(x: np.ndarray, y_raw: np.ndarray) -> int:
    """
    x: increasing TPC array (int)
    y_raw: latency array (float), where lower is better (decreasing curve)
    Returns knee TPC as int. Falls back to last x if cannot decide.
    """
    x = np.array(x, dtype=int)
    y_raw = np.array(y_raw, dtype=float)

    # Filter invalid
    mask = np.isfinite(y_raw) & np.isfinite(x)
    x = x[mask]
    y_raw = y_raw[mask]

    # Optionally ignore small tpcs
    if MIN_TPC_FOR_KNEE is not None:
        keep = x >= int(MIN_TPC_FOR_KNEE)
        if keep.sum() >= 2:
            x = x[keep]
            y_raw = y_raw[keep]

    if x.size == 0:
        return -1
    if x.size < 3:
        return int(x[-1])

    y_smooth = _smooth_curve(y_raw)

    # 1) Kneedle if available
    if _HAVE_KNEED:
        try:
            noise = float(np.std(y_smooth - y_raw))
            signal = float(np.nanmax(y_smooth) - np.nanmin(y_smooth))
            S = max(1.0, signal / (noise + 1e-6))

            kl = KneeLocator(
                x, y_smooth,
                curve="convex", direction="decreasing",
                interp_method="polynomial",
                polynomial_degree=2,
                S=S
            )
            if kl.knee is not None:
                return int(kl.knee)
        except Exception:
            pass

    # 2) Fallback: first point where marginal improvement < threshold
    # Improvement between i and i+1: (y[i] - y[i+1]) / y[i]
    diffs = y_smooth[:-1] - y_smooth[1:]
    denom = np.where(np.abs(y_smooth[:-1]) < 1e-12, 1e-12, np.abs(y_smooth[:-1]))
    rel = diffs / denom
    small = np.where(rel < float(FALLBACK_REL_IMPROVE))[0]
    if small.size > 0:
        return int(x[small[0] + 1])

    # 3) Last resort: max-distance to y=x diagonal in normalized space
    xn = (x - x.min()) / (x.max() - x.min() + 1e-12)
    yn = (y_smooth - np.nanmin(y_smooth)) / (np.nanmax(y_smooth) - np.nanmin(y_smooth) + 1e-12)
    dist = np.abs(xn + yn - 1.0)
    return int(x[int(np.nanargmax(dist))])


def load_perf_knee_series(model: str, batch: int, n_rows_expected: int | None) -> pd.Series:
    """
    Read PERF_DIR/<model>_bz<batch>.csv and compute Knee_TPC for each row.

    Returns a pd.Series of ints length == perf_df rows.
    Raises on missing perf or missing tpc cols.
    """
    model_norm = normalize_model_name(model)
    perf_path = PERF_DIR / f"{model_norm}_bz{batch}.csv"
    if not perf_path.exists():
        raise FileNotFoundError(f"Perf file missing: {perf_path}")

    perf_df = pd.read_csv(perf_path)
    perf_df.columns = perf_df.columns.map(lambda s: str(s).strip())

    tpc_cols = numeric_tpc_cols(perf_df)
    if not tpc_cols:
        raise RuntimeError(f"No numeric TPC columns (e.g., '1','2','4',...) found in {perf_path}")

    x = np.array([int(c) for c in tpc_cols], dtype=int)

    # compute knee per row
    knees = perf_df[tpc_cols].apply(lambda row: find_knee_tpc(x, row.values.astype(float)), axis=1)

    if n_rows_expected is not None and len(knees) != n_rows_expected:
        # Do not crash; but warn loudly.
        warnings.warn(
            f"[WARN] Row count mismatch: fwd rows={n_rows_expected} vs perf rows={len(knees)} "
            f"for model={model_norm}_bz{batch}. Will align by min length and set remaining to -1."
        )

    return knees.astype(int)


def drop_knee_col_case_insensitive(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    drop = [c for c in cols if str(c).strip().lower() == "knee_tpc"]
    if drop:
        return df.drop(columns=drop)
    return df


def fix_one(path: Path, inplace: bool, out_dir: Path, backup: bool):
    if backup and inplace:
        bak = path.with_name(path.name + ".bak")
        if not bak.exists():
            shutil.copy2(path, bak)

    # Read fwd file
    fwd_df = pd.read_csv(path)

    # Remove existing knee_tpc (any case)
    fwd_df = drop_knee_col_case_insensitive(fwd_df)

    # Parse model/batch from filename
    stem = path.name  # for no-extension files, name is fine
    parsed = parse_model_batch_from_fwd_filename(stem)
    if parsed is None:
        # If the file isn't in <model>_<batch>_fwd format, just append -1
        fwd_df["knee_tpc"] = -1
        if inplace:
            fwd_df.to_csv(path, index=False)
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            fwd_df.to_csv(out_dir / path.name, index=False)
        print(f"[SKIP-KNEE] cannot parse model/batch from filename: {path} (wrote knee_tpc=-1)")
        return

    model, batch = parsed

    # Compute knee series from perf
    try:
        knees = load_perf_knee_series(model, batch, n_rows_expected=len(fwd_df))
    except Exception as e:
        # perf missing or error -> fill -1 but do not crash
        fwd_df["knee_tpc"] = -1
        if inplace:
            fwd_df.to_csv(path, index=False)
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            fwd_df.to_csv(out_dir / path.name, index=False)
        print(f"[WARN] {path}: failed to compute knee ({e}); wrote knee_tpc=-1")
        return

    # Align lengths safely
    n = min(len(fwd_df), len(knees))
    out_knee = np.full(shape=(len(fwd_df),), fill_value=-1, dtype=int)
    out_knee[:n] = knees.values[:n]

    # Append as LAST column
    fwd_df["knee_tpc"] = out_knee

    # Write
    if inplace:
        fwd_df.to_csv(path, index=False)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        fwd_df.to_csv(out_dir / path.name, index=False)

    print(f"[OK] {path}: wrote knee_tpc from {normalize_model_name(model)}_bz{batch}.csv (n={len(fwd_df)})")


def main():
    if not ROOT_DIR.exists():
        raise SystemExit(f"ROOT_DIR does not exist: {ROOT_DIR}")

    candidates = sorted(ROOT_DIR.rglob(PATTERN))
    targets = [p for p in candidates if is_probably_csv(p)]

    if not targets:
        raise SystemExit(f"No CSV-like files found under {ROOT_DIR} with pattern {PATTERN}")

    print(f"[INFO] PERF_DIR={PERF_DIR}  (kneed={_HAVE_KNEED}, savgol={_HAVE_SAVGOL})")
    for p in targets:
        fix_one(p, inplace=INPLACE, out_dir=OUT_DIR, backup=BACKUP)

    print(f"\nProcessed {len(targets)} files.")
    if not INPLACE:
        print(f"Outputs under: {OUT_DIR}")


if __name__ == "__main__":
    main()
