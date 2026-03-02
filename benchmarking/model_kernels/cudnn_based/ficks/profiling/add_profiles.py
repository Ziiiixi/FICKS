#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
For each model CSV under MODEL_ROOT (Orion directory), produce a new CSV under OUTPUT_ROOT
with columns:

    Kernel_ID, Name, cluster, knee_coloc_n, exclusive_knee_n, is_short_kernel, excl_1, excl_2, ..., excl_N

where:
  - Kernel_ID: per-kernel stable ID (prefer from input CSV, fallback to row order)
  - Name: kernel name
  - cluster: UT cluster (from A4000_kernels_profiles_all_models.csv)
  - knee_coloc_n: MANUAL per-kernel coloc knee from profiling/knee_coloc/<model>_est_coloc_knee_overrides.csv
  - exclusive_knee_n: MANUAL per-kernel exclusive knee from profiling/knee_excl/<model>_knee_overrides.csv
  - is_short_kernel: 1 if short-running, else 0
  - excl_k: exclusive latency at TPC = k in microseconds (from EXCL_ROOT exclusive profile CSVs)

Knee policy:
  - Only use manual override files
  - Only use knee_tpc_new
  - If knee_tpc_new is missing, fill from sat_tpc_auto and write back to override CSV
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# PATHS (SERVER)
# ============================================================
MODEL_ROOT   = Path("/home/zixi/orion_bu/benchmarking/model_kernels/cudnn_based/others")
OUTPUT_ROOT  = Path("/home/zixi/orion_bu/benchmarking/model_kernels/cudnn_based/ficks")

PROFILING_DIR = OUTPUT_ROOT / "profiling"
CLUSTER_CSV   = PROFILING_DIR / "A4000_kernels_profiles_all_models.csv"

# Manual knee override dirs
KNEE_COLOC_DIR = PROFILING_DIR / "knee_coloc"
KNEE_EXCL_DIR  = PROFILING_DIR / "knee_excl"

# Exclusive profiles (for excl_1..excl_24 curves)
EXCL_ROOT = Path("/home/zixi/orion_bu/artifact_evaluation/FICKS/A4000_TPC_profiles_cudnn_based")


# ============================================================
# SETTINGS
# ============================================================
# duration_proxy = mean of worst-K excl latencies over tpc >= MIN_TPC_FOR_SHORT
MIN_TPC_FOR_SHORT = 1
WORST_K_FOR_SHORT = 20

SHORT_COL_NAME = "is_short_kernel"


# ============================================================
# REGEX
# ============================================================
TPC_COL_RE = re.compile(r"^\s*(\d{1,3})\s*$")  # "1","2","24", ...


# ============================================================
# UTILITIES
# ============================================================
def discover_model_csvs(root: Path):
    """
    Find model kernel CSVs under MODEL_ROOT.
    Your naming style: *_<bz>_fwd or *_<bz>_fwd.csv
    """
    files = []
    for p in sorted(root.iterdir()):
        if p.is_file() and (p.suffix.lower() == ".csv" or p.suffix == ""):
            if re.search(r"_\d+_fwd(\.csv)?$", p.name):
                files.append(p)
    return files


def filestem_to_sheet(stem: str) -> str:
    """
    resnet152_8_fwd -> resnet152_bz8
    """
    m = re.match(r"^(?P<base>.+)_(?P<bz>\d+)_fwd$", stem)
    if m:
        return f"{m.group('base')}_bz{m.group('bz')}"
    m2 = re.search(r"_(\d+)$", stem)
    if m2:
        base = stem[: m2.start()]
        return f"{base}_bz{m2.group(1)}"
    return stem


def sheet_to_model(sheet: str) -> str:
    return re.sub(r"_bz\d+$", "", sheet)


def best_id_col(df: pd.DataFrame) -> Optional[str]:
    """
    Try to find a kernel ID column.
    """
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")

    norm_to_orig = {norm(c): c for c in df.columns}

    kernel_id_candidates = ["kernel_index", "kernel_id", "kernelid", "kernel"]
    generic_id_candidates = ["id", "idx", "index"]

    for cand in kernel_id_candidates + generic_id_candidates:
        if cand in norm_to_orig:
            return norm_to_orig[cand]

    for n, o in norm_to_orig.items():
        if n.startswith("unnamed") and df[o].dtype.kind in "iu":
            return o

    return None


def best_name_col(df: pd.DataFrame) -> Optional[str]:
    """
    Try to find kernel name column robustly.
    """
    candidates = ["Name", "Kernel_Name", "Kernel Name", "kernel_name", "signature", "Signature"]
    for c in candidates:
        if c in df.columns:
            return c

    obj_cols = [c for c in df.columns if df[c].dtype == object]
    return obj_cols[0] if obj_cols else None


def parse_int_like(v):
    """
    Parse value into int if possible.
    Supports:
      - numeric like 12 / 12.0
      - strings like "12", "excl_12", "knee@12", "est_coloc_12"
    """
    if v is None:
        return None
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        if np.isfinite(v):
            return int(round(float(v)))
        return None
    s = str(v).strip()
    if s == "" or s.lower() in ["nan", "none"]:
        return None
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    return int(m.group(1))


def safe_read_csv(path: Path) -> pd.DataFrame:
    """
    More tolerant CSV loader (handles BOM).
    """
    return pd.read_csv(path, encoding="utf-8-sig")


def safe_write_csv(df: pd.DataFrame, path: Path):
    """
    Excel-friendly output.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


# ============================================================
# CLUSTER MAPPING (Kernel_ID -> cluster)
# ============================================================
def build_id_to_cluster_map_for_model(kern_all: pd.DataFrame, model_key: str) -> pd.DataFrame:
    sub = kern_all[kern_all["Model"] == model_key].copy()
    if sub.empty:
        raise ValueError(f"No rows in cluster CSV for model '{model_key}'")

    sub["Kernel_ID"] = pd.to_numeric(sub["Kernel_ID"], errors="coerce")
    sub = sub.dropna(subset=["Kernel_ID"])
    sub["Kernel_ID"] = sub["Kernel_ID"].astype(int)

    if "cluster" not in sub.columns:
        raise KeyError("Cluster CSV must contain a 'cluster' column")

    map_df = sub[["Kernel_ID", "cluster"]].copy()
    map_df = map_df.rename(columns={"Kernel_ID": "_id"})
    map_df = map_df.drop_duplicates(subset=["_id"])
    map_df["_id"] = map_df["_id"].astype(int)
    map_df["cluster"] = map_df["cluster"].astype(int)
    return map_df[["_id", "cluster"]]


# ============================================================
# EXCLUSIVE CSV HELPERS
# ============================================================
def detect_tpc_columns(excl_df: pd.DataFrame):
    """
    Detect columns that are purely numeric like "1","2","24".
    """
    tpcs = []
    for c in excl_df.columns:
        if isinstance(c, str):
            m = TPC_COL_RE.match(c)
            if m:
                tpcs.append((int(m.group(1)), c))
    tpcs.sort(key=lambda x: x[0])
    return tpcs


def build_exclusive_wide_for_df(excl_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a wide df indexed by the detected ID column
    with columns 'excl_<n>' in microseconds.
    """
    if excl_df is None or excl_df.empty:
        return pd.DataFrame()

    id_col = best_id_col(excl_df)
    if id_col is None:
        print("[WARN] Exclusive CSV has no ID-like column.")
        return pd.DataFrame()

    tpcs = detect_tpc_columns(excl_df)
    if not tpcs:
        print("[WARN] Exclusive CSV has no numeric TPC columns.")
        return pd.DataFrame()

    cols = [c for _, c in tpcs]
    df = excl_df[[id_col] + cols].copy()
    df[id_col] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    df = df.dropna(subset=[id_col])

    rename_map = {c: f"excl_{n}" for (n, c) in tpcs}
    df = df.rename(columns=rename_map)

    df = df.drop_duplicates(subset=[id_col]).set_index(id_col)
    df = df.reindex(sorted(df.columns, key=lambda x: int(x.split("_")[1])), axis=1)
    return df


def choose_exclusive_key_for_sheet(sheet: str, available_keys):
    """
    Your exclusive profile CSVs live under EXCL_ROOT and are keyed by file stem.
    """
    s = sheet.lower()
    base = re.sub(r"_bz\d+$", "", s)
    candidates = [
        s,
        s.replace(" ", "_"),
        base,
        base + "_8_fwd",
        base + "_32_fwd",
        base + "_fwd",
    ]
    for cand in candidates:
        if cand in available_keys:
            return cand
    starts = [k for k in available_keys if k.startswith(base)]
    return sorted(starts)[0] if starts else None


# ============================================================
# MANUAL KNEE OVERRIDE LOADING
# ============================================================
def fill_knee_tpc_new_with_sat_tpc_auto_inplace(override_csv: Path) -> int:
    """
    If knee_tpc_new missing/unparseable, fill with sat_tpc_auto.
    Writes back to override_csv.

    Required columns:
      kernel_id
      knee_tpc_new
      sat_tpc_auto
    """
    if not override_csv.exists():
        return 0

    df = safe_read_csv(override_csv)
    if df is None or df.empty:
        return 0

    if "kernel_id" not in df.columns:
        return 0

    if "knee_tpc_new" not in df.columns:
        df["knee_tpc_new"] = np.nan

    if "sat_tpc_auto" not in df.columns:
        return 0

    df["kernel_id"] = pd.to_numeric(df["kernel_id"], errors="coerce")
    df = df.dropna(subset=["kernel_id"]).copy()
    df["kernel_id"] = df["kernel_id"].astype(int)

    filled = 0
    for i in range(len(df)):
        knee = parse_int_like(df.iloc[i].get("knee_tpc_new", None))
        if knee is not None:
            continue

        sat = parse_int_like(df.iloc[i].get("sat_tpc_auto", None))
        if sat is None:
            continue

        df.at[df.index[i], "knee_tpc_new"] = int(sat)
        filled += 1

    if filled > 0:
        safe_write_csv(df, override_csv)

    return filled


def load_manual_knee_map(override_csv: Path) -> Dict[int, int]:
    """
    Load mapping:
      kernel_id -> knee_tpc_new
    """
    if not override_csv.exists():
        return {}

    df = safe_read_csv(override_csv)
    if df is None or df.empty:
        return {}

    if "kernel_id" not in df.columns or "knee_tpc_new" not in df.columns:
        return {}

    df["kernel_id"] = pd.to_numeric(df["kernel_id"], errors="coerce")
    df = df.dropna(subset=["kernel_id"]).copy()
    df["kernel_id"] = df["kernel_id"].astype(int)

    out: Dict[int, int] = {}
    for _, r in df.iterrows():
        kid = int(r["kernel_id"])
        knee = parse_int_like(r.get("knee_tpc_new", None))
        if knee is None:
            continue
        out[kid] = int(knee)
    return out


# ============================================================
# SHORT-KERNEL FLAG
# ============================================================
def duration_proxy_worstk_from_excl(
    updated_df: pd.DataFrame,
    excl_cols: list[str],
    min_tpc: int,
    worst_k: int,
) -> pd.Series:
    """
    duration_proxy per kernel = mean of top-K largest excl_<tpc> values over tpc >= min_tpc
    """
    if not excl_cols:
        return pd.Series([np.nan] * len(updated_df), dtype=float)

    cols_use = []
    for c in excl_cols:
        try:
            t = int(str(c).split("_")[1])
        except Exception:
            continue
        if t >= int(min_tpc):
            cols_use.append(c)

    if not cols_use:
        cols_use = list(excl_cols)

    mat = updated_df[cols_use].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    out = np.full((mat.shape[0],), np.nan, dtype=float)

    for i in range(mat.shape[0]):
        row = mat[i, :]
        row = row[np.isfinite(row)]
        if row.size == 0:
            continue

        row_sorted = np.sort(row)  # ascending
        k = min(int(worst_k), row_sorted.size)
        out[i] = float(np.mean(row_sorted[-k:]))

    return pd.Series(out, dtype=float)


def compute_is_short_kernel_by_mean(duration_series: pd.Series) -> Tuple[pd.Series, float]:
    """
    is_short_kernel = 1 if duration_proxy <= mean(duration_proxy), else 0
    """
    d = pd.to_numeric(duration_series, errors="coerce")
    mean_d = float(d.mean(skipna=True)) if d.notna().sum() > 0 else np.nan



    # mnet Dnet: 1.2
    # DnetDnet twitter: 0.45
    # R1netR1net twitter: 0.6
    # RnetDnet 0.45

    # keep your previous scaling behavior
    mean_d = mean_d * 0.45

    out = pd.Series([pd.NA] * len(d), dtype="Int64")
    ok = d.notna() & np.isfinite(mean_d)
    out.loc[ok] = (d.loc[ok] <= mean_d).astype(int).astype("Int64")
    return out, mean_d


# ============================================================
# MERGE HELPERS
# ============================================================
def attach_cluster_knees_and_exclusive(
    csv_path: Path,
    mapping_df: pd.DataFrame,
    knee_coloc_kernel_map: Dict[int, int],
    knee_excl_kernel_map: Dict[int, int],
    excl_wide: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, int, int]:
    """
    Merge cluster, add manual knee columns, and join exclusive per-TPC columns.

    Returns:
      merged_df
      unmatched_cluster_count
      excl_cols_added
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        out = df.copy()
        out["cluster"] = []
        out["knee_coloc_n"] = []
        out["exclusive_knee_n"] = []
        return out, 0, 0

    id_col = best_id_col(df)
    name_col = best_name_col(df)

    df = df.copy()

    if id_col is not None:
        df[id_col] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
        merged = df.merge(mapping_df, how="left", left_on=id_col, right_on="_id")
        unmatched = int(merged["cluster"].isna().sum())
        merged["cluster"] = merged["cluster"].fillna(-1).astype(int)

        merged["knee_coloc_n"] = merged[id_col].map(
            lambda x: knee_coloc_kernel_map.get(int(x), np.nan) if pd.notna(x) else np.nan
        ).astype("Int64")

        merged["exclusive_knee_n"] = merged[id_col].map(
            lambda x: knee_excl_kernel_map.get(int(x), np.nan) if pd.notna(x) else np.nan
        ).astype("Int64")

        excl_cols_added = 0
        if excl_wide is not None and not excl_wide.empty:
            before_cols = set(merged.columns)
            merged = merged.merge(excl_wide, how="left", left_on=id_col, right_index=True)
            new_cols = [c for c in merged.columns if c.startswith("excl_") and c not in before_cols]
            for c in new_cols:
                merged[c] = pd.to_numeric(merged[c], errors="coerce")
            excl_cols_added = len(new_cols)

        merged = merged.drop(columns=["_id"], errors="ignore")

        if "Name" not in merged.columns:
            if name_col is not None:
                merged = merged.rename(columns={name_col: "Name"})
            else:
                merged["Name"] = ""

        return merged, unmatched, excl_cols_added

    # fallback: row order
    df["_row_id"] = range(len(df))
    merged = df.merge(mapping_df, how="left", left_on="_row_id", right_on="_id")

    unmatched = int(merged["cluster"].isna().sum())
    merged["cluster"] = merged["cluster"].fillna(-1).astype(int)

    merged["knee_coloc_n"] = merged["_row_id"].map(
        lambda x: knee_coloc_kernel_map.get(int(x), np.nan)
    ).astype("Int64")

    merged["exclusive_knee_n"] = merged["_row_id"].map(
        lambda x: knee_excl_kernel_map.get(int(x), np.nan)
    ).astype("Int64")

    excl_cols_added = 0
    if excl_wide is not None and not excl_wide.empty:
        before_cols = set(merged.columns)
        merged = merged.merge(excl_wide, how="left", left_on="_row_id", right_index=True)
        new_cols = [c for c in merged.columns if c.startswith("excl_") and c not in before_cols]
        for c in new_cols:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")
        excl_cols_added = len(new_cols)

    merged = merged.drop(columns=["_row_id", "_id"], errors="ignore")

    if "Name" not in merged.columns:
        if name_col is not None:
            merged = merged.rename(columns={name_col: "Name"})
        else:
            merged["Name"] = ""

    return merged, unmatched, excl_cols_added


# ============================================================
# MAIN
# ============================================================
def main():
    if not CLUSTER_CSV.exists():
        raise FileNotFoundError(f"Cluster CSV not found: {CLUSTER_CSV}")
    if not MODEL_ROOT.exists():
        raise FileNotFoundError(f"MODEL_ROOT not found: {MODEL_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    files = discover_model_csvs(MODEL_ROOT)
    if not files:
        raise RuntimeError(f"No CSV-like files found directly under {MODEL_ROOT}")

    print(f"[INFO] MODEL_ROOT     = {MODEL_ROOT}")
    print(f"[INFO] OUTPUT_ROOT    = {OUTPUT_ROOT}")
    print(f"[INFO] PROFILING_DIR  = {PROFILING_DIR}")
    print(f"[INFO] KNEE_COLOC_DIR = {KNEE_COLOC_DIR}")
    print(f"[INFO] KNEE_EXCL_DIR  = {KNEE_EXCL_DIR}")
    print(f"[INFO] EXCL_ROOT      = {EXCL_ROOT}")
    print(f"[INFO] Short duration proxy: mean(worst-{WORST_K_FOR_SHORT}) over tpc >= {MIN_TPC_FOR_SHORT}")

    kern_all = pd.read_csv(CLUSTER_CSV)

    # load all exclusive CSVs
    excl_raw_by_key: Dict[str, pd.DataFrame] = {
        p.stem.lower(): pd.read_csv(p) for p in EXCL_ROOT.glob("*.csv")
    }

    mapping_cache: Dict[str, pd.DataFrame] = {}
    excl_wide_cache: Dict[str, pd.DataFrame] = {}

    total_files = total_rows = total_unmatched = 0

    for f in files:
        stem  = f.stem
        sheet = filestem_to_sheet(stem)
        model_key = sheet_to_model(sheet)

        print(f"\n[INFO] {stem} -> sheet '{sheet}', model '{model_key}'")

        # cluster mapping
        try:
            if model_key not in mapping_cache:
                mapping_cache[model_key] = build_id_to_cluster_map_for_model(kern_all, model_key)
            mapping_df = mapping_cache[model_key]
        except Exception as e:
            print(f"[WARN] Skipping {f.name}: cannot build cluster map for model '{model_key}'. Reason: {e}")
            continue

        # load manual knee overrides
        knee_coloc_csv = KNEE_COLOC_DIR / f"{stem}_est_coloc_knee_overrides.csv"
        knee_excl_csv  = KNEE_EXCL_DIR  / f"{stem}_knee_overrides.csv"

        coloc_filled = fill_knee_tpc_new_with_sat_tpc_auto_inplace(knee_coloc_csv)
        excl_filled  = fill_knee_tpc_new_with_sat_tpc_auto_inplace(knee_excl_csv)

        knee_coloc_kernel_map = load_manual_knee_map(knee_coloc_csv)
        knee_excl_kernel_map  = load_manual_knee_map(knee_excl_csv)

        print(f"[INFO] Manual knees loaded: coloc_kernels={len(knee_coloc_kernel_map)}, excl_kernels={len(knee_excl_kernel_map)}")
        print(f"[INFO] Manual knees filled: coloc_filled={coloc_filled}, excl_filled={excl_filled}")

        # load exclusive curve file
        excl_key = None
        if excl_raw_by_key:
            excl_key = choose_exclusive_key_for_sheet(sheet, set(excl_raw_by_key.keys()))

        if excl_key is None:
            print(f"[WARN] No exclusive CSV matched for sheet '{sheet}'. excl_* columns become NaN.")
            excl_wide = pd.DataFrame()
        else:
            excl_df = excl_raw_by_key[excl_key]
            if excl_key not in excl_wide_cache:
                excl_wide_cache[excl_key] = build_exclusive_wide_for_df(excl_df)
            excl_wide = excl_wide_cache[excl_key]

        # merge + output
        try:
            updated_df, unmatched, _ = attach_cluster_knees_and_exclusive(
                csv_path=f,
                mapping_df=mapping_df,
                knee_coloc_kernel_map=knee_coloc_kernel_map,
                knee_excl_kernel_map=knee_excl_kernel_map,
                excl_wide=excl_wide,
            )

            # Determine stable Kernel_ID for output
            id_col = best_id_col(updated_df)
            if id_col is not None:
                kid_series = pd.to_numeric(updated_df[id_col], errors="coerce")
                if kid_series.notna().sum() == 0:
                    kernel_ids = np.arange(len(updated_df), dtype=int)
                else:
                    kernel_ids = kid_series.fillna(-1).astype(int).to_numpy()
            else:
                kernel_ids = np.arange(len(updated_df), dtype=int)

            if "Name" not in updated_df.columns:
                name_col = best_name_col(updated_df)
                if name_col is None:
                    updated_df["Name"] = ""
                else:
                    updated_df = updated_df.rename(columns={name_col: "Name"})

            # find excl columns
            excl_cols = [c for c in updated_df.columns if c.startswith("excl_")]
            excl_cols = sorted(excl_cols, key=lambda x: int(x.split("_")[1]))

            # duration proxy for short classification
            dur_proxy = duration_proxy_worstk_from_excl(
                updated_df,
                excl_cols=excl_cols,
                min_tpc=MIN_TPC_FOR_SHORT,
                worst_k=WORST_K_FOR_SHORT,
            )
            is_short, mean_dur = compute_is_short_kernel_by_mean(dur_proxy)

            final_df = pd.DataFrame({
                "Kernel_ID": kernel_ids,
                "Name": updated_df["Name"],
                "cluster": updated_df["cluster"],
                "knee_coloc_n": updated_df["knee_coloc_n"],
                "exclusive_knee_n": updated_df["exclusive_knee_n"],
                SHORT_COL_NAME: is_short,
            })

            # attach excl_*
            for c in excl_cols:
                final_df[c] = updated_df[c]

            out_path = OUTPUT_ROOT / f.name
            final_df.to_csv(out_path, index=False)

            total_files += 1
            total_rows  += len(final_df)
            total_unmatched += unmatched

            matched = len(final_df) - unmatched
            short_cnt = int(pd.to_numeric(final_df[SHORT_COL_NAME], errors="coerce").fillna(0).sum())
            coloc_non_nan = int(final_df["knee_coloc_n"].notna().sum())
            excl_non_nan  = int(final_df["exclusive_knee_n"].notna().sum())

            print(f"[OK] Wrote {out_path.name} | matched={matched}, unmatched={unmatched}")
            print(f"     short flag: mean_duration_proxy={mean_dur:.4f} us | short_kernels={short_cnt}/{len(final_df)}")
            print(f"     manual knees: coloc_non_nan={coloc_non_nan}/{len(final_df)}, excl_non_nan={excl_non_nan}/{len(final_df)}")
            print(f"     excl_cols={len(excl_cols)}")

        except Exception as e:
            print(f"[ERR] Failed on {f.name}: {e}")

    print("\n[DONE]")
    print(f"Files processed: {total_files}")
    print(f"Total rows:      {total_rows}")
    print(f"Total unmatched: {total_unmatched} (cluster=-1)")
    print(f"Outputs in:      {OUTPUT_ROOT}")


if __name__ == "__main__":
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 200)
    main()
