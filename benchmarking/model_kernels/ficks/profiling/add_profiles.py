#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
For each model CSV under MODEL_ROOT (Orion directory), produce a new CSV under OUTPUT_ROOT
with columns:

    Kernel_ID, Name, cluster, knee_coloc_n, exclusive_knee_n, excl_1, excl_2, ..., excl_N

where:
  - Kernel_ID: 0..N-1 (row index in the output)
  - Name: kernel name (from the Orion CSV)
  - cluster: UT cluster (from A4000_kernels_profiles_all_models.csv)
  - knee_coloc_n: universal co-location knee per cluster, per model (from PLAN_CSV)
  - exclusive_knee_n: per-kernel exclusive knee (from exclusive profile CSVs)
  - excl_k: exclusive latency at TPC = k in microseconds (from exclusive profile CSVs)

All other original columns from Orion are discarded in the final output.
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ---------- Paths ----------
MODEL_ROOT   = Path("/home/zixi/orion_bu/benchmarking/model_kernels/orion")
OUTPUT_ROOT  = Path("/home/zixi/orion_bu/benchmarking/model_kernels/ficks")

CLUSTER_CSV = Path("/home/zixi/orion_bu/benchmarking/model_kernels/ficks/profiling/A4000_kernels_profiles_all_models.csv")
PLAN_CSV    = Path("/home/zixi/orion_bu/benchmarking/model_kernels/ficks/profiling/A4000_profile_plan_all_cluster_pairs_up_to3.csv")

EXCL_ROOT = Path("/home/zixi/orion_bu/artifact_evaluation/FICKS/new_A4000_profiles")
KNEE_PARAMS_JSON = Path("/home/zixi/orion_bu/benchmarking/model_kernels/ficks/profiling/knee_params.json")

# ---------- Regex ----------
CONFIG_RE  = re.compile(r"^\(\s*(\d+)\s*,\s*(\d+)\s*\)$")  # "(n,m)" for plan TPC cols
TPC_COL_RE = re.compile(r"^\s*(\d{1,3})\s*$")              # "1","2",...

# ---------- Utils ----------
def discover_model_csvs(root: Path):
    files = []
    for p in sorted(root.iterdir()):
        if p.is_file() and (p.suffix.lower() == ".csv" or p.suffix == ""):
            if re.search(r"_\d+_fwd(\.csv)?$", p.name):
                files.append(p)
    return files

def filestem_to_sheet(stem: str) -> str:
    """Historically used as sheet name; still used as key in knee_params.json."""
    m = re.match(r"^(?P<base>.+)_(?P<bz>\d+)_fwd$", stem)
    if m:
        return f"{m.group('base')}_bz{m.group('bz')}"
    m2 = re.search(r"_(\d+)$", stem)
    if m2:
        base = stem[: m2.start()]
        return f"{base}_bz{m2.group(1)}"
    return stem

def sheet_to_model(sheet: str) -> str:
    """Convert e.g. 'resnet152_bz8' -> 'resnet152'."""
    return re.sub(r"_bz\d+$", "", sheet)

# ---------- Column detection ----------
def best_id_col(df: pd.DataFrame) -> Optional[str]:
    """
    Try to find a kernel ID column.
    Recognizes generic ID names and 'Kernel Index' variants.
    """
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")
    norm_to_orig = {norm(c): c for c in df.columns}

    kernel_id_candidates = ["kernel_index", "kernel_index_ut", "kernel_id", "kernelid", "kernel"]
    generic_id_candidates = ["id", "idx", "index"]

    for cand in kernel_id_candidates + generic_id_candidates:
        if cand in norm_to_orig:
            return norm_to_orig[cand]

    # Heuristic: unnamed integer column exported from spreadsheets
    for n, o in norm_to_orig.items():
        if n.startswith("unnamed") and df[o].dtype.kind in "iu":
            return o
    return None

# ---------- Cluster mapping (kernel ID -> cluster) ----------
def build_id_to_cluster_map_for_model(kern_all: pd.DataFrame, model_key: str) -> pd.DataFrame:
    """Return mapping DataFrame with columns ['_id','cluster'] for a given model name."""
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

# ---------- Knee helpers ----------
def row_knee_derivative_raw(n_list, l_list, tau_abs: float, tau_frac: float, win: int):
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
    tau = max(tau_abs, tau_frac * baseline)

    if gains.size >= win:
        win_means = np.convolve(gains, np.ones(win)/win, mode="valid")
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

def coverage_knee(ns, coverage_p: float):
    arr = np.array([x for x in ns if pd.notna(x)], dtype=float)
    if arr.size == 0:
        return np.nan
    for cand in sorted(set(arr)):
        if (arr <= cand).mean() >= coverage_p:
            return int(cand)
    return int(np.nanmax(arr))

# ---------- Co-location knee per UT cluster (per model) ----------
def find_cfg_triples(df: pd.DataFrame):
    triples = []
    for c in df.columns:
        if isinstance(c, str):
            m = CONFIG_RE.match(c)
            if m:
                triples.append((c, int(m.group(1)), int(m.group(2))))
    triples.sort(key=lambda x: x[1])
    return triples

def build_cluster_to_coloc_knee_map(plan_df: pd.DataFrame,
                                    model_key: str,
                                    tau_abs: float, tau_frac: float, win: int, coverage_p: float) -> Dict[int, int]:
    """Compute a universal co-location knee per UT cluster for a given model (Model_UT == model_key)."""
    df = plan_df[plan_df["Model_UT"] == model_key].copy()
    if df.empty:
        print(f"[WARN] PLAN has no rows with Model_UT == '{model_key}'.")
        return {}
    if "cluster_UT" not in df.columns:
        print(f"[WARN] PLAN CSV lacks 'cluster_UT' column.")
        return {}

    triples = find_cfg_triples(df)
    if not triples:
        print(f"[WARN] PLAN has no '(n,m)' TPC columns.")
        return {}

    cfg_cols = [c for c, _, _ in triples]
    ns       = [n for _, n, _ in triples]
    work = df.copy()
    work[cfg_cols] = work[cfg_cols].apply(pd.to_numeric, errors="coerce")

    row_knees = []
    for _, row in work.iterrows():
        vals = [row[c] for c in cfg_cols]
        k_n = row_knee_derivative_raw(ns, vals, tau_abs=tau_abs, tau_frac=tau_frac, win=win)
        row_knees.append((row.get("cluster_UT"), k_n))
    kdf = pd.DataFrame(row_knees, columns=["cluster_UT", "knee_n"])

    out: Dict[int, int] = {}
    for c_ut, sub in kdf.groupby("cluster_UT", dropna=False):
        n_vals = sub["knee_n"].dropna().astype(float).values
        if n_vals.size == 0:
            continue
        cov_n = coverage_knee(n_vals, coverage_p=coverage_p)
        if pd.notna(cov_n):
            try:
                out[int(c_ut)] = int(cov_n)
            except Exception:
                out[c_ut] = int(cov_n)
    return out

# ---------- Exclusive CSV helpers ----------
def detect_tpc_columns(excl_df: pd.DataFrame):
    tpcs = []
    for c in excl_df.columns:
        if isinstance(c, str):
            m = TPC_COL_RE.match(c)
            if m:
                tpcs.append((int(m.group(1)), c))
    tpcs.sort(key=lambda x: x[0])
    return tpcs

def build_exclusive_knee_map_for_df(excl_df: pd.DataFrame,
                                    tau_abs: float, tau_frac: float, win: int) -> Dict[int, int]:
    """
    Build a map: kernel_id -> exclusive knee TPC (n).
    Works with ID columns named 'Kernel Index', 'id', 'kernel_id', etc.
    """
    if excl_df is None or excl_df.empty:
        return {}

    id_col = best_id_col(excl_df)
    if id_col is None:
        print("[WARN] Exclusive CSV has no ID-like column (expected 'Kernel Index', 'id', ...).")
        return {}

    tpcs = detect_tpc_columns(excl_df)
    if not tpcs:
        return {}

    n_vals = [n for n, _ in tpcs]       # e.g. [1, 2, ..., 24]
    cols   = [c for _, c in tpcs]       # e.g. ['1','2',...,'24']

    df = excl_df.copy()
    df[id_col] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    df = df.dropna(subset=[id_col])
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce") / 1000.0  # ns -> µs

    knee_map: Dict[int, int] = {}
    for _, row in df.iterrows():
        kid = row[id_col]
        if pd.isna(kid):
            continue
        kid = int(kid)
        L = [row[c] for c in cols]
        k_n = row_knee_derivative_raw(n_vals, L, tau_abs=tau_abs, tau_frac=tau_frac, win=win)
        if pd.notna(k_n):
            knee_map[kid] = int(k_n)

    return knee_map

def choose_exclusive_key_for_sheet(sheet: str, available_keys):
    s = sheet.lower()
    base = re.sub(r"_bz\d+$", "", s)
    candidates = [s, s.replace(" ", "_"), base, base + "_8_fwd", base + "_32_fwd", base + "_fwd"]
    for cand in candidates:
        if cand in available_keys:
            return cand
    starts = [k for k in available_keys if k.startswith(base)]
    return sorted(starts)[0] if starts else None

def build_exclusive_wide_for_df(excl_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a wide df indexed by the detected ID column
    with columns 'excl_<n>' in µs.
    Handles exclusive CSVs like:

        id,1,2,...,24
        0,448052.0,...

    """
    if excl_df is None or excl_df.empty:
        return pd.DataFrame()

    id_col = best_id_col(excl_df)
    if id_col is None:
        print("[WARN] Exclusive CSV has no ID-like column (expected 'Kernel Index', 'id', ...).")
        return pd.DataFrame()

    tpcs = detect_tpc_columns(excl_df)
    if not tpcs:
        return pd.DataFrame()

    n_vals = [n for n, _ in tpcs]   # e.g. [1,2,...,24]
    cols   = [c for _, c in tpcs]   # ['1','2',...,'24']

    df = excl_df[[id_col] + cols].copy()

    df[id_col] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    df = df.dropna(subset=[id_col])
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce") / 1000.0  # ns -> µs

    rename_map = {c: f"excl_{n}" for c, n in zip(cols, n_vals)}
    df = df.rename(columns=rename_map)

    df = df.drop_duplicates(subset=[id_col]).set_index(id_col)
    df = df.reindex(sorted(df.columns, key=lambda x: int(x.split("_")[1])), axis=1)

    return df

# ---------- Merge helpers ----------
def attach_cluster_and_knees_and_exclusive(csv_path: Path,
                                           mapping_df: pd.DataFrame,
                                           knee_map_coloc: Dict[int, int],
                                           id_to_excl_knee: Dict[int, int],
                                           excl_wide: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, int, int]:
    """
    Merge cluster (via ID or row-order), add knee_coloc_n (via cluster),
    add exclusive_knee_n (via ID or row-order), and join exclusive per-TPC columns (via ID or row-order).
    Returns: (merged_df, unmatched_cluster_rows, num_excl_cols_added)
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        out = df.copy()
        for c in ("cluster","knee_coloc_n","exclusive_knee_n"):
            out[c] = []
        return out, 0, 0

    id_col = best_id_col(df)

    # -------- Branch A: we DO have a kernel-id column --------
    if id_col is not None:
        df = df.copy()
        df[id_col] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
        merged = df.merge(mapping_df, how="left", left_on=id_col, right_on="_id")
        unmatched = int(merged["cluster"].isna().sum())
        merged["cluster"] = merged["cluster"].fillna(-1).astype(int)

        # coloc knee by cluster
        if knee_map_coloc:
            merged["knee_coloc_n"] = merged["cluster"].map(
                lambda c: knee_map_coloc.get(int(c), np.nan)
            ).astype("Int64")
        else:
            merged["knee_coloc_n"] = pd.Series([np.nan] * len(merged), dtype="Int64")

        # exclusive knee by this ID column
        if id_to_excl_knee:
            merged["exclusive_knee_n"] = merged[id_col].map(
                lambda x: id_to_excl_knee.get(int(x), np.nan) if pd.notna(x) else np.nan
            ).astype("Int64")
        else:
            merged["exclusive_knee_n"] = pd.Series([np.nan] * len(merged), dtype="Int64")

        # exclusive per-TPC columns via wide join (index is ID column)
        excl_cols_added = 0
        if excl_wide is not None and not excl_wide.empty:
            before_cols = set(merged.columns)
            merged = merged.merge(excl_wide, how="left", left_on=id_col, right_index=True)
            new_cols = [c for c in merged.columns if c.startswith("excl_") and c not in before_cols]
            for c in new_cols:
                merged[c] = pd.to_numeric(merged[c], errors="coerce")
            excl_cols_added = len(new_cols)

        merged = merged.drop(columns=["_id"], errors="ignore")
        return merged, unmatched, excl_cols_added

    # -------- Branch B: NO ID column → use row-order fallback --------
    df = df.copy()
    df["_row_id"] = range(len(df))
    merged = df.merge(mapping_df, how="left", left_on="_row_id", right_on="_id")

    unmatched = int(merged["cluster"].isna().sum())
    merged["cluster"] = merged["cluster"].fillna(-1).astype(int)

    # coloc knee by cluster
    if knee_map_coloc:
        merged["knee_coloc_n"] = merged["cluster"].map(
            lambda c: knee_map_coloc.get(int(c), np.nan)
        ).astype("Int64")
    else:
        merged["knee_coloc_n"] = pd.Series([np.nan] * len(merged), dtype="Int64")

    # exclusive knee via ROW-ORDER mapping (assumes Kernel Index == row_id)
    if id_to_excl_knee:
        merged["exclusive_knee_n"] = merged["_row_id"].map(
            lambda x: id_to_excl_knee.get(int(x), np.nan)
        ).astype("Int64")
    else:
        merged["exclusive_knee_n"] = pd.Series([np.nan] * len(merged), dtype="Int64")

    # per-TPC exclusive columns via row-order join to excl_wide (index is Kernel Index)
    excl_cols_added = 0
    if excl_wide is not None and not excl_wide.empty:
        before_cols = set(merged.columns)
        merged = merged.merge(excl_wide, how="left", left_on="_row_id", right_index=True)
        new_cols = [c for c in merged.columns if c.startswith("excl_") and c not in before_cols]
        for c in new_cols:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")
        excl_cols_added = len(new_cols)
        print(f"[INFO] {csv_path.name}: used row-order fallback for exclusive join (added {excl_cols_added} excl_* cols).")

    merged = merged.drop(columns=["_row_id", "_id"], errors="ignore")
    return merged, unmatched, excl_cols_added

# ---------- Params loading ----------
def load_knee_params(json_path: Path):
    if not json_path.exists():
        raise FileNotFoundError(f"knee params JSON not found: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    defaults = data.get("defaults", {})
    models   = data.get("models", {})
    return defaults, models

def params_for_model(model_sheet: str, defaults: dict, models: dict, which: str):
    """
    which ∈ {'coloc','exclusive'}
    """
    m = models.get(model_sheet, {})
    p = m.get(which, {})
    defval = defaults.get(which, {})
    return {
        "tau_abs":    float(p.get("tau_abs",    defval.get("tau_abs",    0.01))),
        "tau_frac":   float(p.get("tau_frac",   defval.get("tau_frac",   0.10))),
        "win":        int(  p.get("win",        defval.get("win",        2))),
        "coverage_p": float(p.get("coverage_p", defval.get("coverage_p", 0.80))),
    }

# ---------- Main ----------
def main():
    if not CLUSTER_CSV.exists():
        raise FileNotFoundError(f"Cluster CSV not found: {CLUSTER_CSV}")
    if not PLAN_CSV.exists():
        raise FileNotFoundError(f"Plan CSV not found: {PLAN_CSV}")

    print(f"[INFO] Cluster CSV: {CLUSTER_CSV}")
    print(f"[INFO] Plan CSV   : {PLAN_CSV}")
    print(f"[INFO] Exclusive root  : {EXCL_ROOT}")
    print(f"[INFO] Knee params JSON: {KNEE_PARAMS_JSON}")

    defaults, models = load_knee_params(KNEE_PARAMS_JSON)

    if not MODEL_ROOT.exists():
        raise FileNotFoundError(f"MODEL_ROOT not found: {MODEL_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    files = discover_model_csvs(MODEL_ROOT)
    if not files:
        raise RuntimeError(f"No CSV-like files found directly under {MODEL_ROOT}")

    # Load cluster & plan CSVs once
    kern_all = pd.read_csv(CLUSTER_CSV)
    plan_df  = pd.read_csv(PLAN_CSV)

    # Load exclusive CSVs (raw) once
    excl_raw_by_key: Dict[str, pd.DataFrame] = {p.stem.lower(): pd.read_csv(p) for p in EXCL_ROOT.glob("*.csv")}
    excl_knee_cache: Dict[Tuple[str, float, float, int], Dict[int, int]] = {}
    excl_wide_cache: Dict[str, pd.DataFrame] = {}

    mapping_cache: Dict[str, pd.DataFrame]    = {}  # keyed by model_key, e.g., 'resnet152'
    coloc_knee_cache: Dict[Tuple[str, float, float, int, float], Dict[int, int]] = {}  # keyed by (model_key, ...)

    total_files = total_rows = total_unmatched = 0

    for f in files:
        stem  = f.stem
        sheet = filestem_to_sheet(stem)     # e.g., "resnet152_bz8"
        model_key = sheet_to_model(sheet)   # e.g., "resnet152"
        print(f"[INFO] {f.name} -> sheet '{sheet}', model '{model_key}'")

        # per-model params
        coloc_p = params_for_model(sheet, defaults, models, "coloc")
        excl_p  = params_for_model(sheet, defaults, models, "exclusive")

        # cluster mapping for this model
        try:
            if model_key not in mapping_cache:
                mapping_cache[model_key] = build_id_to_cluster_map_for_model(kern_all, model_key)
            mapping_df = mapping_cache[model_key]
        except Exception as e:
            print(f"[WARN] Skipping {f.name}: cannot build cluster map for model '{model_key}'. Reason: {e}")
            continue

        # coloc universal knee per UT cluster (per-model)
        ck_key = (model_key, coloc_p["tau_abs"], coloc_p["tau_frac"], coloc_p["win"], coloc_p["coverage_p"])
        if ck_key not in coloc_knee_cache:
            coloc_knee_cache[ck_key] = build_cluster_to_coloc_knee_map(
                plan_df, model_key,
                tau_abs=coloc_p["tau_abs"], tau_frac=coloc_p["tau_frac"],
                win=coloc_p["win"], coverage_p=coloc_p["coverage_p"]
            )
        knee_map_coloc = coloc_knee_cache[ck_key]

        # choose exclusive file key for this sheet
        excl_key = None
        if excl_raw_by_key:
            excl_key = choose_exclusive_key_for_sheet(sheet, set(excl_raw_by_key.keys()))

        if excl_key is None:
            id_to_excl_knee = {}
            excl_wide = pd.DataFrame()
            print(f"[WARN] No exclusive CSV matched for sheet '{sheet}'.")
        else:
            # knees cache
            ek_key = (excl_key, excl_p["tau_abs"], excl_p["tau_frac"], excl_p["win"])
            if ek_key not in excl_knee_cache:
                id_to_excl_knee = build_exclusive_knee_map_for_df(
                    excl_raw_by_key[excl_key],
                    tau_abs=excl_p["tau_abs"], tau_frac=excl_p["tau_frac"], win=excl_p["win"]
                )
                excl_knee_cache[ek_key] = id_to_excl_knee
            else:
                id_to_excl_knee = excl_knee_cache[ek_key]
            # wide curves cache
            if excl_key not in excl_wide_cache:
                excl_wide_cache[excl_key] = build_exclusive_wide_for_df(excl_raw_by_key[excl_key])
            excl_wide = excl_wide_cache[excl_key]

        # merge & write
        try:
            updated_df, unmatched, excl_cols_added = attach_cluster_and_knees_and_exclusive(
                f, mapping_df, knee_map_coloc, id_to_excl_knee, excl_wide
            )

            # ---------- Rebuild final dataframe ----------
            # Kernel_ID: 0 .. N-1
            kernel_ids = np.arange(len(updated_df), dtype=int)

            # Name: must exist in Orion CSV
            if "Name" not in updated_df.columns:
                raise KeyError(f"'Name' column not found in {f.name}")

            # ensure these columns exist (fill NaN if missing)
            for c in ["cluster", "knee_coloc_n", "exclusive_knee_n"]:
                if c not in updated_df.columns:
                    updated_df[c] = np.nan

            # excl_* columns, sorted by numeric suffix
            excl_cols = [c for c in updated_df.columns if c.startswith("excl_")]
            excl_cols = sorted(excl_cols, key=lambda x: int(x.split("_")[1]))

            final_df = pd.DataFrame({
                "Kernel_ID": kernel_ids,
                "Name": updated_df["Name"],
                "cluster": updated_df["cluster"],
                "knee_coloc_n": updated_df["knee_coloc_n"],
                "exclusive_knee_n": updated_df["exclusive_knee_n"],
            })

            # append excl_* columns
            for c in excl_cols:
                final_df[c] = updated_df[c]

            out_path = OUTPUT_ROOT / f.name
            final_df.to_csv(out_path, index=False)

            total_files += 1
            total_rows  += len(final_df)
            total_unmatched += unmatched
            matched = len(final_df) - unmatched
            matched_excl = int(final_df["exclusive_knee_n"].notna().sum())
            print(f"[OK] Wrote {out_path.name} | matched={matched}, unmatched={unmatched} | "
                  f"clusters_with_coloc_knee={len(knee_map_coloc)} | excl_knees={len(id_to_excl_knee)} | "
                  f"exclusive_knee_filled={matched_excl}/{len(final_df)} | excl_cols={len(excl_cols)}")
        except Exception as e:
            print(f"[ERR] Failed on {f.name}: {e}")

    print("\n[DONE]")
    print(f"Files processed: {total_files}")
    print(f"Total rows:      {total_rows}")
    print(f"Total unmatched: {total_unmatched} (cluster=-1, knees may be NaN)")
    print(f"Outputs in:      {OUTPUT_ROOT}")

if __name__ == "__main__":
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 200)
    main()
