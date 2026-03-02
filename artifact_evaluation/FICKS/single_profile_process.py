#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import re

# ============================================================
# CONFIG (no argparse, edit here)
# ============================================================
KERNEL_LOGS_DIR = Path("kernel_logs")                # input logs live here
OUT_DIR         = Path("kernel_profiles_from_logs")  # output profile csvs here
TPC_TOTAL       = 24                                 # A4000 = 24

# If your logs contain warmup iters and you want to drop them
DROP_ITERS_LT   = None   # e.g., 10 to drop iter < 10; set None to keep all

# Clean output dir first (prevents stale files)
CLEAR_OUT_DIR   = True

# You asked: average across logs for the same model
AGG_METHOD      = "mean"  # fixed to mean

# If model_name in logs does NOT contain _bzN, we append default bz here
# (also used to normalize mobilenetv2 -> mobilenet_v2_bz32 output naming)
DEFAULT_BZ_BY_BASE = {
    "densenet201":   8,
    "resnet101":     8,
    "resnet152":     8,
    "vgg19":         8,
    "mobilenet_v2":  32,
    "mobilenetv2":   32,  # accept both spellings from logs
}

# Canonical output base name (so mobilenetv2 becomes mobilenet_v2)
CANONICAL_BASE = {
    "mobilenetv2":  "mobilenet_v2",
    "mobilenet_v2": "mobilenet_v2",
}

# ============================================================
# Helpers
# ============================================================
def ensure_out_dir():
    if CLEAR_OUT_DIR and OUT_DIR.exists():
        if OUT_DIR.name not in ("kernel_profiles_from_logs", "kernel_profiles"):
            raise RuntimeError(f"Refusing to delete suspicious OUT_DIR: {OUT_DIR}")
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def sanitize_filename(name: str) -> str:
    """
    Keep your model naming style like:
      densenet201_bz8.csv, mobilenet_v2_bz32.csv, ...
    Only replace unsafe filesystem chars.
    """
    s = str(name).strip()
    if not s:
        return "unknown_model"
    s = re.sub(r"[^A-Za-z0-9_.]+", "_", s).strip("_")
    return s or "unknown_model"

def strip_bz_suffix(model: str) -> str:
    """'vgg19_bz8' -> 'vgg19' (only strips trailing _bz<digits>)."""
    return re.sub(r"_bz\d+$", "", str(model).strip())

def extract_bz_suffix(model: str):
    """Return int batch size if model ends with _bzN, else None."""
    m = re.search(r"_bz(\d+)$", str(model).strip())
    return int(m.group(1)) if m else None

def canonical_model_with_bz(model_name: str) -> str:
    """
    Produce output model name that always includes _bzN.
    Also canonicalizes mobilenetv2 -> mobilenet_v2.
    """
    raw = str(model_name).strip()
    if not raw:
        return "unknown_model_bz0"

    bz = extract_bz_suffix(raw)
    base = strip_bz_suffix(raw)

    base = CANONICAL_BASE.get(base, base)

    if bz is None:
        bz = DEFAULT_BZ_BY_BASE.get(base)
        if bz is None:
            # unknown model: keep base without adding bz
            # but user asked to keep bz postfix; fallback to bz0
            bz = 0

    return f"{base}_bz{bz}"

def pick_kernel_name(series: pd.Series) -> str:
    """
    Pick a stable kernel name for a (model, kernel_id).
    Prefer most frequent non-empty; else "".
    """
    s = series.astype(str).map(lambda x: x.strip())
    s = s[s != ""]
    if len(s) == 0:
        return ""
    vc = s.value_counts()
    return str(vc.index[0])

def build_model_profile(df_model: pd.DataFrame) -> pd.DataFrame:
    """
    df_model has columns: kernel_id, tpc_used, duration_us, kernel_name
    already for a single model_out and already aggregated to mean for (kernel_id, tpc_used).
    """
    name_by_kid = (
        df_model.groupby("kernel_id")["kernel_name"]
                .apply(pick_kernel_name)
                .to_dict()
    )

    pivot = (
        df_model.pivot(index="kernel_id", columns="tpc_used", values="duration_us")
                .sort_index()
    )

    cols = [str(i) for i in range(1, TPC_TOTAL + 1)]

    prof = pd.DataFrame({
        "id": pivot.index.astype(int),
        "kernel_name": [name_by_kid.get(int(k), "") for k in pivot.index.astype(int)],
    })

    for i in range(1, TPC_TOTAL + 1):
        if i in pivot.columns:
            prof[str(i)] = pivot[i].astype(float).values
        else:
            prof[str(i)] = np.nan

    prof = prof[["id", "kernel_name"] + cols].reset_index(drop=True)
    return prof

def main():
    if not KERNEL_LOGS_DIR.exists():
        raise FileNotFoundError(f"kernel logs dir not found: {KERNEL_LOGS_DIR.resolve()}")

    ensure_out_dir()

    csvs = sorted(KERNEL_LOGS_DIR.rglob("*.csv"))
    if not csvs:
        raise RuntimeError(f"No CSV files found under: {KERNEL_LOGS_DIR.resolve()}")

    print(f"[INFO] Found {len(csvs)} log csv files under {KERNEL_LOGS_DIR.resolve()}")
    print(f"[INFO] Output dir: {OUT_DIR.resolve()} (TPC_TOTAL={TPC_TOTAL}, AGG={AGG_METHOD})")

    all_rows = []
    bad = 0

    for p in csvs:
        try:
            df = pd.read_csv(p)

            # expected header:
            # client_id,iter,kernel_id,event_id,tpc_used,kernel_name,model_name,start_us,end_us,duration_us
            required = ["kernel_id", "tpc_used", "duration_us", "model_name"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise KeyError(f"missing columns {missing}; have {df.columns.tolist()}")

            if DROP_ITERS_LT is not None and "iter" in df.columns:
                df = df[df["iter"] >= int(DROP_ITERS_LT)].copy()

            df["model_name"]  = df["model_name"].astype(str).map(lambda x: x.strip())
            df["model_out"]   = df["model_name"].map(canonical_model_with_bz)

            if "kernel_name" in df.columns:
                df["kernel_name"] = df["kernel_name"].astype(str)
            else:
                df["kernel_name"] = ""

            df["kernel_id"]   = pd.to_numeric(df["kernel_id"], errors="coerce")
            df["tpc_used"]    = pd.to_numeric(df["tpc_used"], errors="coerce")
            df["duration_us"] = pd.to_numeric(df["duration_us"], errors="coerce")

            df = df.dropna(subset=["model_out", "kernel_id", "tpc_used", "duration_us"]).copy()
            df["kernel_id"]   = df["kernel_id"].astype(int)
            df["tpc_used"]    = df["tpc_used"].astype(int)
            df["duration_us"] = df["duration_us"].astype(float)

            df = df[(df["tpc_used"] >= 1) & (df["tpc_used"] <= TPC_TOTAL)].copy()

            if df.empty:
                print(f"[WARN] {p}: no valid rows after filtering")
                continue

            all_rows.append(df[["model_out", "kernel_id", "tpc_used", "duration_us", "kernel_name"]])
            print(f"[OK] loaded {p}  rows={len(df)} -> models={df['model_out'].nunique()}")
        except Exception as e:
            bad += 1
            print(f"[ERR] {p}: {e}")

    if not all_rows:
        raise RuntimeError("No usable log rows loaded (all files failed or empty).")

    df_all = pd.concat(all_rows, ignore_index=True)

    # ---- Average across ALL logs for SAME model_out ----
    dur_agg = (
        df_all.groupby(["model_out", "kernel_id", "tpc_used"], as_index=False)["duration_us"]
              .mean()
    )

    # kernel_name: most frequent per (model_out, kernel_id)
    name_map = (
        df_all.groupby(["model_out", "kernel_id"])["kernel_name"]
              .apply(pick_kernel_name)
              .reset_index()
    )

    dur_agg = dur_agg.merge(name_map, on=["model_out", "kernel_id"], how="left")
    dur_agg["kernel_name"] = dur_agg["kernel_name"].fillna("")

    models = sorted([m for m in dur_agg["model_out"].unique() if str(m).strip() != ""])
    print(f"[INFO] Models discovered: {len(models)} -> {models[:8]}{' ...' if len(models) > 8 else ''}")

    ok_models = 0
    for m in models:
        df_m = dur_agg[dur_agg["model_out"] == m].copy()
        if df_m.empty:
            continue

        prof = build_model_profile(df_m)

        # output exactly: densenet201_bz8.csv, mobilenet_v2_bz32.csv, ...
        out_name = sanitize_filename(m) + ".csv"
        out_path = OUT_DIR / out_name

        prof.to_csv(out_path, index=False)
        print(f"[WRITE] {out_path}  (model={m}, rows={len(prof)})")
        ok_models += 1

    print(f"[DONE] wrote {ok_models} model profile csv files. (bad_input_files={bad})")

if __name__ == "__main__":
    main()
