#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import math
import pandas as pd


# =========================
# CONFIG
# =========================
PLAN_CSV = Path("/home/zixi/orion_bu/benchmarking/A4000_profile_plan_all_cluster_pairs_up_to3.csv")

# Directory containing single model exclusive curves, e.g.:
# densenet201_bz8.csv, mobilenet_v2_bz32.csv, resnet101_bz8.csv, ...
EXCL_DIR = Path("/home/zixi/orion_bu/artifact_evaluation/FICKS/A4000_TPC_profiles_cudnn_based")

OUT_LUT_CSV = Path("A4000_clusterpair_makespan_mult_lut.csv")
OUT_GLOBAL_CSV = Path("A4000_global_makespan_mult_by_split.csv")

TOTAL_TPCS = 24
NO_MASK_SPLIT = "(-1,-1)"
NO_MASK_BASE_TPC = 24

# Clamp multipliers to reduce outlier impact
CLAMP_MIN = 0.25
CLAMP_MAX = 8.0


# =========================
# HELPERS
# =========================
def strip_bz(s: str) -> str:
    return re.sub(r"_bz\d+$", "", str(s))

def normalize_model_name(s: str) -> str:
    """
    Make plan names and excl filenames meet.
    If you sometimes use mobilenetv2 vs mobilenet_v2, unify here.
    """
    s = str(s)
    # common normalization: treat both spellings as the same
    if s == "mobilenetv2":
        return "mobilenet_v2"
    return s

def canon_split_col(col: str) -> str:
    """
    Canonicalize "( 1, 23 )" -> "(1,23)" including negatives.
    """
    m = re.match(r"^\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)$", str(col))
    if not m:
        return col
    a = int(m.group(1))
    b = int(m.group(2))
    return f"({a},{b})"

def parse_split(col: str) -> tuple[int, int]:
    m = re.match(r"^\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)$", col)
    if not m:
        raise ValueError(f"Not a split column: {col}")
    return int(m.group(1)), int(m.group(2))

def safe_log(x: float) -> float:
    # assume x > 0; caller checks
    return math.log(x)

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


# =========================
# LOAD EXCLUSIVE CURVES
# =========================
def load_exclusive_curves(excl_dir: Path) -> dict[tuple[str, int], list[float]]:
    """
    Returns dict: (model_base, kernel_id) -> curve[0..24], where curve[t] is latency at TPC=t.
    curve[0] is unused (set to NaN) so indices match TPC count.
    """
    curves: dict[tuple[str, int], list[float]] = {}

    files = sorted(excl_dir.glob("*.csv"))
    if not files:
        raise RuntimeError(f"No exclusive CSVs found under {excl_dir.resolve()}")

    for fp in files:
        # e.g. resnet152_bz8.csv -> resnet152
        model = strip_bz(fp.stem)
        model = normalize_model_name(model)

        df = pd.read_csv(fp)
        if "id" not in df.columns:
            raise ValueError(f"{fp} missing 'id' column")

        # Expect columns "1".."24"
        needed = [str(i) for i in range(1, TOTAL_TPCS + 1)]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"{fp} missing TPC columns: {missing}")

        for _, r in df.iterrows():
            kid = int(r["id"])
            curve = [float("nan")] * (TOTAL_TPCS + 1)
            for t in range(1, TOTAL_TPCS + 1):
                v = r[str(t)]
                curve[t] = float(v) if pd.notna(v) else float("nan")
            curves[(model, kid)] = curve

        print(f"[EXCL] loaded {fp.name}: {len(df)} kernels")

    print(f"[EXCL] total curves: {len(curves)}")
    return curves


# =========================
# BUILD LUT
# =========================
def build_lut(plan_csv: Path, curves: dict[tuple[str, int], list[float]]):
    df = pd.read_csv(plan_csv)
    df.columns = [canon_split_col(c) for c in df.columns]

    req = ["Model_UT", "Kernel_ID_UT", "Model_CO", "Kernel_ID_CO", "cluster_UT", "cluster_CO"]
    for c in req:
        if c not in df.columns:
            raise KeyError(f"Plan missing required column: {c}")

    # normalize keys
    df["Model_UT"] = df["Model_UT"].astype(str).map(strip_bz).map(normalize_model_name)
    df["Model_CO"] = df["Model_CO"].astype(str).map(strip_bz).map(normalize_model_name)
    df["Kernel_ID_UT"] = pd.to_numeric(df["Kernel_ID_UT"], errors="coerce")
    df["Kernel_ID_CO"] = pd.to_numeric(df["Kernel_ID_CO"], errors="coerce")
    df["cluster_UT"] = pd.to_numeric(df["cluster_UT"], errors="coerce")
    df["cluster_CO"] = pd.to_numeric(df["cluster_CO"], errors="coerce")
    df = df.dropna(subset=["Kernel_ID_UT", "Kernel_ID_CO", "cluster_UT", "cluster_CO"]).copy()
    df["Kernel_ID_UT"] = df["Kernel_ID_UT"].astype(int)
    df["Kernel_ID_CO"] = df["Kernel_ID_CO"].astype(int)
    df["cluster_UT"] = df["cluster_UT"].astype(int)
    df["cluster_CO"] = df["cluster_CO"].astype(int)

    # discover split columns
    split_cols = [c for c in df.columns if isinstance(c, str) and re.match(r"^\(\s*-?\d+\s*,\s*-?\d+\s*\)$", c)]
    if not split_cols:
        raise RuntimeError("No split columns found in plan, expected '(1,23)' etc")

    # keep only valid splits: (-1,-1) OR nonnegative summing to TOTAL_TPCS
    splits: list[tuple[str, int, int]] = []
    for c in split_cols:
        t1, t2 = parse_split(c)
        if (t1, t2) == (-1, -1):
            splits.append((c, t1, t2))
        elif t1 >= 0 and t2 >= 0 and (t1 + t2 == TOTAL_TPCS):
            splits.append((c, t1, t2))

    # stable order: (-1,-1) then increasing t1
    def _k(x):
        _, a, b = x
        if (a, b) == (-1, -1):
            return (-10**9, 0)
        return (a, -b)
    splits.sort(key=_k)

    print(f"[PLAN] rows: {len(df)}")
    print(f"[PLAN] splits used: {[s[0] for s in splits]}")

    # Aggregators in log space:
    # key: (cluster_ut, cluster_co, split_col) -> (n, sum_log, sum_log2)
    agg: dict[tuple[int, int, str], list[float]] = {}

    # Global per split aggregator too
    g_agg: dict[str, list[float]] = {}

    n_used = 0
    n_skip_missing_curve = 0
    n_skip_bad = 0

    for _, r in df.iterrows():
        m_ut = r["Model_UT"]
        m_co = r["Model_CO"]
        id_ut = int(r["Kernel_ID_UT"])
        id_co = int(r["Kernel_ID_CO"])
        cu = int(r["cluster_UT"])
        cc = int(r["cluster_CO"])

        curve_u = curves.get((m_ut, id_ut))
        curve_c = curves.get((m_co, id_co))
        if curve_u is None or curve_c is None:
            n_skip_missing_curve += 1
            continue

        for col, t1, t2 in splits:
            meas = r.get(col, float("nan"))
            if pd.isna(meas):
                continue
            meas = float(meas)
            if meas <= 0:
                continue

            # baseline tpcs
            if (t1, t2) == (-1, -1):
                b1 = NO_MASK_BASE_TPC
                b2 = NO_MASK_BASE_TPC
            else:
                b1, b2 = t1, t2

            if b1 < 1 or b2 < 1 or b1 > TOTAL_TPCS or b2 > TOTAL_TPCS:
                n_skip_bad += 1
                continue

            xu = curve_u[b1]
            xc = curve_c[b2]
            if not (math.isfinite(xu) and math.isfinite(xc)):
                n_skip_bad += 1
                continue
            if xu <= 0 or xc <= 0:
                n_skip_bad += 1
                continue

            baseline = max(xu, xc)
            mult = meas / baseline
            if not math.isfinite(mult) or mult <= 0:
                n_skip_bad += 1
                continue

            mult = clamp(mult, CLAMP_MIN, CLAMP_MAX)
            lm = safe_log(mult)

            k = (cu, cc, col)
            if k not in agg:
                agg[k] = [0.0, 0.0, 0.0]  # n, sum_log, sum_log2
            agg[k][0] += 1.0
            agg[k][1] += lm
            agg[k][2] += lm * lm

            if col not in g_agg:
                g_agg[col] = [0.0, 0.0, 0.0]
            g_agg[col][0] += 1.0
            g_agg[col][1] += lm
            g_agg[col][2] += lm * lm

            n_used += 1

    print(f"[LUT] used samples: {n_used}")
    print(f"[LUT] skipped missing exclusive curves: {n_skip_missing_curve}")
    print(f"[LUT] skipped bad samples: {n_skip_bad}")
    if not agg:
        raise RuntimeError("No valid samples aggregated, check inputs")

    # materialize LUT rows
    out_rows = []
    for (cu, cc, col), (n, sum_log, sum_log2) in agg.items():
        n = int(n)
        mean_log = sum_log / n
        geomean = math.exp(mean_log)
        # log-space stddev (optional)
        var = max(0.0, (sum_log2 / n) - (mean_log * mean_log))
        std_log = math.sqrt(var)

        t1, t2 = parse_split(col)
        out_rows.append({
            "cluster_ut": cu,
            "cluster_co": cc,
            "split": col,
            "tpc1": t1,
            "tpc2": t2,
            "mult_geomean": geomean,
            "n_samples": n,
            "std_log": std_log,
        })

    lut_df = pd.DataFrame(out_rows).sort_values(
        by=["cluster_ut", "cluster_co", "tpc1", "tpc2"]
    ).reset_index(drop=True)

    # global fallback per split
    g_rows = []
    for col, (n, sum_log, sum_log2) in g_agg.items():
        n = int(n)
        mean_log = sum_log / n
        geomean = math.exp(mean_log)
        var = max(0.0, (sum_log2 / n) - (mean_log * mean_log))
        std_log = math.sqrt(var)
        t1, t2 = parse_split(col)
        g_rows.append({
            "split": col,
            "tpc1": t1,
            "tpc2": t2,
            "mult_geomean_global": geomean,
            "n_samples": n,
            "std_log": std_log,
        })
    g_df = pd.DataFrame(g_rows).sort_values(by=["tpc1", "tpc2"]).reset_index(drop=True)

    return lut_df, g_df


def main():
    if not PLAN_CSV.exists():
        raise FileNotFoundError(f"Plan not found: {PLAN_CSV.resolve()}")
    if not EXCL_DIR.exists():
        raise FileNotFoundError(f"Exclusive dir not found: {EXCL_DIR.resolve()}")

    curves = load_exclusive_curves(EXCL_DIR)
    lut_df, g_df = build_lut(PLAN_CSV, curves)

    lut_df.to_csv(OUT_LUT_CSV, index=False)
    g_df.to_csv(OUT_GLOBAL_CSV, index=False)

    print(f"[WRITE] LUT: {OUT_LUT_CSV.resolve()} rows={len(lut_df)}")
    print(f"[WRITE] Global: {OUT_GLOBAL_CSV.resolve()} rows={len(g_df)}")
    print("[DONE]")


if __name__ == "__main__":
    main()
