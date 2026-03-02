#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
import pandas as pd

# ---- how many times to repeat each experiment ----
NUM_RUNS = 1

# (rps_profile, be, hp, tracefile_name, max_be_duration)
trace_files = [
    # ("low",   "", "", "Dnet_8", 160000),
    # ("low",   "", "", "Rnet_8", 160000),
    # ("low",   "", "", "R1net_8", 160000),
    # ("low",   "", "", "Mnet_32", 160000),
    # ("low",   "", "", "Vnet_8", 160000),

    ("low",    "", "", "Rnet_8_Rnet_8", 160000),
    ("poisson", "", "", "Rnet_8_Rnet_8", 160000),
    # ("twitter","", "", "Rnet_8_Rnet_8", 160000),
    # ("apollo", "", "", "Rnet_8_Rnet_8", 160000),

    ("low",    "", "", "Dnet_8_Dnet_8", 160000),
    ("poisson","", "", "Dnet_8_Dnet_8", 160000),
    # ("twitter","", "", "Dnet_8_Dnet_8", 160000),
    # ("apollo", "", "", "Dnet_8_Dnet_8", 160000),

    # ("low",    "", "", "R1net_8_R1net_8", 160000),
    # ("poisson","", "", "R1net_8_R1net_8", 160000),
    # ("twitter","", "", "R1net_8_R1net_8", 160000),
    # ("apollo", "", "", "R1net_8_R1net_8", 160000),

    # ("low",    "", "", "Mnet_32_Mnet_32", 160000),
    # ("poisson","", "", "Mnet_32_Mnet_32", 160000),

    # ("low",    "", "", "Vnet_8_Vnet_8", 160000),
    # ("poisson","", "", "Vnet_8_Vnet_8", 160000),
    # ("twitter","", "", "Vnet_8_Vnet_8", 160000),
    # ("apollo", "", "", "Vnet_8_Vnet_8", 160000),

    # ("low", "", "", "Mnet_32_Dnet_8", 160000),
    # ("low", "", "", "Mnet_32_R1net_8", 160000),

    ("low", "", "", "R1net_8_Dnet_8", 160000),
    ("poisson", "", "", "R1net_8_Dnet_8", 160000),
    # ("twitter", "", "", "R1net_8_Dnet_8", 160000),
    # ("apollo", "", "", "R1net_8_Dnet_8", 160000),

    ("low", "", "", "Rnet_8_Dnet_8", 160000),
    ("poisson", "", "", "Rnet_8_Dnet_8", 160000),
    # ("twitter", "", "", "Rnet_8_Dnet_8", 160000),
    # ("apollo", "", "", "Rnet_8_Dnet_8", 160000),

    ("low", "", "", "Rnet_8_Mnet_32", 160000),
    ("poisson", "", "", "Rnet_8_Mnet_32", 160000),
    # ("twitter", "", "", "Rnet_8_Mnet_32", 160000),
    # ("apollo", "", "", "Rnet_8_Mnet_32", 160000),

    # ("low", "", "", "Rnet_8_R1net_8", 160000),
    # ("poisson", "", "", "Rnet_8_R1net_8", 160000),
    # ("twitter", "", "", "Rnet_8_R1net_8", 160000),
    # ("apollo", "", "", "Rnet_8_R1net_8", 160000),

    # ("low", "", "", "Rnet_8_Vnet_8", 160000),
    # ("poisson", "", "", "Rnet_8_Vnet_8", 160000),
    # ("twitter", "", "", "Rnet_8_Vnet_8", 160000),
    # ("apollo", "", "", "Rnet_8_Vnet_8", 160000),

    # ("low", "", "", "Mnet_32_Vnet_8", 160000),
    # ("poisson", "", "", "Mnet_32_Vnet_8", 160000),

    # ("low", "", "", "Dnet_8_Vnet_8", 160000),
    # ("poisson", "", "", "Dnet_8_Vnet_8", 160000),
    # ("twitter", "", "", "Dnet_8_Vnet_8", 160000),
    # ("apollo", "", "", "Dnet_8_Vnet_8", 160000),

    # ("low",    "", "", "Rnet_8_Rnet_8_Rnet_8_Rnet_8", 160000),
    # ("poisson","", "", "Rnet_8_Rnet_8_Rnet_8_Rnet_8", 160000),
    # ("twitter","", "", "Rnet_8_Rnet_8_Rnet_8_Rnet_8", 160000),
    # ("apollo", "", "", "Rnet_8_Rnet_8_Rnet_8_Rnet_8", 160000),

    # ("low",    "", "", "Dnet_8_Dnet_8_Dnet_8_Dnet_8", 160000),
    # ("poisson","", "", "Dnet_8_Dnet_8_Dnet_8_Dnet_8", 160000),
    # ("twitter","", "", "Dnet_8_Dnet_8_Dnet_8_Dnet_8", 160000),
    # ("apollo", "", "", "Dnet_8_Dnet_8_Dnet_8_Dnet_8", 160000),

    # ("low",    "", "", "Vnet_8_Vnet_8_Vnet_8_Vnet_8", 160000),
    # ("poisson","", "", "Vnet_8_Vnet_8_Vnet_8_Vnet_8", 160000),
    # ("twitter","", "", "Vnet_8_Vnet_8_Vnet_8_Vnet_8", 160000),
    # ("apollo", "", "", "Vnet_8_Vnet_8_Vnet_8_Vnet_8", 160000),

    # ("low",    "", "", "R1net_8_R1net_8_R1net_8_R1net_8", 160000),
    # ("poisson","", "", "R1net_8_R1net_8_R1net_8_R1net_8", 160000),
    # ("twitter","", "", "R1net_8_R1net_8_R1net_8_R1net_8", 160000),
    # ("apollo", "", "", "R1net_8_R1net_8_R1net_8_R1net_8", 160000),

    # ("low",    "", "", "Mnet_32_Mnet_32_Mnet_32_Mnet_32", 160000),
    # ("poisson","", "", "Mnet_32_Mnet_32_Mnet_32_Mnet_32", 160000),
    # ("twitter","", "", "Mnet_32_Mnet_32_Mnet_32_Mnet_32", 160000),
    # ("apollo", "", "", "Mnet_32_Mnet_32_Mnet_32_Mnet_32", 160000),

    # ("low", "", "", "Rnet_8_R1net_8_Mnet_32_Vnet_8", 160000),
    # ("low",    "", "", "Rnet_8_Rnet_8_Dnet_8_Dnet_8", 160000),
]

LOG_ROOT = "logs"
RESULT_DB_DIR = Path("td_results_db")

# ---------------------------------------------------------------------
# SWEEP SETTINGS (for ficks)
# td1 is used to inject depth
# ---------------------------------------------------------------------
TD1_DEPTH_VALUES = list(range(1, 21, 2))  # 1,3,5,...,19
THRESHOLD_SWEEP_VALUES = [round(v, 1) for v in np.arange(0.1, 1.5 + 1e-9, 0.1)]  # 0.1..1.5
TD2_DUMMY_FOR_FICKS = 0.0  # compatibility only

# ---------------------------------------------------------------------
# MIX BASED SHORT KERNEL LABEL CONFIG (used only when algo == "ficks")
# ---------------------------------------------------------------------
MODEL_CSV_ROOT = Path("/home/zixi/orion_bu/benchmarking/model_kernels/cudnn_based/ficks")

# fixed policy: use mean proxy from excl curve
PROXY_MODE = "mean"

# If you need unit conversion, set to 1e-6; otherwise keep 1.0
DURATION_SCALE = 1.0

MODEL_ALIAS_TO_FILE = {
    "Rnet_8": "resnet152_8_fwd",
    "R1net_8": "resnet101_8_fwd",
    "Dnet_8": "densenet201_8_fwd",
    "Mnet_32": "mobilenet_v2_32_fwd",
    "Vnet_8": "vgg19_8_fwd",
}

# IMPORTANT: keep R1net_8 before Rnet_8
MODEL_TOKEN_PATTERN = re.compile(r"(R1net_8|Rnet_8|Dnet_8|Mnet_32|Vnet_8)")
EXCL_COL_PATTERN = re.compile(r'^excl_(\d+)$')


# ============================================================
# Helpers (DB and keys)
# ============================================================
def sanitize_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")


def _norm_depth(v) -> int:
    return int(round(float(v)))


def _norm_thr(v) -> float:
    return round(float(v), 4)


def _cfg_key(trace_name: str, algo: str, rps_profile: str) -> str:
    return f"{trace_name}__{algo}__{rps_profile}"


def _db_path_for_cfg(cfg_key: str) -> Path:
    safe = sanitize_name(cfg_key)
    return RESULT_DB_DIR / f"td_results_{safe}.csv"


def _choose_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    lc = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lc:
            return lc[n.lower()]
    return None


def _load_completed_from_db(
    db_path: Path,
    num_runs: int,
) -> Tuple[bool, Set[Tuple[int, float, int]], Dict[Tuple[int, float], int], int]:
    """
    Returns:
      has_run_idx,
      completed_slots set((depth,thr,run_idx)),
      pair_counts dict((depth,thr)->count_rows),
      db_rows
    """
    completed_slots: Set[Tuple[int, float, int]] = set()
    pair_counts: Dict[Tuple[int, float], int] = {}

    if not db_path.exists():
        return False, completed_slots, pair_counts, 0

    try:
        df = pd.read_csv(db_path)
    except Exception as e:
        print(f"[WARN] cannot read DB {db_path}: {e}")
        return False, completed_slots, pair_counts, 0

    if df.empty:
        return False, completed_slots, pair_counts, 0

    depth_col = _choose_col(df, ["depth", "td1"])
    thr_col = _choose_col(df, ["thr", "threshold", "td2"])
    run_col = _choose_col(df, ["run_idx"])

    if depth_col is None or thr_col is None:
        print(f"[WARN] DB missing depth/thr columns: {db_path}")
        return False, completed_slots, pair_counts, len(df)

    tmp = df.copy()
    tmp["__depth__"] = pd.to_numeric(tmp[depth_col], errors="coerce")
    tmp["__thr__"] = pd.to_numeric(tmp[thr_col], errors="coerce")
    tmp = tmp.dropna(subset=["__depth__", "__thr__"])

    if run_col is not None:
        tmp["__run__"] = pd.to_numeric(tmp[run_col], errors="coerce")
        tmp = tmp.dropna(subset=["__run__"])
        tmp["__run__"] = tmp["__run__"].astype(int)

        for _, r in tmp.iterrows():
            d = _norm_depth(r["__depth__"])
            t = _norm_thr(r["__thr__"])
            run_idx = int(r["__run__"])
            if 1 <= run_idx <= num_runs:
                completed_slots.add((d, t, run_idx))
            pair_counts[(d, t)] = pair_counts.get((d, t), 0) + 1

        return True, completed_slots, pair_counts, len(tmp)

    # no run_idx in DB
    for _, r in tmp.iterrows():
        d = _norm_depth(r["__depth__"])
        t = _norm_thr(r["__thr__"])
        pair_counts[(d, t)] = pair_counts.get((d, t), 0) + 1

    return False, completed_slots, pair_counts, len(tmp)


def _find_missing_jobs_from_db(
    trace_name: str,
    algo: str,
    rps_profile: str,
    depth_values: List[int],
    thr_values: List[float],
    num_runs: int,
) -> Tuple[List[Tuple[int, float, int]], Path, int, int, bool]:
    """
    Returns:
      missing_jobs: list of (depth, thr, run_idx)
      db_path
      expected_total
      completed_effective
      has_run_idx
    """
    cfg = _cfg_key(trace_name, algo, rps_profile)
    db_path = _db_path_for_cfg(cfg)

    has_run_idx, completed_slots, pair_counts, _db_rows = _load_completed_from_db(db_path, num_runs)

    missing: List[Tuple[int, float, int]] = []
    expected_total = len(depth_values) * len(thr_values) * num_runs
    completed_effective = 0

    for d in depth_values:
        dd = _norm_depth(d)
        for t in thr_values:
            tt = _norm_thr(t)
            if has_run_idx:
                local_done = 0
                for run_idx in range(1, num_runs + 1):
                    key = (dd, tt, run_idx)
                    if key in completed_slots:
                        local_done += 1
                    else:
                        missing.append(key)
                completed_effective += local_done
            else:
                # no run_idx in DB; use row count for this pair
                cnt = int(pair_counts.get((dd, tt), 0))
                done = min(cnt, num_runs)
                completed_effective += done
                for run_idx in range(done + 1, num_runs + 1):
                    missing.append((dd, tt, run_idx))

    missing.sort(key=lambda x: (x[0], x[1], x[2]))
    return missing, db_path, expected_total, completed_effective, has_run_idx


def _group_missing_by_pair(missing_jobs: List[Tuple[int, float, int]]) -> Dict[Tuple[int, float], List[int]]:
    out: Dict[Tuple[int, float], List[int]] = {}
    for d, t, r in missing_jobs:
        out.setdefault((d, t), []).append(r)
    for k in out:
        out[k] = sorted(set(out[k]))
    return out


# ============================================================
# Short-kernel rewrite helpers
# ============================================================
def _resolve_model_csv_path(root: Path, stem: str) -> Path:
    candidates = [root / stem, root / f"{stem}.csv"]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"Cannot find model CSV for '{stem}'. Tried: {[str(x) for x in candidates]}")


def _find_excl_cols(df: pd.DataFrame) -> List[str]:
    pairs = []
    for c in df.columns:
        m = EXCL_COL_PATTERN.match(str(c).strip())
        if m:
            pairs.append((int(m.group(1)), c))
    pairs.sort(key=lambda x: x[0])
    return [c for _, c in pairs]


def _duration_proxy_mean(df: pd.DataFrame, excl_cols: List[str]) -> pd.Series:
    mat = df[excl_cols].apply(pd.to_numeric, errors="coerce")
    return mat.mean(axis=1, skipna=True)


def _parse_models_from_trace_name(trace_name: str) -> List[str]:
    return MODEL_TOKEN_PATTERN.findall(trace_name)


def rewrite_is_short_for_trace_mix(trace_name: str, rps_profile: str, threshold_scale: float) -> None:
    """
    Rewrite is_short_kernel IN-PLACE for models participating in trace_name.
    Threshold is computed globally from all kernels in this trace mix:
      threshold = global_mean * threshold_scale * DURATION_SCALE
    """
    aliases = _parse_models_from_trace_name(trace_name)
    if not aliases:
        print(f"[mix-short] skip: no known model tokens parsed from trace '{trace_name}'", flush=True)
        return

    for a in aliases:
        if a not in MODEL_ALIAS_TO_FILE:
            raise KeyError(f"[mix-short] unknown model alias in trace: {a}")

    # load unique model tables once
    cache: Dict[Path, Dict[str, object]] = {}
    unique_aliases = sorted(set(aliases))
    for a in unique_aliases:
        stem = MODEL_ALIAS_TO_FILE[a]
        path = _resolve_model_csv_path(MODEL_CSV_ROOT, stem)
        df = pd.read_csv(path)
        excl_cols = _find_excl_cols(df)
        if not excl_cols:
            raise ValueError(f"[mix-short] no excl_* columns in {path}")
        proxy = _duration_proxy_mean(df, excl_cols)
        cache[path] = {
            "alias": a,
            "df": df,
            "proxy": proxy,
            "out_col": "is_short_kernel",
        }

    # global threshold weighted by occurrences in trace mix
    all_proxy_parts = []
    for a in aliases:
        p = _resolve_model_csv_path(MODEL_CSV_ROOT, MODEL_ALIAS_TO_FILE[a])
        all_proxy_parts.append(pd.to_numeric(cache[p]["proxy"], errors="coerce"))

    all_proxy = pd.concat(all_proxy_parts, ignore_index=True)
    all_proxy = all_proxy[np.isfinite(all_proxy.to_numpy(dtype=float))]
    if len(all_proxy) == 0:
        raise ValueError(f"[mix-short] no valid proxy values for trace {trace_name}")

    global_mean = float(all_proxy.mean())
    threshold = global_mean * float(threshold_scale) * DURATION_SCALE

    print(
        f"[mix-short] trace={trace_name} profile={rps_profile} "
        f"models={aliases} proxy={PROXY_MODE} "
        f"global_mean={global_mean:.6f} scale={float(threshold_scale):.6f} "
        f"duration_scale={DURATION_SCALE:.6f} threshold={threshold:.6f}",
        flush=True,
    )

    # rewrite each unique model file in place
    for path, item in cache.items():
        df = item["df"]
        proxy = pd.to_numeric(item["proxy"], errors="coerce")
        out_col = item["out_col"]

        old = pd.to_numeric(df[out_col], errors="coerce") if out_col in df.columns else pd.Series(np.nan, index=df.index)

        new_label = pd.Series(np.where(proxy <= threshold, 1, 0), index=df.index, dtype="Int64")

        invalid = ~np.isfinite(proxy.to_numpy(dtype=float))
        if invalid.any():
            fallback = old.fillna(0).astype("Int64")
            new_label.loc[invalid] = fallback.loc[invalid]

        df[out_col] = new_label.astype(int)

        old_cmp = old.fillna(-999999).astype(int)
        changed = int((old_cmp != df[out_col].astype(int)).sum())
        short_cnt = int((df[out_col].astype(int) == 1).sum())

        df.to_csv(path, index=False)

        print(
            f"[mix-short] wrote {path} | total={len(df)} short={short_cnt} changed={changed}",
            flush=True,
        )


def _fmt_thr_for_path(thr: float) -> str:
    return str(_norm_thr(thr)).replace(".", "p")


def _run_one_job(
    algo: str,
    rps_profile: str,
    trace_name: str,
    max_be_duration: int,
    run_idx: int,
    combo_log_root: str,
    td1_val: Optional[int] = None,
):
    config_dir = "configs/cudnn_based/ficks" if algo == "ficks" else "configs/cudnn_based/others"
    file_path = f"{config_dir}/{trace_name}.json"
    lib_path = "/home/zixi/orion_bu/src/cuda_capture/libinttemp.so"

    # logs/<combo>/<algo>/<trace_name>/<rps_profile>/
    csv_dir = os.path.join(combo_log_root, algo, trace_name, rps_profile)
    os.makedirs(csv_dir, exist_ok=True)

    cmd = [
        "python3.8",
        "../../benchmarking/launch_jobs.py",
        "--algo", algo,
        "--config_file", file_path,
        "--orion_max_be_duration", str(max_be_duration),
        "--rps_profile", rps_profile,
        "--trace_name", trace_name,
        "--run_idx", str(run_idx),
        "--log_root", combo_log_root,
    ]

    if algo == "ficks":
        cmd.extend(["--td1", str(float(td1_val))])                 # inject depth
        cmd.extend(["--td2", str(float(TD2_DUMMY_FOR_FICKS))])     # compatibility only

    env = os.environ.copy()
    env["LD_PRELOAD"] = lib_path
    subprocess.run(cmd, env=env, check=False)


def main(algo: str, only_list_missing: bool = False) -> None:
    print(f"Number of runs per trace file: {NUM_RUNS}", flush=True)
    RESULT_DB_DIR.mkdir(parents=True, exist_ok=True)

    if algo == "ficks":
        td1_values = [int(v) for v in TD1_DEPTH_VALUES]
        thr_values = [_norm_thr(v) for v in THRESHOLD_SWEEP_VALUES]
        print(f"[sweep] td1(depth) values: {td1_values}", flush=True)
        print(f"[sweep] threshold values: {thr_values}", flush=True)
        print(f"[db] using results db dir: {RESULT_DB_DIR.resolve()}", flush=True)
    else:
        td1_values = [None]
        thr_values = [None]

    total_submitted = 0
    total_missing = 0

    for (rps_profile, _be, _hp, trace_name, max_be_duration) in trace_files:
        if algo != "ficks":
            # Original behavior for non-ficks
            for run_idx in range(1, NUM_RUNS + 1):
                print(
                    f"[{trace_name}] run {run_idx}/{NUM_RUNS} "
                    f"algo={algo} rps={rps_profile}",
                    flush=True,
                )
                _run_one_job(
                    algo=algo,
                    rps_profile=rps_profile,
                    trace_name=trace_name,
                    max_be_duration=max_be_duration,
                    run_idx=run_idx,
                    combo_log_root=LOG_ROOT,
                    td1_val=None,
                )
                total_submitted += 1
            continue

        # --------------------------
        # ficks: query DB and run missing only
        # --------------------------
        missing_jobs, db_path, expected_total, completed_effective, has_run_idx = _find_missing_jobs_from_db(
            trace_name=trace_name,
            algo=algo,
            rps_profile=rps_profile,
            depth_values=td1_values,
            thr_values=thr_values,
            num_runs=NUM_RUNS,
        )

        missing_cnt = len(missing_jobs)
        total_missing += missing_cnt

        print("\n" + "=" * 110, flush=True)
        print(f"[plan] trace={trace_name} algo={algo} rps={rps_profile}", flush=True)
        print(f"[plan] db={db_path}", flush=True)
        print(f"[plan] expected={expected_total} completed≈{completed_effective} missing={missing_cnt}", flush=True)
        print(f"[plan] db_has_run_idx={has_run_idx}", flush=True)

        if missing_cnt == 0:
            print(f"[skip] all sweep points already done for {trace_name}/{rps_profile}", flush=True)
            continue

        # show a compact preview
        preview = ", ".join([f"(d={d},thr={t},run={r})" for d, t, r in missing_jobs[:20]])
        if missing_cnt > 20:
            preview += ", ..."
        print(f"[plan] missing preview: {preview}", flush=True)

        if only_list_missing:
            continue

        jobs_by_pair = _group_missing_by_pair(missing_jobs)

        # rewrite label once per threshold in this trace/rps
        rewritten_thr: Set[float] = set()

        for (td1_val, thr) in sorted(jobs_by_pair.keys(), key=lambda x: (x[0], x[1])):
            run_indices = jobs_by_pair[(td1_val, thr)]

            # rewrite short labels once per threshold
            if thr not in rewritten_thr:
                rewrite_is_short_for_trace_mix(
                    trace_name=trace_name,
                    rps_profile=rps_profile,
                    threshold_scale=float(thr),
                )
                rewritten_thr.add(thr)

            combo_log_root = os.path.join(
                LOG_ROOT,
                f"td1_{int(td1_val):02d}_thr_{_fmt_thr_for_path(float(thr))}",
            )

            for run_idx in run_indices:
                print(
                    f"[{trace_name}] run {run_idx}/{NUM_RUNS} "
                    f"algo={algo} rps={rps_profile} td1(depth)={td1_val} thr={thr}",
                    flush=True,
                )
                _run_one_job(
                    algo=algo,
                    rps_profile=rps_profile,
                    trace_name=trace_name,
                    max_be_duration=max_be_duration,
                    run_idx=run_idx,
                    combo_log_root=combo_log_root,
                    td1_val=td1_val,
                )
                total_submitted += 1

    print("\n" + "#" * 110, flush=True)
    print(f"[done] total_missing_planned={total_missing}", flush=True)
    print(f"[done] total_submitted={total_submitted}", flush=True)
    if only_list_missing:
        print("[done] only listed missing params (no jobs launched)", flush=True)
    print("#" * 110, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        type=str,
        choices=["orion", "reef", "multistream", "krisp", "ficks", "profile"],
        required=True,
        help="Choose one of: orion | reef | multistream | krisp | ficks | profile",
    )
    parser.add_argument(
        "--only_list_missing",
        action="store_true",
        help="Only print missing (depth,thr,run_idx) from td_results_db; do not launch experiments.",
    )
    args = parser.parse_args()

    main(args.algo, only_list_missing=args.only_list_missing)
