#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

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

    # ("low",    "", "", "Rnet_8_Rnet_8", 160000),
    # ("poisson","", "", "Rnet_8_Rnet_8", 160000),
    # ("twitter","", "", "Rnet_8_Rnet_8", 160000),
    # ("apollo", "", "", "Rnet_8_Rnet_8", 160000),

    # ("low",    "", "", "Dnet_8_Dnet_8", 160000),
    # # ("poisson","", "", "Dnet_8_Dnet_8", 160000),
    # # ("twitter","", "", "Dnet_8_Dnet_8", 160000),
    # ("apollo", "", "", "Dnet_8_Dnet_8", 160000),

    # ("low",    "", "", "R1net_8_R1net_8", 160000),
    # # ("poisson","", "", "R1net_8_R1net_8", 160000),
    # # ("twitter","", "", "R1net_8_R1net_8", 160000),
    # ("apollo", "", "", "R1net_8_R1net_8", 160000),

    # # ("low",    "", "", "Mnet_32_Mnet_32", 160000),
    # # ("poisson","", "", "Mnet_32_Mnet_32", 160000),
    # # ("twitter",    "", "", "Mnet_32_Mnet_32", 160000),
    # ("apollo","", "", "Mnet_32_Mnet_32", 160000),

    # ("low",    "", "", "Vnet_8_Vnet_8", 160000),
    # ("poisson","", "", "Vnet_8_Vnet_8", 160000),
    # ("twitter","", "", "Vnet_8_Vnet_8", 160000),
    # ("apollo", "", "", "Vnet_8_Vnet_8", 160000),



    # ("low", "", "", "Mnet_32_Dnet_8", 160000),
    # ("low", "", "", "Mnet_32_R1net_8", 160000),

    # ("low", "", "", "R1net_8_Dnet_8", 160000),
    # ("poisson", "", "", "R1net_8_Dnet_8", 160000),
    # ("twitter", "", "", "R1net_8_Dnet_8", 160000),
    # ("apollo", "", "", "R1net_8_Dnet_8", 160000),

    # ("low", "", "", "Rnet_8_Dnet_8", 160000),
    # ("poisson", "", "", "Rnet_8_Dnet_8", 160000),
    # ("twitter", "", "", "Rnet_8_Dnet_8", 160000),
    # ("apollo", "", "", "Rnet_8_Dnet_8", 160000),

    # ("low", "", "", "Rnet_8_Mnet_32", 160000),
    # ("poisson", "", "", "Rnet_8_Mnet_32", 160000),
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



    #  ("low",    "", "", "Rnet_8_Rnet_8_Rnet_8_Rnet_8", 160000),
    # ("poisson",   "", "", "Rnet_8_Rnet_8_Rnet_8_Rnet_8", 160000),
    # ("twitter","", "", "Rnet_8_Rnet_8_Rnet_8_Rnet_8", 160000),
    # ("apollo", "", "", "Rnet_8_Rnet_8_Rnet_8_Rnet_8", 160000),

    # ("low",    "", "", "Dnet_8_Dnet_8_Dnet_8_Dnet_8", 160000),
    # # ("poisson",   "", "", "Dnet_8_Dnet_8_Dnet_8_Dnet_8", 160000),
    # # ("twitter","", "", "Dnet_8_Dnet_8_Dnet_8_Dnet_8", 160000),
    # ("apollo", "", "", "Dnet_8_Dnet_8_Dnet_8_Dnet_8", 160000),

    # ("low",    "", "", "Vnet_8_Vnet_8_Vnet_8_Vnet_8", 160000),
    # ("poisson",   "", "", "Vnet_8_Vnet_8_Vnet_8_Vnet_8", 160000),
    # ("twitter","", "", "Vnet_8_Vnet_8_Vnet_8_Vnet_8", 160000),
    # ("apollo", "", "", "Vnet_8_Vnet_8_Vnet_8_Vnet_8", 160000),

    ("low",    "", "", "R1net_8_R1net_8_R1net_8_R1net_8", 160000),
    # ("poisson",   "", "", "R1net_8_R1net_8_R1net_8_R1net_8", 160000),
    # ("twitter","", "", "R1net_8_R1net_8_R1net_8_R1net_8", 160000),
    # ("apollo", "", "", "R1net_8_R1net_8_R1net_8_R1net_8", 160000),

    # ("low",   "", "", "Mnet_32_Mnet_32_Mnet_32_Mnet_32", 160000),
    # ("poisson",    "", "", "Mnet_32_Mnet_32_Mnet_32_Mnet_32", 160000),
    # ("twitter","", "", "Mnet_32_Mnet_32_Mnet_32_Mnet_32", 160000),
    # ("apollo", "", "", "Mnet_32_Mnet_32_Mnet_32_Mnet_32", 160000),

#    ("low",    "", "", "Rnet_8_Rnet_8_Dnet_8_Dnet_8", 160000),
#    ("poisson",    "", "", "Rnet_8_Rnet_8_Dnet_8_Dnet_8", 160000),
#    ("twitter",    "", "", "Rnet_8_Rnet_8_Dnet_8_Dnet_8", 160000),
#    ("apollo",    "", "", "Rnet_8_Rnet_8_Dnet_8_Dnet_8", 160000),

#    ("low",    "", "", "Rnet_8_Rnet_8_R1net_8_R1net_8", 160000),
# #    ("poisson",    "", "", "Rnet_8_Rnet_8_R1net_8_R1net_8", 160000),
# #    ("twitter",    "", "", "Rnet_8_Rnet_8_R1net_8_R1net_8", 160000),
#    ("apollo",    "", "", "Rnet_8_Rnet_8_R1net_8_R1net_8", 160000),

#    ("low",    "", "", "Dnet_8_Dnet_8_R1net_8_R1net_8", 160000),
#    ("poisson",    "", "", "Dnet_8_Dnet_8_R1net_8_R1net_8", 160000),
#    ("twitter",    "", "", "Dnet_8_Dnet_8_R1net_8_R1net_8", 160000),
#    ("apollo",    "", "", "Dnet_8_Dnet_8_R1net_8_R1net_8", 160000),
    
]

LOG_ROOT = "logs"


# ---------------------------------------------------------------------
# MIX BASED SHORT KERNEL LABEL CONFIG (used only when algo == "ficks")
# ---------------------------------------------------------------------
MODEL_CSV_ROOT = Path("/home/zixi/orion_bu/benchmarking/model_kernels/cudnn_based/ficks")

# fixed policy: use mean proxy from excl curve
PROXY_MODE = "mean"

# If you need unit conversion, set to 1e-6; otherwise keep 1.0
DURATION_SCALE = 1.0

# threshold scale library:
# priority:
#   (rps_profile, trace_name) > (rps_profile, "default") > ("default", "default")
THRESHOLD_SCALE_LIBRARY: Dict[Tuple[str, str], float] = {
    ("default", "default"): 0.9

    # ("low", "default"): 1.0,
    # ("poisson", "default"): 1.0,
    # ("twitter", "default"): 1.0,
    # ("apollo", "default"): 1.0,

    # examples:
    # ("low", "Mnet_32_Dnet_8"): 1.2,
    # ("twitter", "Dnet_8_Dnet_8"): 0.45,
    # ("twitter", "R1net_8_R1net_8"): 0.6,
}

MODEL_ALIAS_TO_FILE = {
    "Rnet_8": "resnet152_8_fwd",
    "R1net_8": "resnet101_8_fwd",
    "Dnet_8": "densenet201_8_fwd",
    "Mnet_32": "mobilenet_v2_32_fwd",
    "Vnet_8": "vgg19_8_fwd",
}

# IMPORTANT: keep R1net_8 before Rnet_8
MODEL_TOKEN_PATTERN = re.compile(r"(R1net_8|Rnet_8|Dnet_8|Mnet_32|Vnet_8)")
EXCL_COL_PATTERN = re.compile(r"^excl_(\d+)$")


def _get_scale(rps_profile: str, trace_name: str) -> float:
    if (rps_profile, trace_name) in THRESHOLD_SCALE_LIBRARY:
        return THRESHOLD_SCALE_LIBRARY[(rps_profile, trace_name)]
    if (rps_profile, "default") in THRESHOLD_SCALE_LIBRARY:
        return THRESHOLD_SCALE_LIBRARY[(rps_profile, "default")]
    return THRESHOLD_SCALE_LIBRARY.get(("default", "default"), 1.0)


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


def rewrite_is_short_for_trace_mix(trace_name: str, rps_profile: str) -> None:
    """
    Rewrite is_short_kernel IN-PLACE for models participating in trace_name.
    Threshold is computed globally from all kernels in this trace mix.
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
    scale = _get_scale(rps_profile, trace_name)
    threshold = global_mean * scale * DURATION_SCALE

    print(
        f"[mix-short] trace={trace_name} profile={rps_profile} "
        f"models={aliases} proxy={PROXY_MODE} "
        f"global_mean={global_mean:.6f} scale={scale:.6f} "
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


def main(algo: str) -> None:
    print(f"Number of runs per trace file: {NUM_RUNS}", flush=True)

    if algo == "ficks":
        config_dir = "configs/cudnn_based/ficks"
    else:
        config_dir = "configs/cudnn_based/others"

    for (rps_profile, _be, _hp, trace_name, max_be_duration) in trace_files:
        # STRICTLY FICKS ONLY
        # if algo == "ficks":
        #     rewrite_is_short_for_trace_mix(trace_name=trace_name, rps_profile=rps_profile)

        for run_idx in range(1, NUM_RUNS + 1):
            print(
                f"[{trace_name}] run {run_idx}/{NUM_RUNS}  "
                f"rps_profile={rps_profile}, algo={algo}",
                flush=True,
            )

            file_path = f"{config_dir}/{trace_name}.json"
            lib_path = "/home/zixi/orion_bu/src/cuda_capture/libinttemp.so"

            # logs/<algo>/<trace_name>/<rps_profile>/
            csv_dir = os.path.join(LOG_ROOT, algo, trace_name, rps_profile)
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
                "--log_root", LOG_ROOT,
            ]

            env = os.environ.copy()
            env["LD_PRELOAD"] = lib_path

            # print("Running:", flush=True)
            # print("  LD_PRELOAD=" + lib_path, flush=True)
            # print("  " + " ".join(cmd), flush=True)

            # keep running even if one run fails
            subprocess.run(cmd, env=env, check=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        type=str,
        choices=["orion", "reef", "multistream", "krisp", "ficks", "profile"],
        required=True,
        help="Choose one of: orion | reef | multistream | krisp | ficks | profile",
    )
    args = parser.parse_args()
    main(args.algo)
