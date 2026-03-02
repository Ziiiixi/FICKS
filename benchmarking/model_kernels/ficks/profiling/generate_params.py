#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import pandas as pd

# NOTE: now using the CSV with all cluster pairs
PLAN_CSV = Path("/home/zixi/orion_bu/benchmarking/model_kernels/ficks/profiling/"
                "A4000_profile_plan_all_cluster_pairs_up_to3.csv")
OUT_JSON = Path("/home/zixi/orion_bu/benchmarking/model_kernels/ficks/profiling/"
                "knee_params.json")

# Sensible starting defaults (you can change after generation)
DEFAULTS = {
    "coloc":     {"tau_abs": 0.05, "tau_frac": 0.01, "win": 3, "coverage_p": 0.80},
    "exclusive": {"tau_abs": 0.05, "tau_frac": 0.01, "win": 3, "coverage_p": 0.80}
}

def main():
    if not PLAN_CSV.exists():
        raise FileNotFoundError(f"PLAN CSV not found: {PLAN_CSV}")

    df = pd.read_csv(PLAN_CSV)

    # Expect these columns in the plan
    required_cols = ["Model_UT", "Model_CO"]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}' in plan CSV")

    # Collect unique model names from both UT and CO sides
    models_ut = df["Model_UT"].dropna().astype(str).unique().tolist()
    models_co = df["Model_CO"].dropna().astype(str).unique().tolist()
    all_models = sorted(set(models_ut) | set(models_co))

    print(f"[INFO] Found {len(all_models)} unique models in plan CSV:")
    for m in all_models:
        print("   ", m)

    data = {
        "defaults": DEFAULTS,
        "models": {m: DEFAULTS for m in all_models}
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] Wrote params template: {OUT_JSON.resolve()}")

if __name__ == "__main__":
    main()
