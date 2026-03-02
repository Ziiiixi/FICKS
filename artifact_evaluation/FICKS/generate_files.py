#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate mix JSON config files (ONLY low RPS) for 4-client mixed runs.

Output: one JSON per mix under OUT_DIR, named like:
  Rnet_8_Dnet_8_R1net_8_Mnet_32.json

Each JSON file is a list of 4 dict entries, same schema as your example.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple


# =========================
# USER CONFIG
# =========================

OUT_DIR = Path("./generated_mix_jsons")  # change to your desired output directory
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_ITERS = 25
LOW_RPS = 20            # "low" RPS value
UNIFORM = True
DUMMY_DATA = True
TRAIN = False

# Model keys used in mix names -> actual model metadata
# IMPORTANT: update kernel_file paths + num_kernels for Mnet_32 (and any others if needed).
MODEL_DB: Dict[str, Dict[str, Any]] = {
    "Rnet_8": {
        "arch": "resnet152",
        "model_name": "resnet152",
        "batchsize": 8,
        "kernel_file": "/home/zixi/orion_bu/benchmarking/model_kernels/cudnn_based/others/resnet152_8_fwd",
        "num_kernels": 513,
    },
    "R1net_8": {
        "arch": "resnet101",
        "model_name": "resnet101",
        "batchsize": 8,
        "kernel_file": "/home/zixi/orion_bu/benchmarking/model_kernels/cudnn_based/others/resnet101_8_fwd",
        "num_kernels": 343,
    },
    "Dnet_8": {
        "arch": "densenet201",
        "model_name": "densenet201",
        "batchsize": 8,
        "kernel_file": "/home/zixi/orion_bu/benchmarking/model_kernels/cudnn_based/others/densenet201_8_fwd",
        "num_kernels": 705,
    },
    "Vnet_8": {
        "arch": "vgg19",
        "model_name": "vgg19",
        "batchsize": 8,
        "kernel_file": "/home/zixi/orion_bu/benchmarking/model_kernels/cudnn_based/others/vgg19_8_fwd",
        "num_kernels": 56,
    },
    "Mnet_32": {
        "arch": "mobilenetv2",
        "model_name": "mobilenetv2",
        "batchsize": 32,
        "kernel_file": "/home/zixi/orion_bu/benchmarking/model_kernels/cudnn_based/others/mobilenetv2_32_fwd",
        "num_kernels": 0,  # TODO: fill with your real count
    },
}

# 5 mixes (choose 4 out of 5 models). Each model appears once.
MIXES: List[Tuple[str, ...]] = [
    ("Dnet_8", "R1net_8", "Mnet_32", "Vnet_8"),        # no Rnet_8
    ("Rnet_8", "R1net_8", "Mnet_32", "Vnet_8"),        # no Dnet_8
    ("Rnet_8", "Dnet_8",  "Mnet_32", "Vnet_8"),        # no R1net_8
    ("Rnet_8", "Dnet_8",  "R1net_8", "Vnet_8"),        # no Mnet_32
    ("Rnet_8", "Dnet_8",  "R1net_8", "Mnet_32"),       # no Vnet_8
]


# =========================
# HELPERS
# =========================

def make_entry(model_key: str, rps: int) -> Dict[str, Any]:
    if model_key not in MODEL_DB:
        raise KeyError(f"Unknown model key: {model_key}")

    m = MODEL_DB[model_key]
    required = ["arch", "model_name", "batchsize", "kernel_file", "num_kernels"]
    for k in required:
        if k not in m:
            raise KeyError(f"MODEL_DB['{model_key}'] missing '{k}'")

    return {
        "arch": m["arch"],
        "kernel_file": m["kernel_file"],
        "num_kernels": int(m["num_kernels"]),
        "num_iters": int(NUM_ITERS),
        "args": {
            "model_name": m["model_name"],
            "batchsize": int(m["batchsize"]),
            "rps": int(rps),
            "uniform": bool(UNIFORM),
            "dummy_data": bool(DUMMY_DATA),
            "train": bool(TRAIN),
        }
    }


def mix_name(mix: Tuple[str, ...]) -> str:
    return "_".join(mix)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=4)
        f.write("\n")


# =========================
# MAIN
# =========================

def main() -> None:
    for mix in MIXES:
        if len(set(mix)) != len(mix):
            raise ValueError(f"Mix has duplicates (expected all distinct): {mix}")
        for mk in mix:
            if mk not in MODEL_DB:
                raise KeyError(f"Mix references unknown model key: {mk}")

    total = 0
    for mix in MIXES:
        cfg_list: List[Dict[str, Any]] = [make_entry(mk, LOW_RPS) for mk in mix]
        out_path = OUT_DIR / f"{mix_name(mix)}.json"  # no postfix
        write_json(out_path, cfg_list)
        total += 1

    print(f"[OK] Wrote {total} json configs to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
