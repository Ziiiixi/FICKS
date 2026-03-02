#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Optional, Dict, Tuple

# Directory containing the JSON config files
CONFIG_DIR = Path("/home/zixi/orion_bu/artifact_evaluation/FICKS/configs/cudnn_based/test")

# Directory where the kernel CSV files live (these files may NOT end with .csv)
KERNEL_DIR = Path("/home/zixi/orion_bu/benchmarking/model_kernels/cudnn_based/test")

# Map arch -> (kernel file basename, num_kernels)
# Edit here when you add more models or change files.
ARCH_INFO: Dict[str, Tuple[str, int]] = {
    "resnet152":    ("resnet152_8_fwd", 513),
    "resnet101":    ("resnet101_8_fwd", 343),      # change if different on your side
    "densenet201":  ("densenet201_8_fwd", 705),    # change if different on your side
    "vgg19":        ("vgg19_8_fwd", 56),          # change if different on your side
    "mobilenet_v2": ("mobilenet_v2_32_fwd", 150),  # change if different on your side
    "mobilenetv2":  ("mobilenet_v2_32_fwd", 150),
}

def arch_to_kernel_path_and_count(arch: str) -> Tuple[Optional[str], Optional[int]]:
    info = ARCH_INFO.get(arch)
    if info is None:
        return None, None
    base, nk = info
    return str(KERNEL_DIR / base), nk

def process_json_file(path: Path):
    try:
        with path.open("r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}")
        return

    if not isinstance(data, list):
        print(f"[WARN] {path} is not a list of entries, skipping.")
        return

    changed = False
    for entry in data:
        if not isinstance(entry, dict):
            continue

        arch = entry.get("arch")
        if not arch:
            print(f"[WARN] {path.name}: entry without 'arch', skipping that entry.")
            continue

        new_kernel_path, new_num_kernels = arch_to_kernel_path_and_count(arch)
        if new_kernel_path is None or new_num_kernels is None:
            print(f"[WARN] {path.name}: unknown arch '{arch}', skipping that entry.")
            continue

        old_path = entry.get("kernel_file", "")
        if old_path != new_kernel_path:
            entry["kernel_file"] = new_kernel_path
            changed = True

        old_nk = entry.get("num_kernels", None)
        if old_nk != new_num_kernels:
            entry["num_kernels"] = new_num_kernels
            changed = True

    if not changed:
        print(f"[INFO] No changes for {path.name}")
        return

    with path.open("w") as f:
        json.dump(data, f, indent=4)
    print(f"[UPDATED] {path.name}")

if __name__ == "__main__":
    for json_path in sorted(CONFIG_DIR.glob("*.json")):
        process_json_file(json_path)
