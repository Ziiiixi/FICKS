#!/usr/bin/env python3
import os
import json
from pathlib import Path
from typing import Optional

# Directory containing the JSON config files
CONFIG_DIR = Path("/home/zixi/orion_bu/artifact_evaluation/FICKS/configs/krisp")

# Directory where the kernel CSV files live
KERNEL_DIR = Path("/home/zixi/orion_bu/benchmarking/model_kernels/krisp")

# Map from arch -> csv base name (without directory)
# Adjust / extend this mapping if your JSON uses different arch strings.
ARCH_TO_KERNEL_BASENAME = {
    "resnet152":    "resnet152_8_fwd",
    "resnet101":    "resnet152_8_fwd",
    "densenet201":  "densenet201_8_fwd",
    "vgg19":        "vgg19_8_fwd",
    "mobilenet_v2": "mobilenetv2_32_fwd",
    "mobilenetv2":  "mobilenetv2_32_fwd",
}

def kernel_path_for_arch(arch: str) -> Optional[str]:
    """
    Given an architecture string (e.g., 'resnet152'),
    return the full kernel_file path:
      /home/.../krisp/<basename>
    or None if arch is unknown.
    """
    base = ARCH_TO_KERNEL_BASENAME.get(arch)
    if base is None:
        return None
    return str(KERNEL_DIR / base)


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

        new_kernel_path = kernel_path_for_arch(arch)
        if new_kernel_path is None:
            print(f"[WARN] {path.name}: unknown arch '{arch}', skipping that entry.")
            continue

        old = entry.get("kernel_file", "")
        if old != new_kernel_path:
            entry["kernel_file"] = new_kernel_path
            changed = True

    if not changed:
        print(f"[INFO] No changes for {path.name}")
        return

    # Overwrite in place, no .bak
    with path.open("w") as f:
        json.dump(data, f, indent=4)
    print(f"[UPDATED] {path.name}")


if __name__ == "__main__":
    for json_path in sorted(CONFIG_DIR.glob("*.json")):
        process_json_file(json_path)
