import os
import re
import pandas as pd

# ─── MODEL NAME MAPPING ───────────────────────────────────────────────────────
model_name_map = {
    "Mnet":  "mobilenetv2",
    "Rnet":  "resnet152",
    "R1net": "resnet101",
    "Dnet":  "densenet201",
    "Vnet":  "vgg19",
}
# ──────────────────────────────────────────────────────────────────────────────

# ─── CONFIGURE TRACE FILES ───────────────────────────────────────────────────
# Each tuple: (be_client, hp_client, config_name, max_be_duration)
# config_name is also the JSON basename, e.g. "Rnet_8_Rnet_8"
trace_files = [
    # ("", "", "R1net_8_R1net_8", 160000),
    # ("", "", "Dnet_8_Dnet_8", 160000),
    # ("", "", "Vnet_8_Vnet_8", 160000),
    # ("", "", "Mnet_32_Mnet_32", 160000),
    
    # You can add more entries here...
]
# Base directory where your kernel CSVs live
CSV_DIR = "~/orion_bu/benchmarking/model_kernels/small_models/test_critical"
# Base directory for NCU roofline outputs
NCU_BASE_DIR = os.path.expanduser("~/orion_bu/profiling/benchmarks")
# Where your JSON configs live (basename is config_name + ".json")
CONFIG_DIR = "config_files/new_baselines"
# Where to save results
RESULTS_DIR = "results/orion_bu"
# Path to your LD_PRELOAD library
LIB_PATH = "/home/zixi/orion_bu/src/cuda_capture/libinttemp.so"
# Number of repeated runs per config
NUM_RUNS = 1

threshold = 85
# ──────────────────────────────────────────────────────────────────────────────


def update_critical_ops(first_csv, ncu_csv, output_csv):
    df1 = pd.read_csv(first_csv)
    df2 = pd.read_csv(ncu_csv)

    # filter out unwanted kernels
    df2 = df2[
        ~df2['Kernel_Name']
           .str.contains('splitKreduce_kernel|reduce_kernel', case=False, na=False)
    ].copy()
    df2['Kernel_Name'] = df2['Kernel_Name'].str.strip('"')

    # normalize Grid/Block
    for df in (df1, df2):
        df['Grid']  = df['Grid'].astype(str)
        df['Block'] = df['Block'].astype(str)

    # recompute critical based on Compute(SM)(%) > 50
    def is_critical(row):
        pat = re.escape(row['Name'])
        matches = df2[
            df2['Kernel_Name'].str.contains(pat, regex=True, na=False) &
            (df2['Grid'] == row['Grid']) &
            (df2['Block'] == row['Block'])
        ]
        return int((matches['Compute(SM)(%)'] > threshold).any())

    df1['is critical'] = df1.apply(is_critical, axis=1)
    df1.to_csv(output_csv, index=False)
    print(f"✓ Wrote updated CSV: {output_csv}")


if __name__ == "__main__":
    for be, hp, config_name, max_be_duration in trace_files:
        # parse prefix (e.g. "Rnet") and batch (e.g. "8")
        parts = config_name.split("_", 2)
        prefix, batch = parts[0], parts[1]
        model = model_name_map.get(prefix)
        if model is None:
            raise ValueError(f"Unknown model prefix '{prefix}' in '{config_name}'")

        # paths for this model/batch
        first_csv  = os.path.join(CSV_DIR, f"{model}_{batch}_fwd")
        ncu_csv    = os.path.join(
            NCU_BASE_DIR,
            f"{model}_bz{batch}",
            "output_ncu_sms_roofline.csv"
        )
        updated_csv = os.path.join(
            CSV_DIR,
            f"{model}_{batch}_fwd_updated"
        )
        print(f"\n— Processing {model} batch {batch} —")
        update_critical_ops(first_csv, ncu_csv, updated_csv)

        # run benchmarks
        cfg_path = os.path.join(CONFIG_DIR, f"{config_name}.json")
        for run in range(NUM_RUNS):
            print(f"Run {run} for config {config_name}")
            os.system(
                f"LD_PRELOAD='{LIB_PATH}' "
                f"python3.8 ../../benchmarking/launch_jobs.py "
                f"--algo orion --config_file {cfg_path} "
                f"--orion_max_be_duration {max_be_duration}"
            )
            # copy results
            os.system(f"cp client_1.json {RESULTS_DIR}/{be}_{hp}_{run}_hp.json")
            os.system(f"cp client_0.json {RESULTS_DIR}/{be}_{hp}_{run}_be.json")
            os.system("rm client_1.json client_0.json")

