#!/usr/bin/env bash
set -e  # stop on first error

mkdir -p logs

# -------- 1) Main runs for all algos --------
for algo in multistream krisp orion reef ficks; do
    log="logs/${algo}.log"
    echo "Running run_orion.py --algo ${algo}  ->  ${log}"

    python run_orion.py --algo "${algo}" > "${log}" 2>&1
done

# # -------- 2) Tuning at the end (only for ficks) --------
# tune_log="logs/ficks_tuning.log"
# echo "Running run_orion_tuning.py --algo ficks  ->  ${tune_log}"

# python run_orion_tuning.py --algo ficks > "${tune_log}" 2>&1
