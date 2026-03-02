import pandas as pd
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument(
    '--results_dir',
    type=str,
    required=True,
    help='path to directory containing the profiling files'
)
parser.add_argument(
    '--ai_threshold',
    type=float,
    default=9.72,
    help='arithmetic intensity that seperates compute from memory bound kernels'
)

args = parser.parse_args()

# ───────────── Load CSVs ─────────────
df_raw = pd.read_csv(f'{args.results_dir}/raw_ncu.csv')

startp = 0
df_raw = df_raw.iloc[startp:]  # keep original indices, but skip first rows if needed

print(list(df_raw.iloc[0]))   # just to inspect the first row

df_basic = pd.read_csv(f'{args.results_dir}/output_ncu_sms.csv', index_col=0)

dram_throughput = df_basic['DRAM_Throughput(%)']
comp_throughput = df_basic['Compute(SM)(%)']

# ───────────── Metric column names ─────────────
fadd = 'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed [inst/cycle]'
fmul = 'smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed [inst/cycle]'
ffma = 'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed [inst/cycle]'

# Try usecond first, fall back to nsecond
cycles_metric = 'smsp__cycles_elapsed.avg.per_second [cycle/usecond]'
if cycles_metric not in df_raw.columns:
    cycles_metric = 'smsp__cycles_elapsed.avg.per_second [cycle/nsecond]'

bytes_metric = 'dram__bytes.sum.per_second [Gbyte/second]'

ai_list = []
roofline_prof = []  # 1: comp, 0: mem, -1: invalid

comp_bound = 0
mem_bound = 0
rest = 0

# ───────────── Main loop ─────────────
for index, row in df_raw.iterrows():
    # --- parse FLOP stats (inst/cycle) ---
    add_raw = row[fadd]
    mul_raw = row[fmul]
    fma_raw = row[ffma]

    add = float(str(add_raw).replace("'", '')) if not isinstance(add_raw, float) else float(add_raw)
    mul = float(str(mul_raw).replace("'", '')) if not isinstance(mul_raw, float) else float(mul_raw)
    fma = float(str(fma_raw).replace("'", '')) if not isinstance(fma_raw, float) else float(fma_raw)

    inst_per_cycle = add + mul + 2.0 * fma   # approximate FLOPs per cycle

    # --- cycles rate (cycles per µs or ns) ---
    cycles_rate = row[cycles_metric]  # cycles/usecond or cycles/nsecond

    if pd.isna(cycles_rate) or cycles_rate <= 0:
        cycles_per_second = 0.0
    else:
        if 'usecond' in cycles_metric:
            # [cycle/usecond] → [cycle/sec]
            cycles_per_second = cycles_rate * 1e6
        elif 'nsecond' in cycles_metric:
            # [cycle/nsecond] → [cycle/sec]
            cycles_per_second = cycles_rate * 1e9
        else:
            # already per second (fallback)
            cycles_per_second = cycles_rate

    # --- DRAM bytes per second ---
    gbytes_per_second = row[bytes_metric]  # [Gbyte/second]
    if pd.isna(gbytes_per_second) or gbytes_per_second <= 0:
        bytes_per_second = 0.0
    else:
        bytes_per_second = gbytes_per_second * 1e9  # Gbyte/s -> byte/s

    # --- decide how to compute AI / classification ---
    has_flops = (add != 0.0) or (mul != 0.0) or (fma != 0.0)

    if has_flops and cycles_per_second > 0.0 and bytes_per_second > 0.0:
        # FLOPs/s = (FLOPs/cycle) * (cycles/s)
        flops_per_second = inst_per_cycle * cycles_per_second
        ai = flops_per_second / bytes_per_second  # FLOPs per byte

        ai_list.append(ai)
        print(index, ai)

        if ai > args.ai_threshold:
            roofline_prof.append(1)  # compute-bound
            comp_bound += 1
        else:
            roofline_prof.append(0)  # memory-bound
            mem_bound += 1
    else:
        # No valid FLOP / byte stats → fallback to throughput heuristics
        ai_list.append(0.0)

        comp_val = comp_throughput[index - startp]
        dram_val = dram_throughput[index - startp]

        if comp_val >= 60.0:
            roofline_prof.append(1)
        elif dram_val >= 60.0:
            roofline_prof.append(0)
        else:
            roofline_prof.append(-1)
        rest += 1

print(df_basic)

df_basic['AI(flops/bytes)'] = ai_list
df_basic['Roofline_prof'] = roofline_prof
df_basic.to_csv(f'{args.results_dir}/output_ncu_sms_roofline.csv')

print(
    f"comp bound: {comp_bound}, "
    f"mem bound: {mem_bound}, "
    f"rest: {rest}, "
    f"total: {comp_bound + mem_bound + rest}"
)
