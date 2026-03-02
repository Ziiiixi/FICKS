import pandas as pd
import argparse

def get_profile(profile_list, main_prof):
    pset = set(profile_list)
    if -1 in pset:
        pset.remove(-1)
    if len(pset) == 0:
        return -1
    if pset == {0}:
        return 0
    if pset == {1}:
        return 1
    if pset == {0, 1}:
        return main_prof
    return main_prof

def norm_name(s: str) -> str:
    return str(s).replace("<unnamed>", "(anonymous namespace)")

def is_nchw_to_nhwc(name: str) -> bool:
    # FIX: match even if there is no "engines_precompiled::" prefix
    return "nchwToNhwcKernel" in name or "nchwtonhwckernel" in name.lower()

def is_nhwc_to_nchw(name: str) -> bool:
    return "nhwcToNchwKernel" in name or "nhwctonchwkernel" in name.lower()

def is_conv_pre_helper(name: str) -> bool:
    n = name.lower()
    return (
        is_nchw_to_nhwc(name)
        or ("generatewinogradtileskernel" in n)            # covers "void winograd::generateWinogradTilesKernel"
        or ("im2col" in n)
        or ("im2col4d_kernel" in n)
        or ("kern_precompute_indices" in n)
        or ("computeoffsetskernel" in n)
        or ("fft" in n)
        or ("scalepackedtensor_kernel" in n)
        or ("nhwcaddpaddingkernel" in n)
        or ("nhwctofoldednhwckernel" in n)
        or ("foldednhwctonhwckernel" in n)
        or ("cudnn::winograd" in n)
        or ("cudnn::gemm" in n)
    )

def is_cudnn_or_cutlass_compute(name: str) -> bool:
    n = name.lower()
    return (
        ("__cudnn" in n)
        or ("_cudnn" in n)
        or ("cutlass__5x_cudnn::kernel" in n)
        or ("cutlass_cudnn::kernel" in n)
        or ("xmma_cudnn::" in n)
        or ("sm80_xmma" in n)
        or ("sm86_xmma" in n)
        or ("implicit_convolve_sgemm" in n)
        or ("explicit_convolve_sgemm" in n)
        or ("dgrad_engine" in n)
        or ("wgrad_alg" in n)
        or ("scudnn" in n)
        or ("convolve_common_engine" in n)
        or ("precomputed_convolve_sgemm" in n)
        or ("conv2d_grouped_direct_kernel" in n)
    )

def append_entry(out_list, name, prof, mem, sm, dur, grid="", block=""):
    out_list.append([name, prof, mem, sm, dur, grid, block])

def flush_pending_as_individual(processed, pending_rows):
    # If we buffered helpers but never saw a compute kernel, emit them as-is.
    for r in pending_rows:
        name0 = norm_name(r.get("Kernel_Name", "")).split("<")[0]
        append_entry(
            processed,
            name0,
            r.get("Roofline_prof", -1),
            0,
            r.get("SM_needed", 0),
            r.get("Duration(ns)", 0),
            r.get("Grid", ""),
            r.get("Block", "")
        )

parser = argparse.ArgumentParser()
parser.add_argument("--input_file_name", type=str, required=True)
parser.add_argument("--output_file_name", type=str, required=True)
parser.add_argument("--model_type", type=str, required=True)
args = parser.parse_args()

df = pd.read_csv(args.input_file_name)

# robust duration parsing (handles int or "1,234")
df["Duration(ns)"] = df["Duration(ns)"].astype(str).str.replace(",", "", regex=False).astype(int)

rows = df.to_dict("records")
processed_kernel_names = []

# buffer for conv helpers (layout transforms, winograd prep, etc.)
pending_conv_rows = []

i = 0
num_rows = len(rows)

while i < num_rows:
    row = rows[i]
    x_raw = row.get("Kernel_Name", "")
    x = norm_name(x_raw)

    # skip memcpy/memset rows (as you did)
    xl = x.lower()
    if ("memset" in xl) or ("memcpy" in xl):
        i += 1
        continue

    # 1) If we already have pending conv helpers and current is compute, CLOSE as Conv
    if pending_conv_rows and is_cudnn_or_cutlass_compute(x):
        merged_rows = pending_conv_rows + [row]
        pending_conv_rows = []

        # optional trailing nhwcToNchwKernel(s) right after compute
        j = i + 1
        while j < num_rows and is_nhwc_to_nchw(norm_name(rows[j].get("Kernel_Name", ""))):
            merged_rows.append(rows[j])
            j += 1

        sms_max = max(r.get("SM_needed", 0) for r in merged_rows)
        dur_sum = sum(r.get("Duration(ns)", 0) for r in merged_rows)
        profiles = [r.get("Roofline_prof", -1) for r in merged_rows]
        profile = get_profile(profiles, row.get("Roofline_prof", -1))

        append_entry(
            processed_kernel_names,
            "Conv",
            profile,
            0,
            sms_max,
            dur_sum,
            row.get("Grid", ""),
            row.get("Block", "")
        )

        i = j
        continue

    # 2) Buffer conv helper kernels (THIS now catches plain "nchwToNhwcKernel")
    if is_conv_pre_helper(x):
        pending_conv_rows.append(row)
        i += 1
        continue

    # 3) If we have pending helpers but hit a non-conv kernel, flush helpers
    if pending_conv_rows:
        flush_pending_as_individual(processed_kernel_names, pending_conv_rows)
        pending_conv_rows = []

    # 4) Original classification logic for other kernels
    if ("cudnn" in xl) and ("lstm" not in xl):
        if ("bn_fw" in xl) or ("bn_bw" in xl):
            append_entry(processed_kernel_names, "BatchNorm", row.get("Roofline_prof", -1), 0,
                         row.get("SM_needed", 0), row.get("Duration(ns)", 0),
                         row.get("Grid", ""), row.get("Block", ""))

        elif is_cudnn_or_cutlass_compute(x):
            append_entry(processed_kernel_names, "Conv", row.get("Roofline_prof", -1), 0,
                         row.get("SM_needed", 0), row.get("Duration(ns)", 0),
                         row.get("Grid", ""), row.get("Block", ""))

        else:
            append_entry(processed_kernel_names, x, row.get("Roofline_prof", -1), 0,
                         row.get("SM_needed", 0), row.get("Duration(ns)", 0),
                         row.get("Grid", ""), row.get("Block", ""))

        i += 1
        continue

    elif ("sm80_xmma" in xl) or ("sm86_xmma" in xl) or ("implicit_convolve_sgemm" in xl) or ("explicit_convolve_sgemm" in xl):
        append_entry(processed_kernel_names, "Conv", row.get("Roofline_prof", -1), 0,
                     row.get("SM_needed", 0), row.get("Duration(ns)", 0),
                     row.get("Grid", ""), row.get("Block", ""))

    elif args.model_type == "vision" and (("volta_sgemm_128x64_nn" in xl) or ("volta_sgemm_128x64_nt" in xl)):
        append_entry(processed_kernel_names, "Conv", row.get("Roofline_prof", -1), 0,
                     row.get("SM_needed", 0), row.get("Duration(ns)", 0),
                     row.get("Grid", ""), row.get("Block", ""))

    elif "reduce_kernel" in xl:
        # skip
        pass

    elif args.model_type == "transformer" and (("volta_sgemm_32x128_tn" in xl) or ("ampere_sgemm_32x128_tn" in xl)):
        sms = row.get("SM_needed", 0)
        duration = row.get("Duration(ns)", 0)
        profile = row.get("Roofline_prof", -1)

        if i < num_rows - 1:
            next_row = rows[i + 1]
            if "splitKreduce_kernel" in str(next_row.get("Kernel_Name", "")):
                sms = max(sms, next_row.get("SM_needed", 0))
                duration += next_row.get("Duration(ns)", 0)
                profile = get_profile([profile, next_row.get("Roofline_prof", -1)], profile)

        append_entry(processed_kernel_names, x.split("<")[0], profile, 0, sms, duration,
                     row.get("Grid", ""), row.get("Block", ""))

    else:
        append_entry(processed_kernel_names, x.split("<")[0], row.get("Roofline_prof", -1), 0,
                     row.get("SM_needed", 0), row.get("Duration(ns)", 0),
                     row.get("Grid", ""), row.get("Block", ""))

    i += 1

# flush any leftover pending helpers at EOF
if pending_conv_rows:
    flush_pending_as_individual(processed_kernel_names, pending_conv_rows)

with open(args.output_file_name, "w") as f:
    f.write("Name,Profile,Memory_footprint,SM_usage,Duration,Grid,Block\n")
    for r in processed_kernel_names:
        if len(r) < 7:
            r = r + [""] * (7 - len(r))
        f.write(",".join(str(x) for x in r) + "\n")
