#!/usr/bin/env python3
import re
import csv
from collections import defaultdict

# ——— CONFIG —————————————————————————————————————————————
INPUT_LOG  = "123.log"
OUTPUT_CSV = "sm_analysis.csv"

# regex to match lines like:
# At time 5 µs,  kernel 1 is in execution from client 1, tpc used 24, available tpc 0
pattern = re.compile(
    r"At time\s+(\d+)\s*µs,\s*kernel\s+(\d+)\s+is in execution\s+from\s+client\s+(\d+),\s*tpc used\s+(\d+)"
)

# data[time][client] = (kernel, sm_usage)
data = defaultdict(dict)

with open(INPUT_LOG, "r") as f:
    for line in f:
        m = pattern.search(line)
        if not m:
            continue
        time, kernel, client, sm = m.groups()
        t = int(time)
        data[t][int(client)] = (int(kernel), int(sm))

# figure out which client IDs appear (e.g. {0,1})
clients = sorted({c for recs in data.values() for c in recs})
# if you want to rename client 0→“Client 1” etc, you can adjust here

# write CSV with header:
# Time Stamp, KernelID_C0, SMUsage_C0, KernelID_C1, SMUsage_C1, …
with open(OUTPUT_CSV, "w", newline="") as csvf:
    writer = csv.writer(csvf)
    header = ["Time (µs)"]
    for c in clients:
        header += [f"Kernel ID (Client {c})", f"SM Usage (Client {c})"]
    writer.writerow(header)

    for t in sorted(data):
        row = [t]
        recs = data[t]
        for c in clients:
            if c in recs:
                k, sm = recs[c]
                row += [k, sm]
            else:
                row += ["", ""]
        writer.writerow(row)

print(f"Written pivoted data to {OUTPUT_CSV}")
