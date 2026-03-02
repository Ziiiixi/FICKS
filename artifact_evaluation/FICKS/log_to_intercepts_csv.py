#!/usr/bin/env python3
# intercept_catch_only_to_csv.py

import argparse
import csv
import re
from pathlib import Path
from collections import Counter

# Match:
#   [INTERCEPTER-CATCH]-[123] ...
#   [INTERCEPTER-CATCH-0]-[123] ...
#   [INTERCEPTER-CATCH-7]-[123] ...
RE_TAG = re.compile(r"\[(INTERCEPTER-CATCH(?:-\d+)?)\]-\[(\d+)\]\s+(.*)$")
RE_API = re.compile(r"\bCaught\s+([A-Za-z0-9_]+)")

# Optional payload parsing (extend if needed)
RE_CUDNN_BN = re.compile(r"handle is\s+(0x[0-9a-fA-F]+),\s*index is\s+(\d+)")
RE_CUDNN_BE = re.compile(r"CUDNN handle is\s+(0x[0-9a-fA-F]+)")
RE_LAUNCH = re.compile(r"block size:\s*(\d+),\s*grid size:\s*(\d+)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="path to log file")
    ap.add_argument("--out", required=True, help="output csv path")
    ap.add_argument("--summary_out", default="", help="optional summary csv path")
    args = ap.parse_args()

    log_path = Path(args.log)
    out_path = Path(args.out)

    rows = []
    api_counts = Counter()
    tag_counts = Counter()

    with log_path.open("r", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            m = RE_TAG.search(line)
            if not m:
                continue

            catch_tag = m.group(1)   # INTERCEPTER-CATCH or INTERCEPTER-CATCH-0
            catch_id  = m.group(2)   # the number inside [...]
            rest      = m.group(3).rstrip("\n")

            api = ""
            ma = RE_API.search(rest)
            if ma:
                api = ma.group(1)

            cudnn_handle = ""
            cudnn_index = ""
            block = ""
            grid = ""

            mbn = RE_CUDNN_BN.search(rest)
            if mbn:
                cudnn_handle = mbn.group(1)
                cudnn_index = mbn.group(2)

            mbe = RE_CUDNN_BE.search(rest)
            if mbe and not cudnn_handle:
                cudnn_handle = mbe.group(1)

            ml = RE_LAUNCH.search(rest)
            if ml:
                block = ml.group(1)
                grid = ml.group(2)

            rows.append({
                "line_no": line_no,
                "catch_tag": catch_tag,
                "catch_id": catch_id,
                "api": api,
                "cudnn_handle": cudnn_handle,
                "cudnn_index": cudnn_index,
                "block": block,
                "grid": grid,
                "raw": rest,
            })

            tag_counts[catch_tag] += 1
            api_counts[api if api else "(unknown)"] += 1

    fields = ["line_no", "catch_tag", "catch_id", "api",
              "cudnn_handle", "cudnn_index", "block", "grid", "raw"]

    with out_path.open("w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    print("Counts by catch_tag:")
    for k, v in tag_counts.most_common():
        print(f"  {k}: {v}")
    print("Counts by API:")
    for k, v in api_counts.most_common():
        print(f"  {k}: {v}")

    if args.summary_out:
        s_path = Path(args.summary_out)
        with s_path.open("w", newline="") as fs:
            w = csv.writer(fs)
            w.writerow(["kind", "name", "count"])
            for k, v in tag_counts.most_common():
                w.writerow(["catch_tag", k, v])
            for k, v in api_counts.most_common():
                w.writerow(["api", k, v])
        print(f"Wrote summary to {s_path}")

if __name__ == "__main__":
    main()
