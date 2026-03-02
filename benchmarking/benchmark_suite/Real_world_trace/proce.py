#!/usr/bin/env python3
import json
import os
from pathlib import Path

# ----------- 配置 ----------
json_file = "REAL.json"   # 你的 JSON 输入文件
out_dir   = "trace_txt"       # 输出目录
# ---------------------------

Path(out_dir).mkdir(parents=True, exist_ok=True)

with open(json_file, "r") as f:
    data = json.load(f)

for task in data.get("tasks", []):
    load = task.get("load", {})
    if load.get("type") != "trace":
        # 不是 trace 负载就跳过
        continue

    task_id     = task.get("id", "unknown_task")
    trace_ms    = load.get("trace", [])
    trace_sec   = [ms / 1000.0 for ms in trace_ms]

    out_path = Path(out_dir) / f"{task_id}_trace.txt"
    with open(out_path, "w") as fout:
        for t in trace_sec:
            # 保留 6 位小数；如需其它精度可调整 .6f
            fout.write(f"{t:.6f}\n")

    print(f"✔ 生成 {out_path}（{len(trace_sec)} 行）")
