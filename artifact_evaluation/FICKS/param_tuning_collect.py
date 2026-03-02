#!/usr/bin/env python3
import re
import sys
import ast
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ========= CONFIG (can be overridden by CLI) =========
LOG_PATH = "123.log"

# 1) Pivot tables (for humans)
OUT_DIR = Path("td_tables")

# 2) Persistent DB CSVs per config (for incremental reprofiling)
RESULT_DB_DIR = Path("td_results_db")

OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DB_DIR.mkdir(parents=True, exist_ok=True)

ENCODING = "utf-8"
ERRORS = "replace"
# ====================================================


# ====================================================
# Default candidate grid (used only if sweep lists are not found in log)
# depth: 1..20
# thr:   0.1..1.5 step 0.1
# ====================================================
DEFAULT_DEPTH_CANDIDATES = [float(x) for x in range(1, 21)]
DEFAULT_THR_CANDIDATES = [round(float(x), 4) for x in np.arange(0.1, 1.5 + 1e-9, 0.1)]

FLOAT_RE = r"[0-9]+(?:\.[0-9]+)?"

# New run header format:
# [Rnet_8_Rnet_8] run 1/2 algo=ficks rps=poisson td1(depth)=1 thr=0.1
RUN_RE_NEW = re.compile(
    rf"\[(?P<trace>[^\]]+)\]\s+run\s+(?P<run_idx>\d+)\s*/\s*(?P<run_total>\d+)"
    rf".*?td1(?:\s*\(depth\))?\s*=\s*(?P<depth>{FLOAT_RE})"
    rf".*?(?:thr|threshold)\s*=\s*(?P<thr>{FLOAT_RE})",
    re.IGNORECASE,
)

# Backward compatibility with old format:
# [cfg] run 1/2 ... td1 = 0.8, td2 = 1.2
RUN_RE_OLD = re.compile(
    rf"\[(?P<cfg>[^\]]+)\]\s+run\s+(?P<run_idx>\d+)\s*/\s*(?P<run_total>\d+)"
    rf".*?td1\s*=\s*(?P<td1>{FLOAT_RE})\s*,\s*td2\s*=\s*(?P<td2>{FLOAT_RE})",
    re.IGNORECASE,
)

# inline key=value from run header
INLINE_ALGO_RE = re.compile(r"\balgo\s*=\s*([A-Za-z0-9_.-]+)", re.IGNORECASE)
INLINE_RPS_RE = re.compile(r"\brps\s*=\s*([A-Za-z0-9_.-]+)", re.IGNORECASE)

# fallback metadata lines
IM_LOOP_ALGO_RE = re.compile(r"\[imagenet_loop\]\s*algo\s*=\s*([A-Za-z0-9_.-]+)", re.IGNORECASE)
IM_LOOP_RPS_RE = re.compile(r"\[imagenet_loop\]\s*rps_profile\s*=\s*([A-Za-z0-9_.-]+)", re.IGNORECASE)
IM_LOOP_TRACE_RE = re.compile(r"\[imagenet_loop\]\s*trace\s*=\s*([A-Za-z0-9_.-]+)", re.IGNORECASE)
IM_LOOP_RUNIDX_RE = re.compile(r"\[imagenet_loop\]\s*run_idx\s*=\s*(\d+)", re.IGNORECASE)

# additional fallback lines from scheduler/launcher
SCHED_ALGO_RE = re.compile(r"\[Scheduler\]\s*algo\s*=\s*([A-Za-z0-9_.-]+)", re.IGNORECASE)
# example: logs/td1_01_thr_0p1/ficks/Rnet_8_Rnet_8/poisson/run_1
CSV_BASE_RE = re.compile(
    r"csv_base\s*=\s*.*?/(?P<algo>[A-Za-z0-9_.-]+)/(?P<trace>[A-Za-z0-9_.-]+)/(?P<rps>[A-Za-z0-9_.-]+)/run_(?P<run_idx>\d+)",
    re.IGNORECASE,
)

CLIENT_P99_RE = re.compile(
    r"Client\s*(\d+).*?p99\s*[:=]\s*([0-9.eE+-]+)\s*sec",
    re.IGNORECASE,
)

MODEL_LINE_RE = re.compile(
    r"^\s*Model:\s*([A-Za-z0-9_]+)\s*$",
    re.IGNORECASE,
)

SWEEP_DEPTH_RE = re.compile(
    r"\[sweep\].*?td1\s*\(depth\)\s*values\s*:\s*(\[[^\]]*\])",
    re.IGNORECASE,
)

SWEEP_THR_RE = re.compile(
    r"\[sweep\].*?(?:threshold|thr)\s*values\s*:\s*(\[[^\]]*\])",
    re.IGNORECASE,
)


def sanitize_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")


def _round_depth(x: float) -> float:
    return float(int(round(float(x))))


def _round_thr(x: float) -> float:
    return round(float(x), 4)


def _parse_num_list_literal(s: str) -> List[float]:
    """
    Parse string like "[1, 2, 3]" or "[0.1, 0.2]".
    """
    try:
        arr = ast.literal_eval(s.strip())
        if not isinstance(arr, (list, tuple)):
            return []
        out: List[float] = []
        for v in arr:
            try:
                out.append(float(v))
            except Exception:
                pass
        return sorted(set(out))
    except Exception:
        return []


def parse_sweep_candidates(log_path: str) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Read sweep candidate lists from log:
      [sweep] td1(depth) values: [...]
      [sweep] threshold values: [...]
    """
    depth_vals: Optional[List[float]] = None
    thr_vals: Optional[List[float]] = None

    try:
        with open(log_path, "r", encoding=ENCODING, errors=ERRORS) as f:
            for line in f:
                md = SWEEP_DEPTH_RE.search(line)
                if md:
                    vals = _parse_num_list_literal(md.group(1))
                    if vals:
                        depth_vals = [_round_depth(v) for v in vals]

                mt = SWEEP_THR_RE.search(line)
                if mt:
                    vals = _parse_num_list_literal(mt.group(1))
                    if vals:
                        thr_vals = [_round_thr(v) for v in vals]
    except FileNotFoundError:
        return None, None

    return depth_vals, thr_vals


def parse_log(log_path: str) -> List[dict]:
    """
    Return list of dicts (one per run):
      {
        "config": str,          # trace__algo__rps
        "depth": float,
        "thr": float,
        "run_idx": int,
        "avg_p99": float,
        "worst_p99": float,
        "models": str,          # optional
        "p99_c0": float, ...
      }

    IMPORTANT:
      - We skip runs missing algo/rps/trace to avoid unknown_* configs.
    """
    records: List[dict] = []

    cur_trace: Optional[str] = None
    cur_algo: Optional[str] = None
    cur_rps: Optional[str] = None
    cur_depth: Optional[float] = None
    cur_thr: Optional[float] = None
    cur_run_idx: int = 1
    cur_client_p99: Dict[int, float] = {}
    cur_models: List[str] = []

    skipped_missing_meta = 0
    skipped_no_p99 = 0

    def flush_current():
        nonlocal records
        nonlocal cur_trace, cur_algo, cur_rps, cur_depth, cur_thr, cur_run_idx
        nonlocal cur_client_p99, cur_models
        nonlocal skipped_missing_meta, skipped_no_p99

        if cur_depth is None or cur_thr is None:
            cur_trace, cur_algo, cur_rps = None, None, None
            cur_depth, cur_thr, cur_run_idx = None, None, 1
            cur_client_p99, cur_models = {}, []
            return

        if not cur_client_p99:
            skipped_no_p99 += 1
            cur_trace, cur_algo, cur_rps = None, None, None
            cur_depth, cur_thr, cur_run_idx = None, None, 1
            cur_client_p99, cur_models = {}, []
            return

        if not cur_trace or not cur_algo or not cur_rps:
            skipped_missing_meta += 1
            cur_trace, cur_algo, cur_rps = None, None, None
            cur_depth, cur_thr, cur_run_idx = None, None, 1
            cur_client_p99, cur_models = {}, []
            return

        cfg = f"{cur_trace}__{cur_algo}__{cur_rps}"

        p99_vals = list(cur_client_p99.values())
        avg_p99 = float(np.mean(p99_vals))
        worst_p99 = float(np.max(p99_vals))

        rec = {
            "config": cfg,
            "depth": _round_depth(cur_depth),
            "thr": _round_thr(cur_thr),
            "run_idx": int(cur_run_idx),
            "avg_p99": avg_p99,
            "worst_p99": worst_p99,
        }

        if cur_models:
            uniq = []
            seen = set()
            for m in cur_models:
                mm = m.strip()
                if not mm:
                    continue
                k = mm.lower()
                if k in seen:
                    continue
                seen.add(k)
                uniq.append(mm)
            if uniq:
                rec["models"] = ",".join(uniq)

        for cid, v in cur_client_p99.items():
            rec[f"p99_c{cid}"] = float(v)

        records.append(rec)

        cur_trace, cur_algo, cur_rps = None, None, None
        cur_depth, cur_thr, cur_run_idx = None, None, 1
        cur_client_p99, cur_models = {}, []

    try:
        with open(log_path, "r", encoding=ENCODING, errors=ERRORS) as f:
            for raw_line in f:
                line = raw_line.strip()

                m_new = RUN_RE_NEW.search(line)
                if m_new:
                    flush_current()

                    cur_trace = m_new.group("trace").strip()
                    cur_run_idx = int(m_new.group("run_idx"))
                    cur_depth = float(m_new.group("depth"))
                    cur_thr = float(m_new.group("thr"))

                    ma = INLINE_ALGO_RE.search(line)
                    mr = INLINE_RPS_RE.search(line)
                    cur_algo = ma.group(1).strip() if ma else None
                    cur_rps = mr.group(1).strip() if mr else None

                    cur_client_p99 = {}
                    cur_models = []
                    continue

                m_old = RUN_RE_OLD.search(line)
                if m_old:
                    flush_current()

                    cur_trace = m_old.group("cfg").strip()
                    cur_run_idx = int(m_old.group("run_idx"))
                    cur_depth = float(m_old.group("td1"))
                    cur_thr = float(m_old.group("td2"))

                    cur_algo = None
                    cur_rps = None

                    cur_client_p99 = {}
                    cur_models = []
                    continue

                if cur_depth is not None and cur_thr is not None:
                    m_la = IM_LOOP_ALGO_RE.search(line)
                    if m_la:
                        cur_algo = m_la.group(1).strip()

                    m_lr = IM_LOOP_RPS_RE.search(line)
                    if m_lr:
                        cur_rps = m_lr.group(1).strip()

                    m_lt = IM_LOOP_TRACE_RE.search(line)
                    if m_lt and not cur_trace:
                        cur_trace = m_lt.group(1).strip()

                    m_li = IM_LOOP_RUNIDX_RE.search(line)
                    if m_li:
                        try:
                            cur_run_idx = int(m_li.group(1))
                        except Exception:
                            pass

                    m_sa = SCHED_ALGO_RE.search(line)
                    if m_sa and not cur_algo:
                        cur_algo = m_sa.group(1).strip()

                    m_cb = CSV_BASE_RE.search(line)
                    if m_cb:
                        if not cur_algo:
                            cur_algo = m_cb.group("algo").strip()
                        if not cur_trace:
                            cur_trace = m_cb.group("trace").strip()
                        if not cur_rps:
                            cur_rps = m_cb.group("rps").strip()
                        try:
                            cur_run_idx = int(m_cb.group("run_idx"))
                        except Exception:
                            pass

                m_model = MODEL_LINE_RE.match(line)
                if m_model and cur_depth is not None:
                    mm = m_model.group(1).strip()
                    if mm:
                        cur_models.append(mm)
                    continue

                m_cp = CLIENT_P99_RE.search(line)
                if m_cp and cur_depth is not None:
                    try:
                        cid = int(m_cp.group(1))
                        val = float(m_cp.group(2))
                        cur_client_p99[cid] = val
                    except ValueError:
                        pass

        flush_current()
    except FileNotFoundError:
        return []

    if skipped_missing_meta > 0:
        print(f"[INFO] skipped runs (missing trace/algo/rps): {skipped_missing_meta}")
    if skipped_no_p99 > 0:
        print(f"[INFO] skipped runs (no p99 found): {skipped_no_p99}")

    return records


# ====================================================
# Persistent DB helpers
# ====================================================
def _cfg_db_path(cfg: str) -> Path:
    safe = sanitize_name(cfg)
    return RESULT_DB_DIR / f"td_results_{safe}.csv"


def upsert_results_db(cfg: str, new_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Upsert per-config results into RESULT_DB_DIR/td_results_<cfg>.csv

    Key is (depth, thr, run_idx). If repeated, latest overwrites that run.
    """
    db_path = _cfg_db_path(cfg)

    new_rows = new_rows.copy()
    new_rows["depth"] = pd.to_numeric(new_rows["depth"], errors="coerce").round(0)
    new_rows["thr"] = pd.to_numeric(new_rows["thr"], errors="coerce").round(4)
    if "run_idx" not in new_rows.columns:
        new_rows["run_idx"] = 1
    new_rows["run_idx"] = pd.to_numeric(new_rows["run_idx"], errors="coerce").fillna(1).astype(int)

    new_rows = new_rows.dropna(subset=["depth", "thr"])

    if db_path.exists():
        try:
            old = pd.read_csv(db_path)
        except Exception as e:
            print(f"[WARN] Failed to read existing DB {db_path}: {e} (recreate)")
            old = pd.DataFrame()
    else:
        old = pd.DataFrame()

    if not old.empty:
        req = {"depth", "thr"}
        if not req.issubset(set(old.columns)):
            old = pd.DataFrame()
        else:
            old["depth"] = pd.to_numeric(old["depth"], errors="coerce").round(0)
            old["thr"] = pd.to_numeric(old["thr"], errors="coerce").round(4)
            if "run_idx" not in old.columns:
                old["run_idx"] = 1
            old["run_idx"] = pd.to_numeric(old["run_idx"], errors="coerce").fillna(1).astype(int)

    if old.empty:
        merged = new_rows.copy()
    else:
        all_cols = sorted(set(old.columns).union(set(new_rows.columns)))
        old2 = old.reindex(columns=all_cols)
        new2 = new_rows.reindex(columns=all_cols)

        merged = pd.concat([old2, new2], ignore_index=True)
        merged = merged.sort_values(["depth", "thr", "run_idx"], kind="mergesort")
        merged = merged.drop_duplicates(subset=["depth", "thr", "run_idx"], keep="last")

    client_cols = sorted(
        [c for c in merged.columns if c.startswith("p99_c")],
        key=lambda x: int(x.split("p99_c")[1])
    )
    if client_cols:
        merged["avg_p99"] = merged[client_cols].mean(axis=1)
        merged["worst_p99"] = merged[client_cols].max(axis=1)
    else:
        if "avg_p99" in merged.columns and "worst_p99" not in merged.columns:
            merged["worst_p99"] = merged["avg_p99"]

    merged = merged.sort_values(["depth", "thr", "run_idx"], kind="mergesort").reset_index(drop=True)
    merged.to_csv(db_path, index=False, float_format="%.8f")
    print(f"[DB] Updated {db_path}  rows={len(merged)}  (+{len(new_rows)} new/overwrite)")
    return merged


# ====================================================
# Missing-count helper (read DB and count remaining)
# ====================================================
def _choose_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    lc = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lc:
            return lc[n.lower()]
    return None


def count_left_in_db(
    cfg: str,
    depth_candidates: List[float],
    thr_candidates: List[float],
    num_runs: int,
) -> Tuple[int, int, int, bool]:
    """
    Returns:
      expected_total
      completed_effective
      left
      has_run_idx
    """
    db_path = _cfg_db_path(cfg)
    expected_total = len(depth_candidates) * len(thr_candidates) * int(num_runs)

    if not db_path.exists():
        return expected_total, 0, expected_total, False

    try:
        df = pd.read_csv(db_path)
    except Exception:
        return expected_total, 0, expected_total, False

    if df.empty:
        return expected_total, 0, expected_total, False

    depth_col = _choose_col(df, ["depth", "td1"])
    thr_col = _choose_col(df, ["thr", "threshold", "td2"])
    run_col = _choose_col(df, ["run_idx"])

    if depth_col is None or thr_col is None:
        return expected_total, 0, expected_total, False

    tmp = df.copy()
    tmp["__depth__"] = pd.to_numeric(tmp[depth_col], errors="coerce").round(0)
    tmp["__thr__"] = pd.to_numeric(tmp[thr_col], errors="coerce").round(4)
    tmp = tmp.dropna(subset=["__depth__", "__thr__"])

    has_run_idx = run_col is not None
    completed_effective = 0

    # normalize candidate grid
    dset = [float(_round_depth(v)) for v in depth_candidates]
    tset = [float(_round_thr(v)) for v in thr_candidates]

    if has_run_idx:
        tmp["__run__"] = pd.to_numeric(tmp[run_col], errors="coerce")
        tmp = tmp.dropna(subset=["__run__"])
        tmp["__run__"] = tmp["__run__"].astype(int)
        tmp = tmp[(tmp["__run__"] >= 1) & (tmp["__run__"] <= int(num_runs))]

        done = set(zip(tmp["__depth__"].astype(int), tmp["__thr__"].astype(float), tmp["__run__"].astype(int)))
        for d in dset:
            dd = int(d)
            for t in tset:
                tt = float(t)
                for r in range(1, int(num_runs) + 1):
                    if (dd, tt, r) in done:
                        completed_effective += 1
    else:
        # no run_idx: count rows per (depth,thr) as up to num_runs completed
        grp = tmp.groupby(["__depth__", "__thr__"]).size().to_dict()
        for d in dset:
            dd = int(d)
            for t in tset:
                tt = float(t)
                completed_effective += min(int(grp.get((dd, tt), 0)), int(num_runs))

    left = expected_total - completed_effective
    if left < 0:
        left = 0
    return expected_total, completed_effective, left, has_run_idx


def main():
    global LOG_PATH
    if len(sys.argv) > 1:
        LOG_PATH = sys.argv[1]

    print(f"[INFO] OUT_DIR={OUT_DIR.resolve()}")
    print(f"[INFO] RESULT_DB_DIR={RESULT_DB_DIR.resolve()}")

    depth_candidates, thr_candidates = parse_sweep_candidates(LOG_PATH)

    if depth_candidates is None:
        depth_candidates = DEFAULT_DEPTH_CANDIDATES
        print("[INFO] depth candidates not found in log; using default 1..20")
    else:
        print(f"[INFO] depth candidates from log: {depth_candidates}")

    if thr_candidates is None:
        thr_candidates = DEFAULT_THR_CANDIDATES
        print("[INFO] thr candidates not found in log; using default 0.1..1.5 step 0.1")
    else:
        print(f"[INFO] thr candidates from log: {thr_candidates}")

    # ---- Step 1: parse this log and update DB (no table generation, no best/reprofile prints) ----
    records = parse_log(LOG_PATH)
    cfgs_touched: List[str] = []

    if records:
        print(f"[INFO] Parsed {len(records)} runs from THIS log: {LOG_PATH}")
        df_new = pd.DataFrame(records)

        for cfg, df_cfg_new in df_new.groupby("config"):
            db_df = upsert_results_db(cfg, df_cfg_new.drop(columns=["config"], errors="ignore"))
            cfgs_touched.append(cfg)
    else:
        print(f"[INFO] No runs parsed from log '{LOG_PATH}'.")
        print("       (Either log path is wrong, or metadata/p99 lines were not matched.)")

    # ---- Step 2: print "how many profiles left" per config ----
    # If log had zero parsed runs, we still can report left for configs already in td_results_db
    if cfgs_touched:
        cfg_list = sorted(set(cfgs_touched))
    else:
        # fallback: list all DB files
        cfg_list = []
        for p in sorted(RESULT_DB_DIR.glob("td_results_*.csv")):
            safe_cfg = p.stem.replace("td_results_", "", 1)
            cfg_list.append(safe_cfg)

    if not cfg_list:
        print("[INFO] No configs to report (no DB files found).")
        return

    print("\n" + "=" * 110)
    print("[LEFT] Remaining profiles per config (based on td_results_db)")
    print("=" * 110)

    for cfg in cfg_list:
        expected_total, done_eff, left, has_run_idx = count_left_in_db(
            cfg=cfg,
            depth_candidates=depth_candidates,
            thr_candidates=thr_candidates,
            num_runs=1,  # this script doesn't know run_total per cfg; treat as 1 by default
        )
        print(
            f"[left] cfg={cfg} | expected={expected_total} done≈{done_eff} left={left} | db_has_run_idx={has_run_idx}"
        )


if __name__ == "__main__":
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 200)
    main()