#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch process Nsight Systems reports into per-kernel CSVs with:
  - absolute start/end/duration in nanoseconds
  - stream id for each kernel
  - only keep the 11th request (detected by large time gaps between requests)

Robust handling:
  - if <report>.sqlite already exists next to <report>.nsys-rep / <report>.qdrep, use it
  - otherwise export with: nsys export --type sqlite <report>
    and force output into OUT_DIR by running export with cwd=OUT_DIR
  - avoid double-processing when both .nsys-rep and .sqlite exist
"""

from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


# ============================================================
# CONFIG (edit these only)
# ============================================================

ROOT_DIR = Path(".").resolve()

# Output directory
OUT_DIR = ROOT_DIR / "nsys_req11_processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Request boundary gap threshold (seconds)
GAP_SEC = 0.5

# Keep the 11th request (1-based)
KEEP_REQUEST_1BASED = 11

# If fewer than 11 segments are detected:
FALLBACK_TO_LAST_IF_NOT_ENOUGH = True

# Debug dump all requests segmentation result
DUMP_ALL_REQUEST_SEGMENTS_DEBUG = False


# ============================================================
# Helpers
# ============================================================

def run_cmd(cmd: list[str], cwd: Optional[Path] = None) -> None:
    """Run command and raise readable error on failure."""
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(cwd) if cwd is not None else None,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Cannot find 'nsys' command in PATH.\n"
            "Make sure Nsight Systems CLI is installed and 'nsys' is available.\n"
            "Try: which nsys"
        )
    except subprocess.CalledProcessError as e:
        msg = (
            "Command failed:\n"
            f"  {' '.join(cmd)}\n\n"
            f"cwd:\n  {cwd}\n\n"
            f"stdout:\n{e.stdout}\n\n"
            f"stderr:\n{e.stderr}\n"
        )
        raise RuntimeError(msg)


def sqlite_has_table(conn: sqlite3.Connection, table: str) -> bool:
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name=?;"
    return conn.execute(q, (table,)).fetchone() is not None


def sqlite_table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return {r[1] for r in rows}


def pick_inputs(root_dir: Path) -> list[Path]:
    """
    Deduplicate inputs by stem:
      prefer .sqlite > .nsys-rep > .qdrep
    """
    all_sqlite = sorted(root_dir.glob("*.sqlite"))
    all_rep = sorted(root_dir.glob("*.nsys-rep"))
    all_qdrep = sorted(root_dir.glob("*.qdrep"))

    chosen: dict[str, Path] = {}

    # priority: sqlite first
    for p in all_sqlite:
        chosen[p.stem] = p

    # then .nsys-rep if no sqlite exists
    for p in all_rep:
        if p.stem not in chosen:
            chosen[p.stem] = p

    # then .qdrep if no sqlite/nsys-rep exists
    for p in all_qdrep:
        if p.stem not in chosen:
            chosen[p.stem] = p

    return [chosen[k] for k in sorted(chosen.keys())]


def ensure_sqlite(report_path: Path, out_dir: Path) -> Path:
    """
    If input is .sqlite -> return it.
    Else:
      1) if sibling sqlite exists (same directory), return that
      2) else export into out_dir by running nsys export with cwd=out_dir
         and return out_dir/<stem>.sqlite
    """
    if report_path.suffix == ".sqlite":
        return report_path

    sibling_sqlite = report_path.with_suffix(".sqlite")
    if sibling_sqlite.exists():
        return sibling_sqlite

    out_sqlite = out_dir / f"{report_path.stem}.sqlite"
    if out_sqlite.exists():
        return out_sqlite

    # Export: by default nsys writes "<stem>.sqlite" into CURRENT DIRECTORY,
    # so we force cwd=out_dir.
    cmd = ["nsys", "export", "--type", "sqlite", str(report_path)]
    run_cmd(cmd, cwd=out_dir)

    if out_sqlite.exists():
        return out_sqlite

    # Fallback search: any sqlite with stem prefix in out_dir
    candidates = list(out_dir.glob(f"{report_path.stem}*.sqlite"))
    if candidates:
        return sorted(candidates)[0]

    raise RuntimeError(f"Expected sqlite not found after export: {out_sqlite}")


def load_kernels_from_sqlite(sqlite_path: Path) -> pd.DataFrame:
    """
    Load kernel events from CUPTI_ACTIVITY_KIND_KERNEL.
    Returns columns:
      start_ns, end_ns, dur_ns, stream_id, kernel_name
    """
    conn = sqlite3.connect(str(sqlite_path))
    try:
        if not sqlite_has_table(conn, "CUPTI_ACTIVITY_KIND_KERNEL"):
            raise RuntimeError(f"Missing table CUPTI_ACTIVITY_KIND_KERNEL in {sqlite_path.name}")

        kcols = sqlite_table_columns(conn, "CUPTI_ACTIVITY_KIND_KERNEL")

        required = {"start", "end", "streamId"}
        missing = required - kcols
        if missing:
            raise RuntimeError(
                f"Kernel table missing columns {sorted(missing)} in {sqlite_path.name}. "
                f"Columns found: {sorted(kcols)}"
            )

        # Find name id column
        name_key = None
        for cand in ["demangledName", "name"]:
            if cand in kcols:
                name_key = cand
                break

        if name_key is None:
            sql = """
                SELECT
                  k.start AS start_ns,
                  k.end   AS end_ns,
                  (k.end - k.start) AS dur_ns,
                  k.streamId AS stream_id,
                  '' AS kernel_name
                FROM CUPTI_ACTIVITY_KIND_KERNEL k
                ORDER BY k.start;
            """
            return pd.read_sql_query(sql, conn)

        if not sqlite_has_table(conn, "StringIds"):
            sql = f"""
                SELECT
                  k.start AS start_ns,
                  k.end   AS end_ns,
                  (k.end - k.start) AS dur_ns,
                  k.streamId AS stream_id,
                  CAST(k.{name_key} AS TEXT) AS kernel_name
                FROM CUPTI_ACTIVITY_KIND_KERNEL k
                ORDER BY k.start;
            """
            return pd.read_sql_query(sql, conn)

        sql = f"""
            SELECT
              k.start AS start_ns,
              k.end   AS end_ns,
              (k.end - k.start) AS dur_ns,
              k.streamId AS stream_id,
              s.value AS kernel_name
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            LEFT JOIN StringIds s ON s.id = k.{name_key}
            ORDER BY k.start;
        """
        return pd.read_sql_query(sql, conn)

    finally:
        conn.close()


def segment_requests_by_gap(df_k: pd.DataFrame, gap_sec: float) -> Tuple[pd.DataFrame, int]:
    """Add request_id based on large gaps between consecutive kernel start times."""
    if df_k.empty:
        df_k["request_id"] = []
        return df_k, 0

    df = df_k.sort_values("start_ns").reset_index(drop=True)
    gap_ns = int(round(gap_sec * 1e9))

    start_diff = df["start_ns"].diff().fillna(0).astype("int64")
    boundary = (start_diff >= gap_ns).astype("int64")
    df["request_id"] = boundary.cumsum().astype("int64")

    num_requests = int(df["request_id"].max() + 1)
    return df, num_requests


def keep_request(df: pd.DataFrame, keep_req_1based: int, fallback_last: bool) -> Optional[pd.DataFrame]:
    if df.empty:
        return None

    keep_req0 = keep_req_1based - 1
    max_req = int(df["request_id"].max())

    if keep_req0 > max_req:
        if not fallback_last:
            return None
        keep_req0 = max_req

    out = df[df["request_id"] == keep_req0].copy()
    return out


def add_relative_times(df_req: pd.DataFrame) -> pd.DataFrame:
    if df_req.empty:
        df_req["start_ns_rel"] = []
        df_req["end_ns_rel"] = []
        return df_req

    t0 = int(df_req["start_ns"].min())
    df_req["start_ns_rel"] = (df_req["start_ns"] - t0).astype("int64")
    df_req["end_ns_rel"] = (df_req["end_ns"] - t0).astype("int64")
    return df_req


def write_outputs(input_path: Path, df_req11: pd.DataFrame, out_dir: Path) -> None:
    base = input_path.stem
    out_kernels = out_dir / f"{base}__req{KEEP_REQUEST_1BASED}_kernels.csv"
    out_streams = out_dir / f"{base}__req{KEEP_REQUEST_1BASED}_stream_summary.csv"

    cols = [
        "start_ns", "end_ns", "dur_ns",
        "start_ns_rel", "end_ns_rel",
        "stream_id", "kernel_name",
    ]
    for c in cols:
        if c not in df_req11.columns:
            df_req11[c] = ""

    df_req11[cols].to_csv(out_kernels, index=False)

    df_stream = (
        df_req11.groupby("stream_id", as_index=False)
        .agg(
            num_kernels=("dur_ns", "count"),
            total_dur_ns=("dur_ns", "sum"),
            min_start_ns=("start_ns", "min"),
            max_end_ns=("end_ns", "max"),
        )
        .sort_values(["total_dur_ns", "num_kernels"], ascending=False)
        .reset_index(drop=True)
    )
    df_stream.to_csv(out_streams, index=False)

    print(f"[OK] {input_path.name}")
    print(f"     kernels : {out_kernels}")
    print(f"     streams : {out_streams}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(f"[INFO] ROOT_DIR = {ROOT_DIR}")
    print(f"[INFO] OUT_DIR  = {OUT_DIR}")
    print(f"[INFO] GAP_SEC  = {GAP_SEC}")
    print(f"[INFO] KEEP_REQUEST_1BASED = {KEEP_REQUEST_1BASED}")
    print()

    inputs = pick_inputs(ROOT_DIR)
    if not inputs:
        print("[WARN] No report files found in ROOT_DIR.")
        print("       Supported: *.sqlite, *.nsys-rep, *.qdrep")
        return

    for rp in inputs:
        try:
            sqlite_path = ensure_sqlite(rp, OUT_DIR)
            df_k = load_kernels_from_sqlite(sqlite_path)

            if df_k.empty:
                print(f"[SKIP] {rp.name}: no kernels found")
                continue

            df_seg, nreq = segment_requests_by_gap(df_k, GAP_SEC)

            if DUMP_ALL_REQUEST_SEGMENTS_DEBUG:
                dbg_path = OUT_DIR / f"{rp.stem}__all_segments_debug.csv"
                df_seg.to_csv(dbg_path, index=False)
                print(f"[DBG] wrote all segments: {dbg_path}")

            df_req = keep_request(df_seg, KEEP_REQUEST_1BASED, FALLBACK_TO_LAST_IF_NOT_ENOUGH)
            if df_req is None or df_req.empty:
                print(f"[SKIP] {rp.name}: cannot find request {KEEP_REQUEST_1BASED} (detected {nreq})")
                continue

            df_req = add_relative_times(df_req)
            write_outputs(rp, df_req, OUT_DIR)

        except Exception as e:
            print(f"[ERR] {rp.name}: {e}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
