#!/usr/bin/env python3
"""
Migrate already-filled TPC profile cells from an OLD plan CSV into a NEW plan CSV.

Why this script works for your case:
- Old plan may contain kernel IDs that no longer exist (e.g., vgg had more kernels before).
- Direct pair matching may fail.
- So we do TWO passes:
  (A) Direct pair-key migration when pair keys still match.
  (B) Cluster-pair based migration: reuse old filled cells by (cluster_UT, cluster_CO) bucket,
      then fill empty cells in the new plan for rows with the same cluster pair.

We never overwrite existing non-NaN cells in the new plan.

Output: <new_plan>_migrated.csv by default.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import re
import pandas as pd
import numpy as np
from collections import defaultdict, deque


TPC_COL_RE = re.compile(r"^\(\d+,\s*\d+\)$")


def find_tpc_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if isinstance(c, str) and TPC_COL_RE.match(c)]


def ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out


def norm_str_col(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()


def norm_int_col(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")


def has_any_profile_row(df: pd.DataFrame, tpc_cols: list[str]) -> pd.Series:
    # True if any TPC cell is non-NA in that row
    if not tpc_cols:
        return pd.Series(False, index=df.index)
    return ~df[tpc_cols].isna().all(axis=1)


def build_cluster_pool(
    old_df: pd.DataFrame,
    tpc_cols: list[str],
    symmetric: bool = False,
) -> dict[tuple[int, int, str], deque]:
    """
    Build a pool:
      key = (cluster_ut, cluster_co, tpc_col) -> deque([values...])
    from rows in old_df that already have filled cells.
    """
    pool: dict[tuple[int, int, str], deque] = defaultdict(deque)

    if "cluster_UT" not in old_df.columns or "cluster_CO" not in old_df.columns:
        raise KeyError("old_df must contain 'cluster_UT' and 'cluster_CO' for cluster-based migration.")

    # Only take rows that have any profile
    prof_mask = has_any_profile_row(old_df, tpc_cols)
    old_prof = old_df.loc[prof_mask].copy()

    # Normalize cluster types
    old_prof["cluster_UT"] = pd.to_numeric(old_prof["cluster_UT"], errors="coerce").astype("Int64")
    old_prof["cluster_CO"] = pd.to_numeric(old_prof["cluster_CO"], errors="coerce").astype("Int64")
    old_prof = old_prof.dropna(subset=["cluster_UT", "cluster_CO"]).copy()

    # Fill pool
    for _, r in old_prof.iterrows():
        cu = int(r["cluster_UT"])
        cc = int(r["cluster_CO"])
        for c in tpc_cols:
            v = r.get(c, pd.NA)
            if pd.isna(v):
                continue
            pool[(cu, cc, c)].append(v)
            if symmetric:
                pool[(cc, cu, c)].append(v)

    return pool


def migrate_direct_pair_cells(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    tpc_cols: list[str],
    key_cols: list[str],
) -> tuple[pd.DataFrame, int]:
    """
    Directly match (Model_UT, Kernel_ID_UT, Model_CO, Kernel_ID_CO) and fill only NaNs in new_df.
    Returns (new_df_after, num_cells_filled).
    """
    # Keep only necessary columns from old
    keep_cols = [c for c in key_cols if c in old_df.columns] + [c for c in tpc_cols if c in old_df.columns]
    old_small = old_df[keep_cols].copy()

    # Deduplicate old by key (keep last)
    old_small = old_small.drop_duplicates(subset=key_cols, keep="last")

    merged = new_df.merge(
        old_small,
        on=key_cols,
        how="left",
        suffixes=("", "__old"),
    )

    filled = 0
    for c in tpc_cols:
        cold = c + "__old"
        if cold not in merged.columns:
            continue
        mask = merged[c].isna() & merged[cold].notna()
        n = int(mask.sum())
        if n:
            merged.loc[mask, c] = merged.loc[mask, cold]
            filled += n
        merged.drop(columns=[cold], inplace=True)

    return merged, filled


def migrate_by_cluster_pool(
    new_df: pd.DataFrame,
    tpc_cols: list[str],
    pool: dict[tuple[int, int, str], deque],
    max_fill_per_cell: int | None = None,
) -> tuple[pd.DataFrame, int]:
    """
    Fill NaN TPC cells in new_df using values from pool[(cluster_UT, cluster_CO, tpc_col)].

    - Pops from the deque so values distribute across rows.
    - Does NOT overwrite existing non-NaN cells.
    - If max_fill_per_cell is set, we will not fill more than that many cells per (clusterpair,tpc_col)
      (useful if you want to avoid overusing a tiny pool).
    """
    if "cluster_UT" not in new_df.columns or "cluster_CO" not in new_df.columns:
        raise KeyError("new_df must contain 'cluster_UT' and 'cluster_CO' for cluster-based migration.")

    out = new_df.copy()

    out["cluster_UT"] = pd.to_numeric(out["cluster_UT"], errors="coerce").astype("Int64")
    out["cluster_CO"] = pd.to_numeric(out["cluster_CO"], errors="coerce").astype("Int64")

    # Track fills per bucket if max_fill_per_cell is enabled
    filled_counter: dict[tuple[int, int, str], int] = defaultdict(int)

    total_filled = 0

    # Iterate rows in a stable order
    for idx in out.index:
        cu = out.at[idx, "cluster_UT"]
        cc = out.at[idx, "cluster_CO"]
        if pd.isna(cu) or pd.isna(cc):
            continue
        cu_i = int(cu)
        cc_i = int(cc)

        for c in tpc_cols:
            if pd.notna(out.at[idx, c]):
                continue  # don't overwrite

            k = (cu_i, cc_i, c)
            if k not in pool:
                continue
            if not pool[k]:
                continue

            if max_fill_per_cell is not None and filled_counter[k] >= max_fill_per_cell:
                continue

            out.at[idx, c] = pool[k].popleft()
            filled_counter[k] += 1
            total_filled += 1

    return out, total_filled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_plan", type=Path, required=True, help="Old plan CSV that already has some TPC cells filled")
    ap.add_argument("--new_plan", type=Path, required=True, help="New plan CSV (up_to3) to receive migrated cells")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: <new_plan_stem>_migrated.csv next to new_plan)",
    )
    ap.add_argument(
        "--symmetric",
        action="store_true",
        help="Also allow (cluster_CO,cluster_UT) reuse (treat profiles as symmetric). Default off.",
    )
    ap.add_argument(
        "--max_fill_per_bucket",
        type=int,
        default=None,
        help="Optional cap on how many cells we fill per (clusterpair,tpc_col) from the pool.",
    )
    args = ap.parse_args()

    if args.out is None:
        args.out = args.new_plan.with_name(args.new_plan.stem + "_migrated.csv")

    old_df = pd.read_csv(args.old_plan)
    new_df = pd.read_csv(args.new_plan)

    # Identify TPC columns (union across both)
    tpc_old = find_tpc_cols(old_df)
    tpc_new = find_tpc_cols(new_df)
    tpc_cols = sorted(set(tpc_old) | set(tpc_new), key=lambda s: (int(re.findall(r"\d+", s)[0]), s))

    if not tpc_cols:
        raise RuntimeError("No TPC columns like '(1,23)' found in either plan CSV.")

    # Ensure both have all TPC cols
    old_df = ensure_cols(old_df, tpc_cols)
    new_df = ensure_cols(new_df, tpc_cols)

    # Normalize key columns for direct merge
    key_cols = ["Model_UT", "Kernel_ID_UT", "Model_CO", "Kernel_ID_CO"]
    for c in ["Model_UT", "Model_CO"]:
        norm_str_col(old_df, c)
        norm_str_col(new_df, c)
    for c in ["Kernel_ID_UT", "Kernel_ID_CO"]:
        norm_int_col(old_df, c)
        norm_int_col(new_df, c)

    # Drop rows with invalid IDs for direct merge pass
    old_df_direct = old_df.dropna(subset=key_cols).copy()
    new_df_direct = new_df.dropna(subset=key_cols).copy()
    # Convert Int64 to int for stable merge keys
    for c in ["Kernel_ID_UT", "Kernel_ID_CO"]:
        old_df_direct[c] = old_df_direct[c].astype(int)
        new_df_direct[c] = new_df_direct[c].astype(int)

    # PASS A: direct pair-key migration
    new_after_direct, filled_direct = migrate_direct_pair_cells(
        old_df_direct, new_df_direct, tpc_cols=tpc_cols, key_cols=key_cols
    )

    # If new_df had some rows with missing key cols, reattach them unchanged
    if len(new_df_direct) != len(new_df):
        keep_mask = new_df[key_cols].notna().all(axis=1)
        untouched = new_df.loc[~keep_mask].copy()
        new_after_direct = pd.concat([new_after_direct, untouched], ignore_index=True)

    # PASS B: cluster-pair based migration
    # Build pool from the *original old_df* (because cluster cols are there)
    if "cluster_UT" not in old_df.columns or "cluster_CO" not in old_df.columns:
        raise KeyError(
            "Your old plan CSV must include cluster_UT/cluster_CO columns for cluster-based migration.\n"
            "If it doesn't, regenerate old plan or add clusters before using this migration script."
        )
    if "cluster_UT" not in new_after_direct.columns or "cluster_CO" not in new_after_direct.columns:
        raise KeyError(
            "Your new plan CSV must include cluster_UT/cluster_CO columns for cluster-based migration.\n"
            "Your current plan generator includes these, so this usually means the input file is not the plan."
        )

    pool = build_cluster_pool(old_df, tpc_cols=tpc_cols, symmetric=args.symmetric)

    new_final, filled_cluster = migrate_by_cluster_pool(
        new_after_direct, tpc_cols=tpc_cols, pool=pool, max_fill_per_cell=args.max_fill_per_bucket
    )

    # Reorder columns: keep original new_plan order, but ensure all tpc cols present at end
    # (Your generator puts TPC cols at end; we keep that style.)
    new_order = [c for c in pd.read_csv(args.new_plan, nrows=0).columns if c in new_final.columns]
    # add any extra cols (rare)
    extras = [c for c in new_final.columns if c not in new_order]
    # ensure TPC cols come last
    non_tpc = [c for c in (new_order + extras) if c not in tpc_cols]
    new_final = new_final[non_tpc + tpc_cols]

    new_final.to_csv(args.out, index=False)

    # Summary
    prof_before = int(has_any_profile_row(new_df, tpc_cols).sum())
    prof_after = int(has_any_profile_row(new_final, tpc_cols).sum())

    print(f"[DONE] wrote: {args.out}")
    print(f"[TPC]  cols: {len(tpc_cols)}")
    print(f"[PASS A] direct pair-key filled cells: {filled_direct}")
    print(f"[PASS B] cluster-pair filled cells:    {filled_cluster}")
    print(f"[ROWS] profiled rows before: {prof_before} / {len(new_df)}")
    print(f"[ROWS] profiled rows after:  {prof_after} / {len(new_final)}")


if __name__ == "__main__":
    main()
