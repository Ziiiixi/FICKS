from pathlib import Path
import pandas as pd
import re

# ====== EDIT THESE PATHS ======
PLAN_CSV      = Path("A4000_profile_plan_all_cluster_pairs_up_to3.csv")
PAIR_FLAT_CSV = Path("A4000_pair_profiles_flat.csv")

# If you want "generate the same name csv file" meaning overwrite PLAN_CSV:
OUT_CSV       = PLAN_CSV
# ==============================

TOTAL_TPCS = 24

# Add non masking profile column
NON_MASK_COL = "(-1,-1)"

# For A4000: we want columns
# (-1,-1), (1,23), (2,22), (4,20), (6,18), ..., (22,2)
ALL_TPC_COLS = [NON_MASK_COL, f"(1,{TOTAL_TPCS - 1})"] + [
    f"({ut},{TOTAL_TPCS - ut})"
    for ut in range(2, TOTAL_TPCS, 2)   # 2,4,6,...,22
]


def _canon_tpc_col_name(col: str) -> str:
    """
    Canonicalize '( n1 , n2 )' -> '(n1,n2)' and allow negatives.
    If not a TPC-style column, return unchanged.
    """
    m = re.match(r"^\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)$", str(col))
    if not m:
        return col
    a = int(m.group(1))
    b = int(m.group(2))
    return f"({a},{b})"


def _parse_tpc_col(col: str):
    """
    Parse TPC col name '(n1,n2)' -> (n1, n2) as ints.
    Supports negatives (for (-1,-1)).
    """
    m = re.match(r"^\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)$", col)
    if not m:
        raise ValueError(f"Not a TPC column: {col}")
    return int(m.group(1)), int(m.group(2))


def _tpc_sort_key(col: str):
    """
    Keep (-1,-1) first, then (1,23), (2,22), (4,20), ...
    """
    a, b = _parse_tpc_col(col)
    if a == -1 and b == -1:
        return (-10**9, 0)
    return (a, -b)


def merge_flat_into_plan():
    print(f"Loading plan: {PLAN_CSV}")
    plan_df = pd.read_csv(PLAN_CSV)

    # Canonicalize TPC column names in plan so "(-1, -1)" becomes "(-1,-1)"
    plan_df.columns = [_canon_tpc_col_name(c) for c in plan_df.columns]

    # ---- handle missing flat CSV: just report unprofiled rows and exit ----
    if not PAIR_FLAT_CSV.exists():
        print(f"[WARN] Flat pair profile CSV not found: {PAIR_FLAT_CSV}")

        tpc_cols_plan = [c for c in ALL_TPC_COLS if c in plan_df.columns]
        if not tpc_cols_plan:
            unprofiled_rows = len(plan_df)
            print(f"[INFO] No TPC columns found in plan. "
                  f"All {unprofiled_rows} rows are unprofiled.")
        else:
            has_profile = ~plan_df[tpc_cols_plan].isna().all(axis=1)
            unprofiled_rows = (~has_profile).sum()
            print(f"[INFO] Rows with ANY profile in plan: {has_profile.sum()} / {len(plan_df)}")
            print(f"[INFO] Rows still unprofiled in plan: {unprofiled_rows} / {len(plan_df)}")
        return

    print(f"Loading flat pair profiles: {PAIR_FLAT_CSV}")
    flat_df = pd.read_csv(PAIR_FLAT_CSV)

    # Canonicalize TPC column names in flat so "(-1, -1)" becomes "(-1,-1)"
    flat_df.columns = [_canon_tpc_col_name(c) for c in flat_df.columns]

    print("Expected TPC cols:", ALL_TPC_COLS)

    # ---- 0. Ensure all TPC columns exist in BOTH dfs (create empty if missing) ----
    for col in ALL_TPC_COLS:
        if col not in plan_df.columns:
            plan_df[col] = pd.NA
        if col not in flat_df.columns:
            flat_df[col] = pd.NA

    # ---- 1. TPC columns: only use our canonical list ----
    tpc_cols_plan_raw = [c for c in ALL_TPC_COLS if c in plan_df.columns]
    tpc_cols_flat_raw = [c for c in ALL_TPC_COLS if c in flat_df.columns]

    # Sort for safety, keeping (-1,-1) first
    tpc_cols_plan = sorted(tpc_cols_plan_raw, key=_tpc_sort_key)
    tpc_cols_flat = sorted(tpc_cols_flat_raw, key=_tpc_sort_key)

    # Common TPC cols in the SAME sorted order as in the plan
    common_tpc_cols = [c for c in tpc_cols_plan if c in tpc_cols_flat]

    print(f"TPC columns in plan (raw): {len(tpc_cols_plan_raw)}")
    print(f"TPC columns in flat: {len(tpc_cols_flat_raw)}")
    print(f"Common TPC columns to merge (sorted): {len(common_tpc_cols)}")
    print(f"Does common include {NON_MASK_COL}? {NON_MASK_COL in common_tpc_cols}")

    # --- baseline: which rows already had any profile BEFORE merge ---
    had_profile_before = ~plan_df[tpc_cols_plan].isna().all(axis=1)
    print(f"Rows with ANY profile before merge: {had_profile_before.sum()} / {len(plan_df)}")

    # ---- 2. Normalize key columns so merge actually matches ----
    merge_keys = ["Model_UT", "Kernel_ID_UT", "Model_CO", "Kernel_ID_CO"]

    flat_key_map = {
        "model_ut": "Model_UT",
        "kernel_ut_id": "Kernel_ID_UT",
        "model_co": "Model_CO",
        "kernel_co_id": "Kernel_ID_CO",
    }

    # Rename flat columns so keys line up exactly with plan
    flat_df = flat_df.rename(columns=flat_key_map)

    # Normalize key types/whitespace
    for col in merge_keys:
        plan_df[col] = plan_df[col].astype(str).str.strip()
        flat_df[col] = flat_df[col].astype(str).str.strip()

    # ---- 3. Deduplicate flat: one row per pair key ----
    flat_cols_for_merge = merge_keys + common_tpc_cols
    flat_dedup = (
        flat_df[flat_cols_for_merge]
        .sort_values(merge_keys)
        .drop_duplicates(merge_keys, keep="last")
    )

    print(f"Rows in flat (raw): {len(flat_df)}")
    print(f"Unique pair keys in flat (deduped): {len(flat_dedup)}")

    # ---- 4. Merge: plan (base) + flat_dedup (profiles) ----
    merged = plan_df.merge(
        flat_dedup,
        on=merge_keys,
        how="left",
        suffixes=("", "_flat"),
        indicator=True,
    )

    has_flat_match = merged["_merge"] == "both"
    matched_pairs = merged.loc[has_flat_match, merge_keys].drop_duplicates()
    unmatched_pairs = merged.loc[~has_flat_match, merge_keys].drop_duplicates()

    print(f"\nPlan pairs that HAVE profile in flat: {len(matched_pairs)}")
    print(f"Plan pairs with NO profile in flat: {len(unmatched_pairs)}")

    # ---- 5. Fill only NaN cells in plan’s TPC columns with values from flat ----
    total_filled = 0
    for col in common_tpc_cols:
        flat_col = col + "_flat"
        if flat_col not in merged.columns:
            continue

        before_nan = merged[col].isna().sum()
        mask = merged[col].isna() & merged[flat_col].notna()
        filled_here = mask.sum()
        merged.loc[mask, col] = merged.loc[mask, flat_col]
        total_filled += filled_here

        print(
            f"{col}: filled {filled_here} cells "
            f"(NaN before = {before_nan}, after = {merged[col].isna().sum()})"
        )

        merged.drop(columns=[flat_col], inplace=True)

    # drop merge indicator
    merged.drop(columns=["_merge"], inplace=True)

    # --- rows that have any profile AFTER merge ---
    has_profile_after = ~merged[tpc_cols_plan].isna().all(axis=1)
    print(f"\nRows with ANY profile after merge: {has_profile_after.sum()} / {len(merged)}")

    newly_profiled_mask = (~had_profile_before) & has_profile_after
    newly_profiled_pairs = merged.loc[newly_profiled_mask, merge_keys].drop_duplicates()
    print(f"Rows newly profiled by flat: {newly_profiled_mask.sum()}")
    print(f"Unique pairs newly profiled by flat: {len(newly_profiled_pairs)}")

    print(f"\nTotal filled cells across all TPC columns: {total_filled}")

    # ---- 6. Reorder columns: non-TPC first, then sorted TPC cols ----
    non_tpc_cols = [c for c in merged.columns if c not in tpc_cols_plan]
    merged = merged[non_tpc_cols + tpc_cols_plan]

    print(f"Writing merged plan to: {OUT_CSV} (overwriting plan file)")
    merged.to_csv(OUT_CSV, index=False)
    print("Done.")


if __name__ == "__main__":
    merge_flat_into_plan()
