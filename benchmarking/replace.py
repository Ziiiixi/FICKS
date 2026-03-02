#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from collections import defaultdict

# ---------- CONFIG ----------
PLAN_CSV   = Path("A4000_profile_plan_all_cluster_pairs_up_to3.csv")
KERNEL_CSV = Path("A4000_kernels_profiles_all_models.csv")
OUT_PLAN   = Path("A4000_profile_plan_all_cluster_pairs_up_to3_no_vgg.csv")

VGG_NAME    = "vgg19"   # model name to avoid
MAX_VGG_ID  = 55       # vgg kernels are 0..295 only


def main():
    if not PLAN_CSV.exists():
        raise FileNotFoundError(f"Plan CSV not found: {PLAN_CSV.resolve()}")
    if not KERNEL_CSV.exists():
        raise FileNotFoundError(f"Kernel CSV not found: {KERNEL_CSV.resolve()}")

    print(f"[LOAD] {PLAN_CSV}")
    plan_df = pd.read_csv(PLAN_CSV)

    print(f"[LOAD] {KERNEL_CSV}")
    kern_df = pd.read_csv(KERNEL_CSV)

    # ---- basic sanity ----
    required_plan_cols = [
        "Model_UT", "Kernel_ID_UT", "Kernel_Name_UT",
        "Model_CO", "Kernel_ID_CO", "Kernel_Name_CO",
        "cluster_UT", "cluster_CO",
        "kernel0_id", "kernel1_id",
    ]
    for c in required_plan_cols:
        if c not in plan_df.columns:
            raise KeyError(f"Missing column '{c}' in plan CSV")

    required_kern_cols = ["Model", "Kernel_ID", "Kernel_Name", "cluster"]
    for c in required_kern_cols:
        if c not in kern_df.columns:
            raise KeyError(f"Missing column '{c}' in kernel CSV")

    # Ensure types are consistent
    plan_df["cluster_UT"]   = plan_df["cluster_UT"].astype(int)
    plan_df["cluster_CO"]   = plan_df["cluster_CO"].astype(int)
    plan_df["Kernel_ID_UT"] = plan_df["Kernel_ID_UT"].astype(int)
    plan_df["Kernel_ID_CO"] = plan_df["Kernel_ID_CO"].astype(int)

    kern_df["cluster"]   = kern_df["cluster"].astype(int)
    kern_df["Kernel_ID"] = kern_df["Kernel_ID"].astype(int)

    # ---- detect TPC columns (for "has / no profile" test) ----
    tpc_cols = [c for c in plan_df.columns if c.startswith("(") and c.endswith(")")]
    print(f"[INFO] Detected {len(tpc_cols)} TPC columns in plan: {tpc_cols[:5]}{' ...' if len(tpc_cols) > 5 else ''}")

    # ---- build candidate pools ----
    kern_df["is_vgg"] = kern_df["Model"] == VGG_NAME

    # non-VGG candidates per cluster
    cluster_to_candidates = defaultdict(list)  # cluster -> list of (model, kernel_id)
    for _, row in kern_df[~kern_df["is_vgg"]].iterrows():
        cluster_to_candidates[int(row["cluster"])].append(
            (row["Model"], int(row["Kernel_ID"]))
        )

    # <<< NEW: valid VGG candidates per cluster (id <= MAX_VGG_ID)
    cluster_to_vgg_valid = defaultdict(list)   # cluster -> list of vgg kernel_ids
    mask_vgg_valid = (kern_df["is_vgg"]) & (kern_df["Kernel_ID"] <= MAX_VGG_ID)
    for _, row in kern_df[mask_vgg_valid].iterrows():
        cluster_to_vgg_valid[int(row["cluster"])].append(int(row["Kernel_ID"]))

    # lookup for names etc.
    kern_lookup = {
        (row["Model"], int(row["Kernel_ID"])): row
        for _, row in kern_df.iterrows()
    }

    # ---- stats before ----
    plan_df["is_vgg"] = (plan_df["Model_UT"] == VGG_NAME) | (plan_df["Model_CO"] == VGG_NAME)
    total_rows = len(plan_df)
    vgg_rows_before = int(plan_df["is_vgg"].sum())
    print(f"[INFO] total rows: {total_rows}")
    print(f"[INFO] rows involving {VGG_NAME} before: {vgg_rows_before}")

    # ---- replacement logic ----
    new_plan = plan_df.copy()
    changed_rows = 0
    still_vgg_rows = 0

    def pick_candidate(cluster_id):
        """Pick a non-VGG kernel for a cluster (simple: first in list)."""
        cand_list = cluster_to_candidates.get(int(cluster_id), [])
        if not cand_list:
            return None
        return cand_list[0]

    # <<< NEW
    def pick_vgg_candidate(cluster_id):
        """Pick a VGG kernel with id <= MAX_VGG_ID in this cluster."""
        ids = cluster_to_vgg_valid.get(int(cluster_id), [])
        if not ids:
            return None
        # simple strategy: first one
        return ids[0]

    vgg_indices = plan_df.index[plan_df["is_vgg"]].tolist()

    for idx in vgg_indices:
        r = plan_df.loc[idx]

        clu_ut = int(r["cluster_UT"])
        clu_co = int(r["cluster_CO"])
        m_ut   = r["Model_UT"]
        m_co   = r["Model_CO"]

        new_m_ut, new_m_co = m_ut, m_co
        new_id_ut, new_id_co = int(r["Kernel_ID_UT"]), int(r["Kernel_ID_CO"])
        new_name_ut, new_name_co = r["Kernel_Name_UT"], r["Kernel_Name_CO"]

        can_fix = True

        # replace UT if it's VGG with non-VGG
        if m_ut == VGG_NAME:
            cand = pick_candidate(clu_ut)
            if cand is None:
                can_fix = False
            else:
                cm, kid = cand
                new_m_ut = cm
                new_id_ut = kid
                krow = kern_lookup[(cm, kid)]
                new_name_ut = krow["Kernel_Name"]

        # replace CO if it's VGG with non-VGG
        if m_co == VGG_NAME:
            cand = pick_candidate(clu_co)
            if cand is None:
                can_fix = False
            else:
                cm, kid = cand
                new_m_co = cm
                new_id_co = kid
                krow = kern_lookup[(cm, kid)]
                new_name_co = krow["Kernel_Name"]

        # only apply if we managed to remove VGG from BOTH sides that had it
        if can_fix and (new_m_ut != VGG_NAME) and (new_m_co != VGG_NAME):
            new_plan.loc[idx, "Model_UT"] = new_m_ut
            new_plan.loc[idx, "Kernel_ID_UT"] = new_id_ut
            new_plan.loc[idx, "Kernel_Name_UT"] = new_name_ut
            new_plan.loc[idx, "kernel0_id"] = f"{new_m_ut}:{new_id_ut}"

            new_plan.loc[idx, "Model_CO"] = new_m_co
            new_plan.loc[idx, "Kernel_ID_CO"] = new_id_co
            new_plan.loc[idx, "Kernel_Name_CO"] = new_name_co
            new_plan.loc[idx, "kernel1_id"] = f"{new_m_co}:{new_id_co}"

            changed_rows += 1
        else:
            # we keep this row as-is (it still contains vgg)
            still_vgg_rows += 1

            # ---- only touch rows WITHOUT profiles ----
            has_profile = False
            if tpc_cols:
                row_tpcs = r[tpc_cols]
                has_profile = row_tpcs.notna().any()

            if not has_profile:
                # try to "fix" invalid VGG IDs by picking another VGG kernel in same cluster
                # UT side
                if m_ut == VGG_NAME and int(r["Kernel_ID_UT"]) > MAX_VGG_ID:
                    new_kid = pick_vgg_candidate(clu_ut)
                    if new_kid is None:
                        raise ValueError(
                            f"Plan row {idx}: VGG UT kernel_id {r['Kernel_ID_UT']} > {MAX_VGG_ID}, "
                            f"and no alternative VGG kernel (<= {MAX_VGG_ID}) in cluster_UT={clu_ut}"
                        )
                    krow = kern_lookup[(VGG_NAME, new_kid)]
                    new_plan.loc[idx, "Kernel_ID_UT"] = new_kid
                    new_plan.loc[idx, "Kernel_Name_UT"] = krow["Kernel_Name"]
                    new_plan.loc[idx, "kernel0_id"] = f"{VGG_NAME}:{new_kid}"

                # CO side
                if m_co == VGG_NAME and int(r["Kernel_ID_CO"]) > MAX_VGG_ID:
                    new_kid = pick_vgg_candidate(clu_co)
                    if new_kid is None:
                        raise ValueError(
                            f"Plan row {idx}: VGG CO kernel_id {r['Kernel_ID_CO']} > {MAX_VGG_ID}, "
                            f"and no alternative VGG kernel (<= {MAX_VGG_ID}) in cluster_CO={clu_co}"
                        )
                    krow = kern_lookup[(VGG_NAME, new_kid)]
                    new_plan.loc[idx, "Kernel_ID_CO"] = new_kid
                    new_plan.loc[idx, "Kernel_Name_CO"] = krow["Kernel_Name"]
                    new_plan.loc[idx, "kernel1_id"] = f"{VGG_NAME}:{new_kid}"

    # ---- stats after ----
    new_plan["is_vgg_after"] = (new_plan["Model_UT"] == VGG_NAME) | (new_plan["Model_CO"] == VGG_NAME)
    vgg_rows_after = int(new_plan["is_vgg_after"].sum())

    print(f"[INFO] rows involving {VGG_NAME} successfully replaced (to non-VGG): {changed_rows}")
    print(f"[INFO] rows still involving {VGG_NAME}: {still_vgg_rows}")
    print(f"[INFO] rows involving {VGG_NAME} after: {vgg_rows_after}")

    print(f"[WRITE] {OUT_PLAN}")
    new_plan.drop(columns=["is_vgg", "is_vgg_after"], errors="ignore").to_csv(OUT_PLAN, index=False)


if __name__ == "__main__":
    main()
