#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import re
import json
from copy import deepcopy
import shutil

# ---------- CONFIG ----------
PLAN_CSV      = Path("/home/zixi/orion_bu/benchmarking/A4000_profile_plan_all_cluster_pairs_up_to3.csv")

# Where to dump the per-(Model_UT, Model_CO) partition CSVs
PARTITION_DIR = Path("./partition_files")

# Where the *single-model* templates live (e.g. Vnet_8.json, Dnet_8.json, ...)
TEMPLATE_JSON_DIR = Path("/home/zixi/orion_bu/artifact_evaluation/FICKS/configs/cudnn_based/others")

# Where to dump the NEW per-(model_ut,model_co) JSON configs
OUT_JSON_DIR = Path("/home/zixi/orion_bu/artifact_evaluation/FICKS/configs/profile")

TOTAL_TPCS = 24  # used only for sanity checks (tpc1 + tpc2)

# --- filter out configs with too-small TPCs ---
ENFORCE_MIN_TPC = True   # set False to disable this filter
MIN_TPC         = 1      # require tpc1 >= MIN_TPC AND tpc2 >= MIN_TPC

# --- add a "no-mask" baseline config for each kernel pair ---
ADD_NO_MASK_BASELINE = True
NO_MASK_TPC_VALUE    = -1

# --- CLEAN OUTPUT DIRS (delete old files first) ---
CLEAN_OUTPUT_DIRS = True
CLEAN_PARTITION_DIR = True
CLEAN_OUT_JSON_DIR  = True

# Safety: only delete if path looks like what we expect
# (prevents accidental deletion if someone changes PARTITION_DIR/OUT_JSON_DIR)
def _safe_to_delete_dir(p: Path) -> bool:
    p = p.resolve()
    name = p.name
    # allow only these exact directory names by default
    return name in ("partition_files", "profile")

def _recreate_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

if CLEAN_OUTPUT_DIRS:
    if CLEAN_PARTITION_DIR:
        if not _safe_to_delete_dir(PARTITION_DIR):
            raise RuntimeError(f"Refuse to delete unexpected dir: {PARTITION_DIR.resolve()}")
        print(f"[CLEAN] removing {PARTITION_DIR.resolve()}")
        _recreate_dir(PARTITION_DIR)
    else:
        PARTITION_DIR.mkdir(parents=True, exist_ok=True)

    if CLEAN_OUT_JSON_DIR:
        if not _safe_to_delete_dir(OUT_JSON_DIR):
            raise RuntimeError(f"Refuse to delete unexpected dir: {OUT_JSON_DIR.resolve()}")
        print(f"[CLEAN] removing {OUT_JSON_DIR.resolve()}")
        _recreate_dir(OUT_JSON_DIR)
    else:
        OUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
else:
    PARTITION_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

# Map BASE model names (without _bzN) to template JSON names
MODEL_META = {
    "densenet201":   {"alias": "Dnet_8",  "template": "Dnet_8.json"},
    "mobilenetv2":   {"alias": "Mnet_32", "template": "Mnet_32.json"},
    "mobilenet_v2":  {"alias": "Mnet_32", "template": "Mnet_32.json"},  # allow both spellings (input side)
    "resnet101":     {"alias": "R1net_8", "template": "R1net_8.json"},
    "resnet152":     {"alias": "Rnet_8",  "template": "Rnet_8.json"},
    "vgg19":         {"alias": "Vnet_8",  "template": "Vnet_8.json"},
}

# num_iters = (#configs) + OFFSET
NUM_ITERS_OFFS = 12


# ---------- Helpers ----------
def sanitize_name(s: str) -> str:
    """Safe filename component from a model name."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")


def strip_bz(model: str) -> str:
    """'vgg19_bz8' -> 'vgg19' (only strips a trailing _bz<digits>)."""
    return re.sub(r"_bz\d+$", "", str(model))


def normalize_model_name(model: str) -> str:
    """
    Apply all naming normalizations AFTER stripping _bzN.
    Requirement: mobilenet_v2 -> mobilenetv2 so partition filenames match.
    """
    m = str(model)
    # if m == "mobilenet_v2":
    #     return "mobilenetv2"
    return m


def parse_tpc_col(col: str):
    """
    Parse a TPC column name like "(1,23)" -> (1, 23).
    """
    m = re.match(r"^\((\d+)\s*,\s*(\d+)\)$", col)
    if not m:
        raise ValueError(f"Column '{col}' is not a valid TPC column '(n1,n2)'")
    n1 = int(m.group(1))
    n2 = int(m.group(2))
    return n1, n2


# cache for loaded templates
_template_cache = {}


def load_single_template(model_name_base: str) -> dict:
    """
    Load the single-model JSON template for a given BASE model name
    (densenet201, vgg19, ...), and return ONE dict (first entry) as template.
    """
    if model_name_base in _template_cache:
        return deepcopy(_template_cache[model_name_base])

    meta = MODEL_META.get(model_name_base)
    if meta is None:
        raise KeyError(f"No MODEL_META entry for base model '{model_name_base}'")

    template_path = TEMPLATE_JSON_DIR / meta["template"]
    if not template_path.exists():
        raise FileNotFoundError(f"Template JSON not found: {template_path}")

    with template_path.open("r") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0 or not isinstance(data[0], dict):
        raise ValueError(f"Template JSON {template_path} is not a list of dicts")

    _template_cache[model_name_base] = data[0]  # cache first entry
    return deepcopy(_template_cache[model_name_base])


# ---------- Load plan ----------
if not PLAN_CSV.exists():
    raise FileNotFoundError(f"Plan CSV not found: {PLAN_CSV.resolve()}")

print(f"[LOAD] {PLAN_CSV}")
df = pd.read_csv(PLAN_CSV)

required_cols = ["Model_UT", "Model_CO", "Kernel_ID_UT", "Kernel_ID_CO"]
for c in required_cols:
    if c not in df.columns:
        raise KeyError(f"Missing required column '{c}' in plan CSV")

# Normalize basic types
df["Model_UT"] = df["Model_UT"].astype(str)
df["Model_CO"] = df["Model_CO"].astype(str)
df["Kernel_ID_UT"] = pd.to_numeric(df["Kernel_ID_UT"], errors="coerce")
df["Kernel_ID_CO"] = pd.to_numeric(df["Kernel_ID_CO"], errors="coerce")

df = df.dropna(subset=["Model_UT", "Model_CO", "Kernel_ID_UT", "Kernel_ID_CO"]).copy()
df["Kernel_ID_UT"] = df["Kernel_ID_UT"].astype(int)
df["Kernel_ID_CO"] = df["Kernel_ID_CO"].astype(int)

# ---- Strip _bzN, then normalize names (mobilenet_v2 -> mobilenetv2) ----
df["Model_UT"] = df["Model_UT"].map(strip_bz).map(normalize_model_name)
df["Model_CO"] = df["Model_CO"].map(strip_bz).map(normalize_model_name)

# ---------- Identify TPC columns ----------
tpc_cols = [
    c for c in df.columns
    if isinstance(c, str) and re.match(r"^\(\d+,\s*\d+\)$", c)
]
if not tpc_cols:
    raise RuntimeError("No TPC columns like '(1,23)' found in plan CSV.")
print(f"[INFO] Found {len(tpc_cols)} TPC columns: {tpc_cols[:5]}{' ...' if len(tpc_cols) > 5 else ''}")

# Pre-parse TPC column -> (tpc1, tpc2)
tpc_map = {}
for col in tpc_cols:
    t1, t2 = parse_tpc_col(col)
    if t1 + t2 != TOTAL_TPCS:
        print(f"[WARN] Column {col}: tpc1 + tpc2 = {t1 + t2} != {TOTAL_TPCS}")
    tpc_map[col] = (t1, t2)

# ---------- Collect all missing (row, TPC col) into one big list ----------
rows_for_partitions = []
n_missing_cells = 0
n_total_cells   = 0

for _, r in df.iterrows():
    m_ut = r["Model_UT"]
    m_co = r["Model_CO"]
    id1  = int(r["Kernel_ID_UT"])
    id2  = int(r["Kernel_ID_CO"])

    for col in tpc_cols:
        val = r[col]
        n_total_cells += 1
        if pd.isna(val):
            n_missing_cells += 1
            t1, t2 = tpc_map[col]
            rows_for_partitions.append({
                "Model_UT": m_ut,
                "Model_CO": m_co,
                "id1":      id1,
                "id2":      id2,
                "tpc1":     t1,
                "tpc2":     t2,
            })

print(f"[INFO] Checked {n_total_cells} TPC cells.")
print(f"[INFO] Missing (NaN) TPC cells: {n_missing_cells}")

if not rows_for_partitions:
    raise RuntimeError("No missing TPC cells found; nothing to partition.")

# Make DataFrame and dedup so we don’t profile the same config twice
df_missing = pd.DataFrame(rows_for_partitions).drop_duplicates(
    subset=["Model_UT", "Model_CO", "id1", "id2", "tpc1", "tpc2"]
)
print(f"[INFO] Unique missing configs: {len(df_missing)}")

# ---------- filter out too-small TPC configs ----------
if ENFORCE_MIN_TPC:
    before = len(df_missing)
    df_missing = df_missing[
        (df_missing["tpc1"] >= MIN_TPC) &
        (df_missing["tpc2"] >= MIN_TPC)
    ].copy()
    after = len(df_missing)
    print(
        f"[FILTER] ENFORCE_MIN_TPC=True, MIN_TPC={MIN_TPC}: "
        f"kept {after}/{before} (dropped {before - after})"
    )

if df_missing.empty:
    raise RuntimeError("After filtering, no configs remain that need profiling.")

# ---------- Write partition files + JSON per (Model_UT, Model_CO) ----------
n_files     = 0
n_rows_all  = 0
n_json      = 0

for (m_ut, m_co), grp in df_missing.groupby(["Model_UT", "Model_CO"]):
    if grp.empty:
        continue

    safe_ut = sanitize_name(m_ut)
    safe_co = sanitize_name(m_co)
    out_path = PARTITION_DIR / f"{safe_ut}_{safe_co}_partitions.csv"

    grp_out = grp[["id1", "id2", "tpc1", "tpc2"]].copy()

    # ---- Add one "no mask" row per unique kernel pair (id1,id2) ----
    if ADD_NO_MASK_BASELINE:
        uniq_pairs = grp_out[["id1", "id2"]].drop_duplicates()
        nomask = uniq_pairs.copy()
        nomask["tpc1"] = NO_MASK_TPC_VALUE
        nomask["tpc2"] = NO_MASK_TPC_VALUE
        grp_out = pd.concat([grp_out, nomask], ignore_index=True)

    # Final dedup + stable order (so iterations map predictably to rows)
    grp_out = grp_out.drop_duplicates(subset=["id1", "id2", "tpc1", "tpc2"]).copy()
    grp_out = grp_out.sort_values(by=["id1", "id2", "tpc1", "tpc2"], ascending=[True, True, True, True]).reset_index(drop=True)

    grp_out.to_csv(out_path, index=False)

    n_files += 1
    n_rows_all += len(grp_out)
    print(f"[WRITE] {out_path}  ({len(grp_out)} rows) for model pair ({m_ut}, {m_co})")

    # ---- Build corresponding JSON config ----
    if m_ut not in MODEL_META or m_co not in MODEL_META:
        print(f"[WARN] No template mapping for ({m_ut}, {m_co}); skipping JSON.")
        continue

    alias_ut = MODEL_META[m_ut]["alias"]
    alias_co = MODEL_META[m_co]["alias"]

    n_configs = len(grp_out)
    num_iters = int(n_configs + NUM_ITERS_OFFS)

    tmpl_ut = load_single_template(m_ut)
    tmpl_co = load_single_template(m_co)

    tmpl_ut["num_iters"] = num_iters
    tmpl_co["num_iters"] = num_iters

    out_json_name = f"{alias_ut}_{alias_co}.json"
    out_json_path = OUT_JSON_DIR / out_json_name

    with out_json_path.open("w") as f:
        json.dump([tmpl_ut, tmpl_co], f, indent=4)

    n_json += 1
    print(f"[JSON] {out_json_path}  (num_iters={num_iters}, configs={n_configs})")

print(f"\n[SUMMARY] wrote {n_files} partition CSV files under {PARTITION_DIR.resolve()} with {n_rows_all} total rows.")
print(f"[SUMMARY] wrote {n_json} per-pair JSON configs to {OUT_JSON_DIR.resolve()}")
print(f"[SUMMARY] ADD_NO_MASK_BASELINE={ADD_NO_MASK_BASELINE}, NO_MASK_TPC_VALUE={NO_MASK_TPC_VALUE}")
