"""
validate_paper_numbers.py
=========================
Compare paper numbers against JSON result files produced by train.py.
Update RESULTS_BASE below to point to your results directory.
Usage: run from repo root:

    python scripts/validate_paper_numbers.py

Output: bảng so sánh + danh sách lỗi cần sửa.
"""

import json
import os
import glob
import numpy as np
from collections import defaultdict

# ─── Config ───────────────────────────────────────────────────────────────────
# Update RESULTS_BASE to point to your results directory (contains {dataset}_A11_seed*.json)
RESULTS_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
SEEDS = {0, 1, 2, 3, 42}
TOLS = {"mean": 0.015, "std": 0.015}   # tolerance in % points

# ─── Paper numbers to verify (Table V — Main Results, MaxLen Full) ─────────────
# Format: (dataset_key, model_key): (Cold_HR20_mean, Cold_HR20_std, HR20_mean, HR20_std)
# All values in % (i.e., multiply raw by 100)
PAPER_TABLE_V = {
    # --- RetailRocket ---
    ("retailrocket", "A11"):     (41.39, 1.08, 31.33, 1.26),   # CatPro-CL
    ("retailrocket", "M2TRec"):  (28.17, 3.18, 19.33, 1.11),
    ("retailrocket", "DC2R"):    ( 0.00, 0.00, 57.30, 1.43),   # DC2R no cold
    ("retailrocket", "NirGNN"):  ( 0.00, 0.00, 58.35, 1.46),
    ("retailrocket", "GCE-GNN"): ( 0.00, 0.00, 57.25, 1.65),
    ("retailrocket", "CORE"):    ( 0.00, 0.00, 58.76, 0.26),
    ("retailrocket", "NCL-SBR"): ( 0.00, 0.00, 58.34, 0.71),
    ("retailrocket", "CL4SRec"): ( 0.00, 0.00, 51.00, 0.91),
    ("retailrocket", "CCFCRec"): ( 0.00, 0.00, 56.49, 1.10),
    ("retailrocket", "CLCRec"):  ( 0.00, 0.00, 57.08, 0.90),
    ("retailrocket", "LetItGo"): ( 0.00, 0.00, 52.94, 0.81),
    # --- Diginetica ---
    ("diginetica",  "A11"):      (40.10, 0.83, 13.42, 0.92),
    ("diginetica",  "M2TRec"):   (30.84, 1.56,  9.01, 0.43),
    ("diginetica",  "DC2R"):     ( 0.00, 0.00, 30.58, 0.46),
    ("diginetica",  "NirGNN"):   ( 0.00, 0.00, 31.57, 0.34),
    ("diginetica",  "GCE-GNN"):  ( 0.00, 0.00, 31.30, 0.41),
    ("diginetica",  "CORE"):     ( 0.00, 0.00, 33.21, 0.18),
    ("diginetica",  "NCL-SBR"):  ( 0.00, 0.00, 32.99, 0.27),
    ("diginetica",  "CL4SRec"):  ( 0.00, 0.00, 31.61, 0.27),
    ("diginetica",  "CCFCRec"):  ( 0.00, 0.00, 30.19, 0.28),
    ("diginetica",  "CLCRec"):   ( 0.00, 0.00, 31.71, 0.26),
    ("diginetica",  "LetItGo"):  ( 0.00, 0.00, 30.63, 0.39),
    # --- CellPhones ---
    ("cellphones",  "A11"):      ( 0.80, 0.09,  5.66, 0.22),
    ("cellphones",  "M2TRec"):   ( 0.37, 0.12,  4.28, 0.24),
    ("cellphones",  "DC2R"):     ( 0.00, 0.00,  6.85, 0.14),
    ("cellphones",  "NirGNN"):   ( 0.00, 0.00,  6.82, 0.13),
    ("cellphones",  "GCE-GNN"):  ( 0.00, 0.00, 10.37, 0.09),
    ("cellphones",  "CORE"):     ( 0.00, 0.00, 10.13, 0.05),
    ("cellphones",  "NCL-SBR"):  ( 0.00, 0.00,  6.75, 0.15),
    ("cellphones",  "CL4SRec"):  ( 0.00, 0.00,  7.86, 0.11),
    ("cellphones",  "CCFCRec"):  ( 0.00, 0.00,  5.10, 0.09),
    ("cellphones",  "CLCRec"):   ( 0.00, 0.00,  5.16, 0.12),   # paper says 0.16? or 0.13?
    ("cellphones",  "LetItGo"):  ( 0.00, 0.00,  8.40, 0.13),
    # --- Yoochoose ---
    ("yoochoose",   "A11"):      ( 7.15, 2.44, 36.98, 2.58),
    ("yoochoose",   "M2TRec"):   ( 1.25, 0.78, 28.00, 3.19),
    ("yoochoose",   "DC2R"):     ( 0.00, 0.00, 48.05, 0.85),
    ("yoochoose",   "NirGNN"):   ( 0.00, 0.00, 47.58, 0.48),
    ("yoochoose",   "GCE-GNN"):  ( 0.00, 0.00, 47.31, 0.41),
    ("yoochoose",   "CORE"):     ( 0.00, 0.00, 53.57, 0.16),
    ("yoochoose",   "NCL-SBR"):  ( 0.00, 0.00, 52.37, 0.24),
    ("yoochoose",   "CL4SRec"):  ( 0.00, 0.00, 45.78, 0.39),
    ("yoochoose",   "CCFCRec"):  ( 0.00, 0.00, 42.23, 0.53),
    ("yoochoose",   "CLCRec"):   ( 0.00, 0.00, 51.56, 0.27),
    ("yoochoose",   "LetItGo"):  ( 0.00, 0.00, 44.94, 0.51),
}

# ─── Dataset / model name mapping between result filenames and paper keys ──────────
DS_MAP = {
    "retailrocket": "retailrocket",
    "diginetica":   "diginetica",
    "cellphones":   "cellphones",
    "yoochoose":    "yoochoose",
}
MODEL_MAP = {
    "A11":     "A11",
    "M2TRec":  "M2TRec",
    "DC2R":    "DC2R",
    "NirGNN":  "NirGNN",
    "GCE-GNN": "GCE-GNN",
    "GCEGNN":  "GCE-GNN",   # alternate spelling in filenames
    "CORE":    "CORE",
    "NCL-SBR": "NCL-SBR",
    "NCLSBR":  "NCL-SBR",
    "CL4SRec": "CL4SRec",
    "CCFCRec": "CCFCRec",
    "CLCRec":  "CLCRec",
    "LetItGo": "LetItGo",
}

# ─── Load all results JSON files ─────────────────────────────────────────────
def load_results():
    """Returns dict: (ds_key, model_key) -> list of (seed, HR20, ColdHR20)"""
    results = defaultdict(list)
    pattern = os.path.join(RESULTS_BASE, "*", "fullen", "*", "*.json")
    for fpath in glob.glob(pattern):
        try:
            d = json.load(open(fpath))
            ds_raw  = d.get("dataset", "")
            ablation = d.get("ablation", "")
            seed    = d.get("seed", -1)
            test    = d.get("test", {})
            hr20    = test.get("HR@20", 0) * 100
            cold20  = test.get("Cold_HR@20", 0) * 100

            # Map dataset
            ds_key = None
            for k in DS_MAP:
                if k in ds_raw:
                    ds_key = DS_MAP[k]
                    break
            model_key = MODEL_MAP.get(ablation)

            if ds_key and model_key and seed in SEEDS:
                results[(ds_key, model_key)].append((seed, hr20, cold20))
        except Exception as e:
            pass
    return results

# ─── Aggregate ─────────────────────────────────────────────────────────────────
def aggregate(seed_results):
    """seed_results: list of (seed, hr20, cold20)"""
    if not seed_results:
        return None
    hrs   = [r[1] for r in seed_results]
    colds = [r[2] for r in seed_results]
    return (
        np.mean(colds), np.std(colds, ddof=1),
        np.mean(hrs),   np.std(hrs,   ddof=1),
        len(seed_results)
    )

# ─── Main validation ───────────────────────────────────────────────────────────
def main():
    print("=" * 90)
    print("CatPro-CL  —  Paper vs JSON Validation (Table V, MaxLen Full, Cold HR@20 & HR@20)")
    print("=" * 90)
    print(f"{'Key':<35} {'Paper (Cold,Ovr)':<28} {'JSON (Cold,Ovr)':<28}  {'n':>2}  Status")
    print("-" * 90)

    data = load_results()
    errors = []
    warnings = []
    missing = []

    for key in sorted(PAPER_TABLE_V.keys()):
        ds, model = key
        pc_cold, ps_cold, pc_ovr, ps_ovr = PAPER_TABLE_V[key]

        seed_res = data.get(key)
        if not seed_res:
            missing.append(key)
            label = f"({ds},{model})"
            print(f"  {label:<33} {pc_cold:5.2f}±{ps_cold:.2f} / {pc_ovr:5.2f}±{ps_ovr:.2f}   "
                  f"{'???':>26}   {'–':>2}  ⚠ MISSING")
            continue

        agg = aggregate(seed_res)
        jc_cold, js_cold, jc_ovr, js_ovr, n = agg

        issues = []
        if abs(pc_cold - jc_cold) > TOLS["mean"]:
            issues.append(f"Cold mean: paper={pc_cold:.2f} json={jc_cold:.2f} Δ={pc_cold-jc_cold:+.2f}")
        if abs(ps_cold - js_cold) > TOLS["std"] and (pc_cold > 0 or ps_cold > 0):
            issues.append(f"Cold std:  paper={ps_cold:.2f} json={js_cold:.2f} Δ={ps_cold-js_cold:+.2f}")
        if abs(pc_ovr - jc_ovr) > TOLS["mean"]:
            issues.append(f"Ovr mean:  paper={pc_ovr:.2f} json={jc_ovr:.2f} Δ={pc_ovr-jc_ovr:+.2f}")
        if abs(ps_ovr - js_ovr) > TOLS["std"]:
            issues.append(f"Ovr std:   paper={ps_ovr:.2f} json={js_ovr:.2f} Δ={ps_ovr-js_ovr:+.2f}")

        status = "✓ OK" if not issues else ("❌ MISMATCH" if any("mean" in i for i in issues) else "⚠ std diff")
        label = f"({ds},{model})"
        print(f"  {label:<33} "
              f"{pc_cold:5.2f}±{ps_cold:.2f}/{pc_ovr:5.2f}±{ps_ovr:.2f}  "
              f"{jc_cold:5.2f}±{js_cold:.2f}/{jc_ovr:5.2f}±{js_ovr:.2f}  "
              f"  {n:>2}  {status}")
        if issues:
            errors.append((key, issues))
            for iss in issues:
                print(f"      → {iss}")

    # Summary
    print("\n" + "=" * 90)
    print(f"SUMMARY: {len(PAPER_TABLE_V)} entries checked")
    print(f"  Missing from JSON : {len(missing)}")
    print(f"  Mismatches found  : {len(errors)}")
    if errors:
        print("\n  ENTRIES NEEDING CORRECTION IN PAPER:")
        for key, issues in errors:
            print(f"    ({key[0]}, {key[1]})")
            for iss in issues:
                print(f"      {iss}")
    if missing:
        print("\n  JSON FILES NOT FOUND (need to sync from server):")
        for k in missing:
            print(f"    ({k[0]}, {k[1]})")
    print("=" * 90)

if __name__ == "__main__":
    main()
