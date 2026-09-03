"""
collect_ablation_results.py
Thu thập kết quả từ ~/results_ablation/ và in bảng tóm tắt mean±std

Usage:
    python scripts/collect_ablation_results.py
    python scripts/collect_ablation_results.py --results_dir ~/results_ablation

Output:
    Bảng console + ablation_components_summary.csv
"""

import json
import os
import glob
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

SEEDS = [42, 0, 1, 2, 3]
DATASETS = ["retailrocket", "diginetica", "cellphones"]
VARIANTS = ["no_infonce", "no_psm", "no_both"]
METRICS = ["HR@10_overall", "HR@20_overall", "HR@10_cold", "HR@20_cold",
           "MRR@10_overall", "MRR@20_overall", "MRR@10_cold", "MRR@20_cold"]


def load_result_file(path):
    """Load một JSON result file, trả về dict hoặc None nếu lỗi."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [WARN] Cannot read {path}: {e}")
        return None


def extract_metrics(result):
    """
    Trích các metric từ result dict.
    Hỗ trợ cả format flat và nested (test_metrics / eval_metrics).
    """
    metrics = {}

    # Thử lấy từ test_metrics hoặc từ root
    src = result.get("test_metrics", result.get("eval_metrics", result))

    for k in ["HR@10", "HR@20", "MRR@10", "MRR@20"]:
        # Overall
        for overall_key in [f"{k}_overall", f"overall_{k}", f"overall/{k}"]:
            if overall_key in src:
                metrics[f"{k}_overall"] = float(src[overall_key])
                break
        if f"{k}_overall" not in metrics and k in src:
            metrics[f"{k}_overall"] = float(src[k])

        # Cold
        for cold_key in [f"{k}_cold", f"cold_{k}", f"cold/{k}"]:
            if cold_key in src:
                metrics[f"{k}_cold"] = float(src[cold_key])
                break

    return metrics


def collect(results_dir):
    results_dir = Path(results_dir).expanduser()

    # data[dataset][variant][seed] = metric_dict
    data = defaultdict(lambda: defaultdict(dict))

    for ds in DATASETS:
        for variant in VARIANTS:
            for seed in SEEDS:
                # Thử nhiều naming convention
                patterns = [
                    results_dir / ds / variant / f"*seed{seed}*.json",
                    results_dir / ds / variant / f"*_{seed}.json",
                    results_dir / ds / variant / f"seed_{seed}" / "*.json",
                    results_dir / ds / variant / f"seed{seed}.json",
                ]
                found = None
                for pat in patterns:
                    matches = glob.glob(str(pat))
                    if matches:
                        found = matches[0]
                        break

                if found is None:
                    print(f"  [MISSING] {ds}/{variant}/seed{seed}")
                    continue

                result = load_result_file(found)
                if result is None:
                    continue

                metrics = extract_metrics(result)
                if not metrics:
                    print(f"  [EMPTY]   {ds}/{variant}/seed{seed} — no metrics extracted")
                    continue

                data[ds][variant][seed] = metrics
                print(f"  [OK]      {ds}/{variant}/seed{seed}: HR@20_overall={metrics.get('HR@20_overall', 'N/A'):.2f}")

    return data


def summarize(data):
    """Tính mean±std cho từng (ds, variant, metric)."""
    rows = []
    for ds in DATASETS:
        for variant in VARIANTS:
            if variant not in data[ds]:
                continue
            seed_results = data[ds][variant]
            if not seed_results:
                continue

            row = {"dataset": ds, "variant": variant}
            all_metrics = list(seed_results.values())

            for metric in METRICS:
                vals = [m[metric] for m in all_metrics if metric in m]
                if vals:
                    row[f"{metric}_mean"] = np.mean(vals)
                    row[f"{metric}_std"] = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                    row[f"{metric}_n"] = len(vals)
                else:
                    row[f"{metric}_mean"] = None
                    row[f"{metric}_std"] = None
                    row[f"{metric}_n"] = 0

            rows.append(row)
    return rows


def print_table(rows):
    print("\n" + "═" * 100)
    print("  ABLATION COMPONENT RESULTS  (mean ± std over seeds)")
    print("═" * 100)
    header = f"{'Dataset':<15} {'Variant':<15} {'HR@20-OV':>12} {'HR@20-Cold':>12} {'MRR@20-OV':>12} {'MRR@20-Cold':>12}  {'n':>3}"
    print(header)
    print("─" * 100)

    for row in rows:
        def fmt(m):
            mean = row.get(f"{m}_mean")
            std = row.get(f"{m}_std")
            n = row.get(f"{m}_n", 0)
            if mean is None:
                return "    N/A     "
            if n < 2 or std is None:
                return f"{mean:8.2f}      "
            return f"{mean:6.2f}±{std:.2f}"

        n = row.get("HR@20_overall_n", 0)
        print(f"{row['dataset']:<15} {row['variant']:<15} "
              f"{fmt('HR@20_overall'):>12} "
              f"{fmt('HR@20_cold'):>12} "
              f"{fmt('MRR@20_overall'):>12} "
              f"{fmt('MRR@20_cold'):>12}  {n:>3}")

    print("═" * 100)
    print("\nVariants:")
    print("  no_infonce : λ=0.0, ρ=0.05  → PSM only (no InfoNCE)")
    print("  no_psm     : λ=0.5, ρ=0.0   → InfoNCE only (no PSM)")
    print("  no_both    : λ=0.0, ρ=0.0   → prototype bank + cold inference only")
    print("  [A11 full] : λ=0.5, ρ=0.05  → full CatPro-CL (from v6, not re-run)")


def save_csv(rows, out_path):
    import csv
    if not rows:
        return

    fieldnames = ["dataset", "variant"] + [
        f"{m}_{s}" for m in METRICS for s in ["mean", "std", "n"]
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="~/results_ablation",
                        help="Root dir of ablation results")
    parser.add_argument("--out_csv", default="ablation_components_summary.csv")
    args = parser.parse_args()

    print(f"Scanning: {Path(args.results_dir).expanduser()}")
    data = collect(args.results_dir)
    rows = summarize(data)
    print_table(rows)
    save_csv(rows, args.out_csv)


if __name__ == "__main__":
    main()
