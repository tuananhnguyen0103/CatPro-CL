"""
summarize_results.py — Tổng hợp kết quả ablation từ các file JSON
------------------------------------------------------------------
Usage:
    python scripts/summarize_results.py --results_dir ../results
    python scripts/summarize_results.py --results_dir ../results --output summary.csv
"""

import argparse
import json
import os
import csv
from pathlib import Path


ABLATION_DESC = {
    # CatPro-CL ablations
    "A1":         "SR-GNN (backbone)",
    "A2":         "+ category graph (TODO)",
    "A3":         "+ EMA proto (no CL)",
    "A4":         "+ EMA proto + L_proto",
    "A5":         "+ Fixed proto + L_proto",
    "A6":         "+ K-means proto + L_proto",
    "A7":         "CatPro-CL (FULL MODEL)",
    "A8":         "+ cold inference (e_cat)",
    # Non-neural baselines
    "Popularity":  "Popularity (non-neural)",
    "Content-KNN": "Content-KNN (non-neural)",
    # Neural baselines
    "GCE-GNN":    "GCE-GNN (global graph)",
    "CORE":       "CORE (consistent repr.)",
    "NCL-SBR":    "NCL-SBR (session K-means CL)",
    "CL4SRec":    "CL4SRec (Transformer + augCL)",
    "NirGNN":     "NirGNN (cold-transfer GNN)",
}

METRICS = ["HR@10", "HR@20", "MRR@10", "MRR@20",
           "Cold_HR@10", "Cold_HR@20", "Cold_MRR@10", "Cold_MRR@20"]


def load_results(results_dir: str) -> list[dict]:
    rows = []
    for fpath in sorted(Path(results_dir).glob("*.json")):
        with open(fpath) as f:
            data = json.load(f)
        test = data.get("test", {})
        row = {
            "dataset":  data.get("dataset", "?"),
            "ablation": data.get("ablation", "?"),
            "seed":     data.get("seed", "?"),
            "best_epoch": data.get("best_epoch", "?"),
            "desc":     ABLATION_DESC.get(data.get("ablation", ""), ""),
        }
        for m in METRICS:
            row[m] = test.get(m, None)
        rows.append(row)
    return rows


def fmt(v, pct=True) -> str:
    if v is None:
        return "—"
    if pct:
        return f"{v*100:.2f}"
    return str(v)


def print_table(rows: list[dict]) -> None:
    # Sort by ablation
    rows = sorted(rows, key=lambda r: (r["dataset"], r["ablation"]))

    print("\n" + "=" * 110)
    print(f"{'ABL':<4} {'Description':<30} {'HR@10':>7} {'HR@20':>7} "
          f"{'MRR@10':>7} {'MRR@20':>7} | "
          f"{'C_HR@10':>8} {'C_HR@20':>8} {'C_MRR@10':>9} {'C_MRR@20':>9} "
          f"{'Ep':>3} {'Seed':>4}")
    print("-" * 110)

    for r in rows:
        abl  = r["ablation"]
        desc = r["desc"]
        marker = " ← FULL" if abl == "A7" else ""
        print(
            f"{abl:<4} {desc:<30} "
            f"{fmt(r['HR@10']):>7} {fmt(r['HR@20']):>7} "
            f"{fmt(r['MRR@10']):>7} {fmt(r['MRR@20']):>7} | "
            f"{fmt(r['Cold_HR@10']):>8} {fmt(r['Cold_HR@20']):>8} "
            f"{fmt(r['Cold_MRR@10']):>9} {fmt(r['Cold_MRR@20']):>9} "
            f"{r['best_epoch']:>3} {r['seed']:>4}{marker}"
        )
    print("=" * 110)
    print("* Metrics in % (×100). C_ = Cold-only subset.")


def save_csv(rows: list[dict], output: str) -> None:
    fieldnames = ["dataset", "ablation", "desc", "seed", "best_epoch"] + METRICS
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"\nSaved CSV → {output}")


def save_markdown(rows: list[dict], output: str) -> None:
    rows = sorted(rows, key=lambda r: (r["dataset"], r["ablation"]))
    lines = []
    lines.append("| ABL | Description | HR@10 | HR@20 | MRR@10 | MRR@20 | C_HR@10 | C_HR@20 | C_MRR@10 | C_MRR@20 | Epoch | Seed |")
    lines.append("|-----|-------------|------:|------:|-------:|-------:|--------:|--------:|---------:|---------:|------:|-----:|")
    for r in rows:
        marker = " **←FULL**" if r["ablation"] == "A7" else ""
        lines.append(
            f"| {r['ablation']} | {r['desc']}{marker} "
            f"| {fmt(r['HR@10'])} | {fmt(r['HR@20'])} "
            f"| {fmt(r['MRR@10'])} | {fmt(r['MRR@20'])} "
            f"| {fmt(r['Cold_HR@10'])} | {fmt(r['Cold_HR@20'])} "
            f"| {fmt(r['Cold_MRR@10'])} | {fmt(r['Cold_MRR@20'])} "
            f"| {r['best_epoch']} | {r['seed']} |"
        )
    md_path = output.replace(".csv", ".md") if output.endswith(".csv") else output + ".md"
    with open(md_path, "w") as f:
        f.write("# CatPro-CL Ablation Results\n\n")
        f.write("Metrics in % (×100). C_ = Cold-only subset.\n\n")
        f.write("\n".join(lines))
    print(f"Saved Markdown → {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="../results", help="Thư mục chứa *.json")
    parser.add_argument("--output", default=None, help="Tên file output (vd: summary.csv)")
    args = parser.parse_args()

    rows = load_results(args.results_dir)
    if not rows:
        print(f"Không tìm thấy file JSON nào trong: {args.results_dir}")
        exit(1)

    print_table(rows)

    if args.output:
        save_csv(rows, args.output)
        save_markdown(rows, args.output)
    else:
        # Auto-save cạnh results_dir
        out = str(Path(args.results_dir) / "ablation_summary.csv")
        save_csv(rows, out)
        save_markdown(rows, out)
