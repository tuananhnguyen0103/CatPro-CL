"""
plot_cold_overall_tradeoff_landscape.py
========================================
Tái tạo Figure: "Empirical Cold–Overall Trade-off Landscape (5×6 grid, 5-seed means)"
Output: image_in_papers/v10/paper_v1/cold_overall_tradeoff_landscape.png

Usage (from repo root):
    python scripts/plot_cold_overall_tradeoff_landscape.py

Yêu cầu: matplotlib, numpy, pandas
"""

import json
import csv
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)   # repo root
PAPER_DIR  = os.path.join(BASE_DIR, "..")  # Q2_ver2/

GRID_CSV   = os.path.join(PAPER_DIR, "for_advisor/grid_v6_30configs_5maxlen_4datasets.csv")
# V6_DATA: path to canonical v6 paper results — set via environment or update below
V6_DATA    = os.environ.get("V6_DATA", os.path.join(BASE_DIR, "results/v6/data/"))
GS_BASE    = os.environ.get("GS_BASE", os.path.join(BASE_DIR, "results/gridsearch/"))
SS_DIGI    = os.path.join(BASE_DIR, "results"  # update: path to Diginetica full-len results
                          "diginetica/catpro/v3/diginetica/full_len/")
OUT_FILE   = os.path.join(PAPER_DIR, "image_in_papers/v10/paper_v1/cold_overall_tradeoff_landscape.png")

# ─── Helper ───────────────────────────────────────────────────────────────────
def avg_files(pattern):
    """Return (mean_overall_HR20 %, mean_cold_HR20 %) from seed JSON files.
       Returns (None, None) when no files match."""
    files = glob.glob(pattern, recursive=True) if isinstance(pattern, str) else list(pattern)
    if not files:
        return None, None
    hr, cold = [], []
    for f in files:
        raw = json.load(open(f))
        d   = raw.get("test", raw)
        hr.append(  d.get("HR@20",      d.get("hr@20",      0)))
        cold.append(d.get("Cold_HR@20", d.get("cold_hr@20", 0)))
    return float(np.mean(hr)) * 100, float(np.mean(cold)) * 100


def v6(ds_dir, method_subdir):
    """Load from v6/data/{ds_dir}/fullen/{method_subdir}/*.json"""
    return avg_files(os.path.join(V6_DATA, ds_dir, "fullen", method_subdir, "*.json"))


def no_both_full(ablation_ds_dir, prefix):
    """Proto-Sub Only = no_both, full-length session (files in the dir root, not maxlenN)."""
    # Full-length files live directly under the no_both/ folder (not inside maxlenN/)
    pat = os.path.join(GS_BASE, "ablation/results_ablation", ablation_ds_dir,
                       "no_both", f"{prefix}_A11_seed*.json")
    return avg_files(pat)


# ─── 1. Load grid data (Full-length only) ────────────────────────────────────
print("Loading grid CSV …")
grid_data = {k: [] for k in ("RR", "Digi", "Cell", "Yooc")}
with open(GRID_CSV, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["maxlen"] != "Full":
            continue
        lam = float(row["lambda"])
        rho = float(row["rho"])
        for key, ov_col, cold_col in [
            ("RR",   "RR_Overall",   "RR_Cold"),
            ("Digi", "Digi_Overall", "Digi_Cold"),
            ("Cell", "Cell_Overall", "Cell_Cold"),
            ("Yooc", "Yooc_Overall", "Yooc_Cold"),
        ]:
            grid_data[key].append({
                "lambda": lam, "rho": rho,
                "overall": float(row[ov_col]), "cold": float(row[cold_col]),
            })

# ─── 2. Reference methods ─────────────────────────────────────────────────────
print("Loading reference methods …")

refs = {
    "RR":   {"CatPro-CL": v6("retailrocket", "1_A11"),
             "M2TRec":    v6("retailrocket", "2_M2TRec"),
             "LetItGo":   v6("retailrocket", "11_LetItGo")},
    "Digi": {"CatPro-CL": v6("diginetica",   "1_A11"),
             "M2TRec":    v6("diginetica",    "2_M2TRec"),
             "LetItGo":   v6("diginetica",    "11_LetItGo")},
    "Cell": {"CatPro-CL": v6("cellphones",   "1_A11"),
             "M2TRec":    v6("cellphones",    "2_M2TRec"),
             "LetItGo":   v6("cellphones",    "11_LetItGo")},
    # Yoochoose: loaded directly from v6/data/yoochoose
    "Yooc": {"CatPro-CL": v6("yoochoose",    "1_A11"),
             "M2TRec":    v6("yoochoose",     "2_M2TRec"),
             "LetItGo":   v6("yoochoose",     "11_LetItGo")},
}

# ─── 3. SR-GNN baseline (A1 — no cold mechanism → Cold = 0) ──────────────────
sr_gnn = {
    "RR":   avg_files(os.path.join(GS_BASE,
                "retailrocket/retailrocket/retailrocket_fullen/retailrocket_A1_seed*.json")),
    # Diginetica: A1 full-len results (update GS_BASE to your results directory)
    "Digi": avg_files(os.path.join(SS_DIGI, "diginetica_A1_seed*.json")),
    # CellPhones / Yoochoose A1 not available locally — omit (displayed as missing)
    "Cell": (None, None),
    "Yooc": (None, None),
}

# ─── 4. Proto-Sub Only (cold inference only, no InfoNCE, no PSM) ──────────────
# Ablation "no_both" full-length = λ=0, ρ=0 + cold inference
proto_sub = {
    "RR":   no_both_full("retailrocket", "retailrocket"),
    "Digi": no_both_full("diginetica",   "diginetica"),
    "Cell": no_both_full("cellphones",   "cellphones"),
    # Yoochoose: from the grid's l000_p005 results (full-length sessions)
    "Yooc": avg_files(os.path.join(GS_BASE,
                "grid_yoochoose/grid_yoochoose/results_a11_l000_p005/"
                "yoochoose_1_64/yoochoose_1_64_A11_seed*.json")),
}

for ds in ("RR", "Digi", "Cell", "Yooc"):
    print(f"  {ds}: SR-GNN={sr_gnn[ds]}, Proto-Sub={proto_sub[ds]}, "
          f"CatPro-CL={refs[ds]['CatPro-CL']}")

# ─── 5. Visual mappings ───────────────────────────────────────────────────────
LAMBDA_COLORS = {
    0.10: "#5B9BD5",  # blue
    0.20: "#C00000",  # dark red
    0.30: "#9DC3E6",  # light blue
    0.40: "#833C00",  # dark brown
    0.50: "#1F4E79",  # navy
}
RHO_SIZES = {0.05: 40, 0.10: 80, 0.20: 140, 0.30: 210, 0.40: 290, 0.50: 380}

PANELS = [
    ("RR",   "RetailRocket"),
    ("Digi", "Diginetica"),
    ("Yooc", "Yoochoose 1/64"),
    ("Cell", "CellPhones"),
]

# ─── 6. Draw ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(
    "Empirical Cold–Overall Trade-off Landscape  (5×6 grid, 5-seed means)",
    fontsize=13, fontweight="bold", y=0.98,
)

for ax, (ds, title) in zip(axes.flat, PANELS):
    # Grid scatter
    for pt in grid_data[ds]:
        ax.scatter(
            pt["overall"], pt["cold"],
            c=LAMBDA_COLORS.get(round(pt["lambda"], 2), "grey"),
            s=RHO_SIZES.get(round(pt["rho"], 2), 100),
            alpha=0.85, edgecolors="white", linewidths=0.4, zorder=3,
        )

    # Helper to plot one reference marker + label
    def _ref(xy, marker, color, label, size=150, zorder=6):
        if xy is None or xy[0] is None:
            return
        ax.scatter(xy[0], xy[1], marker=marker, c=color, s=size,
                   zorder=zorder, edgecolors="none")
        ax.annotate(label, xy=(xy[0], xy[1]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7.5, color=color, fontweight="bold")

    m2t = refs[ds]["M2TRec"]
    # Dashed horizontal reference at M2TRec Cold HR@20
    if m2t and m2t[0]:
        ax.axhline(m2t[1], color="#9B59B6", linestyle="--",
                   linewidth=1.0, alpha=0.7, zorder=1)

    _ref(sr_gnn[ds],         "s", "black",   "SR-GNN",          size=110)
    _ref(m2t,                "D", "#9B59B6", "M2TRec",          size=120)
    _ref(refs[ds]["LetItGo"],"^", "#27AE60", "LetItGo",         size=120)
    _ref(proto_sub[ds],      "P", "#8B6914", "Proto-Sub\nOnly", size=120)
    _ref(refs[ds]["CatPro-CL"], "*", "#E74C3C", "CatPro-CL\n★",
         size=320, zorder=7)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Overall HR@20 (%)", fontsize=9)
    ax.set_ylabel("Cold HR@20 (%)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

# ─── 7. Legends ──────────────────────────────────────────────────────────────
lambda_patches = [
    mpatches.Patch(color=c, label=f"λ={lam:.2f}")
    for lam, c in sorted(LAMBDA_COLORS.items())
]
rho_handles = [
    Line2D([0], [0], marker="o", color="grey",
           markersize=np.sqrt(s) * 0.55, linestyle="None", label=f"ρ={rho:.2f}")
    for rho, s in sorted(RHO_SIZES.items())
]
ref_handles = [
    Line2D([0],[0], marker="s", color="black",  ms=9, ls="None", label="SR-GNN (A1)"),
    Line2D([0],[0], marker="^", color="#27AE60",ms=9, ls="None", label="LetItGo (adapted)"),
    Line2D([0],[0], marker="D", color="#9B59B6",ms=9, ls="None", label="M2TRec (adapted)"),
    Line2D([0],[0], marker="P", color="#8B6914",ms=9, ls="None", label="Proto-Sub Only"),
    Line2D([0],[0], marker="*", color="#E74C3C",ms=14,ls="None", label="CatPro-CL ★ (λ=0.50, ρ=0.05)"),
    Line2D([0],[0], color="#9B59B6", ls="--",              label="M2TRec Cold HR@20 reference"),
]

fig.legend(
    handles=lambda_patches + rho_handles,
    title="Grid params  (color = λ, size = ρ)",
    loc="lower left", bbox_to_anchor=(0.02, 0.0),
    ncol=4, fontsize=7.5, title_fontsize=8,
    framealpha=0.9, handletextpad=0.4, columnspacing=0.8,
)
fig.legend(
    handles=ref_handles,
    title="Reference methods",
    loc="lower right", bbox_to_anchor=(0.98, 0.0),
    ncol=2, fontsize=7.5, title_fontsize=8,
    framealpha=0.9, handletextpad=0.4, columnspacing=0.8,
)

plt.tight_layout(rect=[0, 0.12, 1, 0.97])
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
plt.savefig(OUT_FILE, dpi=200, bbox_inches="tight")
print(f"\nSaved → {OUT_FILE}")
plt.close()
