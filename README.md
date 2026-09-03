# CatPro-CL: Category-Prototype Alignment for Cold-Start Session-Based Recommendation via Contrastive Learning

Official implementation for the IEEE Access paper:

> **Category-Prototype Alignment for Cold-Start Session-Based Recommendation via Contrastive Learning**  
> *[Author names] — IEEE Access, 2026*

---

## Overview

CatPro-CL addresses the cold-start problem in session-based recommendation. Cold items (items with no interaction history) receive zero score from standard session graph models. Our approach:

1. **SR-GNN backbone** — models each session as a directed graph with a gated GNN
2. **EMA Category Prototype Bank** — maintains one prototype vector per item category, updated online with momentum 0.99
3. **Item-Prototype InfoNCE loss** — aligns warm item embeddings toward their category prototypes during training (temperature τ = 0.1)
4. **Prototype Simulation Masking (PSM)** — during training, randomly replaces a warm target's embedding with its category prototype (probability ρ = 0.05, restricted to categories containing cold items), so the model learns to score prototype-represented cold items correctly
5. **Cold inference** — at test time, substitutes cold item embeddings with their category prototype

The full model (A11) uses λ = 0.5, ρ = 0.05, τ = 0.1.

---

## Requirements

```bash
# Python 3.10+, CUDA 12.x recommended
pip install -r requirements.txt

# Install PyTorch separately (match your CUDA version):
# https://pytorch.org/get-started/locally/
# e.g. pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

---

## Datasets

We use four datasets: **RetailRocket**, **Diginetica**, **Yoochoose 1/64**, and **Amazon CellPhones**.

### Download raw data

| Dataset | Source |
|---|---|
| RetailRocket | [Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) |
| Diginetica | [RecSys 2016 Challenge](https://cikm2016.cs.iupui.edu/cikm-cup/) |
| Yoochoose | [RecSys 2015 Challenge](https://recsys.acm.org/recsys15/challenge/) |
| Amazon CellPhones | [Amazon Review Data (2018)](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) |

### Preprocess

```bash
# Runs all four datasets in sequence
bash prepare_datasets.sh both

# Or individually:
python preprocessing/preprocess_retailrocket.py   --raw_dir ~/data/raw/retailrocket   --out_dir ~/data/retailrocket_unified
python preprocessing/preprocess_diginetica.py     --raw_dir ~/data/raw/diginetica      --out_dir ~/data/diginetica_unified
python preprocessing/preprocess_yoochoose.py      --raw_dir ~/data/raw/yoochoose       --out_dir ~/data/yoochoose_unified
python preprocessing/preprocess_amazon_cellphones.py --raw_dir ~/data/raw/cellphones   --out_dir ~/data/cellphones_unified

# Create cold_20 split (20% of items are cold)
python preprocessing/cold_start_split.py --data_dir ~/data/retailrocket_unified --cold_ratio 0.2

# Sanity check (should report cold_items = 9,052 for RetailRocket cold_20)
python preprocessing/sanity_check.py --data_dir ~/data/retailrocket_unified/cold_20
```

### Data format

Each `cold_NN/` directory contains 7 files:

```
sessions_train.txt   # one session per line, space-separated item ids, last id = target
sessions_val.txt
sessions_test.txt
item2cat.json        # {item_id: cat_id}
cat2items.json       # {cat_id: [item_ids]}
cat_parent.json      # {cat_id: parent_cat_id}
meta.json            # dataset statistics
```

Cold items are **non-leaking**: they are removed from all training/validation sessions and appear only as prediction targets in the test set.

---

## Training

All hyperparameters come from the YAML config. No script edits needed.

```bash
# Full model (A11) — single seed
python catprocl/train.py \
    --config configs/catprocl_retailrocket_a11.yaml \
    --ablation A11 --seed 42

# Override hyperparameters on the command line
python catprocl/train.py \
    --config configs/catprocl_diginetica_a11.yaml \
    --ablation A11 --seed 0 --maxlen 4 \
    --lambda_proto 0.5 --mask_prob 0.05 --temperature 0.1 \
    --output_dir ~/results/diginetica_a11

# Plain SR-GNN baseline (A1)
python catprocl/train.py \
    --config configs/catprocl_retailrocket_a11.yaml \
    --ablation A1 --seed 42

# 5-seed sweep (seeds 42, 0, 1, 2, 3)
bash scripts/run_catprocl_5seeds.sh
```

**Graph cache**: on first run, `data_loader.py` builds a session-graph cache under `data_dir/cache/` (~5–10 min). Subsequent runs load it in seconds.

---

## Baselines

```bash
# Non-neural baselines (no training needed)
python baselines/popularity.py    --data_dir ~/data/retailrocket_unified/cold_20 --output_dir ~/results/baselines --dataset retailrocket --seed 42
python baselines/content_knn.py   --data_dir ~/data/retailrocket_unified/cold_20 --output_dir ~/results/baselines --dataset retailrocket --seed 42

# Graph-based
python baselines/gcegnn.py        --data_dir ~/data/retailrocket_unified/cold_20 --output_dir ~/results/baselines --dataset retailrocket --seed 42

# Adapted cold-start baselines
python baselines/m2trec.py        --data_dir ~/data/retailrocket_unified/cold_20 --output_dir ~/results/baselines --dataset retailrocket --seed 42
python baselines/letitgo.py       --data_dir ~/data/retailrocket_unified/cold_20 --output_dir ~/results/baselines --dataset retailrocket --seed 42

# 5-seed sweep of all baselines
bash scripts/run_baselines_5seeds.sh
```

---

## Evaluation

Evaluation is **full-ranking** (scores computed over all ~45k items, no sampled negatives).  
Metrics: **HR@10, HR@20, MRR@10, MRR@20** — reported separately for **Overall** and **Cold** sessions.  
Statistical results use 5 seeds (42, 0, 1, 2, 3); report sample std (ddof=1).

```bash
# Aggregate all results into a summary table
python scripts/aggregate_results.py

# Print formatted table (matches paper Table II)
python scripts/report_results.py

# Verify paper numbers against result JSON files
python scripts/validate_paper_numbers.py
```

---

## Ablation Study

The paper includes two ablation axes:

**1. Ablation variants (A1–A11):** controlled by the `--ablation` flag (see `catprocl/train.py` docstring).

**2. Component ablation (λ, ρ sensitivity):** uses configs in `configs/ablation_components/`.

```bash
bash scripts/run_ablation_components.sh
python scripts/collect_ablation_results.py
```

---

## Repository Structure

```
CatPro-CL/
├── catprocl/                   # Proposed model
│   ├── model.py                # SR-GNN backbone (GatedSessionGNN + SRGNNEncoder)
│   ├── prototype_bank.py       # EMA category prototype bank
│   ├── losses.py               # rec_loss, item_prototype_infonce, combined_loss
│   ├── cold_inference.py       # build_eval_item_embeddings (cold substitution)
│   ├── data_loader.py          # get_dataloaders + graph cache builder
│   ├── dataset.py              # SessionGraphDataset + collator
│   ├── train.py                # Single training entrypoint
│   └── ablation/               # Variant modules for A6/A9/A10/A12–A15
├── baselines/                  # One self-contained script per competing method
├── preprocessing/              # Raw → unified 7-file format + cold split
├── evaluation/
│   └── evaluator.py            # Full-ranking HR@k / MRR@k (Overall + Cold)
├── configs/                    # One YAML per dataset × variant
│   └── ablation_components/    # λ/ρ sensitivity configs
├── scripts/                    # Sweep runners + result aggregation
├── prepare_datasets.sh         # End-to-end preprocessing pipeline
└── requirements.txt
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{catprocl2026,
  title   = {Category-Prototype Alignment for Cold-Start Session-Based Recommendation via Contrastive Learning},
  author  = {...},
  journal = {IEEE Access},
  year    = {2026},
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
