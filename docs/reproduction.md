# Reproduction Guide

This repository is designed for local reproduction. It does not include cloud
jobs, scheduler wrappers, raw datasets, model weights, or credentials.

## Table Snapshots

Committed CSV snapshots live in `results/tables/`:

| Table source | File |
|---|---|
| Zero-shot leaderboard | `results/tables/zero_shot_leaderboard.csv` |
| Zero-shot per dataset | `results/tables/zero_shot_per_dataset.csv` |
| Supervised per dataset | `results/tables/supervised_per_dataset.csv` |
| Tiling ablation | `results/tables/tiling_ablation.csv` |
| Size metrics | `results/tables/size_distribution_metrics.csv` |

Validate the table directory with:

```bash
python scripts/make_tables.py --output results/tables
```

## Local Commands

```bash
python scripts/download_data.py --provider mendeley --dry-run
python scripts/download_data.py \
  --provider roboflow \
  --api-key <roboflow-api-key> \
  --roboflow-version ronveer=1 \
  --dry-run
python scripts/prepare_data.py --config configs/prepare_data.yaml
python scripts/run_zeroshot.py \
  --config configs/zeroshot/yolo_world_x.yaml \
  --dry-run
python scripts/run_foundational.py --preset main
python scripts/run_foundation_models.py --child-dry-run
python scripts/train_supervised.py \
  --config configs/supervised/yolo11x_obb.yaml \
  --dry-run
python scripts/train_supervised.py \
  --config configs/supervised/rtdetr_x.yaml \
  --dry-run
python scripts/run_tiled_inference.py \
  --config configs/tiling/yolo11x_obb_overlap_0.yaml \
  --stage plan
python scripts/evaluate_detection.py \
  --predictions results/predictions \
  --dataset-root data/prepared/stonebench/yolo_obb
python scripts/evaluate_granulometry.py \
  --predictions results/predictions \
  --dataset-root data/prepared/stonebench/yolo_obb
```

## Hardware

The configs target a local workstation or a single GPU machine. The paper runs
used one A100-class GPU for model inference/training.
