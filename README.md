# StoneBench

StoneBench is a local toolkit for rock-fragment detection experiments across
multiple stages of the mining lifecycle. The repository contains source
manifests, data preparation scripts, model run configurations, local supervised
and tiled pipelines, metric computation, compact table snapshots, and tests.

The repository does not include raw datasets, prepared datasets, model weights,
raw predictions, credentials, or cloud job definitions.

## Install

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

Optional model backends are grouped by extra:

```bash
python -m pip install -e ".[yolo]"
python -m pip install -e ".[foundation]"
python -m pip install -e ".[vlm]"
```

## Data

Dataset sources are listed in `data/manifests/download_sources.csv`.

```bash
python scripts/download_data.py --dry-run
python scripts/download_data.py --provider mendeley --dry-run --open-browser
python scripts/download_data.py --provider mendeley --mendeley-manual-dir "%USERPROFILE%\Downloads"
python scripts/download_data.py --provider roboflow --api-key <roboflow-api-key> --roboflow-version ronveer=1
```

Mendeley archives are handled through the browser flow. Download the ZIP from
the source page, rename it to the `output_name` listed in the manifest, then run
the command with `--mendeley-manual-dir`.

Roboflow exports use the official REST export route and require
a Roboflow API key, passed with `--api-key` or `ROBOFLOW_API_KEY`. The Ronveer
source page does not pin a dataset version, so pass the version explicitly with
`--roboflow-version ronveer=N`.

## Preparation

Extract downloaded archives under `data/raw/extracted/<dataset_slug>/` using the
slugs from `data/manifests/datasets.csv`, then run:

```bash
python scripts/prepare_data.py --config configs/prepare_data.yaml
python scripts/prepare_data.py --config configs/prepare_data.yaml --split-mode resplit --formats yolo_obb yolo_detect
```

The preparer writes these formats under `data/prepared/stonebench/`:

- `yolo_obb` for YOLO OBB training and OBB evaluation.
- `yolo_detect` for AABB detectors such as RT-DETR-X.
- `yolo_seg`, `coco_detect`, and `coco_seg` for compatibility workflows.

## Foundation Models

```bash
python scripts/run_zeroshot.py --config configs/zeroshot/yolo_world_x.yaml --dry-run
python scripts/run_foundational.py --preset main
python scripts/run_foundation_models.py --child-dry-run
```

The package includes local run planning for all configured foundation baselines
and a lightweight YOLO-World execution path for smoke testing. The remaining
foundation configs are versioned as explicit run specifications, including
prompts, thresholds, limits, and output layout.

## Supervised And Tiled Runs

```bash
python scripts/train_supervised.py --config configs/supervised/yolo11x_obb.yaml --dry-run
python scripts/train_supervised.py --config configs/supervised/rtdetr_x.yaml --dry-run
python scripts/run_tiled_inference.py --config configs/tiling/yolo11x_obb_overlap_0.yaml --stage plan
python scripts/run_tiled_inference.py --config configs/tiling/yolo11x_obb_overlap_0.yaml --stage prepare
```

The supervised and tiled scripts are local-first wrappers around prepared data,
Ultralytics training, tile generation, stitched prediction handling, and
full-frame evaluation.

## Metrics And Tables

```bash
python scripts/evaluate_detection.py --predictions results/predictions --dataset-root data/prepared/stonebench/yolo_obb
python scripts/evaluate_granulometry.py --predictions results/predictions --dataset-root data/prepared/stonebench/yolo_obb
python scripts/make_tables.py --output results/tables
```

`scripts/evaluate_detection.py` applies the unified confidence post-filter and
greedy polygon-IoU matching. `scripts/evaluate_granulometry.py` computes CDF
RMSE, KS, Wasserstein-1, D80 absolute error, and missed-small fraction.

Compact table snapshots are stored in `results/tables/`.

## Layout

```text
configs/        Dataset, zero-shot, supervised, and tiling configs.
data/           Manifests, split policy, checksums, and optional samples.
docs/           Dataset, annotation, evaluation, and reproduction notes.
results/        Committed table snapshots; raw outputs are ignored.
scripts/        Local CLI entrypoints.
src/            Python package implementation.
tests/          Deterministic smoke and metric tests.
```
