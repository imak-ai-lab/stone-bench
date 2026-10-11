# StoneBench

Rock-fragment detection and size-distribution evaluation across mining lifecycle
stages. The demo baselines are YOLO11n-OBB and YOLO11s-OBB, fine-tuned on stones
with seed 42.

Companion code for *Are Foundation Models Ready for Rock Fragmentation? A
Multi-Domain Benchmark Across the Mining Lifecycle*: lightweight inference,
evaluation tools, frozen annotations and paper tables. The paper's large supervised
models are YOLO11x-OBB and RT-DETR-X; nano/small provide a lightweight executable
example and do not reproduce their table scores.

## Run the demo

Clone the repository (Git and 64-bit Python 3.12 required):

```bash
git clone https://github.com/imak-ai-lab/stone-bench.git
cd stone-bench
```

Windows PowerShell:

```powershell
.\demo.ps1
```

The launcher installs the dependencies into `.venv-demo`, verifies checkpoint
SHA-256, downloads Transport v1 (72.3 MB) from the exact Mendeley source cited in
the paper, extracts one fixed image, and runs both baselines on CPU in full-frame
and tiled modes. Downloaded images stay outside Git. First setup requires internet access and
several GB of free disk space for the environment; no GPU or API key is required.
See [demo guide](docs/camera_ready.md) for expected outputs and troubleshooting.
Use `-Variant nano` or `-Variant small` to select one model; `-Mode full_frame`
to run without tiling. Supply your own image or directory with `-Source PATH`.

Linux/macOS shell (Windows CPU is the verified platform):

```bash
python3.12 -m venv .venv-demo
source .venv-demo/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[yolo]"
python scripts/download_demo_sample.py
python scripts/run_demo.py --source data/samples/transport.jpg
```

On macOS, omit `--index-url` from the PyTorch installation command. For an existing
Python 3.11/3.12 environment, skip creation and activation above. On Windows use
`demo.ps1` or activate with `.\.venv-demo\Scripts\Activate.ps1`.

Checkpoints and their manifest are in `weights/lightweight/`.
Outputs are in `outputs/demo/{nano,small}/{mode}/`: annotated PNGs,
`predictions.json` and `run_info.json`. Predictions use original-image coordinates.
Default confidence: 0.25; tile size: 1024; overlaps: 0/128/256.
Re-running into a nonempty mode directory is rejected; choose `--output outputs/demo2`
(PowerShell: `-Output outputs/demo2`).

## Minimal metric example

```bash
python scripts/metric_example.py
```

This generates two geometric objects and exercises both evaluation CLIs. Expected:
perfect predictions give F1 = 1 and zero size errors; missing the smaller object
gives F1 = 0.666667 and missed-small = 1. CSV/JSON files appear in
`outputs/metric_example/`. No weights or downloads are needed after base installation.
These are analytical checks, not measured benchmark results.

## Data

[Dataset sources](data/manifests/dataset_sources.md) and
[download manifest](data/manifests/download_sources.csv) retain the links cited in
the article. Raw data are stored externally.
The linked Transport v1 contains images but no instance annotations, so the downloaded
demo image supports inference only. See [reproduction inputs](docs/reproduction.md)
for the remaining requirements for numerical reproduction.
The minimal annotation/configuration bundle is included in Git (7.05 MB);
see [data restoration and evaluation on Transport](docs/data_export.md).

## Paper results

| Paper table | CSV |
|---|---|
| III — Zero-shot detection | [Leaderboard](results/tables/zero_shot_leaderboard.csv) |
| IV — Zero-shot by dataset | [Per-dataset F1](results/tables/zero_shot_per_dataset.csv) |
| V — Supervised detection | [Supervised results](results/tables/supervised_per_dataset.csv) |
| VI — Tiling | [Tiling results](results/tables/tiling_ablation.csv) |
| VII — Size distributions | [Size metrics](results/tables/size_distribution_metrics.csv) |

Values reproduce the corresponding tables in the submitted article. SD columns
follow Table VI; W1 and D80 are in pixels.

## Reproduction

See [experiment instructions](docs/reproduction.md) and
[evaluation protocol](docs/evaluation_protocol.md). The common manifest is
`data/manifests/unified_split_seed42.json`: 518 evaluation images, 60,287 fragments;
Synthetic is train-only. Zero-shot F1 confidence is 0.30; supervised F1 uses 0.25.
The paper reports seeds 42/43/44 and trains tiled models on tiles.
The nano/small demo uses a fixed checkpoint with optional tiled inference.

Training settings are described in the experiment instructions. The executable
examples use the included nano/small checkpoints and require no training.

## Checks

```bash
python scripts/check_camera_ready.py --scope demo --strict
python scripts/prepare_reproduction_inputs.py
python scripts/make_tables.py --output results/tables
```
