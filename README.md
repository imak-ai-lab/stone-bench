# StoneBench

**A multi-domain benchmark for rock-fragment detection and size-distribution analysis across the mining lifecycle.**

Companion repository for *Are Foundation Models Ready for Rock Fragmentation? A Multi-Domain Benchmark Across the Mining Lifecycle*.

[Quick start](#quick-start) · [Data](#data) · [Paper results](#paper-results) · [Documentation](#documentation)

## Overview

StoneBench brings together rock-fragment imagery from blast piles, loading,
haulage, conveyor belts, and synthetic scenes. It provides a common oriented
bounding box (OBB) representation for evaluating individual fragments and the
size distributions derived from them.

This repository includes:

- **A ready-to-run demo** with fine-tuned YOLO11n-OBB and YOLO11s-OBB checkpoints.
- **Detection and size-distribution metrics**, including precision, recall, F1,
  CDF RMSE, KS, Wasserstein-1, D10/D80 errors, and missed-small fraction.
- **Frozen annotations and split manifests** for consistent evaluation.
- **Paper results in CSV format**, with links to dataset sources and protocol details.

## Quick start

The demo runs on CPU and requires no GPU or API key. Use Git and 64-bit Python
3.12; the Python commands also support an existing Python 3.11 environment.

```bash
git clone https://github.com/imak-ai-lab/stone-bench.git
cd stone-bench
```

### Windows

```powershell
.\demo.ps1
```

The launcher creates an environment, installs dependencies, downloads a sample
image, and runs both models in full-frame and tiled modes.

### Linux and macOS

```bash
python3.12 -m venv .venv-demo
source .venv-demo/bin/activate
```

Install PyTorch on Linux:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

On macOS:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1
```

Then install StoneBench and run the demo:

```bash
python -m pip install -e ".[yolo]"
python scripts/download_demo_sample.py
python scripts/run_demo.py --source data/samples/transport.jpg
```

The first run requires internet access and several GB of free disk space.

### Explore the output

Each model and inference mode produces an annotated image, `predictions.json`,
and `run_info.json` under `outputs/demo/`. Predictions use original-image coordinates.

To try your own image or directory in the installed environment:

```bash
python scripts/run_demo.py --source path/to/images --variant nano --mode full_frame --output outputs/my_demo
```

In Windows PowerShell:

```powershell
.\demo.ps1 -Source "C:\images\stones.jpg" -Variant nano -Mode full_frame -Output outputs/my_demo
```

Use a new output directory for each run. See the [demo guide](docs/camera_ready.md)
for inference options, environment setup, and troubleshooting.

The included nano/small checkpoints are lightweight examples. The paper reports
YOLO11x-OBB and RT-DETR-X results; those checkpoints are not included in this release.

## Evaluate predictions

After installation, run a small example with known answers:

```bash
python scripts/metric_example.py
```

The example evaluates perfect predictions and a missed fragment. Expected F1
scores are **1.0** and **0.666667**, respectively. Results are saved as CSV and JSON
in `outputs/metric_example/`.

For evaluation on real images, follow the [Transport walkthrough](docs/data_export.md#evaluate-transport).
The [evaluation protocol](docs/evaluation_protocol.md) defines matching, size metrics,
confidence thresholds, and bootstrap intervals.

## Data

Raw images are hosted by the original dataset providers. The repository includes
a compact annotation bundle with **4,196 OBB label files** and frozen split metadata.
The common evaluation partition contains **518 images and 60,287 fragments**;
the generated Synthetic domain is used only for training.

| Resource | Description |
|---|---|
| [Dataset sources](data/manifests/dataset_sources.md) | Original datasets and download links |
| [Download manifest](data/manifests/download_sources.csv) | Source identifiers and dataset versions |
| [Dataset card](docs/dataset_card.md) | Domains, annotation sources, and limitations |
| [Data preparation](docs/data_export.md) | Restore annotations, prepare images, and evaluate Transport |
| [Frozen split](data/manifests/unified_split_seed42.json) | Image keys for training and evaluation |

The demo downloads an image from Transport v1. For evaluation with annotations,
use the Transport v2 instructions in the data preparation guide.

## Paper results

The following CSV files contain the values reported in the paper.

| Paper table | Results |
|---|---|
| III — Zero-shot detection | [Leaderboard](results/tables/zero_shot_leaderboard.csv) |
| IV — Zero-shot by dataset | [Per-dataset F1](results/tables/zero_shot_per_dataset.csv) |
| V — Supervised detection | [Supervised results](results/tables/supervised_per_dataset.csv) |
| VI — Tiling | [Tiling ablation](results/tables/tiling_ablation.csv) |
| VII — Size distributions | [Size metrics](results/tables/size_distribution_metrics.csv) |

Wasserstein-1 and D80 errors are reported in pixels. See the
[experiment documentation](docs/reproduction.md) for training settings and the
relationship between the demo and paper experiments.

## Documentation

| Guide | Contents |
|---|---|
| [Demo](docs/camera_ready.md) | Setup, inference modes, outputs, and troubleshooting |
| [Data preparation](docs/data_export.md) | Annotation validation and dataset preparation |
| [Annotation format](docs/annotation_format.md) | OBB labels and prediction JSON schema |
| [Evaluation protocol](docs/evaluation_protocol.md) | Metrics, matching, and uncertainty |
| [Experiments](docs/reproduction.md) | Paper settings and reproduction inputs |

To verify the included checkpoints, annotation bundle, and table files:

```bash
python scripts/check_camera_ready.py --scope demo --strict
python scripts/prepare_reproduction_inputs.py
python scripts/make_tables.py --output results/tables
```

The table check validates the saved CSV files; it does not rerun experiments.

## License

Repository code is released under the [MIT License](LICENSE). Refer to the original
dataset providers for data licensing terms.
