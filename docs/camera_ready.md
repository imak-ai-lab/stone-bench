# Demo

Baseline weights: YOLO11n-OBB and YOLO11s-OBB, fine-tuned on stones, seed 42.
Files and SHA-256 manifest are in `weights/lightweight/`.
Both weights are tracked in Git (5.7 MB and 19.8 MB). Use 64-bit Python 3.11/3.12
and allow several GB for dependencies. Windows CPU is verified; GPU and other OS
runs are unverified.
For the demo dependency versions from the verified Windows Python 3.12 setup, run
`python -m pip install -r docs/demo_environment_windows_py312.txt --extra-index-url https://download.pytorch.org/whl/cpu`
in a fresh environment, then `python -m pip install -e . --no-deps`.

## Windows

Install Python 3.12, then run from the repository root:

```powershell
.\demo.ps1
```

The script creates `.venv-demo`, installs dependencies, verifies weights and runs
both models. With no `-Source`, it downloads the 72.3 MB Transport v1 archive and
extracts one checksum-pinned, unmodified 2592x2048 image to `data/samples/transport.jpg`.
Only the sample and its source manifest are retained. An explicit Python path is supported:

```powershell
.\demo.ps1 -Source "C:\images\stones.jpg" -PythonExe "C:\Python312\python.exe"
```

Select one model with `-Variant nano` or `-Variant small`. Use `-Mode full_frame`
for a single full-frame pass. Default: both models, all four modes, CPU.

## Existing environment

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[yolo]"
python scripts/download_demo_sample.py
python scripts/run_demo.py --source data/samples/transport.jpg --variant both --mode all
```

CUDA requires a compatible GPU PyTorch installation; use `--device 0` with the
Python launcher. `demo.ps1` installs the CPU build.

## Output

`outputs/demo/{nano,small}/{mode}/` contains annotated PNGs, `predictions.json`
and `run_info.json`. The default run creates eight mode directories, each with
one 2592x2048 PNG, one prediction JSON and one run-info JSON. Green quadrilaterals
are predicted stone OBBs; an image without boxes is a valid empty prediction.
Prediction records include image SHA-256, pixel dimensions, scores and eight
polygon coordinates `[x1,y1,x2,y2,x3,y3,x4,y4]`. Run info records checkpoint
SHA-256, Python/PyTorch/Ultralytics versions, arguments and counts.
Coordinates refer to the original image. Input may be a
single image or a directory. Confidence is 0.25; tile size is 1024; overlaps are
0/128/256. The demo uses fixed weights in all modes.
Full-frame OBB corners can extend beyond image edges; JSON retains the model's
geometry and PNG drawing clips to the canvas. The demo tiled merger clamps shifted
corners to image bounds before center ownership and polygon NMS.

This nano/small example is separate from the paper's train-on-tiles experiment
with YOLO11x-OBB. A single image cannot reproduce aggregate F1/mAP or Table VI
costs. Transport v1 has no instance annotations; missing GT is not an empty label.
The [metric example](../README.md#minimal-metric-example) checks evaluation independently.

## Repeat runs and troubleshooting

- Use `--output outputs/demo2` (PowerShell: `-Output outputs/demo2`) to preserve
  existing results and create a fresh run.
- For offline inference, install dependencies and download the sample once, then
  invoke `.venv-demo/Scripts/python.exe scripts/run_demo.py --source PATH --output outputs/offline`.
  `demo.ps1` includes installation steps and is intended for online setup.
- If PowerShell blocks scripts, use `powershell -ExecutionPolicy Bypass -File .\demo.ps1`
  for this process only, or follow the Python commands above.
- If `py -3.12` is unavailable, pass `-PythonExe C:\Python312\python.exe`.
- If the sample download fails, open the article's
  [Mendeley v1 page](https://data.mendeley.com/datasets/v9grvgsrh9/1), download
  `transport.zip`, then run `python scripts/download_demo_sample.py --archive PATH_TO_ZIP`.
  Both routes verify the archive and image hashes.
- Python launcher `--conf 0.001` preserves low-score candidates for AP export.
  Apply 0.25 at supervised F1 evaluation; the default demo visualizes scores >= 0.25.

[Experiment reproduction](reproduction.md) · [Evaluation protocol](evaluation_protocol.md)
