# Local Inference And Training

This repository is local-first. Cloud submission files, scheduler wrappers, and
cloud-specific paths are intentionally excluded.

## Zero-Shot Inference

`scripts/run_zeroshot.py` discovers local prepared datasets and writes
reproducible run plans. The public package includes a lightweight YOLO-World
execution path and planning support for the configured foundation baselines.

## Supervised Training

`scripts/train_supervised.py` contains the local Ultralytics training entrypoint
for YOLO11x-OBB and RT-DETR-X. The wrapper scripts select the default configs:
`scripts/train_yolo_obb.py` and `scripts/train_rtdetr_x.py`.

## Tiled Inference

`scripts/run_tiled_inference.py` covers planning, tile generation, local
prediction, stitching with NMS, and full-frame evaluation.

## Outputs

Output payloads remain local and inspectable:

```text
outputs/<run>/
  resolved_config.yaml
  metrics.json
  predictions.json
  run_info.json
```
