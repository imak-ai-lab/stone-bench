# Baseline demo weights

| Model | File | Bytes |
|---|---|---:|
| YOLO11n-OBB | `stones-yolo11n-obb-seed42.pt` | 5695932 |
| YOLO11s-OBB | `stones-yolo11s-obb-seed42.pt` | 19829692 |

Both checkpoints are fine-tuned on stones with seed 42. `manifest.json` records
SHA-256 and checkpoint metrics. `scripts/run_demo.py` verifies both size and hash
before inference. See [demo instructions](../../docs/camera_ready.md).
