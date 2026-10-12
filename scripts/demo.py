"""Local image demo using an explicitly supplied fine-tuned OBB checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stones_fragmentation.metrics.detection import polygon_iou
from predict_ultralytics_obb import parse_result


def starts(length, size, overlap):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require tile-size > 0 and 0 <= overlap < tile-size")
    if length <= size:
        return [0]
    positions = list(range(0, length - size + 1, size - overlap))
    return sorted(set(positions + [length - size]))


def owned(center, position, positions, size, length):
    index = positions.index(position)
    lower = 0 if index == 0 else (position + positions[index - 1] + size) / 2
    upper = length if index == len(positions) - 1 else (position + size + positions[index + 1]) / 2
    return lower <= center < upper


def merge(detections, threshold, limit):
    kept = []
    for item in sorted(detections, key=lambda d: d["score"], reverse=True):
        if any(item["label"] == other["label"] and
               polygon_iou(item["obb_xyxyxyxy"], other["obb_xyxyxyxy"]) >= threshold
               for other in kept):
            continue
        kept.append(item)
        if len(kept) >= limit:
            break
    return kept


def infer(model, image, args, mode):
    kwargs = dict(imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                  max_det=args.max_det, device=args.device, verbose=False)
    if mode == "full_frame":
        return parse_result(model.predict(source=image, **kwargs)[0])
    overlap = int(mode.removeprefix("overlap"))
    width, height = image.size
    xs, ys = starts(width, args.tile_size, overlap), starts(height, args.tile_size, overlap)
    detections = []
    kwargs["max_det"] = args.max_det_per_tile
    for y in ys:
        for x in xs:
            crop = image.crop((x, y, min(width, x + args.tile_size), min(height, y + args.tile_size)))
            for item in parse_result(model.predict(source=crop, **kwargs)[0]):
                poly = item["obb_xyxyxyxy"]
                poly = [max(0, min(width if i % 2 == 0 else height,
                                  value + (x if i % 2 == 0 else y))) for i, value in enumerate(poly)]
                cx = (min(poly[::2]) + max(poly[::2])) / 2
                cy = (min(poly[1::2]) + max(poly[1::2])) / 2
                if owned(cx, x, xs, args.tile_size, width) and owned(cy, y, ys, args.tile_size, height):
                    item["obb_xyxyxyxy"] = poly
                    detections.append(item)
    return merge(detections, args.merge_iou, args.max_det)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="Image or directory of images")
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--mode", choices=["full_frame", "overlap0", "overlap128", "overlap256", "all"], default="full_frame")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--merge-iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=5000)
    parser.add_argument("--max-det-per-tile", type=int, default=3000)
    args = parser.parse_args()
    if not args.weights.is_file() or args.weights.stat().st_size == 0:
        parser.error("Supply a nonempty released fine-tuned .pt file; see docs/camera_ready.md")
    if not 0 <= args.conf <= 1 or not 0 < args.iou <= 1 or not 0 < args.merge_iou <= 1:
        parser.error("Invalid confidence or IoU threshold")
    if min(args.imgsz, args.tile_size, args.max_det, args.max_det_per_tile) <= 0:
        parser.error("Sizes and detection limits must be positive")
    modes = ["full_frame", "overlap0", "overlap128", "overlap256"] if args.mode == "all" else [args.mode]
    for mode in modes:
        if mode != "full_frame":
            starts(1, args.tile_size, int(mode.removeprefix("overlap")))
    images = sorted(p for p in args.source.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}) if args.source.is_dir() else [args.source]
    if not images or any(not p.is_file() for p in images):
        parser.error("No input images found")
    for mode in modes:
        destination = args.output / mode
        if destination.exists() and any(destination.iterdir()):
            parser.error("Output already contains a run; choose a new --output: %s" % destination)
    from ultralytics import YOLO, __version__
    import torch
    model = YOLO(str(args.weights.resolve()))
    if model.task != "obb" or set(model.names.values()) != {"stone"}:
        parser.error("Expected a single-class stone OBB checkpoint, not generic pretrained weights")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(checkpoint_sha256=sha256(args.weights), ultralytics=__version__,
                    python=platform.python_version(), torch=torch.__version__, platform=platform.platform(),
                    arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                    timing_scope="demo wall time including image decoding and postprocessing; not benchmark cost")
    for mode in modes:
        out = args.output / mode
        out.mkdir(parents=True, exist_ok=True)
        records = []
        for index, path in enumerate(images):
            began = time.perf_counter()
            with Image.open(path) as source:
                image = source.convert("RGB")
            detections = infer(model, image, args, mode)
            records.append(dict(image_id=str(path.resolve()), image_sha256=sha256(path), width=image.width, height=image.height,
                                detections=detections, seconds=time.perf_counter() - began))
            draw = ImageDraw.Draw(image)
            for item in detections:
                poly = item["obb_xyxyxyxy"]
                points = list(zip(poly[::2], poly[1::2]))
                draw.line(points + points[:1], fill="lime", width=2)
            image.save(out / ("%04d_%s.png" % (index, path.stem)))
            (out / "predictions.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
            print(f"{mode}: {index + 1}/{len(images)} images", flush=True)
        (out / "run_info.json").write_text(json.dumps(dict(metadata, mode=mode, image_count=len(records),
            detection_count=sum(len(record['detections']) for record in records)), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
