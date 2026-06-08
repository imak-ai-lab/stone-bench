"""Export Ultralytics YOLO OBB predictions to StoneBench predictions.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO OBB prediction and export predictions.json.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=3000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is required: python -m pip install ultralytics") from exc

    data_yaml = resolve_path(Path(args.data_yaml))
    image_paths = load_split_images(data_yaml, args.split)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    model = YOLO(str(resolve_path(Path(args.weights))))
    payload = []
    for image_path in image_paths:
        results = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
        )
        payload.append(
            {
                "image_id": str(image_path.resolve()),
                "detections": parse_result(results[0] if results else None),
            }
        )

    output = resolve_path(Path(args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote: %s" % output)
    print("images: %d" % len(payload))
    print("detections: %d" % sum(len(item["detections"]) for item in payload))
    return 0


def parse_result(result: Any) -> list[dict[str, Any]]:
    if result is None:
        return []
    detections = []
    if getattr(result, "obb", None) is not None and result.obb is not None:
        points_batch = result.obb.xyxyxyxy.cpu().tolist()
        scores = result.obb.conf.cpu().tolist()
        classes = result.obb.cls.cpu().tolist()
        for points, score, class_id in zip(points_batch, scores, classes):
            polygon = []
            for point in points:
                polygon.extend([float(point[0]), float(point[1])])
            detections.append(
                {
                    "label": "stone",
                    "score": float(score),
                    "obb_xyxyxyxy": polygon,
                    "metadata": {"class_id": int(class_id), "adapter": "ultralytics-yolo-obb"},
                }
            )
        return detections

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = boxes.xyxy.cpu().tolist() if boxes.xyxy is not None else []
    scores = boxes.conf.cpu().tolist() if boxes.conf is not None else []
    classes = boxes.cls.cpu().tolist() if boxes.cls is not None else []
    for box, score, class_id in zip(xyxy, scores, classes):
        x_min, y_min, x_max, y_max = [float(value) for value in box[:4]]
        detections.append(
            {
                "label": "stone",
                "score": float(score),
                "bbox_xywh": [x_min, y_min, x_max - x_min, y_max - y_min],
                "obb_xyxyxyxy": [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max],
                "metadata": {"class_id": int(class_id), "adapter": "ultralytics-yolo-box"},
            }
        )
    return detections


def load_split_images(data_yaml: Path, split: str) -> list[Path]:
    config = load_yaml(data_yaml)
    split_file = resolve_dataset_path(data_yaml.parent, config, Path(str(config.get(split, "%s.txt" % split))))
    if not split_file.exists():
        raise SystemExit("Split file not found: %s" % split_file)
    paths = []
    for image_ref in read_lines(split_file):
        image_path = resolve_image_path(data_yaml.parent, config, image_ref)
        if image_path is None:
            raise SystemExit("Image not found from %s: %s" % (split_file, image_ref))
        paths.append(image_path)
    return paths


def resolve_dataset_path(dataset_root: Path, config: dict[str, Any], path: Path) -> Path:
    if path.is_absolute():
        return path
    base = Path(str(config.get("path", ".")))
    for candidate in (dataset_root / base / path, dataset_root / path):
        if candidate.exists():
            return candidate
    return dataset_root / path


def resolve_image_path(dataset_root: Path, config: dict[str, Any], image_ref: str) -> Path | None:
    path = Path(image_ref)
    if path.is_absolute() and path.exists():
        return path.resolve()
    base = Path(str(config.get("path", ".")))
    for candidate in (dataset_root / base / path, dataset_root / path, Path.cwd() / path):
        if candidate.exists():
            return candidate.resolve()
    return None


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj) or {}


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as file_obj:
        return [line.strip() for line in file_obj if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
