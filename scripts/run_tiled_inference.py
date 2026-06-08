"""Local tiled supervised pipeline.

Pipeline stages:
- ``prepare``: cut full images into tiles and write YOLO tile datasets;
- ``train``: train an Ultralytics model on the tile dataset;
- ``predict``: run tile inference and shift predictions back to full-image coordinates;
- ``evaluate``: call ``evaluate_detection.py`` on stitched full-frame predictions;
- ``all``: run every stage.

The implementation is local-only and config-first. It is intentionally explicit
so the generated tile manifests are easy to inspect and debug.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import train_supervised
from stones_fragmentation.metrics import detection as evaluate_detection


COMBINED_SLUG = "all_real_plus_synthetic_train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local tiled supervised pipeline.")
    parser.add_argument("--config", required=True, help="YAML config from configs/tiling.")
    parser.add_argument("--stage", choices=("plan", "prepare", "train", "predict", "evaluate", "all"), default="plan")
    parser.add_argument("--source-root", default="", help="Override full-frame prepared dataset root.")
    parser.add_argument("--prepared-root", default="", help="Override tile output root.")
    parser.add_argument("--output-root", default="", help="Override run output root.")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--scope", choices=("combined", "per-dataset"), default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--tile-overlap", type=int, default=None)
    parser.add_argument("--highres-min-side", type=int, default=None)
    parser.add_argument("--copy-mode", choices=("copy", "hardlink", "symlink", "none"), default="copy")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = resolve_config(load_config(resolve_path(Path(args.config))), args)
    plan = build_plan(config, args)
    output_root = Path(plan["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    save_json(output_root / "tiled_pipeline_plan.json", plan)

    if args.stage == "plan" or args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    if args.stage in ("prepare", "all"):
        prepare_tiled_datasets(config, plan, copy_mode=args.copy_mode)
    if args.stage in ("train", "all"):
        run_training(config, plan)
    if args.stage in ("predict", "all"):
        run_prediction(config, plan)
    if args.stage in ("evaluate", "all"):
        run_evaluation(config, plan)
    return 0


def resolve_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    model = config.setdefault("model", {})
    training = config.setdefault("training", {})
    data = config.setdefault("data", {})
    tiling = config.setdefault("tiling", {})
    output = config.setdefault("output", {})

    if args.source_root:
        data["prepared_root"] = args.source_root
    if args.scope:
        data["scope"] = args.scope
    if args.output_root:
        output["root"] = args.output_root
    if args.prepared_root:
        tiling["prepared_root"] = args.prepared_root
    if args.device:
        training["device"] = args.device
    if args.tile_size is not None:
        tiling["tile_size"] = args.tile_size
    if args.tile_overlap is not None:
        tiling["tile_overlap"] = args.tile_overlap
    if args.highres_min_side is not None:
        tiling["highres_min_side"] = args.highres_min_side

    model.setdefault("family", "yolo-obb")
    model.setdefault("model_id", "yolo11x-obb.pt")
    model.setdefault("label_format", "yolo_obb")
    training.setdefault("epochs", 30)
    training.setdefault("imgsz", 1024)
    training.setdefault("batch", 4)
    training.setdefault("seed", 42)
    training.setdefault("patience", 10)
    training.setdefault("amp", True)
    training.setdefault("device", "cuda")
    training.setdefault("workers", 8)
    data.setdefault("prepared_root", "data/prepared/stonebench/yolo_obb")
    data.setdefault("scope", "combined")
    data.setdefault("combined_slug", COMBINED_SLUG)
    tiling.setdefault("prepared_root", "data/prepared/stonebench_tiled")
    tiling.setdefault("tile_size", 1024)
    tiling.setdefault("tile_overlap", 0)
    tiling.setdefault("highres_min_side", 2048)
    tiling.setdefault("min_retained_area", 0.35)
    tiling.setdefault("nms_iou", 0.50)
    tiling.setdefault("nms_class_aware", True)
    tiling.setdefault("max_detections_per_image", 5000)
    tiling.setdefault("max_detections_per_tile", 3000)
    tiling.setdefault("predict_conf", 0.001)
    tiling.setdefault("predict_iou", 0.70)
    output.setdefault("root", "outputs/supervised_tiled_benchmark")
    return config


def build_plan(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    data = config["data"]
    tiling = config["tiling"]
    source_root = resolve_path(Path(str(data["prepared_root"])))
    tile_root = resolve_path(Path(str(tiling["prepared_root"]))) / tile_run_slug(config) / str(config["model"]["label_format"])
    output_root = resolve_path(Path(str(config["output"]["root"])))
    datasets = discover_dataset_dirs(
        source_root=source_root,
        scope=str(data["scope"]),
        selected=args.datasets,
        combined_slug=str(data.get("combined_slug", COMBINED_SLUG)),
    )
    return {
        "source_root": str(source_root),
        "tile_root": str(tile_root),
        "output_root": str(output_root),
        "datasets": [
            {
                "dataset": path.name,
                "source_data_yaml": str(path / "data.yaml"),
                "tile_data_yaml": str(tile_root / path.name / "data.yaml"),
                "tile_manifest": str(tile_root / path.name / "tile_manifest.json"),
                "run_dir": str(output_root / path.name),
            }
            for path in datasets
        ],
        "model": config["model"],
        "training": config["training"],
        "tiling": config["tiling"],
    }


def prepare_tiled_datasets(config: dict[str, Any], plan: dict[str, Any], copy_mode: str) -> None:
    for item in plan["datasets"]:
        source_dir = Path(item["source_data_yaml"]).parent
        tile_dir = Path(item["tile_data_yaml"]).parent
        prepare_one_dataset(source_dir=source_dir, tile_dir=tile_dir, config=config, copy_mode=copy_mode)


def prepare_one_dataset(source_dir: Path, tile_dir: Path, config: dict[str, Any], copy_mode: str) -> None:
    source_config = evaluate_detection.load_yaml(source_dir / "data.yaml")
    class_names = evaluate_detection.normalize_class_names(source_config.get("names", {0: "stone"}))
    tile_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    split_entries = {}
    for split in ("train", "val"):
        split_path = evaluate_detection.resolve_dataset_path(source_dir, source_config, Path(str(source_config.get(split, "%s.txt" % split))))
        entries = []
        if not split_path.exists():
            write_lines(tile_dir / ("%s.txt" % split), [])
            split_entries[split] = []
            continue
        for image_ref in evaluate_detection.read_lines(split_path):
            image_path = evaluate_detection.resolve_image_path(source_dir, source_config, image_ref)
            if image_path is None:
                continue
            width, height = evaluate_detection.read_image_size(image_path)
            if max(width, height) < int(config["tiling"]["highres_min_side"]):
                continue
            label_path = evaluate_detection.label_path_for_image(source_dir, image_path)
            labels = read_labels(label_path, width, height, str(config["model"]["label_format"]), class_names)
            tile_items = write_image_tiles(
                image_path=image_path,
                labels=labels,
                split=split,
                tile_dir=tile_dir,
                config=config,
                copy_mode=copy_mode,
            )
            entries.extend(item["tile_path"] for item in tile_items)
            manifest.extend(tile_items)
        write_lines(tile_dir / ("%s.txt" % split), entries)
        split_entries[split] = entries

    data_yaml = {
        "path": str(tile_dir.resolve()).replace("\\", "/"),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": class_names,
    }
    write_yaml(tile_dir / "data.yaml", data_yaml)
    write_lines(tile_dir / "test.txt", [])
    save_json(tile_dir / "tile_manifest.json", manifest)
    save_json(
        tile_dir / "dataset_summary.json",
        {
            "source_data_yaml": str(source_dir / "data.yaml"),
            "tile_size": int(config["tiling"]["tile_size"]),
            "tile_overlap": int(config["tiling"]["tile_overlap"]),
            "splits": {split: len(entries) for split, entries in split_entries.items()},
        },
    )


def write_image_tiles(
    image_path: Path,
    labels: list[dict[str, Any]],
    split: str,
    tile_dir: Path,
    config: dict[str, Any],
    copy_mode: str,
) -> list[dict[str, Any]]:
    from PIL import Image

    tile_size = int(config["tiling"]["tile_size"])
    overlap = int(config["tiling"]["tile_overlap"])
    stride = max(1, tile_size - overlap)
    min_retained = float(config["tiling"]["min_retained_area"])
    label_format = str(config["model"]["label_format"])
    items = []
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        for y0 in tile_starts(height, tile_size, stride):
            for x0 in tile_starts(width, tile_size, stride):
                tile_w = min(tile_size, width - x0)
                tile_h = min(tile_size, height - y0)
                clipped = [item for item in (clip_label(label, x0, y0, tile_w, tile_h, min_retained, label_format) for label in labels) if item]
                if split == "train" and not clipped:
                    continue
                stem = "%s__x%d_y%d_w%d_h%d" % (image_path.stem, x0, y0, tile_w, tile_h)
                tile_image = tile_dir / "images" / split / image_path.parent.name / (stem + image_path.suffix)
                tile_label = tile_dir / "labels" / split / image_path.parent.name / (stem + ".txt")
                materialize_tile(image, x0, y0, tile_w, tile_h, tile_image, copy_mode)
                write_tile_label(tile_label, clipped, tile_w, tile_h, label_format)
                items.append(
                    {
                        "source_image_id": str(image_path),
                        "source_image_path": str(image_path),
                        "source_width": width,
                        "source_height": height,
                        "split": split,
                        "x": x0,
                        "y": y0,
                        "width": tile_w,
                        "height": tile_h,
                        "tile_path": str(tile_image.resolve()).replace("\\", "/"),
                        "objects": len(clipped),
                    }
                )
    return items


def run_training(config: dict[str, Any], plan: dict[str, Any]) -> None:
    train_config = json.loads(json.dumps(config))
    train_config["data"]["prepared_root"] = str(Path(plan["tile_root"]))
    train_config["output"]["root"] = str(Path(plan["output_root"]) / "training")
    temp_config = Path(plan["output_root"]) / "resolved_tiled_train_config.yaml"
    write_yaml(temp_config, train_config)
    # Reuse the supervised trainer process to keep behavior identical.
    command = [sys.executable, str(SCRIPTS / "train_supervised.py"), "--config", str(temp_config)]
    subprocess.run(command, cwd=str(ROOT), check=True)


def run_prediction(config: dict[str, Any], plan: dict[str, Any]) -> None:
    model_path = str(config["model"].get("weights") or config["model"].get("model_id"))
    model = build_ultralytics_model(str(config["model"]["family"]), model_path)
    for item in plan["datasets"]:
        tile_dir = Path(item["tile_data_yaml"]).parent
        manifest = [row for row in load_json(tile_dir / "tile_manifest.json") if row.get("split") == "val"]
        grouped: dict[str, list[dict[str, Any]]] = {}
        for tile in manifest:
            results = model.predict(
                source=str(tile["tile_path"]),
                imgsz=int(config["training"]["imgsz"]),
                conf=float(config["tiling"]["predict_conf"]),
                iou=float(config["tiling"]["predict_iou"]),
                device=str(config["training"].get("device", "cuda")),
                max_det=int(config["tiling"]["max_detections_per_tile"]),
                verbose=False,
            )
            detections = parse_ultralytics_result(results[0] if results else None, tile, str(config["model"]["family"]))
            grouped.setdefault(str(tile["source_image_path"]), []).extend(detections)
        prediction_payload = []
        for image_id, detections in sorted(grouped.items()):
            kept = nms(detections, float(config["tiling"]["nms_iou"]), bool(config["tiling"]["nms_class_aware"]))
            kept = kept[: int(config["tiling"]["max_detections_per_image"])]
            prediction_payload.append({"image_id": image_id, "detections": kept})
        run_dir = Path(item["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        save_json(run_dir / "predictions.json", prediction_payload)


def run_evaluation(config: dict[str, Any], plan: dict[str, Any]) -> None:
    for item in plan["datasets"]:
        run_dir = Path(item["run_dir"])
        predictions = run_dir / "predictions.json"
        if not predictions.exists():
            continue
        command = [
            sys.executable,
            str(SCRIPTS / "evaluate_detection.py"),
            "--predictions",
            str(predictions),
            "--data-yaml",
            str(item["source_data_yaml"]),
            "--output",
            str(run_dir / "detection_metrics.csv"),
        ]
        subprocess.run(command, cwd=str(ROOT), check=True)


def read_labels(label_path: Path, width: int, height: int, label_format: str, class_names: dict[int, str]) -> list[dict[str, Any]]:
    if not label_path.exists():
        return []
    labels = []
    for line in evaluate_detection.read_lines(label_path):
        parts = line.split()
        if not parts:
            continue
        class_id = int(float(parts[0]))
        values = [float(value) for value in parts[1:]]
        if len(values) == 4:
            polygon = yolo_detect_to_polygon(values, width, height)
        elif len(values) >= 8 and len(values) % 2 == 0:
            polygon = []
            for index in range(0, len(values), 2):
                polygon.extend([values[index] * width, values[index + 1] * height])
        else:
            continue
        labels.append({"class_id": class_id, "label": class_names.get(class_id, str(class_id)), "polygon": polygon})
    return labels


def clip_label(
    label: dict[str, Any],
    x0: int,
    y0: int,
    tile_w: int,
    tile_h: int,
    min_retained: float,
    label_format: str,
) -> dict[str, Any] | None:
    bbox = polygon_bbox(label["polygon"])
    x_min = max(float(x0), bbox[0])
    y_min = max(float(y0), bbox[1])
    x_max = min(float(x0 + tile_w), bbox[2])
    y_max = min(float(y0 + tile_h), bbox[3])
    if x_max <= x_min or y_max <= y_min:
        return None
    source_area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
    clipped_area = (x_max - x_min) * (y_max - y_min)
    if source_area <= 0 or clipped_area / source_area < min_retained:
        return None
    local_polygon = [x_min - x0, y_min - y0, x_max - x0, y_min - y0, x_max - x0, y_max - y0, x_min - x0, y_max - y0]
    return {"class_id": label["class_id"], "label": label["label"], "polygon": local_polygon}


def write_tile_label(path: Path, labels: Sequence[dict[str, Any]], width: int, height: int, label_format: str) -> None:
    lines = []
    for label in labels:
        polygon = label["polygon"]
        if label_format == "yolo_detect":
            x_min, y_min, x_max, y_max = polygon_bbox(polygon)
            values = [
                ((x_min + x_max) / 2.0) / width,
                ((y_min + y_max) / 2.0) / height,
                (x_max - x_min) / width,
                (y_max - y_min) / height,
            ]
        else:
            values = []
            for index in range(0, len(polygon), 2):
                values.extend([polygon[index] / width, polygon[index + 1] / height])
        lines.append("%d %s" % (label["class_id"], " ".join("%.6f" % clamp01(value) for value in values)))
    write_lines(path, lines)


def parse_ultralytics_result(result: Any, tile: dict[str, Any], family: str) -> list[dict[str, Any]]:
    if result is None:
        return []
    detections = []
    x_offset = float(tile["x"])
    y_offset = float(tile["y"])
    if family == "yolo-obb" and getattr(result, "obb", None) is not None:
        points_batch = result.obb.xyxyxyxy.cpu().tolist()
        scores = result.obb.conf.cpu().tolist()
        classes = result.obb.cls.cpu().tolist()
        for points, score, class_id in zip(points_batch, scores, classes):
            polygon = []
            for point in points:
                polygon.extend([float(point[0]) + x_offset, float(point[1]) + y_offset])
            detections.append({"label": "stone", "score": float(score), "obb_xyxyxyxy": polygon, "metadata": {"class_id": int(class_id)}})
        return detections
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    boxes_xyxy = boxes.xyxy.cpu().tolist() if boxes.xyxy is not None else []
    scores = boxes.conf.cpu().tolist() if boxes.conf is not None else []
    classes = boxes.cls.cpu().tolist() if boxes.cls is not None else []
    for box, score, class_id in zip(boxes_xyxy, scores, classes):
        x_min = float(box[0]) + x_offset
        y_min = float(box[1]) + y_offset
        x_max = float(box[2]) + x_offset
        y_max = float(box[3]) + y_offset
        detections.append(
            {
                "label": "stone",
                "score": float(score),
                "bbox_xywh": [x_min, y_min, x_max - x_min, y_max - y_min],
                "obb_xyxyxyxy": [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max],
                "metadata": {"class_id": int(class_id)},
            }
        )
    return detections


def nms(detections: Sequence[dict[str, Any]], iou_threshold: float, class_aware: bool) -> list[dict[str, Any]]:
    kept = []
    for detection in sorted(detections, key=lambda item: float(item.get("score", 0.0)), reverse=True):
        duplicate = False
        for previous in kept:
            if class_aware and detection.get("label") != previous.get("label"):
                continue
            if evaluate_detection.polygon_iou(prediction_polygon(detection), prediction_polygon(previous)) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(detection)
    return kept


def prediction_polygon(detection: dict[str, Any]) -> list[float]:
    return evaluate_detection.prediction_polygon(detection)


def build_ultralytics_model(family: str, model_id: str) -> Any:
    try:
        from ultralytics import RTDETR, YOLO
    except ImportError as exc:
        raise RuntimeError("Install Ultralytics first: python -m pip install -e .[yolo]") from exc
    return RTDETR(model_id) if family == "rtdetr" else YOLO(model_id)


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def materialize_tile(image: Any, x0: int, y0: int, width: int, height: int, destination: Path, copy_mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    # Tiles are cropped images, so copy/hardlink/symlink modes only apply to full-frame preparers.
    tile = image.crop((x0, y0, x0 + width, y0 + height))
    tile.save(destination)


def polygon_bbox(polygon: Sequence[float]) -> tuple[float, float, float, float]:
    xs = [float(polygon[index]) for index in range(0, len(polygon), 2)]
    ys = [float(polygon[index]) for index in range(1, len(polygon), 2)]
    return min(xs), min(ys), max(xs), max(ys)


def yolo_detect_to_polygon(values: Sequence[float], width: int, height: int) -> list[float]:
    x_center, y_center, box_w, box_h = [float(value) for value in values]
    x_min = (x_center - box_w / 2.0) * width
    x_max = (x_center + box_w / 2.0) * width
    y_min = (y_center - box_h / 2.0) * height
    y_max = (y_center + box_h / 2.0) * height
    return [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]


def discover_dataset_dirs(source_root: Path, scope: str, selected: Sequence[str] | None, combined_slug: str) -> list[Path]:
    selected_set = set(selected or [])
    if (source_root / "data.yaml").exists():
        return [source_root]
    if scope == "combined":
        candidate = source_root / combined_slug
        return [candidate] if (candidate / "data.yaml").exists() else []
    datasets = []
    for path in sorted(source_root.iterdir() if source_root.exists() else []):
        if not path.is_dir() or path.name == combined_slug:
            continue
        if selected_set and path.name not in selected_set:
            continue
        if (path / "data.yaml").exists():
            datasets.append(path)
    return datasets


def tile_run_slug(config: dict[str, Any]) -> str:
    tiling = config["tiling"]
    return "tile%d_overlap%d_minside%d" % (
        int(tiling["tile_size"]),
        int(tiling["tile_overlap"]),
        int(tiling["highres_min_side"]),
    )


def load_config(path: Path) -> dict[str, Any]:
    config = load_yaml(path)
    inherited = config.pop("inherits", None)
    if inherited:
        base = load_config(resolve_path(Path(str(inherited))))
        return deep_merge(base, config)
    return config


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(payload, file_obj, allow_unicode=True, sort_keys=False)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for line in lines:
            file_obj.write(str(line).replace("\\", "/") + "\n")


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


if __name__ == "__main__":
    raise SystemExit(main())
