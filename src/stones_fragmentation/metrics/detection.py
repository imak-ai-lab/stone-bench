"""Compute StoneBench detection metrics from local predictions.

Inputs:
- prepared YOLO OBB dataset directories with ``data.yaml``;
- prediction JSON files written in the parent-project format:
  ``[{image_id, detections: [{label, score, bbox_xywh, obb_xyxyxyxy}]}]``.

The metric follows the paper protocol: a unified confidence post-filter,
greedy per-image matching in descending confidence order, and polygon IoU at
0.50 and 0.75.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


DEFAULT_IOU_THRESHOLDS = (0.50, 0.75)


@dataclass
class Detection:
    label: str
    score: float
    polygon: list[float]


@dataclass
class ImageSample:
    image_id: str
    image_path: Path
    label_path: Path
    width: int
    height: int
    ground_truth: list[Detection]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute detection metrics for StoneBench predictions.")
    parser.add_argument("--predictions", required=True, help="predictions.json file or directory containing them.")
    parser.add_argument("--dataset-root", default="data/prepared/stonebench/yolo_obb")
    parser.add_argument("--data-yaml", default="", help="Explicit dataset data.yaml for single-file evaluation.")
    parser.add_argument("--split", default="val", help="Dataset split to evaluate: val, test, or train.")
    parser.add_argument("--output", default="results/tables/detection_metrics.csv")
    parser.add_argument("--json-output", default="", help="Optional detailed JSON output path.")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--iou-thresholds", nargs="+", type=float, default=list(DEFAULT_IOU_THRESHOLDS))
    parser.add_argument("--class-aware", action="store_true", default=True)
    parser.add_argument("--class-agnostic", dest="class_aware", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prediction_files = discover_prediction_files(resolve_path(Path(args.predictions)))
    if not prediction_files:
        raise SystemExit("No predictions.json files found: %s" % args.predictions)

    rows = []
    details = []
    for prediction_path in prediction_files:
        dataset_slug = infer_dataset_slug(prediction_path)
        data_yaml = resolve_data_yaml(args, dataset_slug)
        samples = load_yolo_obb_dataset(data_yaml, split=args.split)
        predictions = load_predictions(prediction_path)
        result = evaluate_dataset(
            samples=samples,
            predictions=predictions,
            iou_thresholds=args.iou_thresholds,
            confidence_threshold=args.confidence_threshold,
            class_aware=args.class_aware,
        )
        result.update(
            {
                "dataset": dataset_slug,
                "predictions_path": str(prediction_path),
                "data_yaml": str(data_yaml),
                "split": args.split,
            }
        )
        rows.append(flatten_result(result, args.iou_thresholds))
        details.append(result)

    write_csv(resolve_path(Path(args.output)), rows)
    json_output = resolve_path(Path(args.json_output)) if args.json_output else resolve_path(Path(args.output)).with_suffix(".json")
    save_json(json_output, details)
    print("wrote: %s" % resolve_path(Path(args.output)))
    print("wrote: %s" % json_output)
    return 0


def evaluate_dataset(
    samples: Sequence[ImageSample],
    predictions: dict[str, list[Detection]],
    iou_thresholds: Sequence[float],
    confidence_threshold: float,
    class_aware: bool,
) -> dict[str, Any]:
    total_gt = sum(len(sample.ground_truth) for sample in samples)
    filtered_predictions = {
        key: [item for item in detections if item.score >= confidence_threshold]
        for key, detections in predictions.items()
    }
    total_pred = 0
    images_with_predictions = 0
    metrics = {
        "num_images": len(samples),
        "num_ground_truth": total_gt,
        "confidence_threshold": confidence_threshold,
        "class_aware": class_aware,
        "iou": {},
    }

    for threshold in iou_thresholds:
        tp = fp = fn = 0
        matched_ious = []
        for sample in samples:
            sample_predictions = lookup_predictions(filtered_predictions, sample)
            if threshold == iou_thresholds[0]:
                total_pred += len(sample_predictions)
                if sample_predictions:
                    images_with_predictions += 1
            match = match_image(sample.ground_truth, sample_predictions, threshold, class_aware)
            tp += match["tp"]
            fp += match["fp"]
            fn += match["fn"]
            matched_ious.extend(match["ious"])
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2.0 * precision * recall, precision + recall)
        metrics["iou"]["%.2f" % threshold] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "mean_matched_iou": safe_divide(sum(matched_ious), len(matched_ious)),
        }

    metrics["num_predictions"] = total_pred
    metrics["avg_predictions_per_image"] = safe_divide(total_pred, len(samples))
    metrics["coverage"] = safe_divide(images_with_predictions, len(samples))
    return metrics


def match_image(
    ground_truth: Sequence[Detection],
    predictions: Sequence[Detection],
    iou_threshold: float,
    class_aware: bool,
) -> dict[str, Any]:
    matched_gt = set()
    tp = fp = 0
    matched_ious = []
    for prediction in sorted(predictions, key=lambda item: item.score, reverse=True):
        best_index = None
        best_iou = 0.0
        for gt_index, gt in enumerate(ground_truth):
            if gt_index in matched_gt:
                continue
            if class_aware and prediction.label != gt.label:
                continue
            iou = polygon_iou(prediction.polygon, gt.polygon)
            if iou > best_iou:
                best_index = gt_index
                best_iou = iou
        if best_index is not None and best_iou >= iou_threshold:
            matched_gt.add(best_index)
            tp += 1
            matched_ious.append(best_iou)
        else:
            fp += 1
    return {"tp": tp, "fp": fp, "fn": len(ground_truth) - len(matched_gt), "ious": matched_ious}


def load_yolo_obb_dataset(data_yaml: Path, split: str) -> list[ImageSample]:
    config = load_yaml(data_yaml)
    class_names = normalize_class_names(config.get("names", {0: "stone"}))
    split_file = resolve_dataset_path(data_yaml.parent, config, Path(str(config.get(split, "%s.txt" % split))))
    if not split_file.exists():
        raise SystemExit("Split file not found: %s" % split_file)
    samples = []
    for image_ref in read_lines(split_file):
        image_path = resolve_image_path(data_yaml.parent, config, image_ref)
        if image_path is None:
            raise SystemExit("Image not found from %s: %s" % (split_file, image_ref))
        width, height = read_image_size(image_path)
        label_path = label_path_for_image(data_yaml.parent, image_path)
        ground_truth = load_yolo_obb_labels(label_path, width, height, class_names)
        samples.append(
            ImageSample(
                image_id=str(image_path),
                image_path=image_path,
                label_path=label_path,
                width=width,
                height=height,
                ground_truth=ground_truth,
            )
        )
    return samples


def load_yolo_obb_labels(label_path: Path, width: int, height: int, class_names: dict[int, str]) -> list[Detection]:
    if not label_path.exists():
        return []
    detections = []
    for line in read_lines(label_path):
        parts = line.split()
        if len(parts) != 9:
            continue
        class_id = int(float(parts[0]))
        coords = [float(value) for value in parts[1:]]
        polygon = []
        for index in range(0, len(coords), 2):
            polygon.extend([coords[index] * width, coords[index + 1] * height])
        detections.append(Detection(label=class_names.get(class_id, str(class_id)), score=1.0, polygon=polygon))
    return detections


def load_predictions(path: Path) -> dict[str, list[Detection]]:
    raw = load_json(path)
    if isinstance(raw, dict):
        if "predictions" in raw:
            raw = raw["predictions"]
        else:
            raw = [{"image_id": key, "detections": value} for key, value in raw.items()]
    predictions = {}
    for image_record in raw:
        image_id = str(image_record.get("image_id") or image_record.get("file_name") or image_record.get("image_path"))
        detections = []
        for item in image_record.get("detections", []):
            polygon = prediction_polygon(item)
            if not polygon:
                continue
            detections.append(
                Detection(
                    label=str(item.get("label") or item.get("class") or item.get("class_name") or "stone"),
                    score=float(item.get("score", item.get("confidence", 1.0))),
                    polygon=polygon,
                )
            )
        predictions[normalize_key(image_id)] = detections
    return predictions


def prediction_polygon(item: dict[str, Any]) -> list[float]:
    obb = item.get("obb_xyxyxyxy") or item.get("polygon") or item.get("points")
    if obb and len(obb) >= 8:
        return [float(value) for value in obb[:8]]
    bbox = item.get("bbox_xywh") or item.get("bbox")
    if bbox and len(bbox) == 4:
        x_min, y_min, width, height = [float(value) for value in bbox]
        return [x_min, y_min, x_min + width, y_min, x_min + width, y_min + height, x_min, y_min + height]
    return []


def lookup_predictions(predictions: dict[str, list[Detection]], sample: ImageSample) -> list[Detection]:
    keys = [
        sample.image_id,
        str(sample.image_path),
        sample.image_path.as_posix(),
        sample.image_path.name,
        sample.image_path.stem,
    ]
    for key in keys:
        normalized = normalize_key(key)
        if normalized in predictions:
            return predictions[normalized]
    return []


def polygon_iou(left: Sequence[float], right: Sequence[float]) -> float:
    left_points = polygon_points(left)
    right_points = polygon_points(right)
    intersection = convex_clip(left_points, right_points)
    if not intersection:
        return 0.0
    intersection_area = polygon_area_points(intersection)
    union = polygon_area_points(left_points) + polygon_area_points(right_points) - intersection_area
    return safe_divide(intersection_area, union)


def convex_clip(subject: list[tuple[float, float]], clip: list[tuple[float, float]]) -> list[tuple[float, float]]:
    output = list(subject)
    if polygon_signed_area(clip) < 0:
        clip = list(reversed(clip))
    for index, current in enumerate(clip):
        previous = clip[index - 1]
        input_points = output
        output = []
        if not input_points:
            break
        start = input_points[-1]
        for end in input_points:
            if inside(end, previous, current):
                if not inside(start, previous, current):
                    output.append(line_intersection(start, end, previous, current))
                output.append(end)
            elif inside(start, previous, current):
                output.append(line_intersection(start, end, previous, current))
            start = end
    return output


def inside(point: tuple[float, float], edge_start: tuple[float, float], edge_end: tuple[float, float]) -> bool:
    return (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) - (
        edge_end[1] - edge_start[1]
    ) * (point[0] - edge_start[0]) >= -1e-9


def line_intersection(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> tuple[float, float]:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-12:
        return p2
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
    return (px, py)


def polygon_points(flat: Sequence[float]) -> list[tuple[float, float]]:
    return [(float(flat[index]), float(flat[index + 1])) for index in range(0, len(flat), 2)]


def polygon_area_points(points: Sequence[tuple[float, float]]) -> float:
    return abs(polygon_signed_area(points))


def polygon_signed_area(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return area / 2.0


def flatten_result(result: dict[str, Any], thresholds: Sequence[float]) -> dict[str, Any]:
    row = {
        "dataset": result["dataset"],
        "split": result["split"],
        "num_images": result["num_images"],
        "num_ground_truth": result["num_ground_truth"],
        "num_predictions": result["num_predictions"],
        "coverage": result["coverage"],
        "confidence_threshold": result["confidence_threshold"],
        "predictions_path": result["predictions_path"],
        "data_yaml": result["data_yaml"],
    }
    for threshold in thresholds:
        key = "%.2f" % threshold
        metric = result["iou"][key]
        suffix = str(int(round(threshold * 100)))
        row.update(
            {
                "precision_%s" % suffix: metric["precision"],
                "recall_%s" % suffix: metric["recall"],
                "f1_%s" % suffix: metric["f1"],
                "tp_%s" % suffix: metric["tp"],
                "fp_%s" % suffix: metric["fp"],
                "fn_%s" % suffix: metric["fn"],
                "mean_iou_%s" % suffix: metric["mean_matched_iou"],
            }
        )
    return row


def discover_prediction_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("predictions.json")) if path.exists() else []


def infer_dataset_slug(prediction_path: Path) -> str:
    return prediction_path.parent.name


def resolve_data_yaml(args: argparse.Namespace, dataset_slug: str) -> Path:
    if args.data_yaml:
        return resolve_path(Path(args.data_yaml))
    root = resolve_path(Path(args.dataset_root))
    candidates = [root / dataset_slug / "data.yaml", root / "data.yaml"]
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit("Cannot find data.yaml for dataset %s under %s" % (dataset_slug, root))


def label_path_for_image(dataset_root: Path, image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        index = parts.index("images")
        label_parts = parts[:]
        label_parts[index] = "labels"
        return Path(*label_parts).with_suffix(".txt")
    return (dataset_root / "labels" / image_path.name).with_suffix(".txt")


def resolve_dataset_path(dataset_root: Path, config: dict[str, Any], path: Path) -> Path:
    if path.is_absolute():
        return path
    base = Path(str(config.get("path", ".")))
    candidates = [dataset_root / base / path, dataset_root / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_image_path(dataset_root: Path, config: dict[str, Any], image_ref: str) -> Path | None:
    path = Path(image_ref)
    if path.is_absolute() and path.exists():
        return path.resolve()
    base = Path(str(config.get("path", ".")))
    candidates = [dataset_root / base / path, dataset_root / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def read_image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except ImportError as exc:
        raise SystemExit("Pillow is required to read image sizes.") from exc


def normalize_class_names(raw_names: Any) -> dict[int, str]:
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    return {index: str(value) for index, value in enumerate(raw_names)}


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    leading = ["dataset", "split", "num_images", "num_ground_truth", "num_predictions"]
    fieldnames = [key for key in leading if key in fieldnames] + [key for key in fieldnames if key not in leading]
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def normalize_key(value: str) -> str:
    return str(value).replace("\\", "/")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj) or {}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as file_obj:
        return [line.strip() for line in file_obj if line.strip()]


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
