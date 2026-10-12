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
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

import yaml


DEFAULT_IOU_THRESHOLDS = (0.50, 0.75)
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_BOOTSTRAP_SEED = 42


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
    parser.add_argument("--config", default="configs/evaluation.yaml", help="Versioned evaluation YAML.")
    parser.add_argument("--predictions", default=None, help="predictions.json file or directory containing them.")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--data-yaml", default="", help="Explicit dataset data.yaml for single-file evaluation.")
    parser.add_argument("--split", default=None, help="Dataset split to evaluate: val, test, or train.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--json-output", default="", help="Optional detailed JSON output path.")
    parser.add_argument(
        "--confidence-threshold",
        default=None,
        help="Numeric post-filter or 'from_resolved_config' for each prediction file.",
    )
    parser.add_argument("--iou-thresholds", nargs="+", type=float, default=None)
    parser.add_argument("--class-aware", action="store_true", default=None)
    parser.add_argument("--class-agnostic", dest="class_aware", action="store_false", default=None)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=None,
        help="Image-level bootstrap replicates; use 0 to disable confidence intervals.",
    )
    parser.add_argument("--confidence-level", type=float, default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    apply_detection_config(args)
    validate_uncertainty_args(args.bootstrap_samples, args.confidence_level)
    prediction_files = discover_prediction_files(resolve_path(Path(args.predictions)))
    if not prediction_files:
        raise SystemExit("No predictions.json files found: %s" % args.predictions)
    validate_prediction_collection(prediction_files)

    rows = []
    details = []
    for prediction_path in prediction_files:
        dataset_slug = infer_dataset_slug(prediction_path)
        data_yaml = resolve_data_yaml(args, dataset_slug)
        samples = load_yolo_obb_dataset(data_yaml, split=args.split)
        predictions = load_predictions(prediction_path)
        confidence_threshold = resolve_confidence_threshold(args.confidence_threshold, prediction_path)
        result = evaluate_dataset(
            samples=samples,
            predictions=predictions,
            iou_thresholds=args.iou_thresholds,
            confidence_threshold=confidence_threshold,
            class_aware=args.class_aware,
            bootstrap_samples=args.bootstrap_samples,
            confidence_level=args.confidence_level,
            bootstrap_seed=args.bootstrap_seed,
        )
        result.update(
            {
                "dataset": dataset_slug,
                "predictions_path": display_path(prediction_path),
                "data_yaml": display_path(data_yaml),
                "split": args.split,
            }
        )
        rows.append(flatten_result(result, args.iou_thresholds))
        details.append(result)

    if len(details) > 1:
        aggregate = aggregate_dataset_results(
            details,
            args.iou_thresholds,
            args.bootstrap_samples,
            args.confidence_level,
            args.bootstrap_seed,
        )
        rows.append(flatten_result(aggregate, args.iou_thresholds))
        details.append(aggregate)

    write_csv(resolve_path(Path(args.output)), rows)
    json_output = resolve_path(Path(args.json_output)) if args.json_output else resolve_path(Path(args.output)).with_suffix(".json")
    save_json(json_output, [public_result(result) for result in details])
    print("wrote: %s" % resolve_path(Path(args.output)))
    print("wrote: %s" % json_output)
    return 0


def apply_detection_config(args: argparse.Namespace) -> None:
    config_path = resolve_path(Path(args.config))
    config = load_yaml(config_path) if config_path.exists() else {}
    dataset = config.get("dataset", {})
    detection = config.get("detection", {})
    uncertainty = config.get("uncertainty", {})
    defaults = {
        "predictions": dataset.get("predictions", "results/reference_predictions"),
        "dataset_root": dataset.get("root", "data/prepared/stonebench/yolo_obb"),
        "split": dataset.get("split", "val"),
        "output": detection.get("output", "results/tables/detection_metrics.csv"),
        "confidence_threshold": detection.get("confidence_threshold", "from_resolved_config"),
        "iou_thresholds": detection.get("iou_thresholds", list(DEFAULT_IOU_THRESHOLDS)),
        "class_aware": detection.get("class_aware", True),
        "bootstrap_samples": uncertainty.get("samples", DEFAULT_BOOTSTRAP_SAMPLES),
        "confidence_level": uncertainty.get("confidence_level", DEFAULT_CONFIDENCE_LEVEL),
        "bootstrap_seed": uncertainty.get("seed", DEFAULT_BOOTSTRAP_SEED),
    }
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def resolve_confidence_threshold(value: Any, prediction_path: Path) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if text not in ("auto", "from_resolved_config", "resolved"):
        try:
            return float(text)
        except ValueError as exc:
            raise SystemExit("Invalid confidence threshold: %s" % value) from exc
    candidates = [
        prediction_path.parent / "resolved_config.yaml",
        prediction_path.parent / "public_config.yaml",
        prediction_path.parent.parent / "public_config.yaml",
    ]
    for config_path in candidates:
        if not config_path.exists():
            continue
        config = load_yaml(config_path)
        model = config.get("model", config)
        threshold = model.get("confidence_threshold")
        if threshold is not None:
            return float(threshold)
    raise SystemExit(
        "Cannot resolve model confidence threshold next to %s; pass --confidence-threshold explicitly."
        % prediction_path
    )


def evaluate_dataset(
    samples: Sequence[ImageSample],
    predictions: dict[str, list[Detection]],
    iou_thresholds: Sequence[float],
    confidence_threshold: float,
    class_aware: bool,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    validate_uncertainty_args(bootstrap_samples, confidence_level)
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
        "num_images_with_predictions": 0,
        "confidence_threshold": confidence_threshold,
        "class_aware": class_aware,
        "uncertainty": {
            "method": "nonparametric_cluster_bootstrap" if bootstrap_samples else "disabled",
            "resampling_unit": "image",
            "samples": bootstrap_samples,
            "confidence_level": confidence_level,
            "seed": bootstrap_seed,
        },
        "iou": {},
    }

    per_threshold_matches: dict[str, list[dict[str, Any]]] = {}
    for threshold in iou_thresholds:
        image_matches = []
        for sample in samples:
            sample_predictions = lookup_predictions(filtered_predictions, sample)
            if threshold == iou_thresholds[0]:
                total_pred += len(sample_predictions)
                if sample_predictions:
                    images_with_predictions += 1
            image_matches.append(match_image(sample.ground_truth, sample_predictions, threshold, class_aware))
        key = "%.2f" % threshold
        per_threshold_matches[key] = image_matches
        metrics["iou"][key] = aggregate_matches(image_matches)

    if bootstrap_samples and samples:
        for threshold in iou_thresholds:
            key = "%.2f" % threshold
            estimates = {name: [] for name in ("precision", "recall", "f1", "mean_matched_iou")}
            image_matches = per_threshold_matches[key]
            for indices in iter_bootstrap_indices(len(samples), bootstrap_samples, bootstrap_seed):
                replicate = aggregate_matches([image_matches[index] for index in indices])
                for name in estimates:
                    estimates[name].append(float(replicate[name]))
            for name, values in estimates.items():
                metrics["iou"][key]["%s_ci" % name] = confidence_interval(values, confidence_level)

    metrics["num_predictions"] = total_pred
    metrics["num_images_with_predictions"] = images_with_predictions
    metrics["avg_predictions_per_image"] = safe_divide(total_pred, len(samples))
    metrics["coverage"] = safe_divide(images_with_predictions, len(samples))
    metrics["_image_matches"] = per_threshold_matches
    return metrics


def aggregate_dataset_results(
    results: Sequence[dict[str, Any]],
    iou_thresholds: Sequence[float],
    bootstrap_samples: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    combined_matches = {
        "%.2f" % threshold: [
            match
            for result in results
            for match in result["_image_matches"]["%.2f" % threshold]
        ]
        for threshold in iou_thresholds
    }
    aggregate: dict[str, Any] = {
        "dataset": "__micro__",
        "split": results[0]["split"],
        "num_images": sum(int(result["num_images"]) for result in results),
        "num_ground_truth": sum(int(result["num_ground_truth"]) for result in results),
        "num_predictions": sum(int(result["num_predictions"]) for result in results),
        "num_images_with_predictions": sum(
            int(result["num_images_with_predictions"]) for result in results
        ),
        "confidence_threshold": "per_resolved_config",
        "class_aware": bool(results[0]["class_aware"]),
        "predictions_path": "multiple",
        "data_yaml": "multiple",
        "uncertainty": {
            "method": "nonparametric_cluster_bootstrap" if bootstrap_samples else "disabled",
            "resampling_unit": "image",
            "samples": bootstrap_samples,
            "confidence_level": confidence_level,
            "seed": bootstrap_seed,
        },
        "iou": {},
        "_image_matches": combined_matches,
    }
    aggregate["coverage"] = safe_divide(
        aggregate["num_images_with_predictions"], aggregate["num_images"]
    )
    aggregate["avg_predictions_per_image"] = safe_divide(
        aggregate["num_predictions"], aggregate["num_images"]
    )
    for threshold in iou_thresholds:
        key = "%.2f" % threshold
        matches = combined_matches[key]
        metric = aggregate_matches(matches)
        if bootstrap_samples and matches:
            estimates = {name: [] for name in ("precision", "recall", "f1", "mean_matched_iou")}
            for indices in iter_bootstrap_indices(
                aggregate["num_images"], bootstrap_samples, bootstrap_seed
            ):
                replicate = aggregate_matches([matches[index] for index in indices])
                for name in estimates:
                    estimates[name].append(float(replicate[name]))
            for name, values in estimates.items():
                metric["%s_ci" % name] = confidence_interval(values, confidence_level)
        aggregate["iou"][key] = metric
    return aggregate


def public_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if not key.startswith("_")}


def aggregate_matches(matches: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(int(match["tp"]) for match in matches)
    fp = sum(int(match["fp"]) for match in matches)
    fn = sum(int(match["fn"]) for match in matches)
    matched_ious = [float(iou) for match in matches for iou in match["ious"]]
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": safe_divide(2.0 * precision * recall, precision + recall),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "mean_matched_iou": safe_divide(sum(matched_ious), len(matched_ious)),
    }


def make_bootstrap_indices(num_images: int, samples: int, seed: int) -> list[list[int]]:
    return list(iter_bootstrap_indices(num_images, samples, seed))


def iter_bootstrap_indices(num_images: int, samples: int, seed: int) -> Iterable[list[int]]:
    if num_images <= 0 or samples <= 0:
        return
    generator = random.Random(seed)
    for _ in range(samples):
        yield [generator.randrange(num_images) for _ in range(num_images)]


def confidence_interval(values: Sequence[float], confidence_level: float) -> dict[str, float]:
    alpha = 1.0 - confidence_level
    return {
        "low": quantile(values, alpha / 2.0),
        "high": quantile(values, 1.0 - alpha / 2.0),
    }


def quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def validate_uncertainty_args(bootstrap_samples: int, confidence_level: float) -> None:
    if bootstrap_samples < 0:
        raise SystemExit("--bootstrap-samples must be >= 0")
    if not 0.0 < confidence_level < 1.0:
        raise SystemExit("--confidence-level must be between 0 and 1")


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
    seen_images = set()
    for image_ref in read_lines(split_file):
        image_path = resolve_image_path(data_yaml.parent, config, image_ref)
        if image_path is None:
            raise SystemExit("Image not found from %s: %s" % (split_file, image_ref))
        if image_path in seen_images:
            raise ValueError("Duplicate image in evaluation split: %s" % image_path)
        seen_images.add(image_path)
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
    if not samples:
        raise ValueError("Empty evaluation split: %s" % split_file)
    return samples


def load_yolo_obb_labels(label_path: Path, width: int, height: int, class_names: dict[int, str]) -> list[Detection]:
    if not label_path.exists():
        raise FileNotFoundError("Missing annotation file (not an empty negative image): %s" % label_path)
    detections = []
    for line in read_lines(label_path):
        parts = line.split()
        if len(parts) != 9:
            raise ValueError("Expected class ID and eight OBB coordinates: %s" % label_path)
        if float(parts[0]) != int(float(parts[0])):
            raise ValueError("Noninteger class ID: %s" % label_path)
        class_id = int(float(parts[0]))
        coords = [float(value) for value in parts[1:]]
        if class_id not in class_names or any(not math.isfinite(v) or not 0 <= v <= 1 for v in coords):
            raise ValueError("Invalid class or normalized OBB coordinates: %s" % label_path)
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
    records = []
    for image_record in raw:
        image_id = image_record.get("image_id") or image_record.get("file_name") or image_record.get("image_path")
        if not image_id:
            raise ValueError("Missing image ID in predictions: %s" % path)
        image_id = str(image_id)
        detections = []
        for item in image_record.get("detections", []):
            polygon = prediction_polygon(item)
            score = float(item.get("score", item.get("confidence", 1.0)))
            if (not polygon or any(not math.isfinite(v) for v in polygon)
                    or not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError("Invalid prediction geometry/score for %s" % image_id)
            detections.append(
                Detection(
                    label=str(item.get("label") or item.get("class") or item.get("class_name") or "stone"),
                    score=score,
                    polygon=polygon,
                )
            )
        records.append((normalize_key(image_id), detections))
    if len({key for key, _ in records}) != len(records):
        raise ValueError("Duplicate image IDs in predictions: %s" % path)
    predictions = {key: detections for key, detections in records}
    alias_counts: dict[str, int] = {}
    aliases_by_key: dict[str, list[str]] = {}
    for key, _ in records:
        parts = PurePosixPath(key).parts
        aliases = [parts[-1], PurePosixPath(parts[-1]).stem]
        if len(parts) >= 3:
            aliases.append("/".join(parts[-3:]))
        aliases_by_key[key] = aliases
        for alias in set(aliases):
            alias_counts[alias] = alias_counts.get(alias, 0) + 1
    for key, detections in records:
        for alias in aliases_by_key[key]:
            if alias_counts[alias] == 1:
                predictions[alias] = detections
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
    parts = PurePosixPath(sample.image_path.as_posix()).parts
    if len(parts) >= 3:
        keys.append("/".join(parts[-3:]))
    for key in keys:
        normalized = normalize_key(key)
        if normalized in predictions:
            return predictions[normalized]
    raise ValueError("Missing prediction record for %s; include detections: [] for images with no detections" % sample.image_id)


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
        "bootstrap_samples": result["uncertainty"]["samples"],
        "confidence_level": result["uncertainty"]["confidence_level"],
        "bootstrap_seed": result["uncertainty"]["seed"],
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
        for metric_name, column_name in (
            ("precision", "precision"),
            ("recall", "recall"),
            ("f1", "f1"),
            ("mean_matched_iou", "mean_iou"),
        ):
            interval = metric.get("%s_ci" % metric_name)
            if interval:
                row["%s_%s_ci_low" % (column_name, suffix)] = interval["low"]
                row["%s_%s_ci_high" % (column_name, suffix)] = interval["high"]
    return row


def discover_prediction_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("predictions.json")) if path.exists() else []


def validate_prediction_collection(paths: Sequence[Path]) -> None:
    """Only aggregate sibling domain files from one model/run, never model trees."""
    if len(paths) > 1:
        parents = {p.parent.parent.resolve() for p in paths}
        domains = [p.parent.name for p in paths]
        if len(parents) != 1 or len(set(domains)) != len(domains):
            raise ValueError("Evaluate one model/seed at a time: supply its domain directory, not a multi-model tree")


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


def display_path(path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), Path.cwd().resolve())).as_posix()


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
