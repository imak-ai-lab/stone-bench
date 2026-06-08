"""Compute operational size-distribution metrics for StoneBench.

The script reuses the prepared YOLO OBB dataset and prediction JSON format used
by ``evaluate_detection.py``. Fragment size is represented as equivalent-circle
diameter from polygon area. By default sizes are in pixels; pass a global
``--mm-per-px`` or a scale CSV to report millimeters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence


from stones_fragmentation.metrics import detection as evaluate_detection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute StoneBench size-distribution metrics.")
    parser.add_argument("--predictions", default="results/predictions", help="predictions.json file or directory.")
    parser.add_argument("--dataset-root", default="data/prepared/stonebench/yolo_obb")
    parser.add_argument("--data-yaml", default="", help="Explicit dataset data.yaml for single-file evaluation.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", default="results/tables/size_distribution_metrics.csv")
    parser.add_argument("--json-output", default="")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--match-iou-threshold", type=float, default=0.50)
    parser.add_argument("--mm-per-px", type=float, default=1.0)
    parser.add_argument(
        "--pixel-to-mm-csv",
        default="",
        help="Optional CSV with image_id/mm_per_px or dataset/mm_per_px columns.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prediction_files = evaluate_detection.discover_prediction_files(evaluate_detection.resolve_path(Path(args.predictions)))
    if not prediction_files:
        raise SystemExit("No predictions.json files found: %s" % args.predictions)

    scale_lookup = read_scale_lookup(Path(args.pixel_to_mm_csv), default_scale=args.mm_per_px) if args.pixel_to_mm_csv else {}
    rows = []
    details = []
    for prediction_path in prediction_files:
        dataset_slug = evaluate_detection.infer_dataset_slug(prediction_path)
        data_yaml = evaluate_detection.resolve_data_yaml(args, dataset_slug)
        samples = evaluate_detection.load_yolo_obb_dataset(data_yaml, split=args.split)
        predictions = evaluate_detection.load_predictions(prediction_path)
        result = evaluate_size_metrics(
            dataset_slug=dataset_slug,
            samples=samples,
            predictions=predictions,
            scale_lookup=scale_lookup,
            default_scale=args.mm_per_px,
            confidence_threshold=args.confidence_threshold,
            match_iou_threshold=args.match_iou_threshold,
        )
        result.update(
            {
                "dataset": dataset_slug,
                "split": args.split,
                "predictions_path": str(prediction_path),
                "data_yaml": str(data_yaml),
            }
        )
        rows.append(flatten_result(result))
        details.append(result)

    output = evaluate_detection.resolve_path(Path(args.output))
    write_csv(output, rows)
    json_output = evaluate_detection.resolve_path(Path(args.json_output)) if args.json_output else output.with_suffix(".json")
    evaluate_detection.save_json(json_output, details)
    print("wrote: %s" % output)
    print("wrote: %s" % json_output)
    return 0


def evaluate_size_metrics(
    dataset_slug: str,
    samples: Sequence[evaluate_detection.ImageSample],
    predictions: dict[str, list[evaluate_detection.Detection]],
    scale_lookup: dict[str, float],
    default_scale: float,
    confidence_threshold: float,
    match_iou_threshold: float,
) -> dict[str, Any]:
    image_metrics = []
    all_gt = []
    all_pred = []
    small_gt_records = []
    matched_small = 0

    for sample in samples:
        scale = scale_for_sample(dataset_slug, sample, scale_lookup, default_scale)
        gt_sizes = [equivalent_diameter(detection.polygon) * scale for detection in sample.ground_truth]
        pred_items = [
            detection
            for detection in evaluate_detection.lookup_predictions(predictions, sample)
            if detection.score >= confidence_threshold
        ]
        pred_sizes = [equivalent_diameter(detection.polygon) * scale for detection in pred_items]
        all_gt.extend(gt_sizes)
        all_pred.extend(pred_sizes)
        image_metrics.append(
            {
                "image_id": sample.image_id,
                "gt_count": len(gt_sizes),
                "pred_count": len(pred_sizes),
                "cdf_rmse": cdf_rmse(gt_sizes, pred_sizes),
                "ks": ks_distance(gt_sizes, pred_sizes),
                "w1": wasserstein_1(gt_sizes, pred_sizes),
                "d80_gt": percentile(gt_sizes, 0.80),
                "d80_pred": percentile(pred_sizes, 0.80),
            }
        )

    small_threshold = percentile(all_gt, 0.10)
    small_total = 0
    for sample in samples:
        scale = scale_for_sample(dataset_slug, sample, scale_lookup, default_scale)
        pred_items = [
            detection
            for detection in evaluate_detection.lookup_predictions(predictions, sample)
            if detection.score >= confidence_threshold
        ]
        matched_pred = set()
        for gt_index, gt in enumerate(sample.ground_truth):
            gt_size = equivalent_diameter(gt.polygon) * scale
            if small_threshold is None:
                continue
            if gt_size > small_threshold:
                continue
            small_total += 1
            best_index = None
            best_iou = 0.0
            for pred_index, prediction in enumerate(pred_items):
                if pred_index in matched_pred:
                    continue
                iou = evaluate_detection.polygon_iou(gt.polygon, prediction.polygon)
                if iou > best_iou:
                    best_index = pred_index
                    best_iou = iou
            if best_index is not None and best_iou >= match_iou_threshold:
                matched_pred.add(best_index)
                matched_small += 1
            small_gt_records.append({"image_id": sample.image_id, "size": gt_size, "matched": best_index is not None})

    d80_gt = percentile(all_gt, 0.80)
    d80_pred = percentile(all_pred, 0.80)
    return {
        "num_images": len(samples),
        "gt_count": len(all_gt),
        "pred_count": len(all_pred),
        "cdf_rmse": mean([item["cdf_rmse"] for item in image_metrics]),
        "ks": mean([item["ks"] for item in image_metrics]),
        "w1": mean([item["w1"] for item in image_metrics]),
        "d80_gt": d80_gt,
        "d80_pred": d80_pred,
        "d80_abs_error": abs(d80_pred - d80_gt) if d80_gt is not None and d80_pred is not None else None,
        "missed_small": 1.0 - evaluate_detection.safe_divide(matched_small, small_total),
        "small_threshold_p10": small_threshold,
        "confidence_threshold": confidence_threshold,
        "match_iou_threshold": match_iou_threshold,
        "image_metrics": image_metrics,
    }


def equivalent_diameter(polygon: Sequence[float]) -> float:
    area = polygon_area(polygon)
    return math.sqrt(4.0 * area / math.pi) if area > 0 else 0.0


def cdf_rmse(left: Sequence[float], right: Sequence[float]) -> float:
    if not left and not right:
        return 0.0
    grid = sorted(set(float(value) for value in list(left) + list(right)))
    if not grid:
        return 0.0
    errors = [(empirical_cdf(left, value) - empirical_cdf(right, value)) ** 2 for value in grid]
    return math.sqrt(sum(errors) / len(errors))


def ks_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if not left and not right:
        return 0.0
    grid = sorted(set(float(value) for value in list(left) + list(right)))
    if not grid:
        return 0.0
    return max(abs(empirical_cdf(left, value) - empirical_cdf(right, value)) for value in grid)


def wasserstein_1(left: Sequence[float], right: Sequence[float]) -> float:
    if not left and not right:
        return 0.0
    if not left or not right:
        non_empty = list(left or right)
        return max(non_empty) - min(non_empty) if non_empty else 0.0
    grid = sorted(set(float(value) for value in list(left) + list(right)))
    if len(grid) < 2:
        return 0.0
    distance = 0.0
    for start, end in zip(grid[:-1], grid[1:]):
        distance += abs(empirical_cdf(left, start) - empirical_cdf(right, start)) * (end - start)
    return distance


def empirical_cdf(values: Sequence[float], threshold: float) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value <= threshold) / float(len(values))


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower = int(math.floor(position))
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def polygon_area(polygon: Sequence[float]) -> float:
    if len(polygon) < 6:
        return 0.0
    area = 0.0
    count = len(polygon) // 2
    for index in range(count):
        x1 = float(polygon[2 * index])
        y1 = float(polygon[2 * index + 1])
        x2 = float(polygon[2 * ((index + 1) % count)])
        y2 = float(polygon[2 * ((index + 1) % count) + 1])
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def scale_for_sample(
    dataset_slug: str,
    sample: evaluate_detection.ImageSample,
    scale_lookup: dict[str, float],
    default_scale: float,
) -> float:
    keys = [
        sample.image_id,
        str(sample.image_path),
        sample.image_path.as_posix(),
        sample.image_path.name,
        sample.image_path.stem,
        dataset_slug,
    ]
    for key in keys:
        normalized = evaluate_detection.normalize_key(key)
        if normalized in scale_lookup:
            return scale_lookup[normalized]
    return default_scale


def read_scale_lookup(path: Path, default_scale: float) -> dict[str, float]:
    if not path.exists():
        raise SystemExit("Scale CSV not found: %s" % path)
    lookup = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            key = row.get("image_id") or row.get("image") or row.get("file_name") or row.get("dataset")
            value = row.get("mm_per_px") or row.get("scale") or row.get("pixel_to_mm")
            if key and value:
                lookup[evaluate_detection.normalize_key(str(key))] = float(value)
    if not lookup and default_scale <= 0:
        raise SystemExit("Scale CSV has no usable rows: %s" % path)
    return lookup


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": result["dataset"],
        "split": result["split"],
        "num_images": result["num_images"],
        "gt_count": result["gt_count"],
        "pred_count": result["pred_count"],
        "cdf_rmse": result["cdf_rmse"],
        "ks": result["ks"],
        "w1": result["w1"],
        "d80_gt": result["d80_gt"],
        "d80_pred": result["d80_pred"],
        "d80_abs_error": result["d80_abs_error"],
        "missed_small": result["missed_small"],
        "small_threshold_p10": result["small_threshold_p10"],
        "confidence_threshold": result["confidence_threshold"],
        "match_iou_threshold": result["match_iou_threshold"],
        "predictions_path": result["predictions_path"],
        "data_yaml": result["data_yaml"],
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Sequence[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


if __name__ == "__main__":
    raise SystemExit(main())
