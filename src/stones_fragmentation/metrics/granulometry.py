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

import numpy as np

from stones_fragmentation.metrics import detection as evaluate_detection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute StoneBench size-distribution metrics.")
    parser.add_argument("--config", default="configs/evaluation.yaml", help="Versioned evaluation YAML.")
    parser.add_argument("--predictions", default=None, help="predictions.json file or directory.")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--data-yaml", default="", help="Explicit dataset data.yaml for single-file evaluation.")
    parser.add_argument("--split", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--json-output", default="")
    parser.add_argument(
        "--confidence-threshold",
        default=None,
        help="Numeric post-filter or 'from_resolved_config' for each prediction file.",
    )
    parser.add_argument("--match-iou-threshold", type=float, default=None)
    parser.add_argument("--mm-per-px", type=float, default=None)
    parser.add_argument(
        "--pixel-to-mm-csv",
        default="",
        help="Optional CSV with image_id/mm_per_px or dataset/mm_per_px columns.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=None,
        help="Image-level bootstrap replicates; use 0 to disable confidence intervals.",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=None,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    apply_granulometry_config(args)
    evaluate_detection.validate_uncertainty_args(args.bootstrap_samples, args.confidence_level)
    prediction_files = evaluate_detection.discover_prediction_files(evaluate_detection.resolve_path(Path(args.predictions)))
    if not prediction_files:
        raise SystemExit("No predictions.json files found: %s" % args.predictions)
    evaluate_detection.validate_prediction_collection(prediction_files)

    scale_lookup = read_scale_lookup(Path(args.pixel_to_mm_csv), default_scale=args.mm_per_px) if args.pixel_to_mm_csv else {}
    rows = []
    details = []
    for prediction_path in prediction_files:
        dataset_slug = evaluate_detection.infer_dataset_slug(prediction_path)
        data_yaml = evaluate_detection.resolve_data_yaml(args, dataset_slug)
        samples = evaluate_detection.load_yolo_obb_dataset(data_yaml, split=args.split)
        predictions = evaluate_detection.load_predictions(prediction_path)
        confidence_threshold = evaluate_detection.resolve_confidence_threshold(
            args.confidence_threshold, prediction_path
        )
        result = evaluate_size_metrics(
            dataset_slug=dataset_slug,
            samples=samples,
            predictions=predictions,
            scale_lookup=scale_lookup,
            default_scale=args.mm_per_px,
            confidence_threshold=confidence_threshold,
            match_iou_threshold=args.match_iou_threshold,
            bootstrap_samples=args.bootstrap_samples,
            confidence_level=args.confidence_level,
            bootstrap_seed=args.bootstrap_seed,
        )
        result.update(
            {
                "dataset": dataset_slug,
                "split": args.split,
                "predictions_path": evaluate_detection.display_path(prediction_path),
                "data_yaml": evaluate_detection.display_path(data_yaml),
            }
        )
        rows.append(flatten_result(result))
        details.append(result)

    if len(details) > 1:
        aggregate = aggregate_dataset_size_results(
            details,
            args.bootstrap_samples,
            args.confidence_level,
            args.bootstrap_seed,
        )
        rows.append(flatten_result(aggregate))
        details.append(aggregate)

    output = evaluate_detection.resolve_path(Path(args.output))
    write_csv(output, rows)
    json_output = evaluate_detection.resolve_path(Path(args.json_output)) if args.json_output else output.with_suffix(".json")
    evaluate_detection.save_json(json_output, [public_size_result(result) for result in details])
    print("wrote: %s" % output)
    print("wrote: %s" % json_output)
    return 0


def public_size_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "image_metrics"}


def aggregate_dataset_size_results(
    results: Sequence[dict[str, Any]],
    bootstrap_samples: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    image_metrics = [item for result in results for item in result["image_metrics"]]
    aggregate = aggregate_size_records(image_metrics)
    aggregate.update(
        {
            "dataset": "__micro__",
            "split": results[0]["split"],
            "num_images": len(image_metrics),
            "confidence_threshold": "per_resolved_config",
            "match_iou_threshold": results[0]["match_iou_threshold"],
            "predictions_path": "multiple",
            "data_yaml": "multiple",
            "image_metrics": image_metrics,
            "uncertainty": {
                "method": "nonparametric_cluster_bootstrap" if bootstrap_samples else "disabled",
                "resampling_unit": "image",
                "samples": bootstrap_samples,
                "confidence_level": confidence_level,
                "seed": bootstrap_seed,
            },
        }
    )
    if bootstrap_samples and image_metrics:
        aggregate.update(
            bootstrap_size_intervals(
                image_metrics, bootstrap_samples, confidence_level, bootstrap_seed
            )
        )
    return aggregate


def bootstrap_size_intervals(
    image_metrics: Sequence[dict[str, Any]],
    bootstrap_samples: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> dict[str, dict[str, float]]:
    """Vectorized image-cluster bootstrap for large pooled fragment arrays."""
    image_count = len(image_metrics)
    per_image = {
        name: np.asarray([float(item[name]) for item in image_metrics], dtype=np.float64)
        for name in ("cdf_rmse", "ks", "w1")
    }
    gt_values, gt_images, gt_matched = size_observation_arrays(image_metrics, "gt")
    pred_values, pred_images, _ = size_observation_arrays(image_metrics, "pred")
    estimates = {
        name: []
        for name in ("cdf_rmse", "ks", "w1", "d10_abs_error", "d80_abs_error", "missed_small")
    }
    for indices in evaluate_detection.iter_bootstrap_indices(
        image_count, bootstrap_samples, bootstrap_seed
    ):
        image_weights = np.bincount(indices, minlength=image_count).astype(np.int64)
        for name, values in per_image.items():
            estimates[name].append(float(np.dot(image_weights, values) / image_count))
        d80_gt = weighted_percentile(gt_values, gt_images, image_weights, 0.80)
        d80_pred = weighted_percentile(pred_values, pred_images, image_weights, 0.80)
        if d80_gt is not None and d80_pred is not None:
            estimates["d80_abs_error"].append(abs(d80_pred - d80_gt))
        small_threshold = weighted_percentile(gt_values, gt_images, image_weights, 0.10)
        d10_pred = weighted_percentile(pred_values, pred_images, image_weights, 0.10)
        if small_threshold is not None and d10_pred is not None:
            estimates["d10_abs_error"].append(abs(d10_pred - small_threshold))
        if small_threshold is not None:
            mask = gt_values <= small_threshold
            weights = image_weights[gt_images[mask]]
            total = int(weights.sum())
            matched = int(weights[gt_matched[mask]].sum()) if total else 0
            estimates["missed_small"].append(
                1.0 - evaluate_detection.safe_divide(matched, total)
            )
    return {
        "%s_ci" % name: evaluate_detection.confidence_interval(values, confidence_level)
        for name, values in estimates.items()
    }


def size_observation_arrays(
    image_metrics: Sequence[dict[str, Any]], kind: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if kind == "gt":
        values = [
            float(record["size"])
            for item in image_metrics
            for record in item["ground_truth_matches"]
        ]
        matched = [
            bool(record["matched"])
            for item in image_metrics
            for record in item["ground_truth_matches"]
        ]
        image_ids = [
            image_index
            for image_index, item in enumerate(image_metrics)
            for _ in item["ground_truth_matches"]
        ]
    else:
        values = [float(value) for item in image_metrics for value in item["pred_sizes"]]
        matched = [False] * len(values)
        image_ids = [
            image_index
            for image_index, item in enumerate(image_metrics)
            for _ in item["pred_sizes"]
        ]
    if not values:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=bool),
        )
    value_array = np.asarray(values, dtype=np.float64)
    order = np.argsort(value_array, kind="stable")
    return (
        value_array[order],
        np.asarray(image_ids, dtype=np.int64)[order],
        np.asarray(matched, dtype=bool)[order],
    )


def weighted_percentile(
    sorted_values: np.ndarray,
    sorted_image_ids: np.ndarray,
    image_weights: np.ndarray,
    fraction: float,
) -> float | None:
    if sorted_values.size == 0:
        return None
    weights = image_weights[sorted_image_ids]
    total = int(weights.sum())
    if total <= 0:
        return None
    position = (total - 1) * fraction
    lower_rank = int(math.floor(position))
    upper_rank = min(lower_rank + 1, total - 1)
    cumulative = np.cumsum(weights)
    lower_index = int(np.searchsorted(cumulative, lower_rank, side="right"))
    upper_index = int(np.searchsorted(cumulative, upper_rank, side="right"))
    interpolation = position - lower_rank
    return float(
        sorted_values[lower_index] * (1.0 - interpolation)
        + sorted_values[upper_index] * interpolation
    )


def apply_granulometry_config(args: argparse.Namespace) -> None:
    config_path = evaluate_detection.resolve_path(Path(args.config))
    config = evaluate_detection.load_yaml(config_path) if config_path.exists() else {}
    dataset = config.get("dataset", {})
    detection = config.get("detection", {})
    granulometry = config.get("granulometry", {})
    uncertainty = config.get("uncertainty", {})
    defaults = {
        "predictions": dataset.get("predictions", "results/reference_predictions"),
        "dataset_root": dataset.get("root", "data/prepared/stonebench/yolo_obb"),
        "split": dataset.get("split", "val"),
        "output": granulometry.get("output", "results/tables/size_distribution_metrics.csv"),
        "confidence_threshold": granulometry.get(
            "confidence_threshold", detection.get("confidence_threshold", "from_resolved_config")
        ),
        "match_iou_threshold": granulometry.get("match_iou_threshold", 0.50),
        "mm_per_px": granulometry.get("default_mm_per_px", 1.0),
        "bootstrap_samples": uncertainty.get(
            "samples", evaluate_detection.DEFAULT_BOOTSTRAP_SAMPLES
        ),
        "confidence_level": uncertainty.get(
            "confidence_level", evaluate_detection.DEFAULT_CONFIDENCE_LEVEL
        ),
        "bootstrap_seed": uncertainty.get("seed", evaluate_detection.DEFAULT_BOOTSTRAP_SEED),
    }
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def evaluate_size_metrics(
    dataset_slug: str,
    samples: Sequence[evaluate_detection.ImageSample],
    predictions: dict[str, list[evaluate_detection.Detection]],
    scale_lookup: dict[str, float],
    default_scale: float,
    confidence_threshold: float,
    match_iou_threshold: float,
    bootstrap_samples: int = evaluate_detection.DEFAULT_BOOTSTRAP_SAMPLES,
    confidence_level: float = evaluate_detection.DEFAULT_CONFIDENCE_LEVEL,
    bootstrap_seed: int = evaluate_detection.DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    evaluate_detection.validate_uncertainty_args(bootstrap_samples, confidence_level)
    image_metrics = []

    for sample in samples:
        scale = scale_for_sample(dataset_slug, sample, scale_lookup, default_scale)
        gt_sizes = [equivalent_diameter(detection.polygon) * scale for detection in sample.ground_truth]
        pred_items = [
            detection
            for detection in evaluate_detection.lookup_predictions(predictions, sample)
            if detection.score >= confidence_threshold
        ]
        pred_sizes = [equivalent_diameter(detection.polygon) * scale for detection in pred_items]
        matched_gt = matched_ground_truth_indices(
            sample.ground_truth,
            pred_items,
            match_iou_threshold,
        )
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
                "gt_sizes": gt_sizes,
                "pred_sizes": pred_sizes,
                "ground_truth_matches": [
                    {"size": size, "matched": index in matched_gt}
                    for index, size in enumerate(gt_sizes)
                ],
            }
        )

    result = aggregate_size_records(image_metrics)
    result.update({
        "num_images": len(samples),
        "confidence_threshold": confidence_threshold,
        "match_iou_threshold": match_iou_threshold,
        "image_metrics": image_metrics,
        "uncertainty": {
            "method": "nonparametric_cluster_bootstrap" if bootstrap_samples else "disabled",
            "resampling_unit": "image",
            "samples": bootstrap_samples,
            "confidence_level": confidence_level,
            "seed": bootstrap_seed,
        },
    })

    if bootstrap_samples and image_metrics:
        result.update(
            bootstrap_size_intervals(
                image_metrics, bootstrap_samples, confidence_level, bootstrap_seed
            )
        )
    return result


def matched_ground_truth_indices(
    ground_truth: Sequence[evaluate_detection.Detection],
    predictions: Sequence[evaluate_detection.Detection],
    iou_threshold: float,
) -> set[int]:
    """Return GT indices matched by the same score-ordered policy as detection evaluation."""
    matched_gt: set[int] = set()
    for prediction in sorted(predictions, key=lambda item: item.score, reverse=True):
        best_index = None
        best_iou = 0.0
        for gt_index, gt in enumerate(ground_truth):
            if gt_index in matched_gt:
                continue
            iou = evaluate_detection.polygon_iou(gt.polygon, prediction.polygon)
            if iou > best_iou:
                best_index = gt_index
                best_iou = iou
        if best_index is not None and best_iou >= iou_threshold:
            matched_gt.add(best_index)
    return matched_gt


def aggregate_size_records(image_metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    all_gt = [float(value) for item in image_metrics for value in item["gt_sizes"]]
    all_pred = [float(value) for item in image_metrics for value in item["pred_sizes"]]
    small_threshold = percentile(all_gt, 0.10)
    small_records = [
        record
        for item in image_metrics
        for record in item["ground_truth_matches"]
        if small_threshold is not None and float(record["size"]) <= small_threshold
    ]
    matched_small = sum(1 for record in small_records if record["matched"])
    d80_gt = percentile(all_gt, 0.80)
    d80_pred = percentile(all_pred, 0.80)
    d10_pred = percentile(all_pred, 0.10)
    return {
        "gt_count": len(all_gt),
        "pred_count": len(all_pred),
        "cdf_rmse": mean([item["cdf_rmse"] for item in image_metrics]),
        "ks": mean([item["ks"] for item in image_metrics]),
        "w1": mean([item["w1"] for item in image_metrics]),
        "d80_gt": d80_gt,
        "d10_gt": small_threshold,
        "d10_pred": d10_pred,
        "d10_abs_error": abs(d10_pred - small_threshold) if d10_pred is not None and small_threshold is not None else None,
        "d80_pred": d80_pred,
        "d80_abs_error": abs(d80_pred - d80_gt) if d80_gt is not None and d80_pred is not None else None,
        "missed_small": 1.0 - evaluate_detection.safe_divide(matched_small, len(small_records)),
        "small_threshold_p10": small_threshold,
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
    row = {
        "dataset": result["dataset"],
        "split": result["split"],
        "num_images": result["num_images"],
        "gt_count": result["gt_count"],
        "pred_count": result["pred_count"],
        "cdf_rmse": result["cdf_rmse"],
        "ks": result["ks"],
        "w1": result["w1"],
        "d80_gt": result["d80_gt"],
        "d10_gt": result["d10_gt"],
        "d10_pred": result["d10_pred"],
        "d10_abs_error": result["d10_abs_error"],
        "d80_pred": result["d80_pred"],
        "d80_abs_error": result["d80_abs_error"],
        "missed_small": result["missed_small"],
        "small_threshold_p10": result["small_threshold_p10"],
        "confidence_threshold": result["confidence_threshold"],
        "match_iou_threshold": result["match_iou_threshold"],
        "bootstrap_samples": result["uncertainty"]["samples"],
        "confidence_level": result["uncertainty"]["confidence_level"],
        "bootstrap_seed": result["uncertainty"]["seed"],
        "predictions_path": result["predictions_path"],
        "data_yaml": result["data_yaml"],
    }
    for metric_name in ("cdf_rmse", "ks", "w1", "d10_abs_error", "d80_abs_error", "missed_small"):
        interval = result.get("%s_ci" % metric_name)
        if interval:
            row["%s_ci_low" % metric_name] = interval["low"]
            row["%s_ci_high" % metric_name] = interval["high"]
    return row


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
