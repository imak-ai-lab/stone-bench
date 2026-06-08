"""Run or plan one local zero-shot foundation-model pass.

This script is the stable entrypoint for one foundation baseline config. It
already handles config resolution, dataset discovery, run folders, and dry-run
manifests. Foundation backends have different local dependency requirements; if
a config has no bundled execution path, non-dry execution stops with a clear
message instead of producing synthetic predictions.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = "data/prepared/stonebench/yolo_obb"
IMPLEMENTATION_NOTES = {
    "sam-obb": "requires a local SAM/SAM2 backend setup",
    "owlv2-obb": "requires a local OWLv2 backend setup",
    "yolo-world-obb": "available when Ultralytics YOLO-World is installed",
    "yoloe-obb": "requires a local YOLOE backend setup",
    "grounding-dino-obb": "requires a local GroundingDINO backend setup",
    "grounding-dino-sam-obb": "requires local GroundingDINO and SAM/SAM2 backend setup",
    "qwen25-vl-obb": "requires a local Qwen2.5-VL backend setup",
    "detic-obb": "requires a local Detic backend setup",
    "tiled-owlv2-sam-obb": "requires local OWLv2 and SAM/SAM2 backend setup",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or plan one local zero-shot foundation baseline.")
    parser.add_argument("--config", required=True, help="Foundation model YAML under configs/zeroshot.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", default="", help="Override config output.root.")
    parser.add_argument("--run-name", default="", help="Fixed run folder name. Defaults to timestamp.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset slugs to run.")
    parser.add_argument("--device", default=None, help="Override model.device.")
    parser.add_argument("--confidence-threshold", type=float, default=None, help="Override model threshold.")
    parser.add_argument("--limit", type=int, default=None, help="Optional image limit per dataset.")
    parser.add_argument("--split", default="val", help="Dataset split to run.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write run_plan.json without loading model backends.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_path(Path(args.config))
    config = load_yaml(config_path)
    model = dict(config.get("model", {}))
    output_root = resolve_output_root(config, args.output_root)
    run_root = output_root / (args.run_name or default_run_name(config_path, model))
    datasets = discover_datasets(resolve_path(Path(args.dataset_root)), args.datasets)

    if args.device:
        model["device"] = args.device
    if args.confidence_threshold is not None:
        model["confidence_threshold"] = float(args.confidence_threshold)

    plan = {
        "status": "planned",
        "config": str(config_path),
        "run_root": str(run_root),
        "model": model,
        "output": dict(config.get("output", {})),
        "datasets": datasets,
        "limit": args.limit,
        "split": args.split,
        "skip_existing": args.skip_existing,
        "adapter_status": adapter_status(model),
    }
    run_root.mkdir(parents=True, exist_ok=True)
    save_json(run_root / "run_plan.json", plan)

    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    status = plan["adapter_status"]
    if not status["implemented"]:
        raise SystemExit(
            "No bundled local execution path for model.name=%r. %s. "
            "Use --dry-run to write the reproducible run plan."
            % (model.get("name"), status["next_action"])
        )

    if model.get("name") == "yolo-world-obb":
        summary = run_yolo_world_obb(model, datasets, run_root, args)
        save_json(run_root / "run_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    raise SystemExit("Internal error: adapter marked implemented but no inference dispatch is registered.")


def discover_datasets(dataset_root: Path, selected: list[str] | None) -> list[dict[str, Any]]:
    selected_set = set(selected or [])
    candidates = []
    search_roots = [dataset_root]
    if (dataset_root / "yolo_obb").exists():
        search_roots.append(dataset_root / "yolo_obb")

    for root in search_roots:
        if (root / "data.yaml").exists():
            candidates.append(dataset_entry(root))
        for data_yaml in sorted(root.glob("*/data.yaml")):
            candidates.append(dataset_entry(data_yaml.parent))
        for dataset_config in sorted(root.glob("*/dataset_config.yaml")):
            candidates.append(dataset_entry(dataset_config.parent, config_name="dataset_config.yaml"))

    unique = {}
    for item in candidates:
        unique[item["slug"]] = item
    datasets = [unique[key] for key in sorted(unique)]
    if selected_set:
        datasets = [item for item in datasets if item["slug"] in selected_set]
        missing = selected_set - {item["slug"] for item in datasets}
        if missing:
            raise SystemExit("Requested datasets were not found: %s" % ", ".join(sorted(missing)))
    if not datasets:
        raise SystemExit("No dataset configs found under %s" % dataset_root)
    return datasets


def dataset_entry(path: Path, config_name: str = "data.yaml") -> dict[str, Any]:
    return {
        "slug": path.name,
        "root": str(path),
        "config": str(path / config_name),
    }


def adapter_status(model: dict[str, Any]) -> dict[str, Any]:
    name = str(model.get("name", ""))
    return {
        "implemented": name == "yolo-world-obb",
        "next_action": IMPLEMENTATION_NOTES.get(name, "add a local adapter for this model name"),
    }


def run_yolo_world_obb(
    model_config: dict[str, Any],
    datasets: list[dict[str, Any]],
    run_root: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is required: python -m pip install ultralytics") from exc

    yolo = YOLO(str(model_config["model_id"]))
    labels = [str(label) for label in model_config.get("labels", ["stone"])]
    if hasattr(yolo, "set_classes"):
        yolo.set_classes(labels)

    summary = []
    for dataset in datasets:
        dataset_slug = dataset["slug"]
        dataset_root = Path(dataset["root"])
        dataset_run_root = run_root / dataset_slug
        predictions_path = dataset_run_root / "predictions.json"
        if args.skip_existing and predictions_path.exists():
            summary.append({"dataset": dataset_slug, "status": "skipped", "predictions": str(predictions_path)})
            continue

        image_paths = load_split_images(dataset_root / "data.yaml", args.split)
        if args.limit is not None:
            image_paths = image_paths[: args.limit]
        dataset_run_root.mkdir(parents=True, exist_ok=True)

        started = time.time()
        payload = []
        for image_path in image_paths:
            results = yolo.predict(
                source=str(image_path),
                imgsz=int(model_config.get("image_size", 640)),
                conf=float(model_config.get("confidence_threshold", 0.05)),
                iou=float(model_config.get("iou_threshold", 0.70)),
                max_det=int(model_config.get("max_detections", 1000)),
                device=str(model_config.get("device", "cuda")),
                verbose=False,
            )
            detections = parse_yolo_world_result(results[0] if results else None, model_config)
            payload.append({"image_id": str(image_path.resolve()), "detections": detections})

        save_json(predictions_path, payload)
        dataset_summary = {
            "dataset": dataset_slug,
            "status": "ok",
            "images": len(image_paths),
            "predictions": str(predictions_path),
            "detections": sum(len(item["detections"]) for item in payload),
            "duration_seconds": round(time.time() - started, 3),
        }
        save_json(dataset_run_root / "run_info.json", {**dataset_summary, "model": model_config, "split": args.split})
        summary.append(dataset_summary)
    return summary


def load_split_images(data_yaml: Path, split: str) -> list[Path]:
    config = load_yaml(data_yaml)
    split_file = resolve_dataset_path(data_yaml.parent, config, Path(str(config.get(split, "%s.txt" % split))))
    if not split_file.exists():
        return []
    paths = []
    for image_ref in read_lines(split_file):
        image_path = resolve_image_path(data_yaml.parent, config, image_ref)
        if image_path is not None:
            paths.append(image_path)
    return paths


def parse_yolo_world_result(result: Any, model_config: dict[str, Any]) -> list[dict[str, Any]]:
    if result is None or getattr(result, "boxes", None) is None:
        return []
    names = getattr(result, "names", {}) or {}
    output_label = str(model_config.get("output_label") or "stone")
    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().tolist() if boxes.xyxy is not None else []
    scores = boxes.conf.cpu().tolist() if boxes.conf is not None else []
    classes = boxes.cls.cpu().tolist() if boxes.cls is not None else []
    detections = []
    for box, score, class_id in zip(xyxy, scores, classes):
        x_min, y_min, x_max, y_max = [float(value) for value in box[:4]]
        label = str(names.get(int(class_id), output_label)) if isinstance(names, dict) else output_label
        detections.append(
            {
                "label": label or output_label,
                "score": float(score),
                "bbox_xywh": [x_min, y_min, x_max - x_min, y_max - y_min],
                "obb_xyxyxyxy": [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max],
                "metadata": {"class_id": int(class_id), "adapter": "yolo-world-obb"},
            }
        )
    return detections


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
    for candidate in (dataset_root / base / path, dataset_root / path, ROOT / path):
        if candidate.exists():
            return candidate.resolve()
    return None


def resolve_output_root(config: dict[str, Any], override: str) -> Path:
    raw_output = override or config.get("output", {}).get("root") or "results/predictions/foundation"
    return resolve_path(Path(str(raw_output)))


def default_run_name(config_path: Path, model: dict[str, Any]) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_slug = slugify(str(model.get("model_id") or model.get("name") or config_path.stem))
    return "%s-%s" % (timestamp, model_slug)


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj) or {}


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as file_obj:
        return [line.strip() for line in file_obj if line.strip()]


def slugify(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "-" for char in value)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "model"


if __name__ == "__main__":
    raise SystemExit(main())
