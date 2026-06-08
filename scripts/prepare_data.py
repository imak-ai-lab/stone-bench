"""Prepare local StoneBench train/evaluation datasets in several formats.

The script accepts COCO and YOLO-style source exports and writes derived
training/evaluation layouts:

- ``yolo_obb``: YOLO OBB labels, ``class x1 y1 ... x4 y4``.
- ``yolo_detect``: YOLO detection labels, ``class x_center y_center width height``.
- ``yolo_seg``: YOLO segmentation polygon labels.
- ``coco_detect``: COCO annotations with AABB boxes.
- ``coco_seg``: COCO annotations with polygon segmentations.

It is deliberately local and explicit. No cloud jobs, no hidden dataset fetches.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {
    "default_1": "val",
    "valid": "val",
    "validation": "val",
    "val": "val",
    "test": "test",
    "train": "train",
}
YOLO_FORMATS = {"yolo_obb", "yolo_detect", "yolo_seg"}
COCO_FORMATS = {"coco_detect", "coco_seg"}
SUPPORTED_FORMATS = YOLO_FORMATS | COCO_FORMATS


@dataclass
class Annotation:
    class_id: int
    polygon: list[float]
    source: str


@dataclass
class ImageRecord:
    dataset: str
    source_split: str
    target_split: str
    image_path: Path
    width: int
    height: int
    annotations: list[Annotation] = field(default_factory=list)
    synthetic: bool = False
    source_key: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare StoneBench datasets for local training/evaluation.")
    parser.add_argument("--config", default="configs/prepare_data.yaml")
    parser.add_argument("--output-root", help="Override output.root from config.")
    parser.add_argument("--copy-mode", choices=("hardlink", "symlink", "copy", "none"), help="Override output.copy_mode.")
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=sorted(SUPPORTED_FORMATS),
        help="Override output.formats from config.",
    )
    parser.add_argument("--split-mode", choices=("preserve", "resplit"), help="Override split.mode from config.")
    parser.add_argument("--val-fraction", type=float, help="Override split.val_fraction.")
    parser.add_argument("--seed", type=int, help="Override split.seed.")
    parser.add_argument("--datasets", nargs="*", help="Optional source dataset names to prepare.")
    parser.add_argument("--limit", type=int, help="Optional smoke-test image limit per source split.")
    parser.add_argument("--dry-run", action="store_true", help="Print the manifest without writing output files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    config = load_yaml(config_path)
    root = config_path.parent

    output_config = dict(config.get("output", {}))
    split_config = dict(config.get("split", {}))
    dataset_config = dict(config.get("dataset", {}))

    output_root = Path(args.output_root or output_config.get("root", "data/prepared/stonebench"))
    output_root = resolve_path(root, output_root)
    copy_mode = args.copy_mode or str(output_config.get("copy_mode", "hardlink"))
    formats = normalize_formats(args.formats or output_config.get("formats", ["yolo_obb", "yolo_detect"]))
    split_mode = args.split_mode or str(split_config.get("mode", output_config.get("split_mode", "preserve")))
    val_fraction = float(args.val_fraction if args.val_fraction is not None else split_config.get("val_fraction", 0.20))
    seed = int(args.seed if args.seed is not None else split_config.get("seed", 42))
    combined_slug = str(split_config.get("combined_slug", "all_real_plus_synthetic_train"))
    synthetic_names = set(str(name) for name in split_config.get("synthetic_datasets", ["synthetic"]))
    selected = set(args.datasets or [])
    class_names = normalize_class_names(dataset_config.get("names", ["stone"]))

    source_configs = dataset_config.get("sources", [])
    if not source_configs:
        raise SystemExit("No dataset.sources found in %s" % config_path)

    records_by_dataset, source_report = load_sources(
        root=root,
        source_configs=source_configs,
        selected=selected,
        limit=args.limit,
        synthetic_names=synthetic_names,
    )
    split_sets = build_split_sets(
        records_by_dataset=records_by_dataset,
        split_mode=split_mode,
        val_fraction=val_fraction,
        seed=seed,
        combined_slug=combined_slug,
        synthetic_names=synthetic_names,
    )

    manifest: dict[str, Any] = {
        "config": str(config_path),
        "output_root": str(output_root),
        "copy_mode": copy_mode,
        "formats": formats,
        "split_mode": split_mode,
        "val_fraction": val_fraction,
        "seed": seed,
        "class_names": class_names,
        "sources": source_report,
        "datasets": {},
    }

    for format_name in formats:
        manifest["datasets"][format_name] = {}
        for dataset_name, split_records in sorted(split_sets.items()):
            format_root = output_root / format_name / dataset_name
            summary = summarize_split_records(split_records)
            if not args.dry_run:
                if format_name in YOLO_FORMATS:
                    data_file = write_yolo_dataset(format_root, split_records, class_names, format_name, copy_mode)
                else:
                    data_file = write_coco_dataset(format_root, split_records, class_names, format_name, copy_mode)
                summary["data_file"] = str(data_file)
                save_json(format_root / "dataset_summary.json", summary)
            manifest["datasets"][format_name][dataset_name] = summary

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    else:
        save_json(output_root / "prepare_manifest.json", manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def load_sources(
    root: Path,
    source_configs: Sequence[dict[str, Any]],
    selected: set[str],
    limit: int | None,
    synthetic_names: set[str],
) -> tuple[dict[str, list[ImageRecord]], dict[str, Any]]:
    records_by_dataset: dict[str, list[ImageRecord]] = {}
    report: dict[str, Any] = {}

    for source_config in source_configs:
        name = str(source_config["name"])
        if selected and name not in selected:
            continue
        source_type = str(source_config.get("type", "yolo_segmentation")).lower().replace("-", "_")
        source_path = resolve_path(root, Path(source_config["path"]))
        counters: Counter[str] = Counter()
        synthetic = bool(source_config.get("synthetic", name in synthetic_names))

        if source_type.startswith("coco"):
            records = read_coco_source(source_config, source_path, counters, limit, synthetic)
        elif source_type.startswith("yolo"):
            records = read_yolo_source(source_config, source_path, counters, limit, synthetic)
        else:
            raise SystemExit("Unsupported source type %r for %s" % (source_type, name))

        records_by_dataset[name] = records
        report[name] = {
            "type": source_type,
            "path": str(source_path),
            "images": len(records),
            "objects": sum(len(record.annotations) for record in records),
            "synthetic": synthetic,
            "counters": dict(counters),
        }

    if not records_by_dataset:
        raise SystemExit("No source datasets selected or found.")
    return records_by_dataset, report


def read_yolo_source(
    source_config: dict[str, Any],
    source_root: Path,
    counters: Counter[str],
    limit: int | None,
    synthetic: bool,
) -> list[ImageRecord]:
    name = str(source_config["name"])
    data_yaml = load_yaml(source_root / str(source_config.get("data_yaml", "data.yaml")))
    split_map = source_config.get("split_map") or discover_yolo_split_map(data_yaml)
    label_format = str(source_config.get("label_format", source_config.get("type", "yolo_segmentation"))).lower()
    class_id_map = parse_class_id_map(source_config.get("class_id_map"))
    records: list[ImageRecord] = []

    for source_split, target_split in split_map.items():
        image_refs = list_yolo_split_images(source_root, data_yaml, str(source_split))
        if limit:
            image_refs = image_refs[: int(limit)]
        for image_ref in image_refs:
            image_path = resolve_yolo_image_path(source_root, image_ref)
            if image_path is None:
                counters["missing_image"] += 1
                continue
            size = read_image_size(image_path)
            if size is None:
                counters["unreadable_image"] += 1
                continue
            label_path = resolve_yolo_label_path(image_path)
            annotations: list[Annotation] = []
            if label_path is None:
                counters["missing_label"] += 1
                if source_config.get("skip_missing_labels", True):
                    continue
            else:
                annotations = parse_yolo_label_file(label_path, size[0], size[1], label_format, counters, class_id_map)

            if not annotations and source_config.get("skip_empty", True):
                counters["empty_image_skipped"] += 1
                continue
            records.append(
                ImageRecord(
                    dataset=name,
                    source_split=str(source_split),
                    target_split=normalize_split(str(target_split)),
                    image_path=image_path,
                    width=size[0],
                    height=size[1],
                    annotations=annotations,
                    synthetic=synthetic,
                    source_key=source_image_key(image_path),
                )
            )
            counters["images_loaded"] += 1
            counters["objects_loaded"] += len(annotations)
    return records


def read_coco_source(
    source_config: dict[str, Any],
    source_root: Path,
    counters: Counter[str],
    limit: int | None,
    synthetic: bool,
) -> list[ImageRecord]:
    name = str(source_config["name"])
    split_map = source_config.get("split_map") or discover_coco_split_map(source_root)
    records: list[ImageRecord] = []

    for source_split, target_split in split_map.items():
        annotation_path = coco_annotation_path(source_root, str(source_split), source_config)
        if not annotation_path.exists():
            counters["missing_coco_annotations:%s" % source_split] += 1
            continue
        coco = load_json(annotation_path)
        class_id_map = parse_class_id_map(source_config.get("class_id_map")) or infer_coco_class_id_map(coco)
        annotations_by_image = group_by(coco.get("annotations", []), "image_id")
        images = coco.get("images", [])
        if limit:
            images = images[: int(limit)]

        for image in images:
            image_path = resolve_coco_image_path(source_root, annotation_path.parent, image.get("file_name", ""))
            if image_path is None:
                counters["missing_image"] += 1
                continue
            width = int(image.get("width") or 0)
            height = int(image.get("height") or 0)
            if width <= 0 or height <= 0:
                size = read_image_size(image_path)
                if size is None:
                    counters["unreadable_image"] += 1
                    continue
                width, height = size

            annotations = [
                annotation
                for annotation in (
                    coco_annotation_to_annotation(raw_annotation, width, height, counters, class_id_map)
                    for raw_annotation in annotations_by_image.get(image.get("id"), [])
                )
                if annotation is not None
            ]
            if not annotations and source_config.get("skip_empty", True):
                counters["empty_image_skipped"] += 1
                continue
            records.append(
                ImageRecord(
                    dataset=name,
                    source_split=str(source_split),
                    target_split=normalize_split(str(target_split)),
                    image_path=image_path,
                    width=width,
                    height=height,
                    annotations=annotations,
                    synthetic=synthetic,
                    source_key=source_image_key(image_path),
                )
            )
            counters["images_loaded"] += 1
            counters["objects_loaded"] += len(annotations)
    return records


def build_split_sets(
    records_by_dataset: dict[str, list[ImageRecord]],
    split_mode: str,
    val_fraction: float,
    seed: int,
    combined_slug: str,
    synthetic_names: set[str],
) -> dict[str, dict[str, list[ImageRecord]]]:
    split_sets: dict[str, dict[str, list[ImageRecord]]] = {}
    combined: dict[str, list[ImageRecord]] = defaultdict(list)

    for dataset_name, records in sorted(records_by_dataset.items()):
        synthetic = dataset_name in synthetic_names or any(record.synthetic for record in records)
        if split_mode == "resplit":
            dataset_splits = resplit_records(records, val_fraction, seed + stable_offset(dataset_name), train_only=synthetic)
        elif split_mode == "preserve":
            dataset_splits = preserve_splits(records, train_only=synthetic)
        else:
            raise SystemExit("Unsupported split mode: %s" % split_mode)

        if not synthetic:
            split_sets[dataset_name] = dataset_splits
        for split_name, split_records in dataset_splits.items():
            combined[split_name].extend(split_records)

    split_sets[combined_slug] = dict(combined)
    return split_sets


def preserve_splits(records: Sequence[ImageRecord], train_only: bool) -> dict[str, list[ImageRecord]]:
    splits: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        split_name = "train" if train_only else normalize_split(record.target_split)
        splits[split_name].append(record)
    return dict(splits)


def resplit_records(
    records: Sequence[ImageRecord],
    val_fraction: float,
    seed: int,
    train_only: bool,
) -> dict[str, list[ImageRecord]]:
    if train_only:
        return {"train": list(records)}

    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.source_key or str(record.image_path)] += [record]
    keys = list(grouped)
    random.Random(seed).shuffle(keys)
    val_count = max(1, int(round(len(keys) * val_fraction))) if keys else 0
    val_keys = set(keys[:val_count])
    splits: dict[str, list[ImageRecord]] = {"train": [], "val": []}
    for key, group in grouped.items():
        splits["val" if key in val_keys else "train"].extend(group)
    return splits


def write_yolo_dataset(
    output_root: Path,
    split_records: dict[str, list[ImageRecord]],
    class_names: dict[int, str],
    format_name: str,
    copy_mode: str,
) -> Path:
    all_entries: dict[str, list[str]] = {}
    for split_name, records in sorted(split_records.items()):
        entries = []
        for record in records:
            image_out = output_image_path(output_root, split_name, record)
            label_out = label_path_for_image(output_root, image_out)
            materialize_image(record.image_path, image_out, copy_mode)
            write_yolo_label(label_out, record, format_name)
            entries.append(str(image_out.resolve()).replace("\\", "/"))
        all_entries[split_name] = entries
        write_lines(output_root / ("%s.txt" % split_name), entries)

    data_yaml = {
        "path": str(output_root.resolve()).replace("\\", "/"),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": class_names,
    }
    write_yaml(output_root / "data.yaml", data_yaml)
    for expected_split in ("train", "val", "test"):
        if expected_split not in all_entries:
            write_lines(output_root / ("%s.txt" % expected_split), [])
    return output_root / "data.yaml"


def write_coco_dataset(
    output_root: Path,
    split_records: dict[str, list[ImageRecord]],
    class_names: dict[int, str],
    format_name: str,
    copy_mode: str,
) -> Path:
    annotations_root = output_root / "annotations"
    annotations_root.mkdir(parents=True, exist_ok=True)
    dataset_index = {
        "path": str(output_root.resolve()).replace("\\", "/"),
        "format": format_name,
        "annotations": {},
        "categories": class_names,
    }
    for split_name, records in sorted(split_records.items()):
        coco = build_coco_payload(output_root, split_name, records, class_names, format_name, copy_mode)
        annotation_path = annotations_root / ("%s.json" % split_name)
        save_json(annotation_path, coco)
        dataset_index["annotations"][split_name] = str(annotation_path.relative_to(output_root)).replace("\\", "/")
    write_yaml(output_root / "dataset.yaml", dataset_index)
    return output_root / "dataset.yaml"


def build_coco_payload(
    output_root: Path,
    split_name: str,
    records: Sequence[ImageRecord],
    class_names: dict[int, str],
    format_name: str,
    copy_mode: str,
) -> dict[str, Any]:
    images = []
    annotations = []
    annotation_id = 1
    for image_id, record in enumerate(records, start=1):
        image_out = output_image_path(output_root, split_name, record)
        materialize_image(record.image_path, image_out, copy_mode)
        images.append(
            {
                "id": image_id,
                "file_name": str(image_out.relative_to(output_root)).replace("\\", "/"),
                "width": record.width,
                "height": record.height,
            }
        )
        for annotation in record.annotations:
            polygon_px = denormalize_polygon(annotation.polygon, record.width, record.height)
            bbox = polygon_bbox(polygon_px)
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue
            payload = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": annotation.class_id,
                "bbox": [round(value, 3) for value in bbox],
                "area": round(polygon_area(polygon_px), 3),
                "iscrowd": 0,
            }
            if format_name == "coco_seg":
                payload["segmentation"] = [[round(value, 3) for value in polygon_px]]
            annotations.append(payload)
            annotation_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": class_id, "name": name} for class_id, name in sorted(class_names.items())],
    }


def write_yolo_label(path: Path, record: ImageRecord, format_name: str) -> None:
    lines = []
    for annotation in record.annotations:
        if format_name == "yolo_detect":
            values = polygon_to_yolo_detect(annotation.polygon)
        elif format_name == "yolo_obb":
            values = minimum_area_rectangle(annotation.polygon)
        elif format_name == "yolo_seg":
            values = annotation.polygon
        else:
            raise SystemExit("Unsupported YOLO format: %s" % format_name)
        if not values:
            continue
        lines.append("%d %s" % (annotation.class_id, " ".join("%.6f" % clamp01(value) for value in values)))
    write_lines(path, lines)


def parse_yolo_label_file(
    path: Path,
    width: int,
    height: int,
    label_format: str,
    counters: Counter[str],
    class_id_map: dict[int, int],
) -> list[Annotation]:
    annotations = []
    for line_number, line in enumerate(read_lines(path), start=1):
        parts = line.split()
        if len(parts) < 5:
            counters["malformed_label"] += 1
            continue
        try:
            class_id = int(float(parts[0]))
            values = [float(value) for value in parts[1:]]
        except ValueError:
            counters["malformed_label"] += 1
            continue

        if len(values) == 4 and ("detect" in label_format or "bbox" in label_format):
            polygon = yolo_detect_to_polygon(values)
        elif len(values) >= 6 and len(values) % 2 == 0:
            polygon = [clamp01(value) for value in values]
        else:
            counters["malformed_label"] += 1
            continue
        if polygon_area(denormalize_polygon(polygon, width, height)) <= 0:
            counters["degenerate_label"] += 1
            continue
        annotations.append(
            Annotation(class_id=class_id_map.get(class_id, class_id), polygon=polygon, source="%s:%d" % (path, line_number))
        )
    return annotations


def coco_annotation_to_annotation(
    raw_annotation: dict[str, Any],
    width: int,
    height: int,
    counters: Counter[str],
    class_id_map: dict[int, int],
) -> Annotation | None:
    polygon = choose_coco_polygon(raw_annotation.get("segmentation"))
    if polygon is None:
        bbox = raw_annotation.get("bbox")
        if not bbox or len(bbox) != 4:
            counters["invalid_coco_annotation"] += 1
            return None
        polygon = coco_bbox_to_polygon(bbox)
    normalized = []
    for index in range(0, len(polygon), 2):
        normalized.extend([clamp01(float(polygon[index]) / width), clamp01(float(polygon[index + 1]) / height)])
    return Annotation(
        class_id=class_id_map.get(int(raw_annotation.get("category_id", 0)), int(raw_annotation.get("category_id", 0))),
        polygon=normalized,
        source="coco:%s" % raw_annotation.get("id", ""),
    )


def parse_class_id_map(raw_map: Any) -> dict[int, int]:
    if not raw_map:
        return {}
    return {int(source): int(target) for source, target in dict(raw_map).items()}


def infer_coco_class_id_map(coco: dict[str, Any]) -> dict[int, int]:
    category_ids = sorted(int(category["id"]) for category in coco.get("categories", []) if "id" in category)
    if len(category_ids) == 1:
        return {category_ids[0]: 0}
    return {category_id: index for index, category_id in enumerate(category_ids)}


def choose_coco_polygon(segmentation: Any) -> list[float] | None:
    if not isinstance(segmentation, list) or not segmentation:
        return None
    if all(isinstance(value, (float, int)) for value in segmentation):
        polygon = segmentation
    else:
        polygons = [item for item in segmentation if isinstance(item, list)]
        if not polygons:
            return None
        polygon = max(polygons, key=len)
    if len(polygon) < 6 or len(polygon) % 2 != 0:
        return None
    return [float(value) for value in polygon]


def list_yolo_split_images(source_root: Path, data_yaml: dict[str, Any], split_key: str) -> list[str]:
    split_value = data_yaml.get(split_key, split_key)
    candidate = resolve_dataset_path(source_root, data_yaml, Path(str(split_value)))
    if candidate.is_dir():
        return [str(path) for path in sorted(candidate.rglob("*")) if path.suffix.lower() in IMAGE_EXTENSIONS]
    if candidate.is_file():
        return read_lines(candidate)
    return []


def resolve_yolo_image_path(source_root: Path, image_ref: str) -> Path | None:
    candidate = Path(image_ref)
    candidates = [candidate] if candidate.is_absolute() else [
        source_root / candidate,
        source_root / strip_leading_data(candidate),
        Path.cwd() / candidate,
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def resolve_yolo_label_path(image_path: Path) -> Path | None:
    candidates = []
    parts = list(image_path.parts)
    if "images" in parts:
        index = parts.index("images")
        label_parts = parts[:]
        label_parts[index] = "labels"
        candidates.append(Path(*label_parts).with_suffix(".txt"))
    candidates.append(image_path.with_suffix(".txt"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def discover_yolo_split_map(data_yaml: dict[str, Any]) -> dict[str, str]:
    ignored = {"path", "names", "nc", "download"}
    return {
        str(key): normalize_split(str(key))
        for key, value in data_yaml.items()
        if key not in ignored and isinstance(value, (str, list))
    }


def discover_coco_split_map(source_root: Path) -> dict[str, str]:
    split_map = {}
    for path in source_root.iterdir() if source_root.exists() else []:
        if path.is_dir() and (path / "_annotations.coco.json").exists():
            split_map[path.name] = normalize_split(path.name)
    if (source_root / "_annotations.coco.json").exists():
        split_map["."] = "train"
    return split_map


def coco_annotation_path(source_root: Path, split_name: str, source_config: dict[str, Any]) -> Path:
    if source_config.get("annotation_path"):
        return resolve_path(source_root, Path(str(source_config["annotation_path"])))
    if split_name == ".":
        return source_root / "_annotations.coco.json"
    return source_root / split_name / "_annotations.coco.json"


def resolve_coco_image_path(source_root: Path, split_root: Path, file_name: str) -> Path | None:
    candidate = Path(file_name)
    candidates = [candidate] if candidate.is_absolute() else [
        split_root / candidate,
        source_root / candidate,
        source_root / "images" / candidate,
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def output_image_path(output_root: Path, split_name: str, record: ImageRecord) -> Path:
    return output_root / "images" / split_name / record.dataset / safe_image_name(record.image_path)


def label_path_for_image(output_root: Path, output_image: Path) -> Path:
    return (output_root / "labels" / output_image.relative_to(output_root / "images")).with_suffix(".txt")


def materialize_image(source: Path, destination: Path, copy_mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or copy_mode == "none":
        return
    if copy_mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    if copy_mode == "symlink":
        try:
            os.symlink(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def read_image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def minimum_area_rectangle(polygon: Sequence[float]) -> list[float]:
    points = [(float(polygon[index]), float(polygon[index + 1])) for index in range(0, len(polygon), 2)]
    hull = convex_hull(points)
    if len(hull) < 3:
        return []
    if len(hull) == 4:
        return flatten_points(hull)

    best_area = float("inf")
    best_rectangle: list[tuple[float, float]] = []
    for index, point in enumerate(hull):
        next_point = hull[(index + 1) % len(hull)]
        angle = math.atan2(next_point[1] - point[1], next_point[0] - point[0])
        rotated = [rotate_point(item, -angle) for item in hull]
        xs = [item[0] for item in rotated]
        ys = [item[1] for item in rotated]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        area = (max_x - min_x) * (max_y - min_y)
        if area < best_area:
            best_area = area
            rectangle = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
            best_rectangle = [rotate_point(item, angle) for item in rectangle]
    return flatten_points(best_rectangle)


def convex_hull(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return list(unique)

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def rotate_point(point: tuple[float, float], angle: float) -> tuple[float, float]:
    cos_value = math.cos(angle)
    sin_value = math.sin(angle)
    return (point[0] * cos_value - point[1] * sin_value, point[0] * sin_value + point[1] * cos_value)


def polygon_to_yolo_detect(polygon: Sequence[float]) -> list[float]:
    xs = [float(polygon[index]) for index in range(0, len(polygon), 2)]
    ys = [float(polygon[index]) for index in range(1, len(polygon), 2)]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return [(min_x + max_x) / 2.0, (min_y + max_y) / 2.0, max_x - min_x, max_y - min_y]


def yolo_detect_to_polygon(values: Sequence[float]) -> list[float]:
    x_center, y_center, width, height = [float(value) for value in values]
    x_min = x_center - width / 2.0
    x_max = x_center + width / 2.0
    y_min = y_center - height / 2.0
    y_max = y_center + height / 2.0
    return [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]


def coco_bbox_to_polygon(bbox: Sequence[float]) -> list[float]:
    x_min, y_min, width, height = [float(value) for value in bbox]
    x_max = x_min + width
    y_max = y_min + height
    return [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]


def polygon_bbox(polygon_px: Sequence[float]) -> list[float]:
    xs = [float(polygon_px[index]) for index in range(0, len(polygon_px), 2)]
    ys = [float(polygon_px[index]) for index in range(1, len(polygon_px), 2)]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return [min_x, min_y, max_x - min_x, max_y - min_y]


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


def denormalize_polygon(polygon: Sequence[float], width: int, height: int) -> list[float]:
    result = []
    for index in range(0, len(polygon), 2):
        result.extend([float(polygon[index]) * width, float(polygon[index + 1]) * height])
    return result


def flatten_points(points: Sequence[tuple[float, float]]) -> list[float]:
    result = []
    for x_value, y_value in points:
        result.extend([x_value, y_value])
    return result


def summarize_split_records(split_records: dict[str, list[ImageRecord]]) -> dict[str, Any]:
    return {
        "splits": {
            split_name: {
                "images": len(records),
                "objects": sum(len(record.annotations) for record in records),
            }
            for split_name, records in sorted(split_records.items())
        },
        "total_images": sum(len(records) for records in split_records.values()),
        "total_objects": sum(len(record.annotations) for records in split_records.values() for record in records),
    }


def resolve_dataset_path(source_root: Path, data_yaml: dict[str, Any], path: Path) -> Path:
    if path.is_absolute():
        return path
    base = Path(str(data_yaml.get("path", ".")))
    candidates = [source_root / base / path, source_root / path, source_root / strip_leading_data(path)]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_path(root: Path, path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (root / path).resolve()


def safe_image_name(path: Path) -> Path:
    parts = list(path.parts)
    if "images" in parts:
        index = parts.index("images")
        if len(parts) > index + 2:
            return Path(*parts[index + 2 :])
    return Path(path.name)


def source_image_key(path: Path) -> str:
    stem = path.stem
    if ".rf." in stem:
        stem = stem.split(".rf.", 1)[0]
    return stem


def normalize_formats(raw_formats: Iterable[str]) -> list[str]:
    formats = [str(value).lower().replace("-", "_") for value in raw_formats]
    unsupported = sorted(set(formats) - SUPPORTED_FORMATS)
    if unsupported:
        raise SystemExit("Unsupported output format(s): %s" % ", ".join(unsupported))
    return formats


def normalize_class_names(raw_names: Any) -> dict[int, str]:
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    return {index: str(name) for index, name in enumerate(raw_names)}


def normalize_split(value: str) -> str:
    return SPLIT_ALIASES.get(str(value).lower(), str(value).lower())


def stable_offset(value: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(value)) % 100000


def strip_leading_data(path: Path) -> Path:
    parts = list(path.parts)
    if parts and parts[0].lower() == "data":
        return Path(*parts[1:])
    return path


def group_by(items: Iterable[dict[str, Any]], key: str) -> dict[Any, list[dict[str, Any]]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[item.get(key)].append(item)
    return grouped


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit("YAML file not found: %s" % path)
    with path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(payload, file_obj, allow_unicode=True, sort_keys=False)


def load_json(path: Path) -> Any:
    if not path.exists():
        raise SystemExit("JSON file not found: %s" % path)
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as file_obj:
        return [line.strip() for line in file_obj if line.strip()]


def write_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for line in lines:
            file_obj.write(str(line).replace("\\", "/") + "\n")


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


if __name__ == "__main__":
    raise SystemExit(main())
