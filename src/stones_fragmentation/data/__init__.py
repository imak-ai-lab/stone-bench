import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from stones_fragmentation.io import load_yaml, save_json, write_yaml


SPLIT_ALIASES = {
    "train": "train",
    "valid": "val",
    "validation": "val",
    "val": "val",
    "test": "test",
    "default_1": "val",
}
def prepare_data(config: Dict[str, Any], limit: Optional[int] = None) -> Dict[str, Any]:
    output_root = Path(config.get("output", {}).get("root", "data/prepared/stones_seg"))
    copy_mode = str(config.get("output", {}).get("copy_mode", "copy"))
    dataset_config = config.get("dataset", {})
    names = dataset_config.get("names", ["stone"])
    source_configs = dataset_config.get("sources", [])
    if not source_configs:
        raise ValueError("No dataset.sources configured.")

    output_root.mkdir(parents=True, exist_ok=True)
    split_entries = defaultdict(list)
    report = {
        "output_root": str(output_root),
        "copy_mode": copy_mode,
        "sources": {},
        "metadata_xlsx": config.get("input", {}).get("metadata_xlsx"),
    }

    for source_config in source_configs:
        source_name = str(source_config["name"])
        source_type = str(source_config.get("type", "coco_segmentation")).lower()
        counters = Counter()
        if source_type in ("coco", "coco_segmentation", "coco-segmentation"):
            entries = prepare_coco_source(
                source_config=source_config,
                output_root=output_root,
                copy_mode=copy_mode,
                counters=counters,
                limit=limit or dataset_config.get("max_images_per_source_split"),
            )
        elif source_type in ("yolo", "yolo_segmentation", "yolo-segmentation"):
            entries = prepare_yolo_source(
                source_config=source_config,
                output_root=output_root,
                copy_mode=copy_mode,
                counters=counters,
                limit=limit or dataset_config.get("max_images_per_source_split"),
            )
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

        for split_name, image_paths in entries.items():
            split_entries[split_name].extend(image_paths)
        report["sources"][source_name] = dict(counters)

    data_yaml = write_dataset_index(output_root, split_entries, names)
    report["data_yaml"] = str(data_yaml)
    report["splits"] = {split: len(paths) for split, paths in split_entries.items()}
    save_json(output_root / "manifest.json", report)
    return report


def prepare_coco_source(
    source_config: Dict[str, Any],
    output_root: Path,
    copy_mode: str,
    counters: Counter,
    limit: Optional[int],
) -> Dict[str, List[str]]:
    source_root = Path(source_config["path"])
    split_map = source_config.get("split_map") or discover_coco_splits(source_root)
    entries = defaultdict(list)

    for source_split, target_split in split_map.items():
        split_dir = source_root / source_split
        annotation_path = split_dir / "_annotations.coco.json"
        if not annotation_path.exists():
            counters[f"missing_coco_annotations:{source_split}"] += 1
            continue
        with annotation_path.open("r", encoding="utf-8") as file:
            coco = json.load(file)

        annotations_by_image = group_coco_annotations(coco.get("annotations", []))
        images = coco.get("images", [])
        if limit:
            images = images[: int(limit)]

        for image in images:
            image_path = split_dir / image["file_name"]
            if not image_path.exists():
                counters["missing_image"] += 1
                continue

            label_lines = []
            for annotation in annotations_by_image.get(image["id"], []):
                line = coco_annotation_to_yolo_line(annotation, image["width"], image["height"])
                if line is not None:
                    label_lines.append(line)
                else:
                    counters["invalid_annotation"] += 1

            if not label_lines and bool(source_config.get("skip_empty", True)):
                counters["empty_image_skipped"] += 1
                continue

            output_image = output_image_path(
                output_root=output_root,
                split=normalize_split(target_split),
                source_name=str(source_config["name"]),
                relative_name=Path(image["file_name"]).name,
            )
            output_label = label_path_for_image(output_root, output_image)
            materialize_file(image_path, output_image, copy_mode)
            write_label(output_label, label_lines)
            entries[normalize_split(target_split)].append(str(output_image.resolve()))
            counters["images_written"] += 1
            counters["labels_written"] += len(label_lines)

    return entries


def prepare_yolo_source(
    source_config: Dict[str, Any],
    output_root: Path,
    copy_mode: str,
    counters: Counter,
    limit: Optional[int],
) -> Dict[str, List[str]]:
    source_root = Path(source_config["path"])
    data_yaml = load_yaml(str(source_root / source_config.get("data_yaml", "data.yaml")))
    split_map = source_config.get("split_map") or {
        key: normalize_split(key)
        for key, value in data_yaml.items()
        if isinstance(value, str) and key not in {"path", "names", "nc", "download"}
    }
    entries = defaultdict(list)

    for source_split, target_split in split_map.items():
        list_path = resolve_yolo_list_path(source_root, data_yaml.get(source_split, source_split))
        if not list_path.exists():
            counters[f"missing_list:{source_split}"] += 1
            continue
        image_refs = read_lines(list_path)
        if limit:
            image_refs = image_refs[: int(limit)]

        for image_ref in image_refs:
            image_path = resolve_yolo_image_path(source_root, image_ref)
            if image_path is None:
                counters["missing_image"] += 1
                continue
            label_path = resolve_yolo_label_path(image_path)
            if label_path is None:
                counters["missing_label"] += 1
                if bool(source_config.get("skip_missing_labels", True)):
                    continue
                label_lines = []
            else:
                label_lines = normalize_yolo_label_lines(read_lines(label_path))

            if not label_lines and bool(source_config.get("skip_empty", True)):
                counters["empty_image_skipped"] += 1
                continue

            relative_name = relative_image_name(image_path)
            output_image = output_image_path(
                output_root=output_root,
                split=normalize_split(target_split),
                source_name=str(source_config["name"]),
                relative_name=relative_name,
            )
            output_label = label_path_for_image(output_root, output_image)
            materialize_file(image_path, output_image, copy_mode)
            write_label(output_label, label_lines)
            entries[normalize_split(target_split)].append(str(output_image.resolve()))
            counters["images_written"] += 1
            counters["labels_written"] += len(label_lines)

    return entries


def discover_coco_splits(source_root: Path) -> Dict[str, str]:
    split_map = {}
    for path in source_root.iterdir() if source_root.exists() else []:
        if path.is_dir() and (path / "_annotations.coco.json").exists():
            split_map[path.name] = normalize_split(path.name)
    return split_map


def group_coco_annotations(annotations: Iterable[Dict[str, Any]]) -> Dict[Any, List[Dict[str, Any]]]:
    grouped = defaultdict(list)
    for annotation in annotations:
        grouped[annotation["image_id"]].append(annotation)
    return grouped


def coco_annotation_to_yolo_line(
    annotation: Dict[str, Any],
    width: float,
    height: float,
) -> Optional[str]:
    polygon = choose_coco_polygon(annotation.get("segmentation"))
    if polygon is None or width <= 0 or height <= 0:
        return None
    normalized = []
    for index in range(0, len(polygon), 2):
        x_value = clamp(float(polygon[index]) / float(width), 0.0, 1.0)
        y_value = clamp(float(polygon[index + 1]) / float(height), 0.0, 1.0)
        normalized.extend([x_value, y_value])
    if len(normalized) < 6:
        return None
    return "0 " + " ".join(f"{value:.6f}" for value in normalized)


def choose_coco_polygon(segmentation: Any) -> Optional[List[float]]:
    if not isinstance(segmentation, list) or not segmentation:
        return None
    if all(isinstance(value, (int, float)) for value in segmentation):
        polygon = segmentation
    else:
        polygons = [item for item in segmentation if isinstance(item, list)]
        if not polygons:
            return None
        polygon = max(polygons, key=len)
    if len(polygon) < 6 or len(polygon) % 2 != 0:
        return None
    return [float(value) for value in polygon]


def resolve_yolo_list_path(source_root: Path, list_value: Any) -> Path:
    candidate = Path(str(list_value))
    candidates = [candidate] if candidate.is_absolute() else [
        source_root / candidate,
        source_root / strip_leading_data(candidate),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def resolve_yolo_image_path(source_root: Path, image_ref: str) -> Optional[Path]:
    candidate = Path(image_ref)
    candidates = [candidate] if candidate.is_absolute() else [
        source_root / candidate,
        source_root / strip_leading_data(candidate),
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def resolve_yolo_label_path(image_path: Path) -> Optional[Path]:
    parts = list(image_path.parts)
    candidates = []
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


def normalize_yolo_label_lines(lines: Iterable[str]) -> List[str]:
    normalized = []
    for line in lines:
        parts = line.split()
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            continue
        try:
            values = [clamp(float(value), 0.0, 1.0) for value in parts[1:]]
        except ValueError:
            continue
        normalized.append("0 " + " ".join(f"{value:.6f}" for value in values))
    return normalized


def strip_leading_data(path: Path) -> Path:
    parts = list(path.parts)
    if parts and parts[0].lower() == "data":
        return Path(*parts[1:])
    return path


def relative_image_name(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        index = parts.index("images")
        if len(parts) > index + 2:
            return Path(*parts[index + 2 :])
    return Path(image_path.name)


def output_image_path(output_root: Path, split: str, source_name: str, relative_name: Any) -> Path:
    return output_root / "images" / split / source_name / Path(relative_name)


def label_path_for_image(output_root: Path, output_image: Path) -> Path:
    relative = output_image.relative_to(output_root / "images")
    return (output_root / "labels" / relative).with_suffix(".txt")


def materialize_file(source: Path, destination: Path, copy_mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    if copy_mode == "none":
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


def write_label(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        if lines:
            file.write("\n".join(lines) + "\n")


def write_dataset_index(
    output_root: Path,
    split_entries: Dict[str, List[str]],
    names: List[str],
) -> Path:
    for split_name, image_paths in split_entries.items():
        list_path = output_root / f"{split_name}.txt"
        list_path.parent.mkdir(parents=True, exist_ok=True)
        with list_path.open("w", encoding="utf-8") as file:
            for image_path in sorted(image_paths):
                file.write(image_path.replace("\\", "/") + "\n")

    data_yaml = {
        "path": str(output_root.resolve()),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": {index: name for index, name in enumerate(names)},
    }
    data_yaml_path = output_root / "data.yaml"
    write_yaml(data_yaml_path, data_yaml)
    return data_yaml_path


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def normalize_split(split: str) -> str:
    return SPLIT_ALIASES.get(str(split).lower(), str(split).lower())


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(float(value), upper))
