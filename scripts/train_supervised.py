"""Train local supervised StoneBench baselines with Ultralytics.

Supported baselines:
- YOLO11x-OBB on prepared ``yolo_obb`` labels;
- RT-DETR-X on prepared ``yolo_detect`` AABB labels.

The script is local-only. It does not submit jobs and does not assume any cloud
filesystem. Use ``--dry-run`` to validate resolved datasets and train arguments
without importing Ultralytics or loading weights.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
COMBINED_SLUG = "all_real_plus_synthetic_train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO11x-OBB or RT-DETR-X locally.")
    parser.add_argument("--config", required=True, help="YAML config from configs/supervised.")
    parser.add_argument("--prepared-root", default="", help="Override data.prepared_root.")
    parser.add_argument("--output-root", default="", help="Override output.root.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset slugs.")
    parser.add_argument("--scope", choices=("combined", "per-dataset"), default="", help="Override data.scope.")
    parser.add_argument("--device", default="", help="Override training.device.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", default="")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--amp", dest="amp", action="store_true", default=None)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--cache", default="")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--keep-training-artifacts", action="store_true")
    parser.add_argument("--clean-partial-run-before-train", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_path(Path(args.config))
    config = load_yaml(config_path)
    resolved = resolve_config(config, args)
    dataset_dirs = discover_dataset_dirs(
        prepared_root=Path(resolved["data"]["prepared_root"]),
        scope=str(resolved["data"]["scope"]),
        selected=args.datasets,
        combined_slug=str(resolved["data"].get("combined_slug", COMBINED_SLUG)),
    )
    if not dataset_dirs:
        raise SystemExit("No prepared datasets found under %s" % resolved["data"]["prepared_root"])

    output_root = Path(resolved["output"]["root"])
    plan = {
        "config": str(config_path),
        "model": resolved["model"],
        "training": resolved["training"],
        "data": resolved["data"],
        "output_root": str(output_root),
        "datasets": [dataset_plan(dataset_dir, resolved) for dataset_dir in dataset_dirs],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    save_json(output_root / "train_plan.json", plan)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    results = []
    for dataset_dir in dataset_dirs:
        started = time.time()
        try:
            result = train_one_dataset(dataset_dir, resolved, args)
            result["status"] = "ok"
            result["duration_seconds"] = round(time.time() - started, 3)
        except Exception as exc:
            result = {
                "status": "failed",
                "dataset": dataset_dir.name,
                "data_yaml": str(dataset_dir / "data.yaml"),
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "duration_seconds": round(time.time() - started, 3),
            }
        results.append(result)
        save_json(output_root / "run_summary.json", results)

    return 0 if all(item.get("status") == "ok" for item in results) else 1


def resolve_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    resolved = json.loads(json.dumps(config))
    model = resolved.setdefault("model", {})
    training = resolved.setdefault("training", {})
    data = resolved.setdefault("data", {})
    output = resolved.setdefault("output", {})

    data["prepared_root"] = str(resolve_path(Path(args.prepared_root or data.get("prepared_root", ""))))
    data["scope"] = args.scope or data.get("scope", "combined")
    output["root"] = str(resolve_path(Path(args.output_root or output.get("root", "outputs/supervised_benchmark"))))

    if args.device:
        training["device"] = args.device
    for key in ("epochs", "imgsz", "workers", "patience", "seed"):
        value = getattr(args, key)
        if value is not None:
            training[key] = value
    if args.batch:
        training["batch"] = args.batch
    if args.amp is not None:
        training["amp"] = bool(args.amp)
    if args.cache:
        training["cache"] = args.cache
    if args.plots:
        training["plots"] = True

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
    training.setdefault("plots", False)
    return resolved


def discover_dataset_dirs(
    prepared_root: Path,
    scope: str,
    selected: Sequence[str] | None,
    combined_slug: str,
) -> list[Path]:
    selected_set = set(selected or [])
    if scope == "combined":
        dataset_dir = prepared_root / combined_slug
        if selected_set and combined_slug not in selected_set:
            return []
        return [dataset_dir] if (dataset_dir / "data.yaml").exists() else []

    dirs = []
    for path in sorted(prepared_root.iterdir() if prepared_root.exists() else []):
        if not path.is_dir() or path.name == combined_slug:
            continue
        if selected_set and path.name not in selected_set:
            continue
        if (path / "data.yaml").exists():
            dirs.append(path)
    missing = selected_set - {path.name for path in dirs}
    if missing:
        raise SystemExit("Requested datasets were not found: %s" % ", ".join(sorted(missing)))
    return dirs


def dataset_plan(dataset_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    model_id = str(config["model"]["model_id"])
    run_name = "%s-%s" % (dataset_dir.name, slugify(model_id))
    return {
        "dataset": dataset_dir.name,
        "data_yaml": str(dataset_dir / "data.yaml"),
        "run_dir": str(Path(config["output"]["root"]) / run_name),
    }


def train_one_dataset(dataset_dir: Path, config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise RuntimeError("Missing data.yaml: %s" % data_yaml)

    model_id = str(config["model"]["model_id"])
    family = str(config["model"]["family"])
    run_name = "%s-%s" % (dataset_dir.name, slugify(model_id))
    output_root = Path(config["output"]["root"])
    run_dir = output_root / run_name

    if args.skip_existing and (run_dir / "result.json").exists():
        return load_json(run_dir / "result.json")
    if args.clean_partial_run_before_train and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_ultralytics_model(family, model_id)
    train_kwargs = build_train_kwargs(config, data_yaml, output_root, run_name)
    train_result = model.train(**train_kwargs)
    train_payload = serialize_ultralytics_result(train_result)

    val_kwargs = build_val_kwargs(config, data_yaml)
    val_result = model.val(**val_kwargs)
    val_payload = serialize_ultralytics_result(val_result)

    payload = {
        "dataset": dataset_dir.name,
        "model_family": family,
        "model_id": model_id,
        "label_format": config["model"].get("label_format"),
        "data_yaml": str(data_yaml),
        "run_dir": str(run_dir),
        "train_kwargs": train_kwargs,
        "val_kwargs": val_kwargs,
        "train_result": train_payload,
        "val_result": val_payload,
    }
    save_json(run_dir / "val_metrics.json", val_payload)
    save_json(run_dir / "result.json", payload)
    if not args.keep_training_artifacts:
        cleanup_training_artifacts(run_dir)
    return payload


def build_train_kwargs(config: dict[str, Any], data_yaml: Path, project_dir: Path, run_name: str) -> dict[str, Any]:
    training = config["training"]
    family = str(config["model"]["family"])
    kwargs = {
        "data": str(data_yaml),
        "epochs": int(training["epochs"]),
        "imgsz": int(training["imgsz"]),
        "batch": parse_batch(training["batch"]),
        "device": training.get("device", "cuda"),
        "workers": int(training.get("workers", 8)),
        "patience": int(training.get("patience", 10)),
        "seed": int(training.get("seed", 42)),
        "amp": bool(training.get("amp", True)),
        "plots": bool(training.get("plots", False)),
        "project": str(project_dir),
        "name": run_name,
        "exist_ok": True,
        "val": True,
    }
    if family == "yolo-obb":
        kwargs["task"] = "obb"
    if training.get("cache"):
        kwargs["cache"] = parse_cache(str(training["cache"]))
    return kwargs


def build_val_kwargs(config: dict[str, Any], data_yaml: Path) -> dict[str, Any]:
    training = config["training"]
    family = str(config["model"]["family"])
    kwargs = {
        "data": str(data_yaml),
        "split": "val",
        "imgsz": int(training["imgsz"]),
        "batch": parse_batch(training["batch"]),
        "device": training.get("device", "cuda"),
        "workers": int(training.get("workers", 8)),
        "plots": bool(training.get("plots", False)),
    }
    if family == "yolo-obb":
        kwargs["task"] = "obb"
    return kwargs


def build_ultralytics_model(family: str, model_id: str) -> Any:
    try:
        from ultralytics import RTDETR, YOLO
    except ImportError as exc:
        raise RuntimeError("Install Ultralytics first: python -m pip install -e .[yolo]") from exc
    if family == "rtdetr":
        return RTDETR(model_id)
    if family == "yolo-obb":
        return YOLO(model_id)
    raise RuntimeError("Unsupported supervised model family: %s" % family)


def cleanup_training_artifacts(run_dir: Path) -> None:
    keep = {"result.json", "val_metrics.json", "args.yaml", "weights"}
    for path in run_dir.iterdir() if run_dir.exists() else []:
        if path.name in keep:
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def serialize_ultralytics_result(result: Any) -> dict[str, Any]:
    payload = {"class": result.__class__.__name__}
    for attr in ("save_dir", "speed", "fitness", "results_dict"):
        if hasattr(result, attr):
            payload[attr] = to_jsonable(getattr(result, attr))
    for attr in ("box", "obb", "seg"):
        value = getattr(result, attr, None)
        if value is not None:
            payload[attr] = to_jsonable(value)
    return payload


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "__dict__"):
        return {key: to_jsonable(item) for key, item in vars(value).items() if not key.startswith("_")}
    return str(value)


def parse_batch(value: Any) -> Any:
    text = str(value)
    if text.lower() == "auto":
        return -1
    try:
        return int(text)
    except ValueError:
        return text


def parse_cache(value: str) -> Any:
    text = str(value).strip().lower()
    if text in ("1", "true", "ram"):
        return True
    if text in ("0", "false", "none", ""):
        return False
    return text


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def slugify(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "-" for char in value)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "model"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj) or {}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
