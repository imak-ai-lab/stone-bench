from pathlib import Path
from typing import Any, Dict, Iterable, List

from stones_fragmentation.io import list_image_paths, load_json, save_json


def run_inference(config: Dict[str, Any]) -> Dict[str, Any]:
    model_config = config.get("model", {})
    model_type = str(model_config.get("type", "yolo")).lower()
    images = list_image_paths(config.get("data", {}).get("images", []))

    if model_type == "yolo":
        predictions = infer_yolo(images, model_config)
    elif model_type == "sam_auto":
        predictions = infer_sam_auto(images, model_config)
    elif model_type == "sam_prompt":
        predictions = infer_sam_prompt(images, model_config)
    elif model_type == "owlv2":
        predictions = infer_owlv2(images, model_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    output_path = Path(config.get("output", {}).get("predictions", "outputs/predictions.json"))
    save_json(output_path, predictions)
    return {"images": len(images), "predictions": str(output_path)}


def infer_yolo(images: Iterable[Path], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    from ultralytics import YOLO

    model = YOLO(config["weights"])
    predictions = []
    for image in images:
        results = model.predict(
            source=str(image),
            conf=float(config.get("confidence", 0.25)),
            imgsz=int(config.get("image_size", 1024)),
            device=config.get("device"),
            verbose=False,
        )
        predictions.append({"image": str(image), "detections": parse_ultralytics_result(results[0])})
    return predictions


def infer_sam_auto(images: Iterable[Path], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    from ultralytics import SAM

    model = SAM(config["weights"])
    predictions = []
    for image in images:
        results = model.predict(source=str(image), device=config.get("device"), verbose=False)
        predictions.append({"image": str(image), "detections": parse_ultralytics_result(results[0])})
    return predictions


def infer_sam_prompt(images: Iterable[Path], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    from ultralytics import SAM

    model = SAM(config["weights"])
    prompt_boxes = load_prompt_boxes(config.get("prompt_boxes"))
    predictions = []
    for image in images:
        boxes = prompt_boxes.get(str(image), prompt_boxes.get(image.name, []))
        results = model.predict(source=str(image), bboxes=boxes, device=config.get("device"), verbose=False)
        predictions.append({"image": str(image), "detections": parse_ultralytics_result(results[0])})
    return predictions


def infer_owlv2(images: Iterable[Path], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    from transformers import pipeline

    device = 0 if str(config.get("device", "cpu")).startswith("cuda") else -1
    detector = pipeline("zero-shot-object-detection", model=config["name"], device=device)
    labels = config.get("labels", ["stone"])
    threshold = float(config.get("confidence", 0.20))
    predictions = []
    for image in images:
        raw = detector(str(image), candidate_labels=labels, threshold=threshold)
        predictions.append({"image": str(image), "detections": raw})
    return predictions


def parse_ultralytics_result(result: Any) -> List[Dict[str, Any]]:
    if getattr(result, "obb", None) is not None:
        return parse_obb(result)
    if getattr(result, "boxes", None) is not None:
        return parse_boxes(result)
    return []


def parse_obb(result: Any) -> List[Dict[str, Any]]:
    points = result.obb.xyxyxyxy.cpu().tolist()
    scores = result.obb.conf.cpu().tolist()
    classes = result.obb.cls.cpu().tolist()
    names = result.names or {}
    return [
        {
            "label": str(names.get(int(cls), int(cls))),
            "score": float(score),
            "obb_xyxyxyxy": [[float(x), float(y)] for x, y in polygon],
        }
        for polygon, score, cls in zip(points, scores, classes)
    ]


def parse_boxes(result: Any) -> List[Dict[str, Any]]:
    boxes = result.boxes.xyxy.cpu().tolist()
    scores = result.boxes.conf.cpu().tolist()
    classes = result.boxes.cls.cpu().tolist()
    names = result.names or {}
    return [
        {
            "label": str(names.get(int(cls), int(cls))),
            "score": float(score),
            "bbox_xyxy": [float(value) for value in box],
        }
        for box, score, cls in zip(boxes, scores, classes)
    ]


def load_prompt_boxes(path: Any) -> Dict[str, List[List[float]]]:
    if not path:
        return {}
    prompt_path = Path(path)
    if not prompt_path.exists():
        return {}
    return load_json(prompt_path)
