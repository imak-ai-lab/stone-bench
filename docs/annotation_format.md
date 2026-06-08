# Annotation Format

StoneBench uses a single foreground class, `stone`.

## Canonical Prediction JSON

Prediction files contain a list of image records:

```json
[
  {
    "image_id": "path/or/id/of/image.jpg",
    "detections": [
      {
        "label": "stone",
        "score": 0.99,
        "bbox_xywh": [0.0, 0.0, 10.0, 10.0],
        "obb_xyxyxyxy": [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0],
        "metadata": {}
      }
    ]
  }
]
```

`obb_xyxyxyxy` is the preferred geometry for detection and granulometric
metrics. `bbox_xywh` is accepted as an axis-aligned fallback.

## YOLO OBB Labels

YOLO OBB labels use normalized corner coordinates:

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

## YOLO Detect Labels

Axis-aligned labels use the standard normalized YOLO format:

```text
class_id x_center y_center width height
```

## Conversion Rules

- OBB-to-AABB conversion takes the enclosing axis-aligned box of the OBB.
- AABB-to-OBB conversion uses the zero-rotation rectangle corners.
- Segmentation polygons are normalized and clipped to image bounds during data preparation.
- Single-class COCO category IDs are mapped to class `0` for YOLO exports.
