"""Exercise both public metric CLIs on generated geometry with known answers, without weights/data."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/metric_example")
    args = parser.parse_args()
    output = args.output.resolve()
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "labels").mkdir(exist_ok=True)
    boxes = [[16, 16, 24, 16, 24, 24, 16, 24], [40, 40, 64, 40, 64, 64, 40, 64]]
    image = Image.new("RGB", (128, 128), "black")
    draw = ImageDraw.Draw(image)
    for polygon in boxes:
        draw.polygon(list(zip(polygon[::2], polygon[1::2])), fill="gray")
    image.save(output / "images/example.png")
    (output / "labels/example.txt").write_text("\n".join(
        "0 " + " ".join(str(v / 128) for v in polygon) for polygon in boxes
    ) + "\n", encoding="utf-8")
    (output / "test.txt").write_text("images/example.png\n", encoding="utf-8")
    (output / "data.yaml").write_text("path: .\ntest: test.txt\nnames: [stone]\n", encoding="utf-8")
    perfect = [dict(label="stone", score=.9, obb_xyxyxyxy=p) for p in boxes]
    cases = {"perfect": perfect, "miss_small": perfect[1:]}
    for case, predictions in cases.items():
        directory = output / case
        directory.mkdir(exist_ok=True)
        (directory / "predictions.json").write_text(json.dumps([
            dict(image_id="example.png", detections=predictions)
        ], indent=2), encoding="utf-8")
        for script, name in [("evaluate_detection.py", "detection"), ("evaluate_granulometry.py", "size")]:
            subprocess.run([
                sys.executable, str(ROOT / "scripts" / script),
                "--data-yaml", str(output / "data.yaml"), "--split", "test",
                "--predictions", str(directory / "predictions.json"),
                "--confidence-threshold", "0.25", "--bootstrap-samples", "0",
                "--output", str(directory / (name + ".csv")),
            ], check=True, cwd=ROOT)
        detection = json.loads((directory / "detection.json").read_text())[0]
        size = json.loads((directory / "size.json").read_text())[0]
        expected = 1.0 if case == "perfect" else 2 / 3
        for threshold in ["0.50", "0.75"]:
            if not math.isclose(detection["iou"][threshold]["f1"], expected, abs_tol=1e-9):
                raise RuntimeError("Unexpected F1: %s" % case)
        if not math.isclose(size["missed_small"], 0.0 if case == "perfect" else 1.0):
            raise RuntimeError("Unexpected missed-small fraction: %s" % case)
        if case == "perfect" and any(abs(size[key]) > 1e-9 for key in ["ks", "w1", "cdf_rmse", "d80_abs_error"]):
            raise RuntimeError("Perfect geometry must have zero size-distribution error")
        print("%s: F1@50=%.6f, missed_small=%.1f" % (case, expected, size["missed_small"]))
    print("Metric example passed. Generated geometry only, not paper accuracy: %s" % output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
