# Frozen annotations and evaluation data

The included `data/reproduction/annotations_seed42.tar.gz` (7.05 MB) contains
4,196 OBB label files, 11 domain configurations, the fixed split and integrity
metadata. Images are downloaded separately from the article's sources.

## Verify the bundle

From the repository root in the demo environment:

```bash
python scripts/prepare_reproduction_inputs.py
```

The command verifies the archive SHA-256, all 4,230 inventory entries and the
frozen test labels. Expected: **518 images, 60,287 objects**. The report is written
to `outputs/data_export_validation.json`. Checksums are listed in the
[bundle description](../data/reproduction/README.md).

## Prepare local images

Extract source images under `data/raw/images/<dataset_slug>/`. Preserve original
filenames. The importer requires an exact relative-key match or one unique
filename within the domain; missing or ambiguous files stop preparation.

| Paper dataset | Directory slug |
|---|---|
| 40-70 mm | `40_70mm` |
| Mask 2018 | `mask_2018` |
| Open pits | `open_pits_urfu_2025` |
| Rock blasting | `rock_blasting` |
| Rocks UAV | `rock_bpla` |
| Rocks asbest | `rocks_asbest` |
| Ronveer | `ronveer` |
| SAM fragment v7 | `sam_fragment_v7_all_0_105` |
| Transport | `transport` |
| Unloading a wagon | `vgok` |
| Synthetic, train-only | `synthetic` |

```bash
python scripts/prepare_reproduction_inputs.py --images-root data/raw/images --output-root data/prepared/frozen
```

Choose a new empty output directory. The command copies images and labels,
writes per-domain `data.yaml`, `train.txt` and `val.txt` from the frozen manifest,
and records image SHA-256 values in `image_manifest.json`. Use `--datasets transport`
to prepare just Transport. No separate package or preparation step is needed.
The resulting `val` list is the paper evaluation partition (`test` aliases it).
Synthetic has no evaluation images. On Windows, use a short data path if long-path
support is disabled.

Source image hashes were not recorded in the original export. The new hashes
record local inputs; filename matching alone does not prove historical pixel
identity. Keep the source dataset versions with your results.

## Evaluate Transport

The article's [Mendeley v1 URL](https://data.mendeley.com/datasets/v9grvgsrh9/1)
remains in the source manifest and supplies the inference demo image. The checked
annotation example uses `transport.zip` from the updated
[Mendeley v2 release](https://data.mendeley.com/datasets/v9grvgsrh9/2).
Its archive SHA-256 is
`70c6099258d5755877740a32a53b127da3942b1f6cf1ec1c9171ba13fb01c7f1`.
Extract it under `data/raw/images/transport/`, then run:

```bash
python scripts/prepare_reproduction_inputs.py --images-root data/raw/images --datasets transport --output-root data/prepared/frozen_transport
python scripts/predict_ultralytics_obb.py --weights weights/lightweight/stones-yolo11n-obb-seed42.pt --data-yaml data/prepared/frozen_transport/transport/data.yaml --split val --imgsz 1024 --conf 0.25 --max-det 5000 --device cpu --output outputs/transport/nano/predictions.json
python scripts/evaluate_detection.py --data-yaml data/prepared/frozen_transport/transport/data.yaml --split val --predictions outputs/transport/nano/predictions.json --confidence-threshold 0.25 --output outputs/transport/nano/detection.csv
python scripts/evaluate_granulometry.py --data-yaml data/prepared/frozen_transport/transport/data.yaml --split val --predictions outputs/transport/nano/predictions.json --confidence-threshold 0.25 --output outputs/transport/nano/size.csv
```

Expected data: **107 training images / 1,480 objects** and **27 evaluation images /
367 objects**. Use `stones-yolo11s-obb-seed42.pt` and `outputs/transport/small/` for
the second demo model. CSV/JSON outputs describe these nano/small predictions;
the paper's large-model tables remain in `results/tables/`. No training is needed.
