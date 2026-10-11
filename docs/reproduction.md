# Reproducing the examples and paper protocol

The repository provides executable nano/small inference and detection/size
metrics. Start with [the demo](camera_ready.md), then run
`python scripts/metric_example.py` for a small example with known metric values.
[Restore the bundled annotations](data_export.md) to evaluate predictions on the
paper's frozen split. Full experiment training is described below; training
pipelines and large-model checkpoints are outside this compact companion.

## Data and split

Use the [article dataset sources](../data/manifests/dataset_sources.md) and
`data/manifests/unified_split_seed42.json`. The common evaluation contains
518 images and 60,287 objects across 10 real domains; Synthetic is train-only.
The same partition was used for validation. Source-key checks do not establish
independence between acquisition sessions. Keep the fixed partition for all runs.

`scripts/prepare_reproduction_inputs.py` verifies the bundled OBB labels and,
when given local images, writes each domain's `data.yaml`, `train.txt` and
`val.txt` directly from the manifest. No new random split is generated.

## Training protocol

The paper evaluates YOLO11x-OBB and RT-DETR-X in per-domain and combined settings.
Use the pretrained model of the stated architecture, one class (`stone`),
image size 1024, 30 epochs and patience 10; batch size is 4 for YOLO11x-OBB and
2 for RT-DETR-X. Train separate runs with seeds 42, 43 and 44. The split seed
remains 42. For RT-DETR, convert each OBB to its enclosing axis-aligned box.
For combined training, concatenate the real-domain training lists and add
Synthetic only to training; combine the real-domain evaluation lists separately.

For the YOLO tiling experiment in Table VI, train on 1024-pixel tiles at overlaps
0, 128 and 256. At inference, shift tile predictions to original-image coordinates
and merge with class-aware NMS at IoU 0.50. Retain candidates down to confidence
0.001 for AP evaluation, with limits of 3000 detections per tile and 5000 per image.
The lightweight demo instead applies tiled inference to the included fixed weights.

Record the resolved model settings, package versions, image versions, checkpoint
hashes, hardware and prediction files for each run. Exact regeneration of the paper
numbers requires the corresponding original run settings and artifacts; the
included nano/small checkpoints demonstrate the method with smaller models.

## Evaluation

For zero-shot methods, use the paper's models/prompts without stone-specific
training, the same frozen evaluation images and confidence 0.30 for F1. Supervised
F1 uses 0.25. Preserve a prediction record for every image, including empty
predictions. Scores discarded during inference cannot be recovered for AP.

The [Transport example](data_export.md#evaluate-transport) runs both prediction
and evaluation with real annotations. For other domains, substitute their
`data.yaml` and one model/run's predictions in the same commands. Detection tools
report precision, recall and F1; size tools report CDF RMSE, KS, W1, D10/D80 errors
and missed-small fraction. Geometry and uncertainty definitions are in the
[evaluation protocol](evaluation_protocol.md).

Image-bootstrap defaults are 2000 replicates, seed 42, 95% confidence. These
intervals differ from the sample SD across three training seeds. To compute
equal-domain macro-F1, pass one run's multi-domain detection JSON to
`python scripts/summarize_macro_f1.py --input ONE_RUN.json --output MACRO.json`.

The CSV files in `results/tables/` contain the article's reported values.
`python scripts/make_tables.py --output results/tables` checks their structure;
it does not rerun experiments or recalculate paper results.
