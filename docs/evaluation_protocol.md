# Evaluation protocol of the submitted paper

The common evaluation uses `unified_split_seed42.json`: 518 real-domain images and
60,287 GT instances. Source-image variants are grouped; Synthetic is train-only.
The split also served as validation and does not guarantee separation by video or
acquisition session. Do not call it an independent held-out session test.

One class, stone, is evaluated as instances. Mask outputs become minimum-area OBBs;
AABB outputs become zero-angle OBBs and can be penalized on rotated fragments.
Score-ordered greedy matching uses the best unmatched GT by polygon IoU.
Zero-shot confidence is 0.30 with class-agnostic matching; supervised F1 uses 0.25.
Generation thresholds are distinct from evaluation thresholds. The included
detection tools report precision/recall/F1 at IoU 0.50 and 0.75. Paper AP requires
low-cutoff predictions and the corresponding model evaluator.

Training variation is mean plus/minus sample SD over seeds 42/43/44. The public
metric tools additionally offer deterministic 95% image-cluster bootstrap intervals
(2000 replicates, seed 42). All fragments of an image are resampled together.
These are different uncertainties and must not share an unlabeled plus/minus sign.

Size is `2 * sqrt(OBB_area / pi)`. The size evaluator reports CDF RMSE, KS,
Wasserstein-1, absolute D10/D80 error, and the fraction of GT fragments below GT P10
unmatched at IoU 0.50. The submitted Table VII includes D80 and Small, not absolute
D10. D10 and equal-domain macro-F1 are additional outputs, not published table values.
At default scale 1, sizes are pixels. No physical calibration is implied. Report
per-domain results alongside pooled values; differing camera scales restrict pooled
interpretation. Size distributions count individual instances, not union box area.

Paper tiling trains on 1024-pixel tiles, overlap 0/128/256, then shifts predictions
to full-frame coordinates and applies class-aware NMS at 0.50. Demo center ownership
and fixed-checkpoint inference-only experiments are separate. Runtime requires the
actual image count, hardware, warmup and timing boundary recorded with the run.

The [paper tables](../results/README.md) retain the article's individual-run and
three-seed reporting. Table VI concerns YOLO tiling; Table VII concerns granulometry.
