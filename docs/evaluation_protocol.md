# Evaluation Protocol

## Detection Metrics

Reported detection metrics use:

- unified confidence post-filter at 0.25;
- greedy per-image matching in descending confidence order;
- best unmatched ground-truth instance by polygon IoU;
- precision, recall, and F1 at IoU 0.50 and 0.75;
- class-aware matching for the `stone` class.

The implementation is in `src/stones_fragmentation/metrics/detection.py` and
the CLI entrypoint is `scripts/evaluate_detection.py`.

## Size-Distribution Metrics

Operational size-distribution metrics are computed from the same prediction
geometry:

- CDF RMSE;
- Kolmogorov-Smirnov distance;
- Wasserstein-1 distance;
- absolute D80 error;
- missed-small fraction below the ground-truth P10 threshold.

The implementation is in `src/stones_fragmentation/metrics/granulometry.py` and
the CLI entrypoint is `scripts/evaluate_granulometry.py`.

## Tiled Evaluation

Tiled supervised evaluation uses the following policy:

- train on generated tiles;
- shift tile predictions back to full-image coordinates;
- merge stitched predictions with class-aware NMS;
- evaluate on original full frames rather than on tile crops.

The local pipeline entrypoint is `scripts/run_tiled_inference.py`.
