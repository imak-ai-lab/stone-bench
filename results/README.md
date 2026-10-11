# Paper results

`tables/` contains values from Tables III–VII of the submitted article.
Tiling SD is reported separately; W1 and D80 use pixels.
Individual-run results and aggregates over training seeds 42/43/44 retain their
context from the article; the evaluation split is fixed. Table VI concerns
YOLO11x-OBB tiling, while Table VII reports granulometry. Nano/small demo outputs
are separate from these large-model results.
Run `python scripts/make_tables.py --output results/tables` to validate CSV structure.
Generated metrics and prediction outputs are written to ignored local directories.
