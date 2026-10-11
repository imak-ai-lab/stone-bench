# Frozen annotations and dataset configuration

`annotations_seed42.tar.gz` is included in Git: 7,050,583 bytes (7.05 MB).
No separate export download or Cloud.ru access is required.

SHA-256: `830e2cc03c04a1360a7c592151a8aa0d4c474ead9f8262eddecb8c04599ecd11`.

Contents: 4,196 OBB label files, configurations and source image lists for the
11 active domains (10 evaluation domains plus train-only Synthetic), the unchanged
split manifest, a file-integrity inventory and the split validation report.
The excluded historical `rocks_660tj` source has no train/test keys and is omitted.
The frozen split itself still records that exclusion.

This minimal subset comes from the verified October 12 data export (source SHA-256
`fc6642b94ad555de85d79428556aba2df23363d7a2abae4bc305d8037684df15`).
Image bytes, duplicate code, notebook backups, cloud paths and old prepared-output
metadata are omitted. Images remain available through the article's dataset sources.

Run `python scripts/prepare_reproduction_inputs.py` from the repository root to
verify the bundle and 518 test keys / 60,287 objects. See
[restoration and evaluation instructions](../../docs/data_export.md).
