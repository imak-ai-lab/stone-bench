# Data Directory

Raw and prepared data are intentionally kept out of git. This directory stores
manifests, source links, split policy, checksums, and optional tiny samples.

Expected local layout:

```text
data/
  manifests/
  raw/            # ignored
  prepared/       # ignored
  samples/        # ignored except .gitkeep
```

Source archives are downloaded into `data/raw/downloads/`. Prepared datasets are
written under `data/prepared/stonebench/`.
