# Data Directory

Raw images and generated prepared datasets are kept out of Git. The compact
frozen OBB annotation bundle is tracked under `data/reproduction/`. This directory
stores dataset descriptions, article source links and the fixed split manifest.

Expected local layout:

```text
data/
  manifests/
  reproduction/   # bundled annotations and integrity metadata
  raw/            # ignored
  prepared/       # ignored
  samples/        # ignored except .gitkeep
```

The demo downloads one image to `data/samples/`. To join local source images to
the bundled labels and prepare the fixed evaluation split, follow
[the data guide](../docs/data_export.md).
