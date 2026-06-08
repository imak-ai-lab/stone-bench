# StoneBench Dataset Card

StoneBench contains 12 rock-fragment datasets across five lifecycle stages:
blast pile, loading, truck/wagon haulage, conveyor, and synthetic imagery. The
paper table reports 4,523 images and 393,965 annotated fragments.

## Lifecycle Stages

| Stage | Datasets |
|---|---|
| Blast pile | Open pits, Rocks asbest, Rocks UAV, Rock blasting, Mask 2018 |
| Loading | Bucket excavation |
| Haulage | 40-70 mm, Unloading a wagon |
| Conveyor | Transport, Ronveer |
| Synthetic | SAM fragment v7, Synthetic |

## Annotation Sources

Annotations include manual masks, automatic SAM-derived masks, and hybrid
automatic/manual labels. The `annotation` column in
`data/manifests/datasets.csv` records the annotation source for each dataset.

## Public And Own Data

The `own` column in `data/manifests/datasets.csv` marks datasets introduced with
the benchmark. Public sources are linked through Mendeley or Roboflow in
`data/manifests/download_sources.csv`.

## Release Layout

Raw archives are not committed to git. Local downloads are stored under
`data/raw/downloads/`, and prepared benchmark formats are generated under
`data/prepared/stonebench/`.

## Limitations

Some domains contain automatic labels, variable camera geometry, and dense
fragment occlusion. Physical size metrics depend on the configured pixel-to-mm
scale; by default the scripts operate in pixel units unless a scale CSV or
`--mm-per-px` value is supplied.
