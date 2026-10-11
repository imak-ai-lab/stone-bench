# Dataset sources

These links are exactly the sources cited in the article. Dataset descriptions
are in `datasets.csv`; machine-readable source details are in `download_sources.csv`.

| Datasets | Article source |
|---|---|
| Open pits, Rocks asbest, Rocks UAV, Bucket excavation, 40-70 mm, Unloading a wagon, Transport | [Mendeley v1](https://data.mendeley.com/datasets/v9grvgsrh9/1) |
| Rock blasting | [Mendeley v1](https://data.mendeley.com/datasets/z78ghz96bn/1) |
| Mask 2018 | [Mendeley v1](https://data.mendeley.com/datasets/78ht3pjsr4/1) |
| Ronveer | [Roboflow](https://universe.roboflow.com/ghv/ngb) |
| SAM fragment v7 | [Roboflow](https://universe.roboflow.com/sam-fragment/sam-fragment) |
| Synthetic | [Roboflow v1](https://universe.roboflow.com/urfu-uavlq/stone_gen_new_segmentation/dataset/1) |

Download the required images from the source page. Roboflow exports may require
an account; record the export version (the Ronveer article URL does not pin one).
The demo automatically downloads one Transport image. To evaluate on real labels,
follow [data preparation](../../docs/data_export.md) using the bundled annotations
and frozen split.
