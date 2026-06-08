# Dataset Sources

The source links from the paper are captured in `download_sources.csv`.

```bash
python scripts/download_data.py --dry-run
python scripts/download_data.py --provider mendeley --dry-run --open-browser
python scripts/download_data.py --provider mendeley --mendeley-manual-dir "%USERPROFILE%\Downloads"
python scripts/download_data.py --provider roboflow --api-key <roboflow-api-key> --roboflow-version ronveer=1
```

Mendeley sources use a browser download flow: open the source page, download the
archive, and rename it to the `output_name` from `download_sources.csv` before
copying it through `--mendeley-manual-dir`.

Roboflow exports require an API key. The Ronveer project page does not contain a
pinned dataset version in the URL, so the version is supplied with
`--roboflow-version`.

| Dataset | Source | Archive |
|---|---|---|
| Open pits | Mendeley `v9grvgsrh9/1` | `stonebench_mendeley_v1.zip` |
| Rocks asbest | Mendeley `v9grvgsrh9/1` | `stonebench_mendeley_v1.zip` |
| Rocks UAV | Mendeley `v9grvgsrh9/1` | `stonebench_mendeley_v1.zip` |
| Rock blasting | Mendeley `z78ghz96bn/1` | `rock_blasting_mendeley_v1.zip` |
| Mask 2018 | Mendeley `78ht3pjsr4/1` | `mask_2018_mendeley_v1.zip` |
| Bucket excavation | Mendeley `v9grvgsrh9/1` | `stonebench_mendeley_v1.zip` |
| 40-70 mm | Mendeley `v9grvgsrh9/1` | `stonebench_mendeley_v1.zip` |
| Unloading a wagon | Mendeley `v9grvgsrh9/1` | `stonebench_mendeley_v1.zip` |
| Transport | Mendeley `v9grvgsrh9/1` | `stonebench_mendeley_v1.zip` |
| Ronveer | Roboflow `ghv/ngb` | `ronveer_roboflow.zip` |
| SAM fragment v7 | Roboflow `sam-fragment/sam-fragment` | `sam_fragment_v7_roboflow.zip` |
| Synthetic | Roboflow `urfu-uavlq/stone_gen_new_segmentation/dataset/1` | `synthetic_roboflow_v1.zip` |
