"""Download one Roboflow Universe dataset archive."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from download_data import build_roboflow_zip_url, download_roboflow_archive, hide_api_key, SourceRow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download one Roboflow Universe dataset ZIP.")
    parser.add_argument("--url", required=True, help="Article Roboflow Universe URL.")
    parser.add_argument("--version", default="", help="Dataset version if it is not present in the URL.")
    parser.add_argument("--format", default="yolov8", help="Roboflow export format.")
    parser.add_argument("--output", required=True, help="Destination ZIP path.")
    parser.add_argument("--api-key", default="", help="Defaults to ROBOFLOW_API_KEY.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    slug = Path(args.output).stem
    row = SourceRow(
        dataset_slug=slug,
        dataset_name=slug,
        provider="roboflow",
        article_url=args.url,
        dataset_id="",
        version="",
        roboflow_workspace="",
        roboflow_project="",
        roboflow_version=args.version,
        roboflow_format=args.format,
        output_name=Path(args.output).name,
        note="",
    )
    url = build_roboflow_zip_url(
        row,
        api_key=args.api_key or os.environ.get("ROBOFLOW_API_KEY", ""),
        version_overrides={},
    )
    if args.dry_run:
        print(hide_api_key(url))
        return 0
    download_roboflow_archive(url, Path(args.output), overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
