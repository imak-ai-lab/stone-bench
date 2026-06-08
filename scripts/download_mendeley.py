"""Handle one Mendeley Data archive from an article URL."""

from __future__ import annotations

import argparse
from pathlib import Path

from download_data import build_mendeley_zip_url, download_file, handle_mendeley_browser_download, SourceRow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download one Mendeley dataset ZIP.")
    parser.add_argument("--url", required=True, help="Article Mendeley URL, e.g. https://data.mendeley.com/datasets/.../1")
    parser.add_argument("--output", required=True, help="Destination ZIP path.")
    parser.add_argument(
        "--mode",
        choices=("browser", "api"),
        default="browser",
        help="Use browser/manual handling by default; api keeps the old ZIP endpoint for diagnostics.",
    )
    parser.add_argument("--manual-dir", default="", help="Directory with the ZIP downloaded from the browser.")
    parser.add_argument("--open-browser", action="store_true", help="Open the Mendeley article page.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    row = SourceRow(
        dataset_slug=Path(args.output).stem,
        dataset_name=Path(args.output).stem,
        provider="mendeley",
        article_url=args.url,
        dataset_id="",
        version="",
        roboflow_workspace="",
        roboflow_project="",
        roboflow_version="",
        roboflow_format="",
        output_name=Path(args.output).name,
        note="",
    )
    if args.mode == "browser":
        handle_mendeley_browser_download(
            row,
            destination=Path(args.output),
            manual_dir=Path(args.manual_dir) if args.manual_dir else None,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            open_browser=args.open_browser,
        )
    else:
        url = build_mendeley_zip_url(row)
        if args.dry_run:
            print(url)
            return 0
        download_file(url, Path(args.output), overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
