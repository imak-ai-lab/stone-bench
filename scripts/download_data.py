"""Download StoneBench source archives from article links.

The script is intentionally small and local-first:

- Mendeley pages are handled as browser/manual downloads by default.
- Roboflow Universe datasets require ``ROBOFLOW_API_KEY`` and a dataset version.
- Raw archives are written under ``data/raw/downloads`` by default.

Examples:

    python scripts/download_data.py --dry-run
    python scripts/download_data.py --provider mendeley
    python scripts/download_data.py --provider mendeley --mendeley-manual-dir "%USERPROFILE%\\Downloads"
    python scripts/download_data.py --provider roboflow --api-key <roboflow-api-key> --roboflow-version ronveer=1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MENDELEY_ZIP_URL = "https://api.data.mendeley.com/datasets/{dataset_id}/zip/file_downloaded?version={version}"
ROBOFLOW_EXPORT_URL = "https://api.roboflow.com/{workspace}/{project}/{version}/{fmt}?api_key={api_key}"


@dataclass(frozen=True)
class SourceRow:
    dataset_slug: str
    dataset_name: str
    provider: str
    article_url: str
    dataset_id: str
    version: str
    roboflow_workspace: str
    roboflow_project: str
    roboflow_version: str
    roboflow_format: str
    output_name: str
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download dataset archives from article links.")
    parser.add_argument(
        "--manifest",
        default="data/manifests/download_sources.csv",
        help="CSV with Mendeley and Roboflow source links.",
    )
    parser.add_argument(
        "--output-root",
        default="data/raw/downloads",
        help="Where downloaded ZIP archives should be stored.",
    )
    parser.add_argument(
        "--provider",
        choices=("all", "mendeley", "roboflow"),
        default="all",
        help="Limit downloads to one provider.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset slug to download. Can be passed multiple times.",
    )
    parser.add_argument(
        "--roboflow-version",
        action="append",
        default=[],
        metavar="SLUG=VERSION",
        help="Override Roboflow dataset version, e.g. ronveer=1.",
    )
    parser.add_argument(
        "--mendeley-mode",
        choices=("browser", "api"),
        default="browser",
        help="Use browser/manual Mendeley downloads by default; api keeps the old ZIP endpoint for diagnostics.",
    )
    parser.add_argument(
        "--mendeley-manual-dir",
        default="",
        help="Directory containing ZIP archives downloaded manually from Mendeley browser pages.",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open selected Mendeley article pages in the default browser.",
    )
    parser.add_argument("--api-key", default="", help="Roboflow API key. Defaults to ROBOFLOW_API_KEY.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned downloads without network calls.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing archives.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = Path(args.manifest)
    output_root = Path(args.output_root)
    selected = set(args.dataset)
    version_overrides = parse_version_overrides(args.roboflow_version)
    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY", "")

    rows = [
        row
        for row in read_manifest(manifest)
        if (args.provider == "all" or row.provider == args.provider)
        and (not selected or row.dataset_slug in selected)
    ]
    if not rows:
        raise SystemExit("No rows selected from %s" % manifest)

    for row in rows:
        destination = output_root / row.provider / row.output_name
        if row.provider == "mendeley":
            if args.mendeley_mode == "browser":
                handle_mendeley_browser_download(
                    row,
                    destination=destination,
                    manual_dir=Path(args.mendeley_manual_dir) if args.mendeley_manual_dir else None,
                    dry_run=args.dry_run,
                    overwrite=args.overwrite,
                    open_browser=args.open_browser,
                )
                continue
            url = build_mendeley_zip_url(row)
        elif row.provider == "roboflow":
            url = build_roboflow_zip_url(row, api_key=api_key, version_overrides=version_overrides)
        else:
            raise SystemExit("Unsupported provider %r for %s" % (row.provider, row.dataset_slug))

        if args.dry_run:
            print("%s\t%s\t%s" % (row.dataset_slug, row.provider, destination))
            print("  source: %s" % row.article_url)
            print("  export API: %s" % hide_api_key(url))
            if row.note:
                print("  note: %s" % row.note)
            continue

        try:
            if row.provider == "roboflow":
                download_roboflow_archive(url, destination, overwrite=args.overwrite)
            else:
                download_file(url, destination, overwrite=args.overwrite)
        except urllib.error.HTTPError as exc:
            if row.provider == "mendeley" and exc.code == 401:
                raise SystemExit(
                    "Mendeley API returned HTTP 401 for %s. Use the default browser mode instead:\n"
                    "  python scripts/download_data.py --provider mendeley "
                    "--mendeley-manual-dir <folder-with-downloaded-zips>"
                    % row.dataset_slug
                ) from exc
            if row.provider == "roboflow" and exc.code == 401:
                raise SystemExit(
                    "Roboflow rejected the API key for %s. Check that ROBOFLOW_API_KEY is current "
                    "and has access to the selected Universe dataset."
                    % row.dataset_slug
                ) from exc
            raise
    return 0


def read_manifest(path: Path) -> list[SourceRow]:
    with path.open(encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        return [SourceRow(**{key: (value or "") for key, value in row.items()}) for row in reader]


def parse_version_overrides(items: Iterable[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit("--roboflow-version must look like SLUG=VERSION, got %r" % item)
        slug, version = item.split("=", 1)
        overrides[slug.strip()] = version.strip()
    return overrides


def build_mendeley_zip_url(row: SourceRow) -> str:
    dataset_id = row.dataset_id or parse_mendeley_dataset_id(row.article_url)
    version = row.version or parse_mendeley_version(row.article_url)
    if not dataset_id or not version:
        raise SystemExit("Cannot resolve Mendeley id/version for %s" % row.dataset_slug)
    return MENDELEY_ZIP_URL.format(dataset_id=dataset_id, version=version)


def handle_mendeley_browser_download(
    row: SourceRow,
    destination: Path,
    manual_dir: Path | None,
    dry_run: bool,
    overwrite: bool,
    open_browser: bool,
) -> None:
    print("%s\tmendeley\t%s" % (row.dataset_slug, destination))
    print("  source: %s" % row.article_url)
    print("  expected archive name: %s" % row.output_name)
    if row.note:
        print("  note: %s" % row.note)

    if open_browser:
        webbrowser.open(row.article_url)

    manual_archive = manual_dir / row.output_name if manual_dir else None
    if manual_archive and manual_archive.exists():
        copy_manual_archive(manual_archive, destination, overwrite=overwrite)
        return

    if dry_run:
        if manual_archive:
            print("  manual file not found yet: %s" % manual_archive)
        return

    message = [
        "Mendeley downloads are browser/manual for %s." % row.dataset_slug,
        "Open the source page, use the Mendeley download button, and save or rename the ZIP as:",
        "  %s" % (manual_archive or destination),
    ]
    if not manual_dir:
        message.append("Then rerun with --mendeley-manual-dir <folder-with-downloaded-zips>.")
    raise SystemExit("\n".join(message))


def copy_manual_archive(source: Path, destination: Path, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        print("exists: %s" % destination)
        return
    if not zipfile.is_zipfile(source):
        raise SystemExit("Manual Mendeley file is not a readable ZIP archive: %s" % source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print("copied: %s -> %s" % (source, destination))


def build_roboflow_zip_url(row: SourceRow, api_key: str, version_overrides: dict[str, str]) -> str:
    if not api_key:
        raise SystemExit(
            "Roboflow download for %s requires --api-key or ROBOFLOW_API_KEY. "
            "The article link can still be opened manually: %s" % (row.dataset_slug, row.article_url)
        )
    workspace = row.roboflow_workspace or parse_roboflow_workspace(row.article_url)
    project = row.roboflow_project or parse_roboflow_project(row.article_url)
    version = version_overrides.get(row.dataset_slug) or row.roboflow_version or parse_roboflow_version(row.article_url)
    fmt = row.roboflow_format or "yolov8"
    if not workspace or not project:
        raise SystemExit("Cannot resolve Roboflow workspace/project for %s" % row.dataset_slug)
    if not version:
        raise SystemExit(
            "Roboflow URL for %s has no dataset version. Pass --roboflow-version %s=N after checking %s"
            % (row.dataset_slug, row.dataset_slug, row.article_url)
        )
    return ROBOFLOW_EXPORT_URL.format(
        workspace=urllib.parse.quote(workspace, safe=""),
        project=urllib.parse.quote(project, safe=""),
        version=urllib.parse.quote(str(version), safe=""),
        fmt=urllib.parse.quote(fmt, safe=""),
        api_key=urllib.parse.quote(api_key, safe=""),
    )


def download_roboflow_archive(export_url: str, destination: Path, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        print("exists: %s" % destination)
        return
    export_link = resolve_roboflow_export_link(export_url)
    download_file(export_link, destination, overwrite=overwrite)


def resolve_roboflow_export_link(export_url: str) -> str:
    request = urllib.request.Request(export_url, headers={"User-Agent": "stone-bench-downloader/0.1"})
    print("resolving export: %s" % hide_api_key(export_url))
    with urllib.request.urlopen(request) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    export = data.get("export") if isinstance(data, dict) else None
    link = export.get("link") if isinstance(export, dict) else ""
    if not link:
        raise SystemExit("Roboflow export response did not contain export.link")
    return link


def download_file(url: str, destination: Path, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        print("exists: %s" % destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "stone-bench-downloader/0.1"})
    print("downloading: %s" % hide_api_key(url))
    with urllib.request.urlopen(request) as response, temporary.open("wb") as file_obj:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            file_obj.write(chunk)
    if not zipfile.is_zipfile(temporary):
        temporary.unlink(missing_ok=True)
        raise SystemExit("Downloaded file is not a readable ZIP archive: %s" % destination)
    temporary.replace(destination)
    print("wrote: %s" % destination)


def parse_mendeley_dataset_id(url: str) -> str:
    parts = [part for part in urllib.parse.urlparse(url).path.split("/") if part]
    return parts[1] if len(parts) >= 2 and parts[0] == "datasets" else ""


def parse_mendeley_version(url: str) -> str:
    parts = [part for part in urllib.parse.urlparse(url).path.split("/") if part]
    return parts[2] if len(parts) >= 3 and parts[0] == "datasets" else ""


def parse_roboflow_workspace(url: str) -> str:
    parts = [part for part in urllib.parse.urlparse(url).path.split("/") if part]
    return parts[0] if len(parts) >= 2 else ""


def parse_roboflow_project(url: str) -> str:
    parts = [part for part in urllib.parse.urlparse(url).path.split("/") if part]
    return parts[1] if len(parts) >= 2 else ""


def parse_roboflow_version(url: str) -> str:
    parts = [part for part in urllib.parse.urlparse(url).path.split("/") if part]
    for index, part in enumerate(parts):
        if part == "dataset" and index + 1 < len(parts):
            return parts[index + 1]
    return ""


def hide_api_key(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    safe_query = [
        (key, "***" if key in {"api_key", "key"} and value else value)
        for key, value in query
    ]
    return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(safe_query)))


if __name__ == "__main__":
    raise SystemExit(main())
