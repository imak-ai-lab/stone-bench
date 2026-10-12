"""Fetch one fixed Transport image from the exact Mendeley version cited in the paper."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARTICLE_URL = "https://data.mendeley.com/datasets/v9grvgsrh9/1"
DOWNLOAD_URL = "https://data.mendeley.com/public-files/datasets/v9grvgsrh9/files/6bf7d2eb-4400-429d-aafc-0d3fb879fef1/file_downloaded"
ARCHIVE_BYTES = 72335984
ARCHIVE_SHA256 = "3dbcc3f41bec1e551a5f8c9949aa5c2a37451972e5af48ae5823671c504b5d95"
MEMBER = "images/default/10_11_56_13_16-03-2020_1.jpg"
IMAGE_SHA256 = "66415c395edd0d67435b6821e24d2ba8a2c4670fcc618fb9249619a49684297f"


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def extract_sample(archive, output):
    if archive.stat().st_size != ARCHIVE_BYTES or digest(archive) != ARCHIVE_SHA256:
        raise ValueError("Transport v1 archive size/SHA-256 mismatch; no image extracted")
    with zipfile.ZipFile(archive) as bundle:
        image = bundle.read(MEMBER)
    if hashlib.sha256(image).hexdigest() != IMAGE_SHA256:
        raise ValueError("Sample image SHA-256 mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and digest(output) != IMAGE_SHA256:
        raise ValueError("Refusing to overwrite a different image: %s" % output)
    output.write_bytes(image)
    output.with_suffix(".source.json").write_text(json.dumps(dict(
        article_url=ARTICLE_URL, download_url=DOWNLOAD_URL,
        archive_sha256=ARCHIVE_SHA256, member=MEMBER, image_sha256=IMAGE_SHA256,
        purpose="Inference example only; Transport v1 has no instance annotations.",
    ), indent=2), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an already downloaded transport.zip (v1)")
    parser.add_argument("--output", type=Path, default=ROOT / "data/samples/transport.jpg")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.archive:
        extract_sample(args.archive.resolve(), output)
    else:
        print("Downloading Transport v1 (72.3 MB); extracting one unmodified image.", flush=True)
        with tempfile.TemporaryDirectory(prefix="stonebench-sample-") as folder:
            archive = Path(folder) / "transport.zip"
            request = urllib.request.Request(DOWNLOAD_URL, headers={"User-Agent": "StoneBench-demo/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as stream:
                count = 0
                while chunk := response.read(1024 * 1024):
                    count += len(chunk)
                    if count > ARCHIVE_BYTES:
                        raise ValueError("Download exceeds the pinned archive size")
                    stream.write(chunk)
            extract_sample(archive, output)
    print("Sample (SHA-256 verified): %s" % output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
