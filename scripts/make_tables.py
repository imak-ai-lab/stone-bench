"""Validate committed StoneBench table snapshots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED_TABLES = (
    "zero_shot_leaderboard.csv",
    "zero_shot_per_dataset.csv",
    "supervised_per_dataset.csv",
    "tiling_ablation.csv",
    "size_distribution_metrics.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate compact StoneBench CSV table snapshots.")
    parser.add_argument("--output", default="results/tables", help="Directory containing compact CSV tables.")
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional JSON manifest path. Defaults to table_manifest.json in the table directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    table_dir = Path(args.output)
    rows = []
    for table_name in REQUIRED_TABLES:
        table_path = table_dir / table_name
        if not table_path.exists():
            raise SystemExit("Missing table: %s" % table_path)
        with table_path.open("r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            records = list(reader)
            if not reader.fieldnames:
                raise SystemExit("Table has no header: %s" % table_path)
            if not records:
                raise SystemExit("Table has no rows: %s" % table_path)
            rows.append(
                {
                    "table": table_name,
                    "columns": reader.fieldnames,
                    "rows": len(records),
                }
            )

    manifest_path = Path(args.manifest) if args.manifest else table_dir / "table_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"tables": rows}, indent=2), encoding="utf-8")
    print("validated %d tables" % len(rows))
    print("wrote: %s" % manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
