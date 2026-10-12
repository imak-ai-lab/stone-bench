"""Equal-dataset macro-F1 from one model's detection-evaluator JSON output."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def summarize(rows):
    rows = [row for row in rows if not row['dataset'].startswith('__')]
    names = [row['dataset'] for row in rows]
    if not rows or len(set(names)) != len(names):
        raise ValueError('Require exactly one row per dataset for one model/seed')
    thresholds = set(rows[0]['iou'])
    if any(set(row['iou']) != thresholds for row in rows):
        raise ValueError('IoU thresholds differ')
    return dict(aggregation='equal_dataset_mean_of_f1', datasets=names,
                num_datasets=len(rows), macro_f1={
                    threshold: sum(row['iou'][threshold]['f1'] for row in rows) / len(rows)
                    for threshold in sorted(thresholds)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.input.read_text(encoding='utf-8')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
