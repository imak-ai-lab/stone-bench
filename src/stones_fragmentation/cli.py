import argparse
import json

from stones_fragmentation.data import prepare_data
from stones_fragmentation.infer import run_inference
from stones_fragmentation.io import load_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Local stone-fragmentation skeleton.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--limit", type=int, help="Optional smoke-test limit per source split")

    infer = subparsers.add_parser("infer")
    infer.add_argument("--config", required=True)

    args = parser.parse_args()
    config = load_yaml(args.config)

    if args.command == "prepare-data":
        result = prepare_data(config, limit=args.limit)
    else:
        result = run_inference(config)

    print(json.dumps(result, indent=2))
    return 0
