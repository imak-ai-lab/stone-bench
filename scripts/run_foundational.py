"""Convenience entrypoint for the StoneBench foundational-model block.

This is a small wrapper around ``run_foundation_models.py`` with paper-oriented
presets:

- ``--preset main``: main zero-shot foundation baselines from the paper;
- ``--preset ablations``: ablation-only runs such as tiled OWLv2+SAM;
- ``--preset all``: main baselines plus ablations;
- ``--model NAME``: one explicit config stem from ``configs/zeroshot``.

The default mode writes a reproducible plan only. Add ``--execute`` to launch
child runs. Add ``--real-inference`` to omit child dry runs and use configured
local execution paths.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_MODELS = [
    "sam21_l_auto",
    "owlv2_large",
    "yoloe_26x_seg",
    "yolo_world_x",
    "groundingdino_b_sam21_l",
    "qwen25_vl_7b",
    "groundingdino_b",
    "detic_swinb",
]
ABLATION_MODELS = ["tiled_owlv2_large_sam21_l"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or launch the foundational-model benchmark block.")
    parser.add_argument("--preset", choices=("main", "ablations", "all"), default="main")
    parser.add_argument("--model", action="append", default=[], help="Explicit config stem. Can be repeated.")
    parser.add_argument("--dataset-root", default="data/prepared/stonebench/yolo_obb")
    parser.add_argument("--output-root", default="results/predictions/foundation")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--device", default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-prefix", default="")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Launch child commands. Default only writes a plan.")
    parser.add_argument(
        "--real-inference",
        action="store_true",
        help="Do not pass --dry-run to children.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = build_command(args)
    print(" ".join(quote(part) for part in command), flush=True)
    return subprocess.run(command, cwd=str(ROOT), check=False).returncode


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_foundation_models.py",
        "--output-root",
        args.output_root,
        "--dataset-root",
        args.dataset_root,
    ]

    models = selected_models(args)
    if models:
        command.append("--models")
        command.extend(models)
    if args.preset in ("all", "ablations") and not args.model:
        command.append("--include-ablations")
    if args.datasets:
        command.append("--datasets")
        command.extend(args.datasets)
    if args.device:
        command.extend(["--device", args.device])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.run_prefix:
        command.extend(["--run-prefix", args.run_prefix])
    if args.skip_existing:
        command.append("--skip-existing")
    if not args.real_inference:
        command.append("--child-dry-run")
    if args.execute:
        command.append("--execute")
    if args.continue_on_error:
        command.append("--continue-on-error")
    return command


def selected_models(args: argparse.Namespace) -> list[str]:
    if args.model:
        return list(args.model)
    if args.preset == "ablations":
        return list(ABLATION_MODELS)
    if args.preset == "all":
        return list(MAIN_MODELS)
    return list(MAIN_MODELS)


def quote(value: str) -> str:
    if not value or any(char.isspace() for char in value):
        return '"' + value.replace('"', '\\"') + '"'
    return value


if __name__ == "__main__":
    raise SystemExit(main())
