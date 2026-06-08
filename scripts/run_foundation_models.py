"""Plan or launch multiple local foundation zero-shot runs.

By default this script covers the main zero-shot baselines from the paper and
excludes tiled OWLv2+SAM, because that row is an ablation/finding rather than a
main foundation baseline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "zeroshot"
DEFAULT_MODELS = [
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
    parser = argparse.ArgumentParser(description="Plan or launch local foundation model runs.")
    parser.add_argument("--models", nargs="*", default=None, help="Config stems to run. Defaults to paper baselines.")
    parser.add_argument("--include-ablations", action="store_true", help="Also include tiled OWLv2+SAM ablation.")
    parser.add_argument("--config-dir", default=str(CONFIG_DIR))
    parser.add_argument("--dataset-root", default="data/prepared/stonebench/yolo_obb")
    parser.add_argument("--output-root", default="results/predictions/foundation")
    parser.add_argument("--run-prefix", default="", help="Optional prefix added to child run names.")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--child-dry-run", action="store_true", help="Pass --dry-run to each run_zeroshot.py call.")
    parser.add_argument("--execute", action="store_true", help="Actually run child commands. Default only writes a plan.")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configs = select_configs(args)
    commands = [build_command(args, config_path) for config_path in configs]
    plan = {
        "execute": args.execute,
        "child_dry_run": args.child_dry_run,
        "configs": [str(path) for path in configs],
        "commands": [command_to_string(command) for command in commands],
    }
    plan_path = resolve_path(Path(args.output_root)) / "foundation_run_plan.json"
    save_json(plan_path, plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))

    if not args.execute:
        print("Plan only. Re-run with --execute to launch these commands.")
        return 0

    failures = []
    for command in commands:
        result = subprocess.run(command, cwd=str(ROOT), check=False)
        if result.returncode != 0:
            failures.append({"command": command_to_string(command), "returncode": result.returncode})
            if not args.continue_on_error:
                break
    if failures:
        save_json(plan_path.parent / "foundation_run_failures.json", failures)
        return 1
    return 0


def select_configs(args: argparse.Namespace) -> list[Path]:
    config_dir = resolve_path(Path(args.config_dir))
    names = list(args.models) if args.models else list(DEFAULT_MODELS)
    if args.include_ablations:
        names.extend(ABLATION_MODELS)
    if names == ["all"]:
        names = [path.stem for path in sorted(config_dir.glob("*.yaml"))]
        if not args.include_ablations:
            names = [name for name in names if name not in ABLATION_MODELS]

    configs = []
    for name in names:
        path = config_dir / ("%s.yaml" % name)
        if not path.exists():
            raise SystemExit("Foundation config not found: %s" % path)
        configs.append(path)
    return configs


def build_command(args: argparse.Namespace, config_path: Path) -> list[str]:
    output_root = Path(args.output_root) / config_path.stem
    command = [
        sys.executable,
        "scripts/run_zeroshot.py",
        "--config",
        str(config_path),
        "--dataset-root",
        args.dataset_root,
        "--output-root",
        str(output_root),
    ]
    run_name = "%s%s" % ((args.run_prefix + "_") if args.run_prefix else "", config_path.stem)
    command.extend(["--run-name", run_name])
    if args.datasets:
        command.append("--datasets")
        command.extend(args.datasets)
    if args.device:
        command.extend(["--device", args.device])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.skip_existing:
        command.append("--skip-existing")
    if args.child_dry_run:
        command.append("--dry-run")
    return command


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def command_to_string(command: list[str]) -> str:
    return " ".join(quote(value) for value in command)


def quote(value: str) -> str:
    if not value or any(char.isspace() for char in value):
        return '"' + value.replace('"', '\\"') + '"'
    return value


if __name__ == "__main__":
    raise SystemExit(main())
