"""Verify and run the nano/small stone baselines on a supplied image or directory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--variant', choices=['nano', 'small', 'both'], default='both')
    parser.add_argument('--mode', choices=['full_frame', 'overlap0', 'overlap128', 'overlap256', 'all'], default='all')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--conf', type=float, default=0.25, help='Prediction confidence cutoff (use 0.001 for AP export)')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/demo')
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.exists():
        parser.error('Source image/directory does not exist: %s' % source)
    manifest = json.loads((ROOT / 'weights/lightweight/manifest.json').read_text(encoding='utf-8'))
    variants = ['nano', 'small'] if args.variant == 'both' else [args.variant]
    modes = ['full_frame', 'overlap0', 'overlap128', 'overlap256'] if args.mode == 'all' else [args.mode]
    for variant in variants:
        for mode in modes:
            destination = args.output.resolve() / variant / mode
            if destination.exists() and any(destination.iterdir()):
                parser.error('Output already contains a run; choose a new --output: %s' % destination)
    checkpoints = []
    for variant in variants:
        entry = next(e for e in manifest['checkpoints'] if e['variant'] == variant)
        weight = ROOT / 'weights/lightweight' / entry['file']
        if not weight.is_file() or weight.stat().st_size != entry['bytes']:
            parser.error('Missing or wrong-size checkpoint: %s' % weight)
        if hashlib.sha256(weight.read_bytes()).hexdigest() != entry['sha256']:
            parser.error('Checkpoint SHA-256 mismatch: %s' % weight)
        checkpoints.append((variant, weight))
    for variant, weight in checkpoints:
        subprocess.run([sys.executable, str(ROOT / 'scripts/demo.py'), '--weights', str(weight),
                        '--source', str(source), '--device', args.device, '--mode', args.mode,
                        '--conf', str(args.conf),
                        '--output', str(args.output.resolve() / variant)], check=True)
    print('Demo complete: %s' % args.output.resolve())


if __name__ == '__main__':
    main()
