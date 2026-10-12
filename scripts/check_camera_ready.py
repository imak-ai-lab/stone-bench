"""Check local release inputs without loading weights or downloading anything."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check_split(split):
    issues = []
    payload = dict(split)
    expected = payload.pop('manifest_sha256', None)
    actual = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                      separators=(',', ':')).encode('utf-8')).hexdigest()
    if actual != expected:
        issues.append('Split manifest checksum mismatch')
    total = 0
    for slug, entry in split['datasets'].items():
        train, test = entry.get('train_keys', []), entry.get('test_keys', [])
        total += len(test)
        if set(train) & set(test):
            issues.append(f'Train/test overlap in {slug}')
        for name, keys in [('train', train), ('test', test)]:
            if len(set(keys)) != len(keys) or len(keys) != entry.get(name + '_images'):
                issues.append(f'Duplicate keys or wrong {name} count in {slug}')
            checksum = hashlib.sha256('\n'.join(sorted(keys)).encode('utf-8')).hexdigest()
            if checksum != entry.get(name + '_keys_sha256'):
                issues.append(f'Wrong {name} key checksum in {slug}')
        if slug == 'synthetic' and test:
            issues.append('Synthetic must be train-only')
    if total != 518 or split.get('test_images') != total:
        issues.append('Expected exactly 518 common test image keys')
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--scope', choices=['source', 'demo'], default='demo',
                        help='source: split and checkpoint manifests; demo: also verify local lightweight weights')
    args = parser.parse_args()
    issues = []
    split = json.loads((ROOT / 'data/manifests/unified_split_seed42.json').read_text(encoding='utf-8'))
    issues.extend(check_split(split))
    manifest_path = ROOT / 'weights/lightweight/manifest.json'
    if not manifest_path.exists():
        issues.append('Missing weights/lightweight/manifest.json and fine-tuned demo weights')
    else:
        entries = json.loads(manifest_path.read_text(encoding='utf-8')).get('checkpoints', [])
        if len(entries) != 2 or {e.get('variant') for e in entries} != {'nano', 'small'}:
            issues.append('Checkpoint manifest must contain nano and small')
        for entry in entries:
            if entry.get('evaluation_split', {}).get('split_manifest_sha256') != split.get('manifest_sha256'):
                issues.append('Checkpoint evaluation split does not match public manifest')
            path = (manifest_path.parent / entry['file']).resolve()
            if path.parent != manifest_path.parent.resolve():
                issues.append('Invalid checkpoint path')
            elif args.scope == 'source':
                continue
            elif not path.is_file() or path.stat().st_size == 0:
                issues.append(f'Missing/empty {path.name}')
            elif digest(path) != entry.get('sha256') or path.stat().st_size != entry.get('bytes'):
                issues.append(f'Checksum/size mismatch: {path.name}')
    print('Common split:', split['protocol_id'], 'test images:', split['test_images'])
    for issue in issues:
        print('NEEDS INPUT:', issue)
    print('Check scope:', args.scope)
    print('Selected scope passed' if not issues else 'Selected scope incomplete; see docs/camera_ready.md')
    return 1 if args.strict and issues else 0


if __name__ == '__main__':
    raise SystemExit(main())
