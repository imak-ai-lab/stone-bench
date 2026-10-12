"""Verify bundled annotations and optionally prepare frozen YOLO OBB train/val splits."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import tarfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA256 = 'fc6642b94ad555de85d79428556aba2df23363d7a2abae4bc305d8037684df15'
BUNDLED_SHA256 = '830e2cc03c04a1360a7c592151a8aa0d4c474ead9f8262eddecb8c04599ecd11'
BUNDLED_ARCHIVE = ROOT / 'data/reproduction/annotations_seed42.tar.gz'
SPLIT_SHA256 = 'bc3b1676f43b3af59d76552110e293fcf8cb34ffdc0c80f7195246b98a6ea26b'
DATA = 'data/zeroshot_foundation/'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def label_key(key):
    path = PurePosixPath(key)
    if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != 'images':
        raise ValueError('Invalid image key: %s' % key)
    return str(PurePosixPath('labels', *path.parts[1:]).with_suffix('.txt'))


def read_export(archive):
    archive_hash = digest(archive)
    if archive_hash not in {ARCHIVE_SHA256, BUNDLED_SHA256}:
        raise ValueError('Export archive SHA-256 mismatch')
    files = {}
    with tarfile.open(archive, 'r:gz') as bundle:
        for member in bundle:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError('Unsafe archive member: %s' % member.name)
            if member.isfile():
                name = str(PurePosixPath(*path.parts[1:]))
                if name in files:
                    raise ValueError('Duplicate archive member: %s' % name)
                files[name] = bundle.extractfile(member).read()
    inventory = json.loads(files['export_inventory.json'])
    for entry in inventory['copied_files']:
        data = files[entry['path']]
        if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('Inventory mismatch: %s' % entry['path'])
    raw_split = files['data/manifests/unified_split_seed42.json']
    if hashlib.sha256(raw_split).hexdigest() != SPLIT_SHA256:
        raise ValueError('Unexpected split file')
    split = json.loads(raw_split)
    local_split = json.loads((ROOT / 'data/manifests/unified_split_seed42.json').read_text(encoding='utf-8'))
    if split != local_split:
        raise ValueError('Export and repository split manifests differ')
    counts = {}
    for slug, entry in split['datasets'].items():
        keys = entry['test_keys']
        if len(set(keys)) != len(keys) or set(keys) & set(entry['train_keys']):
            raise ValueError('Duplicate/overlapping test keys: %s' % slug)
        if slug == 'synthetic' and keys:
            raise ValueError('Synthetic must be train-only')
        objects = 0
        for key in keys:
            for line in files[DATA + slug + '/' + label_key(key)].decode('utf-8').splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 9 or parts[0] != '0':
                    raise ValueError('Expected single-class eight-coordinate OBB: %s/%s' % (slug, key))
                coords = [float(value) for value in parts[1:]]
                if any(not math.isfinite(value) or not 0 <= value <= 1 for value in coords):
                    raise ValueError('Invalid normalized OBB: %s/%s' % (slug, key))
                objects += 1
        if keys:
            counts[slug] = dict(images=len(keys), gt_objects=objects)
    report = dict(archive_sha256=archive_hash, split_file_sha256=SPLIT_SHA256,
                  split_manifest_sha256=split['manifest_sha256'],
                  verified_inventory_files=len(inventory['copied_files']),
                  dataset_configs=len(inventory['dataset_configs']), by_dataset=counts,
                  images=sum(v['images'] for v in counts.values()),
                  gt_objects=sum(v['gt_objects'] for v in counts.values()),
                  scope='Saved OBB labels and split keys; images and session independence are not certified.')
    saved = json.loads(files['split_validation.json'])
    if report['images'] != 518 or report['gt_objects'] != 60287 or saved['by_dataset'] != counts:
        raise ValueError('Test population differs from 518 images / 60287 GT or exported report')
    report['passed'] = True
    return files, split, report


def materialize(files, split, images_root, output, datasets):
    """Copy only known metadata/labels and matched images; never execute archive code."""
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty --output-root; existing data are preserved')
    selected = datasets or available_datasets(files, split)
    plan = []
    problems = []
    for slug in selected:
        if slug not in split['datasets']:
            raise ValueError('Unknown dataset: %s' % slug)
        if DATA + slug + '/dataset_config.yaml' not in files:
            raise ValueError('Dataset excluded from the bundled frozen-split inputs: %s' % slug)
        source = images_root / slug
        candidates = defaultdict(list)
        for path in source.rglob('*'):
            if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}:
                candidates[path.name].append(path)
        entry = split['datasets'][slug]
        keys = entry['train_keys'] + entry['test_keys']
        for key in keys:
            if not key.strip():
                continue
            label = label_key(key)
            exact = source / key
            matches = [exact] if exact.is_file() else candidates[PurePosixPath(key).name]
            if len(matches) != 1:
                problems.append('%s/%s: %s matches' % (slug, key, len(matches)))
            else:
                plan.append((slug, key, label, matches[0]))
    if problems:
        raise ValueError('Missing/ambiguous images; no output written:\n' + '\n'.join(problems[:20])
                         + '\nTotal problems: %s' % len(problems))
    image_manifest = []
    for slug, key, label, source in plan:
        target = output / slug / key
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        label_path = output / slug / label
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_bytes(files[DATA + slug + '/' + label])
        image_manifest.append(dict(dataset=slug, key=key, sha256=digest(target)))
    for slug in selected:
        directory = output / slug
        directory.mkdir(parents=True, exist_ok=True)
        entry = split['datasets'][slug]
        for name, keys in [('train', entry['train_keys']), ('val', entry['test_keys'])]:
            (directory / (name + '.txt')).write_text(''.join('./' + key + '\n' for key in keys), encoding='utf-8')
        config = dict(path=directory.as_posix(), train='train.txt', val='val.txt', test='val.txt', names=['stone'])
        (directory / 'data.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    (output / 'image_manifest.json').write_text(json.dumps(image_manifest, indent=2), encoding='utf-8')
    return len(plan)


def available_datasets(files, split):
    return sorted(slug for slug in split['datasets'] if DATA + slug + '/dataset_config.yaml' in files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=BUNDLED_ARCHIVE,
                        help='Defaults to the minimal annotation/config bundle included in Git')
    parser.add_argument('--images-root', type=Path, help='Extracted image folders: ROOT/<dataset_slug>/**/<original_filename>')
    parser.add_argument('--output-root', type=Path, help='New directory for frozen per-domain YOLO OBB datasets (outside Git)')
    parser.add_argument('--datasets', nargs='+', help='Optional subset, e.g. transport')
    parser.add_argument('--report', type=Path, default=ROOT / 'outputs/data_export_validation.json')
    args = parser.parse_args()
    if bool(args.images_root) != bool(args.output_root):
        parser.error('--images-root and --output-root must be supplied together')
    files, split, report = read_export(args.archive.resolve())
    if args.images_root:
        report['materialized_images'] = materialize(files, split, args.images_root.resolve(),
                                                     args.output_root.resolve(), args.datasets)
        report['materialized_datasets'] = args.datasets or available_datasets(files, split)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
