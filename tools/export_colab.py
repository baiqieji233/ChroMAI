"""Export a private Composer resume bundle; never stop or modify local training.

Run with the project's Python: python tools/export_colab.py --dry-run
Then omit --dry-run to create dist/colab/composer_colab.tar.gz.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def asset_name(path):
    path = Path(path).resolve()
    windows = PureWindowsPath(str(path))
    if not windows.drive or not windows.is_absolute():
        raise ValueError('Export must run on the original Windows machine')
    return Path('materials', windows.drive.rstrip(':'), *windows.parts[1:]).as_posix()


def collect(checkpoint):
    from maichart.composer.stream_checkpoint import read
    from maichart.composer.stream_data_identity import _annotation_path
    from maichart.training.dataset_catalog import _source_asset

    payload = read(checkpoint)
    if payload.get('checkpoint_kind') != 'full' or not payload.get('training_state'):
        raise ValueError('A complete optimizer/RNG/cursor checkpoint is required')
    args = payload['args']
    files = {}

    def add(path, name=None):
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        files[name or asset_name(path)] = path

    for key in ('manifest', 'validation_manifest'):
        if not args.get(key):
            continue
        manifest = Path(args[key]).resolve()
        add(manifest)
        data = json.loads(manifest.read_text(encoding='utf-8-sig'))
        annotation = _annotation_path(manifest, data)
        if annotation:
            add(annotation)
        for entry in data['songs']:
            for role in ('maidata_path', 'audio_path'):
                source = _source_asset(entry[role], manifest, data)
                add(source)
                if role == 'maidata_path':
                    structure = source.with_name('music_structure.json')
                    if structure.is_file():
                        add(structure)

    # Cache NPZ files are content-addressed and validated by the normal loader.
    # Exclude platform/path-dependent preparation caches, banks and wave maps.
    from maichart.form.cache_support import cache_roots
    for cache in cache_roots(str(Path(args['cache_dir']).resolve())):
        profile = cache / 'rational_audio_queries_v1'
        for path in sorted(profile.glob('*.npz')):
            name = 'cache/rational_audio_queries_v1/' + path.name
            if name not in files:
                add(path, name)

    for path in (ROOT / 'src').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and not path.name.endswith(('.pyc', '.pyo')):
            add(path, 'project/' + path.relative_to(ROOT).as_posix())
    for name in ('pyproject.toml', 'README.md', 'THIRD_PARTY_NOTICES.md', 'tools/colab_resume.py'):
        add(ROOT / name, 'project/' + name)
    metadata = dict(schema='composer_colab_bundle_v1', args=args,
                    epoch=payload['epoch'], optimizer_steps=payload['optimizer_steps'],
                    batch_in_epoch=payload['training_state'].get('batch_in_epoch'),
                    dataset_content_identity=payload['training_state'].get('dataset_content_identity'),
                    drive_letters=sorted({PureWindowsPath(str(p)).drive.rstrip(':') for p in files.values()}))
    del payload
    return files, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/prism_composer_v43/composer_last.pt')
    parser.add_argument('--output', type=Path, default=ROOT / 'dist/colab/composer_colab.tar.gz')
    parser.add_argument('--dry-run', action='store_true')
    options = parser.parse_args(argv)
    files, metadata = collect(options.checkpoint)
    total = sum(p.stat().st_size for p in files.values()) + options.checkpoint.stat().st_size
    print(json.dumps(dict(files=len(files)+1, uncompressed_gib=round(total/2**30, 3),
                         epoch=metadata['epoch'], optimizer_steps=metadata['optimizer_steps'],
                         output=str(options.output.resolve())), ensure_ascii=False), flush=True)
    if options.dry_run:
        return
    output = options.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Choose a new output path; existing exports are preserved')
    if shutil.disk_usage(output.parent).free < total + options.checkpoint.stat().st_size + 2**30:
        raise OSError('Insufficient space for archive plus checkpoint snapshot')
    snapshot = output.with_name('checkpoint_snapshot.pt')
    temporary = output.with_suffix('.partial')
    try:
        # Source checkpoints publish by atomic replacement. Copy the open file
        # before packaging, then inspect that exact snapshot for matching args.
        shutil.copyfile(options.checkpoint, snapshot)
        files, metadata = collect(snapshot)
        metadata['uncompressed_bytes'] = sum(p.stat().st_size for p in files.values()) + snapshot.stat().st_size
        meta_path = output.with_name('bundle.json')
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        files['bundle.json'] = meta_path
        files['checkpoint/composer_last.pt'] = snapshot
        with tarfile.open(temporary, 'w:gz', compresslevel=1) as archive:
            for i, (name, source) in enumerate(sorted(files.items()), 1):
                archive.add(source, arcname=name, recursive=False)
                if i % 250 == 0:
                    print(f'Packed {i}/{len(files)} files', flush=True)
        temporary.replace(output)
        digest = hashlib.sha256()
        with output.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(block)
        output.with_suffix(output.suffix + '.sha256').write_text(digest.hexdigest() + '  ' + output.name + '\n')
        print(f'Ready: {output} ({output.stat().st_size/2**30:.2f} GiB)', flush=True)
    finally:
        snapshot.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
