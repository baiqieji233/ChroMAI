"""Materialize one bounded upload part, keeping the full archive on disk.

python tools/colab_part.py dist/colab/composer_colab.tar.gz 0
Parts are 480 MiB and can be removed after verified upload. SHA256 of the
reassembled archive is checked by Composer_Colab.ipynb.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('index', type=int)
    args = parser.parse_args()
    size = 480 * 1024 * 1024
    if args.index < 0 or args.index * size >= args.archive.stat().st_size:
        raise ValueError('Part index is outside archive')
    target = args.archive.with_name(args.archive.name + f'.part{args.index:03d}')
    with args.archive.open('rb') as source, target.open('wb') as output:
        source.seek(args.index * size)
        remaining = size
        while remaining:
            data = source.read(min(8 * 1024 * 1024, remaining))
            if not data:
                break
            output.write(data)
            remaining -= len(data)
    print(json.dumps(dict(name=target.name, size=target.stat().st_size, path=str(target.resolve()))))


if __name__ == '__main__':
    main()
