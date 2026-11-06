"""Opt-in path relocation without rewriting content-verified manifests."""
from __future__ import annotations

import json
import os
from pathlib import Path


def material_path(value):
    """Map absolute source prefixes using a JSON file named by the environment.

    With no map configured this is exactly Path(value). Manifest and annotation
    bytes remain untouched, so the normal resume identity checks still apply.
    """
    mapping_file = os.environ.get('MAICHART_PATH_MAP')
    if not mapping_file:
        return Path(value)
    mapping = json.loads(Path(mapping_file).read_text(encoding='utf-8'))
    normalized = str(value).replace('\\', '/')
    for prefix, target in sorted(mapping.items(), key=lambda item: -len(item[0])):
        prefix = prefix.replace('\\', '/').rstrip('/')
        if normalized.casefold() == prefix.casefold():
            return Path(target)
        if normalized.casefold().startswith(prefix.casefold() + '/'):
            return Path(target) / normalized[len(prefix) + 1:]
    return Path(value)
