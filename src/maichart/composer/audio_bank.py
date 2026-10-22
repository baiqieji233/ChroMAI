"""Immutable lossless mapped Composer nodes, avoiding repeated NPZ decoding."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

# Bump when composer_audio_nodes changes, including its timing features.
SCHEMA = 'composer_audio_nodes_v1'
BANK_DTYPE = np.dtype([('values', '<f4', (120,)), ('times', '<f8')])


def bank_path(cache_dir, feature_key, source_cache):
    source_cache=Path(source_cache)
    stat=source_cache.stat()
    identity=(SCHEMA,feature_key,str(source_cache.resolve()),
              stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
    digest=hashlib.sha256(json.dumps(identity,ensure_ascii=True).encode()).hexdigest()
    return Path(cache_dir)/'composer_audio_banks'/SCHEMA/(digest+'.npy')


def load_bank(path):
    try:
        bank=np.load(path,mmap_mode='r',allow_pickle=False)
        if bank.ndim!=1 or bank.dtype!=BANK_DTYPE:
            return None
        return bank['values'],bank['times']
    except (OSError,ValueError,EOFError):
        return None


def publish_bank(path,arrays):
    """Atomic publication; a concurrent Windows reader may retain the winner."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    values,times=arrays
    bank=np.empty(len(times),dtype=BANK_DTYPE)
    bank['values']=values;bank['times']=times
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name)
            np.save(stream,bank,allow_pickle=False)
        try:
            temporary.replace(path)
        except PermissionError:
            if load_bank(path) is None:
                raise
        mapped=load_bank(path)
        if mapped is None:
            raise ValueError('Published Composer audio bank is invalid')
        return mapped
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
