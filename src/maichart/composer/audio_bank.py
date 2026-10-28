"""Immutable lossless mapped Composer nodes, avoiding repeated NPZ decoding."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

# Bump when composer_audio_nodes changes, including its timing features.
SCHEMA = 'composer_audio_nodes_native_v3'
BANK_DTYPE = np.dtype([('values', '<f4', (120,)), ('times', '<f8')])


def _digest_path(cache_dir, key_tuple, resolved_source, stat):
    identity = (SCHEMA, key_tuple, resolved_source,
                stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    digest = hashlib.sha256(json.dumps(identity, ensure_ascii=True).encode()).hexdigest()
    return Path(cache_dir) / 'composer_audio_banks' / SCHEMA / (digest + '.npy')


def bank_path(cache_dir, feature_key, source_cache):
    source_cache = Path(source_cache)
    stat = source_cache.stat()
    resolved = str(source_cache.resolve())
    primary = _digest_path(cache_dir, feature_key, resolved, stat)
    if primary.is_file():
        return primary
    if isinstance(feature_key, tuple) and len(feature_key) >= 2 and isinstance(feature_key[0], str):
        content_cand = _digest_path(cache_dir, (source_cache.name, *feature_key[1:]), resolved, stat)
        if content_cand.is_file():
            return content_cand
    return primary


def load_bank(path):
    """Map a bank without scanning every value page on every cache miss.

    Banks are written atomically by :func:`publish_bank`, which validates all
    values before publication.  Repeating a full ``isfinite`` reduction here
    defeated mmap and forced the OS to read each 6--10 MB file from disk every
    time the small hot LRU evicted it.  Validate dtype/shape and a tiny edge
    sample on the load path; full validation remains in ``publish_bank``.
    """
    try:
        bank=np.load(path,mmap_mode='r',allow_pickle=False)
        if bank.ndim!=1 or bank.dtype!=BANK_DTYPE:
            return None
        if len(bank):
            times = bank['times']
            edge = np.asarray([times[0], times[-1]], dtype=np.float64)
            values = np.asarray(bank['values'][[0, -1]], dtype=np.float32)
            if not np.isfinite(edge).all() or not np.isfinite(values).all():
                return None
        return bank['values'],bank['times']
    except (OSError,ValueError,EOFError):
        return None


def publish_bank(path,arrays):
    """Atomic publication; a concurrent Windows reader may retain the winner."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    values,times=arrays
    times=np.asarray(times,dtype=np.float64)
    if len(times) and (not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError('Composer audio bank times must be finite and strictly increasing')
    if not np.isfinite(values).all():
        raise ValueError('Composer audio bank values must be finite')
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
