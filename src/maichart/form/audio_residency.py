"""Budgeted preload of immutable audio features for single-process training."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

from maichart.generation.event_audio import ACOUSTIC_DIM


def physical_memory():
    """Return total/available physical bytes without a new dependency."""
    if os.name == 'nt':
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                        *[(name, ctypes.c_ulonglong) for name in
                          ('total_phys', 'available_phys', 'total_page', 'available_page',
                           'total_virtual', 'available_virtual', 'available_extended')]]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_phys), int(status.available_phys)
    elif Path('/proc/meminfo').exists():
        fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        return int(fields['MemTotal'].split()[0])*1024, int(fields['MemAvailable'].split()[0])*1024
    return None


def preload_audio(store, paths, *, report):
    """Preflight RAM and hard budget, then retain each unique song's features.

    No permanent audio copies are placed on FormSong; _LazyAudio keeps using
    the one process-local store. Random ordering and epoch masks are unchanged.
    """
    paths = list(dict.fromkeys(str(Path(path).resolve()) for path in paths))
    budget = store.budget
    if budget is None or budget <= 0:
        raise ValueError('Full audio residency requires a positive audio-cache-gb budget')
    required = 0
    for path in paths:
        index = store.disk_cache_path(path)
        if index is None or not index.exists():
            raise ValueError(f'Full audio residency requires prepared disk features: {path}')
        if index.suffix=='.npz':
            import numpy as np
            with np.load(index,allow_pickle=False) as arrays:
                required+=arrays['features'].nbytes+arrays['times'].nbytes
        else:
            metadata = json.loads(index.read_text(encoding='utf-8'))
            frames = len(metadata['times'])
            width = int(metadata.get('metadata', {}).get('acoustic_dim', ACOUSTIC_DIM))
            required += frames * (width + 1) * 4
    report(phase='estimate', completed=0, total=len(paths), unique_audio=len(paths),
           required_bytes=required, budget_bytes=budget, bytes=store.bytes)
    if required > budget:
        raise ValueError(f'Full audio residency needs {required / 1024**3:.2f} GiB of feature arrays; '
                         f'audio-cache-gb is {budget / 1024**3:.2f}. Raise the budget or choose lru.')
    memory = physical_memory()
    if memory is not None:
        total, available = memory
        reserve = max(2*1024**3, int(total*.1))
        additional = max(0, required - store.bytes)
        if additional + reserve > available:
            raise ValueError(f'Insufficient available system RAM for full audio residency: '
                             f'{available / 1024**3:.2f} GiB available, '
                             f'{additional / 1024**3:.2f} GiB additional arrays plus '
                             f'{reserve / 1024**3:.2f} GiB reserve required. Choose lru or free RAM.')
    before = store.loads
    resident_keys = set(paths)
    # Remove any speculative preparation entries outside the accepted dataset.
    for key in list(store.entries):
        if key not in resident_keys:
            old = store.entries.pop(key)
            store.bytes -= old.features.nbytes + old.times.nbytes
    for position, path in enumerate(paths, 1):
        store.get(path)
        if position == 1 or position % 50 == 0 or position == len(paths):
            report(phase='loading', completed=position, total=len(paths), bytes=store.bytes,
                   required_bytes=required, budget_bytes=budget, unique_audio=len(paths))
    if len(store.entries) != len(paths) or store.bytes > budget:
        raise ValueError('Audio residency could not retain all features within its budget; choose lru')
    report(phase='ready', completed=len(paths), total=len(paths), bytes=store.bytes,
           required_bytes=required, budget_bytes=budget, unique_audio=len(paths),
           preload_loads=store.loads-before)
    return {'mode':'resident', 'unique_audio':len(paths), 'required_bytes':required,
            'bytes':store.bytes, 'budget_bytes':budget}
