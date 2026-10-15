"""Share native-rate mono waveform pages across spawned training workers."""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from maichart.form.output_guard import training_output_guard


SCHEMA = 'native_mono_float32_v1'


def _load(directory):
    try:
        metadata = json.loads((directory / 'index.json').read_text(encoding='utf-8'))
        if metadata['schema'] != SCHEMA:
            return None
        values = np.load(directory / metadata['array'], mmap_mode='r', allow_pickle=False)
        if values.shape != (metadata['frames'],) or values.dtype != np.dtype('float32'):
            return None
        if not isinstance(metadata['sample_rate'], int) or metadata['sample_rate'] <= 0:
            return None
        return values, metadata['sample_rate']
    except (OSError, ValueError, KeyError, TypeError, EOFError):
        return None


def _promote_waveform(source_dir, target_dir):
    """Copy/link an immutable waveform cache into the active cache root."""
    try:
        metadata_path = source_dir / 'index.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        source_array = source_dir / metadata['array']
        target_dir.mkdir(parents=True, exist_ok=True)
        target_array = target_dir / metadata['array']
        if not target_array.exists():
            try:
                os.link(source_array, target_array)
            except OSError:
                shutil.copyfile(source_array, target_array)
        temporary = target_dir / '.index.tmp'
        temporary.write_text(json.dumps(metadata), encoding='utf-8')
        temporary.replace(target_dir / 'index.json')
        return _load(target_dir)
    except (OSError, ValueError, KeyError, TypeError, EOFError):
        return None


def _waveform_roots(cache_dir):
    if not cache_dir:
        return [Path(tempfile.gettempdir()) / 'maichart-waveforms-v1']
    current = Path(cache_dir).resolve()
    roots = [current / 'rational_audio_queries_v1' / 'waveforms_v1']
    dataset_root = current.parent
    if dataset_root.parent.is_dir():
        for sibling in dataset_root.parent.iterdir():
            candidate = sibling / current.name / 'rational_audio_queries_v1' / 'waveforms_v1'
            if candidate.is_dir() and candidate not in roots:
                roots.append(candidate)
    shared = Path(__file__).resolve().parents[3] / 'webui_workspace' / 'cache' / 'song_form_shared' / 'rational_audio_queries_v1' / 'waveforms_v1'
    if shared.is_dir() and shared not in roots:
        roots.append(shared)
    return roots


@contextmanager
def _build_lock(directory):
    deadline = time.monotonic() + 120.
    while True:
        guard = training_output_guard(directory)
        try:
            guard.__enter__()
            break
        except ValueError:
            if time.monotonic() >= deadline:
                raise ValueError(f'Timed out waiting for waveform cache builder: {directory}')
            time.sleep(.05)
    try:
        yield
    finally:
        guard.__exit__(None, None, None)


@lru_cache(maxsize=4)
def _mapped(path, size, mtime_ns, ctime_ns, sha256, cache_dir):
    from maichart.form.rational_audio import _ShortAudioRead, _digest, _read_mono
    if _digest(path) != sha256:
        raise ValueError('Audio content changed before waveform mapping; reload the audio source')
    roots = _waveform_roots(cache_dir)
    root = roots[0]
    directory = root / sha256
    for candidate_root in roots:
        cached = _load(candidate_root / sha256)
        if cached is not None:
            if candidate_root != root:
                promoted = _promote_waveform(candidate_root / sha256, directory)
                if promoted is not None:
                    return promoted
            return cached
    cached = None
    with _build_lock(directory):
        cached = _load(directory)
        if cached is not None:
            return cached
        # Each published array is immutable; an existing mapping cannot block
        # replacement of a repaired cache on Windows.
        with tempfile.NamedTemporaryFile(dir=directory, suffix='.npy', delete=False) as stream:
            array_path = Path(stream.name)
        temporary_index = None
        published = False
        try:
            import soundfile as sf
            try:
                with sf.SoundFile(path) as source:
                    rate, frames = int(source.samplerate), int(source.frames)
                    mapped = np.lib.format.open_memmap(array_path, mode='w+', dtype='float32', shape=(frames,))
                    block = np.empty((min(65536, max(1, frames)), source.channels), dtype=np.float32)
                    cursor = 0
                    try:
                        while cursor < frames:
                            chunk = source.read(out=block[:min(len(block), frames-cursor)])
                            if not len(chunk):
                                raise _ShortAudioRead(f'Audio ended before declared frame count: {path}')
                            np.mean(chunk, axis=1, dtype=np.float32,
                                    out=mapped[cursor:cursor+len(chunk)])
                            cursor += len(chunk)
                        mapped.flush()
                    finally:
                        del mapped
            except (sf.LibsndfileError, _ShortAudioRead):
                # Unsupported codecs and truncated/inaccurate compressed
                # frame headers use the alternate decoder. Allocation
                # failures still propagate and are never retried here.
                values, rate, _ = _read_mono(path)
                frames = len(values)
                with array_path.open('wb') as stream:
                    np.save(stream, values, allow_pickle=False)
            stable_array = directory / 'waveform.npy'
            if stable_array.exists():
                stable_array.unlink(missing_ok=True)
            array_path.replace(stable_array)
            metadata = dict(schema=SCHEMA, array=stable_array.name, sample_rate=rate,
                            frames=frames, audio_sha256=sha256)
            with tempfile.NamedTemporaryFile(dir=directory, suffix='.json.tmp', delete=False) as stream:
                temporary_index = Path(stream.name)
                stream.write(json.dumps(metadata).encode('utf-8'))
            temporary_index.replace(directory / 'index.json')
            published = True
        finally:
            if temporary_index is not None:
                temporary_index.unlink(missing_ok=True)
            if not published:
                array_path.unlink(missing_ok=True)
        return _load(directory)


def mapped_waveform(path, sha256=None, cache_dir=None):
    from maichart.form.rational_audio import _digest
    source = Path(path).resolve()
    stat = source.stat()
    return _mapped(str(source), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
                   sha256 or _digest(source), str(Path(cache_dir).resolve()) if cache_dir else None)
