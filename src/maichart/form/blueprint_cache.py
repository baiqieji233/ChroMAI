"""Persistent CPU cache for song-form Blueprint planner inputs.

The preparation cache stores parsed chart structure.  This module stores only
the derived tensors consumed by the whole-song planner, using an independent
schema so existing caches remain readable.  Callers provide all effective
input identity fields (tempo map, first, conditions, quality and curve
metadata) to prevent stale tensors from crossing training configurations.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
import tempfile
import threading
import zipfile

import numpy as np

SCHEMA = 'song_form_blueprint_input_v1'


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def _digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def torch_finite(tensor):
    import torch
    return bool(torch.isfinite(tensor).all()) if tensor.is_floating_point() or tensor.is_complex() else True


class BlueprintInputCache:
    """Content-checked cache for CPU planner inputs and targets."""

    def __init__(self, cache_dir):
        self.root = Path(cache_dir) / 'blueprint_inputs' / 'v1'
        package = Path(__file__).resolve().parent
        names = ('blueprint_cache.py', 'train.py', 'data.py', 'blueprint.py',
                 'rational_planner.py', 'rational_audio.py', 'curve.py',
                 'timing.py', '../generation/event_audio.py')
        parts = {}
        for name in names:
            path = package / name
            if path.is_file():
                parts[name] = _digest(path)
        self.code_sha256 = hashlib.sha256(_json_bytes(parts)).hexdigest()
        self._counts = dict(hits=0, misses=0, rebuilt=0, invalid=0, write_errors=0)
        self._lock = threading.RLock()

    def identity(self, chart=None, *, audio_profile=None, max_tokens=None,
                 max_memory_tokens=None, max_region_seconds=None, **extra):
        """Build deterministic key metadata from every effective input.

        ``extra`` should include effective tempo map, first, condition vector,
        quality/region weights, load-curve identity, meter map and any feature
        settings.  It is intentionally not inferred here: the pipeline knows
        the real post-annotation values and must pass them explicitly.
        """
        chart = chart or {}
        identity = {
            'schema': SCHEMA, 'code_sha256': self.code_sha256,
            'source': chart.get('source'), 'difficulty': chart.get('difficulty'),
            'chart_sha256': chart.get('chart_sha256'),
            'audio_sha256': chart.get('audio_sha256'),
            'audio_profile': audio_profile,
            'max_tokens': max_tokens,
            'max_memory_tokens': max_memory_tokens,
            'max_region_seconds': max_region_seconds,
        }
        identity.update(extra)
        # Canonicalise and validate JSON types before deriving the key.
        identity = json.loads(_json_bytes(identity).decode('utf-8'))
        return hashlib.sha256(_json_bytes(identity)).hexdigest(), identity

    def identity_for_dataset(self, dataset, index, *, max_tokens=None,
                             max_region_seconds=None, audio_feature_config=None):
        """Derive a strict per-song identity without loading its lazy audio.

        The chart audit carries content hashes from preparation, while the
        prepared in-memory record carries effective timing, conditions,
        annotation weights, curves and relation descriptors.  Hashing these
        derived values makes the helper safe across first/BPM/quality edits.
        """
        song, points, level, weight, _state, condition, excluded, record = dataset.songs[index]
        chart = dataset.audit['charts'][index]
        if audio_feature_config is None:
            if song.audio_profile == 'rational_audio_queries_v1':
                from maichart.form.rational_audio import ACOUSTIC_DIM, INPUT_DIM, MAX_EVIDENCE_NODES, PROFILE
                audio_feature_config = {'profile': PROFILE, 'acoustic_dim': ACOUSTIC_DIM,
                    'planner_input_dim': INPUT_DIM, 'max_evidence_nodes': MAX_EVIDENCE_NODES,
                    'fixed_feature_rate': None, 'clock': 'native_sample_exact_fraction_v1'}
            else:
                from maichart.generation.event_audio import _feature_config
                audio_feature_config = _feature_config(22050, 441)
        def array_hash(value):
            if value is None:
                return None
            array = np.ascontiguousarray(value)
            digest = hashlib.sha256()
            digest.update(str(array.dtype).encode())
            digest.update(_json_bytes(list(array.shape)))
            digest.update(array.view(np.uint8))
            return digest.hexdigest()
        curve_sha = hashlib.sha256(_json_bytes({
            'times': array_hash(song.load_curve_times_sec),
            'values': array_hash(song.load_curve),
        })).hexdigest()
        effective = {
            'tempo_map': [[str(entry.beat), float(entry.bpm)] for entry in song.tempo],
            'first': float(song.first),
            'meter_map': [meter.to_dict() for meter in song.meters],
            'spans': [span.to_dict() for span in song.spans],
            'regions': [region.to_dict() for region in song.regions],
            'points_sha256': hashlib.sha256(_json_bytes([[str(p.beat), p.note_text]
                                                         for p in points])).hexdigest(),
            'condition': condition.to_vector().detach().cpu().tolist(),
            'level': float(level), 'chart_weight': float(weight),
            'quality_record': record, 'excluded_regions': excluded,
            'curve_sha256': curve_sha,
            'form_descriptor_sha256': array_hash(song.relations.descriptors),
            'audio_profile': song.audio_profile,
            'audio_feature_config': audio_feature_config,
            'max_tokens': max_tokens,
            'max_memory_tokens': getattr(dataset, 'max_memory_tokens', None),
            'max_region_seconds': max_region_seconds,
        }
        return self.identity(chart, **effective)

    def stats(self):
        with self._lock:
            return {**self._counts, 'path': str(self.root)}

    def _count(self, name):
        with self._lock:
            self._counts[name] += 1

    def load(self, key, identity):
        path = self.root / f'{key}.pt'
        if not path.is_file():
            self._count('misses')
            return None
        try:
            import torch
            payload = torch.load(path, map_location='cpu', weights_only=True)
            if payload.get('schema') != SCHEMA or payload.get('identity') != identity:
                raise ValueError('Blueprint cache identity mismatch')
            result = payload['tensors']
            if not isinstance(result, dict) or any(not torch.is_tensor(v) for v in result.values()):
                raise ValueError('Blueprint cache tensors are invalid')
            if any(v.device.type != 'cpu' or not torch.isfinite(v).all() for v in result.values()):
                raise ValueError('Blueprint cache contains non-CPU or nonfinite tensors')
            result['_weight'] = float(payload['weight'])
            self._count('hits')
            return result
        except (OSError, ValueError, TypeError, KeyError, IndexError, EOFError,
                pickle.UnpicklingError,
                RuntimeError, zipfile.BadZipFile):
            self._count('invalid'); self._count('misses')
            return None

    def save(self, key, identity, inputs, targets, weight):
        self._count('rebuilt')
        temporary = []
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            tensors = {}
            for prefix, values in (('input_', inputs), ('target_', targets)):
                for name, value in values.items():
                    if not hasattr(value, 'detach'):
                        raise TypeError(f'Blueprint cache value {name} is not a tensor')
                    tensor = value.detach().cpu().contiguous()
                    if not torch_finite(tensor):
                        raise ValueError(f'Blueprint cache value {name} is nonfinite')
                    tensors[prefix + name] = tensor
            import torch
            payload = {'schema': SCHEMA, 'identity': identity,
                       'tensors': tensors, 'weight': float(weight)}
            with tempfile.NamedTemporaryFile(dir=self.root, suffix='.tmp', delete=False) as stream:
                temp_path = Path(stream.name); temporary.append(temp_path)
            torch.save(payload, temp_path)
            temp_path.replace(self.root / f'{key}.pt')
            return None
        except (OSError, ValueError, TypeError) as error:
            self._count('write_errors'); return str(error)
        finally:
            for path in temporary:
                try: path.unlink(missing_ok=True)
                except OSError: pass

    def get_or_build(self, key, identity, builder):
        """Return ``(inputs, targets, weight, cache_status)``.

        ``builder`` is called only after a cache miss and must return CPU
        tensors.  The status is ``hit``, ``rebuilt`` or ``uncached`` and is useful
        for progress logging without making the cache part of the training
        objective.
        """
        cached = self.load(key, identity)
        if cached is not None:
            inputs = {name[len('input_'):]: value for name, value in cached.items()
                      if name.startswith('input_')}
            targets = {name[len('target_'):]: value for name, value in cached.items()
                       if name.startswith('target_')}
            return inputs, targets, cached['_weight'], 'hit'
        built = builder()
        if len(built) != 3:
                raise ValueError('Blueprint cache builder must return inputs, targets, weight')
        inputs, targets, weight = built
        warning = self.save(key, identity, inputs, targets, weight)
        return inputs, targets, float(weight), ('rebuilt' if warning is None else 'uncached')


def blueprint_cache_identity(cache, chart, *, audio_profile, max_tokens,
                             max_memory_tokens, max_region_seconds, **extra):
    """Small integration helper used by the training pipeline."""
    return cache.identity(chart, audio_profile=audio_profile, max_tokens=max_tokens,
                          max_memory_tokens=max_memory_tokens,
                          max_region_seconds=max_region_seconds, **extra)
