"""Content-checked JSON/NPZ cache for deterministic chart preparation."""
from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import tempfile
import threading
import zipfile

import numpy as np

from maichart.event_timing import EventPoint, EventTempo
from maichart.form.recurrence import Relation, SongRelations
from maichart.form.timing import MeterChange, MetricSpan, normalize_meter_map
from maichart.generation.conditions import GenerationCondition
from maichart.generation.event_audio import _feature_config
from maichart.generation.sustain_state import SustainStateIndex


SCHEMA = 'song_form_preparation_v1'
_DEPENDENCIES = (
    'form/preparation_cache.py', 'form/data.py', 'form/codec.py',
    'form/rational_audio.py', 'form/waveform_cache.py', 'form/rhythm_geometry.py',
    'form/timing.py', 'form/segmentation.py', 'form/recurrence.py', 'form/curve.py',
    'event_timing.py', 'maidata.py', 'notes.py', 'timing.py', 'duration.py', 'ir.py',
    'generation/event_compile.py', 'generation/event_audio.py', 'generation/event_model.py',
    'generation/timeline.py', 'generation/simai_compile.py',
    'generation/event_quality.py', 'generation/conditions.py',
    'generation/sustain_state.py', 'generation/action_state_spec.py',
    'generation/muri/parser.py', 'generation/muri/model.py', 'generation/muri/data.py',
    'training/chart_sanitization.py', 'training/event_dataset.py',
    'training/quality_annotations.py',
)


def _digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


class PreparationCache:
    def __init__(self, cache_dir):
        self.root = Path(cache_dir) / 'song_form_prepare' / 'v1'
        package = Path(__file__).resolve().parents[1]
        dependencies = list(_DEPENDENCIES)
        dependencies.extend(str(p.relative_to(package)).replace('\\', '/')
                            for p in sorted((package / 'generation/muri/reference').glob('*')) if p.is_file())
        self.code_hash = hashlib.sha256(_json_bytes({name: _digest(package / name)
                                                    for name in dependencies})).hexdigest()
        self.runtime = {}
        for name in ('numpy', 'scipy', 'librosa', 'torch'):
            try:
                self.runtime[name] = version(name)
            except PackageNotFoundError:
                self.runtime[name] = None
        self._lock = threading.RLock()
        self._digests = {}
        self._counts = dict(hits=0, misses=0, rebuilt=0, invalid=0, write_errors=0)

    def file_digest(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        with self._lock:
            cached = self._digests.get(path)
            if cached is not None and cached[0] == signature:
                return cached[1]
            checksum = _digest(path)
            self._digests[path] = (signature, checksum)
            return checksum

    def identity(self, source, audio_path, slot, level, record, options):
        source, audio_path = Path(source).resolve(), Path(audio_path).resolve()
        structure = source.with_name('music_structure.json')
        meters = [item.to_dict() for item in normalize_meter_map(options['meters'])]
        identity = {
            'schema': SCHEMA, 'code_sha256': self.code_hash,
            'runtime': self.runtime,
            # Absolute dataset paths are deliberately excluded. The content
            # hashes and structural settings fully determine this artifact;
            # keeping paths here made identical copied datasets miss every
            # preparation cache entry.
            'chart_sha256': self.file_digest(source),
            'audio_sha256': self.file_digest(audio_path),
            'audio_config': ({'profile': options['audio_profile'], 'fixed_feature_rate': None}
                             if options.get('audio_profile') == 'rational_audio_queries_v1'
                             else _feature_config(22050, 441)),
            'structure_sha256': self.file_digest(structure) if structure.exists() else None,
            'difficulty': slot, 'level': level, 'quality': record,
            'settings': {'max_tokens': options['max_tokens'],
                         'audio_profile': options.get('audio_profile', 'legacy_50hz'),
                         'rhythm_geometry_profile': options.get('rhythm_geometry_profile', 'none'),
                         'max_region_seconds': options['max_region_seconds'],
                         'reconstruction': options['reconstruction'], 'meter_map': meters},
        }
        return hashlib.sha256(_json_bytes(identity)).hexdigest(), identity

    def stats(self):
        with self._lock:
            return {**self._counts, 'path': str(self.root)}

    def _count(self, key):
        with self._lock:
            self._counts[key] += 1

    def load(self, key, identity, *, source_path=None, audio_path=None):
        index = self.root / f'{key}.index.json'
        if not index.exists():
            # Migrate caches written before paths were removed from the
            # identity. Compare a normalized identity while scanning only
            # the preparation index files; successful entries are copied to
            # the new stable key on the next save path.
            for candidate in self.root.glob('*.index.json'):
                try:
                    probe = json.loads(candidate.read_text(encoding='utf-8'))
                    old_identity = dict(probe.get('identity') or {})
                    old_identity.pop('source', None); old_identity.pop('audio', None)
                    if old_identity != identity:
                        continue
                    arrays_path = self.root / f"{candidate.name[:-len('.index.json')]}.{probe['arrays_sha256']}.npz"
                    with np.load(arrays_path, allow_pickle=False) as arrays:
                        result = _decode(probe['result'], arrays)
                    if source_path is not None: result['chart']['source'] = str(Path(source_path).resolve())
                    if audio_path is not None: result['audio_path'] = Path(audio_path).resolve()
                    self._count('hits')
                    return result, 'hit_legacy_identity'
                except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
                    continue
            self._count('misses')
            return None, 'miss'
        try:
            metadata = json.loads(index.read_text(encoding='utf-8'))
            if metadata['schema'] != SCHEMA or metadata['identity'] != identity:
                raise ValueError('Preparation index identity mismatch')
            checksum = metadata['arrays_sha256']
            if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
                raise ValueError('Invalid preparation array checksum')
            arrays_path = self.root / f'{key}.{checksum}.npz'
            if self.file_digest(arrays_path) != metadata['arrays_sha256']:
                raise ValueError('Preparation arrays checksum mismatch')
            with np.load(arrays_path, allow_pickle=False) as arrays:
                result = _decode(metadata['result'], arrays)
                if source_path is not None:
                    result['chart']['source'] = str(Path(source_path).resolve())
                if audio_path is not None:
                    result['audio_path'] = Path(audio_path).resolve()
            self._count('hits')
            return result, 'hit'
        except (OSError, ValueError, TypeError, KeyError, IndexError, EOFError, zipfile.BadZipFile):
            self._count('invalid')
            self._count('misses')
            return None, 'invalid'

    def save(self, key, identity, result):
        self._count('rebuilt')
        temporary = []
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            metadata, arrays = _encode(result)
            with tempfile.NamedTemporaryFile(dir=self.root, suffix='.tmp', delete=False) as stream:
                arrays_temp = Path(stream.name)
                temporary.append(arrays_temp)
                np.savez_compressed(stream, **arrays)
            checksum = _digest(arrays_temp)
            arrays_temp.replace(self.root / f'{key}.{checksum}.npz')
            payload = dict(schema=SCHEMA, identity=identity, arrays_sha256=checksum, result=metadata)
            with tempfile.NamedTemporaryFile(dir=self.root, suffix='.tmp', delete=False) as stream:
                index_temp = Path(stream.name)
                temporary.append(index_temp)
                stream.write(_json_bytes(payload))
            # Publish the checked array pair before replacing its index. Each
            # writer has private temporary names, including concurrent runs.
            index_temp.replace(self.root / f'{key}.index.json')
            return None
        except (OSError, ValueError, TypeError) as error:
            self._count('write_errors')
            return str(error)
        finally:
            for path in temporary:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass


def _encode(result):
    song, points, level, weight, state, condition, excluded, record = result['song']
    tempo = [[str(t.beat), t.bpm] for t in song.tempo]
    metadata = {key: result[key] for key in ('skipped', 'filtered', 'items', 'chart')}
    metadata.update(audio_path=str(result['audio_path']), level=level, weight=weight,
                    points=[[str(p.beat), p.note_text] for p in points],
                    condition=condition.to_dict(), excluded=excluded, record=record,
                    state_markers=state.markers,
                    song={'tempo': tempo, 'first': song.first, 'audio_profile': song.audio_profile,
                          'rhythm_geometry_profile': song.rhythm_geometry_profile,
                          'meters': [m.to_dict() for m in song.meters],
                          'spans': [s.to_dict() for s in song.spans],
                          'regions': [s.to_dict() for s in song.regions],
                          'span_region_indices': song.span_region_indices, 'audit': song.audit,
                          'relation_spans': [s.to_dict() for s in song.relations.spans],
                          'relation_links': [asdict(link) for link in song.relations.links],
                          'relation_audit': song.relations.audit})
    arrays = {'descriptors': song.relations.descriptors,
              'load_curve_times_sec': song.load_curve_times_sec, 'load_curve': song.load_curve}
    return metadata, arrays


def _decode(metadata, arrays):
    # Import only after data.py has defined FormSong, avoiding a module cycle.
    from maichart.form.data import FormSong
    values = metadata['song']
    descriptors, times, curve = (arrays[key].copy() for key in
                                ('descriptors', 'load_curve_times_sec', 'load_curve'))
    if (descriptors.ndim != 2 or times.ndim != 1
            or len(descriptors) != len(values['relation_spans'])
            or not all(np.isfinite(a).all() for a in (descriptors, times, curve))):
        raise ValueError('Preparation arrays are invalid')
    tempo = [EventTempo(Fraction(beat), float(bpm)) for beat, bpm in values['tempo']]
    points = [EventPoint(Fraction(beat), text) for beat, text in metadata['points']]
    relations = SongRelations(tuple(MetricSpan.from_dict(s) for s in values['relation_spans']),
                              descriptors, [Relation(**link) for link in values['relation_links']],
                              values['relation_audit'])
    song = FormSong(None, tempo, values['first'], [MeterChange.from_dict(m) for m in values['meters']],
                    [MetricSpan.from_dict(s) for s in values['spans']], relations, values['audit'],
                    [MetricSpan.from_dict(s) for s in values['regions']], values['span_region_indices'],
                    times, curve, values.get('audio_profile','legacy_50hz'),
                    values.get('rhythm_geometry_profile', 'none'))
    # Restore parsed markers instead of recompiling and parsing the same chart.
    # Its code/reference digest participates in the cache identity above.
    state = SustainStateIndex.__new__(SustainStateIndex)
    state.points, state.tempo_changes, state.first = tuple(points), tuple(tempo), float(song.first)
    state.markers, state._references = metadata['state_markers'], None
    condition = GenerationCondition.from_dict(metadata['condition'])
    if 'metadata' in metadata['condition']:
        from maichart.generation.event_quality import EventGenerationCondition
        condition = EventGenerationCondition(**condition.to_dict(), metadata=metadata['condition']['metadata'])
    items = [(item[0], item[1], item[2], item[3]) for item in metadata['items']]
    result = {key: metadata[key] for key in ('skipped', 'filtered', 'chart')}
    result.update(song=(song, points, metadata['level'], metadata['weight'], state,
                        condition,
                        [tuple(pair) for pair in metadata['excluded']], metadata['record']),
                  audio_path=Path(metadata['audio_path']), items=items)
    return result
