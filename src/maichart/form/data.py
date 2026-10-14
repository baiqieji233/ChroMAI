"""Song-level organization and causal training inputs for the new family."""
from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import json
import os
from bisect import bisect_left
from pathlib import Path
import threading
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from maichart.event_timing import EventPoint, parse_event_chart, seconds_at_beat
from maichart.generation.conditions import GenerationCondition, GENERATION_DEFAULTS
from maichart.generation.event_quality import condition_from_events
from maichart.generation.event_audio import (
    audio_segment_tensor, extract_event_audio, event_audio_cache_path, load_event_audio,
    _normalize_tempo_map,
    _beats_at_audio_offsets, segment_frame_times,
)
from maichart.generation.sustain_state import SustainStateIndex
from maichart.form.codec import RationalCodec
from maichart.form.recurrence import analyze_recurrence, CausalMotifMemory
from maichart.form.timing import MetricSpan, normalize_meter_map
from maichart.form.revision import DraftPoint
from maichart.training.chart_sanitization import sanitize_chart
from maichart.training.dataset_catalog import _source_asset
from maichart.training.event_dataset import inspect_chart, numeric_level


class _AudioStore:
    def __init__(self, loader, cache_dir, budget=1024*1024*1024):
        self.loader,self.cache_dir,self.budget=loader,cache_dir,budget
        self.entries=OrderedDict();self.bytes=0
        self.hits=self.misses=self.evictions=self.loads=0
        self.disk_loads=self.source_resolutions=0
        self._resolved_sources={}
        self._lock=threading.RLock()

    def __getstate__(self):
        state=self.__dict__.copy();state.pop('_lock',None)
        # Spawned workers get independent, lazy caches rather than a copy of
        # every resident audio feature array in the parent process.
        state['entries']=OrderedDict();state['bytes']=0
        state['hits']=state['misses']=state['evictions']=state['loads']=0
        state['disk_loads']=state['source_resolutions']=0
        return state

    def __setstate__(self,state):
        self.__dict__.update(state);self._lock=threading.RLock()

    def disk_cache_path(self,path):
        """Return this source's resolved event-audio cache path, if applicable."""
        from maichart.form.rational_audio import extract_rational_audio, cache_path as rational_cache_path
        if self.loader not in (extract_event_audio, extract_rational_audio) or self.cache_dir is None:
            return None
        key=str(path)
        source=Path(path)
        stat=source.stat()
        signature=(stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        with self._lock:
            resolution=self._resolved_sources.get(key)
            if resolution is None or resolution[0] != signature:
                cache_path=(rational_cache_path(source,self.cache_dir) if self.loader is extract_rational_audio
                            else event_audio_cache_path(source,self.cache_dir))
                resolution=(signature,cache_path)
                self._resolved_sources[key]=resolution
                self.source_resolutions+=1
                if key in self.entries:
                    old=self.entries.pop(key)
                    self.bytes-=old.features.nbytes+old.times.nbytes
            return resolution[1]

    def get(self,path):
        key=str(path)
        with self._lock:
            cache_path=self.disk_cache_path(path)
            if key in self.entries:
                self.hits+=1
                self.entries.move_to_end(key);return self.entries[key]
            self.misses+=1
            if cache_path is not None and cache_path.is_file():
                try:
                    from maichart.form.rational_audio import extract_rational_audio, load_rational_audio
                    audio=(load_rational_audio(cache_path) if self.loader is extract_rational_audio
                           else load_event_audio(cache_path,sample_rate=22050,hop_length=441))
                    self.disk_loads+=1
                except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError):
                    audio=self.loader(path,cache_dir=self.cache_dir)
            else:
                audio=self.loader(path,cache_dir=self.cache_dir)
            self.loads+=1
            size=audio.features.nbytes+audio.times.nbytes
            if self.budget is None or self.budget > 0:
                self.entries[key]=audio;self.bytes+=size
                while len(self.entries)>1 and self.budget is not None and self.bytes>self.budget:
                    _,old=self.entries.popitem(last=False);self.bytes-=old.features.nbytes+old.times.nbytes
                    self.evictions+=1
                if self.budget == 0:
                    self.entries.pop(key, None);self.bytes-=size
            return audio

    def stats(self):
        with self._lock:
            return {'hits':self.hits,'misses':self.misses,'loads':self.loads,
                    'evictions':self.evictions,'entries':len(self.entries),
                    'bytes':self.bytes,'budget_bytes':self.budget,
                    'disk_loads':self.disk_loads,'source_resolutions':self.source_resolutions}


class _SampleStore:
    def __init__(self, budget):
        self.budget=budget;self.entries=OrderedDict();self.bytes=0
        self.hits=self.misses=self.evictions=0

    def __getstate__(self):
        return {'budget':self.budget,'entries':OrderedDict(),'bytes':0,
                'hits':0,'misses':0,'evictions':0}

    def __setstate__(self,state): self.__dict__.update(state)

    @staticmethod
    def _size(value):
        if torch.is_tensor(value): return value.numel()*value.element_size()
        if isinstance(value, dict): return sum(_SampleStore._size(v) for v in value.values())
        if isinstance(value, (tuple,list)): return sum(_SampleStore._size(v) for v in value)
        return 0

    @staticmethod
    def _copy(value):
        if torch.is_tensor(value): return value.clone()
        if isinstance(value, dict): return {k:_SampleStore._copy(v) for k,v in value.items()}
        if isinstance(value, list): return [_SampleStore._copy(v) for v in value]
        if isinstance(value, tuple): return tuple(_SampleStore._copy(v) for v in value)
        return value

    def get(self,key):
        if key not in self.entries:
            self.misses+=1;return None
        self.hits+=1;self.entries.move_to_end(key)
        return self._copy(self.entries[key][0])

    def put(self,key,value):
        if self.budget == 0: return
        stored=self._copy(value);size=self._size(stored)
        if self.budget is not None and size > self.budget: return
        old=self.entries.pop(key,None)
        if old: self.bytes-=old[1]
        self.entries[key]=(stored,size);self.bytes+=size
        while self.budget is not None and self.bytes>self.budget:
            _,(_,removed)=self.entries.popitem(last=False)
            self.bytes-=removed;self.evictions+=1

    def stats(self):
        return {'hits':self.hits,'misses':self.misses,'evictions':self.evictions,
                'entries':len(self.entries),'bytes':self.bytes,'budget_bytes':self.budget}


class _LazyAudio:
    def __init__(self,store,path): self.store,self.path=store,path
    def __getstate__(self): return {'store':self.store,'path':self.path}
    def __setstate__(self,state): self.store,self.path=state['store'],state['path']
    def __getattr__(self,key):
        try: store=object.__getattribute__(self,'store')
        except AttributeError: raise AttributeError(key) from None
        path=object.__getattribute__(self,'path')
        return getattr(store.get(path),key)


@dataclass
class FormSong:
    audio: object
    tempo: list
    first: float
    meters: list
    spans: list
    relations: object
    audit: dict
    regions: list[MetricSpan] | None = None
    span_region_indices: list[int] | None = None
    load_curve_times_sec: np.ndarray | None = None
    load_curve: np.ndarray | None = None
    audio_profile: str = "legacy_50hz"

    def __post_init__(self):
        # Old direct constructors treat every supplied span as its own region.
        if self.regions is None:
            self.regions = list(self.spans)
        if self.span_region_indices is None:
            self.span_region_indices = list(range(len(self.spans)))
        if len(self.span_region_indices) != len(self.spans):
            raise ValueError("span_region_indices must map every computational span")
        if any(index < 0 or index >= len(self.regions) for index in self.span_region_indices):
            raise ValueError("span_region_indices contains an unknown musical region")
        if self.load_curve_times_sec is not None or self.load_curve is not None:
            if self.load_curve_times_sec is None or self.load_curve is None:
                raise ValueError("load curve times and values must be supplied together")
            if self.load_curve.ndim != 2 or self.load_curve.shape[1] != 4:
                raise ValueError("load_curve must have shape [T, 4]")
            if len(self.load_curve_times_sec) != len(self.load_curve):
                raise ValueError("load curve times and values length mismatch")


def organize_song(audio, tempo, first, *, meters=None, spans=None, min_seconds=2.0, max_seconds=14.0):
    meters = normalize_meter_map(meters)
    if spans is None:
        from maichart.form.segmentation import segment_music
        segmented = segment_music(audio, tempo, first, meters=meters,
                                  min_seconds=min_seconds, max_seconds=max_seconds)
        spans, regions = segmented.spans, segmented.region_spans
        span_region_indices = list(segmented.span_region_indices)
        audit = dict(segmented.audit)
        audit['musical_regions'] = [{"start":str(r.start),"end":str(r.end),"reason":r.end_reason} for r in segmented.regions]
    else:
        spans, audit = list(spans), {"source": "explicit_spans"}
        regions = list(spans)
        span_region_indices = list(range(len(spans)))
    relations = analyze_recurrence(audio, regions, tempo, first)
    audit['region_count'] = len(regions)
    audit['computational_span_count'] = len(spans)
    return FormSong(audio, list(tempo), float(first), meters, spans, relations, audit,
                    regions, span_region_indices, audio_profile=(audio.metadata or {}).get('profile','legacy_50hz'))


def form_audio(song, start, end):
    if song.audio_profile == 'rational_audio_queries_v1':
        from maichart.form.rational_audio import audio_queries
        return audio_queries(song,start,end,max_nodes=1024)[0]
    from maichart.form.timing import metrical_features
    lo = float(seconds_at_beat(start, song.tempo, song.first))
    hi = float(seconds_at_beat(end, song.tempo, song.first))
    values = audio_segment_tensor(song.audio, lo, hi, start_beat=start, first=song.first,
        tempo_changes=song.tempo, bpm=song.tempo[0].bpm)
    # Legacy bar/phrase phase was hard-coded to 4/16 beats. Do not carry that
    # assumption into the new branch; explicit meter features replace it.
    values[:, -4:] = 0
    stamps = segment_frame_times(lo, hi)
    beats = _beats_at_audio_offsets(np.asarray(stamps) - song.first,
                                   _normalize_tempo_map(song.tempo, song.tempo[0].bpm))
    rational = [Fraction(str(float(b))).limit_denominator(96000) for b in beats]
    metric = torch.from_numpy(metrical_features(rational, song.meters))
    if len(metric) != len(values):
        raise ValueError("Metrical feature clock differs from acoustic clock")
    return torch.cat((values, metric), -1)


def _form_descriptor(song, index, relation=None):
    region_index = song.span_region_indices[index]
    region = song.regions[region_index]
    relative = []
    extent = max(float(song.regions[-1].end), 1.)
    for span in song.regions:
        relative.append([(float(span.start)-float(region.start))/extent,
                         (float(span.end)-float(region.start))/extent,
                         float(span.index == region_index)])
    form = np.concatenate((song.relations.descriptors,
                           np.asarray(relative, dtype=np.float32)), axis=1)
    if len(form) > 128:
        keep = set(np.linspace(0, len(form)-1, 126, dtype=int).tolist()) | {region_index}
        if relation is not None:
            keep.add(relation.source)
        form = form[sorted(keep)]
    return form


def input_example(song, index, history, *, condition=None, codec=None, task=0,
                  draft=(), right=(), max_memory_tokens=512, memory_strength=1.0,
                  state_index=None, load_curve=None, load_curve_times_sec=None,
                  load_curve_condition_mask=None, history_before=None):
    codec = codec or RationalCodec()
    span = song.spans[index]
    region_index = song.span_region_indices[index]
    region = song.regions[region_index]
    history = tuple(history)
    boundary = span.start if history_before is None else Fraction(history_before)
    if history_before is not None and (song.audio_profile != 'rational_audio_queries_v1' or not span.start <= boundary < span.end):
        raise ValueError('Continuation history boundary requires a rational-audio span')
    if any(p.beat >= boundary for p in history):
        raise ValueError("History includes current/future chart targets")
    if task == 0 and (len(draft) or len(right)):
        raise ValueError("Fresh generation cannot read draft/right targets")
    if any(p.beat < span.end for p in right):
        raise ValueError("Right context overlaps edit region")
    memory = CausalMotifMemory(song.regions)
    history_beats = [point.beat for point in history]
    for prior in song.regions:
        if prior.end <= region.start:
            lo = bisect_left(history_beats, prior.start)
            hi = bisect_left(history_beats, prior.end)
            memory.commit(prior.index, history[lo:hi])
    ref = memory.reference(region.index, song.relations)
    # A recurrence is expressed across the full musical region. A compute cap
    # receives only its overlapping portion and never becomes a motif boundary.
    points = [p for p in ref.points if span.start <= p.beat < span.end]
    motif = codec.bounded_memory(points, origin=span.start, max_tokens=max_memory_tokens, newest=False)
    relation = ref.relation
    form = _form_descriptor(song, index, relation)
    hist_origin = history[0].beat if history else span.start
    states, padding, state_meta = (state_index or SustainStateIndex(history,song.tempo,song.first)).at(boundary)
    # A reused index must still be causal. Its implementation filters all heads
    # at or after the boundary, including future-dependent release flags.
    condition = condition or GenerationCondition(radar=GENERATION_DEFAULTS,style=GENERATION_DEFAULTS)
    curve_values = song.load_curve if load_curve is None else np.asarray(load_curve, dtype=np.float32)
    curve_times = song.load_curve_times_sec if load_curve_times_sec is None else np.asarray(load_curve_times_sec, dtype=np.float32)
    if curve_values is not None and curve_times is not None:
        lo_sec = float(seconds_at_beat(span.start, song.tempo, song.first) - song.first)
        hi_sec = float(seconds_at_beat(span.end, song.tempo, song.first) - song.first)
        selected = (curve_times >= lo_sec) & (curve_times < hi_sec)
        local_curve = torch.from_numpy(curve_values[selected].copy())
        local_times = torch.from_numpy((curve_times[selected] - lo_sec).copy())
        local_curve_mask = torch.ones_like(local_curve, dtype=torch.bool) if load_curve_condition_mask is None else torch.as_tensor(load_curve_condition_mask, dtype=torch.bool)[selected].clone()
    else:
        local_curve = torch.zeros((0, 4), dtype=torch.float32)
        local_times = torch.zeros((0,), dtype=torch.float32)
        local_curve_mask = torch.zeros((0, 4), dtype=torch.bool)
    history_tokens = codec.bounded_memory(history,origin=hist_origin,max_tokens=max_memory_tokens)
    history_kept = history_tokens.count(codec.end_id)
    history_seconds = (float(seconds_at_beat(boundary, song.tempo, song.first)
                             - seconds_at_beat(history[-history_kept].beat, song.tempo, song.first))
                       if history_kept else 0.0)
    return {
        "audio": form_audio(song,span.start,span.end), "form": torch.from_numpy(form),
        "history": torch.tensor(history_tokens),
        "motif": torch.tensor(motif),
        "draft": torch.tensor(codec.bounded_memory(draft,origin=span.start,max_tokens=max_memory_tokens,newest=False)),
        "right": torch.tensor(codec.bounded_memory(right,origin=span.end,max_tokens=max_memory_tokens,newest=False)),
        "condition": condition.to_vector(), "condition_null": torch.tensor(False), "task": torch.tensor(task),
        "load_curve": local_curve, "load_curve_times_sec": local_times,
        "load_curve_condition_mask": local_curve_mask,
        "action_state": states, "action_state_padding_mask": padding,
        "relation": torch.tensor([relation.similarity,relation.confidence,relation.duration_ratio,
            (region_index-relation.source)/max(len(song.regions),1)] if relation else [0.,0.,1.,0.]),
        "memory_strength": torch.tensor(float(memory_strength)),
        "skeleton": torch.tensor(codec.bounded_memory((), origin=span.start,
            max_tokens=max_memory_tokens, newest=False)),
        "meta": {"span": span.to_dict(), "reference_source": ref.source,
                 "region": region.to_dict(), "region_index": region_index,
                 "state_count": state_meta["state_count"],
                 "state_truncated_count": state_meta.get("truncated_count", 0),
                 "history_events_available":len(history), "history_events_kept":history_kept,
                 "history_tokens":len(history_tokens), "history_coverage_seconds":history_seconds,
                 "motif_events_kept":motif.count(codec.end_id),
                 "task": "draft" if task==0 else "reconstruction"},
    }


def relation_target(target, reference, span):
    # Soft similarity evidence, not a prescribed chorus/variation class.
    def signature(points):
        rhythm = np.zeros(48,dtype=np.float32)
        types = np.zeros(4,dtype=np.float32)
        for p in points:
            pos = min(47,max(0,int(float((p.beat-span.start)/(span.end-span.start))*48)))
            rhythm[pos] += 1
            types[3 if any(c in p.note_text for c in 'ABCDE') else 2 if '[' in p.note_text and 'h' not in p.note_text else 1 if 'h' in p.note_text else 0] += 1
        return rhythm,types
    a,b=signature(target),signature(reference)
    return torch.tensor([float(np.dot(x,y)/(max(np.linalg.norm(x)*np.linalg.norm(y),1e-8))) for x,y in zip(a,b)])


def _preparation_annotation(annotations, source, slot, audio_path, cache):
    if not annotations:
        return None
    from maichart.training.quality_annotations import chart_sha256, audio_sha256, validate_record
    chart_hash = chart_sha256(source, slot)
    source_identity = Path(source).resolve()
    audio_hash = None
    for record in annotations.get('records', []):
        if int(record.get('difficulty_index', -1)) != slot or record.get('chart_sha256') != chart_hash:
            continue
        try:
            if not record.get('source_path') or Path(record['source_path']).resolve() != source_identity:
                continue
            if record.get('audio_path') and Path(record['audio_path']).resolve() != Path(audio_path).resolve():
                continue
        except (OSError, TypeError, ValueError):
            continue
        if record.get('audio_sha256'):
            if audio_hash is None:
                audio_hash = cache.file_digest(audio_path) if cache else audio_sha256(audio_path)
            if record['audio_sha256'] != audio_hash:
                continue
        return validate_record(record)
    return None


def _prepare_form_entry(entry, options, annotations, audio_store):
    path = options['manifest']
    data = options['source_data']
    codec = RationalCodec()
    results = []
    accepted = 0
    for difficulty in entry.get('difficulties', []):
        slot = int(difficulty.get('difficulty_index', difficulty.get('index', 0)))
        if not options['min_difficulty_index'] <= slot <= options['max_difficulty_index']:
            continue
        if options['limit_charts'] and accepted >= options['limit_charts']:
            break
        result = {'skipped': [], 'filtered': []}
        results.append(result)
        try:
            level = numeric_level(difficulty.get('level_raw', difficulty.get('level')))
            if not options['min_level'] <= level <= options['max_level']:
                continue
            source = _source_asset(entry['maidata_path'], path, data)
            audio_path = _source_asset(entry['audio_path'], path, data)
            cache = options.get('preparation_cache')
            record = _preparation_annotation(annotations, source, slot, audio_path, cache)
            cache_key = cache_identity = None
            cache_status = 'disabled'
            if cache is not None:
                cache_key, cache_identity = cache.identity(source, audio_path, slot, level, record, options)
                cached, cache_status = cache.load(cache_key, cache_identity)
                if cached is not None:
                    cached['cache'] = 'hit'
                    results[-1] = cached
                    accepted += 1
                    continue
            original = parse_event_chart(source, slot)
            filtered = sanitize_chart(original)
            chart = filtered.chart
            inspect_chart(chart)
            if filtered.removed:
                result['filtered'].append({'source': str(source), 'difficulty': slot, **filtered.to_dict()})
            if record and (record.get('label') in ('excluded', 'negative') or float(record.get('weight', 1)) <= 0):
                result['skipped'].append({'source': str(source), 'reason': 'quality_excluded_or_negative'})
                continue
            weight = float((record or {}).get('weight', 1.))
            audio = audio_store.get(audio_path)
            structure_path = source.parent / 'music_structure.json'
            structure = json.loads(structure_path.read_text(encoding='utf-8')) if structure_path.exists() else {}
            song = organize_song(audio, chart.tempo_changes, chart.audio_first,
                                 meters=structure.get('meter_map', options['meters']),
                                 max_seconds=options['max_region_seconds'])
            excluded = [(float(region['start_sec']), float(region['end_sec']))
                        for region in (record or {}).get('regions', []) if float(region.get('weight', 1)) <= 0]
            points = [p for p in chart.points if not any(
                lo <= seconds_at_beat(p.beat, chart.tempo_changes, chart.audio_first) < hi for lo, hi in excluded)]
            condition = condition_from_events(points, chart.tempo_changes, level=level,
                duration_sec=max(float(audio.duration_sec), 1e-3), audio=audio, audio_first=chart.audio_first)
            from maichart.form.curve import build_load_curve
            song.load_curve_times_sec, song.load_curve = build_load_curve(
                points, chart.tempo_changes, chart.audio_first, float(audio.duration_sec))
            state = SustainStateIndex(points, chart.tempo_changes, chart.audio_first)
            items = []
            for span in song.spans:
                lo = seconds_at_beat(span.start, song.tempo, song.first)
                hi = seconds_at_beat(span.end, song.tempo, song.first)
                if any(lo < b and hi > a for a, b in excluded):
                    continue
                target = [p for p in points if span.start <= p.beat < span.end]
                tokens = codec.encode(target, origin=span.start)
                if len(tokens) > options['max_tokens']:
                    result['skipped'].append({'source': str(source), 'span': span.index,
                                             'reason': 'target_token_budget', 'tokens': len(tokens)})
                    continue
                local_weight = _region_weight(lo, hi, record)
                if local_weight <= 0:
                    continue
                items.append((span.index, tokens, 0, local_weight))
                if options['reconstruction'] and target:
                    items.append((span.index, tokens, 1, local_weight))
            # Acoustic arrays stay in the atomic disk cache, not the IPC payload.
            song.audio = None
            result.update(song=(song, points, level, weight, state, condition, excluded, record),
                audio_path=audio_path, items=items,
                chart={'source': str(source), 'difficulty': slot, 'spans': len(song.spans),
                       'links': len(song.relations.links), 'unknown_meter': not bool(song.meters),
                       'chart_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                       'audio_sha256': (audio.metadata or {}).get('audio_sha256'),
                       'meter_map': [m.to_dict() for m in song.meters], 'quality': record, 'weight': weight})
            # The source bytes alone cannot reveal that an old parser ignored
            # difficulty-specific offsets. Make this changed training alignment
            # part of the existing strict resume contract when an override exists.
            from maichart.maidata import parse_maidata_file
            raw_metadata = parse_maidata_file(source)
            if raw_metadata.fields.get(f'first_{slot}', '').strip():
                result['chart']['first_override'] = {
                    'global_first':raw_metadata.first or '0', 'effective_first':float(chart.first),
                    'semantics':'difficulty_overrides_global_v1'}
            if cache is not None:
                warning = cache.save(cache_key, cache_identity, result)
                result.update(cache='rebuilt', cache_reason=cache_status)
                if warning:
                    result['cache_warning'] = warning
            else:
                result['cache'] = 'disabled'
            accepted += 1
        except (OSError, ValueError, KeyError) as error:
            result['skipped'].append({'song': entry.get('song_id'), 'difficulty': slot, 'reason': str(error)})
    return results


_form_thread_local = threading.local()


def _prepare_form_thread_worker(entry, options, annotations):
    audio_store = getattr(_form_thread_local, 'audio_store', None)
    if audio_store is None or audio_store.cache_dir != options['cache_dir']:
        audio_store = _AudioStore(extract_event_audio, options['cache_dir'], budget=64 * 1024 * 1024)
        _form_thread_local.audio_store = audio_store
    return _prepare_form_entry(entry, options, annotations, audio_store)


def _ordered_form_preparation(entries, options, annotations, audio_store, workers):
    if workers == 1:
        for entry in entries:
            yield _prepare_form_entry(entry, options, annotations, audio_store)
        return
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='song-form-prepare')
    try:
        iterator = iter(entries)
        pending = deque()
        for _ in range(min(len(entries), workers * 2)):
            pending.append(pool.submit(_prepare_form_thread_worker, next(iterator), options, annotations))
        while pending:
            # Ordered, bounded admission keeps chart ids/seed offsets stable and
            # prevents a full dataset of completed results accumulating in RAM.
            yield pending.popleft().result()
            entry = next(iterator, None)
            if entry is not None:
                pending.append(pool.submit(_prepare_form_thread_worker, entry, options, annotations))
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


class FormDataset(Dataset):
    """Separate target schema; old checkpoints never consume these examples."""
    def __init__(self, manifest, *, cache_dir, max_tokens=2048, max_memory_tokens=512,
                 meters=None, seed=0, limit_charts=0, reconstruction=True,
                 min_level=12.,max_level=15.,min_difficulty_index=4,max_difficulty_index=6,
                 max_region_seconds=14., quality_enabled=True, audio_loader=extract_event_audio,
                 preparation_workers=1, audio_cache_gb=1, sample_cache_mb=256, audio_profile='legacy_50hz'):
        from maichart.form.rational_audio import PROFILE, extract_rational_audio
        if audio_profile == PROFILE:
            if audio_loader is not extract_event_audio:
                raise ValueError('Rational audio profile owns its waveform query loader')
            audio_loader = extract_rational_audio
        elif audio_profile != 'legacy_50hz':
            raise ValueError('Unknown form audio profile')
        self.codec=RationalCodec(); self.items=[]; self.songs=[]; self.audit={"skipped":[],"filtered":[],"charts":[],"supervision":"draft_and_masked_reconstruction","quality_enabled":bool(quality_enabled)}
        self.max_memory_tokens=max_memory_tokens; self.seed=seed
        if audio_cache_gb is not None and audio_cache_gb < 0:
            raise ValueError('audio_cache_gb must be nonnegative or None')
        if sample_cache_mb is not None and sample_cache_mb < 0:
            raise ValueError('sample_cache_mb must be nonnegative or None')
        audio_budget=None if audio_cache_gb is None else int(audio_cache_gb*1024**3)
        sample_budget=None if sample_cache_mb is None else int(sample_cache_mb*1024**2)
        self.audio_store=_AudioStore(audio_loader,cache_dir,budget=audio_budget)
        self.sample_store=_SampleStore(sample_budget)
        self._point_beats=[]
        path=Path(manifest).resolve(); data=json.loads(path.read_text(encoding='utf-8-sig'))
        annotations=None
        annotation_path=path.with_name('quality_annotations.json')
        if quality_enabled and annotation_path.exists():
            from maichart.training.quality_annotations import load_quality_annotations
            annotations=load_quality_annotations(annotation_path)
        if isinstance(preparation_workers, bool) or not isinstance(preparation_workers, int) or preparation_workers < 0:
            raise ValueError('preparation_workers must be a nonnegative integer')
        entries = data.get('songs', [])
        workers = preparation_workers or min(4, os.cpu_count() or 1)
        workers = max(1, min(workers, len(entries), os.cpu_count() or 1))
        if audio_loader not in (extract_event_audio, extract_rational_audio):
            workers = 1
        preparation_cache = None
        if audio_loader in (extract_event_audio, extract_rational_audio):
            from maichart.form.preparation_cache import PreparationCache
            preparation_cache = PreparationCache(cache_dir)
        options = dict(manifest=path, source_data={'source_root': data.get('source_root')}, cache_dir=cache_dir,
            max_tokens=max_tokens, meters=meters, limit_charts=limit_charts, reconstruction=reconstruction,
            min_level=min_level, max_level=max_level, min_difficulty_index=min_difficulty_index,
            max_difficulty_index=max_difficulty_index, max_region_seconds=max_region_seconds,
            preparation_cache=preparation_cache, audio_profile=audio_profile)
        started = time.monotonic()
        print('form_prepare '+json.dumps({'stage':'preparation','workers':workers,'requested_workers':preparation_workers,
            'total_songs':len(entries),'completed_songs':0,'charts':0,
            'preparation_cache':preparation_cache.stats() if preparation_cache else {'enabled':False},
            'custom_audio_loader':audio_loader is not extract_event_audio}),flush=True)
        prepared = _ordered_form_preparation(entries, options, annotations, self.audio_store, workers)
        try:
            for completed, results in enumerate(prepared, 1):
                for result in results:
                    if limit_charts and len(self.songs) >= limit_charts:
                        break
                    self.audit['skipped'].extend(result['skipped'])
                    self.audit['filtered'].extend(result['filtered'])
                    if result.get('cache_warning'):
                        print('form_warning '+json.dumps({'message':'Preparation cache could not be saved; training continues.',
                            'detail':result['cache_warning']},ensure_ascii=False),flush=True)
                    for filtered in result['filtered']:
                        print(f'form_chart_filter source={filtered["source"]} removed={filtered["removed_count"]}',flush=True)
                    if 'song' not in result:
                        continue
                    idx = len(self.songs)
                    song = result['song'][0]
                    song.audio = _LazyAudio(self.audio_store, result['audio_path'])
                    self.songs.append(result['song'])
                    self._point_beats.append([p.beat for p in result['song'][1]])
                    self.audit['charts'].append(result['chart'])
                    self.items.extend((idx, *item) for item in result['items'])
                    print('form_prepare '+json.dumps({'stage':'chart','charts':len(self.songs),
                        'source':result['chart']['source'],'difficulty':result['chart']['difficulty'],'spans':len(song.spans),
                        'workers':workers,'completed_songs':completed,'total_songs':len(entries),
                        'cache':result.get('cache','disabled'),'cache_reason':result.get('cache_reason'),
                        'elapsed_sec':round(time.monotonic()-started,2)},ensure_ascii=False),flush=True)
                if completed % 25 == 0:
                    print('form_prepare '+json.dumps({'stage':'preparation','workers':workers,
                        'completed_songs':completed,'total_songs':len(entries),'charts':len(self.songs),
                        'preparation_cache':preparation_cache.stats() if preparation_cache else {'enabled':False}}),flush=True)
                if limit_charts and len(self.songs) >= limit_charts:
                    break
        finally:
            prepared.close()
        self.preparation_cache_stats = preparation_cache.stats() if preparation_cache else {'enabled':False}
        self.audit['preparation_cache'] = self.preparation_cache_stats
        print('form_prepare '+json.dumps({'stage':'ready','charts':len(self.songs),'examples':len(self.items),
            'preparation_cache':self.preparation_cache_stats,
            'elapsed_sec':round(time.monotonic()-started,2)},ensure_ascii=False),flush=True)
        if not self.items: raise ValueError(f'No usable song-form examples: {self.audit["skipped"][:4]}')

    def __len__(self): return len(self.items)

    def preload_audio(self, report):
        from maichart.form.audio_residency import preload_audio
        return preload_audio(self.audio_store, [song[0].audio.path for song in self.songs], report=report)

    def cache_stats(self):
        return {'audio':self.audio_store.stats(),'samples':self.sample_store.stats()}

    def stats(self): return self.cache_stats()

    def __getitem__(self,index):
        item=self.items[index]
        if len(item) == 4:
            song_index,span_index,tokens,task=item; local_weight=1.0
        else:
            song_index,span_index,tokens,task,local_weight=item
        cache_key=index if task == 0 else None
        sample_store=getattr(self,'sample_store',None)
        if cache_key is not None and sample_store is not None:
            cached=sample_store.get(cache_key)
            if cached is not None: return cached
        song_record=self.songs[song_index]
        if len(song_record) >= 8:
            song,points,level,weight,state_index,condition,excluded,record=song_record
        else:
            song,points,level,weight,state_index=song_record[:5]
            condition=GenerationCondition(level=level,radar=GENERATION_DEFAULTS,style=GENERATION_DEFAULTS)
            excluded=[]; record=None
        span=song.spans[span_index]
        point_beats=getattr(self,'_point_beats',None)
        beats=point_beats[song_index] if point_beats is not None else [p.beat for p in points]
        lo=bisect_left(beats,span.start);hi=bisect_left(beats,span.end)
        history=points[:lo]
        target=points[lo:hi]
        draft=[];right=[]
        if task:
            rng=np.random.default_rng(self.seed+index)
            clean=rng.random()<.1
            selected=set() if clean else set(rng.choice(len(target),size=max(1,int(len(target)*.3)),replace=False).tolist())
            draft=[DraftPoint(p.beat,None if i in selected else p.note_text,i in selected) for i,p in enumerate(target)]
            right=points[hi:]
        result=input_example(song,span_index,history,condition=condition,codec=self.codec,task=task,
            draft=draft,right=right,max_memory_tokens=self.max_memory_tokens,state_index=state_index)
        ref_index=result['meta']['reference_source']
        reference=[]
        if ref_index is not None and result['motif'].numel()>1:
            # The auxiliary target must see the same phase-aligned, clipped
            # reference as the encoder, not a separate zero-offset projection.
            reference=self.codec.decode(result['motif'].tolist(),origin=span.start)
        # This blueprint contains positions only. Every chart configuration is
        # masked, so training cannot reveal future note types or note text.
        if song.audio_profile == 'rational_audio_queries_v1':
            result['skeleton']=torch.tensor([self.codec.pad_id])
        else:
            skeleton=[DraftPoint(p.beat,None,True) for p in points[lo:]]
            result['skeleton']=torch.tensor(self.codec.bounded_memory(
                skeleton,origin=span.start,max_tokens=self.max_memory_tokens,newest=False))
        result.update(decoder_input=torch.tensor(tokens[:-1]),target=torch.tensor(tokens[1:]),
                      relation_target=relation_target(target,reference,span),
                      relation_valid=torch.tensor(ref_index is not None),weight=torch.tensor(weight*local_weight))
        if cache_key is not None and sample_store is not None: sample_store.put(cache_key,result)
        return result


def collate_form(items):
    result={}
    padded=('audio','form','history','motif','draft','right','skeleton','action_state','decoder_input','target','load_curve','load_curve_times_sec','load_curve_condition_mask')
    for key in padded:
        if key not in items[0]: continue
        result[key]=torch.nn.utils.rnn.pad_sequence([i[key] for i in items],batch_first=True)
        if key in ('audio','form','load_curve','load_curve_times_sec','load_curve_condition_mask'):
            result[key+'_padding_mask']=torch.arange(result[key].size(1))[None,:]>=torch.tensor([len(i[key]) for i in items])[:,None]
    result['action_state_padding_mask']=torch.nn.utils.rnn.pad_sequence([i['action_state_padding_mask'] for i in items],batch_first=True,padding_value=True)
    for key in ('condition','condition_null','task','relation','memory_strength','relation_target','relation_valid','weight'):
        if key in items[0]: result[key]=torch.stack([i[key] for i in items])
    result['meta']=[i['meta'] for i in items]
    return result


def _region_weight(start_sec, end_sec, record):
    """Duration-weight annotation multipliers over one computational span."""
    regions=(record or {}).get('regions',[])
    if not regions:
        return 1.0
    total=max(float(end_sec)-float(start_sec),1e-9)
    weighted=0.0
    covered=0.0
    for region in regions:
        lo=max(float(start_sec),float(region['start_sec']))
        hi=min(float(end_sec),float(region['end_sec']))
        if hi<=lo:
            continue
        duration=hi-lo
        weighted+=duration*float(region.get('weight',1.0))
        covered+=duration
    return (weighted + max(0.0,total-covered))/total
