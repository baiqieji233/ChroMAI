"""Whole-song stream examples built on the existing rational audio contract."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from collections import OrderedDict
from functools import lru_cache
import hashlib
import json
import math
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from maichart.event_timing import EventPoint, seconds_at_beat
from maichart.form.codec import RationalCodec
from maichart.form.data import FormDataset
from maichart.form.rational_audio import planner_audio_queries
from maichart.form.curve import resample_load_curve
from maichart.form.quality_profile import MUSICAL_EVIDENCE
from .context import prefix_action_state
from .quality import event_weight, interval_weight, curve_weights


@dataclass(frozen=True)
class StreamChunk:
    chart_index: int
    chunk_index: int
    decoder_input: tuple[int, ...]
    target: tuple[int, ...]
    history: tuple[int, ...]
    plan_labels: tuple[tuple[float, ...], ...]
    plan_mask: tuple[bool, ...]
    final: bool
    start_event: int
    target_quality: tuple[float, ...] = ()


TRAINING_SEMANTICS = 'chart_stream_training_v4_quality_memory'


def _target_ranges(ids, max_tokens, at_id, eos_id):
    """Partition every target once, without splitting a cell or inventing EOS.

    BOS is input only. EOS is its own final unit, so an exact-fit last cell
    cannot silently lose its end-of-song target.
    """
    if max_tokens < 1 or ids[-1] != eos_id:
        raise ValueError('Invalid continuous stream target or token budget')
    boundaries = [i for i, token in enumerate(ids) if token == at_id]
    boundaries += [len(ids)-1, len(ids)]
    if not boundaries or boundaries[0] != 1:
        raise ValueError('Continuous stream targets must start with an event')
    ranges = []
    start = end = 1
    for left, right in zip(boundaries, boundaries[1:]):
        if right-left > max_tokens:
            raise ValueError(f'单个配置需要 {right-left} token，超过流上下文上限 {max_tokens}')
        if right-start > max_tokens:
            ranges.append((start, end))
            start = left
        end = right
    ranges.append((start, end))
    return ranges


@lru_cache(maxsize=65536)
def _head_labels(text):
    from maichart.form.type_guidance import _counts
    counts = _counts(text)
    if counts is None:
        raise ValueError(f'Cannot supervise stream intent from unparsed configuration: {text!r}')
    chord = float(counts.sum() > 1)
    return chord, float(counts[1] > 0), float(counts[2] + counts[4] > 0), chord


def _plan_for_event(point, previous, text, density=0.):
    delta = max(Fraction(point.beat) - Fraction(previous), Fraction(0))
    return (min(float(delta), 8.) / 8.,
            math.sin(float(point.beat) * math.pi), math.cos(float(point.beat) * math.pi),
            *_head_labels(str(text)),
            float(np.clip(density, 0., 1.)))


class StreamDataset(Dataset):
    """Each item is a continuation chunk; chunk boundaries are not musical EOS."""
    def __init__(self, manifest, *, cache_dir, max_tokens=1024, max_history_tokens=256,
                 max_region_seconds=15., preparation_workers=1, audio_cache_gb=1.,
                 limit_charts=0, quality_profile=MUSICAL_EVIDENCE, quality_enabled=True):
        self.codec = RationalCodec()
        # Split the user budget between acoustic features and final queries.
        # The shared query builder also has its own fixed bounded scratch LRU.
        self.query_budget = int(float(audio_cache_gb)*1024**3/4)
        # The base preparer only uses max_tokens to reject a target before we
        # split it. A stream must retain long songs, so this is deliberately a
        # preparation ceiling rather than a decoder context limit.
        self.base = FormDataset(manifest, cache_dir=cache_dir, max_tokens=1_000_000,
            max_memory_tokens=max_history_tokens, max_region_seconds=max_region_seconds,
            preparation_workers=preparation_workers, audio_cache_gb=float(audio_cache_gb)*.75,
            audio_profile='rational_audio_queries_v1', quality_profile=quality_profile,
            rhythm_geometry_profile='none', limit_charts=limit_charts, preserve_excluded_context=True,
            quality_enabled=quality_enabled)
        self.max_tokens = int(max_tokens)
        self.max_history_tokens = int(max_history_tokens)
        self.songs = []
        self.chunks = []
        self.training_semantics = TRAINING_SEMANTICS
        self._audio_queries = OrderedDict()
        self.query_bytes = self.query_hits = self.query_misses = 0
        self._query_keys = []
        started=time.monotonic()
        print('stream_prepare '+json.dumps({'stage':'targets','completed':0,'total':len(self.base.songs)}),flush=True)
        for index, record in enumerate(self.base.songs):
            song, points, level, weight, state, condition, excluded, annotation = record
            chart = self.base.audit['charts'][index]
            # Different difficulties share queries only with identical clocks
            # and evidence contracts. No chart content enters audio queries.
            self._query_keys.append(json.dumps({
                'audio': chart.get('audio_sha256') or str(getattr(song.audio, 'path', index)),
                'first': song.first, 'tempo': [(str(t.beat), t.bpm) for t in song.tempo],
                'meters': [m.to_dict() for m in song.meters],
                'start': str(song.spans[0].start), 'end': str(song.spans[-1].end),
                'quality_profile': song.quality_profile}, sort_keys=True))
            if not points or weight <= 0:
                continue
            ids = self.codec.encode(points, origin=Fraction(0))
            event_at = {position: event for event, position in enumerate(
                i for i, token in enumerate(ids) if token == self.codec.at_id)}
            event_before = 0
            seconds = [float(seconds_at_beat(p.beat, song.tempo, song.first)) for p in points]
            event_weights = [event_weight(float(song.first) if i == 0 else seconds[i-1], second, annotation)
                             for i, second in enumerate(seconds)]
            intent_quality = curve_weights(song, seconds, annotation)
            eos_weight = (interval_weight(seconds[-1], float(song.audio.duration_sec), annotation)
                          if (annotation or {}).get('regions') else 1.)
            all_weights = [0.]
            for point, local_weight in zip(points, event_weights):
                all_weights.extend([local_weight]*len(self.codec.cell(point)))
            all_weights.append(eos_weight)
            for chunk_index, (start, end) in enumerate(_target_ranges(
                    ids, self.max_tokens, self.codec.at_id, self.codec.eos_id)):
                decoder = [self.codec.bos_id] + ids[start:end-1]
                target = ids[start:end]
                history_start = max(0, start-self.max_history_tokens)
                history = ids[history_start:start] if start > 1 else []
                # Map event starts to local target rows. Labels supervise the
                # next-event intent, not every character in a configuration.
                labels=[]; masks=[]
                for row, token in enumerate(target):
                    if token == self.codec.at_id:
                        global_index = start + row
                        event_index = event_at[global_index]
                        point = points[min(len(points) - 1, event_index)]
                        previous = Fraction(0) if event_index == 0 else points[event_index - 1].beat
                        event_second = float(seconds_at_beat(point.beat, song.tempo, song.first) - song.first)
                        density = float(np.interp(event_second, song.load_curve_times_sec,
                            song.load_curve[:, 0], left=0., right=0.))
                        labels.append(_plan_for_event(point, previous, point.note_text, density))
                        masks.append(bool(intent_quality[event_index] > 0))
                    else:
                        labels.append((0.,)*8); masks.append(False)
                start_event = event_before
                event_before += target.count(self.codec.at_id)
                if any(value > 0 for value in all_weights[start:end]):
                    self.chunks.append(StreamChunk(index, chunk_index, tuple(decoder), tuple(target),
                        tuple(history), tuple(labels), tuple(masks), end >= len(ids), start_event,
                        tuple(all_weights[start:end])))
            self.songs.append(record)
            if (index+1)%25==0 or index+1==len(self.base.songs):
                print('stream_prepare '+json.dumps({'stage':'targets','completed':index+1,
                    'total':len(self.base.songs),'chunks':len(self.chunks),
                    'elapsed_sec':round(time.monotonic()-started,2)}),flush=True)
        # Content and annotation identity, not cache-hit counts or disk mtime.
        identity = {'semantics': TRAINING_SEMANTICS, 'max_tokens': self.max_tokens,
            'history': self.max_history_tokens, 'charts': self.base.audit['charts'],
            'quality_enabled':bool(quality_enabled),
            'chunks': len(self.chunks)}
        self.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def _queries(self, index, song):
        key = self._query_keys[index]
        if key in self._audio_queries:
            self.query_hits += 1
            self._audio_queries.move_to_end(key)
            return self._audio_queries[key][0]
        self.query_misses += 1
        query = planner_audio_queries(song)
        size = query[0].numel()*query[0].element_size() + query[1].nbytes + len(query[2])*64
        if size <= self.query_budget:
            while self._audio_queries and self.query_bytes + size > self.query_budget:
                _, (_, removed) = self._audio_queries.popitem(last=False)
                self.query_bytes -= removed
            self._audio_queries[key] = (query, size)
            self.query_bytes += size
        return query

    def cache_stats(self):
        return {'audio': self.base.audio_store.stats(), 'stream_queries': {
            'bytes': self.query_bytes, 'budget_bytes': self.query_budget,
            'entries': len(self._audio_queries), 'hits': self.query_hits, 'misses': self.query_misses}}

    def __len__(self): return len(self.chunks)
    def __getitem__(self, index):
        chunk = self.chunks[index]
        song, points, level, weight, state, condition, excluded, annotation = self.base.songs[chunk.chart_index]
        audio, times, _ = self._queries(chunk.chart_index, song)
        action, action_mask, _ = prefix_action_state(state, points, chunk.start_event)
        return {'audio': audio, 'audio_times_sec': torch.as_tensor(times, dtype=torch.float64),
                'curve_target': torch.from_numpy(resample_load_curve(
                    song.load_curve_times_sec, song.load_curve,
                    np.asarray(times, dtype=np.float64) - float(song.first))),
                'condition': condition.to_vector(),
                'quality_weight': torch.tensor(float(weight)),
                'target_quality': torch.tensor(chunk.target_quality, dtype=torch.float32),
                'curve_quality': torch.from_numpy(curve_weights(song, times, annotation)),
                'decoder_input': torch.tensor(chunk.decoder_input, dtype=torch.long),
                'target': torch.tensor(chunk.target, dtype=torch.long),
                'history': torch.tensor(chunk.history, dtype=torch.long),
                'plan_labels': torch.tensor(chunk.plan_labels, dtype=torch.float32),
                'plan_mask': torch.tensor(chunk.plan_mask, dtype=torch.bool),
                'action_state': action, 'action_state_padding_mask': action_mask,
                'final': torch.tensor(chunk.final), 'chart_index': chunk.chart_index,
                'chunk_index': chunk.chunk_index}


def collate_stream(items):
    from torch.nn.utils.rnn import pad_sequence
    result={}
    for key in ('audio','decoder_input','target','history','plan_labels','plan_mask','action_state','curve_target',
                'target_quality','curve_quality'):
        values=[item[key] for item in items]
        result[key]=pad_sequence(values,batch_first=True,padding_value=0)
        result[key+'_padding_mask']=torch.arange(result[key].size(1))[None] >= torch.tensor([len(v) for v in values])[:,None]
    for key in ('condition','final','chart_index','chunk_index','quality_weight'):
        result[key]=torch.stack([item[key] if torch.is_tensor(item[key]) else torch.tensor(item[key]) for item in items])
    result['audio_times_sec']=pad_sequence([item['audio_times_sec'] for item in items],batch_first=True)
    result['audio_padding_mask']=torch.arange(result['audio'].size(1))[None] >= torch.tensor([len(v['audio']) for v in items])[:,None]
    result['action_state_padding_mask']=pad_sequence([item['action_state_padding_mask'] for item in items],batch_first=True,padding_value=True)
    return result
