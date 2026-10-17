"""Whole-song stream examples built on the existing rational audio contract."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np
import torch
from torch.utils.data import Dataset

from maichart.event_timing import EventPoint, seconds_at_beat
from maichart.form.codec import RationalCodec
from maichart.form.data import FormDataset, _form_descriptor
from maichart.form.rational_audio import planner_audio_queries
from maichart.form.curve import resample_load_curve
from maichart.form.quality_profile import MUSICAL_EVIDENCE


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


TRAINING_SEMANTICS = 'chart_stream_training_v2_contiguous_targets'


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


def _plan_for_event(point, previous, text, density=0.):
    delta = max(Fraction(point.beat) - Fraction(previous), Fraction(0))
    chars = str(text)
    return (min(float(delta), 8.) / 8.,
            math.sin(float(point.beat) * math.pi), math.cos(float(point.beat) * math.pi),
            float('/' in chars), float('-' in chars or '>' in chars or '<' in chars),
            float('h' in chars), float(len(chars.split('/')) > 1),
            float(np.clip(density, 0., 1.)))


class StreamDataset(Dataset):
    """Each item is a continuation chunk; chunk boundaries are not musical EOS."""
    def __init__(self, manifest, *, cache_dir, max_tokens=1024, max_history_tokens=256,
                 max_region_seconds=15., preparation_workers=1, audio_cache_gb=1.,
                 limit_charts=0, quality_profile=MUSICAL_EVIDENCE):
        self.codec = RationalCodec()
        # The base preparer only uses max_tokens to reject a target before we
        # split it. A stream must retain long songs, so this is deliberately a
        # preparation ceiling rather than a decoder context limit.
        self.base = FormDataset(manifest, cache_dir=cache_dir, max_tokens=1_000_000,
            max_memory_tokens=max_history_tokens, max_region_seconds=max_region_seconds,
            preparation_workers=preparation_workers, audio_cache_gb=audio_cache_gb,
            audio_profile='rational_audio_queries_v1', quality_profile=quality_profile,
            rhythm_geometry_profile='none', limit_charts=limit_charts)
        self.max_tokens = int(max_tokens)
        self.max_history_tokens = int(max_history_tokens)
        self.songs = []
        self.chunks = []
        self.training_semantics = TRAINING_SEMANTICS
        self._audio_queries = {}
        for index, record in enumerate(self.base.songs):
            song, points, level, weight, state, condition, excluded, annotation = record
            if not points or weight <= 0:
                continue
            ids = self.codec.encode(points, origin=Fraction(0))
            event_at = {position: event for event, position in enumerate(
                i for i, token in enumerate(ids) if token == self.codec.at_id)}
            event_before = 0
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
                        masks.append(True)
                    else:
                        labels.append((0.,)*8); masks.append(False)
                start_event = event_before
                self.chunks.append(StreamChunk(index, chunk_index, tuple(decoder), tuple(target),
                    tuple(history), tuple(labels), tuple(masks), end >= len(ids), start_event))
                event_before += target.count(self.codec.at_id)
            self.songs.append(record)

    def __len__(self): return len(self.chunks)
    def __getitem__(self, index):
        chunk = self.chunks[index]
        song, points, level, weight, state, condition, excluded, annotation = self.base.songs[chunk.chart_index]
        if chunk.chart_index not in self._audio_queries:
            self._audio_queries[chunk.chart_index] = planner_audio_queries(song)
        audio, times, _ = self._audio_queries[chunk.chart_index]
        boundary = (points[chunk.start_event].beat if chunk.start_event < len(points)
                    else points[-1].beat+Fraction(1, 1_000_000)) if points else Fraction(0)
        action, action_mask, _ = state.at(boundary)
        return {'audio': audio, 'audio_times_sec': torch.as_tensor(times, dtype=torch.float64),
                'form': torch.from_numpy(_form_descriptor(song, 0)),
                'curve_target': torch.from_numpy(resample_load_curve(
                    song.load_curve_times_sec, song.load_curve,
                    np.asarray(times, dtype=np.float64) - float(song.first))),
                'condition': condition.to_vector(),
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
    for key in ('audio','decoder_input','target','history','plan_labels','plan_mask','action_state','curve_target'):
        values=[item[key] for item in items]
        result[key]=pad_sequence(values,batch_first=True,padding_value=0)
        result[key+'_padding_mask']=torch.arange(result[key].size(1))[None] >= torch.tensor([len(v) for v in values])[:,None]
    for key in ('condition','final','chart_index','chunk_index'):
        result[key]=torch.stack([item[key] if torch.is_tensor(item[key]) else torch.tensor(item[key]) for item in items])
    result['audio_times_sec']=pad_sequence([item['audio_times_sec'] for item in items],batch_first=True)
    result['audio_padding_mask']=torch.arange(result['audio'].size(1))[None] >= torch.tensor([len(v['audio']) for v in items])[:,None]
    result['action_state_padding_mask']=pad_sequence([item['action_state_padding_mask'] for item in items],batch_first=True,padding_value=True)
    return result
