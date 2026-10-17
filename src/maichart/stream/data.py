"""Whole-song stream examples built on the existing rational audio contract."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np
import torch
from torch.utils.data import Dataset

from maichart.event_timing import EventPoint
from maichart.form.codec import RationalCodec
from maichart.form.data import FormDataset, _form_descriptor
from maichart.form.rational_audio import planner_audio_queries
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


def _plan_for_event(point, previous, text):
    delta = max(Fraction(point.beat) - Fraction(previous), Fraction(0))
    chars = str(text)
    return (min(float(delta), 8.) / 8.,
            math.sin(float(point.beat) * math.pi), math.cos(float(point.beat) * math.pi),
            float('/' in chars), float('-' in chars or '>' in chars or '<' in chars),
            float('h' in chars), float(len(chars.split('/')) > 1),
            min(float(delta), 4.) / 4.)


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
        for index, record in enumerate(self.base.songs):
            song, points, level, weight, state, condition, excluded, annotation = record
            if not points or weight <= 0:
                continue
            ids = self.codec.encode(points, origin=Fraction(0))
            # Start later chunks at an event boundary so a configuration is not
            # cut in half. The carried token history still makes the stream causal.
            starts = [0]
            cursor = self.max_tokens
            at_id = self.codec.at_id
            while cursor < len(ids):
                candidates = [i for i in range(cursor, max(1, cursor-self.max_tokens//4), -1)
                              if ids[i] == at_id]
                start = min(candidates) if candidates else cursor
                starts.append(start); cursor = start + self.max_tokens
            for chunk_index, start in enumerate(starts):
                end = min(len(ids), start + self.max_tokens)
                if end < len(ids):
                    candidates = [i for i in range(end, start, -1) if ids[i] == at_id]
                    if candidates: end = max(candidates)
                if end <= start + 1: continue
                decoder = ids[start:end-1] if start else ids[:end-1]
                target = ids[start+1:end] if start else ids[1:end]
                history_start = max(0, start-self.max_history_tokens)
                history = ids[history_start:start]
                # Map event starts to local target rows. Labels supervise the
                # next-event intent, not every character in a configuration.
                labels=[]; masks=[]
                for row, token in enumerate(target):
                    if token == self.codec.at_id:
                        global_index = start + 1 + row
                        event_index = ids[:global_index].count(self.codec.at_id)
                        point = points[min(len(points) - 1, event_index)]
                        previous = Fraction(0) if event_index == 0 else points[event_index - 1].beat
                        labels.append(_plan_for_event(point, previous, point.note_text))
                        masks.append(True)
                    else:
                        labels.append((0.,)*8); masks.append(False)
                start_event = ids[:start].count(self.codec.at_id)
                self.chunks.append(StreamChunk(index, chunk_index, tuple(decoder), tuple(target),
                    tuple(history), tuple(labels), tuple(masks), end >= len(ids), start_event))
            self.songs.append(record)

    def __len__(self): return len(self.chunks)
    def __getitem__(self, index):
        chunk = self.chunks[index]
        song, points, level, weight, state, condition, excluded, annotation = self.base.songs[chunk.chart_index]
        audio, times, _ = planner_audio_queries(song)
        boundary = points[min(chunk.start_event, len(points)-1)].beat if points else Fraction(0)
        action, action_mask, _ = state.at(boundary)
        return {'audio': audio, 'audio_times_sec': torch.as_tensor(times, dtype=torch.float64),
                'form': torch.from_numpy(_form_descriptor(song, 0)),
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
    for key in ('audio','decoder_input','target','history','plan_labels','plan_mask','action_state'):
        values=[item[key] for item in items]
        result[key]=pad_sequence(values,batch_first=True,padding_value=0)
        result[key+'_padding_mask']=torch.arange(result[key].size(1))[None] >= torch.tensor([len(v) for v in values])[:,None]
    for key in ('condition','final','chart_index','chunk_index'):
        result[key]=torch.stack([item[key] if torch.is_tensor(item[key]) else torch.tensor(item[key]) for item in items])
    result['audio_times_sec']=pad_sequence([item['audio_times_sec'] for item in items],batch_first=True)
    result['audio_padding_mask']=torch.arange(result['audio'].size(1))[None] >= torch.tensor([len(v['audio']) for v in items])[:,None]
    result['action_state_padding_mask']=pad_sequence([item['action_state_padding_mask'] for item in items],batch_first=True,padding_value=True)
    return result
