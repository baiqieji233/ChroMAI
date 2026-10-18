"""Whole-song stream examples built on the existing rational audio contract."""
from __future__ import annotations

from dataclasses import dataclass,replace
from fractions import Fraction
from collections import OrderedDict
from functools import lru_cache
import hashlib
import json
import math
import time
import random
from collections import Counter
from bisect import bisect_left

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


TRAINING_SEMANTICS = 'chart_stream_training_v5_grounded_context'


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
                 limit_charts=0, quality_profile=MUSICAL_EVIDENCE, quality_enabled=True,
                 grounding_config=None,history_note_dropout=0.,prefix_note_dropout=0.,seed=0):
        self.codec = RationalCodec()
        self.grounding_config=grounding_config
        self.grounded=bool(grounding_config is not None and grounding_config.grounding_profile!='none')
        self.history_note_dropout=float(history_note_dropout);self.prefix_note_dropout=float(prefix_note_dropout)
        self.seed,self.epoch=int(seed),0
        self.cache_dir=cache_dir
        # Split the user budget between acoustic features and final queries.
        # The shared query builder also has its own fixed bounded scratch LRU.
        query_fraction=.75 if self.grounded else .25
        self.query_budget = int(float(audio_cache_gb)*1024**3*query_fraction)
        # The base preparer only uses max_tokens to reject a target before we
        # split it. A stream must retain long songs, so this is deliberately a
        # preparation ceiling rather than a decoder context limit.
        self.base = FormDataset(manifest, cache_dir=cache_dir, max_tokens=1_000_000,
            max_memory_tokens=max_history_tokens, max_region_seconds=max_region_seconds,
            preparation_workers=preparation_workers, audio_cache_gb=float(audio_cache_gb)*(1-query_fraction),
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
        self._chart_ids=[];self._chart_positions=[];self._probe_beats=[];self._probe_indices=[]
        started=time.monotonic()
        print('stream_prepare '+json.dumps({'stage':'targets','completed':0,'total':len(self.base.songs)}),flush=True)
        for index, record in enumerate(self.base.songs):
            song, points, level, weight, state, condition, excluded, annotation = record
            chart = self.base.audit['charts'][index]
            ids=self.codec.encode(points,origin=Fraction(0)) if points else [self.codec.bos_id,self.codec.eos_id]
            positions=[i for i,t in enumerate(ids) if t==self.codec.at_id]
            self._chart_ids.append(ids);self._chart_positions.append(positions)
            probes=sorted({Fraction(0),*(p.beat for p in points)})
            self._probe_beats.append(probes);self._probe_indices.append({b:i for i,b in enumerate(probes)})
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
        if self.grounded:
            from dataclasses import asdict
            identity['grounding']={k:v for k,v in asdict(grounding_config).items() if k.startswith(('grounding','detail_','local_','archive_','alignment_'))}
        identity.update(history_note_dropout=self.history_note_dropout,prefix_note_dropout=self.prefix_note_dropout)
        self.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        self._chunk_counts=Counter(c.chart_index for c in self.chunks)

    def set_epoch(self,epoch):self.epoch=int(epoch)

    def _cached_query(self,key,builder):
        if key in self._audio_queries:
            self.query_hits+=1;self._audio_queries.move_to_end(key)
            return self._audio_queries[key][0]
        self.query_misses+=1;query=builder()
        size=query[0].numel()*query[0].element_size()+query[1].nbytes+len(query[2])*64
        if size<=self.query_budget:
            while self._audio_queries and self.query_bytes+size>self.query_budget:
                _,(_,removed)=self._audio_queries.popitem(last=False);self.query_bytes-=removed
            self._audio_queries[key]=(query,size);self.query_bytes+=size
        return query

    def _queries(self, index, song):
        key = self._query_keys[index]
        return self._cached_query(key,lambda:planner_audio_queries(song))

    def _grounded_item(self,chunk,item,song,points,annotation,rng):
        from .grounding import clock_inputs,history_clock,exact_probe_cache,detail_queries,time_cell_weights,masked_note_inputs,action_summaries
        from maichart.form.alignment import build_alignment_targets
        from maichart.form.data import _LazyAudio
        if isinstance(song.audio,_LazyAudio):song=replace(song,audio=song.audio.store.get(song.audio.path))
        cfg=self.grounding_config;index=chunk.chart_index
        detail,times,beats=self._cached_query(('detail',self._query_keys[index],cfg.detail_audio_nodes),
            lambda:detail_queries(song,cfg.detail_audio_nodes))
        probes=self._probe_beats[index]
        def probe_builder():
            values,_=exact_probe_cache(song,probes,self.cache_dir)
            return values,np.asarray([seconds_at_beat(b,song.tempo,song.first) for b in probes],dtype=np.float64),tuple(probes)
        exact,_,_=self._cached_query(('probes',index),probe_builder)
        prior=points[chunk.start_event-1].beat if chunk.start_event else Fraction(0)
        before=points[chunk.start_event-2].beat if chunk.start_event>1 else Fraction(0)
        clocks=clock_inputs(chunk.decoder_input,self.codec,song,previous=prior,previous_previous=before,committed=chunk.start_event)
        query_beats=clocks.pop('query_beats');pending=clocks.pop('query_pending')
        item.update(clocks)
        item['query_actions']=action_summaries(self.base.songs[index][4],query_beats,pending,clocks['query_committed'])
        item['query_audio']=exact[[self._probe_indices[index][b] for b in query_beats]]
        positions=self._chart_positions[index]
        token_start=positions[chunk.start_event] if chunk.start_event<len(points) else len(self._chart_ids[index])-1
        history_start=max(0,token_start-self.max_history_tokens)
        item['history_clock']=history_clock(self._chart_ids[index],history_start,token_start,positions,points,self.codec,song) if chunk.history else torch.zeros(0,20)
        item.update(detail_audio=detail,detail_times_sec=torch.as_tensor(times,dtype=torch.float64),
            detail_audio_padding_mask=torch.as_tensor((times<0)|(times>=song.audio.duration_sec)),
            detail_time_weight=time_cell_weights(times,max(0.,song.first),song.audio.duration_sec),
            curve_time_weight=time_cell_weights(item['audio_times_sec'].numpy(),max(0.,song.first),song.audio.duration_sec))
        item['curve_example_weight']=torch.tensor(len(self.chunks)/max(1,len(self._chunk_counts))/self._chunk_counts[index])
        # Only complete events older than the detailed history enter archives.
        archive_count=bisect_left(positions,history_start)
        selected=list(range(archive_count))
        if len(selected)>cfg.archive_events:
            selected=np.linspace(0,len(selected)-1,cfg.archive_events,dtype=int).tolist()
        item['archive_cells']=[[self.codec.ids['N:'+c] for c in points[i].note_text] for i in selected]
        item['archive_audio']=exact[[self._probe_indices[index][points[i].beat] for i in selected]]
        item['archive_beats']=torch.tensor([float(points[i].beat) for i in selected],dtype=torch.float64)
        item['archive_seconds']=torch.tensor([float(seconds_at_beat(points[i].beat,song.tempo,song.first)) for i in selected],dtype=torch.float64)
        item['archive_event_index']=torch.tensor(selected,dtype=torch.long)
        item['archive_padding_mask']=torch.zeros(len(selected),dtype=torch.bool)
        # Unknown history is explicit MASK, never invented wrong configurations.
        history,_=masked_note_inputs(chunk.history,self.codec,rng,self.history_note_dropout)
        for cell in item['archive_cells']:
            if rng.random()<self.history_note_dropout:cell[:]=[self.codec.mask_id]*len(cell)
        decoder,quality=masked_note_inputs(chunk.decoder_input,self.codec,rng,self.prefix_note_dropout,
            targets=chunk.target,quality=chunk.target_quality)
        item['decoder_input']=torch.tensor(decoder,dtype=torch.long)
        item['history']=torch.tensor(history,dtype=torch.long)
        item['target_quality']=torch.tensor(quality,dtype=torch.float32)
        event_rows=[i for i,t in enumerate(chunk.target) if t==self.codec.at_id]
        event_points=points[chunk.start_event:chunk.start_event+len(event_rows)]
        stamps=np.asarray([seconds_at_beat(p.beat,song.tempo,song.first) for p in event_points],dtype=np.float64)
        right=np.searchsorted(times,stamps).clip(0,len(times)-1);left=(right-1).clip(0)
        nearest=np.where(abs(times[left]-stamps)<=abs(times[right]-stamps),left,right)
        item['alignment_event_indices']=torch.tensor(event_rows,dtype=torch.long)
        item['alignment_audio_index']=torch.as_tensor(nearest,dtype=torch.long)
        item['alignment_event_mask']=torch.tensor([bool(abs(times[j]-s)<=cfg.alignment_tolerance_ms/1000. and s>=0 and quality[row]>0)
            for j,s,row in zip(nearest,stamps,event_rows)],dtype=torch.bool)
        item['alignment_event_weight']=item['target_quality'][event_rows]
        from .quality import region_weight_at
        def supervision_builder():
            alignment=build_alignment_targets(points,song.tempo,song.first,times,tolerance_ms=cfg.alignment_tolerance_ms)
            weights=np.asarray([region_weight_at(float(t),annotation) for t in times],dtype=np.float32)
            # An excluded event must not label nearby evidence in an adjacent
            # included region, or silently become a negative example there.
            for point in points:
                second=float(seconds_at_beat(point.beat,song.tempo,song.first))
                if region_weight_at(second,annotation)<=0:
                    weights[np.abs(times-second)<=cfg.alignment_tolerance_ms/1000.]=0.
            values=torch.stack((alignment['labels'],alignment['mask'].float(),torch.from_numpy(weights)),1)
            return values,np.empty(0,dtype=np.float64),()
        supervised,_,_=self._cached_query(('alignment',index,cfg.detail_audio_nodes,cfg.alignment_tolerance_ms),supervision_builder)
        item['position_alignment_target']=supervised[:,0]
        item['alignment_audio_quality']=supervised[:,2]
        item['position_alignment_mask']=supervised[:,1].bool()&~item['detail_audio_padding_mask']&(supervised[:,2]>0)
        return item

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
        item={'audio': audio, 'audio_times_sec': torch.as_tensor(times, dtype=torch.float64),
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
        if self.grounded:
            rng=random.Random(self.seed+self.epoch*1_000_003+index*7919)
            item=self._grounded_item(chunk,item,song,points,annotation,rng)
        return item


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
    if 'decoder_clock' in items[0]:
        for key in ('decoder_clock','history_clock','query_clock','query_seconds','query_audio','query_committed','query_actions',
                    'decoder_query_index','detail_audio','detail_times_sec','curve_time_weight','detail_time_weight',
                    'archive_audio','archive_beats','archive_seconds','archive_event_index',
                    'alignment_event_indices','alignment_audio_index','alignment_event_mask','alignment_event_weight',
                    'position_alignment_target','position_alignment_mask','alignment_audio_quality'):
            result[key]=pad_sequence([v[key] for v in items],batch_first=True,
                                     padding_value=(float('inf') if key=='detail_times_sec' else
                                                    -1 if key=='alignment_event_indices' else 0))
        for key in ('detail_audio_padding_mask','archive_padding_mask'):
            result[key]=pad_sequence([v[key] for v in items],batch_first=True,padding_value=True)
        result['curve_example_weight']=torch.stack([v['curve_example_weight'] for v in items])
        tokens=[];owners=[];positions=[];inverse=[]
        width=result['archive_padding_mask'].size(1)
        for row,item in enumerate(items):
            for event,cell in enumerate(item['archive_cells']):
                tokens.extend(cell);owners.extend([row*width+event]*len(cell))
                positions.extend((i+.5)/len(cell) for i in range(len(cell)));inverse.extend([1/len(cell)]*len(cell))
        result.update(archive_flat_tokens=torch.tensor(tokens,dtype=torch.long),
            archive_token_event=torch.tensor(owners,dtype=torch.long),
            archive_token_position=torch.tensor(positions,dtype=torch.float32),
            archive_token_inverse_length=torch.tensor(inverse,dtype=torch.float32))
    return result
