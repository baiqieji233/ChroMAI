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
import threading
from types import SimpleNamespace
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
    # Event used to build the causal decoder context. With overlap training
    # this can precede the first supervised event in the chunk.
    start_event: int
    supervised_event: int = 0
    target_offset: int = 0
    target_quality: tuple[float, ...] = ()
    cycle_exit_mask: tuple[bool, ...] = ()


# v7 records the corrected contract where warm replay tokens are actually
# present in decoder_input. v6 checkpoints were labelled overlap_v1 before
# this was true and must not be resumed as if they had learned the contract.
# v8: plan label 6 is Touch (was a duplicate chord flag), masked cells keep
# timing supervision, structural cycle exits, local next-onset alignment,
# kinematic/phrase grounding and game-mirror augmentation.
# v9: sustain release buttons in kinematics (Slide end / Hold release) and
# Break/EX as plain attribute channels of the next-event plan target.
# v10: multi-scale local audio lookahead spanning local_audio_seconds,
# clock interval de-echo on masked/dropped events, and rhythm cycle-exit targets.
# v11: Slides in flight are visible to the causal hand kinematics (destination,
# origin phase, active count, time to arrival, and effective hand step).
# v12: planner geometry uses symmetry-invariant relative hand motion instead of
# absolute song beat phase. Old v11 checkpoints remain generation-loadable but
# must be finetuned under this new label contract rather than resumed.
# v13 bounds the extra per-token cycle-exit weight by the main CE denominator.
TRAINING_SEMANTICS = 'chart_stream_training_v14_causal_motor_masks'


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
    # HEAD_TYPES = tap, slide, hold, touch, touch_hold.
    return chord, float(counts[1] > 0), float(counts[2] + counts[4] > 0), float(counts[3] + counts[4] > 0)


@lru_cache(maxsize=65536)
def _attribute_labels(text):
    """Break/EX are note attributes, supervised exactly like the head types."""
    from maichart.form.type_guidance import _break_count, _ex_count
    return float(_break_count(text) > 0), float(_ex_count(text) > 0)


# delta beat, relative motion phase (2), chord, slide, hold, touch, density,
# break, ex.
PLAN_DIM = 10
LEGACY_PLAN_DIM = 8


def _motion_phase(previous_text, text):
    """Return reflection-invariant relative hand motion phase.

    Signed angular direction would change under the training mirrors and create
    contradictory labels. The absolute shortest angular displacement preserves
    movement magnitude while remaining invariant under left/right reflection.
    """
    if not previous_text:
        return 0., 1.
    from .symmetry import _angle, event_lanes
    previous_angle = _angle(event_lanes(str(previous_text)))
    current_angle = _angle(event_lanes(str(text)))
    if previous_angle is None or current_angle is None:
        return 0., 1.
    delta = abs(math.atan2(math.sin(current_angle - previous_angle),
                           math.cos(current_angle - previous_angle)))
    return math.sin(delta), math.cos(delta)


def _plan_for_event(point, previous, text, density=0., previous_text=None):
    delta = max(Fraction(point.beat) - Fraction(previous), Fraction(0))
    motion_sin, motion_cos = _motion_phase(previous_text, text)
    return (min(float(delta), 8.) / 8.,
            motion_sin, motion_cos,
            *_head_labels(str(text)),
            float(np.clip(density, 0., 1.)),
            *_attribute_labels(str(text)))


@lru_cache(maxsize=262144)
def _mirror_ok(text, transform):
    from maichart.form.type_guidance import _counts
    from .symmetry import transform_text
    mirrored = transform_text(text, transform)
    before, after = _counts(text), _counts(mirrored)
    if before is None or after is None or not (before == after).all():
        return False
    if not any(c in mirrored for c in '-<>^pqszvwV'):
        return True
    try:
        from maichart.generation.muri.parser import parse_markers
        _, errors, _ = parse_markers(f'(120){{4}}{mirrored},', first=0.)
    except (ValueError, TypeError, KeyError, OverflowError, IndexError):
        return False
    return not errors


def _valid_symmetries(points, mode='mirror4'):
    """Game mirror options valid for every cell of a chart (identity first)."""
    if mode == 'none':
        return ('identity',)
    from .symmetry import TRANSFORMS
    candidates = ('mirror_lr',) if mode == 'mirror_lr' else TRANSFORMS[1:]
    texts = {str(p.note_text) for p in points}
    return ('identity',) + tuple(t for t in candidates if all(_mirror_ok(text, t) for text in texts))


class StreamDataset(Dataset):
    """Each item is a continuation chunk; chunk boundaries are not musical EOS."""
    def __init__(self, manifest, *, cache_dir, max_tokens=1024, max_history_tokens=256,
                 max_region_seconds=15., preparation_workers=1, audio_cache_gb=1.,
                 limit_charts=0, quality_profile=MUSICAL_EVIDENCE, quality_enabled=True,
                 grounding_config=None,history_note_dropout=0.,prefix_note_dropout=0.,seed=0,
                 input_cache_mode='mapped',structure_cache_mb=256,prefix_mask_span_events=1,
                 symmetry_augmentation='none'):
        self.codec = RationalCodec()
        if symmetry_augmentation not in {'none','mirror_lr','mirror4'}:raise ValueError('Unknown symmetry_augmentation')
        self.symmetry_augmentation=symmetry_augmentation
        self._symmetries=[];self._event_lanes=[];self._event_trajectories=[];self._oriented_ids={}
        self.symmetry_stats=Counter()
        self.grounding_config=grounding_config
        self.grounded=bool(grounding_config is not None and grounding_config.grounding_profile!='none')
        self.history_note_dropout=float(history_note_dropout);self.prefix_note_dropout=float(prefix_note_dropout)
        self.prefix_mask_span_events=int(prefix_mask_span_events)
        from .continuation import replay_budget
        self.replay_tokens=replay_budget(grounding_config) if grounding_config is not None else 0
        self.seed,self.epoch=int(seed),0
        self.cache_dir=cache_dir
        if input_cache_mode not in {'mapped','legacy'}:raise ValueError('Unknown input_cache_mode')
        from .input_cache import MappedInputCache,fingerprint
        self.input_cache=MappedInputCache(cache_dir) if input_cache_mode=='mapped' else None
        self._durations={};self._chart_identities=[]
        self._structures=OrderedDict();self.structure_bytes=0
        self.structure_budget=int(structure_cache_mb*1024**2)
        self.structure_hits=self.structure_misses=0
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
        self._query_lock=threading.RLock()
        self._query_build_locks=[threading.Lock() for _ in range(64)]
        self.query_prepare_seconds=Counter();self.query_prepare_counts=Counter()
        self.query_bytes = self.query_hits = self.query_misses = 0
        self._query_keys = []
        self._chart_ids=[];self._chart_positions=[];self._probe_beats=[];self._probe_indices=[]
        self._point_seconds=[];self._point_values=[];self._event_note_ids=[]
        started=time.monotonic()
        print('stream_prepare '+json.dumps({'stage':'targets','completed':0,'total':len(self.base.songs)}),flush=True)
        for index, record in enumerate(self.base.songs):
            song, points, level, weight, state, condition, excluded, annotation = record
            chart = self.base.audit['charts'][index]
            ids=self.codec.encode(points,origin=Fraction(0)) if points else [self.codec.bos_id,self.codec.eos_id]
            positions=[i for i,t in enumerate(ids) if t==self.codec.at_id]
            self._chart_ids.append(ids);self._chart_positions.append(positions)
            from maichart.tempo_clock import TempoClock
            timeline=TempoClock(song.tempo,song.first)
            self._point_seconds.append(np.asarray([timeline.at(p.beat) for p in points],dtype=np.float64))
            self._point_values.append(np.asarray([float(p.beat) for p in points],dtype=np.float64))
            self._event_note_ids.append([tuple(self.codec.ids['N:'+c] for c in p.note_text) for p in points])
            from .symmetry import event_lanes,sustain_trajectories
            self._event_lanes.append([event_lanes(p.note_text) for p in points])
            self._event_trajectories.append(sustain_trajectories(getattr(state,'markers',()),self._point_seconds[-1]))
            self._symmetries.append(_valid_symmetries(points,self.symmetry_augmentation))
            self.symmetry_stats[len(self._symmetries[-1])]+=1
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
            # Audio probes/selection labels did not change with continuation
            # windows. Keep their existing identity and avoid rebuilding GBs
            # of identical native audio evidence when upgrading supervision.
            self._chart_identities.append(fingerprint({'semantics':'chart_stream_training_v5_grounded_context',
                'query':self._query_keys[-1],'chart':chart,'annotation':annotation,
                'events':[(str(p.beat),p.note_text) for p in points]}))
            if not points or weight <= 0:
                continue
            event_at = {position: event for event, position in enumerate(
                i for i, token in enumerate(ids) if token == self.codec.at_id)}
            seconds = self._point_seconds[index]
            event_weights = [event_weight(float(song.first) if i == 0 else seconds[i-1], second, annotation)
                             for i, second in enumerate(seconds)]
            intent_quality = curve_weights(song, seconds, annotation)
            eos_weight = (interval_weight(seconds[-1], float(song.audio.duration_sec), annotation)
                          if (annotation or {}).get('regions') else 1.)
            all_weights = [0.]
            for point, local_weight in zip(points, event_weights):
                all_weights.extend([local_weight]*len(self.codec.cell(point)))
            all_weights.append(eos_weight)
            from .continuation import continuation_ranges,cycle_exit_targets
            exits=cycle_exit_targets(ids,self.codec)
            for chunk_index, (warm, supervised_start, end) in enumerate(continuation_ranges(
                    ids, self.max_tokens, self.codec.at_id, self.codec.eos_id,self.replay_tokens)):
                # The overlap is part of the decoder's self-attention prefix.
                # Its targets are retained with zero quality, so the model is
                # trained on exactly the same replay path used by generation.
                decoder = [self.codec.bos_id] + ids[warm:end-1]
                target = ids[warm:end]
                history_start = max(0, warm-self.max_history_tokens)
                history = ids[history_start:warm] if warm > 1 else []
                # Map event starts to local target rows. Labels supervise the
                # next-event intent, not every character in a configuration.
                labels=[]; masks=[]
                for row, token in enumerate(target):
                    if token == self.codec.at_id:
                        global_index = warm + row
                        event_index = event_at[global_index]
                        point = points[min(len(points) - 1, event_index)]
                        previous = Fraction(0) if event_index == 0 else points[event_index - 1].beat
                        event_second = float(seconds_at_beat(point.beat, song.tempo, song.first) - song.first)
                        density = float(np.interp(event_second, song.load_curve_times_sec,
                            song.load_curve[:, 0], left=0., right=0.))
                        previous_text = None if event_index == 0 else points[event_index - 1].note_text
                        labels.append(_plan_for_event(point, previous, point.note_text, density,
                                                      previous_text=previous_text))
                        masks.append(bool(global_index >= supervised_start and intent_quality[event_index] > 0))
                    else:
                        labels.append((0.,)*PLAN_DIM); masks.append(False)
                context_event = bisect_left(positions,warm)
                supervised_event = bisect_left(positions,supervised_start)
                target_offset = max(0,supervised_start-warm)
                weights=[0. if i<supervised_start else all_weights[i] for i in range(warm,end)]
                if any(value > 0 for value in weights):
                    self.chunks.append(StreamChunk(index, chunk_index, tuple(decoder), tuple(target),
                        tuple(history), tuple(labels), tuple(masks), end >= len(ids), context_event,
                        supervised_event, target_offset, tuple(weights),tuple(exits[warm:end])))
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
        identity.update(history_note_dropout=self.history_note_dropout,prefix_note_dropout=self.prefix_note_dropout,
            replay_tokens=self.replay_tokens,prefix_mask_span_events=self.prefix_mask_span_events,
            symmetry_augmentation=self.symmetry_augmentation)
        if self.grounded:
            identity['motor_profile']=grounding_config.motor_profile
        self.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        self._chunk_counts=Counter(c.chart_index for c in self.chunks)

    def set_epoch(self,epoch):self.epoch=int(epoch)

    def augmented_atoms(self):
        """Mirrored atoms seen in training must also exist in the generation grammar."""
        from .symmetry import transform_text
        from maichart.generation.event_cli import _NoteGrammar
        atoms=set()
        for index,record in enumerate(self.base.songs):
            for transform in self._symmetries[index][1:] if index<len(self._symmetries) else ():
                for point in record[1]:
                    atoms.update(_NoteGrammar._split_slashes(transform_text(str(point.note_text),transform)))
        return sorted(atoms)

    def _orientation(self,index):
        choices=self._symmetries[self.chunks[index].chart_index]
        if len(choices)==1:return 'identity'
        rng=random.Random(self.seed*7+self.epoch*1_000_033+index*104729)
        return choices[rng.randrange(len(choices))]

    def _oriented_chart(self,chart_index,transform):
        key=(chart_index,transform)
        if key not in self._oriented_ids:
            from .symmetry import transform_text
            ids=np.asarray(self._chart_ids[chart_index],dtype=np.int16)
            positions=self._chart_positions[chart_index]
            points=self.base.songs[chart_index][1]
            for event,start in enumerate(positions):
                note=self._chart_ids[chart_index].index(self.codec.note_id,start)+1
                text=transform_text(str(points[event].note_text),transform)
                ids[note:note+len(text)]=[self.codec.ids['N:'+c] for c in text]
            self._oriented_ids[key]=ids
        return self._oriented_ids[key]

    def _orient_chunk(self,chunk,transform):
        """Same token positions, mirrored cells; non-note tokens are unchanged."""
        if transform=='identity':return chunk,transform
        ids=self._oriented_chart(chunk.chart_index,transform)
        positions=self._chart_positions[chunk.chart_index]
        warm=positions[chunk.start_event] if chunk.start_event<len(positions) else len(ids)-1
        end=warm+len(chunk.target)
        history_start=max(0,warm-self.max_history_tokens)
        target=tuple(int(v) for v in ids[warm:end])
        note=lambda token:self.codec.tokens[token].startswith('N:')
        if len(target)!=len(chunk.target) or any(a!=b and not (note(a) and note(b))
                                                  for a,b in zip(target,chunk.target)):
            self.symmetry_stats['orientation_fallbacks']+=1
            return chunk,'identity'
        return replace(chunk,target=target,decoder_input=(self.codec.bos_id,*target[:-1]),
            history=tuple(int(v) for v in ids[history_start:warm]) if chunk.history else ()),transform

    def prepare_concurrency_key(self,index):
        # Difficulties sharing one source serialize mutable audio-view and
        # sustain caches; independent songs can prepare on separate threads.
        return str(self.base.songs[self.chunks[index].chart_index][0].audio.path)

    def _cached_query(self,key,builder):
        # Test/legacy lightweight dataset constructors may omit runtime locks.
        from contextlib import nullcontext
        lock=getattr(self,'_query_lock',nullcontext())
        build_locks=getattr(self,'_query_build_locks',())
        with build_locks[hash(key)%len(build_locks)] if build_locks else nullcontext():
            with lock:
                if key in self._audio_queries:
                    self.query_hits+=1;self._audio_queries.move_to_end(key)
                    return self._audio_queries[key][0]
                self.query_misses+=1
            started=time.perf_counter();query=builder();elapsed=time.perf_counter()-started
            size=query[0].numel()*query[0].element_size()+query[1].nbytes+len(query[2])*64
            with lock:
                if hasattr(self,'query_prepare_seconds'):
                    family=key[0] if isinstance(key,tuple) else 'planner'
                    self.query_prepare_seconds[family]+=elapsed;self.query_prepare_counts[family]+=1
                if size<=self.query_budget:
                    while self._audio_queries and self.query_bytes+size>self.query_budget:
                        _,(_,removed)=self._audio_queries.popitem(last=False);self.query_bytes-=removed
                    self._audio_queries[key]=(query,size);self.query_bytes+=size
            return query

    def _queries(self, index, song):
        key = self._query_keys[index]
        return self._cached_query(key,lambda:self._mapped_query(index,'planner',1024,
            lambda:planner_audio_queries(self._source_song(index)))) if getattr(self,'input_cache',None) else self._cached_query(key,lambda:planner_audio_queries(song))

    def _source_song(self,index):
        from maichart.form.data import _LazyAudio
        song=self.base.songs[index][0]
        return replace(song,audio=song.audio.store.get(song.audio.path)) if isinstance(song.audio,_LazyAudio) else song

    def _mapped_query(self,index,kind,settings,builder):
        from .input_cache import LazyBeats
        if self.input_cache is None:return builder()
        identity=self._query_identity(index,kind,settings)
        def build():
            values,times,beats=builder()
            duration=self._durations.get(index)
            if duration is None:duration=float(self.base.songs[index][0].audio.duration_sec)
            return {'values':values,'times':torch.as_tensor(times)}, {'beats':list(map(str,beats)),'duration':duration}
        tensors,extra=self.input_cache.get(identity,build)
        self._durations[index]=extra['duration']
        return tensors['values'],tensors['times'].numpy(),LazyBeats(extra['beats'])

    def _query_identity(self,index,kind,settings):
        identity={'type':kind,'settings':settings,'query':self._query_keys[index],
            'features':self.base.audio_store.feature_policy+'_'+self.base.audio_store.evidence_profile}
        if kind in {'probes','alignment'}:identity['chart']=self._chart_identities[index]
        return identity

    def _structure_identity(self,chunk):
        return {'type':'causal_structure_v1','chart':self._chart_identities[chunk.chart_index],
            'chunk':chunk.chunk_index,'tokens':self.max_tokens,'history':self.max_history_tokens,
            'replay_tokens':self.replay_tokens}

    def prepare_residency(self,*,policy='auto',budget_bytes=None,reserve_bytes=0,validation_records=0,
                          report=lambda **_:None):
        if self.input_cache is None:return {'mode':'legacy'}
        identities=[]
        for index in sorted({c.chart_index for c in self.chunks}):
            identities.append(self._query_identity(index,'planner',1024))
            if self.grounded:
                cfg=self.grounding_config
                for kind,setting in (('detail',cfg.detail_audio_nodes),('probes',None),
                        ('alignment',(cfg.detail_audio_nodes,cfg.alignment_tolerance_ms))):
                    identities.append(self._query_identity(index,kind,setting))
        if self.grounded:identities.extend(self._structure_identity(c) for c in self.chunks)
        if budget_bytes is None:
            budget_bytes=self.query_budget+self.base.audio_store.budget+self.structure_budget
        if policy in {'auto','resident'}:
            with self._query_lock:
                self._audio_queries.clear();self.query_bytes=0
                self._structures.clear();self.structure_bytes=0
            store=getattr(self.base,'audio_store',None)
            if store is not None and hasattr(store,'entries'):
                store.entries.clear();store.bytes=0
        self.residency=self.input_cache.preload(identities,budget_bytes=budget_bytes,policy=policy,
            reserve_bytes=reserve_bytes,validation_records=validation_records,report=report)
        return self.residency

    def warm_input_cache(self, report=lambda **_:None):
        """Build all deterministic mapped inputs before GPU training starts.

        Without this pass, an automatic/partial residency run discovers missing
        artifacts inside random training batches, causing disk I/O and cache
        rebuilds to compete with CUDA for the whole first epoch.
        """
        if self.input_cache is None:
            return {'enabled':False,'items':0}
        started=time.monotonic(); total=len(self.chunks); completed=0
        seen=set();rebuilt_samples=validated_artifacts=0
        for index in range(total):
            chunk=self.chunks[index];chart=chunk.chart_index
            identities=[self._query_identity(chart,'planner',1024)]
            if self.grounded:
                cfg=self.grounding_config
                identities.extend(self._query_identity(chart,kind,setting) for kind,setting in (
                    ('detail',cfg.detail_audio_nodes),('probes',None),
                    ('alignment',(cfg.detail_audio_nodes,cfg.alignment_tolerance_ms))))
                identities.append(self._structure_identity(chunk))
            ready=True
            for identity in identities:
                key=self.input_cache.key(identity)
                if key in seen:continue
                # Same identity/checksum/finiteness checks used by sampling;
                # no weaker metadata-only definition of a cache hit.
                if self.input_cache.validate_existing(identity):validated_artifacts+=1
                else:ready=False
                seen.add(key)
            if not ready:
                self[index]
                rebuilt_samples+=1
            completed += 1
            if completed % 100 == 0 or completed == total:
                report(stage='warmup',completed=completed,total=total,
                       validated_artifacts=validated_artifacts,rebuilt_samples=rebuilt_samples,
                       elapsed_sec=round(time.monotonic()-started,2))
        return {'enabled':True,'items':completed,'validated_artifacts':validated_artifacts,
                'rebuilt_samples':rebuilt_samples,'elapsed_sec':time.monotonic()-started}

    def _structure(self,chunk,song,points):
        """Cache prefix-only geometry before any epoch-dependent masking."""
        from .grounding import clock_inputs,history_clock,action_summaries
        from .input_cache import tensor_bytes
        key=(chunk.chart_index,chunk.chunk_index)
        with self._query_lock:
            if key in self._structures:
                self.structure_hits+=1;self._structures.move_to_end(key)
                return self._structures[key][0]
            self.structure_misses+=1
        def build():
            previous=points[chunk.start_event-1].beat if chunk.start_event else Fraction(0)
            before=points[chunk.start_event-2].beat if chunk.start_event>1 else Fraction(0)
            rows=clock_inputs(chunk.decoder_input,self.codec,song,previous=previous,previous_previous=before,committed=chunk.start_event)
            beats=rows.pop('query_beats');pending=rows.pop('query_pending')
            rows['probe_index']=torch.tensor([self._probe_indices[chunk.chart_index][b] for b in beats],dtype=torch.long)
            rows['query_actions']=action_summaries(self.base.songs[chunk.chart_index][4],beats,pending,rows['query_committed'])
            action,mask,_=prefix_action_state(self.base.songs[chunk.chart_index][4],points,chunk.start_event)
            rows['action_state'],rows['action_state_padding_mask']=action,mask
            positions=self._chart_positions[chunk.chart_index]
            start=positions[chunk.start_event] if chunk.start_event<len(points) else len(self._chart_ids[chunk.chart_index])-1
            rows['history_clock']=(history_clock(self._chart_ids[chunk.chart_index],max(0,start-self.max_history_tokens),
                start,positions,points,self.codec,song) if chunk.history else torch.zeros(0,20))
            return rows,{}
        identity=self._structure_identity(chunk)
        rows,_=self.input_cache.get(identity,build) if self.input_cache is not None else build()
        size=tensor_bytes(rows)
        with self._query_lock:
            if size<=self.structure_budget:
                while self._structures and self.structure_bytes+size>self.structure_budget:
                    _,(_,removed)=self._structures.popitem(last=False);self.structure_bytes-=removed
                self._structures[key]=(rows,size);self.structure_bytes+=size
        return rows

    def _grounded_item(self,chunk,item,song,points,annotation,rng,*,source=None,transform='identity'):
        from .grounding import (clock_inputs,history_clock,exact_probe_cache,detail_queries,
                                time_cell_weights,masked_note_inputs,masked_clock_inputs,action_summaries)
        from .symmetry import transform_state,kinematic_rows
        from maichart.form.alignment import build_alignment_targets
        cfg=self.grounding_config;index=chunk.chart_index
        detail,times,beats=self._cached_query(('detail',self._query_keys[index],cfg.detail_audio_nodes),
            lambda:self._mapped_query(index,'detail',cfg.detail_audio_nodes,lambda:detail_queries(self._source_song(index),cfg.detail_audio_nodes)))
        probes=self._probe_beats[index]
        def probe_builder():
            values,_=exact_probe_cache(self._source_song(index),probes,self.cache_dir)
            return values,np.asarray([seconds_at_beat(b,song.tempo,song.first) for b in probes],dtype=np.float64),tuple(probes)
        exact,_,_=self._cached_query(('probes',index),lambda:self._mapped_query(index,'probes',None,probe_builder))
        if index in self._durations:song=replace(song,audio=SimpleNamespace(duration_sec=self._durations[index]))
        structure=self._structure(source if source is not None else chunk,song,points)
        item.update({k:v for k,v in structure.items() if k!='probe_index'})
        item['action_state']=transform_state(structure['action_state'],transform)
        item['query_actions']=transform_state(structure['query_actions'],transform)
        item['query_audio']=exact[structure['probe_index']]
        positions=self._chart_positions[index]
        token_start=positions[chunk.start_event] if chunk.start_event<len(points) else len(self._chart_ids[index])-1
        history_start=max(0,token_start-self.max_history_tokens)
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
        item['archive_cells']=[self._archive_cell(index,i,transform) for i in selected]
        item['archive_audio']=exact[[self._probe_indices[index][points[i].beat] for i in selected]]
        item['archive_beats']=torch.from_numpy(self._point_values[index][selected])
        item['archive_seconds']=torch.from_numpy(self._point_seconds[index][selected])
        item['archive_event_index']=torch.tensor(selected,dtype=torch.long)
        item['archive_padding_mask']=torch.zeros(len(selected),dtype=torch.bool)
        # Unknown history is explicit MASK, never invented wrong configurations.
        history,_,history_hidden=masked_note_inputs(chunk.history,self.codec,rng,self.history_note_dropout,return_events=True)
        for cell in item['archive_cells']:
            if rng.random()<self.history_note_dropout:cell[:]=[self.codec.mask_id]*len(cell)
        decoder,quality,prefix_hidden=masked_note_inputs(chunk.decoder_input,self.codec,rng,self.prefix_note_dropout,
            targets=chunk.target,quality=chunk.target_quality,span_events=self.prefix_mask_span_events,return_events=True)
        item['decoder_input']=torch.tensor(decoder,dtype=torch.long)
        item['history']=torch.tensor(history,dtype=torch.long)
        item['target_quality']=torch.tensor(quality,dtype=torch.float32)
        item['decoder_clock'],item['query_clock'],prefix_intervals=masked_clock_inputs(
            structure['decoder_clock'],chunk.decoder_input,self.codec,rng,prefix_hidden,
            dropout=.5*self.prefix_note_dropout,query_clock=structure['query_clock'],
            query_index=structure['decoder_query_index'],targets=chunk.target,return_interval_events=True)
        item['history_clock'],history_intervals=masked_clock_inputs(
            structure['history_clock'],chunk.history,self.codec,rng,history_hidden,
            dropout=.5*self.history_note_dropout,return_interval_events=True)
        # Hand kinematics read only committed events (query_committed counts
        # complete cells), so pure-Tap passages are no longer an empty state.
        # Cells hidden by note dropout stay hidden here as well.
        lanes=list(self._lane_sets(index,transform))
        stored_trajectories=self._trajectory_sets(index,transform)
        trajectories=list(stored_trajectories) if stored_trajectories is not None else None
        history_first=chunk.start_event-sum(1 for t in chunk.history if t==self.codec.at_id)
        masked_events=set()
        for event in [history_first+e for e in history_hidden]+[chunk.start_event+e for e in prefix_hidden]:
            if 0<=event<len(lanes):
                lanes[event]=()
                masked_events.add(event)
                if trajectories is not None:trajectories[event]=()
        interval_hidden={chunk.start_event+e for e in prefix_intervals}|{history_first+e for e in history_intervals}
        pending=structure['query_clock'][:,18].numpy()>0
        if cfg.motor_profile=='causal_hands_v1':
            owners=structure['query_committed'].numpy()-1+pending.astype(np.int64)
            hidden_query=torch.tensor([int(owner) in interval_hidden for owner in owners],dtype=torch.bool)
            item['query_clock']=item['query_clock'].clone()
            item['query_clock'][hidden_query,15:17]=0.
            item['decoder_clock']=item['decoder_clock'].clone()
            item['decoder_clock'][hidden_query[structure['decoder_query_index']],15:17]=0.
        item['query_kinematics']=kinematic_rows(lanes,self._point_seconds[index],
            structure['query_committed'].numpy(),structure['query_seconds'].numpy(),
            trajectories=trajectories,masked_events=masked_events,interval_hidden=interval_hidden,
            pending=pending,profile=cfg.motor_profile)
        if cfg.motor_profile=='causal_hands_v1':
            from .hand_tracking import HandHistory
            histories=getattr(self,'_hand_histories',None)
            if histories is None:histories=self._hand_histories={}
            key=(index,transform)
            if key not in histories:
                histories[key]=HandHistory(self._lane_sets(index,transform),self._point_seconds[index],stored_trajectories,
                                           stride=128,keep_recent=False)
            counts=structure['query_committed'].numpy()
            item['query_hands']=histories[key].rows(counts,structure['query_seconds'].numpy(),masked_events=masked_events)
            # Hide explicit age echoes with the same interval decision. Position
            # tokens and real sustain geometry remain causal and observable.
            hidden_age=torch.tensor([((int(c) if p else int(c)-1) in interval_hidden
                                     or int(c)-1 in interval_hidden) for c,p in zip(counts,pending)],dtype=torch.bool)
            item['query_hands'][hidden_age,4]=0.
            item['query_hands'][hidden_age,12]=0.
        event_rows=[i for i,t in enumerate(chunk.target)
                    if t==self.codec.at_id and i>=chunk.target_offset]
        event_points=points[chunk.supervised_event:chunk.supervised_event+len(event_rows)]
        stamps=self._point_seconds[index][chunk.supervised_event:chunk.supervised_event+len(event_rows)]
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
        supervised,_,_=self._cached_query(('alignment',index,cfg.detail_audio_nodes,cfg.alignment_tolerance_ms),
            lambda:self._mapped_query(index,'alignment',(cfg.detail_audio_nodes,cfg.alignment_tolerance_ms),supervision_builder))
        item['position_alignment_target']=supervised[:,0]
        item['alignment_audio_quality']=supervised[:,2]
        item['position_alignment_mask']=supervised[:,1].bool()&~item['detail_audio_padding_mask']&(supervised[:,2]>0)
        return item

    def cache_stats(self):
        with self._query_lock:
            queries={'bytes':self.query_bytes,'budget_bytes':self.query_budget,'entries':len(self._audio_queries),
                'hits':self.query_hits,'misses':self.query_misses,
                'prepare_seconds':dict(self.query_prepare_seconds),'prepare_counts':dict(self.query_prepare_counts)}
        return {'audio':self.base.audio_store.stats(),'stream_queries':queries,
            'mapped_inputs':self.input_cache.stats() if self.input_cache else {'enabled':False},
            'structures':{'hits':self.structure_hits,'misses':self.structure_misses,
                'bytes':self.structure_bytes,'budget_bytes':self.structure_budget}}

    def __len__(self): return len(self.chunks)
    def __getitem__(self, index):
        source = self.chunks[index]
        chunk, transform = (self._orient_chunk(source, self._orientation(index))
                            if self.symmetry_augmentation != 'none' else (source, 'identity'))
        song, points, level, weight, state, condition, excluded, annotation = self.base.songs[chunk.chart_index]
        audio, times, _ = self._queries(chunk.chart_index, song)
        if self.grounded:
            clock_song=(replace(song,audio=SimpleNamespace(duration_sec=self._durations[chunk.chart_index]))
                        if chunk.chart_index in self._durations else song)
            # Mirroring changes lanes, never clocks: the cached structure of
            # the authored chunk is shared by all orientations.
            structure=self._structure(source,clock_song,points)
            action,action_mask=structure['action_state'],structure['action_state_padding_mask']
        else:action, action_mask, _ = prefix_action_state(state, points, chunk.start_event)
        from .symmetry import transform_state
        action=transform_state(action,transform)
        item={'audio': audio, 'audio_times_sec': torch.as_tensor(times, dtype=torch.float64),
                'curve_target': torch.from_numpy(resample_load_curve(
                    song.load_curve_times_sec, song.load_curve,
                    np.asarray(times, dtype=np.float64) - float(song.first))),
                'condition': condition.to_vector(),
                'quality_weight': torch.tensor(float(weight)),
                'target_quality': torch.tensor(chunk.target_quality, dtype=torch.float32),
                'cycle_exit_mask':torch.tensor(chunk.cycle_exit_mask or (False,)*len(chunk.target)),
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
            item=self._grounded_item(chunk,item,song,points,annotation,rng,source=source,transform=transform)
        return item

    def _lane_sets(self,chart_index,transform):
        if transform=='identity':return self._event_lanes[chart_index]
        cache=getattr(self,'_oriented_lanes',None)
        if cache is None:cache=self._oriented_lanes={}
        key=(chart_index,transform)
        if key not in cache:
            from .symmetry import _BUTTON
            mapping=_BUTTON[transform]
            cache[key]=[tuple(sorted(mapping(lane) for lane in lanes)) for lanes in self._event_lanes[chart_index]]
        return cache[key]

    def _trajectory_sets(self,chart_index,transform):
        trajectories=getattr(self,'_event_trajectories',())
        if chart_index>=len(trajectories):return None
        if transform=='identity':return trajectories[chart_index]
        cache=getattr(self,'_oriented_trajectories',None)
        if cache is None:cache=self._oriented_trajectories={}
        key=(chart_index,transform)
        if key not in cache:
            from .symmetry import transform_trajectories
            cache[key]=transform_trajectories(trajectories[chart_index],transform)
        return cache[key]

    def _archive_cell(self,chart_index,event,transform):
        if transform=='identity':return list(self._event_note_ids[chart_index][event])
        from .symmetry import transform_text
        text=transform_text(str(self.base.songs[chart_index][1][event].note_text),transform)
        return [self.codec.ids['N:'+c] for c in text]


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
    result['cycle_exit_mask']=pad_sequence([item.get('cycle_exit_mask',torch.zeros_like(item['target'],dtype=torch.bool)) for item in items],batch_first=True)
    if 'decoder_clock' in items[0]:
        from .symmetry import KINEMATIC_DIM
        for item in items:
            if 'query_kinematics' not in item:
                item['query_kinematics']=torch.zeros(len(item['query_clock']),KINEMATIC_DIM)
        for key in ('decoder_clock','history_clock','query_clock','query_seconds','query_audio','query_committed','query_actions',
                    'query_kinematics',
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
        if 'query_hands' in items[0]:
            if any('query_hands' not in item for item in items):raise ValueError('Mixed stream motor profiles in one batch')
            result['query_hands']=pad_sequence([item['query_hands'] for item in items],batch_first=True)
        # Keep the same ordered per-event moments, but construct repeated
        # owners and positions in NumPy instead of a Python loop per token.
        from itertools import chain
        cells=[];owners=[]
        width=result['archive_padding_mask'].size(1)
        for row,item in enumerate(items):
            for event,cell in enumerate(item['archive_cells']):
                cells.append(cell);owners.append(row*width+event)
        lengths=np.fromiter(map(len,cells),dtype=np.int64,count=len(cells))
        count=int(lengths.sum())
        starts=np.cumsum(lengths)-lengths
        sizes=np.repeat(lengths,lengths)
        offsets=np.arange(count,dtype=np.float64)-np.repeat(starts,lengths)
        positions=((offsets+.5)/sizes).astype(np.float32)
        inverse=np.reciprocal(sizes.astype(np.float64)).astype(np.float32)
        result.update(archive_flat_tokens=torch.from_numpy(np.fromiter(chain.from_iterable(cells),dtype=np.int64,count=count)),
            archive_token_event=torch.from_numpy(np.repeat(np.asarray(owners,dtype=np.int64),lengths)),
            archive_token_position=torch.from_numpy(positions),
            archive_token_inverse_length=torch.from_numpy(inverse))
    return result
