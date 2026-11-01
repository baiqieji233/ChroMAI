"""Independent, causal event-set stream training data contract.

No Composer candidate lattice is constructed.  A sample covers one fixed
absolute-time window, split into bounded target chunks when necessary.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
import math
import threading
import sys

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from maichart.form.data import FormDataset
from maichart.event_timing import EventPoint
from maichart.generation.action_state_spec import STATE_DIM
from maichart.generation.sustain_state import SustainStateIndex
from maichart.stream.grounded_inference import IncrementalState
from maichart.tempo_clock import TempoClock
from .acoustics import AcousticContext
from .candidates import composer_audio_nodes, interp_candidate_audio
from .event_set_codec import EventSetCodec, _representation
from .event_views import EventViews
from .events import event_metadata
from .stream_bank_cache import StreamBankDiskCache, bank_identity
from .stream_continuation import ContinuationState, continuation_rows, continuation_dim, POLICIES as CONTINUATION_POLICIES
from .stream_timing import encode_history_gap, TIMING_POLICIES
from .stream_phrase import (PhraseIndex, PHRASE_POLICIES, PHRASE_ROW_FIELDS,
    collate_phrase_sequences, collate_candidate_phrase_sequences, share_aligned_phrase_sequences,
    collate_phrase_music,
    prepare_candidate_phrase_inputs)

PROFILE = 'event_set_stream_v1'
PAD, BOS, NOTE, WAIT, EOS = range(5)
GRID_SECONDS = .01
NATIVE_DYNAMICS_SECONDS = .005


def _grid_seconds(value):
    value = float(value)
    if not math.isfinite(value) or not .005 <= value <= .05:
        raise ValueError('audio_grid_seconds must be between .005 and .05 seconds')
    return value


def _fraction(value):
    return value if isinstance(value, Fraction) else Fraction(str(value))


def beat_at_second(clock, second):
    """Invert the piecewise clock, extending its first tempo before beat 0."""
    sec = _fraction(second)
    starts = [_fraction(x) for x in clock.seconds]
    i = max(0, bisect_right(starts, sec) - 1)
    return clock.beats[i] + (sec - starts[i]) * _fraction(clock.bpms[i]) / 60


def second_at_beat(clock, beat):
    beat = _fraction(beat)
    if beat < 0:
        return float(_fraction(clock.seconds[0]) + beat * 60 / _fraction(clock.bpms[0]))
    return float(clock.at(beat))


def _float_beats_at_seconds(clock, seconds):
    seconds = np.asarray(seconds, dtype=np.float64)
    indices = np.clip(np.searchsorted(clock._seconds_arr, seconds, side='right')-1, 0, len(clock.beats)-1)
    return clock._beat_floats[indices] + (seconds-clock._seconds_arr[indices]) / clock._scale_arr[indices]


@dataclass(frozen=True)
class StreamStep:
    kind: int
    start: Fraction
    end: Fraction
    start_second: float
    end_second: float
    text: str | None = None
    weight: float = 1.


def build_steps(points, clock, duration, *, views=None):
    """Compile exact beats; EOS means no further events through audio end.

    EOS has no delta class. WAIT advances exactly two beats until the next
    event or end is within two beats. Unknown spans never become rest labels.
    """
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('audio duration must be finite and positive')
    start = Fraction(0)
    end = beat_at_second(clock, duration)
    if end < start:
        raise ValueError('audio ends before chart beat zero')
    cursor = start
    steps = []
    last_beat = None
    def confidence(a, b):
        if views is None:
            return 1.
        hi = min(float(b), np.nextafter(duration, -np.inf))
        return float(views.window_weight(max(0., float(a)), hi))
    for index, point in enumerate(points):
        beat = _fraction(point.beat)
        sec = second_at_beat(clock, beat)
        if not 0 <= sec < duration:
            continue
        if beat < cursor or (last_beat is not None and beat <= last_beat):
            raise ValueError('events must have strictly increasing beats')
        while beat - cursor > 2:
            nxt = cursor + 2
            a, b = max(0., second_at_beat(clock, cursor)), max(0., second_at_beat(clock, nxt))
            wait_weight = 0. if second_at_beat(clock,nxt) <= 0. else confidence(a,b)
            steps.append(StreamStep(WAIT, cursor, nxt, a, b, weight=wait_weight))
            cursor = nxt
        a = max(0., second_at_beat(clock, cursor))
        w = confidence(a, sec)
        if views is not None:
            w *= float(views.local_weights[index]) if views.trusted[index] else 0.
        steps.append(StreamStep(NOTE, cursor, beat, a, sec, str(point.note_text), w))
        cursor = beat
        last_beat = beat
    while end - cursor > 2:
        nxt = cursor + 2
        a, b = max(0., second_at_beat(clock, cursor)), max(0., second_at_beat(clock, nxt))
        wait_weight = 0. if second_at_beat(clock,nxt) <= 0. else confidence(a,b)
        steps.append(StreamStep(WAIT, cursor, nxt, a, b, weight=wait_weight))
        cursor = nxt
    a = max(0., second_at_beat(clock, cursor))
    steps.append(StreamStep(EOS, cursor, end, a, duration, weight=confidence(a, duration)))
    return steps


class DeltaVocabulary:
    """Exact training deltas with reserved index zero for PAD/BOS/EOS."""
    def __init__(self, values):
        self.values = tuple(sorted({_fraction(x) for x in values}))
        if any(x < 0 or x > 2 for x in self.values):
            raise ValueError('stream delta must be between 0 and 2 beats')
        self.indices = {x: i + 1 for i, x in enumerate(self.values)}

    @classmethod
    def fit(cls, streams):
        common = {Fraction(2), Fraction(0)}
        common.update(Fraction(n, d) for d in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
                      for n in range(1, 2*d + 1))
        return cls(common | {step.end-step.start for stream in streams for step in stream if step.kind != EOS})

    def encode(self, delta):
        try:
            return self.indices[_fraction(delta)]
        except KeyError as exc:
            raise ValueError(f'out-of-vocabulary stream delta: {delta}') from exc

    def to_dict(self):
        return {'schema': 'event_set_stream_delta_v1', 'values': [str(x) for x in self.values]}

    @classmethod
    def from_dict(cls, state):
        if state.get('schema') != 'event_set_stream_delta_v1':
            raise ValueError('invalid stream delta vocabulary')
        return cls(state['values'])


def build_audio_bank(song, clock, audio_grid_seconds=GRID_SECONDS):
    grid = _grid_seconds(audio_grid_seconds)
    values, times = composer_audio_nodes(song)
    duration = float(song.audio.duration_sec)
    # Composer's column 107 normalizes beat by the authored chart extent.
    # Replace that future-derived extent with the audio-end beat instead.
    values = values.copy()
    beats_at_nodes = _float_beats_at_seconds(clock, times)
    values[:,107] = beats_at_nodes / max(1., float(beat_at_second(clock, duration)))
    seconds = np.arange(0., duration, grid, dtype=np.float64)
    context = AcousticContext(values, times, duration)
    macro = context.at(seconds)
    local = interp_candidate_audio(values, times, seconds)
    # Retain narrow source-node attacks/flux within each fixed cell rather
    # than sampling them only at its left boundary. Signed gradient keeps
    # whichever excursion has greater magnitude.
    def preserve_peaks(sample_times, sample_values, columns, *, signed=()):
        indices = np.searchsorted(seconds, sample_times, side='right')-1
        valid = (indices >= 0) & (sample_times < duration)
        indices = indices[valid]
        for source_column, target_column in columns:
            sample = np.asarray(sample_values[valid,source_column],dtype=np.float32)
            high = np.full(len(seconds),-np.inf,dtype=np.float32)
            np.maximum.at(high,indices,sample)
            present = np.isfinite(high)
            if target_column in signed:
                low = np.full(len(seconds),np.inf,dtype=np.float32)
                np.minimum.at(low,indices,sample)
                high = np.where(np.abs(low)>np.abs(high),low,high)
                current = local[:,target_column]
                local[:,target_column] = np.where(present & (np.abs(high)>np.abs(current)),high,current)
            else:
                local[:,target_column] = np.where(present,np.maximum(local[:,target_column],high),local[:,target_column])
    if grid != .05:
        preserve_peaks(times,values,[(92,92),(94,94),(95,95),(96,96),(97,97)],signed=(96,))
        # Sample waveform-derived short dynamics at 5 ms independently of the
        # model grid. These are measured peaks, not inferred sound sources.
        from .acoustics import native_dynamics
        probe_times = np.arange(0.,duration,NATIVE_DYNAMICS_SECONDS,dtype=np.float64)
        measured = native_dynamics(song.audio,probe_times)
        preserve_peaks(probe_times,measured,[(0,92),(2,94),(3,96)],signed=(96,))
        local[:,98] = local[:,99] = local[:,92]
        local[:,100] = local[:,94]
    # Eight frequency-band means and six transient/dynamics channels; each is
    # relative to the real elapsed span of the preceding four beats. These are
    # acoustic contrasts, not claims about individual instruments or voices.
    grouped = np.column_stack((local[:, :80].reshape(-1, 8, 10).mean(-1), local[:, 92:98]))
    beats = _float_beats_at_seconds(clock, seconds)
    previous = clock.at_floats(beats-4.)
    previous = np.maximum(0., previous)
    widths = np.diff(np.r_[seconds, duration])
    prefix = np.vstack((np.zeros((1, 14)), np.cumsum(grouped.astype(np.float64)*widths[:, None], axis=0)))
    ix = np.clip(np.searchsorted(seconds, previous, side='right')-1, 0, len(seconds)-1)
    area = prefix[:-1] - (prefix[ix] + grouped[ix] * (previous-seconds[ix])[:, None])
    past = area / np.maximum(seconds-previous, 1e-8)[:, None]
    contrast = np.where((seconds > previous)[:, None], grouped-past, 0.)
    features = np.concatenate((macro, local, contrast), axis=1).astype(np.float32)
    return torch.from_numpy(features), torch.from_numpy(seconds), torch.from_numpy(beats)


def _audio_plan(views, duration, audio_grid_seconds=GRID_SECONDS, plan_blank_window_seconds=.5):
    """[blank, head count, event count] per exact fixed cell, label-only."""
    grid = _grid_seconds(audio_grid_seconds)
    edges = np.r_[np.arange(0., duration, grid, dtype=np.float64), duration]
    count = len(edges)-1
    events = np.zeros(count, dtype=np.float32)
    heads = np.zeros(count, dtype=np.float32)
    for point, second, known in zip(views.context_points, views.seconds, views.trusted):
        if known and 0. <= float(second) < duration:
            index = min(int(np.searchsorted(edges, float(second), side='right') - 1), count-1)
            events[index] += 1.
            heads[index] += len(event_metadata(point.note_text).heads)
    blank_window = float(plan_blank_window_seconds)
    if not math.isfinite(blank_window) or blank_window < 0:
        raise ValueError('plan_blank_window_seconds must be finite and nonnegative')
    blank = events == 0
    weights = views.window_weight(edges[:-1], np.nextafter(edges[1:], -np.inf)).astype(np.float32)
    weights *= np.diff(edges).astype(np.float32)
    weights /= max(float(weights.sum()), 1e-8)
    blank_weights = weights.copy()
    if blank_window:
        centers=(edges[:-1]+edges[1:])*.5
        lo,hi=centers-blank_window*.5,centers+blank_window*.5
        points=views.seconds
        blank=(np.searchsorted(points,hi,side='left')-np.searchsorted(points,lo,side='left')) == 0
        complete=(lo>=0.) & (hi<=duration)
        known=views.window_weight(np.maximum(lo,0.),np.nextafter(np.minimum(hi,duration),-np.inf))
        blank_weights *= complete.astype(np.float32)*np.asarray(known,dtype=np.float32)
        blank_weights /= max(float(blank_weights.sum()),1e-8)
    target = np.column_stack((blank, heads, events)).astype(np.float32)
    return (torch.from_numpy(target), torch.from_numpy(weights),
            torch.from_numpy(np.diff(edges).astype(np.float32)),torch.from_numpy(blank_weights))


class StreamDataset(Dataset):
    """Causal chart streams backed by FormDataset, not ComposerDataset."""
    def __init__(self, manifest=None, *, cache_dir='cache/chart_composer_v1', base=None,
                 codec=None, delta_vocab=None, lead_in=64, targets=16, window_seconds=8.,
                 preparation_workers=1, audio_cache_gb=4., limit_charts=0, unknown_policy='error',
                 audio_grid_seconds=GRID_SECONDS, plan_blank_window_seconds=.5,
                 history_window_policy='rolling_legacy_v1',continuation_policy='off',timing_output_policy='relative_delta_v1',
                 window_balance_policy='equal_window_v1',phrase_context_policy='off',
                 candidate_state_policy='cursor_v1'):
        if lead_in < 0 or targets < 1 or window_seconds <= 0:
            raise ValueError('invalid stream window settings')
        if unknown_policy not in ('error','mask'):
            raise ValueError('unknown_policy must be error or mask')
        self.unknown_policy = unknown_policy
        if phrase_context_policy not in PHRASE_POLICIES:raise ValueError('invalid phrase_context_policy')
        self.phrase_context_policy = phrase_context_policy
        if candidate_state_policy not in ('cursor_v1','selected_native_v1'):
            raise ValueError('invalid candidate_state_policy')
        self.candidate_state_policy = candidate_state_policy
        if timing_output_policy not in TIMING_POLICIES:raise ValueError('invalid timing output policy')
        self.timing_output_policy=timing_output_policy
        if window_balance_policy not in ('equal_window_v1','sqrt_window_v2'):raise ValueError('invalid window balance policy')
        self.window_balance_policy=window_balance_policy
        if history_window_policy not in ('block_aligned_v2','rolling_legacy_v1'):
            raise ValueError('invalid history_window_policy')
        self.history_window_policy = history_window_policy
        if continuation_policy not in CONTINUATION_POLICIES: raise ValueError('invalid continuation_policy')
        self.continuation_policy = continuation_policy
        self.audio_grid_seconds = _grid_seconds(audio_grid_seconds)
        self.plan_blank_window_seconds = float(plan_blank_window_seconds)
        if not math.isfinite(self.plan_blank_window_seconds) or self.plan_blank_window_seconds < 0:
            raise ValueError('plan_blank_window_seconds must be finite and nonnegative')
        if not math.isfinite(float(audio_cache_gb)) or float(audio_cache_gb) < 0:
            raise ValueError('audio_cache_gb must be finite and nonnegative')
        self.lead_in, self.targets, self.window_seconds = int(lead_in), int(targets), float(window_seconds)
        self._stream_cache_dir = Path(cache_dir) if base is None else None
        self._disk_bank = StreamBankDiskCache(cache_dir) if base is None else None
        self.base = base if base is not None else FormDataset(manifest, cache_dir=cache_dir,
            max_tokens=1_000_000, max_memory_tokens=256, max_region_seconds=999.,
            preparation_workers=preparation_workers, audio_cache_gb=float(audio_cache_gb)*.25,
            limit_charts=limit_charts, quality_enabled=True, preserve_excluded_context=True,
            quality_profile='musical_evidence_v1', rhythm_geometry_profile='none',
            audio_profile='rational_audio_queries_v1')
        self.streams, self.clocks, self.conditions, self.views, self.plans, self.chart_weights = [], [], [], [], [], []
        self._durations, self._end_beats, self._bank_keys = [], [], []
        for song, points, level, weight, state, condition, excluded, annotation in self.base.songs:
            duration = float(song.audio.duration_sec)
            clock = TempoClock(song.tempo, song.first)
            self._durations.append(duration)
            self._end_beats.append(float(beat_at_second(clock,duration)))
            source = getattr(song.audio,'path',None) or getattr(song.audio,'source',None)
            self._bank_keys.append((str(source) if source else ('object',id(song.audio)),float(song.first),
                tuple((str(t.beat),float(t.bpm)) for t in song.tempo),duration,
                tuple((str(m.beat),int(m.numerator),int(m.denominator),tuple(getattr(m,'grouping',None) or ()))
                      for m in (getattr(song,'meters',None) or ()))))
            views = EventViews.build(points, song.tempo, song.first, duration, excluded=excluded,
                                     annotation=annotation, action_slots=64)
            stream = build_steps(views.context_points, clock, duration, views=views)
            # Explicit difficulty/style controls are distinct from inputting
            # exact future events. Training currently uses reference-derived
            # controls; generation must supply user/default controls.
            controls = condition.to_vector() if hasattr(condition, 'to_vector') else torch.as_tensor(condition)
            self.streams.append(stream); self.clocks.append(clock); self.conditions.append(controls); self.views.append(views)
            self.plans.append(_audio_plan(views, duration, self.audio_grid_seconds,self.plan_blank_window_seconds))
            self.chart_weights.append(max(0., float(weight)))
        all_text = [s.text for stream in self.streams for s in stream if s.kind == NOTE]
        self.codec = codec or EventSetCodec.fit(all_text)
        from .stream_actions import semantic_catalog
        self._literal_atom_types=semantic_catalog(self.codec)[0]
        self.delta_vocab = delta_vocab or DeltaVocabulary.fit(self.streams)
        self.encoded, self.widths, self.unknown_steps = [], [], []
        self.continuations = []
        self.oov = {'total_steps': 0, 'delta_oov': 0, 'action_oov': 0, 'fallback_recovered': 0,
                    'masked_steps': 0}
        self.samples, self.plan_weights, self.target_window_weights = [], [], []
        total_windows = covered_windows = 0
        for song_index, stream in enumerate(self.streams):
            encoded, unknown = [], []
            for step in stream:
                missing_delta = missing_action = False
                try:
                    delta = encode_history_gap(self.delta_vocab,step.end-step.start,self.timing_output_policy) if step.kind != EOS else 0
                except ValueError:
                    if unknown_policy == 'error': raise
                    delta, missing_delta = 0, True
                event = None
                if step.kind == NOTE:
                    try:
                        event = self.codec.encode(step.text)
                        if event['fallback'] and not _representation(step.text)[4]:
                            self.oov['fallback_recovered'] += 1
                    except KeyError:
                        # A new arrangement or relative detail can still be
                        # represented exactly by the literal atom vocabulary.
                        atoms = sorted(event_metadata(step.text).atoms)
                        ids = [self.codec._atoms.get(atom,0) for atom in atoms]
                        if ids and all(ids):
                            event = dict(shape=0,phase=0,atoms=ids,details=ids+[0],
                                         phase_mask=[True]*8,fallback=True)
                            self.oov['fallback_recovered'] += 1
                        elif unknown_policy == 'error':
                            raise
                        else:
                            missing_action = True
                bad = missing_delta or missing_action
                self.oov['total_steps'] += 1
                self.oov['delta_oov'] += int(missing_delta)
                self.oov['action_oov'] += int(missing_action)
                self.oov['masked_steps'] += int(bad)
                if bad:
                    # Preserve the clock and real sustain state, but never
                    # expose a partially encoded unknown step as a known one.
                    delta, event = 0, None
                encoded.append((delta,event)); unknown.append(bad)
            self.encoded.append(encoded)
            self.unknown_steps.append(unknown)
            if self.continuation_policy != 'off':
                self.continuations.append(continuation_rows(stream,policy=self.continuation_policy))
            else:
                self.continuations.append((torch.zeros(len(stream),continuation_dim(self.continuation_policy)),torch.zeros(len(stream),dtype=torch.bool),
                                           torch.zeros(len(stream),dtype=torch.bool)))
            self.widths.append((max(1, max((len(v['atoms']) for _,v in encoded if v is not None), default=0)),
                                max(1, max((len(v['details']) for _,v in encoded if v is not None), default=0))))
            windows = {}
            for i, step in enumerate(stream):
                window = int(step.start_second // self.window_seconds)
                windows.setdefault(window, []).append(i)
            event_weights = [0.] * len(stream)
            for indices in windows.values():
                for i in indices:
                    ratio=self.targets/len(indices)
                    event_weights[i] = (min(2.,max(.5,math.sqrt(ratio)))
                        if self.window_balance_policy=='sqrt_window_v2' else ratio)
            if self.window_balance_policy=='sqrt_window_v2' and event_weights:
                # Preserve this chart's former total mass while removing the
                # extreme per-transition advantage of sparse windows.
                scale=self.targets*len(windows)/sum(event_weights)
                event_weights=[value*scale for value in event_weights]
            self.target_window_weights.append(event_weights)
            duration = self._durations[song_index]
            total_windows += math.ceil(duration/self.window_seconds)
            covered_windows += len(windows)
            if self.history_window_policy == 'block_aligned_v2':
                for start in range(0,len(stream),self.targets):
                    end = min(start+self.targets,len(stream))
                    self.samples.append((song_index,start,end,1.))
                    self.plan_weights.append(sum(event_weights[start:end])/self.targets)
                continue
            for window, indices in sorted(windows.items()):
                chunks = [indices[i:i+self.targets] for i in range(0, len(indices), self.targets)]
                for chunk in chunks:
                    # Every time window has total target mass K before quality
                    # and unknown masks, independent of its event density.
                    self.samples.append((song_index, chunk[0], chunk[-1]+1, self.targets/len(indices)))
                    self.plan_weights.append(sum(event_weights[i] for i in chunk)/self.targets
                        if self.window_balance_policy=='sqrt_window_v2' else 1./len(chunks))
        count = sum(len(s) for s in self.streams)
        self.coverage = {'total_steps': count, 'supervised_steps': sum(end-start for _,start,end,_ in self.samples),
                         'note_steps': sum(s.kind == NOTE for stream in self.streams for s in stream),
                         'window_chunks': len(self.samples), 'total_time_windows': total_windows,
                         'covered_time_windows': covered_windows,
                         'empty_time_windows': max(0,total_windows-covered_windows)}
        if self.coverage['supervised_steps'] != count:
            raise AssertionError('stream samples do not cover each target exactly once')
        self.oov['known_steps'] = self.oov['total_steps']-self.oov['masked_steps']
        self.oov['known_fraction'] = self.oov['known_steps']/max(1,self.oov['total_steps'])
        self._banks = OrderedDict()
        self._states = OrderedDict()
        self._bank_budget = int(float(audio_cache_gb)*.50*1024**3)
        self._state_budget = int(float(audio_cache_gb)*.25*1024**3)
        # Symbolic labels share the existing non-audio cache allowance.
        self._symbol_budget = self._state_budget // 4
        self._state_budget -= self._symbol_budget
        self._symbols = OrderedDict()
        self._symbol_bytes = self._symbol_hits = self._symbol_misses = self._symbol_evictions = 0
        self._bank_bytes = self._state_bytes = 0
        self._bank_hits = self._bank_misses = self._bank_evictions = 0
        self._state_hits = self._state_misses = self._state_evictions = self._state_uncached = 0
        self._bank_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self.symbolic_only = False
        self._symbolic_bank = (torch.zeros(1,238), torch.zeros(1,dtype=torch.float64),
                               torch.zeros(1,dtype=torch.float64))
        self._symbolic_plan = (torch.zeros(1,3), torch.zeros(1), torch.zeros(1),torch.zeros(1))
        self._phrases = OrderedDict()
        self._phrase_bytes = 0
        self._phrase_budget = 64*1024**2
        self._phrase_uncached = 0

    def _phrase_index(self, song):
        with self._bank_lock, self._state_lock:
            if song in self._phrases:
                self._phrases.move_to_end(song)
                return self._phrases[song]
            ordered = self.phrase_context_policy in ('ordered_phrase_v2','aligned_phrase_v3')
            index = PhraseIndex(self._bank(song), self._end_beats[song], self.streams[song],
                                sequence_codec=self.codec if ordered else None,encoded=self.encoded[song] if ordered else None)
            index.supervision_valid = np.ones(len(index.boundaries)-1,dtype=np.bool_)
            index.supervision_counts = np.zeros(len(index.boundaries)-1,dtype=np.int64)
            for step, unknown in zip(self.streams[song],self.unknown_steps[song]):
                if unknown:index.supervision_valid[index.segment(step.end)] = False
                if not unknown and step.weight>0:index.supervision_counts[index.segment(step.start)] += 1
            if index.nbytes > self._phrase_budget:
                self._phrase_uncached += 1
                return index
            while self._phrases and self._phrase_bytes+index.nbytes > self._phrase_budget:
                _, old = self._phrases.popitem(last=False)
                self._phrase_bytes -= old.nbytes
            self._phrases[song] = index
            self._phrase_bytes += index.nbytes
            return index

    def rebuild_phrase_row(self, row, song, modified, begin, first, end):
        """Edited prefix features are rebuilt; future auxiliary labels disabled."""
        if self.phrase_context_policy != 'off':
            index = self._phrase_index(song).edited(modified, begin, end)
            row.update(index.rows(modified, first, end))
            if self.phrase_context_policy in ('ordered_phrase_v2','aligned_phrase_v3'):
                row.update(index.sequence_rows(np.arange(first,end),row['step_beat'].numpy()))
            if self.phrase_context_policy=='aligned_phrase_v3':
                row.update(prepare_candidate_phrase_inputs(index,np.arange(first,end),row['target_beat'].numpy()))
                row['candidate_step_beat'] = row['target_beat'].clone()
        return row

    def __len__(self): return len(self.samples)

    def set_symbolic_only(self, enabled):
        """Switch only between loader lifetimes; labels and clock stay intact."""
        self.symbolic_only = bool(enabled)

    def _bank(self, song_index):
        with self._bank_lock:
            return self._bank_locked(song_index)

    def _bank_locked(self, song_index):
        song = self.base.songs[song_index][0]
        audio = song.audio
        key = self._bank_keys[song_index]
        if key not in self._banks:
            self._bank_misses += 1
            def build():
                source_song = song
                if hasattr(audio,'store') and hasattr(audio,'path'):
                    source_song = replace(song,audio=audio.store.get(audio.path))
                return build_audio_bank(source_song,self.clocks[song_index],self.audio_grid_seconds)
            bank = (self._disk_bank.get(bank_identity(song,self.audio_grid_seconds,key),build)
                    if self._disk_bank is not None else build())
            size = sum(t.numel()*t.element_size() for t in bank)
            if size > self._bank_budget:
                return bank
            while self._banks and self._bank_bytes+size > self._bank_budget:
                _,old = self._banks.popitem(last=False)
                self._bank_bytes -= sum(t.numel()*t.element_size() for t in old)
                self._bank_evictions += 1
            self._banks[key] = bank
            self._bank_bytes += size
        else:
            self._bank_hits += 1
        self._banks.move_to_end(key)
        return self._banks[key]

    @staticmethod
    def _state_at(index, stream, absolute):
        step = stream[absolute]
        if index is None or step.start < 0 or absolute == 0:
            return torch.zeros(1,STATE_DIM),torch.ones(1,dtype=torch.bool)
        previous = stream[absolute-1]
        # Include the committed head at the cursor, but never admit the
        # current target even for arbitrarily close exact Fraction timings.
        cutoff = min(step.start + Fraction(1,10**12), step.end) if previous.kind == NOTE else step.start
        state,mask,_ = index.at(step.start,include_current=True,visible_before_beat=cutoff,
                                visibility_version='causal_v2',include_metadata=False)
        return state,mask

    @staticmethod
    def _object_bytes(value, seen=None):
        """Conservative Python/container and backing-array estimate, once per song."""
        if seen is None: seen=set()
        identity=id(value)
        if identity in seen: return 0
        seen.add(identity)
        if torch.is_tensor(value): return sys.getsizeof(value)+value.numel()*value.element_size()
        if isinstance(value,np.ndarray): return sys.getsizeof(value)+(0 if value.flags.owndata else value.nbytes)
        size=sys.getsizeof(value)
        if isinstance(value,dict): return size+sum(StreamDataset._object_bytes(k,seen)+StreamDataset._object_bytes(v,seen) for k,v in value.items())
        if isinstance(value,(list,tuple,set,frozenset)):
            return size+sum(StreamDataset._object_bytes(v,seen) for v in value)
        return size

    def _song_states(self,song):
        with self._state_lock:
            if song in self._states:
                self._state_hits += 1
                self._states.move_to_end(song)
                index,rows,_ = self._states[song]
                return index,rows
            self._state_misses += 1
            source=self.base.songs[song][0]
            points=self.views[song].context_points
            index=SustainStateIndex(points,source.tempo,source.first) if points else None
            stream=self.streams[song]
            def store(entry_index,entry_rows,size):
                while self._states and self._state_bytes+size > self._state_budget:
                    _,(_,_,old_size)=self._states.popitem(last=False)
                    self._state_bytes -= old_size
                    self._state_evictions += 1
                self._states[song]=(entry_index,entry_rows,size)
                self._state_bytes += size
            def index_only():
                self._state_uncached += 1
                if index is not None and self._state_budget:
                    # Populate per-index acceleration tables before sizing;
                    # later on-demand queries must not silently grow an
                    # unaccounted cache entry past its byte budget.
                    if len(stream)>1:
                        self._state_at(index,stream,len(stream)-1)
                    for marker in getattr(index,'markers',()):
                        if float(marker.get('end_second',0.)) > float(marker.get('second',0.)):
                            beat=marker['source_beat']
                            index.at(beat,include_current=True,visible_before_beat=beat+Fraction(1,10**12),
                                     visibility_version='causal_v2',include_metadata=False)
                            break
                    # Judge references are process-global shared objects, not
                    # independently retained data per chart index.
                    owned={key:value for key,value in vars(index).items() if key != '_references'}
                    size=self._object_bytes(owned)
                    if size <= self._state_budget:
                        store(index,None,size)
                return index,None
            # Do not build a full-song cache if even one empty state per step
            # cannot fit. The fallback computes only requested rows.
            if not self._state_budget or len(stream)*(STATE_DIM*4+1)>self._state_budget:
                return index_only()
            rows=[]; row_bytes=sys.getsizeof(rows)
            for absolute in range(len(stream)):
                state,mask=self._state_at(index,stream,absolute)
                if self.candidate_state_policy=='selected_native_v1':
                    from .stream_candidate_state import candidate_state_at
                    candidate_state,candidate_mask=candidate_state_at(index,stream,absolute)
                    rows.append((state,mask,candidate_state,candidate_mask))
                else:rows.append((state,mask))
                row_bytes += self._object_bytes(rows[-1]) + 8
                if row_bytes > self._state_budget:
                    return index_only()
            # Once rows are materialized the parsed index is unnecessary.
            # It can reference large process-global judge data, which must not
            # be retained or charged as a separate cache entry per song.
            size=self._object_bytes(rows)
            if size > self._state_budget:
                return index_only()
            store(None,tuple(rows),size)
            return None,rows

    def cache_stats(self):
        with self._bank_lock, self._state_lock:
            return {'bank':{'bytes':self._bank_bytes,'budget_bytes':self._bank_budget,
                            'entries':len(self._banks),'hits':self._bank_hits,'misses':self._bank_misses,
                            'evictions':self._bank_evictions},
                    'state':{'estimated_bytes':self._state_bytes,'budget_bytes':self._state_budget,
                             'entries':len(self._states),'hits':self._state_hits,'misses':self._state_misses,
                             'evictions':self._state_evictions,'uncached':self._state_uncached},
                    'symbol':{'estimated_bytes':self._symbol_bytes,'budget_bytes':self._symbol_budget,
                              'entries':len(self._symbols),'hits':self._symbol_hits,
                              'misses':self._symbol_misses,'evictions':self._symbol_evictions},
                    'phrase':{'estimated_bytes':self._phrase_bytes,'budget_bytes':self._phrase_budget,
                              'entries':len(self._phrases),'uncached':self._phrase_uncached},
                    'plan':{'bytes':sum(t.numel()*t.element_size() for plan in self.plans for t in plan),
                            'policy':'eager_labels_not_in_cache_budget'},
                    'disk_bank':self._disk_bank.stats() if self._disk_bank is not None else None,
                    'form_audio':self.base.audio_store.stats() if hasattr(self.base,'audio_store') else None}

    def _symbol_rows(self, song, first, end):
        """Copy a bounded chart-label cache; callers may freely mutate rows."""
        with self._state_lock:
            if song in self._symbols:
                self._symbol_hits += 1
                self._symbols.move_to_end(song)
                return {key: value[first:end].clone() for key,value in self._symbols[song][0].items()}
            self._symbol_misses += 1
            count = len(self.streams[song])
            atoms, details = self.widths[song]
            # Reject oversized charts before allocating full-song tensors.
            estimate = count * (7*8 + atoms*8 + 16 + details*8 + 4*8)
            cache = bool(self._symbol_budget and estimate <= self._symbol_budget)
            lo, hi = (0,count) if cache else (first,end)
            rows = self._build_symbol_rows(song,lo,hi)
            size = self._object_bytes(rows)
            if cache and size <= self._symbol_budget:
                while self._symbols and self._symbol_bytes+size > self._symbol_budget:
                    _,(_,old_size) = self._symbols.popitem(last=False)
                    self._symbol_bytes -= old_size
                    self._symbol_evictions += 1
                self._symbols[song] = (rows,size)
                self._symbol_bytes += size
                return {key: value[first:end].clone() for key,value in rows.items()}
            return {key: value[first-lo:end-lo].clone() for key,value in rows.items()}

    def _build_symbol_rows(self, song, first, end):
        stream, encoded = self.streams[song], self.encoded[song]
        sequence = stream[first:end]
        length = len(sequence)
        atom_width, detail_width = self.widths[song]
        row = {key: torch.zeros(length,dtype=torch.long) for key in
               ('history_kind','history_delta','history_shape','history_phase','target_kind','target_delta','target_shape')}
        row['history_atoms'] = torch.zeros(length,atom_width,dtype=torch.long)
        row['history_phase_mask'] = torch.zeros(length,8,dtype=torch.bool)
        row['target_phase_mask'] = torch.zeros(length,8,dtype=torch.bool)
        row['target_details'] = torch.full((length,detail_width),-1,dtype=torch.long)
        row['step_beat'] = torch.tensor([float(s.start) for s in sequence],dtype=torch.float64)
        row['step_index'] = torch.arange(first,end,dtype=torch.long)
        row['step_second'] = torch.tensor([s.start_second for s in sequence],dtype=torch.float64)
        row['target_second'] = torch.tensor([s.end_second for s in sequence],dtype=torch.float64)
        row['target_beat'] = torch.tensor([float(s.end) for s in sequence],dtype=torch.float64)
        row['history_gap_beats'] = torch.zeros(length,dtype=torch.float64)
        for i,step in enumerate(sequence):
            absolute = first+i
            prior_delta, prior_event = encoded[absolute-1] if absolute else (0,None)
            row['history_kind'][i] = stream[absolute-1].kind if absolute else BOS
            row['history_delta'][i] = prior_delta
            if absolute:row['history_gap_beats'][i]=float(stream[absolute-1].end-stream[absolute-1].start)
            if prior_event is not None:
                row['history_shape'][i],row['history_phase'][i] = prior_event['shape'],prior_event['phase']
                row['history_atoms'][i,:len(prior_event['atoms'])] = torch.tensor(prior_event['atoms'])
                row['history_phase_mask'][i] = torch.tensor(prior_event['phase_mask'],dtype=torch.bool)
            delta,event = encoded[absolute]
            row['target_kind'][i],row['target_delta'][i] = step.kind,delta
            if event is not None:
                row['target_shape'][i] = event['shape']
                row['target_phase_mask'][i] = torch.tensor(event['phase_mask'])
                row['target_details'][i,:len(event['details'])] = torch.tensor(event['details'])
        return row

    def __getitem__(self, index):
        song, start, end, window_weight = self.samples[index]
        first = max(0, start-self.lead_in)
        stream, encoded = self.streams[song], self.encoded[song]
        sequence = stream[first:end]
        length = len(sequence)
        row = self._symbol_rows(song,first,end)
        row['loss_mask'] = torch.tensor([first+i >= start for i in range(length)], dtype=torch.bool)
        row['configuration_mask'] = torch.ones(length,dtype=torch.bool)
        row['sample_weight'] = torch.tensor([s.weight*(self.target_window_weights[song][first+i]
                                             if self.history_window_policy == 'block_aligned_v2' or self.window_balance_policy=='sqrt_window_v2' else window_weight)*self.chart_weights[song]
                                             if first+i >= start and not self.unknown_steps[song][first+i] else 0.
                                             for i,s in enumerate(sequence)], dtype=torch.float32)
        features,exit_mask,exit_labels = self.continuations[song]
        row['continuation_features'] = features[first:end].clone()
        row['transition_exit_mask'] = exit_mask[first:end].clone() & row['loss_mask'] & row['sample_weight'].gt(0)
        row['transition_exit_labels'] = exit_labels[first:end].clone()
        state_index,cached_states = self._song_states(song)
        states, state_masks = [], []
        candidate_states,candidate_state_masks = [], []
        for i, step in enumerate(sequence):
            absolute = first+i
            cached = cached_states[absolute] if cached_states is not None else None
            state,state_mask=(cached[:2] if cached is not None else self._state_at(state_index,stream,absolute))
            states.append(state); state_masks.append(state_mask)
            if self.candidate_state_policy=='selected_native_v1':
                if cached is not None:candidate_state,candidate_mask=cached[2:]
                else:
                    from .stream_candidate_state import candidate_state_at
                    candidate_state,candidate_mask=candidate_state_at(state_index,stream,absolute)
                candidate_states.append(candidate_state)
                candidate_state_masks.append(candidate_mask)
        row['action_state'] = pad_sequence(states, batch_first=True)
        row['action_state_mask'] = pad_sequence(state_masks, batch_first=True, padding_value=True)
        if candidate_states:
            row['candidate_action_state'] = pad_sequence(candidate_states,batch_first=True)
            row['candidate_action_state_mask'] = pad_sequence(candidate_state_masks,batch_first=True,padding_value=True)
        row['condition'] = torch.as_tensor(self.conditions[song], dtype=torch.float32)
        row['audio_end_sec'] = torch.tensor(self._durations[song],dtype=torch.float64)
        row['audio_end_beat'] = torch.tensor(self._end_beats[song],dtype=torch.float64)
        row['song_index'] = song
        row['_sample_index'] = int(index)
        row['_window_weight'] = self.plan_weights[index]*self.chart_weights[song]
        row['_song_bank'] = self._symbolic_bank if self.symbolic_only else self._bank(song)
        row['_song_plan'] = self._symbolic_plan if self.symbolic_only else self.plans[song]
        row['_literal_atom_types']=self._literal_atom_types
        if self.phrase_context_policy != 'off':
            phrase_index = self._phrase_index(song)
            row.update(phrase_index.rows(stream,first,end,supervise=True))
            trusted = row['loss_mask'] & row['sample_weight'].gt(0)
            segments = np.clip(np.searchsorted(phrase_index.boundaries,row['step_beat'].numpy(),side='right')-1,
                               0,len(phrase_index.supervision_valid)-1)
            trusted &= torch.from_numpy(phrase_index.supervision_valid[segments])
            row['target_phrase_mask'] &= trusted
            counts = torch.from_numpy(np.maximum(1,phrase_index.supervision_counts[segments]).astype(np.float32))
            row['target_phrase_weight'] *= row['sample_weight']/counts
            if self.phrase_context_policy in ('ordered_phrase_v2','aligned_phrase_v3'):
                row.update(phrase_index.sequence_rows(row['step_index'].numpy(),row['step_beat'].numpy(),
                                                      target_mask=row['target_phrase_mask'].numpy()))
            if self.phrase_context_policy=='aligned_phrase_v3':
                # Candidate clock is kept in a separate namespace. The main
                # timing path continues to consume the departure segment only.
                row.update(prepare_candidate_phrase_inputs(phrase_index,row['step_index'].numpy(),
                    row['target_beat'].numpy()))
                row['candidate_step_beat'] = row['target_beat'].clone()
        return row


def collate_stream(items):
    if not items: raise ValueError('empty stream batch')
    output = {}
    output['_sample_indices'] = [item.get('_sample_index') for item in items]
    if any('phrase_progress' in item for item in items):
        if not all('phrase_progress' in item for item in items):
            raise ValueError('Cannot mix phrase-enabled and legacy stream rows')
        for key in PHRASE_ROW_FIELDS:
            output[key] = pad_sequence([item[key] for item in items], batch_first=True)
    if any('_phrase_sequence_owner' in item for item in items):
        output.update(collate_phrase_sequences(items))
    if any('candidate__phrase_sequence_owner' in item for item in items):
        output.update(collate_candidate_phrase_sequences(items))
        share_aligned_phrase_sequences(items,output)
    if any('candidate_phrase_progress' in item for item in items):
        if not all('candidate_phrase_progress' in item for item in items):
            raise ValueError('Cannot mix aligned and legacy phrase rows')
        for key in PHRASE_ROW_FIELDS:
            output['candidate_'+key] = pad_sequence([item['candidate_'+key] for item in items],batch_first=True)
    for key in ('history_kind','history_delta','history_shape','history_phase','history_atoms','history_phase_mask',
                'target_kind','target_delta','target_shape','target_phase_mask','target_details',
                'step_beat','step_second','step_index','target_second','target_beat','history_gap_beats','loss_mask','configuration_mask','sample_weight',
                'continuation_features','transition_exit_mask','transition_exit_labels'):
        if key in ('target_beat','history_gap_beats') and any(key not in item for item in items):continue
        values = [item[key] for item in items]
        # Atom/detail widths may differ by song; preserve every member.
        if key in ('history_atoms','target_details'):
            width = max(x.size(1) for x in values)
            fill = -1 if key == 'target_details' else 0
            values = [torch.nn.functional.pad(x, (0,width-x.size(1)), value=fill) for x in values]
        output[key] = pad_sequence(values, batch_first=True, padding_value=-1 if key == 'target_details' else 0)
    output['step_mask'] = torch.arange(output['history_kind'].size(1))[None] < torch.tensor([len(x['history_kind']) for x in items])[:,None]
    state_width = max(x['action_state'].size(1) for x in items)
    states = [torch.nn.functional.pad(x['action_state'], (0,0,0,state_width-x['action_state'].size(1))) for x in items]
    masks = [torch.nn.functional.pad(x['action_state_mask'], (0,state_width-x['action_state_mask'].size(1)), value=True) for x in items]
    output['action_state'] = pad_sequence(states, batch_first=True)
    output['action_state_mask'] = pad_sequence(masks, batch_first=True, padding_value=True)
    if any('candidate_action_state' in item for item in items):
        if not all('candidate_action_state' in item for item in items):
            raise ValueError('Cannot mix candidate state policies')
        width = max(item['candidate_action_state'].size(1) for item in items)
        candidate_states = [torch.nn.functional.pad(item['candidate_action_state'],
            (0,0,0,width-item['candidate_action_state'].size(1))) for item in items]
        candidate_masks = [torch.nn.functional.pad(item['candidate_action_state_mask'],
            (0,width-item['candidate_action_state_mask'].size(1)),value=True) for item in items]
        output['candidate_action_state'] = pad_sequence(candidate_states,batch_first=True)
        output['candidate_action_state_mask'] = pad_sequence(candidate_masks,batch_first=True,padding_value=True)
    if any('candidate_step_beat' in item for item in items):
        output['candidate_step_beat'] = pad_sequence([item['candidate_step_beat'] for item in items],batch_first=True)
    output['condition'] = torch.stack([item['condition'] for item in items])
    output['audio_end_sec'] = torch.stack([item['audio_end_sec'] for item in items])
    output['audio_end_beat'] = torch.stack([item['audio_end_beat'] for item in items])
    output['audio_plan_target'] = pad_sequence([item['_song_plan'][0] for item in items], batch_first=True)
    output['audio_plan_exposure'] = pad_sequence([item['_song_plan'][2] for item in items], batch_first=True)
    output['audio_plan_weight'] = pad_sequence([item['_song_plan'][1] * item.get('_plan_scale', item['_window_weight'])
                                                for item in items], batch_first=True)
    output['audio_plan_blank_weight'] = pad_sequence([item['_song_plan'][3] * item.get('_plan_scale', item['_window_weight'])
                                                      for item in items], batch_first=True)
    unique, inverse, identities = [], [], {}
    for item in items:
        identity = id(item['_song_bank'])
        if identity not in identities:
            identities[identity] = len(unique); unique.append(item['_song_bank'])
        inverse.append(identities[identity])
    output['song_audio'] = pad_sequence([bank[0] for bank in unique], batch_first=True)
    output['song_seconds'] = pad_sequence([bank[1] for bank in unique], batch_first=True)
    output['song_beats'] = pad_sequence([bank[2] for bank in unique], batch_first=True)
    output['song_mask'] = torch.arange(output['song_audio'].size(1))[None] < torch.tensor([len(bank[0]) for bank in unique])[:,None]
    output['song_inverse'] = torch.tensor(inverse, dtype=torch.long)
    active = output['target_kind'].eq(NOTE) & output['loss_mask'] & output['configuration_mask'] & output['sample_weight'].gt(0)
    flat_shapes = output['target_shape'].reshape(-1)
    flat_active = active.reshape(-1)
    output['_detail_groups'] = [(int(shape), (flat_active & flat_shapes.eq(shape)).nonzero(as_tuple=False).flatten())
                                for shape in torch.unique(flat_shapes[flat_active]).tolist() if shape != 0]
    flat_phases = output['target_phase_mask'].reshape(-1,8)
    output['_detail_phase_groups'] = []
    for shape,indices in output['_detail_groups']:
        pairs = flat_phases[indices].nonzero(as_tuple=False)
        output['_detail_phase_groups'].append((shape,indices,pairs[:,0],pairs[:,1]))
    output['_literal_indices'] = (flat_active & flat_shapes.eq(0)).nonzero(as_tuple=False).flatten()
    if all('_literal_atom_types' in item for item in items):
        from .stream_literal import pack_literal_type_blocks
        literal_targets=output['target_details'].flatten(0,1)[output['_literal_indices']]
        output['_literal_type_blocks']=pack_literal_type_blocks(literal_targets,items[0]['_literal_atom_types'])
    if 'phrase_sequence_inverse' in output:
        output.update(collate_phrase_music(output))
    if 'candidate_phrase_sequence_inverse' in output:
        candidate = {key[len('candidate_'):]:value for key,value in output.items() if key.startswith('candidate_')}
        candidate['song_inverse'] = output['song_inverse']
        output.update({'candidate_'+key:value for key,value in collate_phrase_music(candidate).items()})
    return output


class StreamInferenceContext:
    """Construct the same causal inputs from an actual committed stream.

    The full prefix is used for sustained state, even when symbolic context is
    truncated. Pass NOTE and WAIT StreamStep records; never pass an EOS record.
    """
    def __init__(self, song, codec, delta_vocab, condition, *, lead_in=64, audio_bank=None,
                 audio_grid_seconds=GRID_SECONDS, history_window_policy='rolling_legacy_v1',rollout_steps=16,
                 continuation_policy='off',timing_output_policy='relative_delta_v1',phrase_context_policy='off',
                 candidate_state_policy='cursor_v1'):
        self.song, self.codec, self.delta_vocab = song, codec, delta_vocab
        if phrase_context_policy not in PHRASE_POLICIES:raise ValueError('invalid phrase_context_policy')
        self.phrase_context_policy = phrase_context_policy
        if candidate_state_policy not in ('cursor_v1','selected_native_v1'):
            raise ValueError('invalid candidate_state_policy')
        self.candidate_state_policy = candidate_state_policy
        if timing_output_policy not in TIMING_POLICIES:raise ValueError('invalid timing output policy')
        self.timing_output_policy=timing_output_policy
        self.clock = TempoClock(song.tempo, song.first)
        self.condition = condition.to_vector() if hasattr(condition, 'to_vector') else torch.as_tensor(condition, dtype=torch.float32)
        self.lead_in = int(lead_in)
        if self.lead_in < 0: raise ValueError('lead_in must be nonnegative')
        if history_window_policy not in ('block_aligned_v2','rolling_legacy_v1'):
            raise ValueError('invalid history_window_policy')
        self.history_window_policy = history_window_policy
        if continuation_policy not in CONTINUATION_POLICIES:raise ValueError('invalid continuation_policy')
        self.continuation_policy = continuation_policy
        self.rollout_steps = int(rollout_steps)
        if self.rollout_steps < 1: raise ValueError('rollout_steps must be positive')
        self.audio_grid_seconds = _grid_seconds(audio_grid_seconds)
        self.bank = audio_bank if audio_bank is not None else build_audio_bank(song, self.clock, self.audio_grid_seconds)
        self._reset_prefix()

    def _reset_prefix(self):
        self._phrase = (PhraseIndex(self.bank, beat_at_second(self.clock,self.song.audio.duration_sec),
                                   sequence_codec=self.codec if self.phrase_context_policy in ('ordered_phrase_v2','aligned_phrase_v3') else None)
                        if self.phrase_context_policy != 'off' else None)
        self._committed = []
        self._incremental_state = IncrementalState(self.song.tempo,self.song.first)
        self._max_atoms = 1
        self._rows = [(BOS,0,None,Fraction(0),torch.zeros(1,STATE_DIM),torch.ones(1,dtype=torch.bool))]
        self._continuation = ContinuationState(self.continuation_policy)
        self._continuation_rows = [torch.zeros(continuation_dim(self.continuation_policy))]

    def _append_step(self, step):
        cursor = self._committed[-1].end if self._committed else Fraction(0)
        if step.kind not in (NOTE, WAIT) or step.start != cursor or step.end < step.start:
            raise ValueError('inference prefix must be contiguous NOTE/WAIT steps from beat zero')
        if step.kind == WAIT and not 0 < step.end-step.start <= 2:
            raise ValueError('WAIT must advance by at most two beats')
        if step.kind == NOTE and not step.text:
            raise ValueError('NOTE must contain its committed event')
        if step.kind == NOTE and (step.end-step.start > 2 or
                                  (step.end == step.start and self._committed)):
            raise ValueError('NOTE delta must be positive except at the first beat-zero step')
        delta = encode_history_gap(self.delta_vocab,step.end-step.start,self.timing_output_policy)
        event = self.codec.encode(step.text) if step.kind == NOTE else None
        if event is not None:
            self._incremental_state.append(EventPoint(step.end,step.text))
            self._max_atoms = max(self._max_atoms,len(event['atoms']))
        if not self._incremental_state.points or step.end < 0:
            state, mask = torch.zeros(1,STATE_DIM), torch.ones(1,dtype=torch.bool)
        else:
            cutoff = step.end + Fraction(1,10**12) if step.kind == NOTE else step.end
            state, mask, _ = self._incremental_state.at(step.end,include_current=True,
                visible_before_beat=cutoff,visibility_version='causal_v2',include_metadata=False)
        self._committed.append(step)
        if self._phrase is not None:self._phrase.append(step,encoded=event)
        self._rows.append((step.kind,delta,event,step.end,state,mask))
        if self.continuation_policy != 'off':
            self._continuation.append(step)
            self._continuation_rows.append(self._continuation.features())
        else:self._continuation_rows.append(torch.zeros(continuation_dim(self.continuation_policy)))
        j = len(self._committed)
        first = max(0,(j//self.rollout_steps)*self.rollout_steps-self.lead_in) if self.history_window_policy == 'block_aligned_v2' else max(0,j-self.lead_in)
        keep = j-first+1
        if len(self._rows) > keep:
            del self._rows[:len(self._rows)-keep]
            del self._continuation_rows[:len(self._continuation_rows)-keep]

    def commit(self, step):
        """Append a previously accepted NOTE/WAIT without replaying its prefix."""
        self._append_step(step)

    def current_batch(self):
        cursor = self._committed[-1].end if self._committed else Fraction(0)
        return self.batch(self._committed,cursor)

    def batch(self, steps, cursor):
        own_prefix = steps is self._committed
        if not own_prefix: steps = list(steps)
        cursor = _fraction(cursor)
        if cursor != (steps[-1].end if steps else Fraction(0)):
            raise ValueError('cursor does not match committed prefix')
        if not own_prefix and (len(steps) < len(self._committed) or steps[:len(self._committed)] != self._committed):
            self._reset_prefix()
        if not own_prefix:
            for step in steps[len(self._committed):]:
                self._append_step(step)
        length = len(self._rows)
        atom_width = self._max_atoms
        result = {key: torch.zeros(1,length,dtype=torch.long) for key in
                  ('history_kind','history_delta','history_shape','history_phase')}
        result['history_atoms'] = torch.zeros(1,length,atom_width,dtype=torch.long)
        result['history_phase_mask'] = torch.zeros(1,length,8,dtype=torch.bool)
        result['continuation_features'] = torch.stack(self._continuation_rows).unsqueeze(0)
        result['step_beat'] = torch.zeros(1,length,dtype=torch.float64)
        result['step_second'] = torch.zeros(1,length,dtype=torch.float64)
        result['history_gap_beats'] = torch.zeros(1,length,dtype=torch.float64)
        result['step_index'] = torch.arange(len(self._committed)-length+1,len(self._committed)+1,dtype=torch.long).unsqueeze(0)
        states, masks = [], []
        for row, (kind,delta,event,beat,state,mask) in enumerate(self._rows):
            previous=len(self._committed)-length+row
            if previous>=0:
                committed=self._committed[previous]
                result['history_gap_beats'][0,row]=float(committed.end-committed.start)
            result['step_beat'][0,row] = float(beat)
            result['step_second'][0,row] = max(0.,second_at_beat(self.clock,beat))
            result['history_kind'][0,row],result['history_delta'][0,row] = kind,delta
            if event is not None:
                result['history_shape'][0,row], result['history_phase'][0,row] = event['shape'], event['phase']
                result['history_atoms'][0,row,:len(event['atoms'])] = torch.tensor(event['atoms'])
                result['history_phase_mask'][0,row] = torch.tensor(event['phase_mask'],dtype=torch.bool)
            states.append(state); masks.append(mask)
        result['action_state'] = pad_sequence(states,batch_first=True).unsqueeze(0)
        result['action_state_mask'] = pad_sequence(masks,batch_first=True,padding_value=True).unsqueeze(0)
        result['step_mask'] = torch.ones(1,length,dtype=torch.bool)
        result['loss_mask'] = torch.zeros(1,length,dtype=torch.bool)
        result['condition'] = self.condition.unsqueeze(0)
        result['audio_end_sec'] = torch.tensor([float(self.song.audio.duration_sec)],dtype=torch.float64)
        result['audio_end_beat'] = torch.tensor([float(beat_at_second(self.clock,self.song.audio.duration_sec))],dtype=torch.float64)
        result['song_audio'] = self.bank[0].unsqueeze(0)
        result['song_seconds'] = self.bank[1].unsqueeze(0)
        result['song_beats'] = self.bank[2].unsqueeze(0)
        result['song_mask'] = torch.ones(1,len(self.bank[0]),dtype=torch.bool)
        result['song_inverse'] = torch.zeros(1,dtype=torch.long)
        if self._phrase is not None:
            first_query = len(self._committed)-length+1
            phrase_rows = self._phrase.rows_at(np.arange(first_query,first_query+length),result['step_beat'][0].numpy())
            result.update({key:value.unsqueeze(0) for key,value in phrase_rows.items()})
            if self.phrase_context_policy in ('ordered_phrase_v2','aligned_phrase_v3'):
                sequences = self._phrase.sequence_rows(np.arange(first_query,first_query+length),result['step_beat'][0].numpy())
                for key in ('phrase_sequence_inverse','target_phrase_sequence_inverse'):sequences[key] = sequences[key].unsqueeze(0)
                result.update(sequences)
                result.update(collate_phrase_music(result))
        return result

    def candidate_batch(self, beat):
        """One landing query after the current committed prefix.

        Return candidate_* inputs with batch/time dimensions of one. Callers
        may merge these into current_batch without changing timing inputs.
        """
        beat = _fraction(beat)
        cursor = self._committed[-1].end if self._committed else Fraction(0)
        if beat < cursor:raise ValueError('candidate precedes committed cursor')
        result = {}
        if self.candidate_state_policy=='selected_native_v1':
            if not self._incremental_state.points or beat < 0:
                state,mask = torch.zeros(1,STATE_DIM),torch.ones(1,dtype=torch.bool)
            else:
                last_note = bool(self._committed and self._committed[-1].kind==NOTE)
                cutoff = min(cursor+Fraction(1,10**12),beat) if last_note else cursor
                state,mask,_=self._incremental_state.at(beat,include_current=True,
                    visible_before_beat=cutoff,visibility_version='causal_v2',include_metadata=False)
            result['candidate_action_state'] = state.unsqueeze(0).unsqueeze(0)
            result['candidate_action_state_mask'] = mask.unsqueeze(0).unsqueeze(0)
        if self.phrase_context_policy!='aligned_phrase_v3' or self._phrase is None:
            return result
        query = np.asarray([len(self._committed)],dtype=np.int64)
        candidate = np.asarray([float(beat)],dtype=np.float64)
        result.update(prepare_candidate_phrase_inputs(self._phrase,query,candidate))
        for key in ('candidate_phrase_sequence_inverse','candidate_target_phrase_sequence_inverse'):
            result[key] = result[key].unsqueeze(0)
        for key in PHRASE_ROW_FIELDS:
            result['candidate_'+key] = result['candidate_'+key].unsqueeze(0)
        candidate_music = {key[len('candidate_'):]:value for key,value in result.items()
                           if key.startswith('candidate_')}
        candidate_music['song_inverse'] = torch.zeros(1,dtype=torch.long)
        result.update({'candidate_'+key:value for key,value in collate_phrase_music(candidate_music).items()})
        result['candidate_step_beat'] = torch.as_tensor(candidate).reshape(1,1)
        return result

    candidate_phrase_batch = candidate_batch


def recovery_sample(dataset, song_index, prefix, target, *, weight=1.,continuation_policy=None,
                    configuration_trusted=True,correspondence='authored'):
    """One supervised reference decision after a real committed prefix."""
    song = dataset.base.songs[song_index][0]
    bank = dataset._bank(song_index)
    cursor = prefix[-1].end if prefix else Fraction(0)
    policy=dataset.continuation_policy if continuation_policy is None else continuation_policy
    context = StreamInferenceContext(song,dataset.codec,dataset.delta_vocab,dataset.conditions[song_index],
                                     lead_in=dataset.lead_in,audio_bank=bank,audio_grid_seconds=dataset.audio_grid_seconds,
                                     history_window_policy=dataset.history_window_policy,rollout_steps=dataset.targets,
                                     continuation_policy=policy,timing_output_policy=getattr(dataset,'timing_output_policy','relative_delta_v1'),
                                     phrase_context_policy=getattr(dataset,'phrase_context_policy','off'),
                                     candidate_state_policy=getattr(dataset,'candidate_state_policy','cursor_v1'))
    batch = context.batch(prefix,cursor)
    row = {key: value[0] for key,value in batch.items() if key.startswith('history_') or key in PHRASE_ROW_FIELDS or key in
           ('step_beat','step_second','step_index','loss_mask','action_state','action_state_mask','continuation_features')}
    if getattr(dataset,'phrase_context_policy','off') in ('ordered_phrase_v2','aligned_phrase_v3'):
        for key,value in batch.items():
            if key.startswith(('phrase_sequence_','target_phrase_sequence_','_phrase_sequence_','_target_phrase_sequence_')):
                row[key] = value[0] if key in ('phrase_sequence_inverse','target_phrase_sequence_inverse') else value
    length = len(row['history_kind'])
    for key in ('target_kind','target_delta','target_shape'):
        row[key] = torch.zeros(length,dtype=torch.long)
    row['target_phase_mask'] = torch.zeros(length,8,dtype=torch.bool)
    event = dataset.codec.encode(target.text) if target.kind == NOTE else None
    width = max(1,len(event['details']) if event is not None else 0)
    row['target_details'] = torch.full((length,width),-1,dtype=torch.long)
    row['target_kind'][-1] = target.kind
    row['target_delta'][-1] = encode_history_gap(dataset.delta_vocab,target.end-target.start,
        getattr(dataset,'timing_output_policy','relative_delta_v1')) if target.kind != EOS else 0
    if event is not None:
        row['target_shape'][-1] = event['shape']
        row['target_phase_mask'][-1] = torch.tensor(event['phase_mask'])
        row['target_details'][-1,:len(event['details'])] = torch.tensor(event['details'])
    row['target_second'] = row['step_second'].clone()
    row['target_second'][-1] = target.end_second
    row['target_beat'] = row['step_beat'].clone()
    row['target_beat'][-1] = float(target.end)
    if getattr(dataset,'phrase_context_policy','off')=='aligned_phrase_v3':
        result = prepare_candidate_phrase_inputs(context._phrase,row['step_index'].numpy(),
            row['target_beat'].numpy())
        row.update(result)
        row['candidate_step_beat'] = row['target_beat'].clone()
    if getattr(dataset,'candidate_state_policy','cursor_v1')=='selected_native_v1':
        selected = context.candidate_batch(target.end)
        row['candidate_action_state'] = row['action_state'].clone()
        row['candidate_action_state_mask'] = row['action_state_mask'].clone()
        state = selected['candidate_action_state'][0,0]
        mask = selected['candidate_action_state_mask'][0,0]
        width = max(row['candidate_action_state'].size(1),state.size(0))
        row['candidate_action_state'] = torch.nn.functional.pad(row['candidate_action_state'],
            (0,0,0,width-row['candidate_action_state'].size(1)))
        row['candidate_action_state_mask'] = torch.nn.functional.pad(row['candidate_action_state_mask'],
            (0,width-row['candidate_action_state_mask'].size(1)),value=True)
        row['candidate_action_state'][-1] = torch.nn.functional.pad(state,(0,0,0,width-state.size(0)))
        row['candidate_action_state_mask'][-1] = torch.nn.functional.pad(mask,(0,width-mask.size(0)),value=True)
    row['loss_mask'][-1] = True
    row['configuration_mask'] = torch.zeros(length,dtype=torch.bool)
    row['configuration_mask'][-1] = bool(configuration_trusted)
    row['sample_weight'] = torch.zeros(length,dtype=torch.float32)
    row['sample_weight'][-1] = float(weight)*dataset.chart_weights[song_index]
    row['transition_exit_mask'] = torch.zeros(length,dtype=torch.bool)
    row['transition_exit_labels'] = torch.zeros(length,dtype=torch.bool)
    if policy != 'off':
        eligible,label = context._continuation.exit_label(target)
        row['transition_exit_mask'][-1] = eligible and row['sample_weight'][-1] > 0
        row['transition_exit_labels'][-1] = label
    row['condition'] = dataset.conditions[song_index]
    row['audio_end_sec'] = torch.tensor(float(song.audio.duration_sec),dtype=torch.float64)
    row['audio_end_beat'] = torch.tensor(float(beat_at_second(dataset.clocks[song_index],song.audio.duration_sec)),dtype=torch.float64)
    row['song_index'] = song_index
    row['_song_bank'] = bank
    row['_song_plan'] = dataset.plans[song_index]
    row['_window_weight'] = 1.
    row['_plan_scale'] = 0.
    row['_literal_atom_types']=dataset._literal_atom_types
    row['_configuration_correspondence']=correspondence if configuration_trusted else 'time_only'
    if not configuration_trusted and 'target_phrase_mask' in row:
        row['target_phrase_mask'].zero_()
        row['target_phrase_weight'].zero_()
    return row
