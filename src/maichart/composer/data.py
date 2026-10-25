"""Event-level composer dataset built on the existing FormDataset/audio cache."""
from __future__ import annotations

from dataclasses import dataclass,replace
from fractions import Fraction
import json
import hashlib
import os
from pathlib import Path
from collections import OrderedDict
import threading
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from maichart.form.data import FormDataset
from maichart.stream.context import prefix_action_state
from maichart.generation.sustain_state import SustainStateIndex
from maichart.stream.audio_bounds import generation_bounds
from maichart.tempo_clock import TempoClock
from .candidates import (
    COMPOSER_AUDIO_FEATURE_POLICY,
    _ALL_CANONICAL_OFFSETS,
    _select_candidate_offsets,
    candidate_beats,
    candidate_features,
    candidate_index,
    candidate_spec,
    composer_audio_nodes,
    interp_candidate_audio,
)
from .events import (EVENT_SEMANTICS, HEAD_PADS, event_metadata,
                     head_timing_features, history_event_features)
from .event_views import EVENT_VIEW_POLICY, EventViews
from .model import MUSIC_PROFILES,GROUNDED_PROFILE
from .runtime_memory import cache_capacity, host_memory
from .audio_bank import bank_path,load_bank,publish_bank
from .acoustics import AcousticContext
from .planning import PLAN_POLICY
from .workload import ExactWorkloadQuery, prefix_ledger, trailing_workload_targets, event_statistics
from .candidates import CANDIDATE_POLICY, MUSIC_CANDIDATE_POLICY_V6, offset_universe
from maichart.generation.event_cli import _NoteGrammar
from maichart.form.type_guidance import _counts


@dataclass(frozen=True, slots=True)
class ComposerSample:
    song_index: int
    event_index: int
    previous: Fraction | None
    target: Fraction
    note_text: str
    target_index: int
    valid_candidates: int
    eos_target: float
    weight: float = 1.
    eos_weight: float = 1.
    pointer_weight: float = 1.


class ComposerDataset(Dataset):
    def __init__(self, manifest, *, cache_dir='cache/chart_composer_v1',
                 candidate_count=96, history_events=128, action_slots=8,
                 preparation_workers=1, audio_cache_gb=4., limit_charts=0,
                 quality_enabled=True, seed=0, architecture_profile='music_context_v2',
                 action_state_policy='committed_legacy_v1', motif_memory_policy='off', motif_max_events=16):
        started = time.monotonic()
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.codec_atoms=[]; self.atom_to_id={}; self.samples=[]; self.songs=[]
        self.candidate_count=int(candidate_count);self.history_events=int(history_events)
        self.architecture_profile = architecture_profile
        self.action_state_policy=action_state_policy
        self.motif_memory_policy=motif_memory_policy
        self.motif_max_events=int(motif_max_events)
        self.candidate_policy = MUSIC_CANDIDATE_POLICY_V6 if architecture_profile in MUSIC_PROFILES else CANDIDATE_POLICY
        self.canonical_offsets = offset_universe(self.candidate_policy)
        if self.candidate_count < 16:
            raise ValueError('Composer candidate_count must be at least 16 for common onset and rest proposals')
        self.action_slots=int(action_slots);self.seed=int(seed);self.epoch=0
        if not 1 <= self.action_slots <= 64:
            raise ValueError('Composer action_slots must be in [1,64]')
        self.action_slots_requested = self.action_slots
        # The exposed budget covers both stores rather than granting each a
        # full independent budget. Parsed labels/history are separate CPU data.
        self._feature_cache_budget_bytes=int(max(0.,float(audio_cache_gb))*(1024**3)*.75)
        self._feature_cache_effective_budget_bytes=self._feature_cache_budget_bytes
        self._memory_checked_at=float('-inf')
        self._feature_cache_hits=0;self._feature_cache_misses=0
        self._feature_keys={};self._bank_paths={}
        self._bank_hits=self._bank_builds=self._bank_write_errors=0
        self._bank_mmap_loads=self._bank_view_hits=0
        self._data_timing=dict(song_table_sec=0.,audio_lookup_sec=0.,action_state_sec=0.)
        self.base=FormDataset(manifest,cache_dir=cache_dir,max_tokens=1_000_000,
            max_memory_tokens=max(256,history_events),max_region_seconds=999.,
            preparation_workers=preparation_workers,audio_cache_gb=max(0.,float(audio_cache_gb))*.25,
            limit_charts=limit_charts,quality_enabled=quality_enabled,
            preserve_excluded_context=True,quality_profile='musical_evidence_v1',
            rhythm_geometry_profile='none',audio_profile='rational_audio_queries_v1')
        records = list(self.base.songs)
        self._split_cache: dict[str, tuple[str, ...]] = {}
        self._text_meta_cache: dict[str, tuple] = {}
        atom_set: set[str] = set()
        self.max_required_action_slots = 1
        for record in records:
            for point in record[1]:
                text = str(point.note_text)
                atoms = self._split_cache.get(text)
                if atoms is None:
                    atoms = tuple(_NoteGrammar._split_slashes(text))
                    self._split_cache[text] = atoms
                    atom_set.update(atoms)
                self.max_required_action_slots = max(self.max_required_action_slots, len(atoms))
        if self.max_required_action_slots > 64:
            raise ValueError(f'Composer source event requires {self.max_required_action_slots} atoms; maximum is 64')
        self.action_slots = max(self.action_slots, self.max_required_action_slots)
        if self.action_slots != self.action_slots_requested:
            print('composer_prepare ' + json.dumps({'stage': 'action_slots_expanded',
                'requested': self.action_slots_requested, 'effective': self.action_slots}), flush=True)
        self.codec_atoms = sorted(atom_set)
        self.atom_to_id = {atom: index + 1 for index, atom in enumerate(self.codec_atoms)}
        self.candidate_total = 0
        self.candidate_misses = 0
        self.phase_filter_misses = 0
        self.candidate_miss_examples = []
        self.candidate_miss_categories = {}
        self.action_grammar = _NoteGrammar.from_atoms(self.codec_atoms)
        self._song_tables: dict[int, dict] = {}
        self._song_table_locks = {}
        self._context_cache = OrderedDict()
        self._context_cache_limit = 256
        self._feature_cache = OrderedDict()
        self._feature_cache_bytes = 0
        # Keep mapped bank views alive independently of the small resident
        # feature LRU. Random event sampling otherwise reopens the same mmap
        # files thousands of times after the hot cache cycles through charts.
        self._mapped_bank_cache = OrderedDict()
        self._mapped_bank_cache_limit = 768
        self._feature_lock = threading.RLock()

        bounds_cache, bounds_path = self._load_bounds_cache()
        self._song_audio_ends = {}
        self.event_views = {}
        self.supervised_event_indices = {}
        self._song_conditions = {}
        bounds_dirty = False
        total_charts = len(records)
        full_hi = len(self.canonical_offsets)
        max_offset = self.canonical_offsets[-1]
        for song_index, record in enumerate(records):
            song, points, level, weight, state, condition, excluded, annotation = record
            if not points or weight <= 0:
                continue
            audio_end_sec, updated = self._audio_end_sec(song_index, bounds_cache)
            self._song_audio_ends[song_index] = audio_end_sec
            bounds_dirty = bounds_dirty or updated
            clock = TempoClock(song.tempo, song.first)
            views = EventViews.build(points, song.tempo, song.first, audio_end_sec,
                excluded=excluded, annotation=annotation, action_slots=self.action_slots)
            self.event_views[song_index] = views
            if len(views.context_points)!=len(points):
                from maichart.generation.event_quality import condition_from_events, MIACODE_DURATION_SEMANTICS
                condition=condition_from_events(
                    [p for p,known in zip(views.context_points,views.trusted) if known],
                    song.tempo, level=level, duration_sec=max(audio_end_sec,1e-6),
                    audio=song.audio, audio_first=song.first,
                    duration_semantics=MIACODE_DURATION_SEMANTICS)
            self._song_conditions[song_index]=condition
            points = views.context_points
            previous = None
            prev_sec = 0.0
            supervised=[]
            for event_index, point in enumerate(points):
                beat = point.beat if isinstance(point.beat, Fraction) else Fraction(point.beat)
                beat_sec = float(clock.at(beat))
                local_weight = float(weight) * float(views.local_weights[event_index])
                if not views.trusted[event_index]:
                    previous = beat
                    prev_sec = beat_sec
                    continue
                if (previous is not None and previous>=0
                        and (far_second:=float(clock.at(previous+max_offset)))<audio_end_sec
                        and far_second-prev_sec<=(64. if architecture_profile in MUSIC_PROFILES else 16.)-1e-12):
                    valid_hi=full_hi
                    offsets,lookup,_=_select_candidate_offsets(False,full_hi,self.candidate_count,previous,self.candidate_policy)
                else:
                    _, _, valid_hi, offsets, lookup, _ = candidate_spec(
                        previous,tempo=song.tempo,first=song.first,audio_end_sec=audio_end_sec,
                        candidate_count=self.candidate_count,clock=clock,policy=self.candidate_policy)
                self.candidate_total += 1
                if not offsets:
                    previous = beat
                    prev_sec = beat_sec
                    continue
                offset_needed = beat if previous is None else beat - previous
                target_index = lookup.get(offset_needed, -1)
                if target_index < 0:
                    if previous is not None and offset_needed in self.canonical_offsets:
                        self.phase_filter_misses += 1
                    self.candidate_misses += 1
                    reason = ('beyond_horizon' if offset_needed > max_offset else
                              'pruned' if offset_needed in self.canonical_offsets else 'phase_or_subdivision')
                    self.candidate_miss_categories[reason] = self.candidate_miss_categories.get(reason, 0) + 1
                    if len(self.candidate_miss_examples) < 8:
                        self.candidate_miss_examples.append({
                            'song_index': song_index,
                            'event_index': event_index,
                            'target': str(beat),
                            'previous': str(previous),
                        })
                    previous = beat
                    prev_sec = beat_sec
                    continue
                eos_target, eos_weight = views.eos(event_index)
                self.samples.append(ComposerSample(
                    song_index,
                    event_index,
                    previous,
                    beat,
                    str(point.note_text),
                    target_index,
                    valid_hi,
                    eos_target,
                    local_weight,
                    eos_weight,
                    float(views.window_weight(prev_sec, beat_sec)),
                ))
                supervised.append(event_index)
                previous = beat
                prev_sec = beat_sec
            # A filtered/excluded final target is unknown, not evidence that
            # the earlier accepted event should end the chart.
            self.songs.append(record)
            self.supervised_event_indices[song_index]=tuple(supervised)
            if (song_index + 1) % 250 == 0:
                print('composer_prepare ' + json.dumps({
                    'stage': 'scanning',
                    'completed': song_index + 1,
                    'total': total_charts,
                    'samples': len(self.samples),
                    'elapsed_sec': round(time.monotonic() - started, 2),
                }), flush=True)
        if bounds_dirty and bounds_path is not None:
            self._save_bounds_cache(bounds_path, bounds_cache)
        manifest_path = Path(manifest)
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.is_file() else None
        self.fingerprint = json.dumps({
            'manifest': str(manifest_path.resolve()),
            'manifest_sha256': manifest_hash,
            'candidate_count': self.candidate_count,
            'history_events': self.history_events,
            'action_slots': self.action_slots,
            'samples': len(self.samples),
            'candidate_total': self.candidate_total,
            'candidate_misses': self.candidate_misses,
            'phase_filter_misses': self.phase_filter_misses,
            'audio_feature_policy': COMPOSER_AUDIO_FEATURE_POLICY,
            'candidate_policy': self.candidate_policy,
            'event_view_policy': EVENT_VIEW_POLICY,
            'event_semantics': EVENT_SEMANTICS,
            'architecture_profile':architecture_profile,
            'grounded_inputs':('lens31x26_plan4beat_placement25ms_v1' if architecture_profile==GROUNDED_PROFILE else None),
            'grounded_plan_policy':(PLAN_POLICY
                if architecture_profile==GROUNDED_PROFILE else None),
            'supervised_indices_sha256':hashlib.sha256(json.dumps(
                self.supervised_event_indices,sort_keys=True).encode()).hexdigest(),
            'workload_profile': 'exact_uniform_seconds_trailing_v3',
            'source_sha256': hashlib.sha256(json.dumps([
                (i, views.source_digest, self._song_feature_key(i),
                 self._audio_identity(records[i][0].audio,
                    (self.base.audit.get('charts',[])[i].get('audio_sha256')
                     if i<len(self.base.audit.get('charts',[])) else None)),
                 self._song_conditions[i].to_vector().tolist())
                for i, views in sorted(self.event_views.items())],
                sort_keys=True, default=str).encode()).hexdigest(),
            'quality_sha256': hashlib.sha256(json.dumps(
                [(r[3],r[6],r[7]) for r in records],sort_keys=True,default=str).encode()).hexdigest(),
        }, sort_keys=True)
        if action_state_policy!='committed_legacy_v1' or motif_memory_policy!='off':
            fingerprint=json.loads(self.fingerprint)
            fingerprint.update(action_state_policy=action_state_policy,motif_memory_policy=motif_memory_policy,
                               motif_max_events=self.motif_max_events)
            self.fingerprint=json.dumps(fingerprint,sort_keys=True)
        if not self.samples:
            raise ValueError('Composer has no eligible events after candidate filtering; check manifest and candidate recall')
        print('composer_prepare ' + json.dumps({
            'stage': 'ready',
            'charts': len(self.songs),
            'samples': len(self.samples),
            'candidate_total': self.candidate_total,
            'candidate_misses': self.candidate_misses,
            'phase_filter_misses': self.phase_filter_misses,
            'context_events': sum(len(v.context_points) for v in self.event_views.values()),
            'trusted_label_events': sum(int(v.trusted.sum()) for v in self.event_views.values()),
            'elapsed_sec': round(time.monotonic() - started, 2),
        }), flush=True)
        miss_rate = self.candidate_misses / max(1, self.candidate_total)
        if self.candidate_misses and miss_rate > 0.05:
            examples = '; '.join(f"{item['target']} after {item['previous']}" for item in self.candidate_miss_examples)
            raise ValueError(
                f'Composer candidate coverage failed for {self.candidate_misses}/{self.candidate_total} '
                f'authored events; increase candidate_count or extend the canonical grid. Examples: {examples}'
            )

    def _load_bounds_cache(self) -> tuple[dict[str, float], Path | None]:
        if not self.cache_dir:
            return {}, None
        path = Path(self.cache_dir) / 'composer_prepare' / 'audio_bounds_v1.json'
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
                if isinstance(data, dict):
                    return {str(k): float(v) for k, v in data.items()}, path
            except Exception:
                pass
        return {}, path

    @staticmethod
    def _audio_identity(audio, digest=None):
        source = getattr(audio, 'path', None) or getattr(audio, 'source', None)
        if source and Path(source).is_file():
            path = Path(source).resolve()
            stat = path.stat()
            return (str(path), stat.st_size, stat.st_mtime_ns, digest)
        metadata = getattr(audio, 'metadata', None) or {}
        return (str(source), metadata.get('audio_sha256'), float(audio.duration_sec))

    @staticmethod
    def _save_bounds_cache(path: Path, cache: dict[str, float]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile('w', dir=path.parent, suffix='.tmp', encoding='utf-8', delete=False) as f:
                tmp = Path(f.name)
                json.dump(cache, f)
            tmp.replace(path)
        except Exception:
            pass

    def _audio_end_sec(self, song_index: int, bounds_cache: dict[str, float]) -> tuple[float, bool]:
        song = self.base.songs[song_index][0]
        audio = getattr(song, 'audio', None)
        if hasattr(audio, 'path') and hasattr(audio, 'store'):
            try:
                path = Path(audio.path).resolve()
                stat = path.stat()
                key = f'{path}:{stat.st_size}:{stat.st_mtime_ns}'
                if key in bounds_cache:
                    return bounds_cache[key], False
                if self.cache_dir:
                    from maichart.form.rational_audio import _source_digest
                    from maichart.form.waveform_cache import mapped_waveform
                    from maichart.stream.audio_bounds import exact_signal_end
                    digest = _source_digest(path, self.cache_dir)
                    signal, sr = mapped_waveform(path, digest, self.cache_dir)
                    duration = float(len(signal) / sr)
                    measured = exact_signal_end(signal, sr)
                    end_sec = float(measured if duration - measured >= 0.05 else duration)
                else:
                    end_sec = float(generation_bounds(self._source_song(song_index).audio, trim_exact_silence=True)['effective_end_sec'])
                bounds_cache[key] = end_sec
                return end_sec, True
            except Exception:
                pass
        return float(generation_bounds(self._source_song(song_index).audio, trim_exact_silence=True)['effective_end_sec']), False

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def prepare_concurrency_key(self, index):
        """Serialize samples sharing one source while preparing others in parallel."""
        song_index = self.samples[index].song_index
        audio = self.base.songs[song_index][0].audio
        source = getattr(audio, 'path', None) or getattr(audio, 'source', None)
        if not source:
            source = (getattr(audio, 'metadata', None) or {}).get('source')
        if source:
            return os.path.normcase(os.path.abspath(str(source)))
        return f'composer-song:{song_index}'

    def __len__(self):
        return len(self.samples)

    def _song_table(self, song_index: int) -> dict:
        lock = self._song_table_locks.setdefault(song_index, threading.RLock())
        with lock:
            return self._song_table_unlocked(song_index)

    def _song_table_unlocked(self, song_index: int) -> dict:
        table = self._song_tables.get(song_index)
        if table is not None:
            return table
        song, _points, _level, _weight, _state, condition, _excluded, _annotation = self.base.songs[song_index]
        views = self.event_views[song_index]
        condition = self._song_conditions[song_index]
        points = views.context_points
        n = len(points)
        clock = TempoClock(song.tempo, song.first)
        point_seconds = views.seconds
        atom_ids = np.zeros((n, self.action_slots), dtype=np.int64)
        hist_feats = np.zeros((n, 8), dtype=np.float32)
        counts = [0] * n
        sigs = [None] * n
        chords = [0.0] * n
        holds = [0.0] * n
        slides = [0.0] * n
        touches = [0.0] * n
        last_head_seconds = np.full((n, len(HEAD_PADS)), np.nan, dtype=np.float32)
        last_head_beats = np.full((n, len(HEAD_PADS)), np.nan, dtype=np.float32)
        seen_seconds = np.full(len(HEAD_PADS), np.nan)
        seen_beats = np.full(len(HEAD_PADS), np.nan)
        prior_beat = Fraction(0)
        prior_second = float(clock.at(prior_beat))
        for i, p in enumerate(points):
            text = str(p.note_text)
            meta = event_metadata(text)
            atoms = meta.atoms
            count, sig = len(meta.heads), meta.structure
            is_chord, is_hold, is_slide, is_touch = meta.chord, meta.hold, meta.slide, meta.touch
            if self.architecture_profile in MUSIC_PROFILES:
                atoms=tuple(sorted(atoms))
            for slot, atom in enumerate(atoms):
                atom_ids[i, slot] = self.atom_to_id[atom]
            cur_sec = point_seconds[i]
            hist_feats[i] = history_event_features(text, p.beat-prior_beat, cur_sec-prior_second,
                parsed=self.architecture_profile in MUSIC_PROFILES)
            # Snapshot BEFORE this target: current/future heads never enter
            # candidate-relative recency, including on the first event.
            last_head_seconds[i], last_head_beats[i] = seen_seconds, seen_beats
            for pad in meta.head_pad_indices:
                seen_seconds[pad], seen_beats[pad] = cur_sec, float(p.beat)
            prior_beat = p.beat
            prior_second = cur_sec
            counts[i] = count
            sigs[i] = sig
            chords[i] = is_chord
            holds[i] = is_hold
            slides[i] = is_slide
            touches[i] = is_touch

        future_targets = np.zeros((n, 8), dtype=np.float32)
        future_masks = np.zeros((n, 8), dtype=np.float32)
        log1p_8 = float(np.log1p(8.0))
        for fi in range(n):
            end_w = min(n, fi + 4)
            w_len = end_w - fi
            if w_len > 0:
                before_sec = point_seconds[fi - 1] if fi else 0.0
                gap_clock = np.r_[before_sec, point_seconds[fi:end_w]]
                gaps = np.maximum(0., np.diff(gap_clock))
                head_rate = float(sum(counts[fi:end_w]) / max(sum(gaps), 1e-3))
                future_targets[fi] = (
                    min(1., float(np.log1p(gaps[0]) / log1p_8)),
                    min(1., head_rate / 16.),
                    float(max(gaps) > 1.),
                    sum(chords[fi:end_w]) / w_len,
                    sum(slides[fi:end_w]) / w_len,
                    sum(holds[fi:end_w]) / w_len,
                    sum(touches[fi:end_w]) / w_len,
                    float(fi > 0 and sigs[fi] != sigs[fi - 1]),
                )
                future_masks[fi] = views.future_mask(fi, end_w)
        known_indices=np.flatnonzero(views.trusted)
        known_point_seconds=np.asarray(point_seconds,dtype=np.float64)[known_indices]
        known_head_prefix=np.r_[0.,np.cumsum(np.asarray(counts,dtype=np.float64)[known_indices])]
        action_states = torch.zeros((n, 50), dtype=torch.float32)
        action_ready = bytearray(n)
        if n > 0:
            action_ready[0] = 1
        duration = float(song.audio.duration_sec)
        workload_times = np.arange(.25, duration, .5)
        if not len(workload_times): workload_times = np.asarray([duration * .5])
        workload_query = ExactWorkloadQuery(points, song.tempo, song.first, views.end_sec,
            unknown_intervals=views.unknown_intervals, regions=views.regions)
        workload_targets = workload_query.query(workload_times)
        transition_weights = np.ones(n, dtype=np.float32)
        for i in range(8, n):
            # Weight real authored exits only, once even if several periods
            # match. This does not declare longer repetitions invalid.
            gaps = np.diff(point_seconds[i-8:i])
            current_gap = point_seconds[i] - point_seconds[i-1]
            median = float(np.median(gaps))
            rhythm_exit = (median > 0 and np.all(np.abs(gaps-median)<.01)
                           and abs(current_gap-median)>max(.02,median*.5))
            type_exit = any(i >= 4*p and all(sigs[i-j]==sigs[i-j-p] for j in range(1,3*p+1))
                            and sigs[i]!=sigs[i-p] for p in (1,2,3,4))
            known = np.all(views.trusted[max(0,i-16):i+1]) and views.window_weight(point_seconds[max(0,i-16)],point_seconds[i])>0
            if known and (rhythm_exit or (self.architecture_profile in MUSIC_PROFILES and type_exit)):
                transition_weights[i] = 2.
        table = {
            'clock': clock,
            'audio_duration': duration,
            'points': points,
            'state_index': _state if len(points)==len(_points) else SustainStateIndex(points, song.tempo, song.first),
            'atom_ids': torch.from_numpy(atom_ids),
            'history_features': torch.from_numpy(hist_feats),
            'future_targets': torch.from_numpy(future_targets),
            'future_masks': torch.from_numpy(future_masks),
            'last_head_seconds': last_head_seconds,
            'last_head_beats': last_head_beats,
            'condition': condition.to_vector().detach().cpu().to(torch.float32),
            'action_states': action_states,
            'action_ready': action_ready,
            'point_seconds': np.asarray(point_seconds, dtype=np.float64),
            'head_counts': np.asarray(counts, dtype=np.float32),
            'head_prefix': np.r_[0., np.cumsum(counts, dtype=np.float64)],
            'known_point_seconds': known_point_seconds,
            'known_head_prefix': known_head_prefix,
            'transition_weights': transition_weights,
            'workload_times': workload_times,
            'workload_targets': workload_targets,
            'workload_query': workload_query,
        }
        if self.action_state_policy!='committed_legacy_v1' and any('source_beat' not in m for m in table['state_index'].markers):
            table['state_index']=SustainStateIndex(points,song.tempo,song.first)
        if self.motif_memory_policy!='off':
            from .motif_memory import ComposerMotifIndex
            table['motif_index']=ComposerMotifIndex(getattr(song,'relations',None),
                [p.beat for p in points],atom_ids)
        self._song_tables[song_index] = table
        return table

    def _action_state(self, song_index: int, event_index: int, table: dict) -> torch.Tensor:
        with self._song_table_locks.setdefault(song_index, threading.RLock()):
            return self._action_state_unlocked(song_index,event_index,table)

    def _action_state_unlocked(self, song_index: int, event_index: int, table: dict) -> torch.Tensor:
        if not table['action_ready'][event_index]:
            if self.action_state_policy=='committed_causal_v2':
                from .context_features import composer_action_state
                table['action_states'][event_index]=composer_action_state(
                    table['state_index'],event_index,(),self.action_state_policy)
                table['action_ready'][event_index]=1
                return table['action_states'][event_index]
            action, mask, _ = prefix_action_state(table['state_index'], table['points'], event_index)
            active = action[~mask]
            if active.numel():
                table['action_states'][event_index] = active.mean(0)
            table['action_ready'][event_index] = 1
        return table['action_states'][event_index]

    def _song_feature_key(self, song_index: int):
        cached=self._feature_keys.get(song_index)
        if cached is not None:
            return cached
        song = self.base.songs[song_index][0]
        audio = getattr(song, 'audio', None)
        audio_key = str(getattr(audio, 'path', None) or getattr(audio, 'source', None) or song_index)
        tempo_key = tuple((str(t.beat), float(t.bpm)) for t in song.tempo)
        meter_key = tuple(
            (str(m.beat), int(m.numerator), int(m.denominator), tuple(getattr(m, 'grouping', None) or ()))
            for m in (getattr(song, 'meters', None) or ())
        )
        regions = getattr(song, 'regions', None)
        spans = getattr(song, 'spans', None)
        extent_key = str(regions[-1].end if regions else (spans[-1].end if spans else 0))
        key=(audio_key, float(song.first), tempo_key, meter_key, extent_key)
        self._feature_keys[song_index]=key
        return key

    def _audio_bank_path(self,song_index,key):
        if song_index not in self._bank_paths:
            path=None
            audio=self.base.songs[song_index][0].audio
            if self.cache_dir and hasattr(audio,'store') and hasattr(audio,'path'):
                source_cache=audio.store.disk_cache_path(audio.path)
                if source_cache is not None and source_cache.is_file():
                    path=bank_path(self.cache_dir,key,source_cache)
            self._bank_paths[song_index]=path
        return self._bank_paths[song_index]

    def _song_features(self, song_index: int):
        self._check_cache_headroom()
        key = self._song_feature_key(song_index)
        with self._feature_lock:
            arrays = self._feature_cache.get(key)
        if arrays is None:
            self._feature_cache_misses+=1
            path=self._audio_bank_path(song_index,key)
            arrays = None
            if path is not None:
                cache_key = str(path)
                with self._feature_lock:
                    arrays = self._mapped_bank_cache.get(cache_key)
                    if arrays is not None:
                        self._bank_view_hits += 1
                        self._mapped_bank_cache.move_to_end(cache_key)
                if arrays is None:
                    arrays = load_bank(path)
                    if arrays is not None:
                        self._bank_mmap_loads += 1
                        with self._feature_lock:
                            self._mapped_bank_cache[cache_key] = arrays
                            self._mapped_bank_cache.move_to_end(cache_key)
                            while len(self._mapped_bank_cache) > self._mapped_bank_cache_limit:
                                self._mapped_bank_cache.popitem(last=False)
            if arrays is not None:
                self._bank_hits+=1
            else:
                song = self._source_song(song_index, cache_in_base=False)
                arrays = composer_audio_nodes(song)
                self._bank_builds+=1
                if path is not None:
                    try:
                        arrays=publish_bank(path,arrays)
                    except (OSError,ValueError):
                        self._bank_write_errors+=1
            item_bytes = int(arrays[0].nbytes + arrays[1].nbytes)
            if item_bytes>self._feature_cache_effective_budget_bytes:
                return arrays
            with self._feature_lock:
                self._feature_cache[key] = arrays
                self._feature_cache.move_to_end(key)
                self._feature_cache_bytes += item_bytes
                while self._feature_cache_bytes > self._feature_cache_effective_budget_bytes and self._feature_cache:
                    _, evicted = self._feature_cache.popitem(last=False)
                    self._feature_cache_bytes -= int(evicted[0].nbytes + evicted[1].nbytes)
        else:
            self._feature_cache_hits+=1
            with self._feature_lock:
                self._feature_cache.move_to_end(key)
        return arrays

    def _check_cache_headroom(self):
        # The sole producer owns this LRU. Checking once per second avoids
        # system calls for every event and retains the requested ceiling.
        now=time.monotonic()
        if now-self._memory_checked_at < 1.:
            return
        self._memory_checked_at=now
        memory = host_memory()
        budget=cache_capacity(self._feature_cache_budget_bytes,
                              self._feature_cache_bytes,memory)
        self._feature_cache_effective_budget_bytes=budget
        if memory is not None:
            free_physical = memory['free_physical']
            self._mapped_bank_cache_limit = (128 if free_physical < 3*1024**3 else
                                              256 if free_physical < 4*1024**3 else
                                              512 if free_physical < 6*1024**3 else 768)
            self._context_cache_limit = (64 if free_physical < 3*1024**3 else
                                         128 if free_physical < 5*1024**3 else
                                         256 if free_physical < 8*1024**3 else 512)
        with self._feature_lock:
            while self._feature_cache_bytes>budget and self._feature_cache:
                _,evicted=self._feature_cache.popitem(last=False)
                self._feature_cache_bytes-=int(evicted[0].nbytes+evicted[1].nbytes)
            while len(self._mapped_bank_cache) > self._mapped_bank_cache_limit:
                self._mapped_bank_cache.popitem(last=False)
            context_cache=getattr(self,'_context_cache',None)
            if context_cache is not None:
                while len(context_cache) > getattr(self,'_context_cache_limit',256):
                    context_cache.popitem(last=False)

    def cache_stats(self):
        return {'hits':self._feature_cache_hits,'misses':self._feature_cache_misses,
                'bytes':self._feature_cache_bytes,'budget_bytes':self._feature_cache_budget_bytes,
                'effective_budget_bytes':self._feature_cache_effective_budget_bytes,
                'mapped_bank_hits':self._bank_hits,'bank_builds':self._bank_builds,
                'bank_mmap_loads':self._bank_mmap_loads,
                'bank_view_hits':self._bank_view_hits,
                'bank_write_errors':self._bank_write_errors,
                'cpu_timing':dict(self._data_timing),
                'entries':len(self._feature_cache),
                'mapped_bank_entries':len(self._mapped_bank_cache),
                'mapped_bank_limit':self._mapped_bank_cache_limit,
                'context_entries':len(self._context_cache),
                'context_limit':self._context_cache_limit,
                'scope':'composer_audio_bank'}

    def _source_song(self, index, *, cache_in_base=True):
        song = self.base.songs[index][0]
        audio = getattr(song, 'audio', None)
        if hasattr(audio, 'store') and hasattr(audio, 'path'):
            if not cache_in_base:
                try:
                    from maichart.form.rational_audio import load_rational_audio
                    cache_path = audio.store.disk_cache_path(audio.path)
                    if cache_path is not None and cache_path.is_file():
                        return replace(song, audio=load_rational_audio(cache_path, source=audio.path, cache_dir=audio.store.cache_dir))
                except Exception:
                    pass
            return replace(song, audio=audio.store.get(audio.path))
        return song

    def __getitem__(self, index):
        sample = self.samples[index]
        song = self.base.songs[sample.song_index][0]
        stamp=time.perf_counter()
        table = self._song_table(sample.song_index)
        self._data_timing['song_table_sec']+=time.perf_counter()-stamp
        clock = table['clock']
        stamp=time.perf_counter()
        values, times = self._song_features(sample.song_index)
        self._data_timing['audio_lookup_sec']+=time.perf_counter()-stamp
        offsets, _, valid_floats = _select_candidate_offsets(
            sample.previous is None, sample.valid_candidates, self.candidate_count, sample.previous, self.candidate_policy
        )
        actual_len = len(offsets)
        if actual_len <= 0:
            raise ValueError('Composer sample has no candidate beats')
        candidate_mask = np.zeros(self.candidate_count, dtype=bool)
        base_float = 0.0 if sample.previous is None else float(sample.previous)
        if actual_len < self.candidate_count:
            candidate_mask[actual_len:] = True
            beat_floats = np.empty(self.candidate_count, dtype=np.float64)
            beat_floats[:actual_len] = base_float + valid_floats
            beat_floats[actual_len:] = beat_floats[actual_len - 1]
        else:
            beat_floats = base_float + valid_floats
        if len(clock.beats) == 1:
            seconds = clock.seconds[0] + beat_floats * (60.0 / clock.bpms[0])
            candidates = None
        else:
            base_frac = Fraction(0) if sample.previous is None else sample.previous
            candidates = [base_frac + off for off in offsets]
            if len(candidates) < self.candidate_count:
                candidates.extend([candidates[-1]] * (self.candidate_count - len(candidates)))
            seconds = np.asarray([clock.at(b) for b in candidates], dtype=np.float64)
        candidate_mask |= (seconds < 0.) | (seconds >= self._song_audio_ends[sample.song_index])
        feature = candidate_features(
            candidates,
            sample.previous,
            song.tempo,
            song.first,
            clock=clock,
            candidate_seconds=seconds,
            beat_floats=beat_floats,
            rich_time=True,
        )
        audio = interp_candidate_audio(values, times, seconds)
        features = np.concatenate((audio, feature), axis=1)
        target = sample.target_index
        if target < 0 or target >= self.candidate_count:
            raise ValueError('Composer candidate generator missed an authored target')
        if candidate_mask[target]:
            raise ValueError('Composer supervised target lies outside the valid audio range')
        ev = sample.event_index
        start = max(0, ev - self.history_events)
        context_key = self._song_feature_key(sample.song_index)
        with self._feature_lock:
            context = self._context_cache.get(context_key)
            if context is not None:
                self._context_cache.move_to_end(context_key)
        if context is None:
            context = AcousticContext(values, times, table['audio_duration'], validated_bank=True)
            with self._feature_lock:
                existing = self._context_cache.get(context_key)
                if existing is None:
                    self._context_cache[context_key] = context
                    self._context_cache.move_to_end(context_key)
                    while len(self._context_cache) > self._context_cache_limit:
                        self._context_cache.popitem(last=False)
                else:
                    context = existing
        duration = table['audio_duration']
        # Uniform whole-song probes include silence and are independent of
        # event density; deterministic phases vary the centers across samples.
        uniform = (np.arange(16) + ((index % 31) + .5) / 31.) * duration / 16.
        workload_seconds = np.r_[seconds, uniform]
        workload_audio = context.at(workload_seconds)
        workload_target = table['workload_query'].query(workload_seconds)
        workload_weight = np.zeros(len(workload_seconds), dtype=np.float32)
        if actual_len:
            edges = np.r_[seconds[0], (seconds[:actual_len - 1] + seconds[1:actual_len]) * .5,
                          seconds[actual_len - 1]]
            widths = np.maximum(np.diff(edges), 1e-4)
            workload_weight[:actual_len] = .5 * widths / widths.sum()
        workload_weight[self.candidate_count:] = .5 / 16.
        if self.architecture_profile in MUSIC_PROFILES:
            # Candidate windows are sampled at authored events; uniform-only
            # curve supervision avoids overweighting the busy sections.
            workload_weight[:self.candidate_count] = 0.
            workload_weight[self.candidate_count:] = 1. / 16.
        probe_weight = workload_weight.copy()
        views = self.event_views[sample.song_index]
        workload_weight *= table['workload_query'].weight_at(workload_seconds)
        ledger = prefix_ledger(table['point_seconds'], table['head_prefix'], seconds, committed_count=ev)
        exit_weight = table['transition_weights'][ev]
        stamp=time.perf_counter()
        if self.action_state_policy=='candidate_prefix_v2':
            from .context_features import composer_action_state
            if candidates is None:
                base_frac=Fraction(0) if sample.previous is None else sample.previous
                candidates=[base_frac+off for off in offsets]
                candidates.extend([candidates[-1]]*(self.candidate_count-len(candidates)))
            action_state=composer_action_state(table['state_index'],ev,candidates,self.action_state_policy)
        else:
            action_state=self._action_state(sample.song_index,ev,table)
        self._data_timing['action_state_sec']+=time.perf_counter()-stamp
        result = {
            'candidate_features': torch.from_numpy(features),
            'candidate_target': torch.tensor(target, dtype=torch.long),
            'candidate_mask': torch.from_numpy(candidate_mask),
            'action_target': table['atom_ids'][ev],
            'history': table['atom_ids'][start:ev],
            'history_features': table['history_features'][start:ev],
            'action_state': action_state,
            'condition': table['condition'],
            'future_target': table['future_targets'][ev],
            'future_weight': table['future_masks'][ev],
            'eos_target': torch.tensor(sample.eos_target, dtype=torch.float32),
            'eos_weight': torch.tensor(sample.eos_weight, dtype=torch.float32),
            'pointer_weight': torch.tensor(sample.pointer_weight, dtype=torch.float32),
            'song_index': torch.tensor(sample.song_index),
            'event_index': torch.tensor(ev),
            'workload_audio': torch.from_numpy(workload_audio),
            'workload_target': torch.from_numpy(workload_target),
            'workload_weight': torch.from_numpy(workload_weight),
            'workload_seconds': torch.from_numpy(workload_seconds),
            'ledger': torch.from_numpy(ledger),
            'sample_weight': torch.tensor(sample.weight, dtype=torch.float32),
            'transition_weight': torch.tensor(exit_weight, dtype=torch.float32),
            'head_timing': torch.from_numpy(head_timing_features(table['last_head_seconds'][ev],
                table['last_head_beats'][ev], seconds, beat_floats)),
        }
        if self.motif_memory_policy!='off':
            from .context_features import motif_tensors
            result.update(motif_tensors(table['motif_index'],sample.target,ev,
                                        self.history_events,self.motif_max_events))
        if self.architecture_profile in MUSIC_PROFILES:
            # Trailing labels have their own support; a future exclusion must
            # not mask an otherwise valid past-only window (or vice versa).
            trailing_weight = np.column_stack([
                probe_weight * views.window_weight(workload_seconds-width, workload_seconds)
                for width in (1.,1.,4.,4.)])
            result.update({
                'interval_audio': torch.from_numpy(context.interval(float(clock.at(sample.previous)) if sample.previous is not None else 0., seconds)),
                'trailing_audio': torch.from_numpy(context.trailing(workload_seconds)),
                'trailing_target': torch.from_numpy(trailing_workload_targets(table['known_point_seconds'],
                    table['known_head_prefix'], workload_seconds)),
                'trailing_weight': torch.from_numpy(trailing_weight),
            })
        if self.architecture_profile==GROUNDED_PROFILE:
            result.update(self._grounded_features(sample.song_index,workload_seconds,seconds,table=table))
            # Candidates are a biased proposal set. Equal total weight for
            # candidates and uniform probes; the latter cover authored rests.
            place_weight=result['placement_weight']
            place_weight[:self.candidate_count]*=torch.from_numpy((~candidate_mask).astype(np.float32))[:,None]/max(1,actual_len)
            place_weight[self.candidate_count:]/=16.
            result['planner_weight']=result['planner_weight']/max(1,len(result['planner_weight']))
        return result

    def _grounded_features(self,song_index,query_seconds,candidate_seconds,table=None):
        from .audio_lens import LensBank,PlacementTargets
        from .planning import PlanGrid
        table=self._song_table(song_index) if table is None else table
        song=self.base.songs[song_index][0]
        key=self._song_feature_key(song_index)
        with self._feature_lock:context=self._context_cache.get(key)
        if context is None:
            values,times=self._song_features(song_index)
            context=AcousticContext(values,times,table['audio_duration'],validated_bank=True)
            with self._feature_lock:
                self._context_cache[key]=context
                while len(self._context_cache)>self._context_cache_limit:self._context_cache.popitem(last=False)
        lock = self._song_table_locks.setdefault(song_index, threading.RLock())
        with lock:
            if not hasattr(context,'lens_bank'):
                values,times=self._song_features(song_index)
                context.lens_bank=LensBank(values,times,table['audio_duration'])
            if 'planner_data' not in table:
                table['planner_data']=PlanGrid(table['clock'],self._song_audio_ends[song_index],context,self.event_views[song_index])
                table['placement_data']=PlacementTargets(table['points'],song.tempo,song.first,
                    self._song_audio_ends[song_index],self.event_views[song_index])
        plan=table['planner_data']
        patches,valid=context.lens_bank.query(query_seconds)
        targets,weights=table['placement_data'].query(query_seconds)
        indices,mix=plan.locate(candidate_seconds)
        return {'audio_lens':torch.from_numpy(patches),'audio_lens_mask':torch.from_numpy(valid),
            'placement_target':torch.from_numpy(targets),'placement_weight':torch.from_numpy(weights),
            'planner_audio':torch.from_numpy(plan.audio),'planner_target':torch.from_numpy(plan.targets),
            'planner_weight':torch.from_numpy(plan.weights.copy()),
            'planner_mask':torch.zeros(len(plan.centers),dtype=torch.bool),
            'planner_indices':torch.from_numpy(indices),'planner_mix':torch.from_numpy(mix)}


def collate_composer(items):
    from torch.nn.utils.rnn import pad_sequence
    result={}
    result['candidate_features']=torch.stack([item['candidate_features'] for item in items])
    result['candidate_target']=torch.stack([item['candidate_target'] for item in items])
    result['action_target']=torch.stack([item['action_target'] for item in items])
    if 'audio_lens' in items[0]:
        # Grounded v9 ignores slots after the first STOP. Keep every authored
        # atom and its terminating STOP, without projecting padding through
        # both 18k-class heads. Checkpoint slot capacity stays full.
        targets=result['action_target']
        required=min(targets.size(1),int(targets.ne(0).sum(-1).max())+1)
        result['action_target']=targets[:,:required].contiguous()
    result['action_state']=torch.stack([item['action_state'] for item in items])
    if 'motif_ids' in items[0]:
        width=max(item['motif_ids'].size(-1) for item in items)
        result['motif_ids']=torch.stack([torch.nn.functional.pad(item['motif_ids'],
            (0,width-item['motif_ids'].size(-1))) for item in items])
        for key in ('motif_relative_beats','motif_mask'):
            result[key]=torch.stack([item[key] for item in items])
    result['future_target']=torch.stack([item['future_target'] for item in items])
    result['condition']=torch.stack([item['condition'] for item in items])
    result['history']=pad_sequence([item['history'] for item in items],batch_first=True,padding_value=0)
    if 'audio_lens' in items[0] and result['history'].ndim==3:
        # History sums atom embeddings after masking STOP. Empty trailing atom
        # columns add exactly zero, but expanding them through every factor
        # embedding is expensive (33 columns for ordinary one/two-note events).
        # Locate the last occupied column rather than counting, retaining any
        # internal holes and full-screen events. Empty histories keep one slot.
        occupied=result['history'].ne(0).any(dim=(0,1)).nonzero()
        width=int(occupied[-1,0])+1 if occupied.numel() else 1
        result['history']=result['history'][...,:width].contiguous()
    result['history_features']=pad_sequence([item['history_features'] for item in items],batch_first=True,padding_value=0.)
    result['history_padding_mask']=torch.arange(result['history'].size(1))[None]>=torch.tensor(
        [len(item['history']) for item in items])[:,None]
    result['candidate_padding_mask']=torch.stack([item['candidate_mask'] for item in items])
    result['song_index']=torch.stack([item['song_index'] for item in items])
    result['event_index']=torch.stack([item['event_index'] for item in items])
    result['eos_target']=torch.stack([item['eos_target'] for item in items])
    for key in ('workload_audio', 'workload_target', 'workload_weight', 'workload_seconds',
                'ledger', 'sample_weight', 'transition_weight', 'interval_audio',
                'trailing_audio', 'trailing_target', 'trailing_weight', 'head_timing',
                'eos_weight', 'future_weight', 'pointer_weight','action_weight',
                'audio_lens','audio_lens_mask','placement_target','placement_weight','planner_indices','planner_mix'):
        if key in items[0]: result[key] = torch.stack([item[key] for item in items])
    if 'planner_audio' in items[0]:
        for key in ('planner_audio','planner_target','planner_weight'):
            result[key]=pad_sequence([item[key] for item in items],batch_first=True,padding_value=0.)
        result['planner_mask']=pad_sequence([item['planner_mask'] for item in items],batch_first=True,padding_value=True)
    return result
