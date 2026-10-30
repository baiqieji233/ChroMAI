"""Event-level composer dataset built on the existing FormDataset/audio cache."""
from __future__ import annotations

from dataclasses import dataclass,replace
from fractions import Fraction
import json
import hashlib
import math
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
from maichart.form.recurrence import RECURRENCE_DESCRIPTOR_POLICY,LEGACY_RECURRENCE_DESCRIPTOR_POLICY
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
                     head_timing_features, history_event_features,head_pads_for_policy,head_pad_indices_for_policy)
from .event_views import EVENT_VIEW_POLICY, EventViews
from .model import MUSIC_PROFILES,GROUNDED_PROFILE
from .runtime_memory import cache_capacity, host_memory
from .audio_bank import bank_path,load_bank,publish_bank
from .acoustics import AcousticContext
from .planning import PLAN_POLICY
from .workload import ExactWorkloadQuery, prefix_ledger, prefix_ledger_batch, trailing_workload_targets, event_statistics
from .candidates import CANDIDATE_POLICY, MUSIC_CANDIDATE_POLICY_V6, is_playable_candidate_transition, offset_universe
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
                 action_state_policy='committed_legacy_v1', motif_memory_policy='off', motif_max_events=16,
                 head_timing_policy='merged_pads_v1', structure_policy='off', structure_backend='native',
                 atom_catalog=None, defer_audio_queries=False, quality_inputs_enabled=False, quality_inputs_policy=None,
                 action_order_policy='off', fixed_future_audio=False,
                 timing_distribution_policy='hierarchical_v1', density_feedback_policy='off'):
        started = time.monotonic()
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.codec_atoms=[]; self.atom_to_id={}; self.samples=[]; self.songs=[]
        self.candidate_count=int(candidate_count);self.history_events=int(history_events)
        self.architecture_profile = architecture_profile
        self.timing_distribution_policy = timing_distribution_policy
        self.density_feedback_policy = density_feedback_policy
        self.defer_audio_queries = bool(defer_audio_queries)
        self.quality_inputs_policy = quality_inputs_policy
        self.quality_inputs_enabled = bool(quality_inputs_enabled or quality_inputs_policy == 'song_paired_v3')
        self.compact_audio_queries = False  # runtime execution only; no label/cache contract change
        if action_order_policy not in ('off', 'symmetric_v1'):
            raise ValueError('Unknown action order policy')
        self.action_order_policy=action_order_policy
        self.fixed_future_audio=bool(fixed_future_audio)
        self.action_state_policy=action_state_policy
        self.head_timing_policy=head_timing_policy
        self.structure_policy=structure_policy
        self.structure_backend=structure_backend
        self.fixed_catalog=atom_catalog is not None
        self.head_pads=head_pads_for_policy(head_timing_policy)
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
            rhythm_geometry_profile='none',audio_profile='rational_audio_queries_v1',
            recurrence_descriptor_policy=(RECURRENCE_DESCRIPTOR_POLICY if motif_memory_policy!='off'
                                          else LEGACY_RECURRENCE_DESCRIPTOR_POLICY))
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
        self.action_slots = self.action_slots if self.fixed_catalog else max(self.action_slots, self.max_required_action_slots)
        if self.action_slots != self.action_slots_requested:
            print('composer_prepare ' + json.dumps({'stage': 'action_slots_expanded',
                'requested': self.action_slots_requested, 'effective': self.action_slots}), flush=True)
        self.codec_atoms = list(atom_catalog) if self.fixed_catalog else sorted(atom_set)
        if len(set(self.codec_atoms))!=len(self.codec_atoms):
            raise ValueError('Composer fixed atom catalog contains duplicates')
        self.atom_to_id = {atom: index + 1 for index, atom in enumerate(self.codec_atoms)}
        self.candidate_total = 0
        self.candidate_misses = 0
        self.phase_filter_misses = 0
        self.playability_unreachable_targets = 0
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
                    song.tempo, level=level, duration_sec=max(float(song.audio.duration_sec),1e-6),
                    audio=song.audio, audio_first=song.first,
                    duration_semantics=MIACODE_DURATION_SEMANTICS)
            self._song_conditions[song_index]=condition
            points = views.context_points
            point_secs = views.seconds
            trusted_arr = views.trusted
            local_weights_arr = views.local_weights
            single_tempo_scale = (60.0 / clock.bpms[0]) if len(clock.beats) == 1 else None
            max_offset_float = float(max_offset)
            max_rest_limit = (64. if architecture_profile in MUSIC_PROFILES else 16.) - 1e-12
            previous = None
            prev_sec = 0.0
            supervised=[]
            for event_index, point in enumerate(points):
                beat = point.beat if type(point.beat) is Fraction else Fraction(point.beat)
                beat_sec = float(point_secs[event_index])
                if not trusted_arr[event_index]:
                    previous = beat
                    prev_sec = beat_sec
                    continue
                local_weight = float(weight) * float(local_weights_arr[event_index])
                if previous is not None and previous >= 0:
                    far_second = (prev_sec + max_offset_float * single_tempo_scale
                                  if single_tempo_scale is not None
                                  else float(clock.at(previous + max_offset)))
                    if far_second < audio_end_sec and far_second - prev_sec <= max_rest_limit:
                        valid_hi = full_hi
                        offsets, lookup, _ = _select_candidate_offsets(False, full_hi, self.candidate_count, previous, self.candidate_policy)
                    else:
                        _, _, valid_hi, offsets, lookup, _ = candidate_spec(
                            previous, tempo=song.tempo, first=song.first, audio_end_sec=audio_end_sec,
                            candidate_count=self.candidate_count, clock=clock, policy=self.candidate_policy)
                else:
                    _, _, valid_hi, offsets, lookup, _ = candidate_spec(
                        previous, tempo=song.tempo, first=song.first, audio_end_sec=audio_end_sec,
                        candidate_count=self.candidate_count, clock=clock, policy=self.candidate_policy)
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
                if not is_playable_candidate_transition(beat, previous, self.candidate_policy):
                    self.playability_unreachable_targets += 1
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
            **({'action_order_policy':self.action_order_policy} if self.action_order_policy!='off' else {}),
            **({'fixed_future_audio':'ten_exact_beat_probes_v1'} if self.fixed_future_audio else {}),
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
            if motif_memory_policy!='off':
                from .motif_memory import motif_relations_fingerprint
                fingerprint.update(motif_input_policy=RECURRENCE_DESCRIPTOR_POLICY,
                    motif_relations_sha256=motif_relations_fingerprint(
                        (i, getattr(records[i][0], 'relations', None)) for i in sorted(self.event_views)))
            self.fingerprint=json.dumps(fingerprint,sort_keys=True)
        if head_timing_policy!='merged_pads_v1':
            fingerprint=json.loads(self.fingerprint)
            fingerprint['head_timing_policy']=head_timing_policy
            self.fingerprint=json.dumps(fingerprint,sort_keys=True)
        if self.fixed_catalog:
            # Unseen actions must never expand a held-out model vocabulary.
            # Keep their authored history timing, but exclude unrepresentable
            # next-event labels and report the exact evaluation coverage.
            before=len(self.samples)
            def representable(sample):
                atoms=event_metadata(sample.note_text).atoms
                return len(atoms)<=self.action_slots and all(a in self.atom_to_id for a in atoms)
            self.samples=[s for s in self.samples if representable(s)]
            self.catalog_coverage={'eligible_events':before,'representable_events':len(self.samples),
                'excluded_unknown_or_oversize':before-len(self.samples),
                'unknown_atom_types':len(atom_set-set(self.codec_atoms))}
        if structure_policy!='off':
            from .structure import structure_schema_for_policy
            fingerprint=json.loads(self.fingerprint)
            fingerprint.update(structure_policy=structure_policy,structure_backend=structure_backend,
                               structure_schema=structure_schema_for_policy(structure_policy))
            from .structure_backend import backend_identity
            fingerprint['structure_backend_identity']=backend_identity(structure_backend)
            self.fingerprint=json.dumps(fingerprint,sort_keys=True)
        if self.quality_inputs_policy == 'song_paired_v3':
            fingerprint=json.loads(self.fingerprint)
            fingerprint['quality_inputs_policy']=self.quality_inputs_policy
            self.fingerprint=json.dumps(fingerprint,sort_keys=True)
        if self.timing_distribution_policy=='survival_integrated_v2':
            fingerprint=json.loads(self.fingerprint)
            fingerprint.update(timing_distribution_policy=self.timing_distribution_policy,
                               survival_grid_seconds=.05)
            self.fingerprint=json.dumps(fingerprint,sort_keys=True)
        if self.density_feedback_policy != 'off':
            fingerprint=json.loads(self.fingerprint)
            fingerprint['density_feedback_policy']=self.density_feedback_policy
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

    def ordered_action_target(self, target, sample_index):
        """Augment only a private target; committed history/cache stay canonical."""
        target = target.clone()
        if getattr(self, 'action_order_policy', 'off') == 'symmetric_v1':
            # Integer mixing is process-independent and independent of loader order.
            mixed = (self.seed * 0x9E3779B1 + self.epoch * 0x85EBCA77
                     + int(sample_index) * 0xC2B2AE3D) & 0xFFFFFFFF
            mixed ^= mixed >> 16
            if mixed & 1:
                occupied = target.ne(0).nonzero().flatten()
                target[occupied] = target[occupied].flip(0)
        return target

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
        table = self._song_tables.get(song_index)
        if table is not None:
            return table
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
        last_head_seconds = np.full((n, len(self.head_pads)), np.nan, dtype=np.float32)
        last_head_beats = np.full((n, len(self.head_pads)), np.nan, dtype=np.float32)
        seen_seconds = np.full(len(self.head_pads), np.nan)
        seen_beats = np.full(len(self.head_pads), np.nan)
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
            for slot, atom in enumerate(atoms[:self.action_slots]):
                atom_ids[i, slot] = self.atom_to_id.get(atom,0)
            cur_sec = point_seconds[i]
            hist_feats[i] = history_event_features(text, p.beat-prior_beat, cur_sec-prior_second,
                parsed=self.architecture_profile in MUSIC_PROFILES)
            # Snapshot BEFORE this target: current/future heads never enter
            # candidate-relative recency, including on the first event.
            last_head_seconds[i], last_head_beats[i] = seen_seconds, seen_beats
            for pad in head_pad_indices_for_policy(meta,self.head_timing_policy):
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
        all_gaps = [max(0.0, point_seconds[i] - (point_seconds[i - 1] if i else 0.0)) for i in range(n)]
        for fi in range(n):
            end_w = min(n, fi + 4)
            w_len = end_w - fi
            if w_len > 0:
                gaps = all_gaps[fi:end_w]
                head_rate = float(sum(counts[fi:end_w]) / max(sum(gaps), 1e-3))
                future_targets[fi] = (
                    min(1., float(math.log1p(gaps[0]) / log1p_8)),
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
        from .structure import _audio_duration_sec, _identity
        if vars(song).get('_composer_structure_identity_base') is None and hasattr(song.audio, 'store'):
            try:
                _identity(song, self.structure_backend if self.structure_policy != 'off' else 'native',
                          self.structure_policy if self.structure_policy in ('metrical_context_v1', 'metrical_context_v2') else 'metrical_context_v1')
            except Exception:
                pass
        duration = _audio_duration_sec(song)
        workload_times = np.arange(.25, duration, .5)
        if not len(workload_times): workload_times = np.asarray([duration * .5])
        workload_query = ExactWorkloadQuery(points, song.tempo, song.first, views.end_sec,
            unknown_intervals=views.unknown_intervals, regions=views.regions)
        workload_targets = workload_query.query(workload_times)
        transition_weights = np.ones(n, dtype=np.float32)
        in_music_profile = self.architecture_profile in MUSIC_PROFILES
        step_gaps = [point_seconds[j + 1] - point_seconds[j] for j in range(max(0, n - 1))]
        trusted_arr = views.trusted
        for i in range(8, n):
            # Weight real authored exits only, once even if several periods
            # match. This does not declare longer repetitions invalid.
            lo_k = max(0, i - 16)
            if not (bool(trusted_arr[lo_k:i+1].all()) and views._window_positive(float(point_seconds[lo_k]), float(point_seconds[i]))):
                continue
            gaps = step_gaps[i-8:i-1]
            current_gap = step_gaps[i-1]
            rhythm_exit = False
            if max(gaps) - min(gaps) < 0.02:
                median = sorted(gaps)[3]
                rhythm_exit = (median > 0 and all(abs(g - median) < .01 for g in gaps)
                               and abs(current_gap - median) > max(.02, median * .5))
            if rhythm_exit or (in_music_profile and any(
                    i >= 4*p and sigs[i] != sigs[i-p] and all(sigs[i-j] == sigs[i-j-p] for j in range(1, 3*p+1))
                    for p in (1, 2, 3, 4))):
                transition_weights[i] = 2.
        table = {
            'clock': clock,
            'audio_duration': duration,
            'points': points,
            'state_index': _state if len(points)==len(_points) else SustainStateIndex(points, song.tempo, song.first),
            'atom_ids': torch.from_numpy(atom_ids),
            'atom_widths': np.max(np.where(atom_ids != 0, np.arange(atom_ids.shape[1]) + 1, 0), axis=1),
            'history_features': torch.from_numpy(hist_feats),
            'future_targets': torch.from_numpy(future_targets),
            'future_masks': torch.from_numpy(future_masks),
            'last_head_seconds': last_head_seconds,
            'last_head_beats': last_head_beats,
            'condition': condition.to_vector().detach().cpu().to(torch.float32),
            'action_states': action_states,
            'action_ready': action_ready,
            'point_seconds': np.asarray(point_seconds, dtype=np.float64),
            'point_beats': np.asarray([float(p.beat) for p in points], dtype=np.float64),
            'head_counts': np.asarray(counts, dtype=np.float32),
            'head_prefix': np.r_[0., np.cumsum(counts, dtype=np.float64)],
            'known_point_seconds': known_point_seconds,
            'known_head_prefix': known_head_prefix,
            'transition_weights': transition_weights,
            'workload_times': workload_times,
            'workload_targets': workload_targets,
            'workload_query': workload_query,
        }
        if (self.action_state_policy!='committed_legacy_v1' or getattr(self, 'quality_inputs_enabled', False)) and any('source_beat' not in m for m in table['state_index'].markers):
            table['state_index']=SustainStateIndex(points,song.tempo,song.first)
        if self.motif_memory_policy!='off':
            from .motif_memory import ComposerMotifIndex
            table['motif_index']=ComposerMotifIndex(getattr(song,'relations',None),
                [p.beat for p in points],atom_ids)
        self._song_tables[song_index] = table
        return table

    def _action_state(self, song_index: int, event_index: int, table: dict) -> torch.Tensor:
        if table['action_ready'][event_index]:
            return table['action_states'][event_index]
        with self._song_table_locks.setdefault(song_index, threading.RLock()):
            return self._action_state_unlocked(song_index,event_index,table)

    def _action_state_unlocked(self, song_index: int, event_index: int, table: dict) -> torch.Tensor:
        if not table['action_ready'][event_index]:
            if self.action_state_policy in {'committed_causal_v2','selected_prefix_v3'}:
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
        if self._bank_paths.get(song_index) is None:
            path=None
            audio=self.base.songs[song_index][0].audio
            if self.cache_dir and hasattr(audio,'store') and hasattr(audio,'path'):
                source_cache=audio.store.disk_cache_path(audio.path)
                if source_cache is not None and source_cache.is_file():
                    path=bank_path(self.cache_dir,key,source_cache)
            if path is not None:self._bank_paths[song_index]=path
            return path
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
                if path is None:
                    path=self._audio_bank_path(song_index,key)
                if path is not None:
                    try:
                        arrays=publish_bank(path,arrays)
                        audio=self.base.songs[song_index][0].audio
                        if hasattr(audio,'store') and hasattr(audio,'path') and len(key)>=2:
                            sc=audio.store.disk_cache_path(audio.path)
                            if sc is not None and sc.is_file():
                                alias=bank_path(self.cache_dir,(sc.name,*key[1:]),sc)
                                if alias!=path and not alias.exists():
                                    try:os.link(path,alias)
                                    except OSError:pass
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

    def _audio_context(self, song_index, table):
        stamp = time.perf_counter()
        values, times = self._song_features(song_index)
        self._data_timing['audio_lookup_sec'] += time.perf_counter() - stamp
        context_key = self._song_feature_key(song_index)
        with self._feature_lock:
            context = self._context_cache.get(context_key)
            if context is not None:
                self._context_cache.move_to_end(context_key)
        if context is None:
            raw_times = np.ascontiguousarray(np.asarray(times).view(np.ndarray), dtype=np.float64)
            context = AcousticContext(values, raw_times, table['audio_duration'], validated_bank=True)
            context._raw_times_contig = raw_times
            if self.architecture_profile == GROUNDED_PROFILE:
                from .audio_lens import LensBank
                context.lens_bank = LensBank(values, raw_times, table['audio_duration'], validated_bank=True)
            with self._feature_lock:
                existing = self._context_cache.get(context_key)
                if existing is None:
                    self._context_cache[context_key] = context
                    while len(self._context_cache) > self._context_cache_limit:
                        self._context_cache.popitem(last=False)
                else:
                    context = existing
        interp_times = getattr(context, '_raw_times_contig', times)
        if interp_times.shape != times.shape:
            interp_times = times
        return values, times, context, interp_times

    def _sample_query(self, index, table):
        """Exact clocks, shared by scalar and batched sample assembly."""
        sample = self.samples[index]
        offsets, _, valid_floats = _select_candidate_offsets(
            sample.previous is None, sample.valid_candidates, self.candidate_count,
            sample.previous, self.candidate_policy)
        actual_len = len(offsets)
        if actual_len <= 0:
            raise ValueError('Composer sample has no candidate beats')
        mask = np.zeros(self.candidate_count, dtype=bool)
        base = 0. if sample.previous is None else float(sample.previous)
        if actual_len < self.candidate_count:
            mask[actual_len:] = True
            beats = np.empty(self.candidate_count, dtype=np.float64)
            beats[:actual_len] = base + valid_floats
            beats[actual_len:] = beats[actual_len - 1]
        else:
            beats = base + valid_floats
        clock = table['clock']
        seconds = (clock.seconds[0] + beats * (60. / clock.bpms[0])
                   if len(clock.beats) == 1 else clock.at_floats(beats))
        mask |= (seconds < 0.) | (seconds >= self._song_audio_ends[sample.song_index])
        ev = sample.event_index
        previous_second = (0. if sample.previous is None else
            float(table['point_seconds'][ev - 1])
            if ev > 0 and sample.previous == table['points'][ev - 1].beat
            else float(clock.at(sample.previous)))
        workload_seconds = np.empty(self.candidate_count + 16, dtype=np.float64)
        workload_seconds[:self.candidate_count] = seconds
        workload_seconds[self.candidate_count:] = (
            (np.arange(16) + ((index % 31) + .5) / 31.) * table['audio_duration'] / 16.)
        return dict(offsets=offsets, actual_len=actual_len, base_float=base,
                    beat_floats=beats, seconds=seconds, candidate_mask=mask,
                    prev_sec=previous_second, workload_seconds=workload_seconds)

    def __getitems__(self, indices):
        """Query immutable same-chart banks once per batch, preserving row order.

        Only exact, row-independent queries are merged. Causal symbolic inputs
        and per-index probe phases are still assembled separately. Temporary
        arrays live for this call only; nothing grows with the epoch length.
        """
        indices = list(indices)
        items = [None] * len(indices)
        groups = {}
        for position, index in enumerate(indices):
            groups.setdefault(self.samples[index].song_index, []).append((position, index))
        for song_index, rows in groups.items():
            stamp = time.perf_counter()
            table = self._song_table(song_index)
            self._data_timing['song_table_sec'] += time.perf_counter() - stamp
            runtime = self._audio_context(song_index, table)
            values, _, context, interp_times = runtime
            queries = [self._sample_query(index, table) for _, index in rows]
            seconds = np.stack([q['seconds'] for q in queries])
            workload_seconds = np.stack([q['workload_seconds'] for q in queries])
            count, width = workload_seconds.shape
            audio = interp_candidate_audio(values, interp_times, seconds.reshape(-1)).reshape(count, self.candidate_count, -1)
            target = table['workload_query'].query(workload_seconds.reshape(-1)).reshape(count, width, -1)
            confidence = table['workload_query'].weight_at(workload_seconds.reshape(-1)).reshape(count, width)
            ledger = prefix_ledger_batch(table['point_seconds'], table['head_prefix'], seconds,
                [self.samples[index].event_index for _, index in rows])
            trailing = (trailing_workload_targets(table['known_point_seconds'], table['known_head_prefix'],
                workload_seconds.reshape(-1)).reshape(count, width, -1)
                if self.architecture_profile in MUSIC_PROFILES else None)
            workload_audio = (None if self.defer_audio_queries else
                context.at(workload_seconds.reshape(-1)).reshape(count, width, -1))
            grounded = None
            if self.architecture_profile == GROUNDED_PROFILE:
                grounded = self._grounded_features(song_index, workload_seconds.reshape(-1), seconds.reshape(-1),
                    table=table, context=context, scale_planner_weight=True)
            structures = None
            if self.structure_policy != 'off':
                from .structure import structure_features_and_transition_batch
                structures = structure_features_and_transition_batch(
                    self.base.songs[song_index][0], np.stack([q['beat_floats'] for q in queries]), seconds,
                    [q['base_float'] if self.samples[index].previous is not None else None for (_, index), q in zip(rows, queries)],
                    [q['prev_sec'] if self.samples[index].previous is not None else None for (_, index), q in zip(rows, queries)],
                    policy=self.structure_policy, cache_dir=self.cache_dir, backend=self.structure_backend)
            future_audio = future_inverse = None
            if self.fixed_future_audio:
                from .quality_inputs import QUALITY_FUTURE_OFFSETS, exact_history_audio
                # Phrase rows repeat exact candidate clocks. Query immutable
                # future audio once per distinct beat, then restore every row.
                beats = np.stack([q['beat_floats'] for q in queries])
                unique_beats, inverse = np.unique(beats, return_inverse=True)
                future_inverse = inverse.reshape(beats.shape)
                offsets = np.asarray(QUALITY_FUTURE_OFFSETS, dtype=np.float64)
                probe_seconds = table['clock'].at_floats(
                    (unique_beats[:, None] + offsets[None, :]).reshape(-1)).reshape(len(unique_beats), len(offsets))
                probe_valid = np.isfinite(probe_seconds) & (probe_seconds >= 0.) & (probe_seconds <= float(table['audio_duration']))
                probe_audio = np.zeros((*probe_seconds.shape, 104), dtype=np.float32)
                if probe_valid.any():
                    unique_seconds, probe_inverse = np.unique(probe_seconds[probe_valid], return_inverse=True)
                    probe_audio[probe_valid] = exact_history_audio(context, unique_seconds)[probe_inverse]
                future_audio = {
                    'quality_future_audio': torch.from_numpy(probe_audio),
                    'quality_future_mask': torch.from_numpy(probe_valid),
                    'quality_future_offsets': torch.tensor(QUALITY_FUTURE_OFFSETS, dtype=torch.float32)}
            for row, ((position, index), query) in enumerate(zip(rows, queries)):
                shared = dict(audio=audio[row], workload_target=target[row], workload_confidence=confidence[row], ledger=ledger[row],
                              workload_audio=None if workload_audio is None else workload_audio[row])
                if future_audio is not None:
                    take = future_inverse[row]
                    shared['future_audio'] = {
                        key: value if key == 'quality_future_offsets' else value[take]
                        for key, value in future_audio.items()}
                if trailing is not None:
                    shared['trailing_target'] = trailing[row]
                if structures is not None:
                    shared['structure'] = structures[0][row], structures[1][row]
                if grounded is not None:
                    # All planner-grid tensors are immutable shared song data.
                    # Placement/lens rows retain the original per-sample clocks.
                    sample_grounded = dict(grounded)
                    for name in ('placement_target', 'placement_weight', 'placement_proximity',
                                 'audio_lens', 'audio_lens_mask'):
                        if name in grounded:
                            sample_grounded[name] = grounded[name][row * width:(row + 1) * width]
                    weights = sample_grounded['placement_weight'].clone()
                    weights[:self.candidate_count] *= 1. / max(1, query['actual_len'])
                    weights[:self.candidate_count][query['candidate_mask']] = 0.
                    weights[self.candidate_count:] *= 1. / 16.
                    sample_grounded['placement_weight'] = weights
                    for name in ('planner_indices', 'planner_mix'):
                        sample_grounded[name] = grounded[name][row * self.candidate_count:(row + 1) * self.candidate_count]
                    shared['grounded'] = sample_grounded
                items[position] = self._item(index, query=query, runtime=runtime, table=table, shared=shared)
        return items

    def __getitem__(self, index):
        return self._item(index)

    def _item(self, index, *, query=None, runtime=None, table=None, shared=None):
        sample = self.samples[index]
        song = self.base.songs[sample.song_index][0]
        if table is None:
            stamp=time.perf_counter()
            table = self._song_table(sample.song_index)
            self._data_timing['song_table_sec']+=time.perf_counter()-stamp
        clock = table['clock']
        values, times, context, interp_times = (self._audio_context(sample.song_index, table)
                                               if runtime is None else runtime)
        query = self._sample_query(index, table) if query is None else query
        offsets, actual_len = query['offsets'], query['actual_len']
        base_float, beat_floats = query['base_float'], query['beat_floats']
        seconds, candidate_mask = query['seconds'], query['candidate_mask']
        prev_sec = query['prev_sec']
        candidates = None
        ev = sample.event_index
        feature = candidate_features(
            candidates,
            sample.previous,
            song.tempo,
            song.first,
            clock=clock,
            candidate_seconds=seconds,
            beat_floats=beat_floats,
            previous_second=max(0.0, prev_sec),
            rich_time=True,
        )
        target = sample.target_index
        if target < 0 or target >= self.candidate_count:
            raise ValueError('Composer candidate generator missed an authored target')
        if candidate_mask[target]:
            raise ValueError('Composer supervised target lies outside the valid audio range')
        start = max(0, ev - self.history_events)
        from .history_clock import history_beat_ages
        past_beats = (table['point_beats'][start:ev] if 'point_beats' in table
                      else [p.beat for p in table['points'][start:ev]])
        audio = (interp_candidate_audio(values, interp_times, seconds)
                 if shared is None else shared['audio'])
        features = np.concatenate((audio, feature), axis=1)
        duration = table['audio_duration']
        # Uniform whole-song probes include silence and are independent of
        # event density; deterministic phases vary the centers across samples.
        workload_seconds = query['workload_seconds']
        workload_audio = ((None if self.defer_audio_queries else context.at(workload_seconds))
                          if shared is None else shared['workload_audio'])
        workload_target = (table['workload_query'].query(workload_seconds)
                           if shared is None else shared['workload_target'])
        workload_weight = np.zeros(len(workload_seconds), dtype=np.float32)
        if self.architecture_profile in MUSIC_PROFILES:
            # Candidate windows are sampled at authored events; uniform-only
            # curve supervision avoids overweighting the busy sections.
            workload_weight[self.candidate_count:] = 1. / 16.
        else:
            if actual_len:
                edges = np.empty(actual_len + 1, dtype=np.float64)
                edges[0] = seconds[0]
                edges[1:actual_len] = (seconds[:actual_len - 1] + seconds[1:actual_len]) * .5
                edges[actual_len] = seconds[actual_len - 1]
                widths = np.maximum(np.diff(edges), 1e-4)
                workload_weight[:actual_len] = .5 * widths / widths.sum()
            workload_weight[self.candidate_count:] = .5 / 16.
        probe_weight = workload_weight.copy()
        views = self.event_views[sample.song_index]
        workload_weight *= (table['workload_query'].weight_at(workload_seconds)
                            if shared is None else shared['workload_confidence'])
        event_count, event_exposure, event_rate_weight = event_rate_targets(
            table['point_seconds'], workload_seconds, duration, views,
            probe_weight, self.candidate_count)
        ledger = (prefix_ledger(table['point_seconds'], table['head_prefix'], seconds, committed_count=ev)
                  if shared is None else shared['ledger'])
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
            'action_target': self.ordered_action_target(table['atom_ids'][ev], index),
            'history': table['atom_ids'][start:ev],
            'history_features': table['history_features'][start:ev],
            'history_beat_ages': torch.from_numpy(history_beat_ages(past_beats)),
            'history_second_ages': torch.from_numpy(np.asarray(
                table['point_seconds'][ev-1] - table['point_seconds'][start:ev]
                if ev else [], dtype=np.float32)),
            'action_state': action_state,
            'condition': table['condition'],
            'future_target': table['future_targets'][ev],
            'future_weight': table['future_masks'][ev],
            'eos_target': torch.tensor(sample.eos_target, dtype=torch.float32),
            'eos_weight': torch.tensor(sample.eos_weight, dtype=torch.float32),
            'pointer_weight': torch.tensor(sample.pointer_weight, dtype=torch.float32),
            'song_index': torch.tensor(sample.song_index),
            'event_index': torch.tensor(ev),
            **({'workload_audio': torch.from_numpy(workload_audio)} if workload_audio is not None else {}),
            'workload_target': torch.from_numpy(workload_target),
            'workload_weight': torch.from_numpy(workload_weight),
            'workload_seconds': torch.from_numpy(workload_seconds),
            'event_count_target': torch.from_numpy(event_count),
            'event_exposure': torch.from_numpy(event_exposure),
            'event_rate_weight': torch.from_numpy(event_rate_weight),
            'survival_start': torch.tensor(0., dtype=torch.float64),
            'ledger': torch.from_numpy(ledger),
            'sample_weight': torch.tensor(sample.weight, dtype=torch.float32),
            'transition_weight': torch.tensor(exit_weight, dtype=torch.float32),
            'head_timing': torch.from_numpy(head_timing_features(table['last_head_seconds'][ev],
                table['last_head_beats'][ev], seconds, beat_floats)),
        }
        if self.architecture_profile == GROUNDED_PROFILE:
            # Last occupied column (including any internal holes), computed
            # before collating rather than scanning a padded [B,H,33] tensor.
            result['_history_atom_width'] = int(table['atom_widths'][start:ev].max(initial=1))
        if getattr(self, 'quality_inputs_enabled', False):
            from .quality_inputs import quality_context_inputs, quality_targets, phrase_audio_bank
            song_paired = getattr(self, 'quality_inputs_policy', None) == 'song_paired_v3'
            if song_paired:
                if 'planner_data' not in table:
                    self._grounded_features(sample.song_index, workload_seconds, seconds, table=table, context=context)
                with self._song_table_locks.setdefault(sample.song_index, threading.RLock()):
                    if 'quality_history_audio' not in table:
                        table['quality_history_audio'] = context.at(table['point_seconds'])
                        table['quality_history_plan_mapping'] = table['planner_data'].locate(table['point_seconds'])
            quality_start = start if song_paired else 0
            result.update(quality_context_inputs(points=(), atom_ids=table['atom_ids'][quality_start:ev],
                include_future=getattr(self,'fixed_future_audio',False) and not (shared is not None and 'future_audio' in shared),
                candidate_beats=beat_floats, candidate_seconds=seconds, state_index=table['state_index'],
                audio_features=values, audio_times=interp_times, acoustic_context=context, clock=clock,
                audio_duration=duration, visible_before_beat=table['points'][ev].beat,
                point_beats=table['point_beats'][quality_start:ev], point_seconds=table['point_seconds'][quality_start:ev],
                policy=getattr(self, 'quality_inputs_policy', None), plan_grid=table.get('planner_data'),
                history_audio=table['quality_history_audio'][start:ev] if song_paired else None,
                history_plan_mapping=tuple(a[start:ev] for a in table['quality_history_plan_mapping']) if song_paired else None))
            if 'quality_target_data' not in table:
                with self._song_table_locks.setdefault(sample.song_index, threading.RLock()):
                    if 'quality_target_data' not in table:
                        table['quality_target_data'] = quality_targets(table['points'], clock,
                            phrase_audio_bank(context, clock, duration), views, policy=getattr(self, 'quality_inputs_policy', None))
            quality = table['quality_target_data']
            result.update(quality_phrase_target=quality['phrase_target'], quality_phrase_weight=quality['phrase_weight'],
                quality_gap_target=quality['gap_target'][ev], quality_gap_weight=quality['gap_weight'][ev])
        if shared is not None and 'future_audio' in shared:
            result.update(shared['future_audio'])
        if getattr(self,'fixed_future_audio',False) and 'quality_future_audio' not in result:
            from .quality_inputs import fixed_future_audio_inputs
            result.update(fixed_future_audio_inputs(context,clock,beat_floats,duration))
        if getattr(self, 'fixed_future_audio', False) and self.timing_distribution_policy!='survival_integrated_v2':
            result.update(empty_gap_inputs(context, views, prev_sec, table['point_seconds'][ev]))
        if self.structure_policy!='off':
            from .structure import structure_features_and_transition
            song=self.base.songs[sample.song_index][0]
            if shared is not None and 'structure' in shared:
                structure,transition=shared['structure']
            else:
                structure,transition=structure_features_and_transition(song,beat_floats,seconds,
                    base_float if sample.previous is not None else None,
                    prev_sec if sample.previous is not None else None,
                    policy=self.structure_policy,cache_dir=self.cache_dir,backend=self.structure_backend)
            result['structure_features']=torch.from_numpy(structure)
            result['structure_transition']=torch.from_numpy(transition)
        if self.motif_memory_policy!='off':
            from .context_features import motif_tensors
            result.update(motif_tensors(table['motif_index'],sample.target,ev,
                                        self.history_events,self.motif_max_events))
        if self.action_state_policy=='selected_prefix_v3':
            from .context_features import selected_action_state
            # Dataset tables represent authored prefixes only. Recovery builds
            # its own state and never consults this training-target cache.
            with self._song_table_locks.setdefault(sample.song_index,threading.RLock()):
                selected_cache=table.setdefault('selected_target_states',{})
                selected_key=(ev,sample.target)
                if selected_key not in selected_cache:
                    selected_cache[selected_key]=selected_action_state(table['state_index'],ev,sample.target)
                result['selected_action_state']=selected_cache[selected_key]
        if self.architecture_profile in MUSIC_PROFILES:
            # Trailing labels have their own support; a future exclusion must
            # not mask an otherwise valid past-only window (or vice versa).
            w1 = probe_weight * views.window_weight(workload_seconds - 1., workload_seconds)
            w4 = probe_weight * views.window_weight(workload_seconds - 4., workload_seconds)
            trailing_weight = np.column_stack((w1, w1, w4, w4))
            result.update({
                **({} if self.defer_audio_queries else {
                    'interval_audio': torch.from_numpy(context.interval(prev_sec, seconds)),
                    'trailing_audio': torch.from_numpy(context.trailing(workload_seconds))}),
                'trailing_target': torch.from_numpy(shared['trailing_target'] if shared is not None else
                    trailing_workload_targets(table['known_point_seconds'],table['known_head_prefix'],workload_seconds)),
                'trailing_weight': torch.from_numpy(trailing_weight),
            })
        if self.architecture_profile==GROUNDED_PROFILE:
            result.update(shared['grounded'] if shared is not None and 'grounded' in shared else
                self._grounded_features(sample.song_index,workload_seconds,seconds,table=table,context=context,
                    candidate_mask=candidate_mask,actual_len=actual_len,scale_planner_weight=True))
        if self.timing_distribution_policy == 'survival_integrated_v2':
            from .density_grid import survival_grid_inputs
            result.update({key: torch.from_numpy(value) for key, value in survival_grid_inputs(
                context, max(0., prev_sec), seconds, duration,
                plan_grid=table.get('planner_data'), views=views,
                point_seconds=table['point_seconds'],
                prefix_seconds=(table['point_seconds'][:ev]
                    if getattr(self, 'density_feedback_policy', 'off') == 'causal_count_debt_v1' else None)).items()})
        if self.defer_audio_queries:
            # Shared immutable CPU bank only; CUDA work belongs to the trainer.
            result['_audio_bank'] = context
            result['_audio_music'] = self.architecture_profile in MUSIC_PROFILES
            result['_audio_grounded'] = self.architecture_profile == GROUNDED_PROFILE
            result['_compact_audio_queries'] = getattr(self, 'compact_audio_queries', False) and result['_audio_grounded']
            result['audio_query_seconds'] = torch.from_numpy(workload_seconds)
            result['audio_previous_seconds'] = torch.tensor(prev_sec, dtype=torch.float64)
        return result

    def _grounded_features(self,song_index,query_seconds,candidate_seconds,table=None,context=None,defer_queries=None,
                           *,candidate_mask=None,actual_len=None,scale_planner_weight=False):
        from .audio_lens import LensBank,PlacementTargets
        from .planning import PlanGrid
        table=self._song_table(song_index) if table is None else table
        song=self.base.songs[song_index][0]
        key=self._song_feature_key(song_index)
        if context is None:
            with self._feature_lock:context=self._context_cache.get(key)
        if context is None:
            values,times=self._song_features(song_index)
            raw_times_contig=np.ascontiguousarray(np.asarray(times).view(np.ndarray),dtype=np.float64)
            context=AcousticContext(values,raw_times_contig,table['audio_duration'],validated_bank=True)
            context._raw_times_contig=raw_times_contig
            with self._feature_lock:
                self._context_cache[key]=context
                while len(self._context_cache)>self._context_cache_limit:self._context_cache.popitem(last=False)
        if not hasattr(context,'lens_bank') or 'planner_data' not in table:
            lock = self._song_table_locks.setdefault(song_index, threading.RLock())
            with lock:
                if not hasattr(context,'lens_bank'):
                    values,times=self._song_features(song_index)
                    raw_times_contig=getattr(context,'_raw_times_contig',None)
                    if raw_times_contig is None or raw_times_contig.shape != times.shape:
                        raw_times_contig=np.ascontiguousarray(np.asarray(times).view(np.ndarray),dtype=np.float64)
                        context._raw_times_contig=raw_times_contig
                    context.lens_bank=LensBank(values,raw_times_contig,table['audio_duration'],validated_bank=True)
                if 'planner_data' not in table:
                    plan_grid=PlanGrid(table['clock'],self._song_audio_ends[song_index],context,self.event_views[song_index],
                        share_audio=getattr(self,'quality_inputs_policy',None)=='song_paired_v3')
                    table['planner_data']=plan_grid
                    shared_plan=getattr(plan_grid,'_shared_raw_plan_bank',None)
                    table['planner_audio_tensor']=shared_plan['audio_tensor'] if shared_plan is not None else torch.from_numpy(plan_grid.audio)
                    table['planner_target_tensor']=torch.from_numpy(plan_grid.targets)
                    table['planner_weight_tensor']=torch.from_numpy(plan_grid.weights.copy())
                    table['planner_weight_scaled_tensor']=table['planner_weight_tensor']/max(1,len(plan_grid.weights))
                    table['planner_mask_tensor']=shared_plan['mask'] if shared_plan is not None else torch.zeros(len(plan_grid.centers),dtype=torch.bool)
                    table['placement_data']=PlacementTargets(table['points'],song.tempo,song.first,
                        self._song_audio_ends[song_index],self.event_views[song_index])
        plan=table['planner_data']
        lens={}
        if not (self.defer_audio_queries if defer_queries is None else defer_queries):
            patches,valid=context.lens_bank.query(query_seconds)
            lens={'audio_lens':torch.from_numpy(patches),'audio_lens_mask':torch.from_numpy(valid)}
        placement_data=table['placement_data']
        if hasattr(placement_data,'query_with_proximity'):
            targets,weights,proximity=placement_data.query_with_proximity(query_seconds)
        else:
            targets,weights=placement_data.query(query_seconds)
            proximity=(placement_data.proximity(query_seconds) if hasattr(placement_data,'proximity') else targets.copy())
        if candidate_mask is not None and actual_len is not None:
            weights[:self.candidate_count] *= (1. / max(1, actual_len))
            if candidate_mask.any():
                weights[:self.candidate_count][candidate_mask] = 0.
            weights[self.candidate_count:] *= (1. / 16.)
        indices,mix=plan.locate(candidate_seconds)
        planner_audio = table['planner_audio_tensor'] if 'planner_audio_tensor' in table else torch.from_numpy(plan.audio)
        planner_target = table['planner_target_tensor'] if 'planner_target_tensor' in table else torch.from_numpy(plan.targets)
        if scale_planner_weight:
            planner_weight = (table['planner_weight_scaled_tensor']
                              if 'planner_weight_scaled_tensor' in table
                              else torch.from_numpy(plan.weights.copy()) / max(1, len(plan.weights)))
        else:
            planner_weight = (table['planner_weight_tensor'].clone()
                              if 'planner_weight_tensor' in table
                              else torch.from_numpy(plan.weights.copy()))
        planner_mask = table['planner_mask_tensor'] if 'planner_mask_tensor' in table else torch.zeros(len(plan.centers), dtype=torch.bool)
        return {**lens,
            'placement_target':torch.from_numpy(targets),'placement_weight':torch.from_numpy(weights),'placement_proximity':torch.from_numpy(proximity),
            'planner_audio':planner_audio,
            'planner_target':planner_target,
            'planner_weight':planner_weight,
            'planner_mask':planner_mask,
            'planner_indices':torch.from_numpy(indices),'planner_mix':torch.from_numpy(mix)}


def empty_gap_inputs(context, views, previous_second, target_second):
    """Four zero-count rate-only probes inside the known pre-target gap.

    These authored-target-dependent tensors are auxiliary supervision, never
    candidate/pointer/action context. Confidence covers the entire gap so an
    unknown span cannot be mislabeled as silence by midpoint sampling.
    Sample weight is deliberately applied by the loss, once.
    """
    audio = np.zeros((4, 104), dtype=np.float32)
    exposure = np.zeros(4, dtype=np.float32)
    weight = np.zeros(4, dtype=np.float32)
    lower = max(0., float(previous_second))
    upper = float(target_second) - .025
    duration = upper - lower
    if np.isfinite(duration) and duration > 0.:
        confidence = float(views.window_weight(lower, upper))
        if confidence > 0.:
            exposure.fill(duration / 4.)
            centers = lower + (np.arange(4, dtype=np.float64) + .5) * duration / 4.
            audio[:] = context.at(centers)
            weight.fill(confidence)
    return {'empty_gap_audio': torch.from_numpy(audio),
            'empty_gap_exposure': torch.from_numpy(exposure),
            'empty_gap_weight': torch.from_numpy(weight)}


def event_rate_targets(point_seconds, centers, duration, views, probe_weight, candidate_count):
    """Count authored event onsets in time probes; unknown spans are unsupervised.

    Candidate rows never supervise the rate: their sampling follows authored
    events. Counts concern event onsets, not the number of heads in a chord.
    """
    centers = np.asarray(centers, dtype=np.float64)
    lower = np.clip(centers - .125, 0., duration)
    upper = np.clip(centers + .125, 0., duration)
    exposure = np.maximum(upper - lower, 0.).astype(np.float32)
    points = np.asarray(point_seconds, dtype=np.float64)
    counts = (np.searchsorted(points, upper, side='left') -
              np.searchsorted(points, lower, side='left')).astype(np.float32)
    # window_weight uses an inclusive right endpoint; this is a half-open
    # counting interval, including a legitimately known audio-tail probe.
    confidence = views.window_weight(lower, np.nextafter(upper, lower))
    weight = np.asarray(probe_weight, dtype=np.float32).copy()
    weight[:candidate_count] = 0.
    weight *= confidence * (exposure > 0.)
    return counts, exposure, weight


def collate_composer(items):
    from torch.nn.utils.rnn import pad_sequence
    result={}
    result['candidate_features']=torch.stack([item['candidate_features'] for item in items])
    result['candidate_target']=torch.stack([item['candidate_target'] for item in items])
    result['action_target']=torch.stack([item['action_target'] for item in items])
    if ('audio_lens' in items[0] or items[0].get('_audio_grounded',False)):
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
    known_width = (max(item['_history_atom_width'] for item in items)
                   if all('_history_atom_width' in item for item in items) else None)
    histories = [item['history'] if known_width is None else item['history'][..., :known_width] for item in items]
    result['history']=pad_sequence(histories,batch_first=True,padding_value=0)
    if known_width is None and ('audio_lens' in items[0] or items[0].get('_audio_grounded',False)) and result['history'].ndim==3:
        # History sums atom embeddings after masking STOP. Empty trailing atom
        # columns add exactly zero, but expanding them through every factor
        # embedding is expensive (33 columns for ordinary one/two-note events).
        # Locate the last occupied column rather than counting, retaining any
        # internal holes and full-screen events. Empty histories keep one slot.
        occupied=result['history'].ne(0).any(dim=(0,1)).nonzero()
        width=int(occupied[-1,0])+1 if occupied.numel() else 1
        result['history']=result['history'][...,:width].contiguous()
    result['history_features']=pad_sequence([item['history_features'] for item in items],batch_first=True,padding_value=0.)
    if 'history_beat_ages' in items[0]:
        result['history_beat_ages']=pad_sequence([item['history_beat_ages'] for item in items],batch_first=True,padding_value=0.)
    if 'history_second_ages' in items[0]:
        result['history_second_ages']=pad_sequence([item['history_second_ages'] for item in items],batch_first=True,padding_value=0.)
    result['history_padding_mask']=torch.arange(result['history'].size(1))[None]>=torch.tensor(
        [len(item['history']) for item in items])[:,None]
    result['candidate_padding_mask']=torch.stack([item['candidate_mask'] for item in items])
    result['song_index']=torch.stack([item['song_index'] for item in items])
    result['event_index']=torch.stack([item['event_index'] for item in items])
    result['eos_target']=torch.stack([item['eos_target'] for item in items])
    if 'survival_audio' in items[0]:
        for key in ('survival_audio', 'survival_plan_indices', 'survival_plan_mix',
                    'survival_count_target', 'survival_count_weight', 'survival_widths'):
            result[key] = pad_sequence([item[key] for item in items], batch_first=True, padding_value=0.)
        width = result['survival_audio'].size(1) + 1
        result['survival_edges'] = torch.stack([
            torch.cat((item['survival_edges'], item['survival_edges'][-1:].expand(width - len(item['survival_edges']))))
            for item in items])
        for key in ('survival_candidate_seconds', 'survival_start_seconds'):
            result[key] = torch.stack([item[key] for item in items])
        if 'survival_prefix_counts' in items[0]:
            result['survival_prefix_counts'] = pad_sequence(
                [item['survival_prefix_counts'] for item in items], batch_first=True, padding_value=0.)
    for key in ('workload_audio', 'workload_target', 'workload_weight', 'workload_seconds',
                'empty_gap_audio', 'empty_gap_exposure', 'empty_gap_weight',
                'event_count_target', 'event_exposure', 'event_rate_weight', 'survival_start',
                'ledger', 'sample_weight', 'transition_weight', 'interval_audio',
                'trailing_audio', 'trailing_target', 'trailing_weight', 'head_timing',
                'eos_weight', 'future_weight', 'pointer_weight','action_weight',
                'audio_lens','audio_lens_mask','placement_target','placement_weight','placement_proximity','planner_indices','planner_mix',
                'selected_action_state','structure_features','structure_transition',
                'audio_query_seconds','audio_previous_seconds',
                'quality_lifecycle','quality_anchor_audio','quality_anchor_ids','quality_anchor_ages',
                'quality_anchor_mask','quality_phrase_indices','quality_gap_target','quality_gap_weight',
                'quality_future_audio','quality_future_mask','quality_future_offsets'):
        if key in items[0]: result[key] = torch.stack([item[key] for item in items])
    if 'quality_history_audio' in items[0]:
        for key in ('quality_history_audio', 'quality_history_clock', 'quality_history_plan_indices', 'quality_history_plan_mix',
                    'quality_phrase_plan_indices', 'quality_phrase_plan_mix'):
            result[key] = pad_sequence([item[key] for item in items], batch_first=True, padding_value=0)
        # A single-row recovery batch still uses the original planner inputs.
        if len(items) == 1:
            result['quality_plan_clock'] = pad_sequence([item['quality_plan_clock'] for item in items], batch_first=True, padding_value=0)
    if 'quality_candidate_offsets' in items[0]:
        result['quality_candidate_offsets']=torch.stack([item['quality_candidate_offsets'] for item in items])
    if 'quality_phrase_audio' in items[0]:
        for key in ('quality_phrase_audio', 'quality_phrase_target', 'quality_phrase_weight'):
            result[key] = pad_sequence([item[key] for item in items], batch_first=True, padding_value=0.)
    if 'planner_audio' in items[0]:
        for key in ('planner_target','planner_weight'):
            result[key]=pad_sequence([item[key] for item in items],batch_first=True,padding_value=0.)
        if 'quality_history_audio' not in items[0] or len(items) == 1:
            result['planner_audio']=pad_sequence([item['planner_audio'] for item in items],batch_first=True,padding_value=0.)
            result['planner_mask']=pad_sequence([item['planner_mask'] for item in items],batch_first=True,padding_value=True)
    if '_audio_bank' in items[0]:
        banks=[]; rows=[]; identities={}
        for row,item in enumerate(items):
            identity=id(item['_audio_bank'])
            if identity not in identities:
                identities[identity]=len(banks)
                banks.append(item['_audio_bank']);rows.append([])
            rows[identities[identity]].append(row)
        result['_audio_banks']=tuple(banks)
        result['_audio_bank_rows']=tuple(tuple(group) for group in rows)
        result['_audio_bank_row_indices']=torch.tensor([r for group in rows for r in group],dtype=torch.long)
        result['_audio_bank_row_splits']=tuple(len(group) for group in rows)
        result['_audio_music']=items[0]['_audio_music']
        result['_audio_grounded']=items[0]['_audio_grounded']
        if items[0].get('_compact_audio_queries', False):
            # Group by actual bank identity and exact float64 query time, never
            # rounded beats or cross-song acoustic similarity. Do this on CPU
            # so CUDA never synchronizes for a variable-sized unique result.
            query = result['audio_query_seconds'].numpy()
            inverse = np.empty(query.shape, dtype=np.int64)
            unique_seconds = []
            splits = []
            offset = 0
            for group in rows:
                times, indices = np.unique(query[group].reshape(-1), return_inverse=True)
                inverse[group] = indices.reshape(len(group), -1) + offset
                unique_seconds.append(times)
                splits.append(len(times))
                offset += len(times)
            result['audio_unique_seconds'] = torch.from_numpy(np.concatenate(unique_seconds))
            result['audio_query_inverse'] = torch.from_numpy(inverse)
            result['_audio_unique_splits'] = tuple(splits)
    if 'planner_audio' in items[0]:
        unique=[];inverse=[];groups={}
        for row,item in enumerate(items):
            # Dataset rows of one song share the immutable plan.audio storage.
            # Independently allocated tables remain separate without scanning
            # whole-song audio on the collate hot path.
            audio=item['planner_audio']
            key=(int(item['song_index']),item['condition'].numpy().tobytes(),
                 audio.data_ptr(),tuple(audio.shape),tuple(audio.stride()),
                 item['planner_mask'].numpy().tobytes())
            group=groups.get(key)
            if group is None:
                group=len(unique);unique.append(row);groups[key]=group
            inverse.append(group)
        result['planner_unique_indices']=torch.tensor(unique,dtype=torch.long)
        result['planner_inverse']=torch.tensor(inverse,dtype=torch.long)
        if 'quality_history_audio' in items[0]:
            unique=[]; inverse=[]; groups={}
            for row,item in enumerate(items):
                audio=item['planner_audio']; clock=item['quality_plan_clock']
                identity=(audio.data_ptr(),tuple(audio.shape),clock.data_ptr(),tuple(clock.shape),item['planner_mask'].numpy().tobytes())
                if identity not in groups:
                    groups[identity]=len(unique);unique.append(row)
                inverse.append(groups[identity])
            result['song_unique_indices']=torch.tensor(unique,dtype=torch.long)
            result['song_inverse']=torch.tensor(inverse,dtype=torch.long)
            result['song_audio']=pad_sequence([items[row]['planner_audio'] for row in unique],batch_first=True,padding_value=0.)
            result['song_mask']=pad_sequence([items[row]['planner_mask'] for row in unique],batch_first=True,padding_value=True)
            result['song_clock']=pad_sequence([items[row]['quality_plan_clock'] for row in unique],batch_first=True,padding_value=0.)

    if 'planner_audio' in items[0]:
        targets=result['action_target']
        active=torch.ones_like(targets,dtype=torch.bool)
        active[:,1:]=targets[:,:-1].ne(0)
        result['packed_slot_indices']=active.nonzero(as_tuple=False)
        counts=active.sum(-1)
        result['packed_first_indices']=counts.cumsum(0)-counts
    return result
