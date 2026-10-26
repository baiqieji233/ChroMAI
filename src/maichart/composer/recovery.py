"""Conservative reference-continuation supervision after real short rollouts.

The reference continuation is not an optimal recovery oracle. Legacy training
supervises its time only; explicit v3 training also supervises its configuration
when it remains legal after the generated prefix. ``propose(item, sample_index)`` must sample the actual model;
the builder never fabricates generated events by perturbing authored labels.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import math

import numpy as np
import torch

from maichart.stream.grounded_inference import IncrementalState
from .acoustics import AcousticContext
from .audio_lens import PLACEMENT_TOLERANCE_SEC
from .candidates import candidate_beats, candidate_features, interp_candidate_audio
from .data import collate_composer
from .events import HEAD_PADS, event_metadata, head_timing_features
from .generate import _history_features
from .guard import EventGuard
from .model import MUSIC_PROFILES
from .workload import prefix_ledger


class RecoveryBuilder:
    def __init__(self, dataset, model=None, device='cpu', *, propose=None,
                 tolerance_sec=PLACEMENT_TOLERANCE_SEC, guard_factory=EventGuard,
                 supervise_actions=False):
        self.dataset, self.model = dataset, model
        self.device, self.propose = torch.device(device), propose
        self.tolerance_sec = float(tolerance_sec)
        if not math.isfinite(self.tolerance_sec) or self.tolerance_sec < PLACEMENT_TOLERANCE_SEC:
            raise ValueError('Recovery tolerance must be finite and at least placement tolerance')
        self.guard_factory = guard_factory
        self.supervise_actions = bool(supervise_actions)
        self._state_cache = None
        self._guard_cache = None

    def _environment(self, sample_index):
        sample = self.dataset.samples[sample_index]
        song = self.dataset.base.songs[sample.song_index][0]
        views = self.dataset.event_views[sample.song_index]
        table = self.dataset._song_table(sample.song_index)
        return sample, song, views, table

    def _guard(self, song, views, prefix, anchor_count):
        identity=(id(views),anchor_count,tuple((p.beat,p.note_text) for p in prefix))
        if self._guard_cache is not None and self._guard_cache[0]==identity:
            guard=self._guard_cache[1]
            self._guard_cache=None
            return guard
        options=({'touch_layout_policy':getattr(self.dataset,'touch_layout_policy','legacy_two_heads')}
                 if self.guard_factory is EventGuard else {})
        guard = self.guard_factory(song.tempo, song.first, views.end_sec,**options)
        for index, point in enumerate(prefix):
            # Authored events are context; every newly generated event is a
            # checked transaction against that context and earlier proposals.
            if index >= anchor_count and not guard.accepts(point):
                return None
            guard.commit(point)
        return guard

    def _audio_context(self, song_index, song, values, times):
        ds = self.dataset
        if not hasattr(ds, '_context_cache') or not hasattr(ds, '_feature_lock'):
            return AcousticContext(values, times, song.audio.duration_sec, validated_bank=True)
        key = ds._song_feature_key(song_index)
        with ds._feature_lock:
            context = ds._context_cache.get(key)
            if context is not None:
                ds._context_cache.move_to_end(key)
                return context
        context = AcousticContext(values, times, song.audio.duration_sec, validated_bank=True)
        with ds._feature_lock:
            context = ds._context_cache.setdefault(key, context)
            ds._context_cache.move_to_end(key)
            while len(ds._context_cache) > ds._context_cache_limit:
                ds._context_cache.popitem(last=False)
        return context

    def _inputs(self, prefix, sample_index):
        ds = self.dataset
        sample, song, views, table = self._environment(sample_index)
        clock = table['clock']
        previous = Fraction(prefix[-1].beat) if prefix else None
        candidates = candidate_beats(previous, tempo=song.tempo, first=song.first,
            audio_end_sec=views.end_sec, candidate_count=ds.candidate_count,
            clock=clock, policy=ds.candidate_policy)
        if not candidates:
            return None, candidates
        actual = len(candidates)
        padded = candidates + [candidates[-1]] * (ds.candidate_count - actual)
        seconds = np.asarray([clock.at(beat) for beat in padded], dtype=np.float64)
        mask = (seconds < 0.) | (seconds >= views.end_sec)
        mask[actual:] = True
        values, times = ds._song_features(sample.song_index)
        context = self._audio_context(sample.song_index, song, values, times)
        features = np.concatenate((interp_candidate_audio(values, times, seconds),
            candidate_features(padded, previous, song.tempo, song.first,
                clock=clock, candidate_seconds=seconds, rich_time=True)), axis=-1)
        history, prefix_seconds, counts = [], [], []
        signature=tuple((p.beat,p.note_text) for p in prefix)
        cached=self._state_cache
        if cached is not None and cached[0]==id(views) and signature[:len(cached[1])]==cached[1]:
            state=cached[2]
            state_start=len(cached[1])
        else:
            state=IncrementalState(song.tempo,song.first)
            state_start=0
        last_seconds = np.full(len(HEAD_PADS), np.nan)
        last_beats = np.full(len(HEAD_PADS), np.nan)
        for position,point in enumerate(prefix):
            meta = event_metadata(str(point.note_text))
            if len(meta.atoms) > ds.action_slots or any(atom not in ds.atom_to_id for atom in meta.atoms):
                raise ValueError('Recovery prefix has unknown or truncated actions')
            atoms = (sorted(meta.atoms) if getattr(ds, 'architecture_profile', '') in MUSIC_PROFILES
                     else meta.atoms)
            ids = [ds.atom_to_id[atom] for atom in atoms]
            history.append((ids + [0] * (ds.action_slots - len(ids)), Fraction(point.beat)))
            second = float(clock.at(point.beat))
            prefix_seconds.append(second)
            counts.append(len(meta.heads))
            for pad in meta.head_pad_indices:
                last_seconds[pad], last_beats[pad] = second, float(point.beat)
            if position>=state_start:state.append(point)
        self._state_cache=(id(views),signature,state)
        committed = previous if previous is not None else Fraction(0)
        from .context_features import composer_action_state
        action_state=composer_action_state(state,len(prefix),padded,
            getattr(ds,'action_state_policy','committed_legacy_v1'))
        tail = history[-ds.history_events:]
        preceding = history[-ds.history_events-1][1] if len(history) > ds.history_events else Fraction(0)
        history_features = _history_features(tail, ds.codec_atoms, song.tempo, song.first,
            ds.action_slots, clock=clock, preceding_beat=preceding, parsed=True)
        count = ds.candidate_count
        item = {
            '_candidate_beats': tuple(padded),
            '_prefix_points': tuple(prefix),
            '_action_index': state,
            '_tempo': song.tempo,
            '_first': song.first,
            '_clock': clock,
            'candidate_features': torch.from_numpy(features),
            'candidate_mask': torch.from_numpy(mask),
            'candidate_target': torch.tensor(0, dtype=torch.long),
            'action_target': torch.zeros(ds.action_slots, dtype=torch.long),
            'history': torch.tensor([ids for ids, _ in tail], dtype=torch.long).reshape(-1, ds.action_slots),
            'history_features': torch.tensor(history_features, dtype=torch.float32).reshape(-1, 8),
            'action_state': action_state,
            'condition': table['condition'],
            'head_timing': torch.from_numpy(head_timing_features(last_seconds, last_beats,
                seconds, [float(beat) for beat in padded])),
            'ledger': torch.from_numpy(prefix_ledger(prefix_seconds,
                np.r_[0., np.cumsum(counts)], seconds)),
            'workload_audio': torch.from_numpy(context.at(seconds)),
            'workload_seconds': torch.from_numpy(seconds),
            'workload_target': torch.zeros(count, 5),
            'workload_weight': torch.zeros(count),
            'interval_audio': torch.from_numpy(context.interval(prefix_seconds[-1] if prefix else 0., seconds)),
            'trailing_audio': torch.from_numpy(context.trailing(seconds)),
            'trailing_target': torch.zeros(count, 4),
            'trailing_weight': torch.zeros(count, 4),
            'future_target': torch.zeros(8), 'future_weight': torch.zeros(8),
            'eos_target': torch.tensor(0.), 'eos_weight': torch.tensor(0.),
            'action_weight': torch.tensor(0.), 'pointer_weight': torch.tensor(1.),
            'sample_weight': torch.tensor(1.), 'transition_weight': torch.tensor(1.),
            'song_index': torch.tensor(sample.song_index),
            'event_index': torch.tensor(sample.event_index),
        }
        if getattr(ds,'motif_memory_policy','off')!='off':
            from .motif_memory import ComposerMotifIndex
            item['_motif_index']=ComposerMotifIndex(getattr(song,'relations',None),
                [p.beat for p in prefix],np.asarray([ids for ids,_ in history],dtype=np.int64).reshape(-1,ds.action_slots))
        if getattr(ds,'architecture_profile',None)=='audio_grounded_v3' and hasattr(ds, '_grounded_features'):
            item.update(ds._grounded_features(sample.song_index, seconds, seconds, table=table))
        return item, candidates

    def build(self, prefix, sample_index, *, generated_count):
        """Return ``(item, reason)``; rejected samples return ``(None, reason)``.

        Prefix must consist of the exact context before the anchor followed by
        two to four actual model events. Only continuation time and explicitly
        enabled legal reference actions are supervised. Independent authored
        placement, workload and future labels remain masked.
        """
        sample, song, views, table = self._environment(sample_index)
        prefix = list(prefix)
        anchor = sample.event_index
        if not 2 <= generated_count <= 4 or len(prefix) != anchor + generated_count:
            return None, 'invalid_rollout_length'
        if any((p.beat, p.note_text) != (q.beat, q.note_text)
               for p, q in zip(prefix[:anchor], views.context_points[:anchor])):
            return None, 'changed_authored_prefix'
        if any(Fraction(right.beat) <= Fraction(left.beat) for left, right in zip(prefix, prefix[1:])):
            return None, 'nonmonotonic_prefix'
        clock = table['clock']
        last_second = float(clock.at(prefix[-1].beat))
        target_index = int(np.searchsorted(views.seconds,
            last_second + self.tolerance_sec + 1e-9, side='right'))
        if target_index >= len(views.context_points):
            return None, 'no_reference_continuation'
        boundary = float(views.seconds[anchor-1]) if anchor else 0.
        target = views.context_points[target_index]
        if (not views.trusted[target_index]
                or float(views.window_weight(boundary, views.seconds[target_index])) <= 0.):
            return None, 'unknown_reference_span'
        try:
            guard = self._guard(song, views, prefix, anchor)
            if guard is None:
                return None, 'illegal_generated_prefix'
            if not guard.accepts(target):
                return None, 'illegal_reference_continuation'
            item, candidates = self._inputs(prefix, sample_index)
        except (ValueError, KeyError):
            return None, 'unsupported_generated_prefix'
        if item is None or target.beat not in candidates:
            return None, 'unreachable_reference_time'
        selected = candidates.index(target.beat)
        if bool(item['candidate_mask'][selected]):
            return None, 'masked_reference_time'
        item['candidate_target'] = torch.tensor(selected, dtype=torch.long)
        if getattr(self.dataset,'action_state_policy','committed_legacy_v1')=='selected_prefix_v3':
            from .context_features import selected_action_state
            item['selected_action_state']=selected_action_state(item['_action_index'],len(prefix),target.beat)
        if '_motif_index' in item:
            from .context_features import motif_tensors
            item.update(motif_tensors(item['_motif_index'],target.beat,len(prefix),
                self.dataset.history_events,self.dataset.motif_max_events))
        item['event_index'] = torch.tensor(target_index)
        item['sample_weight'] = torch.tensor(float(sample.weight))
        # New feature helpers may attach authored labels. Recovery never copies
        # those supervision tensors onto a different generated chart state.
        for key, value in tuple(item.items()):
            if key.endswith(('_target', '_weight')) and key not in {
                    'candidate_target', 'pointer_weight', 'sample_weight', 'transition_weight'}:
                item[key] = torch.zeros_like(torch.as_tensor(value))
        if self.supervise_actions:
            # The same legality check above covers the COMPLETE target event.
            # Never supervise empty placeholder actions or truncate a chord.
            try:
                atoms = sorted(event_metadata(target.note_text).atoms)
            except ValueError:
                return None, 'unsupported_reference_actions'
            if (not atoms or len(atoms) > self.dataset.action_slots
                    or any(atom not in self.dataset.atom_to_id for atom in atoms)):
                return None, 'unsupported_reference_actions'
            ids = [self.dataset.atom_to_id[atom] for atom in atoms]
            item['action_target'] = torch.tensor(ids + [0] * (self.dataset.action_slots-len(ids)), dtype=torch.long)
            item['action_weight'] = torch.tensor(1.)
        return item, 'accepted'

    def rollout(self, sample_index, *, steps=2, propose=None):
        """Run a real model callback for each event; return prefix or rejection."""
        if steps not in (2, 3, 4):
            raise ValueError('Recovery rollouts must contain two to four events')
        propose = propose or self.propose
        if propose is None:
            raise ValueError('Recovery requires an actual model proposal callback')
        sample, song, views, table = self._environment(sample_index)
        prefix = list(views.context_points[:sample.event_index])
        guard = self._guard(song, views, prefix, sample.event_index)
        for _ in range(steps):
            item, candidates = self._inputs(prefix, sample_index)
            if item is None:
                return None, 'rollout_candidates_exhausted'
            item['_proposal_guard']=guard
            point = propose(item, sample_index)
            if point is None:
                return None, 'rollout_stopped'
            if point.beat not in candidates or (prefix and point.beat <= prefix[-1].beat):
                return None, 'rollout_unreachable_proposal'
            second = float(table['clock'].at(point.beat))
            boundary = float(views.seconds[sample.event_index-1]) if sample.event_index else 0.
            if float(views.window_weight(boundary, second)) <= 0.:
                return None, 'unknown_rollout_span'
            if not guard.accepts(point):
                return None, 'illegal_rollout_proposal'
            guard.commit(point)
            prefix.append(point)
        self._guard_cache=((id(views),sample.event_index,tuple((p.beat,p.note_text) for p in prefix)),guard)
        return prefix, 'accepted'

    def batch_from_indices(self, indices, *, steps=2, propose=None):
        """Return ``(collated_batch_or_None, rejection_counts)`` on CPU.

        The callback controls model evaluation and sampling. No optimizer step
        occurs here, and model train/eval state is never changed implicitly.
        """
        items, diagnostics = [], Counter()
        for index in indices:
            prefix, reason = self.rollout(int(index), steps=steps, propose=propose)
            if prefix is not None:
                item, reason = self.build(prefix, int(index), generated_count=steps)
                if item is not None:
                    items.append({key: value for key, value in item.items() if not key.startswith('_')})
            diagnostics[reason] += 1
        if not items:
            return None, dict(diagnostics)
        batch = collate_composer(items)
        # Keep optional/new-profile tensors even with an older collator.
        for key in items[0]:
            if key not in batch and key != 'candidate_mask' and not key.startswith('_'):
                batch[key] = torch.stack([torch.as_tensor(item[key]) for item in items])
        return batch, dict(diagnostics)
