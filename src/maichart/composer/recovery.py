"""Reference-continuation supervision on retained, actually generated histories.

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
from .candidates import candidate_beats, candidate_features, interp_candidate_audio, is_playable_candidate_transition
from .data import collate_composer, ComposerDataset
from .events import event_metadata, head_timing_features,head_pads_for_policy,head_pad_indices_for_policy
from .generate import _history_features
from .guard import EventGuard
from .model import MUSIC_PROFILES
from .workload import prefix_ledger
from maichart.event_timing import EventPoint

RECOVERY_EPISODE_SCHEMA='committed_generated_episodes_v1'


def recovery_episode_state(model):
    import copy
    return {'schema':RECOVERY_EPISODE_SCHEMA,'episodes':copy.deepcopy(getattr(model,'_recovery_episodes',[]))}


def restore_recovery_episodes(model,state):
    import copy
    if state is None:model._recovery_episodes=[];return
    if state.get('schema')!=RECOVERY_EPISODE_SCHEMA or not isinstance(state.get('episodes'),list) or len(state['episodes'])>8:
        raise ValueError('Invalid recovery episode state')
    for record in state['episodes']:
        if record is None:continue
        if (not isinstance(record,dict) or not isinstance(record.get('sample_index'),int)
                or record['sample_index']<0 or not isinstance(record.get('events'),list) or len(record['events'])>132):
            raise ValueError('Invalid recovery episode record')
        for item in record['events']:
            if not isinstance(item,(tuple,list)) or len(item)!=2 or not isinstance(item[1],str):
                raise ValueError('Invalid recovery episode event')
            Fraction(item[0])
    model._recovery_episodes=copy.deepcopy(state['episodes'])


class RecoveryBuilder:
    def __init__(self, dataset, model=None, device='cpu', *, propose=None,
                 tolerance_sec=PLACEMENT_TOLERANCE_SEC, guard_factory=EventGuard,
                 supervise_actions=False, supervise_eos=False, maximum_generated_events=4):
        self.dataset, self.model = dataset, model
        self.device, self.propose = torch.device(device), propose
        self.tolerance_sec = float(tolerance_sec)
        if not math.isfinite(self.tolerance_sec) or self.tolerance_sec < PLACEMENT_TOLERANCE_SEC:
            raise ValueError('Recovery tolerance must be finite and at least placement tolerance')
        self.guard_factory = guard_factory
        self.supervise_actions = bool(supervise_actions)
        self.supervise_eos = bool(supervise_eos)
        if not 4<=maximum_generated_events<=132:raise ValueError('Invalid recovery episode horizon')
        self.maximum_generated_events=int(maximum_generated_events)
        self.last_reference_kind=None
        self.playability_unreachable_references=0
        self.reference_reachability_counts=Counter()
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
        timing_policy=getattr(ds,'head_timing_policy','merged_pads_v1')
        last_seconds = np.full(len(head_pads_for_policy(timing_policy)), np.nan)
        last_beats = np.full(len(last_seconds), np.nan)
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
            for pad in head_pad_indices_for_policy(meta,timing_policy):
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
        from .history_clock import history_beat_ages
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
            'history_beat_ages': torch.from_numpy(history_beat_ages([beat for _, beat in tail])),
            'history_second_ages': torch.tensor(
                [prefix_seconds[-1] - float(clock.at(beat)) for _, beat in tail], dtype=torch.float32),
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
            # A generated prefix is context, never an authored event-rate label.
            'event_count_target': torch.zeros(count),
            'event_exposure': torch.zeros(count),
            'event_rate_weight': torch.zeros(count),
            'survival_start': torch.tensor(0., dtype=torch.float64),
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
        if getattr(ds, 'fixed_future_audio', False):
            # A generated prefix provides no authored empty-gap supervision.
            item.update(empty_gap_audio=torch.zeros(4, 104),
                        empty_gap_exposure=torch.zeros(4),
                        empty_gap_weight=torch.zeros(4))
        if getattr(ds, 'quality_inputs_enabled', False):
            from .quality_inputs import quality_context_inputs
            song_paired = getattr(ds, 'quality_inputs_policy', None) == 'song_paired_v3'
            if song_paired and 'planner_data' not in table:
                ds._grounded_features(sample.song_index, seconds, seconds, table=table, context=context, defer_queries=False)
            quality_start = max(0, len(prefix)-ds.history_events) if song_paired else 0
            item.update(quality_context_inputs(points=prefix[quality_start:],
                include_future=getattr(ds,'fixed_future_audio',False),
                atom_ids=np.asarray([ids for ids, _ in history[quality_start:]], dtype=np.int64).reshape(-1, ds.action_slots),
                candidate_beats=padded, candidate_seconds=seconds, state_index=state,
                audio_features=values, audio_times=times, acoustic_context=context, clock=clock,
                audio_duration=table['audio_duration'], point_seconds=prefix_seconds[quality_start:],
                policy=getattr(ds, 'quality_inputs_policy', None), plan_grid=table.get('planner_data')))
            # A generated prefix has changed the problem. Authored future
            # phrase/gap labels are not silently reused as recovery targets.
            phrases = len(item['quality_phrase_audio'])
            item.update(quality_phrase_target=torch.zeros(phrases, 9 if song_paired else 8), quality_phrase_weight=torch.zeros(phrases),
                quality_gap_target=torch.tensor(0, dtype=torch.long), quality_gap_weight=torch.tensor(0.))
        if getattr(ds,'motif_memory_policy','off')!='off':
            from .motif_memory import ComposerMotifIndex
            item['_motif_index']=ComposerMotifIndex(getattr(song,'relations',None),
                [p.beat for p in prefix],np.asarray([ids for ids,_ in history],dtype=np.int64).reshape(-1,ds.action_slots))
        if getattr(ds,'architecture_profile',None)=='audio_grounded_v3' and hasattr(ds, '_grounded_features'):
            item.update(ds._grounded_features(sample.song_index, seconds, seconds, table=table,defer_queries=False))
            if 'planner_data' in table:
                item['_planner_centers']=table['planner_data'].centers
        if getattr(ds,'structure_policy','off')!='off':
            from .structure import structure_features_and_transition
            features,transition=structure_features_and_transition(song,padded,seconds,
                float(previous) if previous is not None else None,
                float(clock.at(previous)) if previous is not None else None,
                policy=ds.structure_policy,cache_dir=ds.cache_dir,backend=ds.structure_backend)
            item['structure_features']=torch.from_numpy(features)
            item['structure_transition']=torch.from_numpy(transition)
        if getattr(ds,'fixed_future_audio',False) and 'quality_future_audio' not in item:
            from .quality_inputs import fixed_future_audio_inputs
            item.update(fixed_future_audio_inputs(context,clock,padded,table['audio_duration']))
        if getattr(ds, 'timing_distribution_policy', 'hierarchical_v1') == 'survival_integrated_v2':
            from .density_grid import survival_grid_inputs
            item.update({key: torch.from_numpy(value) for key, value in survival_grid_inputs(
                context, prefix_seconds[-1] if prefix else 0., seconds, table['audio_duration'],
                plan_grid=table.get('planner_data'),
                prefix_seconds=(prefix_seconds if getattr(ds, 'density_feedback_policy', 'off')
                                == 'causal_count_debt_v1' else None)).items()})
        return item, candidates

    def build(self, prefix, sample_index, *, generated_count):
        """Return ``(item, reason)``; rejected samples return ``(None, reason)``.

        Prefix must consist of the exact context before the anchor followed by
        actual model events within the bounded episode. Each visit adds only
        two to four new events. Only continuation time and explicitly
        enabled legal reference actions are supervised. Independent authored
        placement, workload and future labels remain masked.
        """
        sample, song, views, table = self._environment(sample_index)
        prefix = list(prefix)
        self.last_reference_kind=None
        anchor = sample.event_index
        if not 2 <= generated_count <= self.maximum_generated_events or len(prefix) != anchor + generated_count:
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
        counts=event_metadata(target.note_text).type_counts
        self.last_reference_kind=('mixed' if sum(counts[:3]) and sum(counts[3:]) else 'touch' if sum(counts[3:]) else 'button')
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
        playable_transition = is_playable_candidate_transition(
            target.beat,
            Fraction(prefix[-1].beat) if prefix else None,
            getattr(self.dataset, 'candidate_policy', None),
        )
        if item is None or target.beat not in candidates:
            self.reference_reachability_counts['generator_unreachable'] += 1
            if not playable_transition:
                self.playability_unreachable_references += 1
            return None, 'unreachable_reference_time'
        selected = candidates.index(target.beat)
        if bool(item['candidate_mask'][selected]):
            self.reference_reachability_counts['prefilter_masked'] += 1
            if not playable_transition:
                self.playability_unreachable_references += 1
            return None, 'masked_reference_time'
        if not playable_transition:
            self.playability_unreachable_references += 1
            self.reference_reachability_counts['playability_filtered'] += 1
        else:
            self.reference_reachability_counts['reachable'] += 1
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
        if self.supervise_eos:
            eos_target, eos_weight = views.eos(target_index)
            item['eos_target'] = torch.tensor(float(eos_target), dtype=torch.float32)
            item['eos_weight'] = torch.tensor(float(eos_weight), dtype=torch.float32)
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
            item['action_target'] = ComposerDataset.ordered_action_target(self.dataset,
                torch.tensor(ids + [0] * (self.dataset.action_slots-len(ids)), dtype=torch.long), sample_index)
            item['action_weight'] = torch.tensor(1.)
        return item, 'accepted'

    def rollout(self, sample_index, *, steps=2, propose=None,initial_prefix=None):
        """Run a real model callback for each event; return prefix or rejection."""
        if steps not in (2, 3, 4):
            raise ValueError('Recovery rollouts must contain two to four events')
        propose = propose or self.propose
        if propose is None:
            raise ValueError('Recovery requires an actual model proposal callback')
        sample, song, views, table = self._environment(sample_index)
        prefix = list(views.context_points[:sample.event_index] if initial_prefix is None else initial_prefix)
        if (not sample.event_index<=len(prefix)<=sample.event_index+self.maximum_generated_events-steps
                or any((p.beat,p.note_text)!=(q.beat,q.note_text)
                       for p,q in zip(prefix[:sample.event_index],views.context_points[:sample.event_index]))):
            return None,'invalid_episode_prefix'
        guard = self._guard(song, views, prefix, sample.event_index)
        if guard is None:return None,'illegal_generated_prefix'
        for _ in range(steps):
            item, candidates = self._inputs(prefix, sample_index)
            if item is None:
                return None, 'rollout_candidates_exhausted'
            item['_proposal_guard']=guard
            point = propose(item, sample_index)
            if point is None:
                return None, item.get('_proposal_stop_reason', 'rollout_stopped')
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

    def batch_from_episodes(self,indices,episodes,*,steps=2):
        """Advance a bounded number of real proposals, retaining generated histories."""
        items=[];outcomes=Counter();updated=[];reference_types={};lengths=[]
        for slot,default_index in enumerate(indices):
            record=episodes[slot] if slot<len(episodes) else None
            if record and len(record['events'])+steps>self.maximum_generated_events:record=None
            index=int(record['sample_index'] if record else default_index)
            if not 0<=index<len(self.dataset.samples):raise ValueError('Recovery episode sample index is outside the dataset')
            sample,_,views,_=self._environment(index)
            generated=[] if record is None else [EventPoint(Fraction(beat),text) for beat,text in record['events']]
            prefix=list(views.context_points[:sample.event_index])+generated
            prefix,reason=self.rollout(index,steps=steps,initial_prefix=prefix)
            keep=None
            if prefix is not None:
                generated=prefix[sample.event_index:];lengths.append(len(generated))
                item,reason=self.build(prefix,index,generated_count=len(generated))
                kind=self.last_reference_kind
                if kind:
                    values=reference_types.setdefault(kind,{'accepted':0,'rejected':0,'accepted_weight':0.,'rejected_weight':0.})
                    outcome='accepted' if item is not None else 'rejected'
                    values[outcome]+=1;values[outcome+'_weight']+=float(sample.weight)
                if item is not None:
                    items.append({key:value for key,value in item.items() if not key.startswith('_')})
                if reason not in {'no_reference_continuation','unknown_reference_span','unsupported_generated_prefix'}:
                    keep={'sample_index':index,'events':[[str(point.beat),point.note_text] for point in generated]}
            updated.append(keep);outcomes[reason]+=1
        diagnostics={'generated_history_events':lengths,'maximum_generated_events':self.maximum_generated_events,
                     'reference_types':reference_types,'self_generated_window':int(self.dataset.history_events),
                     'playability_unreachable_references':self.playability_unreachable_references,
                     'reference_reachability':dict(self.reference_reachability_counts)}
        return (collate_composer(items) if items else None),dict(outcomes),updated,diagnostics

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
