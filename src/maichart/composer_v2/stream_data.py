"""Event-aligned computational slices. No synthetic end-of-window targets."""
from collections import OrderedDict
from fractions import Fraction
import bisect
import hashlib
import json
import math
import numpy as np
import torch
from .data import ComposerV2Dataset,collate_composer_v2
from .defaults import (MUSICAL_CONTEXT_PROBABILITY, AUDIO_WRITING_POLICIES, BOUNDARY_WRITING,
                       GRID_WRITING_POLICIES, MOTION_WRITING_POLICIES)
from .stream_codec import POLICY,StreamCodec
from . import musical_context, ordered_configuration, relation_sampling, timing_recovery, phrase_replay


_SELECT_REFERENCE = object()
_REFERENCE_CACHE_SONGS = 128
_REFERENCE_QUERY_ROWS = 1024
_CONFIGURATION_CACHE_SONGS = 32


def _song_cache_key(song):
    clock = song.clock
    content = (str(song.audio_path) + str(clock.beats) + str(clock.bpms) + str(clock.seconds)
               + json.dumps(song.audio.identity, sort_keys=True))
    return hashlib.sha256(content.encode()).hexdigest()


class StreamDataset(ComposerV2Dataset):
    def __init__(self,*args,refresh_events=16,timing_recovery_policy='off',augmentation_seed=0,onset_readout_policy='off',
                 structure_policy='off',reference_augmentation=False,intensity_policy='off',rhythm_learning_policy=None,
                 music_context_policy='off',reviewed_reference_policy='approved_only',
                 zero_unmarked_policy='explicit_good_v1',reference_quality_policy='event_mask_v1',
                 local_audio_block_beats=1,training_window_events=None,
                 musical_context_policy='off',musical_context_probability=MUSICAL_CONTEXT_PROBABILITY,
                 musical_relation_policy='off', musical_relation_fraction=.3,
                 future_configuration_policy='off', auxiliary_validation=False, **kwargs):
        if type(local_audio_block_beats) is not int or local_audio_block_beats not in (1, 2):
            raise ValueError('local_audio_block_beats must be 1 or 2')
        if musical_context_policy not in musical_context.POLICIES:
            raise ValueError('Unknown musical context view policy')
        musical_context_probability = float(musical_context_probability)
        if not 0 <= musical_context_probability <= 1:
            raise ValueError('musical_context_probability must be between 0 and 1')
        if musical_relation_policy not in ('off', relation_sampling.POLICY):
            raise ValueError('Unknown musical relation policy')
        musical_relation_fraction = float(musical_relation_fraction)
        if not 0 <= musical_relation_fraction < 1:
            raise ValueError('musical_relation_fraction must be between 0 and 1 (exclusive)')
        if future_configuration_policy not in ('off', ordered_configuration.POLICY, *AUDIO_WRITING_POLICIES):
            raise ValueError('Unknown future configuration policy')
        self.musical_relation_policy = musical_relation_policy
        self.musical_relation_fraction = musical_relation_fraction
        self.future_configuration_policy = future_configuration_policy
        self.auxiliary_validation = bool(auxiliary_validation)
        self._configuration_cache = OrderedDict()
        self._writing_audio_cache = OrderedDict()
        self.musical_context_policy = (
            musical_context.POLICY if musical_context_policy == 'auto' else musical_context_policy)
        self.musical_context_probability = musical_context_probability
        self.local_audio_block_beats = local_audio_block_beats
        super().__init__(*args,sequence_policy=POLICY,**kwargs)
        from .rhythm_structure import POLICIES
        if structure_policy not in POLICIES:raise ValueError('Unknown rhythm structure policy')
        if reviewed_reference_policy not in ('approved_only','off'):raise ValueError('Unknown reviewed reference policy')
        self.structure_policy=structure_policy
        self.intensity_policy=intensity_policy
        self.music_context_policy=music_context_policy
        self.reviewed_reference_policy=reviewed_reference_policy
        if zero_unmarked_policy not in ('explicit_good_v1','legacy_weights_v1'):
            raise ValueError('Unknown zero-unmarked supervision policy')
        self.zero_unmarked_policy=zero_unmarked_policy
        if reference_quality_policy not in ('legacy_v1','event_mask_v1'):
            raise ValueError('Unknown reference quality policy')
        self.reference_quality_policy=reference_quality_policy
        from .rhythm_learning import POLICY as LEGACY_LEARNING
        self.rhythm_learning_policy=rhythm_learning_policy or LEGACY_LEARNING
        self.reference_augmentation=bool(reference_augmentation) and structure_policy!='off'
        from .onset_readout import POLICY as ONSET_POLICY, phase_values
        if onset_readout_policy not in ('off', ONSET_POLICY):raise ValueError('Unknown onset readout policy')
        self.onset_readout_policy = onset_readout_policy
        self.onset_phases = phase_values(self.codec) if onset_readout_policy != 'off' else None
        if not isinstance(self.codec,StreamCodec):raise ValueError('Continuous training requires a new vocabulary and fresh checkpoint')
        self.refresh_events=int(refresh_events)
        if self.refresh_events<1:raise ValueError('refresh_events must be positive')
        self.training_window_events = self.refresh_events if training_window_events is None else training_window_events
        if type(self.training_window_events) is not int or self.training_window_events < 1:
            raise ValueError('training_window_events must be a positive integer')
        if timing_recovery_policy not in timing_recovery.POLICIES:
            raise ValueError('Unknown timing recovery policy')
        self.timing_recovery_policy=timing_recovery_policy
        # Recovery keeps its original target boundaries and random seeds when
        # authored targets are grouped into longer training windows.
        recovery_slices = [] if (timing_recovery_policy != 'off'
                                 and self.training_window_events != self.refresh_events) else None
        self.tile_cache=OrderedDict();self.windows=[];self.ranges=[];self.lengths=[]
        self._remaining_references = []
        from maichart.training.quality_annotations import region_component_weights, zero_unmarked_region_weights
        legacy_zero_resume = self.zero_unmarked_policy == 'legacy_weights_v1'
        for song_index,song in enumerate(self.songs):
            first_window = len(self.windows)
            reviewed = (song.quality or {}).get('unmarked_weight') == 0
            song._legacy_zero_unmarked_resume = legacy_zero_resume
            if reviewed:
                song.reviewed_event_seconds = song.clock.at_floats(np.asarray(song.point_beats, dtype=np.float64))
            token_lengths = song.event_token_lengths
            song.event_token_lengths = []
            weight_fn = region_component_weights if legacy_zero_resume else zero_unmarked_region_weights
            for begin, end, length in self._authored_slices(
                    song, token_lengths, self.training_window_events, reviewed, weight_fn):
                self.windows.append((song_index, begin))
                self.ranges.append(end)
                self.lengths.append(length)
            if recovery_slices is not None:
                recovery_slices.extend(
                    (song_index, begin, end, length)
                    for begin, end, length in self._authored_slices(
                        song, token_lengths, self.refresh_events, reviewed, weight_fn)
                )
            self._remaining_references.append(len(self.windows) - first_window)
        reference_count = len(self.windows)
        self._relation_scores = None
        if musical_relation_policy != 'off':
            self._prepare_relation_scores()
        if future_configuration_policy != 'off':
            self.audit['future_configuration'] = dict(
                policy=future_configuration_policy, horizons=ordered_configuration.HORIZONS,
                dimension=ordered_configuration.TARGET_DIM, max_queries=ordered_configuration.MAX_QUERIES,
                target='authored_successor_events_and_ordered_beat_intervals',
                quality='continuous_reviewed_audio_coverage', timing_recovery=False,
                probability=1. if self.auxiliary_validation else ordered_configuration.TRAINING_PROBABILITY,
                chart_condition='level_only_on_auxiliary_windows', curve_teacher=False)
            if future_configuration_policy in AUDIO_WRITING_POLICIES:
                from .local_writing import FIELDS, MOTION_FIELDS, BEAT_INTERVALS
                motion = future_configuration_policy in MOTION_WRITING_POLICIES
                self.audit['future_configuration'].update(
                    horizons=BEAT_INTERVALS,dimension=len(MOTION_FIELDS if motion else FIELDS),max_queries=None,
                    probability=1.,target='audio_predicted_ordered_writing',
                    chart_condition='planner_level_only_decoder_optional_conditions',
                    timing_recovery=timing_recovery_policy!='off', curve_teacher='ordinary_views' if motion else False,
                    execution='predicted_motion_and_next_integer' if motion else 'predicted_inputs_only_at_configuration_decisions')
                if future_configuration_policy in GRID_WRITING_POLICIES:
                    from .phrase_decisions import GRID_FIELDS, DECISION_FIELDS
                    self.audit['future_configuration'].update(dimension=len(MOTION_FIELDS)+len(GRID_FIELDS),
                        successor_fields=DECISION_FIELDS if future_configuration_policy==BOUNDARY_WRITING else (),
                        extra_grid_rows_per_window=2)
        self.recovery_sources = {}
        self._recovery_seed_indices = {}
        self.training_epoch = 0
        self.augmentation_seed = int(augmentation_seed)
        self._recovery_span = min(timing_recovery.SPAN, Fraction(self.history_beats))
        self._recovery_phases = frozenset(self.codec.phases.values())
        self._timing_ids = timing_recovery.timing_ids(self.codec)
        if timing_recovery_policy != 'off':
            recovery_count = reference_count if recovery_slices is None else len(recovery_slices)
            for index in range(7, recovery_count, 8):
                if recovery_slices is None:
                    si, begin = self.windows[index]
                    end, length = self.ranges[index], self.lengths[index]
                else:
                    si, begin, end, length = recovery_slices[index]
                if begin<2:continue
                song=self.songs[si]
                first=bisect.bisect_left(song.point_beats,song.point_beats[begin-1]-self._recovery_span,hi=begin)
                if any(timing_recovery.is_tap(p) for p in song.points[first:begin-1]):
                    recovery_index = len(self.windows)
                    source_index = index
                    if recovery_slices is not None:
                        source_index = recovery_index
                        self._recovery_seed_indices[source_index] = index
                        self._remaining_references[si] += 1
                    self.recovery_sources[recovery_index] = source_index
                    self.windows.append((si, begin))
                    self.ranges.append(end)
                    self.lengths.append(length)
        if recovery_slices is not None:
            reference_count = len(self.windows)
        self._reference_starts = [None] * reference_count
        self._reference_ready = bytearray(reference_count)
        self._reference_descriptors = OrderedDict()
        self.audit.update(windows=len(self.windows),sequence_policy=POLICY,
            segmentation='whole_events_compute_only',refresh_events=self.refresh_events,eos='true_chart_end_only',
            reviewed_reference_policy=self.reviewed_reference_policy)
        if self.training_window_events != self.refresh_events:
            self.audit['training_window_events'] = self.training_window_events
        self.audit['timing_recovery'] = dict(policy=timing_recovery_policy, views=len(self.recovery_sources),
            original_windows=len(self.windows)-len(self.recovery_sources),target='authored_future_time_only')
        if timing_recovery_policy == timing_recovery.LAYOUT_POLICY:
            self.audit['timing_recovery']['target']='timing_or_sampled_tap_prefix_with_two_authored_contacts'
        if timing_recovery_policy == timing_recovery.GESTURE_POLICY:
            self.audit['timing_recovery']['target']='timing_or_same_song_gesture_pair'
        if self.musical_context_policy != 'off':
            self.audit['musical_context'] = dict(policy=self.musical_context_policy,
                probability=self.musical_context_probability, recent_beats=musical_context.RECENT_BEATS,
                original_windows=len(self.windows)-len(self.recovery_sources), appended_windows=0,
                timing_recovery_windows_excluded=len(self.recovery_sources),
                target='unchanged_authored_future', reference='disabled',
                chart_condition='level_only', curve_teacher='disabled')
        if structure_policy!='off':
            from . import rhythm_structure as rs
            self.audit['rhythm_structure']=dict(policy=structure_policy,reference='per_beat_evidence_full_coverage_v2' if self._evidence_v2() else 'aligned_integer_beat_shift_trust_v1',
                min_score=rs.REF_MIN_SCORE,lead_beats=rs.REF_LEAD_BEATS,match_ahead_beats=self.window_beats if self._evidence_v2() else rs.REF_MATCH_AHEAD_BEATS,
                display_ahead_beats=self.window_beats,augmentation=self.reference_augmentation,
                drop_probability=rs.REF_DROP_PROBABILITY if self.reference_augmentation else 0.,
                alternative_probability=rs.REF_ALTERNATIVE_PROBABILITY if self.reference_augmentation else 0.,
                history_budget='low_trust_uses_spare_only' if self._evidence_v2() else 'reference_first',
                learning=self.rhythm_learning_policy if self._evidence_v2() and self.reference_augmentation else 'off')

    def _prepare_relation_scores(self):
        scores = np.zeros(len(self.windows), dtype=np.float32)
        counts = dict.fromkeys(relation_sampling.STRATA, 0)
        first = 0
        for si, count in enumerate(self._remaining_references):
            windows = [(index, self.windows[index][1], self.ranges[index])
                       for index in range(first, first + count)]
            first += count
            if not windows or self.musical_relation_fraction == 0:
                continue
            song = self.songs[si]
            chart = self._configuration(si) if self.future_configuration_policy != 'off' else (
                ordered_configuration.ChartConfiguration(song))
            for index, (score, name) in relation_sampling.source_scores(song, chart, windows).items():
                scores[index] = score
                counts[name] += 1
        self._relation_scores = scores
        self.audit['musical_relation'] = dict(
            policy=self.musical_relation_policy, maximum_extra_supervision_fraction=self.musical_relation_fraction,
            scored_windows=int(np.count_nonzero(scores)), original_windows=len(scores),
            strata=counts, mode='reweight_existing_authored_windows', timing_recovery=False)

    def _configuration(self, si):
        cache = self._configuration_cache
        if si not in cache:
            cache[si] = ordered_configuration.ChartConfiguration(self.songs[si])
            if len(cache) > _CONFIGURATION_CACHE_SONGS:
                cache.popitem(last=False)
        cache.move_to_end(si)
        return cache[si]

    def _authored_slices(self, song, token_lengths, event_limit, reviewed, weight_fn):
        # Empty charts and actual song endings still have one real EOS target.
        if not song.points:
            yield 0, 0, 1
        target_budget = self.max_tokens // 2
        stops = ()
        allowed = None
        if reviewed and self.future_configuration_policy in MOTION_WRITING_POLICIES and song.points:
            # Start a new target exactly where a reviewed use changes. build_stream
            # still receives the real previous events and reconstructs their carries.
            quality = ordered_configuration.ConfigurationQuality(song)
            lanes = quality.at(song.reviewed_event_seconds) > 0
            allowed = lanes.any(1)
            stops = np.flatnonzero(np.any(lanes[1:] != lanes[:-1], axis=1)) + 1
        begin = 0
        while begin < len(song.points):
            boundary = len(song.points)
            if allowed is not None:
                i = bisect.bisect_right(stops, begin)
                if i < len(stops):
                    boundary = int(stops[i])
                if not allowed[begin]:
                    begin = boundary
                    continue
            end, tokens = begin, 0
            origin = math.floor(song.points[begin - 1].beat) if begin else 0
            while end < boundary and end - begin < event_limit:
                length = token_lengths[end]
                if tokens + length + 1 > target_budget:
                    break
                tokens += length
                end += 1
                # This boundary adds no synthetic note, rest or EOS.
                if self._evidence_v2() and song.points[end - 1].beat >= origin + self.window_beats:
                    break
            if end == begin:
                raise ValueError(f'One whole event exceeds target budget: {song.source_path}, {song.points[begin].beat}')
            if allowed is not None or not reviewed or any(
                region['start_sec'] <= stamp < region['end_sec']
                for region in song.quality.get('regions', [])
                if max(weight_fn(region)) > 0
                for stamp in song.reviewed_event_seconds[begin:end]
            ):
                yield begin, end, tokens + 1
            begin = end

    def set_epoch(self, epoch):
        self.training_epoch = int(epoch)
    def _structured(self):
        return getattr(self,'structure_policy','off')!='off'
    def _evidence_v2(self):
        return getattr(self,'structure_policy','off')=='aligned_rhythm_v2'
    def _paired_music(self):
        return getattr(self,'music_context_policy','off')=='song_paired_v2'
    def _reference_points_approved(self,song,reference,*,beat_offset=0):
        """Approve [REF] on zero-unmarked charts only when every post-budget
        reference event supplied to the model lies inside an explicit ``good``
        configuration approval (``component_weight > 0``); when ``auto_quality_policy``
        is active, also reject [REF] spans overlapping masked problem regions."""
        quality = song.quality or {}
        if quality.get('unmarked_weight') != 0:
            legacy = getattr(self,'reference_quality_policy','event_mask_v1') == 'legacy_v1'
            if not reference or (legacy and getattr(self, 'auto_quality_policy', 'off') != 'auto_v1'):
                return True
            from maichart.training.quality_annotations import region_component_weights
            bad_spans = []
            for region in quality.get('regions', ()):
                if region.get('label') == 'problem' or region_component_weights(region)[1] <= 0:
                    start_sec, end_sec = float(region['start_sec']), float(region['end_sec'])
                    if end_sec > start_sec:
                        bad_spans.append((start_sec, end_sec))
            if not bad_spans:
                return True
            seconds = song.clock.at_floats(np.asarray([float(point.beat - beat_offset) for point in reference], dtype=np.float64))
            if legacy:
                first, last = seconds.min(), seconds.max()
                return not any(first < end_sec and last >= start_sec for start_sec, end_sec in bad_spans)
            return not any(np.any((seconds >= start_sec) & (seconds < end_sec))
                           for start_sec,end_sec in bad_spans)
        if getattr(self,'reviewed_reference_policy','approved_only') == 'off' or not reference:
            return False
        if quality.get('label') in ('excluded', 'negative') or float(quality.get('weight', 1.)) <= 0:
            return False
        from maichart.training.quality_annotations import zero_unmarked_region_weights
        spans = []
        for region in quality.get('regions', ()):
            if zero_unmarked_region_weights(region, require_explicit_good=True)[1] <= 0:
                continue
            start_sec, end_sec = float(region['start_sec']), float(region['end_sec'])
            if end_sec > start_sec:
                spans.append((start_sec, end_sec))
        if not spans:
            return False
        seconds = song.clock.at_floats(np.asarray([float(point.beat - beat_offset) for point in reference], dtype=np.float64))
        approved = np.zeros(len(seconds), dtype=bool)
        for start_sec, end_sec in spans:
            approved |= (seconds >= start_sec) & (seconds < end_sec)
        return bool(approved.all())
    def _profile(self,song):
        from .rhythm_structure import ProfileCache,song_profile
        cache=getattr(self,'_profiles',None)
        if cache is None:cache=self._profiles=ProfileCache()
        if song.stream_cache_key is None:song.stream_cache_key=_song_cache_key(song)
        return cache.get(song.stream_cache_key,lambda:song_profile(song.audio,song.clock,song.end_beat))
    def _aligned_match(self,song,cursor,alternative_draw=None):
        """Audio-only search; committed chart notes are only read after the match."""
        from .rhythm_structure import match_reference
        return match_reference(self._profile(song),cursor,min_shift=self.window_beats,alternative_draw=alternative_draw,
            ahead_beats=self.window_beats if self._evidence_v2() else 8)
    def _aligned_legacy_start(self,song,cursor):
        """Training-free aligned choice for checkpoints that learned the legacy
        REF layout: same original-position 16-beat window, chosen by the
        beat-synchronous match and omitted when the music does not repeat."""
        from .rhythm_structure import match_reference,REF_MIN_SCORE
        match=match_reference(self._profile(song),cursor,min_shift=self.history_beats+self.window_beats)
        if match is None or match['best_score']<REF_MIN_SCORE:
            self._last_aligned=None;return None
        self._last_aligned=(match['best_shift'],match['best_score'])
        return Fraction(math.floor(Fraction(cursor)))-match['best_shift']
    def _selected_reference(self,song,cursor):
        from .rhythm_structure import REF_MIN_SCORE
        match=self._aligned_match(song,cursor)
        if match is None or match['best_score']<REF_MIN_SCORE:return None
        return (match['best_shift'],match['best_score'])
    def _reference_argument(self,index):
        zero_unmarked = (self.songs[self.windows[index][0]].quality or {}).get('unmarked_weight') == 0
        if zero_unmarked and getattr(self,'reviewed_reference_policy','approved_only') == 'off':
            return None
        if not self._structured():return self._prepare_reference(index)
        from . import rhythm_structure as rs
        info=self._prepare_reference(index)
        if info is None:return None
        if getattr(self,'reference_augmentation',False):
            # Deterministic per window and epoch: occasionally hide the match or
            # show an arbitrary passage with its true (usually low) trust, so
            # trust, not mere presence, decides how much is reused.
            seed_index = getattr(self, '_recovery_seed_indices', {}).get(index, index)
            draw=float(np.random.default_rng((self.augmentation_seed,seed_index,self.training_epoch,7)).random())
            if draw<rs.REF_DROP_PROBABILITY:return None
            if not zero_unmarked and draw<rs.REF_DROP_PROBABILITY+rs.REF_ALTERNATIVE_PROBABILITY:
                return (info['alternative_shift'],info['alternative_score'])
        if info['best_score']<rs.REF_MIN_SCORE:return None
        return (info['best_shift'],info['best_score'])
    def _reference_stream(self, song, points, cursor, *, prefix_end=None):
        stop = cursor - self.history_beats - self.window_beats
        if prefix_end is None:
            beats = [point.beat for point in points if point.beat <= stop]
            count = len(beats)
        else:
            beats = song.point_beats
            count = bisect.bisect_right(beats, stop, hi=prefix_end)
        if count == 0:
            return None
        indices = np.linspace(0, count - 1, 64, dtype=int) if count > 64 else range(count)
        candidates = [beats[index] for index in indices]
        starts = [cursor, *candidates]
        cache = song.descriptor_cache
        descriptors = {start: cache[start] for start in starts if start in cache}
        missing = list(dict.fromkeys(start for start in starts if start not in descriptors))
        if missing:
            offsets = np.arange(self.window_beats * 4) / 4
            starts_per_read = max(1, _REFERENCE_QUERY_ROWS // len(offsets))
            for index in range(0, len(missing), starts_per_read):
                chunk = missing[index:index + starts_per_read]
                beat_queries = np.asarray(chunk, dtype=np.float64)[:, None] + offsets
                unique, inverse = np.unique(beat_queries.ravel(), return_inverse=True)
                mel, sem, valid = song.audio.sample(song.clock.at_floats(unique))
                for start, rows in zip(chunk, inverse.reshape(beat_queries.shape)):
                    # Keep the original reduction order and validity per window.
                    values = sem if np.any(valid[rows]) else mel
                    vec = values[rows].mean(0)
                    descriptors[start] = vec / max(float(np.linalg.norm(vec)), 1e-8)
        for start in starts:
            if start not in cache:
                if len(cache) >= 128:
                    cache.pop(next(iter(cache)))
                cache[start] = descriptors[start]
        query = descriptors[cursor]
        return max(candidates, key=lambda beat: float(np.dot(query, descriptors[beat])))
    def local_anchor(self, beat):
        width = getattr(self, 'local_audio_block_beats', 1)
        return (math.floor(beat) // width) * width

    def beat_features(self, song, anchors):
        result = {}
        if getattr(self, 'onset_readout_policy', 'off') != 'off':
            from .onset_readout import audio_tiles
            result.update(audio_tiles(song, anchors, self.onset_phases))
        if self._paired_music():
            from .music_evidence import entry_preview
            result['entry_preview_features'] = torch.as_tensor(entry_preview(self._profile(song), anchors))
        if getattr(self,'future_configuration_policy','off') in AUDIO_WRITING_POLICIES:
            result.update(self.writing_features(song,anchors))
        return result

    def writing_features(self,song,anchors):
        from .local_writing import audio_rows, inputs
        if not hasattr(self,'_writing_audio_cache'):
            self._writing_audio_cache=OrderedDict()
        key=song.stream_cache_key or _song_cache_key(song)
        if key not in self._writing_audio_cache:
            self._writing_audio_cache[key]=audio_rows(self._profile(song),song.clock)
            if len(self._writing_audio_cache)>8:self._writing_audio_cache.popitem(last=False)
        self._writing_audio_cache.move_to_end(key)
        return inputs(self._writing_audio_cache[key],song,anchors)

    def tile(self,song,anchor,*,include_beat_features=True):
        if not hasattr(self,'tile_cache'):self.tile_cache=OrderedDict()
        key=(str(song.audio_path),str(song.clock.beats),str(song.clock.bpms),str(song.clock.seconds),int(anchor),
             self.future_beats,include_beat_features)
        if key in self.tile_cache:self.tile_cache.move_to_end(key);return self.tile_cache[key]
        ticks=np.arange(-48,(self.future_beats+2)*24,dtype=np.int64)
        beats=(int(anchor)*24+ticks)/24.;seconds=song.clock.at_floats(beats)
        mel,sem,valid=song.audio.sample(seconds)
        result=dict(local_audio=torch.as_tensor(mel,dtype=torch.float32),
            local_semantic=torch.as_tensor(sem,dtype=torch.float32),local_semantic_mask=torch.as_tensor(valid,dtype=torch.bool),
            local_mask=torch.as_tensor((seconds>=0)&(seconds<=song.audio.duration_sec),dtype=torch.bool),
            local_beats=torch.tensor(beats,dtype=torch.float32))
        if self._structured():
            from .rhythm_structure import onset_envelope
            result['local_envelope']=torch.as_tensor(onset_envelope(song.audio,song.clock,beats))
        if include_beat_features:
            result.update({k: v[0] for k, v in self.beat_features(song, [anchor]).items()})
        self.tile_cache[key]=result
        while len(self.tile_cache)>64:self.tile_cache.popitem(last=False)
        return result
    def build_stream(self,song,points,cursor,*,target=(),ended=False,training=False,
                     reference_start=_SELECT_REFERENCE,_prefix_end=None,context_recent_beats=None,
                     _reference_index=None,history_exclude=()):
        cursor=Fraction(cursor);boundary=max(Fraction(0),cursor-self.history_beats)
        if _prefix_end is None:
            known = list(points)
            if any(p.beat > cursor for p in known):
                raise ValueError('Uncommitted future chart in continuous prefix')
            history = [p for p in known if p.beat >= boundary]
        else:
            # Dataset windows use the sorted source with an exclusive prefix index.
            known = points
            history_index = bisect.bisect_left(song.point_beats, boundary, hi=_prefix_end)
            history = known[history_index:_prefix_end]
        # Training can reuse the immutable native index, with strict causal filtering.
        state=song.sustain if training else None
        if not training and any('[' in p.note_text for p in known):
            from maichart.generation.sustain_state import SustainStateIndex
            from maichart.event_timing import EventTempo
            state=SustainStateIndex(known,[EventTempo(b,t) for b,t in zip(song.clock.beats,song.clock.bpms)],song.clock.seconds[0])
        def active_sources(at):
            if state is None:return set()
            second=song.clock.at(at)
            return {Fraction(m['source_beat']) for m in state._candidate_markers(second)
                if m['source_beat']<=cursor and float(m['end_second'])>second}
        older=active_sources(boundary)
        active=active_sources(cursor)
        if history_exclude:
            history=[p for p in history if p.beat not in history_exclude or p.beat in active]
        carry = []
        if _prefix_end is None:
            carry = [p for p in known if p.beat < boundary and p.beat in older]
        else:
            beats = song.point_beats
            for beat in sorted(older):
                first = bisect.bisect_left(beats, beat, hi=history_index)
                end = bisect.bisect_right(beats, beat, lo=first, hi=history_index)
                carry.extend(known[first:end])
        codec = self.codec
        budget = self.max_tokens // 2
        carry_ids = codec.encode_points(carry)
        shortened_ids = None
        if training and context_recent_beats is not None:
            shortened = musical_context.shorten_history(history, cursor, context_recent_beats, active)
            if len(shortened) < len(history):
                candidate = codec.encode_points(shortened)
                if 5 + len(carry_ids) + len(candidate) <= budget:
                    shortened_ids = candidate
        if shortened_ids is None and _reference_index is not None:
            reference_start = self._reference_argument(_reference_index)
        structured=self._structured()
        zero_unmarked = (song.quality or {}).get('unmarked_weight') == 0
        if shortened_ids is not None or (zero_unmarked and getattr(self, 'reviewed_reference_policy', 'approved_only') == 'off'):
            reference_start=None
        aligned=None
        if structured:
            aligned=self._selected_reference(song,cursor) if reference_start is _SELECT_REFERENCE else reference_start
            reference_start=None
        legacy_aligned=None
        if not structured and reference_start is _SELECT_REFERENCE:
            if getattr(self,'reference_selection','model')=='aligned':
                reference_start = self._aligned_legacy_start(song, cursor)
                legacy_aligned = self._last_aligned
            else:
                reference_start = self._reference_stream(song, known, cursor, prefix_end=_prefix_end)
        reference = []
        if reference_start is not None:
            if _prefix_end is None:
                stop = min(reference_start + self.window_beats, boundary)
                reference = [p for p in known if reference_start <= p.beat < stop]
            else:
                first = bisect.bisect_left(beats, reference_start, hi=history_index)
                end = bisect.bisect_left(beats, reference_start + self.window_beats, lo=first, hi=history_index)
                reference = known[first:end]
        evidence_v2=self._evidence_v2()
        checked_start=math.floor(cursor)-4
        checked_end=math.floor(cursor)+self.window_beats
        if aligned is not None:
            from .rhythm_structure import reference_points,REF_LEAD_BEATS
            shift,score=aligned
            ahead=2*getattr(self,'refresh_events',16)
            if _prefix_end is None:
                reference=reference_points(known,cursor,shift,display_ahead=min(self.window_beats,max(0,self._profile(song).beats-math.floor(cursor))) if evidence_v2 else self.window_beats,max_ahead_events=ahead)
            else:
                reference=reference_points(known,cursor,shift,display_ahead=min(self.window_beats,max(0,self._profile(song).beats-math.floor(cursor))) if evidence_v2 else self.window_beats,max_ahead_events=ahead,
                                           beats=song.point_beats,end=_prefix_end)
            # Low-trust examples teach rejection, without destroying the true
            # history needed to learn it. Strong references retain the v1 cap.
            from .rhythm_structure import REF_MIN_SCORE
            reference_budget=budget//3
            if evidence_v2 and score<REF_MIN_SCORE:
                reference_budget=min(reference_budget,max(0,budget-5-
                    len(codec.encode_points(carry))-len(codec.encode_points(history))))
            lengths=[len(codec.encode_points([p])) for p in reference]
            total=sum(lengths);left=0;right=len(reference)
            while total>reference_budget and left<right:
                if reference[right-1].beat>cursor:
                    right-=1;total-=lengths[right]
                else:
                    total-=lengths[left];left+=1
            reference=reference[left:right]
            reference_start=Fraction(math.floor(cursor))-REF_LEAD_BEATS-shift
            if evidence_v2 and not reference and lengths:aligned=None;reference_start=None
        check_ref_quality = (zero_unmarked or getattr(self, 'auto_quality_policy', 'off') == 'auto_v1'
            or (getattr(self,'reference_quality_policy','event_mask_v1') != 'legacy_v1'
                and bool((song.quality or {}).get('regions'))))
        if check_ref_quality and not self._reference_points_approved(song, reference, beat_offset=aligned[0] if aligned is not None else 0):
            reference = []
            aligned = None
            reference_start = None
            legacy_aligned = None
        history_ids = codec.encode_points(history) if shortened_ids is None else shortened_ids
        reference_ids = codec.encode_points(reference)
        prefix_length = 5 + len(carry_ids) + len(history_ids) + len(reference_ids)
        if prefix_length > budget:
            if not structured:
                prefix_length -= len(reference_ids)
                reference_ids = []
            kept_start = 0
            carried_beats = {point.beat for point in carry}
            while prefix_length > budget and kept_start < len(history):
                removed = history[kept_start]
                kept_start += 1
                if removed.beat in active and removed.beat not in carried_beats:
                    carry.append(removed)
                    carried_beats.add(removed.beat)
                else:
                    prefix_length -= len(codec.encode_points([removed]))
            if structured and prefix_length > budget and aligned is not None:
                # A trusted reference outranks the oldest history, not mandatory carry.
                prefix_length -= len(reference_ids)
                reference_ids = [];aligned=None;reference_start=None
            if prefix_length > budget:
                raise ValueError('Mandatory sustained context exceeds token budget; increase max_tokens')
            if kept_start:
                carry_ids = codec.encode_points(carry)
                history_ids = codec.encode_points(history[kept_start:])
        prefix = ([codec.bos_id, codec.carry_id] + carry_ids + [codec.ctx_id] + history_ids
                  + [codec.ref_id] + reference_ids + [codec.tgt_id])
        suffix=self.codec.encode_points(target)
        if ended:suffix.append(self.codec.eos_id)
        full=prefix+suffix
        if training and len(full)-1>self.max_tokens:raise ValueError('Complete event slice exceeds token budget')
        ids=full[:-1] if training else full
        full_clocks = self.codec.token_clocks(full, cursor=cursor)
        clocks = full_clocks[:-1] if training else full_clocks
        labels=np.full(len(ids),-100,dtype=np.int64)
        if training:labels[len(prefix)-1:]=full[len(prefix):]
        active=(labels!=-100).astype(np.float32)
        unmarked=float((song.quality or {}).get('unmarked_weight',1.))
        weights=active*unmarked
        regions = (song.quality or {}).get('regions', [])
        if training and regions:
            from .quality_data import stream_region_weight_fn
            weight_fn = stream_region_weight_fn(unmarked, getattr(song, '_legacy_zero_unmarked_resume', False))
            # Causal query clocks intentionally stay at the preceding event
            # while its successor's time is written. Supervision ownership is
            # different: every token of a complete event owns its final time.
            ownership = np.asarray(full_clocks, dtype=np.float64).copy()
            event_start = None
            for position, token in enumerate(full):
                if token == codec.at_id:
                    event_start = position
                elif token == codec.event_end_id and event_start is not None:
                    ownership[event_start:position + 1] = full_clocks[position]
                    event_start = None
            seconds = song.clock.at_floats(ownership[1:])
            timing_ids = {codec.at_id, codec.phase_id, codec.eos_id, *codec.digits, *codec.phases}
            timing = np.asarray([int(token) in timing_ids for token in full[1:]])
            for region in regions:
                selected = (seconds >= float(region['start_sec'])) & (seconds < float(region['end_sec']))
                timing_weight, component_weight = weight_fn(region)
                weights[selected & timing] = active[selected & timing] * timing_weight
                weights[selected & ~timing] = active[selected & ~timing] * component_weight
            if unmarked == 0:
                # Keep contiguous approved timing across layout-only boundaries.
                stamps=getattr(song,'reviewed_event_seconds',None)
                if stamps is None:
                    stamps=song.clock.at_floats(np.asarray([float(p.beat) for p in song.points]))
                    song.reviewed_event_seconds=stamps
                spans=[]
                for region in regions:
                    if weight_fn(region)[0]<=0:continue
                    a,b=float(region['start_sec']),float(region['end_sec'])
                    if spans and a<=spans[-1][1]:spans[-1][1]=max(spans[-1][1],b)
                    else:spans.append([a,b])
                previous=np.searchsorted(stamps,seconds,side='left')-1
                prior=np.where(previous>=0,stamps[np.maximum(previous,0)],0.) if len(stamps) else np.zeros_like(seconds)
                trusted=np.zeros(len(seconds),dtype=bool)
                for a,b in spans:trusted|=(seconds>=a)&(seconds<b)&(prior>=a)
                # The integer jump may cross unreviewed material, while this
                # approved onset's absolute sub-beat phase is still trustworthy.
                coarse=np.isin(labels,[codec.at_id,codec.eos_id,*codec.digits])
                # A specifically approved incoming transition owns only the
                # first original event at/after its saved review boundary.
                for region in regions:
                    if 'entry_start_sec' not in region or weight_fn(region)[0]<=0:continue
                    entry=int(np.searchsorted(stamps,float(region['entry_start_sec']),side='left'))
                    if entry<len(stamps) and float(region['start_sec'])<=stamps[entry]<float(region['end_sec']):
                        trusted|=np.abs(seconds-stamps[entry])<1e-8
                weights[coarse & ~trusted]=0
                weights[labels==codec.eos_id]=0
        anchors=sorted({math.floor(t) for t in clocks});lookup={a:i for i,a in enumerate(anchors)}
        local_anchors = sorted({self.local_anchor(a) for a in anchors})
        local_lookup = {anchor: index for index, anchor in enumerate(local_anchors)}
        # Adjacent query tiles overlap heavily. Read/project each source frame
        # once; gather the small learned vectors into independent encoder tiles.
        relative=np.arange(-48,(self.future_beats+2)*24,dtype=np.int64)
        all_ticks=np.asarray(local_anchors)[:,None]*24+relative[None,:]
        ticks=np.unique(all_ticks);tile_frames=np.searchsorted(ticks,all_ticks)
        seconds=song.clock.at_floats(ticks/24.)
        envelope=None
        if structured:
            from .rhythm_structure import onset_envelope
            envelope=onset_envelope(song.audio,song.clock,ticks/24.)
        sample_frames = getattr(song.audio, 'sample_for_training', None) if training else None
        if sample_frames is not None:
            mel, semantic, semantic_right, semantic_weight, valid = sample_frames(seconds)
        else:
            mel,semantic,valid=song.audio.sample(seconds)
        summary=song.summary
        out=dict(input_ids=torch.tensor(ids),labels=torch.tensor(labels),token_weights=torch.tensor(weights),
            token_beats=torch.tensor(clocks,dtype=torch.float32),token_mask=torch.ones(len(ids),dtype=torch.bool),
            tile_ids=[lookup[math.floor(t)] for t in clocks],anchors=anchors,
            local_anchors=local_anchors, local_tile_ids=[local_lookup[self.local_anchor(t)] for t in clocks],
            bank_audio=torch.as_tensor(mel,dtype=torch.float32),bank_semantic=torch.as_tensor(semantic),
            bank_semantic_mask=torch.as_tensor(valid,dtype=torch.bool),
            tile_frames=torch.tensor(tile_frames,dtype=torch.long),
            local_beats=torch.tensor(all_ticks/24.,dtype=torch.float32),
            local_mask=torch.as_tensor(((seconds>=0)&(seconds<=song.audio.duration_sec))[tile_frames],dtype=torch.bool),
            condition=torch.as_tensor(song.condition),condition_known=torch.as_tensor(song.condition_known if song.condition_known is not None else [True]+[False]*15),
            sample_weight=torch.tensor(float((song.quality or {}).get('weight',1.))),
            song_audio=torch.as_tensor(summary['mel']),song_semantic=torch.as_tensor(summary['semantic']),
            song_semantic_mask=torch.as_tensor(summary['semantic_valid']),song_beats=torch.as_tensor(summary['beats'],dtype=torch.float32),
            song_mask=torch.ones(len(summary['times_sec']),dtype=torch.bool),start_beat=Fraction(0),cursor=cursor,
            song_id=song.song_id,reference_start=reference_start,pad_id=self.codec.pad_id,
            song_cache_key=song.stream_cache_key or _song_cache_key(song),
            curve_teacher_allowed=shortened_ids is None)
        if shortened_ids is not None:
            for field in ('condition', 'condition_known'):
                out[field] = out[field].clone()
                out[field][1:] = 0
        if 'detail' in summary:
            out['song_detail'] = torch.as_tensor(summary['detail'], dtype=torch.float32)
        if evidence_v2:
            from .rhythm_structure import local_evidence
            evidence=local_evidence(self._profile(song),clocks,aligned[0] if aligned else None,
                                    start=checked_start,end=checked_end)
            evidence[:prefix.index(codec.ref_id)]=0
            out['reference_evidence']=torch.as_tensor(evidence)
            out['reference_checked_start']=checked_start
            out['reference_checked_end']=min(checked_end,self._profile(song).beats)
            if training and getattr(self,'reference_augmentation',False):
                from .rhythm_learning import supervision_weights,POLICY as LEGACY_LEARNING
                if getattr(self,'rhythm_learning_policy',LEGACY_LEARNING)==LEGACY_LEARNING:
                    out['token_weights']=torch.as_tensor(weights*supervision_weights(codec,ids,labels,target,cursor,weights,
                        context_phases=[p.beat%1 for p in history if p.beat>=cursor-2]))
        if sample_frames is not None:
            out['bank_semantic_right'] = torch.as_tensor(semantic_right)
            out['bank_semantic_weight'] = torch.as_tensor(semantic_weight)
        if structured:
            from .rhythm_structure import trust_features
            out['bank_envelope']=torch.as_tensor(envelope)
            out['reference_trust']=torch.as_tensor(trust_features(aligned[1] if aligned is not None else None))
            out['reference_shift']=aligned[0] if aligned is not None else None
            out['reference_score']=aligned[1] if aligned is not None else None
        elif getattr(self,'reference_selection','model')=='aligned':
            last=legacy_aligned if reference_start is not None else None
            out['reference_shift']=last[0] if last else None
            out['reference_score']=last[1] if last else None
        if getattr(self, 'onset_readout_policy', 'off') != 'off':
            from .onset_readout import audio_tiles, query_rows
            out.update(audio_tiles(song, anchors, self.onset_phases))
            out['phase_queries_cpu'] = query_rows(ids, out['tile_ids'], self.codec.ids['[PHASE]'])
        if getattr(self,'intensity_policy','off')!='off':
            from .intensity import add_item
            add_item(out,song,clocks,training=training)
        if self._paired_music():
            from .music_evidence import paired_music_features,entry_preview
            profile=self._profile(song)
            paired=paired_music_features(profile,clocks,aligned[0] if aligned else None,
                                        start=checked_start,end=checked_end)
            paired[:prefix.index(codec.ref_id)]=0
            out['paired_music_features']=torch.as_tensor(paired)
            out['entry_preview_features']=torch.as_tensor(entry_preview(profile,anchors))
        if getattr(self,'future_configuration_policy','off') in AUDIO_WRITING_POLICIES:
            out.update(self.writing_features(song,anchors))
        return out
    def __getitem__(self,index):
        source = self.recovery_sources.get(index)
        if source is not None:
            return self._recovery_item(source)
        si,begin=self.windows[index];song=self.songs[si];end=self.ranges[index]
        cursor=song.points[begin-1].beat if begin else Fraction(0)
        recent_beats = None
        if begin and getattr(self,'musical_context_policy','off') == musical_context.POLICY:
            recent_beats = musical_context.recent_history_beats(seed=self.augmentation_seed,
                epoch=self.training_epoch, index=index, probability=self.musical_context_probability,
                history_beats=self.history_beats)
        if self.future_configuration_policy in MOTION_WRITING_POLICIES and (song.quality or {}).get('unmarked_weight') == 0:
            recent_beats = None
        if song.stream_cache_key is None:
            song.stream_cache_key = _song_cache_key(song)
        item = self.build_stream(song, song.points, cursor, target=song.points[begin:end],
                                 ended=end==len(song.points), training=True,
                                 _reference_index=index,_prefix_end=begin,
                                 context_recent_beats=recent_beats)
        scores = getattr(self, '_relation_scores', None)
        if scores is not None and scores[index] > 0:
            fraction = self.musical_relation_fraction
            extra = float(scores[index]) * fraction / (1. - fraction)
            item['token_weights'] *= 1. + extra
            item['musical_relation_extra_fraction'] = extra / (1. + extra)
        if self.future_configuration_policy in AUDIO_WRITING_POLICIES:
            self._writing_targets(item,self._configuration(si),song,begin,end,
                                  offset=((index*17+self.training_epoch*37)%101+.5)/101)
        elif self._use_future_configuration(index):
            ordered_configuration.add_item(item, self._configuration(si), self.codec, begin, end)
            if 'future_configuration_positions' in item:
                for field in ('condition', 'condition_known'):
                    item[field] = item[field].clone()
                    item[field][1:] = 0
                item['curve_teacher_allowed'] = False
        return item

    def _use_future_configuration(self, index):
        if getattr(self, 'future_configuration_policy', 'off') == 'off':
            return False
        if self.auxiliary_validation:
            return True
        rng = np.random.default_rng((self.augmentation_seed, self.training_epoch, index, 31))
        return float(rng.random()) < ordered_configuration.TRAINING_PROBABILITY

    def _writing_targets(self,item,configuration,song,begin,end,*,offset=.5,layout=True):
        if self.future_configuration_policy not in AUDIO_WRITING_POLICIES:
            return
        from .local_writing import add_targets
        if self.future_configuration_policy in GRID_WRITING_POLICIES and begin<end:
            item['writing_stop_beat']=song.points[end-1].beat
        if self.future_configuration_policy == BOUNDARY_WRITING:
            from .phrase_decisions import add_decisions
            add_decisions(item,configuration,song,self.codec,begin,end,layout=layout)
        add_targets(item,configuration,song,self.future_configuration_policy,dataset=self,offset=offset)

    def _prepare_reference(self,index):
        si,begin=self.windows[index];song=self.songs[si]
        cursor=song.points[begin-1].beat if begin else Fraction(0)
        if self._structured():
            if not self._reference_ready[index]:
                seed_index = getattr(self, '_recovery_seed_indices', {}).get(index, index)
                draw=float(np.random.default_rng((self.augmentation_seed,seed_index,11)).random())
                self._reference_starts[index]=self._aligned_match(song,cursor,alternative_draw=draw)
                self._reference_ready[index]=1
            return self._reference_starts[index]
        if getattr(self,'reference_selection','model')=='aligned':
            if not self._reference_ready[index]:
                self._reference_starts[index]=self._aligned_legacy_start(song,cursor)
                self._reference_ready[index]=1
            return self._reference_starts[index]
        if not self._reference_ready[index]:
            self._reference_starts[index] = self._reference_stream(song, song.points, cursor, prefix_end=begin)
            self._reference_ready[index] = 1
            self._remaining_references[si] -= 1
            if self._remaining_references[si] and song.descriptor_cache:
                self._reference_descriptors[si] = song.descriptor_cache
                self._reference_descriptors.move_to_end(si)
                if len(self._reference_descriptors) > _REFERENCE_CACHE_SONGS:
                    _, evicted = self._reference_descriptors.popitem(last=False)
                    evicted.clear()
            else:
                self._reference_descriptors.pop(si, None)
                song.descriptor_cache.clear()
        return self._reference_starts[index]

    def _recovery_item(self, source):
        si,begin=self.windows[source];song=self.songs[si];end=self.ranges[source]
        cursor=song.points[begin-1].beat
        seed_index = self._recovery_seed_indices.get(source, source)
        seed=self.augmentation_seed+seed_index*1009+self.training_epoch*1000003
        if (self.timing_recovery_policy==timing_recovery.GESTURE_POLICY
                and (seed_index//16+self.training_epoch//2)%2==0):
            from .gesture_pairs import training_item
            pair=training_item(self,song,self._configuration(si),seed)
            if pair is not None:
                return pair
        # Alternate independently of dense/sparse timing practice below, so
        # both original timing views keep their share of the remaining slots.
        if (self.timing_recovery_policy==timing_recovery.LAYOUT_POLICY
                and (seed_index//16+self.training_epoch//2)%2==0):
            replay=phrase_replay.training_item(self,song,self._configuration(si),begin,end,seed)
            if replay is not None:
                return replay
        changed=timing_recovery.perturb_prefix(song.points[:begin],self._recovery_phases,
            seed,dense=(seed_index//8+self.training_epoch)%2==0,span=self._recovery_span)
        if changed is None:
            raise RuntimeError('Timing recovery view lost its eligible committed prefix')
        prefix,_=changed
        if self._evidence_v2() and (seed_index//8+self.training_epoch)%3==0:
            phase=timing_recovery.perturb_phase_prefix(song.points[:begin],self._recovery_phases,
                seed,next_beat=song.points[begin].beat,span=self._recovery_span)
            if phase is not None:prefix=phase
        cursor=prefix[-1].beat
        item=self.build_stream(song,prefix,cursor,target=song.points[begin:end],
            ended=end==len(song.points),training=True,reference_start=self._reference_argument(source))
        keep=torch.tensor([int(token) in self._timing_ids for token in item['labels']],dtype=torch.bool)
        item['labels']=item['labels'].masked_fill(~keep,-100)
        item['token_weights']=item['token_weights']*keep*2.
        if self.future_configuration_policy in MOTION_WRITING_POLICIES:
            item['curve_teacher_allowed'] = False
        if self.future_configuration_policy in AUDIO_WRITING_POLICIES:
            self._writing_targets(item,self._configuration(si),song,begin,end,layout=False)
        return item


def collate_stream(items):
    # The existing collator handles sequence fields; tiles have a distinct batch axis.
    from . import gesture_pairs
    if any('gesture_peer' in item for item in items):
        items=gesture_pairs.expand(items)
    temporary=[{**item,'local_audio':item['bank_audio'][:1],
        'local_semantic':item['bank_semantic'][:1],'local_semantic_mask':item['bank_semantic_mask'][:1],
        'local_mask':item['local_mask'][0,:1],'local_beats':item['local_beats'][0,:1]} for item in items]
    out=collate_composer_v2(temporary)
    ordered_configuration.collate(items, out)
    from .local_writing import collate as collate_writing
    collate_writing(items,out)
    if any('musical_relation_extra_fraction' in item for item in items):
        out['musical_relation_extra_fraction'] = torch.tensor([
            item.get('musical_relation_extra_fraction', 0.) for item in items], dtype=torch.float32)
    if any(not item.get('curve_teacher_allowed', True) for item in items):
        out['curve_teacher_allowed'] = torch.tensor([item.get('curve_teacher_allowed',True) for item in items],
                                                    dtype=torch.bool)
    if 'curve_left' in items[0]:
        from .intensity import collate
        collate(items,out)
    if 'curve_loss_scale' in items[0]:
        out['curve_loss_scale']=torch.stack([item['curve_loss_scale'] for item in items])
    for k in ('bank_audio','bank_semantic','bank_semantic_mask','local_mask','local_beats'):out[k]=torch.cat([item[k] for item in items],0)
    if 'bank_envelope' in items[0]:out['bank_envelope']=torch.cat([item['bank_envelope'] for item in items],0)
    if 'reference_trust' in items[0]:out['reference_trust']=torch.stack([item['reference_trust'] for item in items])
    if 'reference_evidence' in items[0]:
        from torch.nn import functional as F
        width=out['input_ids'].size(1)
        out['reference_evidence']=torch.stack([F.pad(item['reference_evidence'],(0,0,0,width-len(item['input_ids']))) for item in items])
    if 'paired_music_features' in items[0]:
        out['paired_music_features']=torch.nn.utils.rnn.pad_sequence([x['paired_music_features'] for x in items],batch_first=True)
        out['entry_preview_features']=torch.cat([x['entry_preview_features'] for x in items])
    if any('bank_semantic_right' in item for item in items):
        out['bank_semantic_right'] = torch.cat([item.get('bank_semantic_right', item['bank_semantic']) for item in items])
        out['bank_semantic_weight'] = torch.cat([item['bank_semantic_weight']
            if 'bank_semantic_weight' in item else torch.zeros(len(item['bank_semantic']), dtype=torch.float64)
            for item in items])
    owners=[];ids=[];offset=0;bank_owners=[];frames=[];bank_offset=0
    local_ids=[];local_offset=0
    for row,item in enumerate(items):
        padding = [-1]*(out['input_ids'].size(1)-len(item['tile_ids']))
        ids.append([i+offset for i in item['tile_ids']] + padding)
        offset+=len(item['anchors'])
        owners.extend([row]*len(item['tile_frames']))
        local_ids.append([i+local_offset for i in item.get('local_tile_ids', item['tile_ids'])] + padding)
        local_offset+=len(item['tile_frames'])
        bank_owners.extend([row]*len(item['bank_audio']));frames.append(item['tile_frames']+bank_offset);bank_offset+=len(item['bank_audio'])
    out['tile_owner']=torch.tensor(owners);out['tile_ids_cpu']=ids
    if any(item.get('local_anchors', item['anchors']) != item['anchors'] for item in items):
        out['local_tile_ids_cpu']=local_ids
    out['bank_owner']=torch.tensor(bank_owners);out['tile_frames']=torch.cat(frames)
    out['cursor']=[item['cursor'] for item in items]
    if 'onset_audio' in items[0]:
        from .onset_readout import collate_tiles
        out.update(collate_tiles(items))
        queries=[];offset=0
        for row,item in enumerate(items):
            queries.extend((row,p,tile+offset) for _,p,tile in item['phase_queries_cpu'])
            offset+=len(item['anchors'])
        out['phase_queries_cpu']=queries
    phrase_replay.collate(items,out)
    gesture_pairs.collate(items,out)
    return out
