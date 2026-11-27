"""Causal component windows, sharing the established manifest and audio clocks.

No legacy FormDataset, draft targets, load curves or latent labels are built.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from maichart.event_timing import EventPoint, EventTempo, _parse_event_metadata
from maichart.maidata import decode_text_with_encoding, parse_maidata_metadata
from maichart.tempo_clock import TempoClock
from maichart.training.dataset_catalog import _source_asset
from maichart.training.event_dataset import inspect_chart, numeric_level
from . import chart_dialect
from .chart_cache import ChartCache
from .codec import VocabularyCoverageError
from .conditions import RADAR_STYLE, condition_contract, resolve_condition, training_condition
from .difficulty import DIFFICULTY_SLOTS, metadata_exclusion
from .quality_data import TrainingQuality, quality_excludes_chart


def _beat(value):
    return value if isinstance(value, Fraction) else Fraction(str(value))


def _beats_at_seconds(clock, seconds):
    seconds = np.asarray(seconds, dtype=np.float64)
    ix = np.clip(np.searchsorted(clock._seconds_arr, seconds, side='right') - 1,
                 0, len(clock.beats) - 1)
    return clock._beat_floats[ix] + (seconds - clock._seconds_arr[ix]) / clock._scale_arr[ix]


def _report_preparation(stage, completed, total, started, **details):
    message = dict(stage=stage, completed=completed, total=total,
                   elapsed_sec=round(time.perf_counter() - started, 3), **details)
    print('composer_v2_prepare ' + json.dumps(message, ensure_ascii=False), flush=True)


@dataclass
class ComposerV2Song:
    song_id: str
    source_path: Path
    audio_path: Path
    difficulty: int
    level: float
    points: list
    clock: TempoClock
    end_beat: Fraction
    condition: np.ndarray
    quality: dict | None = None
    audio: object | None = None
    sustain: object | None = None
    summary: dict | None = None
    point_beats: list = field(default_factory=list)
    reference_cache: dict = field(default_factory=dict)
    authored_end_beat: Fraction | None = None
    descriptor_cache: dict = field(default_factory=dict)
    condition_known: np.ndarray | None = None
    chart_sha256: str | None = None
    event_token_lengths: list[int] = field(default_factory=list)
    stream_cache_key: str | None = None
    learning_role: str = 'full'


class ComposerV2Dataset(Dataset):
    max_condition_level = 15.

    def __init__(self, manifest, cache_dir, codec=None, window_beats=16,
                 history_beats=16, future_beats=4, ticks_per_beat=48,
                 max_tokens=2048, audio_backend='mert', mert_model=None,
                 prepare_device='cpu', limit_charts=0, prepare=True,
                 *, audio_store=None, min_level=12., max_level=15.,
                 min_difficulty_index=4, max_difficulty_index=6,
                 quality_enabled=True, condition_policy=RADAR_STYLE,sequence_policy='window_v1',
                 extend_vocabulary=False, dialect_policy=chart_dialect.POLICY,
                 auto_quality_policy='off', difficulty_scorer=None, min_declared_level=0.,
                 difficulty_slots=None, validate_fixed_vocabulary=False, chart_subset=None, full_sources_only=False):
        if window_beats<1 or history_beats<0 or future_beats<0 or any(int(v)!=v for v in (window_beats,history_beats,future_beats)):
            raise ValueError('Window lengths must be integral beats; target positive, history/future nonnegative')
        if int(max_tokens) < 4:
            raise ValueError('max_tokens must be at least 4')
        if dialect_policy not in (*chart_dialect.NORMALIZED_POLICIES, chart_dialect.LEGACY_STRICT_POLICY):
            raise ValueError(f'Unknown Composer v2 chart dialect policy: {dialect_policy}')
        if auto_quality_policy not in ('off', 'auto_v1'):
            raise ValueError(f'Unknown auto_quality_policy: {auto_quality_policy}')
        if auto_quality_policy == 'auto_v1' and (sequence_policy != 'continuous_events_v1' or not quality_enabled):
            raise ValueError('自动质检标注需要启用质量加权的 Composer v2 连续事件训练')
        self.auto_quality_policy = auto_quality_policy
        self.manifest = Path(manifest).resolve()
        self.condition_policy = condition_policy
        self.condition_contract = condition_contract(condition_policy)
        self.cache_dir = Path(cache_dir)
        self.window_beats, self.history_beats = window_beats, history_beats
        self.future_beats, self.max_tokens = future_beats, int(max_tokens)
        self.songs, self.windows = [], []
        self.audit = {'skipped': [], 'filtered': [], 'charts': [], 'windows': 0}
        if difficulty_scorer is not None:
            self.max_condition_level = max_level
        if difficulty_scorer is not None or min_declared_level or difficulty_slots not in (None, DIFFICULTY_SLOTS):
            selection_audit = dict(source='declared', min_level=min_level, max_level=max_level,
                                   min_declared_level=min_declared_level,
                                   difficulty_slots=list(difficulty_slots or DIFFICULTY_SLOTS))
            if difficulty_scorer is not None:
                selection_audit.update(source=getattr(difficulty_scorer, 'source', 'fitted'),
                                       scorer=difficulty_scorer.identity)
            self.audit['difficulty_selection'] = selection_audit
        supported_slots = getattr(difficulty_scorer,'supported_slots',None)
        self.audit['chart_dialect'] = {'policy': dialect_policy, 'changes': []}
        manifest_bytes = self.manifest.read_bytes()
        payload = json.loads(manifest_bytes.decode('utf-8-sig'))
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        quality_data = TrainingQuality(self.manifest, payload, enabled=quality_enabled,
                                       use_auto=auto_quality_policy == 'auto_v1')
        if auto_quality_policy == 'auto_v1':
            self.auto_annotations_sha256 = quality_data.auto_fingerprint
            self.audit['auto_quality'] = {
                'policy': auto_quality_policy,
                'path': str(quality_data.auto_path) if quality_data.auto_path is not None else None,
            }
        entries = payload.get('songs', [])
        chart_start = time.perf_counter()
        normalize_dialect = dialect_policy in chart_dialect.NORMALIZED_POLICIES
        chart_cache = ChartCache(cache_dir, dialect_policy if normalize_dialect else None)
        prepared_charts = {}
        reused_charts = cache_hits = 0
        _report_preparation('charts', 0, len(entries), chart_start, accepted=0)
        manifest_zero_unmarked = bool(payload.get('candidate_pool') or payload.get('reviewed_training'))
        if manifest_zero_unmarked and not quality_enabled:
            raise ValueError('人工确认片段必须启用质量加权；未确认部分仅作上下文，不能作为整谱训练')
        if manifest_zero_unmarked and sequence_policy != 'continuous_events_v1':
            raise ValueError('人工确认片段当前仅支持 Composer v2 连续事件训练')
        for examined, entry in enumerate(entries, 1):
            if (entry.get('reviewed_only') or entry.get('candidate_pool')) and not quality_enabled:
                raise ValueError('人工确认片段必须启用质量加权；未确认部分仅作上下文，不能作为整谱训练')
            if (entry.get('reviewed_only') or entry.get('candidate_pool')) and sequence_policy != 'continuous_events_v1':
                raise ValueError('人工确认片段当前仅支持 Composer v2 连续事件训练')
            if limit_charts and len(self.songs) >= limit_charts:
                break
            source = None
            metadata = None
            for difficulty in entry.get('difficulties', []):
                if limit_charts and len(self.songs) >= limit_charts:
                    break
                role = difficulty.get('learning_role',entry.get('learning_role','full'))
                if role not in ('full','mechanics'):
                    raise ValueError('未知的素材学习用途')
                if role=='mechanics':
                    if full_sources_only:
                        self.audit.setdefault('mechanics_validation_excluded',0)
                        self.audit['mechanics_validation_excluded']+=1
                        continue
                    if getattr(self,'hand_plan_policy','off')=='off':
                        raise ValueError('用于动作学习的素材需要开启手顺预判')
                slot = int(difficulty.get('difficulty_index', difficulty.get('index', 0)))
                if not min_difficulty_index <= slot <= max_difficulty_index:
                    continue
                declared = difficulty.get('level_raw', difficulty.get('level'))
                exclusion = metadata_exclusion(slot, declared, difficulty_slots, supported_slots, min_declared_level)
                if exclusion is not None:
                    if difficulty.get('has_chart', True):
                        self.audit['filtered'].append(dict(source=str(entry.get('maidata_path','')),
                            difficulty=slot,declared_level=declared,reason=exclusion))
                    continue
                try:
                    if difficulty_scorer is None:
                        level = numeric_level(declared)
                        if not min_level <= level <= max_level:
                            continue
                    if source is None:
                        chart_path = _source_asset(entry['maidata_path'], self.manifest, payload)
                        audio_path = _source_asset(entry['audio_path'], self.manifest, payload)
                        chart_bytes = chart_path.read_bytes()
                        chart_sha = hashlib.sha256(chart_bytes).hexdigest()
                        source = chart_path
                    chart_key = (chart_sha, slot)
                    if chart_subset is not None and chart_key not in chart_subset:
                        continue
                    prepared_chart = prepared_charts.get(chart_key)
                    if prepared_chart is None:
                        prepared_chart = chart_cache.load(*chart_key)
                        cache_hits += prepared_chart is not None
                    else:
                        reused_charts += 1
                    if prepared_chart is None:
                        if metadata is None:
                            text, _ = decode_text_with_encoding(chart_bytes)
                            metadata = parse_maidata_metadata(text)
                        chart = _parse_event_metadata(metadata, slot, strict_notes=False)
                        dialect_edits = {}
                        if normalize_dialect:
                            chart, dialect_edits = chart_dialect.normalize_training_chart(chart, dialect_policy)
                        inspect_chart(chart)
                    else:
                        chart, sustain, dialect_edits = prepared_chart
                    quality = quality_data.record_for(source, slot, audio_path)
                    if entry.get('reviewed_only') and (quality is None or quality.get('unmarked_weight') != 0):
                        raise ValueError('人工确认片段的范围标注缺失或已失效，请重新整理校准素材')
                    if quality_excludes_chart(quality):
                        self.audit['skipped'].append({'source': str(source), 'reason': 'quality_excluded'})
                        continue
                    if difficulty_scorer is not None:
                        level = difficulty_scorer.score(chart_sha, slot, source, chart_bytes)
                        if not min_level <= level <= max_level:
                            self.audit['filtered'].append(dict(source=str(source), difficulty=slot,
                                declared_level=declared, fitted_level=level, reason='fitted_level_out_of_range'))
                            continue
                    condition, known = resolve_condition(level, policy=condition_policy,
                                                         max_level=self.max_condition_level)
                    song = ComposerV2Song(str(entry.get('song_id', source.parent.name)),
                        source, audio_path, slot, level, list(chart.points),
                        TempoClock(chart.tempo_changes, chart.audio_first),
                        _beat(chart.end_beat), condition, quality)
                    song.learning_role = role
                    song.condition_known = known
                    song.point_beats = [p.beat for p in song.points]
                    song.authored_end_beat = song.end_beat
                    song.chart_sha256 = chart_sha
                    if prepared_chart is None:
                        sustain = None
                        if any('[' in p.note_text for p in song.points):
                            from maichart.generation.sustain_state import SustainStateIndex
                            sustain = SustainStateIndex(song.points, chart.tempo_changes, chart.audio_first)
                        chart_cache.save(*chart_key, chart, sustain, dialect_edits)
                    prepared_charts[chart_key] = (chart, sustain, dialect_edits)
                    song.sustain = sustain
                    self.songs.append(song)
                    self.audit['charts'].append({'source': str(source), 'difficulty': slot,
                        'chart_sha256': chart_sha,
                        'first': chart.audio_first, 'quality': quality})
                    if role!='full':
                        self.audit['charts'][-1]['learning_role']=role
                    if difficulty_scorer is not None:
                        self.audit['charts'][-1].update(declared_level=declared, fitted_level=level)
                    if any(dialect_edits.values()):
                        self.audit['chart_dialect']['changes'].append({
                            'source': str(source), 'difficulty': slot, **dialect_edits})
                except (OSError, ValueError, KeyError) as error:
                    rejected = {'song': entry.get('song_id'), 'difficulty': slot, 'reason': str(error)}
                    self.audit['skipped'].append(rejected)
                    print('composer_v2_chart_rejected ' + json.dumps(rejected, ensure_ascii=False), flush=True)
            if examined % 50 == 0 or examined == len(entries):
                _report_preparation('charts', examined, len(entries), chart_start,
                                    accepted=len(self.songs), cache_hits=cache_hits)
        self.audit['preparation'] = {'charts_sec': time.perf_counter() - chart_start,
                                     'duplicate_chart_parses_saved': reused_charts,
                                     'chart_cache_hits': cache_hits}
        del prepared_charts, manifest_bytes, payload, entries, quality_data
        if not self.songs:
            if full_sources_only and self.audit.get('mechanics_validation_excluded'):
                raise ValueError('验证集需要有通过筛选的完整学习谱面，仅学动作的谱面不用于选择 best')
            if 'difficulty_selection' in self.audit:
                raise ValueError('当前难度筛选条件下没有可用谱面。'
                                 f'已筛除 {len(self.audit["filtered"])} 张。解析或估分失败示例：{self.audit["skipped"][:4]}')
            raise ValueError(f'No usable composer_v2 charts: {self.audit["skipped"][:4]}')
        continuous = sequence_policy == 'continuous_events_v1'
        vocabulary_start = time.perf_counter()
        _report_preparation('vocabulary', 0, len(self.songs), vocabulary_start)
        # Auxiliary sources cannot silently expand the writer's output grammar.
        writing_sources=[s for s in self.songs if s.learning_role=='full']
        if not writing_sources:
            raise ValueError('请至少选择一个通过筛选的完整学习谱面')
        if codec is None:
            from .codec import ComponentCodec
            if continuous:
                from .stream_codec import StreamCodec
                codec=StreamCodec.fit((p.note_text for s in writing_sources for p in s.points),
                    ticks_per_beat=ticks_per_beat,phases=(p.beat for s in writing_sources for p in s.points))
            else:
                codec = ComponentCodec.fit((p.note_text for s in writing_sources for p in s.points),
                    ticks_per_beat=ticks_per_beat, window_beats=window_beats)
                # Train-observed phases are available at each window-relative integer.
                phases = {p.beat % 1 for song in writing_sources for p in song.points}
                relative_times = {Fraction(integer) + phase
                                  for phase in phases for integer in range(-history_beats, window_beats)}
                codec.configure_times(relative_times)
            self.audit['vocabulary_policy'] = 'fit_training_components_and_phases_once'
        elif extend_vocabulary:
            parent_codec = codec
            phases = {p.beat % 1 for song in writing_sources for p in song.points} if continuous else ()
            codec = parent_codec.extended((p.note_text for song in writing_sources for p in song.points),
                                          extra_tokens=('F:' + str(phase) for phase in phases))
            self.audit['vocabulary_policy'] = 'extend_checkpoint_for_finetune'
            added_tokens = codec.tokens[parent_codec.vocab_size:]
            self.audit['vocabulary_added_tokens'] = added_tokens
            self.audit['vocabulary_added_templates'] = len(codec.templates) - len(parent_codec.templates)
            _report_preparation('vocabulary_extension', len(self.songs), len(self.songs), vocabulary_start,
                                added_token_count=len(added_tokens),
                                added_components=[token for token in added_tokens if not token.startswith('F:')],
                                added_templates=self.audit['vocabulary_added_templates'])
        else:
            self.audit['vocabulary_policy'] = 'checkpoint_fixed_no_extension'
        unknown_phases = set()
        lengths_by_chart = {}
        compatible_songs = []
        vocabulary_exclusions = []
        for index, song in enumerate(self.songs, 1):
            chart_key = (song.chart_sha256, song.difficulty)
            try:
                if continuous and chart_key in lengths_by_chart:
                    song.event_token_lengths = lengths_by_chart[chart_key]
                else:
                    for point in song.points:
                        if continuous:
                            song.event_token_lengths.append(len(codec.encode_points([point])))
                        else:
                            relative = point.beat % window_beats
                            codec.encode_points([point], origin=point.beat - relative, region='target')
                            if 'T:' + str(relative) not in codec.ids:
                                unknown_phases.add(str(relative % 1))
                    if continuous:
                        lengths_by_chart[chart_key] = song.event_token_lengths
            except VocabularyCoverageError as error:
                if not validate_fixed_vocabulary and song.learning_role!='mechanics':
                    raise VocabularyCoverageError(
                        f'composer_v2 checkpoint cannot encode {song.source_path} '
                        f'difficulty={song.difficulty} beat={point.beat}: {error}') from error
                # The unknown event could also occur in a later input history,
                # a carried sustain or a reference, so exclude the whole record.
                record = dict(chart_index=index-1, source=str(song.source_path), difficulty=song.difficulty,
                              first_unsupported_beat=str(point.beat), reason=str(error))
                vocabulary_exclusions.append(record)
                prefix='composer_v2_validation_vocab_excluded'
                if song.learning_role=='mechanics':
                    record['learning_role']='mechanics'
                    prefix='composer_v2_mechanics_vocab_excluded'
                print(prefix+' '+json.dumps(record,ensure_ascii=False),flush=True)
            else:
                compatible_songs.append(song)
            if index % 100 == 0 or index == len(self.songs):
                _report_preparation('vocabulary', index, len(self.songs), vocabulary_start)
        self.songs = compatible_songs
        if vocabulary_exclusions:
            self.audit['vocabulary_exclusions'] = vocabulary_exclusions
        self.audit['preparation']['vocabulary_sec'] = time.perf_counter() - vocabulary_start
        self.audit['unseen_phase_numeric_fallback'] = sorted(unknown_phases)
        if unknown_phases:
            print('composer_v2_phase_fallback ' + json.dumps({
                'phases': sorted(unknown_phases), 'vocabulary_extended': False,
                'policy': 'lossless_numeric_time_escape'}, ensure_ascii=False), flush=True)
        self.codec = codec
        if audio_store is None:
            from .audio import AudioFeatureStore
            audio_store = AudioFeatureStore(cache_dir=cache_dir, backend=audio_backend,
                model_path=mert_model, device=prepare_device)
        self.audio_store = audio_store
        if prepare and self.songs:
            self.prepare()
        if not continuous:
            for song_index, song in enumerate(self.songs):
                for offset in range(max(1, math.ceil(float(song.end_beat) / window_beats))):
                    self.windows.append((song_index, Fraction(offset * window_beats)))
        self.audit['windows'] = len(self.windows)

    def prepare(self):
        """Materialize offline features once; item fetching never executes MERT."""
        try:
            self._prepare_audio()
        finally:
            self.close()
        if not self.songs:
            raise ValueError(f'No usable composer_v2 charts within audio extent: {self.audit["skipped"][-4:]}')
        return self

    def _prepare_audio(self):
        store = self.audio_store
        to_reference = getattr(store, 'reference', None)
        by_path = {}
        references = {}
        summaries = {}
        conditions = {}
        usable = []
        total = len({str(song.audio_path) for song in self.songs})
        started = time.perf_counter()
        initial_hits = getattr(store, 'cache_hits', 0)
        initial_builds = getattr(store, 'cache_builds', 0)
        _report_preparation('audio', 0, total, started, cache_hits=0, cache_builds=0)
        for song in self.songs:
            path = str(song.audio_path)
            if path not in by_path:
                prepared = store.prepare(song.audio_path)
                if prepared is None:
                    prepared = store.load(song.audio_path)
                audio_key = getattr(prepared, 'identity', {}).get('audio_sha256', path)
                if audio_key not in references:
                    references[audio_key] = to_reference(prepared) if to_reference else prepared
                by_path[path] = audio_key
                _report_preparation('audio', len(by_path), total, started, audio=path,
                    cache_hits=getattr(store, 'cache_hits', 0) - initial_hits,
                    cache_builds=getattr(store, 'cache_builds', 0) - initial_builds)
            audio_key = by_path[path]
            song.audio = references[audio_key]
            duration = float(song.audio.duration_sec)
            outside = [point for point in song.points if song.clock.at(point.beat) >= duration]
            if outside:
                rejected = dict(source=str(song.source_path), difficulty=song.difficulty,
                    reason='authored_onset_at_or_after_exclusive_audio_end',
                    first_beat=str(outside[0].beat), events=len(outside), duration_sec=duration,
                    policy='reject_whole_chart_preserve_authored_notes')
                self.audit['skipped'].append(rejected)
                print('composer_v2_chart_rejected ' + json.dumps(rejected, ensure_ascii=False), flush=True)
                continue
            audio_end = _beat(float(_beats_at_seconds(song.clock, duration)))
            if audio_end <= 0:
                rejected = dict(source=str(song.source_path), difficulty=song.difficulty,
                                reason='audio_ends_before_chart_beat_zero')
                self.audit['skipped'].append(rejected)
                print('composer_v2_chart_rejected ' + json.dumps(rejected, ensure_ascii=False), flush=True)
                continue
            if audio_key not in summaries:
                summaries[audio_key] = song.audio.summary(160)
            summary = summaries[audio_key]
            song.summary = dict(summary)
            song.summary['beats'] = _beats_at_seconds(song.clock, summary['times_sec'])
            # Only source E is capped/extended to the input-audio extent. No note moves.
            song.end_beat = audio_end
            late_sustains = ([] if song.sustain is None else [marker for marker in song.sustain.markers
                if float(marker['end_second']) > duration + 1e-8])
            if late_sustains:
                risk = dict(source=str(song.source_path), difficulty=song.difficulty,
                    actions=len(late_sustains), duration_sec=duration,
                    latest_native_end_sec=max(float(marker['end_second']) for marker in late_sustains),
                    guard_tolerance_sec=1e-8,
                    policy='preserve_source_duration_report_generation_guard_mismatch')
                self.audit.setdefault('sustain_audio_end_risks', []).append(risk)
                print('composer_v2_sustain_audio_end_risk ' + json.dumps(risk, ensure_ascii=False), flush=True)
            self.audit.setdefault('audio_extents', []).append(dict(source=str(song.source_path),
                difficulty=song.difficulty, authored_end_beat=str(song.authored_end_beat),
                window_end_beat=str(audio_end)))
            if (song.quality or {}).get('unmarked_weight') == 0:
                # An unreviewed remainder cannot define chart-wide radar/style.
                song.condition, song.condition_known = resolve_condition(
                    song.level, policy=self.condition_policy, max_level=self.max_condition_level)
                usable.append(song)
                continue
            condition_key = (song.chart_sha256 or song.source_path, song.difficulty, song.level, audio_key)
            if condition_key not in conditions:
                conditions[condition_key] = training_condition(song.points,
                    [EventTempo(beat, bpm) for beat, bpm in zip(song.clock.beats, song.clock.bpms)],
                    level=song.level, duration_sec=duration, policy=self.condition_policy,
                    audio=song.audio, first=song.clock.seconds[0], max_level=self.max_condition_level)
            condition, known = conditions[condition_key]
            song.condition, song.condition_known = condition.copy(), known.copy()
            usable.append(song)
        self.songs = usable
        self.audit.setdefault('preparation', {}).update(audio_sec=time.perf_counter() - started,
            cache_hits=getattr(store, 'cache_hits', 0) - initial_hits,
            cache_builds=getattr(store, 'cache_builds', 0) - initial_builds,
            distinct_audio_contents=len(references))

    def close(self):
        close = getattr(self.audio_store, 'close', None)
        if close:
            close()

    def __len__(self):
        return len(self.windows)

    @staticmethod
    def _slice(points, start, end):
        return [p for p in points if start <= p.beat < end]

    def _carry(self, song, points, history_start):
        if history_start <= 0:
            return []
        # For generated prefixes, rebuild native markers only when explicitly supplied.
        state = song.sustain
        if points is not song.points:
            from maichart.event_timing import EventTempo
            from maichart.generation.sustain_state import SustainStateIndex
            tempo = [EventTempo(beat, bpm) for beat, bpm in zip(song.clock.beats, song.clock.bpms)]
            state = SustainStateIndex(points, tempo, song.clock.seconds[0])
        if state is None:
            return []
        boundary = song.clock.at(history_start)
        active = {m['source_beat'] for m in state._candidate_markers(boundary)
                  if m['source_beat'] < history_start and float(m['end_second']) > boundary}
        return [p for p in points if p.beat in active]

    def _reference(self, song, start):
        """Nearest earlier acoustic phrase; no future symbolic target participates."""
        if song.audio is None or start < self.history_beats + self.window_beats:
            return None
        if start in song.reference_cache:
            return song.reference_cache[start]
        candidates = list(range(0, int(start) - self.history_beats - self.window_beats + 1,
                                self.window_beats))
        if not candidates:
            return None
        def descriptor(origin):
            if origin in song.descriptor_cache:return song.descriptor_cache[origin]
            beats = origin + np.arange(self.window_beats * 4) / 4.
            mel, semantic, valid = song.audio.sample(song.clock.at_floats(beats))
            values = np.asarray(semantic, dtype=np.float32)
            valid = np.asarray(valid, dtype=bool)
            if not valid.any():
                values = np.asarray(mel, dtype=np.float32)
                valid = np.ones(len(values), dtype=bool)
            vec = values[valid].mean(0)
            vector=vec / max(float(np.linalg.norm(vec)), 1e-8)
            song.descriptor_cache[origin]=vector
            return vector
        # The complete input audio is available; only future chart notes are hidden.
        query = descriptor(float(start))
        scores = [float(np.dot(query, descriptor(candidate))) for candidate in candidates]
        selected = Fraction(candidates[int(np.argmax(scores))])
        song.reference_cache[start] = selected
        return selected

    def build_window(self, song, points, start, reference_points=None, training=False):
        start = _beat(start)
        history_start = max(Fraction(0), start - self.history_beats)
        history = self._slice(points, history_start, start)
        carry = self._carry(song, points, history_start)
        active_at_target = {p.beat for p in self._carry(song, points, start)}
        target = self._slice(points, start, start + self.window_beats) if training else []
        reference_start = self._reference(song, start)
        if reference_points is None:
            reference_points = ([] if reference_start is None else self._slice(
                points, reference_start, reference_start + self.window_beats))
        if any(p.beat >= history_start for p in reference_points):
            raise ValueError('REF must end before the immediate history')
        codec = self.codec
        target_ids = codec.encode_points(target, origin=start, region='target')
        def prefix():
            return ([codec.bos_id, codec.carry_id]
                + codec.encode_points(carry, origin=start, region='carry')
                + [codec.ctx_id]
                + codec.encode_points(history, origin=start, region='history')
                + [codec.ref_id]
                + codec.encode_points(reference_points, origin=start, region='reference')
                + [codec.tgt_id])
        # Training and inference use the same target-independent prefix budget.
        # Mandatory carry survives; context sheds whole onsets, never target notes.
        prefix_ids = prefix()
        suffix = target_ids + [codec.eos_id] if training else []
        prefix_budget = self.max_tokens // 2
        while len(prefix_ids) > prefix_budget:
            if reference_points:
                reference_points = []
            elif history:
                removed = history[0]
                history = history[1:]
                # A trimmed history onset can still occupy the current window.
                # Move the whole authored group into mandatory carry in that case.
                if removed.beat in active_at_target and all(p.beat != removed.beat for p in carry):
                    carry = sorted([*carry, removed], key=lambda point: point.beat)
            else:
                raise ValueError(f'Complete target/carry exceeds fixed prefix budget={prefix_budget}, '
                                 f'prefix_tokens={len(prefix_ids)}, max_tokens={self.max_tokens}: '
                                 f'{song.source_path}, beat={start}; increase max_tokens')
            prefix_ids = prefix()
        if training and len(prefix_ids) + len(suffix) - 1 > self.max_tokens:
            raise ValueError(f'Complete target/carry exceeds max_tokens={self.max_tokens}: '
                f'{song.source_path}, beat={start}, prefix_tokens={len(prefix_ids)}, '
                f'target_tokens={len(target_ids)}, target_capacity={self.max_tokens-len(prefix_ids)}; '
                'target was preserved; increase max_tokens')
        full = prefix_ids + suffix
        clocks = np.asarray(codec.token_clocks(full), dtype=np.float32)
        if len(clocks) != len(full):
            raise ValueError('codec.token_clocks must return one causal clock per token')
        ids = full[:-1] if training else full
        labels = np.full(len(ids), -100, dtype=np.int64)
        if training:
            first = len(prefix_ids) - 1
            labels[first:] = full[len(prefix_ids):]
        weights = np.zeros(len(ids), dtype=np.float32)
        if training:
            active = (labels != -100).astype(np.float32)
            weights = active * float((song.quality or {}).get('unmarked_weight', 1.))
            # Excluded targets remain visible teacher-forcing context, with zero CE.
            target_clocks = clocks[1:]
            for region in (song.quality or {}).get('regions', []):
                sec = song.clock.at_floats(float(start) + target_clocks)
                mask = ((sec >= float(region['start_sec'])) & (sec < float(region['end_sec'])))
                weights[mask] = active[mask] * float(region.get('weight', 1.))
            if (song.quality or {}).get('unmarked_weight') == 0:
                weights[labels == codec.eos_id] = 0
        if song.audio is None:
            raise RuntimeError('Audio features are not prepared; call dataset.prepare()')
        local_beats = np.arange(-self.history_beats, self.window_beats + self.future_beats,
                                1 / 24., dtype=np.float64)
        seconds = song.clock.at_floats(local_beats + float(start))
        mel, semantic, valid = song.audio.sample(seconds)
        summary = song.summary
        output = {'input_ids': torch.tensor(ids, dtype=torch.long),
            'labels': torch.from_numpy(labels), 'token_beats': torch.from_numpy(clocks[:len(ids)]),
            'token_mask': torch.ones(len(ids), dtype=torch.bool),
            'token_weights': torch.from_numpy(weights),
            'condition': torch.as_tensor(song.condition, dtype=torch.float32),
            'condition_known': torch.as_tensor(song.condition_known if song.condition_known is not None
                else np.asarray([True] + [False] * 15), dtype=torch.bool),
            'sample_weight': torch.tensor(float((song.quality or {}).get('weight', 1.))),
            'local_audio': torch.as_tensor(np.asarray(mel), dtype=torch.float32),
            'local_semantic': torch.as_tensor(np.asarray(semantic), dtype=torch.float32),
            'local_semantic_mask': torch.as_tensor(np.asarray(valid), dtype=torch.bool),
            'local_mask': torch.as_tensor((seconds >= 0) & (seconds <= song.audio.duration_sec)
                                          & np.isfinite(seconds), dtype=torch.bool),
            'local_beats': torch.tensor(local_beats, dtype=torch.float32),
            'song_audio': torch.as_tensor(np.asarray(summary['mel']), dtype=torch.float32),
            'song_semantic': torch.as_tensor(np.asarray(summary['semantic']), dtype=torch.float32),
            'song_semantic_mask': torch.as_tensor(np.asarray(summary['semantic_valid']), dtype=torch.bool),
            'song_beats': torch.as_tensor(summary['beats'] - float(start), dtype=torch.float32),
            'song_mask': torch.ones(len(summary['times_sec']), dtype=torch.bool),
            'start_beat': start, 'song_id': song.song_id,
            'song_cache_key': hashlib.sha256((str(song.audio_path.resolve())+str(song.clock.beats)+str(song.clock.bpms)+str(song.clock.seconds)+json.dumps(getattr(song.audio,'identity',{}),sort_keys=True,default=str)).encode()).hexdigest(),
            'reference_start': reference_start, 'pad_id': codec.pad_id}
        if 'detail' in summary:
            output['song_detail']=torch.as_tensor(np.asarray(summary['detail']),dtype=torch.float32)
        return output

    def __getitem__(self, index):
        song_index, start = self.windows[index]
        song = self.songs[song_index]
        return self.build_window(song, song.points, start, training=True)


def collate_composer_v2(items):
    if not items:
        raise ValueError('Cannot collate an empty batch')
    output = {}
    sequence_keys = ('input_ids', 'labels', 'token_beats', 'token_mask', 'token_weights',
        'local_audio', 'local_semantic', 'local_semantic_mask', 'local_beats', 'local_mask',
        'song_audio', 'song_semantic', 'song_semantic_mask', 'song_beats', 'song_mask')
    for key in sequence_keys:
        fill = items[0]['pad_id'] if key == 'input_ids' else (-100 if key == 'labels' else 0)
        output[key] = torch.nn.utils.rnn.pad_sequence([item[key] for item in items],
            batch_first=True, padding_value=fill)
    token_mask = output['token_mask'].bool()
    output['right_padded_tokens'] = not bool((token_mask[:, 1:] & ~token_mask[:, :-1]).any())
    if all('song_detail' in item for item in items):
        output['song_detail'] = torch.nn.utils.rnn.pad_sequence([item['song_detail'] for item in items], batch_first=True)
    for key in ('condition', 'condition_known', 'sample_weight'):
        output[key] = torch.stack([item[key] for item in items])
    for key in ('start_beat', 'song_id', 'reference_start'):
        output[key] = [item[key] for item in items]
    if all('song_cache_key' in item for item in items):output['song_cache_key']=[item['song_cache_key'] for item in items]
    return output


GLOBAL_BATCH_ORDER = 'length_bucket_seed_epoch_v1'
AUDIO_LOCAL_BATCH_ORDER = 'audio_local_length_bucket_v2'


class LengthBucketBatchSampler:
    """Shuffle whole batches of similarly sized targets without dropping examples."""
    def __init__(self, dataset, batch_size, *, seed=0, bucket_batches=32, shuffle=True,
                 order=GLOBAL_BATCH_ORDER):
        if batch_size < 1 or bucket_batches < 1:
            raise ValueError('batch_size and bucket_batches must be positive')
        if order not in (GLOBAL_BATCH_ORDER, AUDIO_LOCAL_BATCH_ORDER):
            raise ValueError(f'Unknown Composer v2 batch order: {order}')
        self.dataset, self.batch_size = dataset, int(batch_size)
        self.seed, self.epoch = int(seed), 0
        self.start_batch = 0
        self.bucket_batches, self.shuffle = int(bucket_batches), bool(shuffle)
        self.audio_groups = None
        if shuffle and order == AUDIO_LOCAL_BATCH_ORDER:
            audio_keys = [song.audio.identity['audio_sha256'] for song in dataset.songs]
            groups = {}
            for index, (song_index, _) in enumerate(dataset.windows):
                groups.setdefault(audio_keys[song_index], []).append(index)
            self.audio_groups = list(groups.values())
        if hasattr(dataset,'lengths'):
            self.lengths=list(dataset.lengths);return
        self.lengths = []
        for song_index, start in dataset.windows:
            song = dataset.songs[song_index]
            target = dataset._slice(song.points, start, start + dataset.window_beats)
            self.lengths.append(len(dataset.codec.encode_points(target, origin=start, region='target')))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        if hasattr(self.dataset,'set_epoch'):
            self.dataset.set_epoch(epoch)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.audio_groups is not None:
            yield from self._audio_local_batches(rng)
            return
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            rng.shuffle(indices)
        batches = []
        width = self.batch_size * self.bucket_batches
        for offset in range(0, len(indices), width):
            bucket = sorted(indices[offset:offset + width], key=lambda index: self.lengths[index])
            batches.extend(bucket[i:i + self.batch_size] for i in range(0, len(bucket), self.batch_size))
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches[self.start_batch:]

    def _audio_local_batches(self, rng):
        # Mix songs inside shuffled blocks so both lengths and audio pages
        # can be reused. Carry partial batches forward without changing the
        # number of steps, even for blocks containing very short charts.
        group_order = rng.permutation(len(self.audio_groups))
        songs_per_block = 16
        width = self.batch_size * self.bucket_batches
        remainder = []
        batch_index = 0
        for start in range(0, len(group_order), songs_per_block):
            indices = [index for group in group_order[start:start + songs_per_block]
                       for index in self.audio_groups[group]]
            rng.shuffle(indices)
            indices[:0] = remainder
            remainder = []
            batches = []
            for offset in range(0, len(indices), width):
                bucket = sorted(indices[offset:offset + width], key=self.lengths.__getitem__)
                batches.extend(bucket[i:i + self.batch_size] for i in range(0, len(bucket), self.batch_size))
            if len(batches[-1]) < self.batch_size:
                remainder = batches.pop()
            rng.shuffle(batches)
            for batch in batches:
                if batch_index >= self.start_batch:
                    yield batch
                batch_index += 1
        if remainder and batch_index >= self.start_batch:
            yield remainder
