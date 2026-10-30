"""Difficulty and workload curve prediction and interactive curve guidance for chart_composer_v1."""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from maichart.event_timing import EventTempo
from maichart.form.clock import AUDIO_CURVE_AXIS
from maichart.form.curve import encode_density, load_curve_bin_centers, resample_load_curve
from maichart.form.data import organize_song
from maichart.form.rational_audio import extract_rational_audio
from maichart.generation.conditions import GENERATION_DEFAULTS, GenerationCondition
from maichart.generation.music_intensity import SCHEMA, audio_sha256, validate_intensity_profile
from maichart.generation.timeline import normalize_tempo_map
from maichart.stream.audio_bounds import generation_bounds
from maichart.tempo_clock import TempoClock

from .acoustics import AcousticContext
from .candidates import COMPOSER_AUDIO_FEATURE_POLICY, composer_audio_nodes
from .checkpoint import load
from .generation_settings import resolve_condition
from .model import GROUNDED_PROFILE, WORKLOAD_PROFILES
from .planning import PlanGrid


CURVE_MODES = ('follow', 'soft', 'hard', 'strict')
CURVE_SPAN_MODES = ('full', 'trim')
INTENSITY_MAPPING_POLICY = 'bounded_odds_v1'
INTENSITY_RATE_CAP = 64.0


def rate_to_intensity(rate, baseline, rate_cap=INTENSITY_RATE_CAP):
    """Invertible on [0, rate_cap], including a finite intensity=1 endpoint."""
    r = np.clip(np.asarray(rate, dtype=np.float64), 0., float(rate_cap))
    b = float(baseline)
    return (r / (r + b)) * ((float(rate_cap) + b) / float(rate_cap))


def intensity_to_rate(intensity, baseline, rate_cap=INTENSITY_RATE_CAP):
    x = np.clip(np.asarray(intensity, dtype=np.float64), 0., 1.)
    b = float(baseline)
    return b * x / ((float(rate_cap) + b) / float(rate_cap) - x)



def _select_device(value: str = 'auto') -> torch.device:
    text = str(value or 'auto').strip().lower()
    if text == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(text)


def _resolve_curve_condition(condition: Any, level: float = 13.) -> GenerationCondition:
    if isinstance(condition, GenerationCondition):
        return condition
    if isinstance(condition, dict):
        merged = dict(condition)
        merged.setdefault('level', float(level))
        merged.setdefault('difficulty', 5)
        merged.setdefault('radar', dict(GENERATION_DEFAULTS))
        merged.setdefault('style', dict(GENERATION_DEFAULTS))
        return GenerationCondition.from_dict(merged)
    resolved, _ = resolve_condition(float(level), condition_file=None)
    return resolved


@torch.inference_mode()
def generate_learned_profile(
    audio_path: str | Path,
    checkpoint_path: str | Path,
    *,
    resolution_sec: float = 2.0,
    local_context_sec: float = 12.0,
    condition: Any = None,
    bpm: float | None = None,
    first: float = 0.0,
    tempo_mode: str = 'constant',
    tempo_map: Any = None,
    meter_map: Any = None,
    device: str = 'cpu',
    level: float = 13.0,
) -> dict[str, Any]:
    """Predict a song-long workload and planner intensity curve from a Composer checkpoint."""
    from .checkpoint import read_payload
    checkpoint_header = read_payload(checkpoint_path)
    stream = (checkpoint_header.get('schema') == 'maichart-composer-event-set-checkpoint-v1'
              or (checkpoint_header.get('model_config') or {}).get('architecture_profile') == 'event_set_stream_v1')
    del checkpoint_header
    if stream:
        from .stream_curve import generate_learned_profile as generate_stream_profile
        return generate_stream_profile(audio_path, checkpoint_path, resolution_sec=resolution_sec,
            local_context_sec=local_context_sec, condition=condition, bpm=bpm, first=first,
            tempo_mode=tempo_mode, tempo_map=tempo_map, meter_map=meter_map, device=device, level=level)
    del local_context_sec, tempo_mode
    if not math.isfinite(float(first)) or not math.isfinite(float(resolution_sec)) or float(resolution_sec) <= 0:
        raise ValueError('Curve first/resolution must be finite with positive resolution')
    if bpm not in (None, '') and (not math.isfinite(float(bpm)) or float(bpm) <= 0):
        raise ValueError('Curve BPM must be finite and positive')

    dev = _select_device(device)
    model, payload = load(checkpoint_path, device=dev)
    model.eval()
    learned = model.config.architecture_profile in WORKLOAD_PROFILES
    grounded = model.config.architecture_profile == GROUNDED_PROFILE
    if not learned:
        raise ValueError('Composer checkpoint does not contain a learned workload/planner head')

    audio_file = Path(audio_path)
    ckpt_file = Path(checkpoint_path)
    audio = extract_rational_audio(
        audio_file,
        cache_dir=ckpt_file.parent / 'audio_cache',
        evidence_profile='balanced_evidence_v1',
    )
    tempo = (
        normalize_tempo_map(tempo_map)
        if tempo_map is not None
        else [EventTempo(Fraction(0), float(audio.tempo_bpm if bpm in (None, '') else bpm))]
    )
    song = organize_song(
        audio,
        tempo,
        float(first),
        meters=meter_map,
        max_seconds=999.,
        quality_profile='musical_evidence_v1',
    )
    clock = TempoClock(song.tempo, song.first)
    values, times = composer_audio_nodes(song, policy=COMPOSER_AUDIO_FEATURE_POLICY)
    bounds = generation_bounds(audio, trim_exact_silence=True)
    end_sec = float(bounds['effective_end_sec'])
    acoustic_context = AcousticContext(values, times, audio.duration_sec, validated_bank=True)

    resolved_condition = _resolve_curve_condition(condition, level=level)
    control = resolved_condition.to_vector(device=dev)[None]

    bins = load_curve_bin_centers(audio.duration_sec, float(resolution_sec))
    workload_inputs = torch.from_numpy(acoustic_context.at(bins))[None].to(dev)
    workload_pred = model.predict_workload(workload_inputs, control)[0].float().cpu().numpy()
    # workload_pred shape: [N, 5] -> (log1p(event_rate), log1p(head_rate), hold, slide, touch)
    wl_event_rate = np.expm1(np.clip(workload_pred[:, 0], 0., 5.))
    wl_head_rate = np.expm1(np.clip(workload_pred[:, 1], 0., 5.))
    hold_frac = np.clip(workload_pred[:, 2], 0., 1.)
    slide_frac = np.clip(workload_pred[:, 3], 0., 1.)
    touch_frac = np.clip(workload_pred[:, 4], 0., 1.)

    if grounded:
        plan_grid = PlanGrid(clock, end_sec, acoustic_context)
        planner_audio = torch.from_numpy(plan_grid.audio)[None].to(dev)
        planner_mask = torch.zeros(planner_audio.shape[:2], device=dev, dtype=torch.bool)
        if getattr(model.config, 'configuration_quality_policy', 'off') == 'song_paired_v3':
            from .quality_inputs import plan_clock_features
            song_state = model.planner.encode(planner_audio, planner_mask,
                plan_clock_features(plan_grid, clock)[None].to(dev))
            plan_pred = model.planner.heads(song_state, control)[0][0].float().cpu().numpy()
        else:
            plan_pred = model.planner(planner_audio, control, planner_mask)[0].float().cpu().numpy()
        # plan_pred shape: [T_plan, 6] -> (log_event_rate, log_head_rate, hold, slide, touch, rest)
        plan_centers = np.asarray(plan_grid.centers, dtype=np.float64)
        if len(plan_centers) > 0:
            plan_event_rate = np.interp(bins, plan_centers, np.expm1(np.clip(plan_pred[:, 0], 0., 5.)))
            plan_head_rate = np.interp(bins, plan_centers, np.expm1(np.clip(plan_pred[:, 1], 0., 5.)))
            plan_rest = np.interp(bins, plan_centers, np.clip(plan_pred[:, 5], 0., 1.))
            active_gate = np.clip(1. - 0.65 * plan_rest, 0.15, 1.0)
            combined_head_rate = (0.65 * wl_head_rate + 0.35 * plan_head_rate) * active_gate
            combined_event_rate = (0.65 * wl_event_rate + 0.35 * plan_event_rate) * active_gate
        else:
            combined_head_rate = wl_head_rate
            combined_event_rate = wl_event_rate
    else:
        combined_head_rate = wl_head_rate
        combined_event_rate = wl_event_rate

    in_bounds = (bins >= float(bounds.get('effective_start_sec', 0.))) & (bins <= end_sec)
    combined_head_rate = np.where(in_bounds, combined_head_rate, 0.)
    combined_event_rate = np.where(in_bounds, combined_event_rate, 0.)

    positive = combined_head_rate[combined_head_rate > 1e-4]
    reference = float(np.mean(positive)) if len(positive) else 2.5
    p95_rate = float(np.percentile(positive, 95)) if len(positive) else 5.0
    intensity = rate_to_intensity(combined_head_rate, reference)

    event_density = np.clip(encode_density(combined_head_rate, 'musical_log_v1'), 0., 1.)
    sustain_comp = np.clip(hold_frac + 0.5 * slide_frac, 0., 1.)
    slide_touch_comp = np.clip(slide_frac + touch_frac, 0., 1.)
    tap_comp = np.clip(1. - np.maximum(hold_frac, slide_touch_comp), 0., 1.)
    comp_matrix = np.stack([event_density, tap_comp, sustain_comp, slide_touch_comp], axis=-1)

    editor_times = np.unique(
        np.concatenate(([0.0], np.asarray(bins, dtype=np.float64), [float(audio.duration_sec)]))
    )
    editor_values = np.clip(np.interp(editor_times, bins, intensity), 0., 1.)
    if editor_times[0] < float(bounds.get('effective_start_sec', 0.)):
        editor_values[0] = float(editor_values[min(1, len(editor_values) - 1)] * 0.25)
    if editor_times[-1] > end_sec:
        editor_values[-1] = 0.0
    editor_components = np.clip(
        resample_load_curve(np.asarray(bins, dtype=np.float64), comp_matrix, editor_times),
        0.,
        1.,
    )

    points = [
        {'time_sec': float(t), 'intensity': float(v)}
        for t, v in zip(editor_times, editor_values)
    ]
    profile = {
        'schema': SCHEMA,
        'method': f'learned_{model.config.architecture_profile}',
        'model_family': 'chart_composer_v1',
        'architecture_profile': model.config.architecture_profile,
        'time_axis': AUDIO_CURVE_AXIS,
        'device': str(dev),
        'bpm': float(tempo[0].bpm),
        'first': float(first),
        'duration_sec': float(audio.duration_sec),
        'curve_bin_seconds': float(resolution_sec),
        'baseline_head_rate': float(reference),
        'intensity_mapping_policy': INTENSITY_MAPPING_POLICY,
        'intensity_rate_cap': INTENSITY_RATE_CAP,
        'baseline_p95_rate': round(float(p95_rate), 4),
        'points': points,
        'reference_points': [dict(point) for point in points],
        'reference_event_rates': np.interp(editor_times, bins, combined_event_rate).tolist(),
        'reference_head_rates': np.interp(editor_times, bins, combined_head_rate).tolist(),
        'components': {
            'event_density': editor_components[:, 0].tolist(),
            'tap': editor_components[:, 1].tolist(),
            'sustain': editor_components[:, 2].tolist(),
            'slide_touch': editor_components[:, 3].tolist(),
        },
        'component_times_sec': [float(t) for t in editor_times],
        'rest_regions': [],
        'checkpoint': str(ckpt_file),
        'audio_sha256': audio_sha256(audio_file),
        'condition': resolved_condition.to_dict(),
    }
    return validate_intensity_profile(profile)


@dataclass
class ComposerCurveController:
    enabled: bool = False
    mode: str = 'follow'
    span_mode: str = 'full'
    strength: float = 1.0
    chord_multiplier: float = 1.0
    duration_sec: float = 0.0
    effective_start_sec: float = 0.0
    effective_end_sec: float = 0.0
    baseline_head_rate: float = 3.0
    mapping_policy: str = 'legacy_linear_v1'
    rate_cap: float = INTENSITY_RATE_CAP
    point_times: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    point_values: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    reference_times: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    reference_values: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    rest_regions: list[tuple[float, float]] = field(default_factory=list)
    source_method: str = 'none'

    @property
    def active(self) -> bool:
        return self.enabled and self.strength > 0.

    def _profile_seconds(self, seconds):
        secs = np.asarray(seconds, dtype=np.float64)
        if self.span_mode == 'trim' and self.effective_end_sec > self.effective_start_sec:
            mapped = (secs - self.effective_start_sec) * self.duration_sec / (self.effective_end_sec - self.effective_start_sec)
        else:
            mapped = secs
        return mapped

    def _values_at(self, seconds, *, reference=False):
        query = self._profile_seconds(seconds)
        times = self.reference_times if reference else self.point_times
        values = self.reference_values if reference else self.point_values
        sampled = np.interp(query, times, values)
        sampled = np.where((query >= 0.) & (query <= self.duration_sec), sampled, 0.)
        if not reference:
            for low, high in self.rest_regions:
                sampled = np.where((query >= low) & (query <= high), 0., sampled)
        return sampled

    def target_at(self, second: float) -> float:
        if not self.active or self.point_times.size == 0:
            return 0.5
        return float(self._values_at(second))

    def target_array(self, seconds: np.ndarray) -> np.ndarray:
        secs = np.asarray(seconds, dtype=np.float64).reshape(-1)
        return (self._values_at(secs).astype(np.float32) if self.active
                else np.full(secs.shape, .5, dtype=np.float32))

    def rest_mask(self, seconds: np.ndarray) -> np.ndarray:
        secs = np.asarray(seconds, dtype=np.float64).reshape(-1)
        result = np.zeros(secs.shape, dtype=bool)
        if not self.active:
            return result
        query = self._profile_seconds(secs)
        for low, high in self.rest_regions:
            result |= (query >= low) & (query <= high)
        if self.mode == 'strict':
            zero_target = self._values_at(secs) <= 0.015
            if self.reference_times.size:
                zero_target &= self._values_at(secs, reference=True) > 0.015
            result |= zero_target
        elif self.mode in {'follow', 'hard'} and self.point_times.size >= 2:
            # A zero-intensity outro tail (all remaining control points <= 0.015)
            # means the chart has ended; prevent placing stray notes at audio_end.
            active_pts = np.flatnonzero(self.point_values > 0.015)
            if active_pts.size and int(active_pts[-1]) < self.point_times.size - 1:
                outro_zero_start = float(self.point_times[int(active_pts[-1]) + 1])
                result |= (query >= outro_zero_start) & (self._values_at(secs) <= 0.015)
        return result

    def is_in_rest_region(self, second: float) -> bool:
        return bool(self.rest_mask(np.asarray([second]))[0])

    def has_future_activity(self, after_sec: float, end_sec: float, *, min_intensity: float = 0.06) -> bool:
        if not self.active or self.point_times.size == 0:
            return False
        probe = np.arange(float(after_sec) + .25, float(end_sec), .5)
        if not probe.size:
            return False
        active_probe = (~self.rest_mask(probe)) & (self.target_array(probe) >= min_intensity)
        if not np.any(active_probe):
            return False
        mapped_after = float(self._profile_seconds(float(after_sec)))
        future_points = self.point_values[self.point_times > mapped_after + 1e-6]
        if future_points.size and np.all(future_points < min_intensity) and int(np.count_nonzero(active_probe)) <= 1:
            # Do not veto EOS on a single sub-second linear-interpolation tail
            # tapering directly into a zero-intensity outro.
            return False
        return True

    def _blend_alpha(self) -> float:
        factor = {'soft': .45, 'follow': .75, 'hard': .95, 'strict': 1.}[self.mode]
        return float(np.clip(factor * self.strength, 0., 1.))

    def _decode(self, values):
        if self.mapping_policy == INTENSITY_MAPPING_POLICY:
            return intensity_to_rate(values, self.baseline_head_rate, self.rate_cap)
        return 2. * np.asarray(values) * self.baseline_head_rate

    def _rates_at(self, seconds, *, reference=False):
        return self._decode(self._values_at(seconds, reference=reference))

    def _window_rates(self, seconds, width, *, reference=False):
        # A bounded vectorized quadrature over the same past windows as ledger;
        # no GPU readback, recursive sampling or per-candidate model evaluation.
        nodes = (np.arange(32, dtype=np.float64) + .5) * (width / 32.)
        return self._rates_at(np.asarray(seconds)[:, None] - nodes, reference=reference).mean(axis=-1)

    def _adjust_rates(self, original, seconds, *, width=None):
        """Edit head density while retaining each branch's own heads/event ratio.

        The immutable prediction is a reference for edits, not a replacement for
        the differently sized planner/workload windows. A saved, unedited model
        curve therefore leaves all of those predictions exactly unchanged.
        """
        target = (self._rates_at(seconds) if width is None else self._window_rates(seconds, width))
        reference = None
        if self.reference_times.size:
            reference = (self._rates_at(seconds, reference=True) if width is None
                         else self._window_rates(seconds, width, reference=True))
        target_t = torch.as_tensor(target, device=original.device, dtype=original.dtype)
        native_event, native_head = original.unbind(-1)
        if reference is None:
            desired_head = target_t
        else:
            ref = torch.as_tensor(reference, device=original.device, dtype=original.dtype)
            desired_head = torch.where(ref > 1e-6, native_head * target_t / ref.clamp_min(1e-6),
                                       native_head + target_t - ref)
        new_head = native_head + self._blend_alpha() * (desired_head - native_head)
        ratio = torch.where(native_head > 1e-6, native_event / native_head.clamp_min(1e-6),
                            torch.ones_like(native_head)).clamp_min(0.)
        new_event = (new_head * ratio).clamp(0., self.rate_cap)
        chord_scale = 1. + self._blend_alpha() * (self.chord_multiplier - 1.)
        if self.chord_multiplier != 1.:
            # The main curve owns total note density. Chord size redistributes
            # that same material across fewer/more event times, never creates it.
            new_event = torch.minimum(new_event / chord_scale, new_head)
        new_head = new_head.clamp(0., self.rate_cap)
        adjusted = torch.stack((new_event, new_head), dim=-1)
        if reference is not None and self.chord_multiplier == 1.:
            adjusted = torch.where((target_t == ref)[:, None], original, adjusted)
        return adjusted

    def _neutral(self):
        return (self.chord_multiplier == 1. and not self.rest_regions and self.reference_times.size
                and np.array_equal(self.point_times, self.reference_times)
                and np.array_equal(self.point_values, self.reference_values))

    def transform_survival_rates(self,rates,head_rates,seconds):
        """Apply actual curve edits to the same fixed-grid timing intensity.

        Unedited learned curves remain identity; learned planner rests already
        affect native survival independently of the editor and its strength.
        Explicit user rest regions still set their exact grid cells to zero.
        """
        if not self.active or self._neutral():
            return rates
        if rates.shape!=head_rates.shape or rates.ndim!=2 or rates.size(0)!=1:
            raise ValueError('Generation survival curve expects [1,K] rates')
        seconds=np.asarray(seconds,dtype=np.float64).reshape(-1)
        if len(seconds)!=rates.size(1):raise ValueError('Survival curve clocks must match grid rates')
        real=torch.stack((rates[0].float(),head_rates[0].float()),-1)
        adjusted=self._adjust_rates(real,seconds)[:,0]
        rests=torch.as_tensor(self.rest_mask(seconds),device=rates.device)
        return adjusted.masked_fill(rests,0.)[None]

    def transform_planner(self, planner_prediction, plan_centers):
        if not self.active or planner_prediction is None or self._neutral():
            return planner_prediction
        centers = np.asarray(plan_centers, dtype=np.float64).reshape(-1)
        count = min(len(centers), planner_prediction.shape[1])
        result = planner_prediction.clone()
        real = planner_prediction[0, :count, :2].float().clamp_min(0.).expm1()
        rates = self._adjust_rates(real, centers[:count])
        rests = torch.as_tensor(self.rest_mask(centers[:count]), device=result.device)
        result[0, :count, :2] = rates.masked_fill(rests[:, None], 0.).log1p().to(result.dtype)
        # Rest and note-type channels are independent controls. Editing density
        # never silently turns a busy prediction into rests or changes types.
        result[0, :count, 2:5] = result[0, :count, 2:5].masked_fill(rests[:, None], 0.)
        result[0, :count, 5] = result[0, :count, 5].masked_fill(rests, 1.)
        return result

    def build_window_overrides(self, *, candidate_seconds, model_workload, model_trailing=None):
        secs = np.asarray(candidate_seconds, dtype=np.float64).reshape(-1)
        result = {'workload_override': None, 'trailing_override': None,
                  'rest_mask': self.rest_mask(secs), 'target_intensities': self.target_array(secs),
                  'total_heads_budget': None}
        if not self.active or not len(secs) or self._neutral():
            return result
        if model_workload is not None:
            out = model_workload.clone()
            count = min(len(secs), out.shape[1])
            real = out[0, :count, :2].float().clamp_min(0.).expm1()
            rates = self._adjust_rates(real, secs[:count])
            rests = torch.as_tensor(self.rest_mask(secs[:count]), device=out.device)
            out[0, :count, :2] = rates.masked_fill(rests[:, None], 0.).log1p().to(out.dtype)
            out[0, :count, 2:5] = out[0, :count, 2:5].masked_fill(rests[:, None], 0.)
            result['workload_override'] = out
            # Optional chord reduction is a per-selected-event count budget,
            # including Touch, separate from the physical two-button budget.
            if self.mode in {'hard', 'strict'} and self.chord_multiplier < 1.:
                result['total_heads_budget'] = (rates[:, 1] / rates[:, 0].clamp_min(1e-6)).clamp_min(1.)
        if model_trailing is not None:
            out = model_trailing.clone()
            count = min(len(secs), out.shape[1])
            for offset, width in ((0, 1.), (2, 4.)):
                real = out[0, :count, offset:offset + 2].float().clamp_min(0.).expm1()
                rates = self._adjust_rates(real, secs[:count], width=width)
                out[0, :count, offset:offset + 2] = rates.log1p().to(out.dtype)
            result['trailing_override'] = out
        return result

    def summarize_diagnostics(self, *, points, clock, curve_times, actual_head_rates):
        if not self.active or len(curve_times) == 0:
            return {'enabled': False}
        target_rates = self._rates_at(curve_times)
        err = float(np.mean(np.abs(actual_head_rates - target_rates))) if len(actual_head_rates) else None
        corr = (float(np.corrcoef(target_rates, actual_head_rates)[0, 1])
                if len(curve_times) > 1 and np.std(target_rates) > 1e-8 and np.std(actual_head_rates) > 1e-8 else None)
        return {'enabled': True, 'mode': self.mode, 'span_mode': self.span_mode,
                'strength': self.strength, 'chord_multiplier': self.chord_multiplier,
                'source_method': self.source_method, 'mapping_policy': self.mapping_policy,
                'baseline_head_rate': self.baseline_head_rate,
                'reference_relative_edits': bool(self.reference_times.size),
                'rest_regions': [[a, b] for a, b in self.rest_regions],
                'rest_region_violations': sum(self.is_in_rest_region(float(clock.at(p.beat))) for p in points),
                'target_head_rate': target_rates.tolist(), 'mean_target_head_rate_error': err,
                'target_head_rate_correlation': corr}


def apply_curve_to_composer(curve_payload_or_file, *, duration_sec, effective_start_sec=0.,
                            effective_end_sec=None, mode='follow', span_mode='full', strength=1.,
                            baseline_head_rate=3., chord_multiplier=1.):
    eff_end = float(duration_sec if effective_end_sec is None else effective_end_sec)
    if curve_payload_or_file is None or curve_payload_or_file == '':
        return ComposerCurveController(duration_sec=float(duration_sec), effective_start_sec=float(effective_start_sec),
                                       effective_end_sec=eff_end)
    raw = curve_payload_or_file
    if isinstance(raw, (str, Path)):
        raw = json.loads(Path(raw).read_text(encoding='utf-8-sig'))
    normalized = validate_intensity_profile(raw)
    resolved_mode = str(mode or raw.get('mode') or 'follow').lower()
    resolved_span = str(span_mode or raw.get('span_mode') or 'full').lower()
    if resolved_mode not in CURVE_MODES or resolved_span not in CURVE_SPAN_MODES:
        raise ValueError('Unsupported Composer intensity curve mode/span')
    if not math.isfinite(float(strength)) or not 0. <= float(strength) <= 2.:
        raise ValueError('intensity_curve_strength must be between 0 and 2')
    if not math.isfinite(float(chord_multiplier)) or not .25 <= float(chord_multiplier) <= 4.:
        raise ValueError('intensity_curve_chord_multiplier must be between 0.25 and 4')
    base = float(raw.get('baseline_head_rate') or baseline_head_rate)
    cap = float(raw.get('intensity_rate_cap', INTENSITY_RATE_CAP))
    if not math.isfinite(base) or base <= 0. or not math.isfinite(cap) or cap <= 0.:
        raise ValueError('Intensity curve rate baseline/cap must be finite and positive')
    policy = str(raw.get('intensity_mapping_policy', 'legacy_linear_v1'))
    if policy not in {INTENSITY_MAPPING_POLICY, 'legacy_linear_v1'}:
        raise ValueError('Unsupported Composer intensity mapping policy')
    ref = raw.get('reference_points') or []
    if ref:
        ref = validate_intensity_profile({**raw, 'points': ref, 'components': {}})['points']
    points = normalized['points']
    return ComposerCurveController(
        enabled=float(strength) > 0., mode=resolved_mode, span_mode=resolved_span,
        strength=float(strength), chord_multiplier=float(chord_multiplier), duration_sec=float(normalized['duration_sec']),
        effective_start_sec=float(effective_start_sec), effective_end_sec=eff_end,
        baseline_head_rate=base, mapping_policy=policy, rate_cap=cap,
        point_times=np.asarray([p['time_sec'] for p in points], dtype=np.float64),
        point_values=np.asarray([p['intensity'] for p in points], dtype=np.float64),
        reference_times=np.asarray([p['time_sec'] for p in ref], dtype=np.float64),
        reference_values=np.asarray([p['intensity'] for p in ref], dtype=np.float64),
        rest_regions=[(r['start_sec'], r['end_sec']) for r in normalized.get('rest_regions', [])],
        source_method=str(normalized.get('method', 'custom')))
