"""Editor-compatible intensity predictions from the event-set shared audio bank."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import math

import numpy as np
import torch
from torch.nn import functional as F

from maichart.event_timing import EventTempo
from maichart.form.clock import AUDIO_CURVE_AXIS
from maichart.form.curve import load_curve_bin_centers
from maichart.form.data import organize_song
from maichart.form.rational_audio import extract_rational_audio
from maichart.generation.music_intensity import SCHEMA, audio_sha256, validate_intensity_profile
from maichart.generation.timeline import normalize_tempo_map
from maichart.tempo_clock import TempoClock

from .stream_checkpoint import load, PROFILE
from .stream_data import build_audio_bank


def profile_from_predictions(seconds, logits, duration, resolution_sec, *, metadata=None,
                             plan_target_units='cell_counts_v1'):
    """Convert fine-cell rates/counts to interval-averaged editor intensities.

    Integrate the physical rates before reducing temporal resolution. Sampling
    one onset-sized cell at a bin center aliases beat pulses into fake rests.
    Blank probability is diagnostic evidence, never a hard rest region.
    """
    from .curve import rate_to_intensity, INTENSITY_MAPPING_POLICY, INTENSITY_RATE_CAP
    seconds = np.asarray(seconds, dtype=np.float64).reshape(-1)
    duration, resolution_sec = float(duration), float(resolution_sec)
    logits = torch.as_tensor(logits).detach().float().cpu()
    if (not math.isfinite(duration) or duration <= 0 or not math.isfinite(resolution_sec)
            or resolution_sec <= 0 or not len(seconds) or logits.shape != (len(seconds), 3)
            or not np.isfinite(seconds).all() or not bool(torch.isfinite(logits).all())):
        raise ValueError('Invalid stream plan clocks, predictions, duration or resolution')
    widths = np.diff(np.r_[seconds, duration])
    if seconds[0] != 0 or np.any(widths <= 0):
        raise ValueError('Stream plan cells must begin at zero and increase through audio end')
    values = F.softplus(logits[:, 1:]).numpy().astype(np.float64)
    if plan_target_units == 'rates_per_second_v2':
        rates = values
        counts = rates * widths[:, None]
    elif plan_target_units in {'cell_counts_v1', 'counts_per_cell_v1', 'counts_per_audio_cell'}:
        counts = values
        rates = counts / widths[:, None]
    else:
        raise ValueError(f'Unknown stream plan units: {plan_target_units}')
    head_rates, event_rates = rates.T
    blank = logits[:, 0].sigmoid().numpy()
    baseline = max(float(counts[:, 0].sum() / duration), 1e-6)
    # The editor validator allows at most 2048 points including both endpoints.
    bin_width=max(resolution_sec, float(np.nextafter(duration / 2046., np.inf)))
    bins = load_curve_bin_centers(duration, bin_width)
    edges=np.r_[np.arange(len(bins),dtype=np.float64)*bin_width,duration]
    cell_edges=np.r_[seconds,duration]
    # Each source prediction owns [seconds[i], seconds[i+1]); integrating
    # piecewise-constant values matches the plan count/exposure contract.
    values=np.column_stack((head_rates,event_rates,blank))
    area=np.vstack((np.zeros((1,3)),np.cumsum(values*widths[:,None],axis=0)))
    at_edges=np.column_stack([np.interp(edges,cell_edges,area[:,i]) for i in range(3)])
    averaged=np.diff(at_edges,axis=0)/np.diff(edges)[:,None]
    editor_times = np.unique(np.r_[0., bins, duration])
    # Endpoint knots repeat their adjacent interval, rather than sampling a
    # potentially unrelated final 10 ms cell or extrapolating into the song.
    heads = np.interp(editor_times, bins, averaged[:,0])
    events = np.interp(editor_times, bins, averaged[:,1])
    blanks = np.interp(editor_times, bins, averaged[:,2])
    intensities = rate_to_intensity(heads, baseline)
    points = [dict(time_sec=float(t), intensity=float(v)) for t, v in zip(editor_times, intensities)]
    profile = dict(metadata or {})
    profile.update(schema=SCHEMA, method='learned_chart_composer_event_set_stream_v1',
        model_family='chart_composer_v1', architecture_profile=PROFILE, time_axis=AUDIO_CURVE_AXIS,
        duration_sec=duration, curve_bin_seconds=bin_width,
        curve_aggregation_policy='time_integrated_bins_v1',
        baseline_head_rate=baseline, baseline_p95_rate=float(np.percentile(averaged[:,0], 95)),
        interval_edges_sec=edges.tolist(),interval_head_rates=averaged[:,0].tolist(),
        interval_event_rates=averaged[:,1].tolist(),
        intensity_mapping_policy=INTENSITY_MAPPING_POLICY, intensity_rate_cap=INTENSITY_RATE_CAP,
        points=points, reference_points=[dict(point) for point in points],
        reference_event_rates=events.tolist(), reference_head_rates=heads.tolist(),
        reference_blank_probabilities=blanks.tolist(), component_times_sec=editor_times.tolist(),
        components={'event_density': intensities.tolist(), 'blank_probability': blanks.tolist()},
        rest_regions=[], plan_target_units=plan_target_units,
        plan_contract='fixed_cell_blank_head_rate_event_rate_v2' if plan_target_units=='rates_per_second_v2' else 'fixed_cell_blank_head_count_event_count_v1')
    return validate_intensity_profile(profile)


@torch.inference_mode()
def generate_learned_profile(audio_path, checkpoint_path, *, resolution_sec=2., local_context_sec=12.,
        condition=None, bpm=None, first=0., tempo_mode='constant', tempo_map=None, meter_map=None,
        device='cpu', level=13.):
    from .curve import _select_device, _resolve_curve_condition
    if not math.isfinite(float(first)) or (bpm not in (None, '') and (not math.isfinite(float(bpm)) or float(bpm) <= 0)):
        raise ValueError('Curve first/BPM must be finite with positive BPM')
    dev = _select_device(device)
    model, payload = load(checkpoint_path, dev)
    model.eval()
    # Release mapped model/optimizer references before audio extraction.
    payload.clear()
    audio_file, checkpoint_file = Path(audio_path), Path(checkpoint_path)
    audio = extract_rational_audio(audio_file, cache_dir=checkpoint_file.parent / 'audio_cache',
                                   evidence_profile='balanced_evidence_v1')
    tempo = normalize_tempo_map(tempo_map) if tempo_map is not None else [EventTempo(Fraction(0), float(bpm or audio.tempo_bpm))]
    song = organize_song(audio, tempo, float(first), meters=meter_map,
                         max_seconds=999., quality_profile='musical_evidence_v1')
    grid_seconds = float(getattr(model.config, 'audio_grid_seconds', .05))
    bank = build_audio_bank(song, TempoClock(song.tempo, song.first), grid_seconds)
    features, seconds, beats = [value[None].to(dev) for value in bank]
    cache = model.encode_audio(features, torch.ones(features.shape[:2], dtype=torch.bool, device=dev), seconds, beats)
    resolved = _resolve_curve_condition(condition, level)
    control = resolved.to_vector(device=dev)[None]
    logits = model.plan_head(cache['local'])[0] + F.linear(model.condition(control), model.plan_head.weight)[0]
    return profile_from_predictions(bank[1].numpy(), logits, audio.duration_sec, resolution_sec,
        plan_target_units=getattr(model.config, 'plan_target_units', 'cell_counts_v1'),
        metadata=dict(checkpoint=str(checkpoint_file), audio_sha256=audio_sha256(audio_file),
            device=str(dev), audio_grid_seconds=grid_seconds, bpm=float(tempo[0].bpm), first=float(first), condition=resolved.to_dict(),
            tempo_map=[dict(beat=str(item.beat), bpm=float(item.bpm)) for item in tempo],
            meter_map=meter_map))
