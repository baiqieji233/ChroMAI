"""Learned load-curve inference for chart_stream_v1."""
from __future__ import annotations

import numpy as np
import torch

from maichart.event_timing import EventTempo
from maichart.form.clock import AUDIO_CURVE_AXIS
from maichart.form.curve import load_curve_bin_centers, resample_load_curve, decode_density
from maichart.form.data import organize_song
from maichart.form.rational_audio import extract_rational_audio, planner_audio_queries
from maichart.generation.music_intensity import audio_sha256
from maichart.generation.conditions import GenerationCondition, GENERATION_DEFAULTS
from maichart.generation.timeline import normalize_tempo_map
from .checkpoint import load
from .runtime import resolve_device


def generate_learned_profile(audio_path, checkpoint_path, *, resolution_sec=1., condition=None,
                             bpm=120., first=0., tempo_map=None, meter_map=None, device='cpu'):
    device = resolve_device(device)
    model, payload = load(checkpoint_path, device=device)
    del payload
    model.eval()
    audio = extract_rational_audio(audio_path,
        cache_dir=str(__import__('pathlib').Path(checkpoint_path).parent/'curve_audio_cache'),
        evidence_profile='balanced_evidence_v1')
    tempo = normalize_tempo_map(tempo_map) if tempo_map is not None else [EventTempo(0, float(bpm))]
    song = organize_song(audio, tempo, first, meters=meter_map, max_seconds=999., quality_profile='musical_evidence_v1')
    values, times, _ = planner_audio_queries(song)
    condition = GenerationCondition.from_dict(condition) if isinstance(condition, dict) else condition
    condition = condition or GenerationCondition(level=13.3, radar=GENERATION_DEFAULTS, style=GENERATION_DEFAULTS)
    batch = {'audio': values[None].to(device), 'condition': condition.to_vector(device=device)[None],
             'audio_padding_mask': torch.zeros((1, len(values)), dtype=torch.bool, device=device),
             'condition_null': torch.zeros(1, dtype=torch.bool, device=device)}
    with torch.inference_mode():
        _, _, prediction = model.encode(batch['audio'], batch['condition'],
            audio_padding_mask=batch['audio_padding_mask'], condition_null=batch['condition_null'],
            return_curve=True)
    predicted = prediction[0].sigmoid().float().cpu().numpy()
    bins = load_curve_bin_centers(audio.duration_sec, resolution_sec)
    values = resample_load_curve(np.asarray(times, dtype=np.float64) - float(first), predicted,
                                 np.asarray(bins, dtype=np.float64) - float(first))
    values = np.clip(values, 0., 1.)
    density = decode_density(values[:, 0], 'musical_log_v1')
    positive = density[density > 1e-6]
    reference = float(np.mean(positive)) if len(positive) else 1.
    intensity = np.clip(.5 * density / max(reference, 1e-6), 0., 1.)
    editor_times = np.unique(np.concatenate(([0.0], np.asarray(bins, dtype=np.float64),
                                              [float(audio.duration_sec)])))
    editor_values = np.interp(editor_times, bins, intensity)
    editor_components = resample_load_curve(
        np.asarray(bins, dtype=np.float64), values, editor_times)
    points = [{'time_sec': float(t), 'intensity': float(v)}
              for t, v in zip(editor_times, editor_values)]
    return {'schema':'maichart-intensity-curve-v1', 'method':'learned_chart_stream_v1',
            'model_family':'chart_stream_v1', 'time_axis':AUDIO_CURVE_AXIS,
            'device':str(device), 'audio_evidence_profile':audio.evidence_profile,
            'audio_feature_policy':audio.feature_policy,
            'duration_sec':float(audio.duration_sec), 'curve_bin_seconds':float(resolution_sec),
            'density_profile':'musical_log_v1', 'points':points,
            'components': {'event_density': editor_components[:,0].tolist(),
                           'tap': editor_components[:,1].tolist(),
                           'sustain': editor_components[:,2].tolist(),
                           'slide_touch': editor_components[:,3].tolist()},
            'component_times_sec':[float(t) for t in editor_times], 'rest_regions':[],
            'checkpoint':str(checkpoint_path), 'audio_sha256':audio_sha256(audio_path)}
