"""CPU-only timing audit; ideal-label evidence is not checkpoint quality evidence.

Run from the installed project environment. No checkpoint, corpus, cache,
training task, or GPU is opened. --output writes only the requested JSON.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from maichart.event_timing import EventPoint, EventTempo, seconds_at_beat
from maichart.form.blueprint import BlueprintConfig, build_blueprint_targets
from maichart.form.onset_selection import local_peak_mask, merge_close_onsets
from maichart.generation.event_audio import FEATURE_RATE_HZ, _resample_rows


def ideal_run(bpm, division, phase_ms, *, count=32):
    """Feed the actual authored target directly to the actual peak selector.

    The target is a soft label, not a calibrated predicted probability. Using
    it as a score isolates representation/peak selection; it does not estimate
    how a trained checkpoint performs.
    """
    tempo = (EventTempo(Fraction(0), float(bpm)),)
    first = 0.3 + phase_ms / 1000.0
    points = [EventPoint(Fraction(index * 4, division), "1") for index in range(count)]
    expected = np.asarray([seconds_at_beat(point.beat, tempo, first) for point in points])
    times = np.arange(int(np.ceil((expected[-1] + .3) * FEATURE_RATE_HZ))) / FEATURE_RATE_HZ
    targets = build_blueprint_targets(points, tempo, first, times)
    labels = targets["onset_target"].numpy()
    peaks = local_peak_mask(labels)
    selected = np.flatnonzero(peaks & (labels >= .5))
    candidates = [dict(second=float(times[i]), score=float(labels[i]), beat=str(Fraction(i)))
                  for i in selected]
    # Every selected peak must still be close to some actual authored event.
    errors_ms = (np.min(np.abs(times[selected, None] - expected[None, :]), axis=1) * 1000
                 if len(selected) else np.zeros(0))
    return {
        "bpm": bpm, "simai_division": division, "phase_ms": phase_ms,
        "authored_events": count, "spacing_ms": 240000.0 / (bpm * division),
        "ideal_label_peaks_at_0_5": len(selected),
        "ideal_label_peaks_at_0_641": int((peaks & (labels >= .641)).sum()),
        "distinct_type_frames": int((targets["type_target"] >= 0).sum()),
        "maximum_peak_distance_to_authored_ms": float(errors_ms.max()) if len(errors_ms) else None,
        "after_merge": {str(ms): len(merge_close_onsets(candidates, ms)) for ms in (0, 30, 40, 56)},
    }


def interpolation_probe():
    values = np.asarray([[1.0], [2.0]], dtype=np.float32)
    source = np.asarray([0.0, .02])
    target = np.asarray([.03])
    return {
        "default_after_last_feature": float(_resample_rows(values, source, target)[0, 0]),
        "unused_fill_outside_extended_interval": float(
            _resample_rows(values, source, target, fill_outside=(0.0, .04))[0, 0]),
        "current_callers_use_fill_outside": False,
        "scope": "The optional extended interval extrapolates; current callers do not use it.",
    }


def projection_probe(bpm, division, phase_ms, *, count=32):
    """Compare ideal-frame point projection and sequence snapping to truth."""
    from maichart.form.generate import exact_candidate
    from maichart.form.rhythm_snap import approximate_onsets
    tempo = (EventTempo(Fraction(0), float(bpm)),)
    first = .3 + phase_ms / 1000
    points = [EventPoint(Fraction(i * 4, division), "1") for i in range(count)]
    expected = np.asarray([seconds_at_beat(point.beat, tempo, first) for point in points])
    duration = float(expected[-1] + .3)
    times = np.arange(int(np.ceil(duration * FEATURE_RATE_HZ))) / FEATURE_RATE_HZ
    labels = build_blueprint_targets(points, tempo, first, times)["onset_target"].numpy()
    selected = np.flatnonzero(local_peak_mask(labels) & (labels >= .5))
    report = {"bpm": bpm, "simai_division": division, "phase_ms": phase_ms,
              "authored_events": count, "selected_peaks": len(selected)}
    if len(selected) != count:
        report["comparison"] = "insufficient_independent_peaks_for_one_to_one_comparison"
        return report
    records = []
    for index in selected:
        second = float(times[index])
        beat = exact_candidate(second, tempo, first)
        records.append(dict(beat=str(beat), second=float(seconds_at_beat(beat, tempo, first)),
                            source_second=second, score=float(labels[index])))
    snapped, _ = approximate_onsets(records, tempo, first, tolerance_ms=30,
        source_tolerance_ms=30, min_interval_ms=0, duration_sec=duration,
        divisions=(4, 8, 12, 16, 24, 32), mode="sequence")
    def stats(values):
        return {"different_authored_beat_count": sum(Fraction(item["beat"]) != point.beat
                                                     for item, point in zip(values, points)),
                "maximum_authored_time_error_ms": float(np.max(np.abs(
                    np.asarray([item["second"] for item in values]) - expected)) * 1000)}
    report.update(point_projection=stats(records), sequence_projection=stats(snapped),
                  sequence_snap_ms=30, raw_source_limit_ms=30, merge_ms=0)
    return report


def transient_probe():
    """Measure feature maxima on synthetic attacks, not a learned onset model."""
    import librosa
    from maichart.generation.event_audio import _spectral_flux
    sr, hop, n_fft = 22050, 441, 2048
    results = []
    for waveform in ("impulse", "decaying_tone"):
        for requested in (1.0, 1.007, 1.01):
            signal = np.zeros(sr * 2, dtype=np.float32)
            start = round(requested * sr)
            if waveform == "impulse":
                signal[start] = 1.0
            else:
                length = int(sr * .1)
                local = np.arange(length) / sr
                signal[start:start + length] = np.cos(2 * np.pi * 1000 * local) * np.exp(-local / .01)
            channels = {
                "onset_strength": librosa.onset.onset_strength(y=signal, sr=sr, hop_length=hop),
                "spectral_flux": _spectral_flux(signal, sr, hop, n_fft),
                "rms": librosa.feature.rms(y=signal, frame_length=n_fft, hop_length=hop, center=True)[0],
            }
            actual = start / sr
            channel_results = {}
            for name, values in channels.items():
                times = np.arange(len(values)) * hop / sr
                selected = np.flatnonzero((times >= .8) & (times <= 1.2))
                peak = selected[int(np.argmax(values[selected]))]
                channel_results[name] = {
                    "peak_second": float(times[peak]), "offset_ms": float((times[peak] - actual) * 1000),
                }
            results.append({"waveform": waveform, "attack_second": actual, "features": channel_results})
    return {"librosa_version": librosa.__version__, "sample_rate": sr,
            "hop_length": hop, "analysis_window_ms": 1000 * n_fft / sr,
            "scope": "Feature maxima differ by channel/waveform; this is not a universal model delay.",
            "cases": results}


def run_audit(*, include_audio=True):
    config = BlueprintConfig()
    dilations = [((1, 2, 4, 8)[index % 4]) for index in range(config.temporal_layers)]
    receptive_frames = 1 + 2 * sum(dilations)
    report = {
        "schema": "song_form_timing_resolution_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {Path(inspect.getfile(function)).name:
                          hashlib.sha256(Path(inspect.getfile(function)).read_bytes()).hexdigest()
                          for function in (build_blueprint_targets, local_peak_mask, _resample_rows)},
        "scope": "CPU synthetic implementation audit; not model inference, music quality acceptance, or validation data.",
        "contract": {"feature_rate_hz": FEATURE_RATE_HZ, "frame_interval_ms": 1000 / FEATURE_RATE_HZ,
                     "label_sigma_ms": 20, "default_temporal_layers": config.temporal_layers,
                     "default_local_receptive_frames": receptive_frames,
                     "default_local_receptive_center_span_ms": (receptive_frames - 1) * 1000 / FEATURE_RATE_HZ},
        "ideal_target_runs": [ideal_run(bpm, division, phase)
                              for bpm in (60, 80, 120, 130, 180, 187.5, 240)
                              for division in (16, 24, 32, 48, 64)
                              for phase in (0, 5, 10, 15)],
        "interpolation": interpolation_probe(),
        "ideal_projection": [projection_probe(bpm, division, phase)
                             for bpm in (130, 180, 187.5)
                             for division in (16, 24, 32)
                             for phase in (0, 10)],
    }
    if include_audio:
        report["synthetic_transients"] = transient_probe()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-audio", action="store_true", help="Skip optional librosa synthetic attacks")
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = run_audit(include_audio=not args.skip_audio)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(str(args.output.resolve()))
    else:
        print(text)


if __name__ == "__main__":
    main()
