"""Select real score peaks and suppress near-duplicate onset candidates."""
from bisect import bisect_left
from fractions import Fraction
import math

import numpy as np


def local_peak_mask(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError('Onset scores must be a finite one-dimensional series')
    if not len(scores):
        return np.zeros(0, dtype=np.bool_)
    # A plateau has one representative, not one event per plateau frame.
    return np.r_[True, scores[1:] > scores[:-1]] & np.r_[scores[:-1] >= scores[1:], True]


def odds_modulate_probabilities(probabilities, factors):
    """Apply positive evidence multipliers to probability odds.

    A factor of one returns the input probability exactly. A zero factor is a
    hard exclusion; positive factors change odds without collapsing confident
    probabilities as direct multiplication would.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    factors = np.asarray(factors, dtype=np.float64)
    try:
        probabilities, factors = np.broadcast_arrays(probabilities, factors)
    except ValueError as exc:
        raise ValueError("probabilities and factors must be broadcast-compatible") from exc
    if (not np.isfinite(probabilities).all() or not np.isfinite(factors).all()
            or np.any((probabilities < 0) | (probabilities > 1)) or np.any(factors < 0)):
        raise ValueError("probabilities must be in [0, 1] and factors finite and nonnegative")
    result = probabilities.copy()
    active = (factors != 1) & (probabilities != 0)
    result[factors == 0] = 0.0
    positive = active & (factors > 0)
    result[positive] = (probabilities[positive] * factors[positive] /
                        (1.0 - probabilities[positive] + probabilities[positive] * factors[positive]))
    return result


def onset_gap_diagnostics(*, times, probabilities, adjusted, peaks, activity,
                          rest_mask, rest_regions, onsets, threshold,
                          curve_intensity=None, curve_factors=None,
                          start_sec=None, end_sec=None,
                          gap_threshold_sec=2.0, max_gaps=16):
    """Explain long final onset gaps using planner, audio, and curve stages."""
    arrays = [np.asarray(value, dtype=np.float64) for value in
              (times, probabilities, adjusted, activity)]
    times, probabilities, adjusted, activity = arrays
    peaks = np.asarray(peaks, dtype=np.bool_)
    rest_mask = np.asarray(rest_mask, dtype=np.bool_)
    if curve_intensity is None:
        curve_intensity = np.full(times.shape, np.nan, dtype=np.float64)
    else:
        curve_intensity = np.asarray(curve_intensity, dtype=np.float64)
    if curve_factors is None:
        curve_factors = np.ones(times.shape, dtype=np.float64)
    else:
        curve_factors = np.asarray(curve_factors, dtype=np.float64)
    if any(value.shape != times.shape for value in
           (probabilities, adjusted, peaks, activity, rest_mask, curve_intensity, curve_factors)):
        raise ValueError("onset diagnostic frame arrays must have matching shapes")
    if not np.isfinite(times).all() or not np.isfinite(probabilities).all() \
            or not np.isfinite(adjusted).all() or not np.isfinite(activity).all():
        raise ValueError("onset diagnostic frame values must be finite")
    if not math.isfinite(float(threshold)) or not 0 < float(threshold) < 1:
        raise ValueError("threshold must be finite and in (0, 1)")
    if not math.isfinite(float(gap_threshold_sec)) or gap_threshold_sec <= 0 or max_gaps < 1:
        raise ValueError("gap diagnostic limits must be positive")

    thresholds = sorted({0.4, 0.5, float(threshold)})
    audible = activity >= 0.05
    eligible = peaks & audible & ~rest_mask
    totals = {
        "raw_local_peaks": int(peaks.sum()),
        "audible_peaks": int((peaks & audible).sum()),
        "eligible_peaks": int(eligible.sum()),
        "raw_pass_by_threshold": {str(value): int((eligible & (probabilities >= value)).sum())
                                   for value in thresholds},
        "adjusted_pass_by_threshold": {str(value): int((eligible & (adjusted >= value)).sum())
                                       for value in thresholds},
        "curve_suppressed_by_threshold": {
            str(value): int((eligible & (probabilities >= value) & (adjusted < value)).sum())
            for value in thresholds},
    }

    selected = sorted(((float(item["second"]), item.get("source_second")) for item in onsets),
                      key=lambda item: item[0])
    intervals = [(left[0], right[0], left[1], right[1])
                 for left, right in zip(selected, selected[1:])
                 if right[0] - left[0] >= gap_threshold_sec]
    if start_sec is not None:
        start_sec = float(start_sec)
        if selected and selected[0][0] - start_sec >= gap_threshold_sec:
            intervals.append((start_sec, selected[0][0], None, selected[0][1]))
        elif not selected and end_sec is not None and float(end_sec) - start_sec >= gap_threshold_sec:
            intervals.append((start_sec, float(end_sec), None, None))
    if end_sec is not None and selected:
        end_sec = float(end_sec)
        if end_sec - selected[-1][0] >= gap_threshold_sec:
            intervals.append((selected[-1][0], end_sec, selected[-1][1], None))
    intervals.sort(key=lambda item: (-(item[1] - item[0]), item[0]))
    rows = []
    for start, end, left_source, right_source in intervals[:max_gaps]:
        frame = (times >= start) & (times < end)
        # The final beat may be snapped a few milliseconds away from its source
        # peak. Exclude each selected boundary's original frame from gap counts.
        for source_second in (left_source, right_source):
            if source_second is not None and len(times):
                nearest = int(np.argmin(np.abs(times - float(source_second))))
                if abs(float(times[nearest]) - float(source_second)) <= 0.011:
                    frame[nearest] = False
        frame_peaks = frame & peaks
        frame_eligible = frame & eligible
        raw_counts = {str(value): int((frame_eligible & (probabilities >= value)).sum())
                      for value in thresholds}
        adjusted_counts = {str(value): int((frame_eligible & (adjusted >= value)).sum())
                           for value in thresholds}
        rest_overlap = sum(max(0.0, min(end, float(region["end_sec"])) -
                               max(start, float(region["start_sec"])))
                           for region in rest_regions)
        audible_fraction = float(audible[frame].mean()) if frame.any() else None
        if rest_overlap >= end - start - 1e-6:
            classification = "declared_rest"
        elif audible_fraction is not None and audible_fraction < 0.5:
            classification = "low_audio_activity"
        elif raw_counts[str(float(threshold))] == 0:
            classification = "planner_below_threshold"
        elif adjusted_counts[str(float(threshold))] == 0:
            classification = "curve_suppressed"
        else:
            classification = "post_threshold_suppressed"

        def extrema(values, selected_mask):
            sample = values[selected_mask]
            return ({"min": float(sample.min()), "mean": float(sample.mean()),
                     "max": float(sample.max())} if len(sample) else None)

        intensity = extrema(curve_intensity, frame)
        factors = extrema(curve_factors, frame)
        raw_peaks = probabilities[frame_peaks]
        adjusted_peaks = adjusted[frame_peaks]
        rows.append({
            "start_sec": float(start), "end_sec": float(end), "duration_sec": float(end - start),
            "raw_peak_count": int(frame_peaks.sum()),
            "audible_peak_count": int((frame & peaks & audible).sum()),
            "raw_max": float(raw_peaks.max()) if len(raw_peaks) else None,
            "adjusted_max": float(adjusted_peaks.max()) if len(adjusted_peaks) else None,
            "curve_intensity_min": None if intensity is None or not math.isfinite(intensity["min"]) else intensity["min"],
            "curve_intensity_mean": None if intensity is None or not math.isfinite(intensity["mean"]) else intensity["mean"],
            "curve_intensity_max": None if intensity is None or not math.isfinite(intensity["max"]) else intensity["max"],
            "curve_factor_min": float(factors["min"]) if factors else None,
            "curve_factor_max": float(factors["max"]) if factors else None,
            "rest_overlap_sec": float(rest_overlap), "audible_fraction": audible_fraction,
            "raw_pass_by_threshold": raw_counts,
            "adjusted_pass_by_threshold": adjusted_counts,
            "classification": classification,
        })
    return {"schema": "song_form_onset_diagnostics_v1",
            "gap_threshold_sec": float(gap_threshold_sec), "max_gap_records": int(max_gaps),
            "gap_count": len(intervals), "gaps_truncated": len(intervals) > max_gaps,
            "thresholds": thresholds, "totals": totals, "gaps": rows}


def merge_close_onsets(records, window_ms):
    """Keep the strongest event within a time radius; never average its beat.

    Work on the final tempo-integrated positions so rational snapping cannot
    reintroduce close pairs. Greedy suppression avoids transitively combining
    an arbitrarily long run of genuine rapid notes into one event.
    """
    window_ms = float(window_ms)
    if not math.isfinite(window_ms) or not 0 <= window_ms <= 250:
        raise ValueError('onset_merge_ms must be in [0, 250]')
    if window_ms == 0:
        return sorted(records, key=lambda item: Fraction(item['beat']))
    radius = window_ms / 1000
    ranked = sorted(records, key=lambda item: (-float(item['score']), float(item['second']),
                                               Fraction(item['beat'])))
    selected, times = [], []
    for item in ranked:
        stamp = float(item['second'])
        index = bisect_left(times, stamp)
        left = index and stamp - times[index-1] < radius - 1e-9
        right = index < len(times) and times[index] - stamp < radius - 1e-9
        if left or right:
            continue
        times.insert(index, stamp)
        selected.insert(index, item)
    return selected


def interval_statistics(records):
    times = sorted(float(item['second']) for item in records)
    intervals = np.diff(times)
    return {'count': len(records),
            'minimum_interval_ms': float(intervals.min()*1000) if len(intervals) else None,
            'pairs_below_30ms': int(np.sum(intervals < .03-1e-9)),
            'pairs_below_50ms': int(np.sum(intervals < .05-1e-9))}
