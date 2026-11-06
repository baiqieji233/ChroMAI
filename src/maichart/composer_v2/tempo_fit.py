"""Robust clock fitting on detector peaks, independent of notes and model weights.

Peak density can change without a tempo change. Search a whole-song pulse
consensus first, then jointly refine phase and period with fresh beat indices.
Local estimates are diagnostics, not automatically authored tempo directives.
"""
from __future__ import annotations

import math
import numpy as np

FIT_POLICY = 'whole_song_pulse_consensus_v2'


def _wrapped(times, period, first):
    return (times - first + period / 2.) % period - period / 2.


def _phase_consensus(times, period):
    # A bounded kernel prefers the dominant pulse, without letting subdivision
    # peaks pull the median halfway between two legitimate phase clusters.
    sigma = min(.020, period * .065)
    phases = np.arange(512, dtype=np.float64) * period / 512.
    scores = []
    for chunk in np.array_split(phases, 8):
        errors = _wrapped(times[None, :], period, chunk[:, None])
        scores.extend(np.exp(-.5 * (errors / sigma)**2).sum(axis=1))
    phase = float(phases[int(np.argmax(scores))])
    for _ in range(5):
        errors = _wrapped(times, period, phase)
        weight = np.exp(-.5 * (errors / sigma)**2)
        phase = (phase + float(np.sum(errors * weight) / np.sum(weight))) % period
    return phase


def _refine(times, period):
    phase = _phase_consensus(times, period)
    tolerance = min(.050, period * .18)
    for _ in range(8):
        # Reassign on EVERY iteration; freezing these integers locks the fit to
        # the initial local run when its period was slightly biased.
        index = np.rint((times - phase) / period)
        residual = times - (phase + index * period)
        keep = np.abs(residual) <= tolerance
        if int(keep.sum()) < 8:
            return None
        x = index[keep]
        y = times[keep]
        weight = np.minimum(1., min(.020, period * .08) / np.maximum(np.abs(residual[keep]), 1e-9))
        mean_x = float(np.average(x, weights=weight))
        mean_y = float(np.average(y, weights=weight))
        variance = float(np.sum(weight * (x - mean_x)**2))
        if variance <= 0:
            return None
        new_period = float(np.sum(weight * (x - mean_x) * (y - mean_y)) / variance)
        if not .12 <= new_period <= 2.:
            return None
        phase = (mean_y - mean_x * new_period) % new_period
        if abs(new_period - period) < 1e-10:
            period = new_period
            break
        period = new_period
    residual = _wrapped(times, period, phase)
    keep = np.abs(residual) <= min(.050, period * .18)
    if keep.sum() < 8:
        return None
    # Smooth support avoids rewarding a poor fit simply for reaching an
    # arbitrary inlier cutoff. Off-pulse peaks receive negligible influence.
    sigma = min(.025, period * .08)
    support = float(np.exp(-.5 * (residual / sigma)**2).sum())
    return dict(period=period, first=phase, keep=keep, residual=residual, support=support)


def fit_global_clock(times):
    intervals = np.diff(times)
    typical = float(np.median(intervals))
    if not .12 <= typical <= 2.:
        raise ValueError('Detected beat intervals are outside a usable global tempo range')
    span = float(np.ptp(times))
    if span <= typical * 4:
        raise ValueError('Too little audio span for a reliable clock fit')
    # Full-song frequency resolution; cap temporary arrays rather than build
    # an unbounded [all frequencies, all song peaks] matrix.
    low = max(.5, .65 / typical)
    high = min(500. / 60., 1.50 / typical)
    frequencies = np.arange(low, high, .08 / span)
    scores = np.empty(len(frequencies), dtype=np.float64)
    centered = times - times[0]
    for start in range(0, len(frequencies), 128):
        freq = frequencies[start:start+128]
        z = np.exp(2j * np.pi * freq[:, None] * centered)
        # The second harmonic survives a mixture of on-beat and half-beat
        # peaks (the 205 BPM regression case). It does not double the output
        # tempo: the period candidates retain the detector's pulse convention.
        scores[start:start+128] = np.abs(z.mean(axis=1)) + .7 * np.abs((z*z).mean(axis=1))
    maxima = np.flatnonzero((scores[1:-1] > scores[:-2]) & (scores[1:-1] >= scores[2:])) + 1
    seeds = sorted(maxima, key=lambda i: -scores[i])[:8]
    seeds.append(int(np.argmax(scores)))
    fits = [_refine(times, 1. / frequencies[i]) for i in set(seeds)]
    fits = [fit for fit in fits if fit is not None]
    if not fits:
        raise ValueError('No consistent global beat grid')
    best = max(fits, key=lambda fit: fit['support'])
    measured = 60. / best['period']
    precision = 'continuous_fit'
    # Prefer a simple integer ONLY when its accumulated drift is below 5 ms
    # over the entire observed song, and its consensus is no worse. This is a
    # resolution rule, not blanket rounding of real fractional tempos.
    integer = round(measured)
    drift = span * abs(integer / measured - 1.)
    if integer >= 30 and drift <= .005:
        period = 60. / integer
        phase = _phase_consensus(times, period)
        residual = _wrapped(times, period, phase)
        sigma = min(.025, period * .08)
        support = float(np.exp(-.5 * (residual / sigma)**2).sum())
        if support >= best['support'] * .995:
            best = dict(period=period, first=phase, residual=residual,
                keep=np.abs(residual) <= min(.050, period*.18), support=support)
            precision = 'integer_equivalent_within_5ms_song_drift'
    keep = best['keep']
    report = dict(clock_fit_policy=FIT_POLICY, interval_median_sec=typical,
        interval_mad_sec=float(np.median(np.abs(intervals - typical))),
        off_pulse_peak_count=int((~keep).sum()), global_fit_inliers=int(keep.sum()),
        global_fit_coverage=float(keep.mean()),
        global_fit_residual_p90_sec=float(np.quantile(np.abs(best['residual'][keep]), .9)),
        continuous_fit_bpm=measured, bpm_precision_policy=precision,
        fitted_first_sec=float(best['first']))
    return 60. / best['period'], report


def fit_phase(times, clock):
    period = float(clock._scale_arr[0])

    def errors(offset):
        seconds = times - offset
        ix = np.clip(np.searchsorted(clock._seconds_arr, seconds, side='right') - 1, 0, len(clock.beats)-1)
        beat = clock._beat_floats[ix] + (seconds-clock._seconds_arr[ix]) / clock._scale_arr[ix]
        return (beat-np.rint(beat))*clock._scale_arr[ix], clock._scale_arr[ix]

    if len(clock.beats) == 1:
        first = _phase_consensus(times, period)
    else:
        # Manual variable-tempo maps already specify beat coordinates. Search
        # their shared time translation, never fit each section independently.
        phases = np.arange(1024, dtype=np.float64)*period/1024.
        scores = []
        for phase in phases:
            residual, scale = errors(phase)
            scores.append(float(np.exp(-.5*(residual/np.minimum(.020, scale*.065))**2).sum()))
        first = float(phases[int(np.argmax(scores))])
        for _ in range(5):
            residual, scale = errors(first)
            weights = np.exp(-.5*(residual/np.minimum(.020, scale*.065))**2)
            # No modulo during refinement: maps with tempo changes are not
            # periodic under a one-initial-beat translation of every change.
            first = float(np.clip(first+np.average(residual, weights=weights), 0., np.nextafter(period, 0.)))
    residual, scale = errors(first)
    keep = np.abs(residual) <= np.minimum(.050, scale*.18)
    if keep.sum() < 8:
        raise ValueError('Too few coherent beat peaks to estimate first reliably')
    return first, dict(phase_fit_scope='dominant_consistent_pulse',
        phase_fit_inliers=int(keep.sum()), phase_fit_coverage=float(keep.mean()),
        phase_residual_median_sec=float(np.median(np.abs(residual[keep]))),
        phase_residual_p90_sec=float(np.quantile(np.abs(residual[keep]), .9)),
        all_peak_phase_residual_p90_sec=float(np.quantile(np.abs(residual), .9)))


def inspect_tempo_regions(times, global_bpm):
    """Report sustained local pulse changes in seconds, never guessed beat IDs.

    Sparse/doubled detector pulses are normalized to the global pulse octave.
    A candidate requires sustained evidence from overlapping eight-second
    windows; it is a measurement aid, since peaks can lag the real change.
    """
    regions = []
    observations = []
    first_time = float(times[0])
    last_time = float(times[-1])
    for start in np.arange(first_time, last_time-7.99, 4.):
        sample = times[(times >= start) & (times < start+8.)]
        if len(sample) < 8:
            observations.append(None)
            continue
        intervals = np.diff(sample)
        typical = float(np.median(intervals))
        # The small within-window fit only supplies a diagnostic slope. It is
        # never used to seed or bias the global full-song consensus.
        candidate = _refine(sample, typical)
        if candidate is None or candidate['keep'].mean() < .80:
            observations.append(None)
            continue
        p90 = float(np.quantile(np.abs(candidate['residual'][candidate['keep']]), .9))
        if p90 > .035:
            observations.append(None)
            continue
        bpm = 60. / candidate['period']
        raw_bpm = bpm
        octave = min((.5, 1., 2.), key=lambda scale: abs(math.log(bpm*scale/global_bpm)))
        # Fold only a convincing half/double match. Merely selecting the
        # nearest octave can turn a genuine 112 -> 180 change into 224 -> 180.
        if abs(math.log(bpm*octave/global_bpm)) > .03:
            octave = 1.
        bpm *= octave
        observation = dict(start_sec=float(start), end_sec=float(start+8.), bpm=float(bpm),
            detector_pulse_bpm=float(raw_bpm), pulse_octave_adjustment=octave,
            residual_p90_sec=p90)
        observations.append(observation)
        if regions and abs(regions[-1]['end_sec']-float(start+4.)) < 1e-8 and abs(math.log(bpm/regions[-1]['bpm'])) < .02:
            previous = regions[-1]
            count = previous['windows']
            previous['bpm'] = (previous['bpm']*count+bpm)/(count+1)
            previous['end_sec'] = float(start+8.)
            previous['windows'] = count+1
        else:
            regions.append(dict(observation, windows=1))
    sustained = [region for region in regions if region['windows'] >= 2]
    changed = any(abs(math.log(a['bpm']/b['bpm'])) > .035
                  for a, b in zip(sustained, sustained[1:]))
    # An accelerando need not sit at one fixed BPM for two windows. Likewise,
    # a global fit may lock onto the faster of two mutually incompatible pulse
    # grids. Sustained disagreement is a warning, never proof of a tempo map.
    off_global = [window is not None and abs(math.log(window['bpm']/global_bpm)) > .035
                  for window in observations]
    changed = changed or any(a and b for a,b in zip(off_global,off_global[1:]))
    return dict(tempo_analysis_policy='local_pulse_regions_diagnostic_v1',
        local_tempo_regions=sustained, local_tempo_windows=observations,
        variable_tempo_suspected=bool(changed), automatic_tempo_map_applied=False)
