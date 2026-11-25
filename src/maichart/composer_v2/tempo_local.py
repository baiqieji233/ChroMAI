"""Second-pass local tempo estimation for an otherwise constant audio clock."""

import itertools
from fractions import Fraction

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from maichart.event_timing import EventTempo
from maichart.tempo_clock import TempoClock

from .tempo_fit import _wrapped

RATE = 200  # onset frames per second
STEP = 0.4  # seconds between six-second spectral windows
SAMPLE_RATE = 16000


def _mono_blocks(path):
    import soundfile as sf
    import soxr

    with sf.SoundFile(path) as audio:
        resampler = soxr.ResampleStream(audio.samplerate, SAMPLE_RATE, 1)
        while True:
            block = audio.read(65536, dtype="float32", always_2d=True)
            last = len(block) < 65536
            samples = resampler.resample_chunk(block.mean(axis=1), last=last)
            if len(samples):
                yield samples
            if last:
                break


def _band_flux(blocks):
    from itertools import chain

    window_size, hop = 512, SAMPLE_RATE // RATE
    freq = np.fft.rfftfreq(window_size, 1 / SAMPLE_RATE)
    cuts = freq.searchsorted([25, 180, 650, 2300, 7800])
    window = np.hanning(window_size).astype("f")
    padding = np.zeros(window_size // 2, dtype="f")
    pending = padding
    previous = None
    bands = []
    for block in chain(blocks, (padding,)):
        pending = np.concatenate((pending, block))
        if len(pending) < window_size:
            continue
        frames = np.lib.stride_tricks.sliding_window_view(pending, window_size)[::hop]
        for start in range(0, len(frames), 192):
            spectrum = np.abs(np.fft.rfft(frames[start : start + 192] * window)).astype("f")
            spectrum = np.log1p(6 * spectrum)
            history = spectrum[:1].repeat(2, axis=0) if previous is None else previous
            lagged = np.concatenate((history, spectrum))
            diff = np.maximum(0, lagged[2:] - lagged[:-2])
            bands.append(np.stack([diff[:, a:b].mean(axis=1) for a, b in itertools.pairwise(cuts)]))
            previous = lagged[-2:]
        pending = pending[len(frames) * hop :]
    flux = np.concatenate(bands, axis=1)
    if flux.shape[1] > 1:
        flux[:, :-1] = flux[:, 1:]
    norm = np.maximum(
        gaussian_filter1d(flux, 180, axis=1), 0.20 * flux.mean(axis=1, keepdims=True) + 1e-5
    )
    return np.minimum(flux / norm, 7)


def _onset_bands(path):
    try:
        return _band_flux(_mono_blocks(path))
    except (OSError, RuntimeError):
        # Restart from the beginning if libsndfile rejects the format midstream.
        import soxr

        from maichart.form.rational_audio import _read_mono_fallback

        samples, rate, _ = _read_mono_fallback(path)
        return _band_flux((soxr.resample(samples, rate, SAMPLE_RATE),))


def _periodicity(flux, beats, base):
    times = np.arange(STEP / 2, (flux.shape[1] - 1) / RATE, STEP)
    bpms = np.unique(np.r_[np.arange(40.0, 300.01, 0.5), base])
    beat_hz = bpms / 60
    half = 3 * RATE
    window = np.hanning(2 * half)
    fft_size = 16384
    hz = np.fft.rfftfreq(fft_size, 1 / RATE)
    padded = np.pad(flux, ((0, 0), (half, half)))
    impulses = np.zeros(flux.shape[1])
    np.add.at(impulses, np.rint(beats * RATE).astype(int).clip(0, len(impulses) - 1), 1)
    impulses = np.pad(impulses, (half, half))
    intervals = np.diff(beats)
    midpoints = (beats[1:] + beats[:-1]) / 2
    shape = (len(times), len(bpms))
    acoustic, pulse, spacing = (np.zeros(shape, dtype=np.float32) for _ in range(3))
    for i, time in enumerate(times):
        start = round(time * RATE)
        local = padded[:, start : start + 2 * half] * window
        spectrum = abs(np.fft.rfft(local, fft_size, axis=1)) / (
            local.sum(axis=1, keepdims=True) + 1e-5
        )
        spectrum = np.minimum(spectrum, 0.85).mean(axis=0)
        powers = sum(
            w * np.interp(h * beat_hz, hz, spectrum)
            for h, w in ((1, 0.42), (2, 0.72), (3, 0.16), (4, 0.60), (8, 0.10))
        )
        centre = np.median(powers)
        acoustic[i] = (powers - centre) / max(0.25, powers.max() - centre)
        local = impulses[start : start + 2 * half] * window
        if local.sum() >= 2:
            pulse[i] = np.interp(beat_hz, hz, abs(np.fft.rfft(local, fft_size))) / (
                local.sum() + 1e-5
            )
        nearby = intervals[midpoints.searchsorted(time - 3) : midpoints.searchsorted(time + 3)]
        if len(nearby):
            errors = np.log(np.maximum(nearby, 1e-4)[:, None] * beat_hz)
            spacing[i] = np.exp(-0.5 * (errors / 0.095) ** 2).mean(axis=0)
    return times, bpms, (acoustic, pulse, spacing)


def _segments(scores, rates, base_id):
    """Constant switch penalty; backtrack the best path across tempo families."""
    values = scores[0].astype(float)
    keep = np.arange(len(rates), dtype=np.int32)
    previous = np.empty((len(scores) - 1, len(rates)), dtype=np.int32)
    for i, row in enumerate(scores[1:]):
        best = values.argmax()
        floor = values[best] - 2.7
        previous[i] = np.where(values < floor, best, keep)
        values = np.maximum(values, floor) + row
    path = np.empty(len(scores), dtype=np.int32)
    path[-1] = values.argmax()
    for i in range(len(previous) - 1, -1, -1):
        path[i] = previous[i, path[i + 1]]
    edges = np.r_[0, np.flatnonzero(np.diff(path)) + 1, len(path)]
    merged = []
    for left, right in itertools.pairwise(edges):
        k = path[left]
        if abs(np.log(rates[k] / rates[base_id])) < 0.019:
            k = base_id
        if merged and abs(np.log(rates[merged[-1][2]] / rates[k])) < 0.027:
            start, _, old = merged[-1]
            k = base_id if base_id in (old, k) else (old if left - start >= right - left else k)
            merged[-1] = (start, right, k)
        else:
            merged.append((left, right, k))
    return merged


def _find_plateaus(flux, beats, base):
    times, bpms, features = _periodicity(flux, beats, base)
    rates = bpms[(bpms >= base / np.sqrt(2)) & (bpms < base * np.sqrt(2))]
    base_id = np.argmin(abs(rates - base))
    variants = rates[:, None] * np.array([0.5, 1.0, 2.0])
    pos = bpms.searchsorted(variants).clip(1, len(bpms) - 1)
    nearest = np.where(variants - bpms[pos - 1] <= bpms[pos] - variants, pos - 1, pos)
    valid = abs(bpms[nearest] / variants - 1) < 0.007
    families = [
        np.maximum.reduce(
            [np.where(valid[:, k], feature[:, nearest[:, k]], -2.0) for k in range(3)]
        )
        for feature in features
    ]
    evidence = 0.92 * families[0] + 0.24 * families[1] + 0.30 * families[2]
    energy = [flux[:, max(0, int((t - 1.2) * RATE)) : int((t + 1.2) * RATE)].mean() for t in times]
    active = (np.asarray(energy) > 0.14) & (
        beats.searchsorted(times + 2) - beats.searchsorted(times - 2) >= 3
    )
    scores = (evidence - evidence[:, base_id, None]) * active[:, None] - 0.085 * (
        abs(np.log(rates / base)) > 0.016
    )
    plateaus = []
    for left, right, k in _segments(scores, rates, base_id):
        use = np.flatnonzero(active[left:right]) + left
        if k != base_id and len(use) * STEP >= 3.6:
            acoustic, pulse, spacing = ((f[use, k] - f[use, base_id]).mean() for f in families)
            if not (acoustic > 0.105 and (pulse > 0.12 or spacing > 0.15 or acoustic > 0.42)):
                k = base_id
        else:
            k = base_id
        if plateaus and plateaus[-1][2] == k:
            plateaus[-1][1] = right
        else:
            plateaus.append([left, right, k])
    if len(plateaus) > 1 and (plateaus[0][1] - plateaus[0][0]) * STEP < 5:
        plateaus.pop(0)
        plateaus[0][0] = 0
    if len(plateaus) > 1 and (plateaus[-1][1] - plateaus[-1][0]) * STEP < 5:
        plateaus.pop()
        plateaus[-1][1] = len(times)
    if not 2 <= len(plateaus) <= 10 or not any(k == base_id for _, _, k in plateaus):
        return []

    # Select the absolute metrical level after locating changes in the ratio.
    levels, lengths = [], []
    for left, right, k in plateaus:
        use = np.flatnonzero(active[left:right]) + left
        lengths.append(max(1, len(use)))
        level = np.zeros(3)
        if len(use):
            level = sum(
                weight * f[np.ix_(use, nearest[k])].mean(0)
                for weight, f in zip((1.0, 0.08, 0.33), features)
            )
        levels.append(np.where(valid[k], level, -10.0))
    common = np.average(levels, axis=0, weights=lengths)
    common[1] += 0.065
    shared = common.argmax()
    out = []
    for (left, right, k), level in zip(plateaus, levels):
        choice = level.argmax()
        if valid[k, shared] and level[choice] - level[shared] < 0.23:
            choice = shared
        end = right * STEP if right < len(times) else (flux.shape[1] - 1) / RATE
        out.append((left * STEP, end, variants[k, choice], k == base_id and choice == 1))
    return out


def _refine(rate, left, right, beats, events, weights, fixed):
    margin = min(0.7, (right - left) * 0.1)
    mask = (events >= left + margin) & (events < right - margin)
    events, weights = events[mask], weights[mask]
    beats = beats[(beats >= left + margin) & (beats < right - margin)]
    centre = (left + right) / 2

    def coherence(rates, data, weight, harmonic):
        total = np.zeros(len(rates), dtype=complex)
        for i in range(0, len(data), 256):
            phase = (2j * np.pi * harmonic / 60) * rates[:, None] * (data[i : i + 256] - centre)
            total += np.einsum("ij,j->i", np.exp(phase), weight[i : i + 256])
        return abs(total) / (weight.sum() + 1e-8)

    def score(rates):
        return 0.88 * coherence(rates, beats, np.ones(len(beats)), 1) + sum(
            w * coherence(rates, events, weights, h) for h, w in ((1, 0.25), (2, 0.65), (4, 0.55))
        )

    if not fixed:
        rates = np.linspace(rate * 0.981, rate * 1.019, 171)
        best = rates[score(rates).argmax()]
        step = rates[1] - rates[0]
        rates = np.linspace(best - step, best + step, 13)
        rate = rates[score(rates).argmax()]
        if abs(rate - round(rate)) < 0.065 and (right - left) * abs(round(rate) / rate - 1) < 0.025:
            rate = round(rate)
    period = 60 / rate
    phases = np.linspace(0, period, max(40, round(period * 700)), endpoint=False)

    def grid(data, weight, period, sigma):
        total = np.zeros(len(phases))
        for i in range(0, len(data), 256):
            residual = _wrapped(data[i : i + 256], period, phases[:, None])
            total += np.einsum(
                "ij,j->i", np.exp(-0.5 * (residual / sigma) ** 2), weight[i : i + 256]
            )
        return total / (weight.sum() + 1e-8)

    support = (
        1.15 * grid(beats, np.ones(len(beats)), period, 0.022)
        + 0.5 * grid(events, weights, period, 0.018)
        + 0.35 * grid(events, weights, period / 2, 0.015)
    )
    return float(rate), float(phases[support.argmax()])


def _junction(coarse, left_rate, intercept, right_rate, phase, last_beat, events, weights, beats):
    old, new = 60 / left_rate, 60 / right_rate
    lo, hi = max(0, coarse - 4.5), coarse + 4.5
    first_tick = max(int(np.ceil((lo - intercept) / old * 48)), int(last_beat * 48) + 1)
    ticks = np.arange(first_tick, int(np.floor((hi - intercept) / old * 48)) + 1)
    if not len(ticks):
        return None
    positions = ticks / 48
    times = intercept + positions * old
    bins = np.arange(lo - 0.11, hi + 0.121, 0.025)

    def likelihood(period, origin):
        def evidence(data, div, sigma):
            residual = _wrapped(data, period / div, origin)
            return np.exp(-0.5 * (residual / sigma) ** 2) - min(0.9, 2.5066 * sigma * div / period)

        onset = sum(
            w * evidence(events, div, sigma)
            for w, div, sigma in ((0.43, 1, 0.023), (0.62, 2, 0.019), (0.33, 4, 0.013))
        )
        return (
            np.histogram(events, bins, weights=weights * onset)[0]
            + np.histogram(beats, bins, weights=0.85 * evidence(beats, 1, 0.025))[0]
        )

    prefix = np.r_[0, np.cumsum(likelihood(old, intercept) - likelihood(new, phase))]
    index = bins.searchsorted(times).clip(1, len(bins) - 1) - 1
    errors = abs(_wrapped(times - positions * new, new, phase))
    score = prefix[index] - 11 * (errors / 0.045) ** 2 - 0.6 * abs(times - coarse)
    if (errors <= 0.025).any():
        score[errors > 0.025] = -np.inf
    k = score.argmax()
    return Fraction(int(ticks[k]), 48), float(times[k]), float(errors[k])


def fit_local_clock(path, beats, base, first, *, first_override=None):
    """Return a continuous Simai clock, or leave the first pass in control."""
    report = {"policy": "local_octave_families_v1", "decision": "insufficient_local_support"}
    beats = np.asarray(beats, dtype=float)
    if len(beats) < 8 or beats[-1] - beats[0] < 8 or not np.isfinite(base) or base <= 0:
        return None, None, report
    flux = _onset_bands(path)
    if flux.shape[1] / RATE < 8:
        return None, None, report
    plateaus = _find_plateaus(flux, beats, base)
    if not plateaus:
        return None, None, report
    onset = flux.mean(axis=0)
    peaks = find_peaks(onset, distance=4, prominence=0.18)[0]
    peaks = peaks[onset[peaks] > 0.65]
    events, weights = peaks / RATE, np.minimum(onset[peaks], 5.0) ** 1.4
    fitted = []
    # Reserve one pulse in four for the final comparison with the original
    # clock. These pulses participate in candidate discovery, not fine fitting.
    reservation = np.arange(len(beats)) % 4 == 3
    fit_beats, probes = beats[~reservation], beats[reservation]
    for left, right, rate, fixed in plateaus:
        rate, phase = _refine(rate, left, right, fit_beats, events, weights, fixed)
        if fixed and abs(_wrapped(phase, 60 / rate, first)) < 0.018:
            phase = first % (60 / rate)
        fitted.append((left, rate, phase))
    _, rate, origin = fitted[0]
    if first_override is not None:
        if abs(_wrapped(origin, 60 / rate, first_override)) > 0.035:
            report["decision"] = "manual_first_inconsistent"
            return None, None, report
        origin = first_override
    elif rate == base and abs(_wrapped(origin, 60 / rate, first)) < 0.018:
        origin = first
    changes = [EventTempo(Fraction(0), rate)]
    seconds = origin
    bounds = []
    for coarse, rate, phase in fitted[1:]:
        last = changes[-1]
        intercept = seconds - float(last.beat) * 60 / last.bpm
        i = beats.searchsorted(coarse)
        gap = None
        if 0 < i < len(beats) and beats[i] - beats[i - 1] > 2.3:
            gap = [float(beats[i - 1]), float(beats[i])]
            coarse = gap[1]
        found = _junction(
            coarse, last.bpm, intercept, rate, phase, last.beat, events, weights, fit_beats
        )
        if gap and found is not None:
            # A short bridge can rejoin both grids across an unobserved interval.
            start = round((gap[0] - intercept) * last.bpm / 60)
            t0 = intercept + start * 60 / last.bpm
            t1 = phase + round((gap[1] - phase) * rate / 60) * 60 / rate
            count = max(1, round((t1 - t0) * last.bpm / 60))
            bridge = 60 * count / (t1 - t0) if t1 > t0 else 0.0
            if start > last.beat and abs(bridge / last.bpm - 1) < 0.04:
                if abs(bridge / last.bpm - 1) > 1e-6:
                    changes.append(EventTempo(Fraction(start), bridge))
                found = Fraction(start + count), t1, 0.0
        if found is None or found[2] > 0.035:
            report["decision"] = "ambiguous_junction"
            return None, None, report
        position, seconds, error = found
        changes.append(EventTempo(position, rate))
        bounds.append(
            {"beat": str(position), "seconds": seconds, "phase_error_sec": error, "gap_sec": gap}
        )
    clock = TempoClock(changes, origin)
    checks = []
    for i, change in enumerate(changes):
        if abs(change.bpm / base - 1) < 0.027:
            continue
        lo = clock.seconds[i]
        hi = clock.seconds[i + 1] if i + 1 < len(changes) else flux.shape[1] / RATE
        sample = probes[(probes >= lo + 0.2) & (probes < hi - 0.2)]
        if len(sample) < 3:
            # A bridge has no observed beats by construction. Its neighbouring
            # musical segments carry the evidence for accepting the map.
            if any(
                b["gap_sec"] and lo >= b["gap_sec"][0] - 0.5 and hi <= b["gap_sec"][1] + 0.5
                for b in bounds
            ):
                continue
            report.update(decision="insufficient_validation_pulses", validation=checks)
            return None, None, report
        intercept = lo - float(change.beat) * 60 / change.bpm
        previous = float(np.mean(abs(_wrapped(sample, 60 / base, first)) <= 0.05))
        proposed = float(np.mean(abs(_wrapped(sample, 60 / change.bpm, intercept)) <= 0.05))
        checks.append(
            {
                "start_sec": lo,
                "end_sec": hi,
                "pulses": len(sample),
                "base_coverage": previous,
                "local_coverage": proposed,
            }
        )
    report["validation"] = checks
    if not checks or any(
        c["local_coverage"] < 0.55 or c["local_coverage"] - c["base_coverage"] < 0.15
        for c in checks
    ):
        report["decision"] = "no_validation_gain"
        return None, None, report
    report.update(decision="accepted", boundaries=bounds)
    return changes, float(origin), report
