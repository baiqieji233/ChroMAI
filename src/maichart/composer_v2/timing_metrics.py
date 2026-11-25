"""Offline authored-onset comparisons, never labels or generation constraints.

Phase families describe landing positions, not rhythmic style. Acoustic evidence
is conditional on authored selections: uncharted acoustic peaks are not misses.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from fractions import Fraction
import math
import weakref

import numpy as np

SCHEMA = 'composer_v2_timing_metrics_v1'
_FLUX_CACHE = OrderedDict()
_FAMILIES = ('shared', 'binary', 'ternary', 'other')


def _finite_json(value):
    """Extreme but finite inputs must not leak NaN/Infinity into artifacts."""
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def phase_family(value: Fraction) -> str:
    """Classify exact evidence; integer/half beats are shared by both grids.

    Ternary includes powers of three (e.g. denominator 9 or 27), and is
    deliberately broader than the 24th-note/sixth-beat grid.
    """
    denominator = Fraction(value).denominator
    if denominator <= 2:
        return 'shared'
    has_three = denominator % 3 == 0
    for factor in (2, 3):
        while denominator % factor == 0:
            denominator //= factor
    if denominator != 1:
        return 'other'
    return 'ternary' if has_three else 'binary'


def _beats(points):
    return sorted({Fraction(getattr(point, 'beat', point)) for point in points})


def _counts(values):
    counts = Counter(phase_family(value) for value in values)
    return {family: counts[family] for family in _FAMILIES}


def rhythm_profile(points, *, window_beats=2):
    width = Fraction(window_beats)
    if width <= 0:
        raise ValueError('window_beats must be positive')
    beats = _beats(points)
    windows = {}
    for beat in beats:
        windows.setdefault(int(beat // width), []).append(beat)
    descriptions = []
    for index, values in sorted(windows.items()):
        counts = _counts(values)
        quarter = any((4 * beat).denominator == 1 and (2 * beat).denominator != 1 for beat in values)
        sixth = any((6 * beat).denominator == 1 and (2 * beat).denominator != 1 for beat in values)
        descriptions.append(dict(start_beat=str(index * width),
            end_beat=str((index + 1) * width), phase_counts=counts,
            mixed_16_24=bool(quarter and sixth),
            mixed_binary_ternary=bool(counts['binary'] and counts['ternary'])))
    evidence = [phase_family(beat) for beat in beats
                if phase_family(beat) in ('binary', 'ternary')]
    return dict(unique_onsets=len(beats), phase_counts=_counts(beats),
        interval_counts=_counts(b - a for a, b in zip(beats, beats[1:])),
        window_beats=str(width), windows=descriptions,
        mixed_16_24_windows=sum(row['mixed_16_24'] for row in descriptions),
        binary_ternary_switches=sum(a != b for a, b in zip(evidence, evidence[1:])),
        interpretation='Landing/interval evidence only; ternary includes powers of three, not only sixth-beat/24th-note positions. mixed_16_24 requires exclusive quarter-beat and sixth-beat evidence. Mixtures, switches and rests are descriptive, not penalties.')


def _head_entries(points, clock):
    """Retain authored atoms/multiplicity, separate from unique-onset counts.

    Bare-beat inputs have no note content and cannot establish head counts.
    Slash atoms follow the existing codec contract; slide paths joined by *
    belong to their atom rather than becoming additional note heads.
    """
    from .codec import explicit_note_atoms
    entries = []
    for point in points:
        if not hasattr(point, 'note_text'):
            return None
        entries.append((float(clock.at(point.beat)), len(explicit_note_atoms(point.note_text))))
    return entries


def _head_summary(reference, generated, duration, start=None, end=None, final=False):
    def count(entries):
        if entries is None:
            return None
        return sum(heads for time, heads in entries if start is None or
                   (time >= start and (time <= end if final else time < end)))
    ref, gen = count(reference), count(generated)
    return dict(reference_note_heads=ref, generated_note_heads=gen,
        note_head_count_ratio=gen / ref if ref and gen is not None else None,
        reference_note_head_rate=ref / duration if ref is not None and duration else None,
        generated_note_head_rate=gen / duration if gen is not None and duration else None,
        unit='authored slash atoms via event_set_codec._atoms; unavailable for bare-beat inputs')


def _times(beats, clock):
    result = np.asarray([clock.at(beat) for beat in beats], dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError('Chart times must be finite')
    return np.unique(result)


def _match(reference, generated, tolerance):
    # Earliest feasible pairing maximizes cardinality on sorted time lines.
    i = j = 0
    pairs = []
    while i < len(reference) and j < len(generated):
        delta = float(generated[j] - reference[i])
        if abs(delta) <= tolerance + 1e-12:
            pairs.append((i, j, delta))
            i += 1
            j += 1
        elif delta < 0:
            j += 1
        else:
            i += 1
    return pairs


def _score(reference, generated, tolerance):
    pairs = _match(reference, generated, tolerance)
    matched = len(pairs)
    errors = [pair[2] * 1000 for pair in pairs]
    return dict(reference_onsets=len(reference), generated_onsets=len(generated),
        matched_onsets=matched,
        precision=matched / len(generated) if len(generated) else None,
        recall=matched / len(reference) if len(reference) else None,
        f1=2 * matched / (len(reference) + len(generated)) if len(reference) + len(generated) else None,
        mean_signed_error_ms=float(np.mean(errors)) if errors else None,
        mean_absolute_error_ms=float(np.mean(np.abs(errors))) if errors else None,
        max_absolute_error_ms=max(map(abs, errors)) if errors else None)


def _gaps(times, duration):
    inside = times[(times >= 0) & (times <= duration)]
    edges = np.r_[0., inside, duration]
    gaps = np.diff(edges)
    return dict(empty=not len(inside), leading_blank_sec=float(edges[1]) if len(inside) else duration,
        trailing_blank_sec=float(duration - inside[-1]) if len(inside) else duration,
        max_onset_gap_sec=float(np.max(np.diff(inside))) if len(inside) > 1 else None,
        long_gaps=[dict(start_sec=float(a), end_sec=float(b), duration_sec=float(b-a))
                   for a, b, gap in zip(edges, edges[1:], gaps) if gap >= 2.],
        long_gap_threshold_sec=2., interpretation='Descriptive silence/onset spacing; long rests are not errors.')


def _section(reference, generated, start, end, *, final=False):
    def select(times):
        return times[(times >= start) & ((times <= end) if final else (times < end))]
    ref, gen = select(reference), select(generated)
    length = end - start
    return dict(start_sec=start, end_sec=end,
        matching={name: _score(ref, gen, tolerance) for name, tolerance in (('10ms', .010), ('25ms', .025))},
        reference_event_rate=len(ref) / length if length else None,
        generated_event_rate=len(gen) / length if length else None,
        event_count_ratio=len(gen) / len(ref) if len(ref) else None)


def _audio_evidence(audio, reference, generated):
    key = id(audio)
    cached = _FLUX_CACHE.get(key)
    if cached is not None and cached[0]() is audio:
        times, flux = cached[1:]
        _FLUX_CACHE.move_to_end(key)
    else:
        times = np.asarray(audio.mel_times_sec, dtype=np.float64)
        flux = np.asarray(audio.mel[:, 96], dtype=np.float64)
        valid = np.isfinite(times) & np.isfinite(flux)
        times, flux = times[valid], np.maximum(flux[valid], 0.)
        order = np.argsort(times, kind='stable')
        times, unique = np.unique(times[order], return_index=True)
        flux = flux[order][unique]
        try:
            _FLUX_CACHE[key] = (weakref.ref(audio), times, flux)
            while len(_FLUX_CACHE) > 4:
                _FLUX_CACHE.popitem(last=False)
        except TypeError:
            pass
    queries = np.r_[reference, generated]
    support = np.interp(queries, times, flux, left=np.nan, right=np.nan) if len(times) else np.full(len(queries), np.nan)
    ref_support, gen_support = support[:len(reference)], support[len(reference):]
    authored = ref_support[np.isfinite(ref_support)]
    threshold = float(np.median(authored)) if len(authored) else None
    strong = np.flatnonzero(np.isfinite(ref_support) & (ref_support > 0) &
                            (ref_support >= threshold)) if threshold is not None else []
    matched_ref = {i for i, _, _ in _match(reference, generated, .025)}
    covered = sum(int(i) in matched_ref for i in strong)
    def summary(values):
        finite = values[np.isfinite(values)]
        return dict(onsets=len(values), covered_onsets=len(finite),
                    mean_flux=float(np.mean(finite)) if len(finite) else None)
    return dict(selected_attack_support=dict(reference=summary(ref_support), generated=summary(gen_support)),
        authored_strong_attack_recall=dict(reference_strong_onsets=len(strong), matched_onsets=covered,
            recall=covered / len(strong) if len(strong) else None, threshold_flux=threshold,
            threshold_basis='positive support at or above median of authored reference onset support', tolerance_sec=.025),
        limitations='Cached mel column 96 proxy, linearly interpolated once; no FFT. No pitch/melody or mandatory peak labels. Weak/missing flux is not an error; uncharted peaks are never misses.')


def compare_charts(reference_points, reference_clock, generated_points, generated_clock,
                   duration_sec, *, audio=None, interval_sec=2.0):
    """Compare unique attack times in absolute audio seconds, including offsets.

    Whole-song matching includes chart onsets outside the audio range; intervals
    cover [0,duration], using half-open bins and an inclusive final endpoint.
    Each interval matches independently, so boundary-near pairs can differ from
    whole-song matching. All rates count unique onsets, not simultaneous heads.
    """
    duration, interval = float(duration_sec), float(interval_sec)
    if not math.isfinite(duration) or duration < 0 or not math.isfinite(interval) or interval <= 0:
        raise ValueError('duration must be finite/nonnegative and interval finite/positive')
    reference_points, generated_points = list(reference_points), list(generated_points)
    ref_beats, gen_beats = _beats(reference_points), _beats(generated_points)
    ref_heads = _head_entries(reference_points, reference_clock)
    gen_heads = _head_entries(generated_points, generated_clock)
    ref, gen = _times(ref_beats, reference_clock), _times(gen_beats, generated_clock)
    whole = _section(ref, gen, 0., duration, final=True)
    whole['matching'] = {name: _score(ref, gen, tolerance) for name, tolerance in (('10ms', .01), ('25ms', .025))}
    whole.update(reference_event_rate=len(ref) / duration if duration else None,
                 generated_event_rate=len(gen) / duration if duration else None,
                 event_count_ratio=len(gen) / len(ref) if len(ref) else None)
    whole['note_heads'] = _head_summary(ref_heads, gen_heads, duration)
    intervals = [_section(ref, gen, float(start), min(float(start + interval), duration),
                          final=start + interval >= duration) for start in np.arange(0., duration, interval)]
    for section in intervals:
        section['note_heads'] = _head_summary(ref_heads, gen_heads,
            section['end_sec'] - section['start_sec'], section['start_sec'], section['end_sec'],
            final=section['end_sec'] == duration)
    result = dict(schema=SCHEMA, onset_unit='unique onset; simultaneous notes count once',
        whole=whole, intervals=intervals,
        reference_spacing=_gaps(ref, duration), generated_spacing=_gaps(gen, duration),
        rhythm_profile=dict(reference=rhythm_profile(ref_beats), generated=rhythm_profile(gen_beats)),
        limitations='Reference similarity is not musical quality. Authored omissions, repeats, mixtures and rests remain valid. Event rates/ratios use unique onsets, while note_heads reports authored atom workload. Interval matching is independent per bin: cross-boundary pairs may be lost and interval matches need not sum to whole-song matches.')
    result['audio_evidence'] = _audio_evidence(audio, ref, gen) if audio is not None else None
    return _finite_json(result)
