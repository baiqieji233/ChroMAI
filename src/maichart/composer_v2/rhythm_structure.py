"""Beat-domain rhythm structure: sub-beat phase geometry and aligned, trusted references.

Everything here lives on the chart's own (possibly variable) tempo clock. No meter,
bar length or compulsory grid is assumed. Phases remain exact rationals in the
token language; the periodic features only expose which subdivision families a
position belongs to: cos(2*pi*n*phase) == 1 exactly when the phase lies on the
1/n grid, so 1/4 (16th) and 1/3 (24th family) become directly separable.

The aligned reference replaces the old "mean of 16 beats, argmax, always shown"
retrieval. It compares the upcoming music beat by beat (song-centred MERT plus a
within-beat onset pattern), searches every integer-beat shift, reports the score
as an explicit trust signal and shows nothing when no earlier passage matches.
The reference events are displayed on the target timeline (shifted by the found
integer offset), so the model compares a candidate continuation with the very
audio it would be written to instead of inferring an offset from absolute beats.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import OrderedDict
from fractions import Fraction
import math

import numpy as np
import torch

LEGACY_POLICY = 'aligned_rhythm_v1'
POLICY = 'aligned_rhythm_v2'
POLICIES = ('off', LEGACY_POLICY, POLICY)
CHOICES = ('auto', *POLICIES)
EVIDENCE_FEATURES = 8

HARMONICS = (1, 2, 3, 4, 6, 8, 12, 16, 24)
PHASE_FEATURES = 2 * len(HARMONICS)

# Reference geometry, all in beats of the chart clock.
REF_LEAD_BEATS = 4          # already-written beats shown for correspondence
REF_MATCH_AHEAD_BEATS = 8   # upcoming beats that decide the match
REF_MAX_LEAD_EVENTS = 8
REF_MIN_VALID_BEATS = 4
REF_SAMPLES_PER_BEAT = 8
REF_ONSET_BINS = 12         # lcm(4, 3): separates 16th and triplet patterns
REF_ONSET_SAMPLES = 4       # samples per onset bin for the generic audio path
REF_SEMANTIC_DIM = 256
REF_SEMANTIC_WEIGHT = .7
REF_PROFILE_CACHE = 48
REF_MIN_SCORE = .35         # calibrated on authored charts; see docs
REF_TRUST_BINS = 8
TRUST_FEATURES = 2 + 1 + REF_TRUST_BINS   # present, score, below-gate, bins
REF_DROP_PROBABILITY = .1
REF_ALTERNATIVE_PROBABILITY = .15

ENVELOPE_FEATURES = 2
ENVELOPE_HALF_WIDTH = 1 / 48.  # half of the 1/24-beat bank spacing
_FLUX, _ENERGY = 96, 97
STATE_PREFIXES = ('phase_position.', 'audio_phase_position.', 'phase_token.',
                  'audio_envelope.', 'reference_trust.', 'segment.', 'reference_evidence.')


def resolve_policy(requested, parent=None, *, resume=False):
    """auto: fresh and continuous finetune use the latest; resume inherits."""
    previous = (parent or {}).get('structure_policy', 'off')
    continuous = parent is None or parent.get('sequence_policy') == 'continuous_events_v1'
    latest = POLICY if continuous else previous
    policy = (previous if resume else latest) if requested in (None, 'auto') else requested
    if policy not in POLICIES:
        raise ValueError('Unknown rhythm structure policy')
    if policy != 'off' and not continuous:
        raise ValueError('Rhythm structure requires continuous event checkpoints')
    if resume and policy != previous:
        raise ValueError('Changing rhythm structure requires finetune in a new directory')
    return policy, parent is not None and policy != previous


# ---------------------------------------------------------------------------
# Sub-beat phase geometry
# ---------------------------------------------------------------------------

def phase_features(beats, harmonics=None):
    """[..., 18] periodic features of continuous beat positions (torch)."""
    beats = beats.float()
    fraction = beats - torch.floor(beats)
    if harmonics is None:
        harmonics = torch.tensor(HARMONICS, dtype=torch.float32, device=beats.device)
    angle = (2 * math.pi) * fraction[..., None] * harmonics
    return torch.cat((angle.sin(), angle.cos()), -1)


def exact_phase_features(phase):
    """Exact features of a rational phase: grid membership is exactly cos == 1."""
    phase = Fraction(phase)
    values = []
    for harmonic in HARMONICS:
        values.append(float((harmonic * phase) % 1))
    angles = 2 * math.pi * np.asarray(values, dtype=np.float64)
    result = np.concatenate((np.sin(angles), np.cos(angles)))
    for index, harmonic in enumerate(HARMONICS):
        if (harmonic * phase).denominator == 1:
            result[index], result[len(HARMONICS) + index] = 0., 1.
    return result.astype(np.float32)


def phase_token_table(codec):
    """[vocab, 18] exact phase features for F: tokens, zero rows elsewhere."""
    table = np.zeros((codec.vocab_size, PHASE_FEATURES), dtype=np.float32)
    for index, phase in getattr(codec, 'phases', {}).items():
        table[index] = exact_phase_features(phase)
    return torch.from_numpy(table)


# ---------------------------------------------------------------------------
# Pooled onset envelope (bank input channel)
# ---------------------------------------------------------------------------

def _mel_arrays(audio):
    try:
        mel, times = audio.mel, audio.mel_times_sec
    except AttributeError:
        return None
    mel = np.asarray(mel) if not hasattr(mel, 'shape') else mel
    if getattr(mel, 'ndim', 0) != 2 or mel.shape[1] <= _ENERGY or len(times) != len(mel) or len(times) < 2:
        return None
    return mel, np.asarray(times, dtype=np.float64)


def onset_envelope(audio, clock, beats):
    """Max positive flux and energy rise inside each sample's own beat interval.

    The bank samples 24 positions per beat by interpolation, which can halve a
    one-frame transient when the clock is slightly off. These pooled values keep
    the peak of every interval, independent of the exact sampling instant.
    """
    beats = np.asarray(beats, dtype=np.float64).reshape(-1)
    result = np.zeros((len(beats), ENVELOPE_FEATURES), dtype=np.float32)
    if not len(beats):
        return result
    arrays = _mel_arrays(audio)
    if arrays is None:
        offsets = np.linspace(-ENVELOPE_HALF_WIDTH, ENVELOPE_HALF_WIDTH, 4)
        queries = (beats[:, None] + offsets[None, :]).ravel()
        mel, _, _ = audio.sample(clock.at_floats(queries))
        mel = np.asarray(mel, dtype=np.float32)
        if mel.ndim != 2 or mel.shape[1] <= _ENERGY:
            return result
        mel = mel.reshape(len(beats), len(offsets), -1)
        result[:, 0] = np.maximum(mel[:, :, _FLUX], 0).max(1)
        rise = np.diff(mel[:, :, _ENERGY], axis=1)
        result[:, 1] = np.maximum(rise, 0).max(1) if rise.shape[1] else 0.
        return result
    mel, times = arrays
    lo = np.searchsorted(times, clock.at_floats(beats - ENVELOPE_HALF_WIDTH), side='left')
    hi = np.searchsorted(times, clock.at_floats(beats + ENVELOPE_HALF_WIDTH), side='right')
    lo = np.clip(lo, 0, len(times) - 1)
    hi = np.clip(np.maximum(hi, lo + 1), 1, len(times))
    width = int(min(16, max(1, int((hi - lo).max()))))
    first, last = int(lo.min()), int(min(len(times), hi.max() + 1))
    block = np.asarray(mel[max(0, first - 1):last, _FLUX:_ENERGY + 1], dtype=np.float32)
    origin = max(0, first - 1)
    flux = np.maximum(block[:, 0], 0)
    rise = np.zeros(len(block), dtype=np.float32)
    rise[1:] = np.maximum(np.diff(block[:, 1]), 0)
    index = (lo - origin)[:, None] + np.arange(width)[None, :]
    valid = index < (hi - origin)[:, None]
    index = np.clip(index, 0, len(block) - 1)
    result[:, 0] = np.where(valid, flux[index], 0.).max(1)
    result[:, 1] = np.where(valid, rise[index], 0.).max(1)
    return result


# ---------------------------------------------------------------------------
# Beat-synchronous song profile for reference matching
# ---------------------------------------------------------------------------

_PROJECTIONS = {}


def _projection(dimension):
    if dimension <= REF_SEMANTIC_DIM:
        return None
    matrix = _PROJECTIONS.get(dimension)
    if matrix is None:
        rng = np.random.default_rng(20261108 + dimension)
        matrix = (rng.standard_normal((dimension, REF_SEMANTIC_DIM)) / math.sqrt(REF_SEMANTIC_DIM)).astype(np.float32)
        _PROJECTIONS[dimension] = matrix
    return matrix


def _unit_rows(values, floor=1e-8):
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.where(norms > floor, values / np.maximum(norms, floor), 0.).astype(np.float32)


class BeatProfile:
    """Per integer beat: song-centred music identity and within-beat onset pattern."""

    def __init__(self, semantic, onset, onset_raw=None):
        self.semantic = semantic
        self.onset = onset
        self.onset_raw = np.maximum(onset, 0) if onset_raw is None else onset_raw

    @property
    def beats(self):
        return len(self.semantic)


def song_profile(audio, clock, end_beat, *, chunk_beats=64):
    beats = int(math.floor(float(end_beat)))
    if beats <= 0:
        return BeatProfile(np.zeros((0, 1), np.float32), np.zeros((0, REF_ONSET_BINS), np.float32))
    sums = None
    onset = np.zeros((beats, REF_ONSET_BINS), dtype=np.float32)
    arrays = _mel_arrays(audio)
    per_bin = REF_ONSET_SAMPLES
    for start in range(0, beats, chunk_beats):
        stop = min(beats, start + chunk_beats)
        grid = (np.arange(start, stop, dtype=np.float64)[:, None]
                + np.arange(REF_SAMPLES_PER_BEAT)[None, :] / REF_SAMPLES_PER_BEAT)
        mel, semantic, valid = audio.sample(clock.at_floats(grid.ravel()))
        semantic = np.asarray(semantic, dtype=np.float32)
        mel = np.asarray(mel, dtype=np.float32)
        valid = np.asarray(valid, dtype=bool)
        use_semantic = semantic.ndim == 2 and semantic.shape[1] > 0 and valid.any()
        values = semantic if use_semantic else mel[:, :min(96, mel.shape[1])]
        values = values.reshape(stop - start, REF_SAMPLES_PER_BEAT, -1).mean(1)
        if sums is None:
            sums = np.zeros((beats, values.shape[1]), dtype=np.float32)
        if sums.shape[1] != values.shape[1]:
            raise ValueError('Inconsistent audio feature width while profiling a song')
        sums[start:stop] = values
        fine = (np.arange(start, stop, dtype=np.float64)[:, None]
                + np.arange(REF_ONSET_BINS * per_bin)[None, :] / (REF_ONSET_BINS * per_bin))
        seconds = clock.at_floats(fine.ravel())
        if arrays is not None:
            flux = np.interp(seconds, arrays[1], np.asarray(arrays[0][:, _FLUX], dtype=np.float32))
        else:
            flux = np.asarray(audio.sample(seconds)[0], dtype=np.float32)
            flux = flux[:, _FLUX] if flux.ndim == 2 and flux.shape[1] > _FLUX else np.zeros(len(seconds), np.float32)
        onset[start:stop] = np.maximum(flux, 0).reshape(stop - start, REF_ONSET_BINS, per_bin).max(2)
    centred = sums - sums.mean(0, keepdims=True)
    projection = _projection(centred.shape[1])
    if projection is not None:
        centred = centred @ projection
    semantic = _unit_rows(centred)
    pattern = onset - onset.mean(1, keepdims=True)
    spread = onset.max(1) - onset.min(1)
    typical = float(np.median(spread[spread > 0])) if np.any(spread > 0) else 0.
    pattern[spread < .1 * typical] = 0.
    return BeatProfile(semantic, _unit_rows(pattern), onset)


def match_reference(profile, cursor, *, min_shift, alternative_draw=None, ahead_beats=REF_MATCH_AHEAD_BEATS):
    """Best integer-beat shift of the upcoming passage to an earlier one.

    Returns dict(best_shift, best_score, alternative_shift, alternative_score)
    or None when no shift is admissible. Scores are mean per-beat similarities
    over [floor(cursor)-lead, floor(cursor)+ahead), restricted to audio beats.
    """
    anchor = int(math.floor(Fraction(cursor)))
    span = np.arange(anchor - REF_LEAD_BEATS, anchor + int(ahead_beats))
    span = span[(span >= 0) & (span < profile.beats)]
    if len(span) < REF_MIN_VALID_BEATS:
        return None
    min_shift = max(1, int(min_shift))
    max_shift = int(span[0])
    if max_shift < min_shift:
        return None
    shifts = np.arange(min_shift, max_shift + 1)
    source = span[None, :] - shifts[:, None]
    semantic = (profile.semantic[span][None] * profile.semantic[source]).sum(-1)
    onset = (profile.onset[span][None] * profile.onset[source]).sum(-1)
    scores = (REF_SEMANTIC_WEIGHT * semantic + (1 - REF_SEMANTIC_WEIGHT) * onset).mean(1)
    best = int(np.argmax(scores))
    if alternative_draw is None:
        alternative = best
    else:
        alternative = min(len(shifts) - 1, int(float(alternative_draw) * len(shifts)))
    return dict(best_shift=int(shifts[best]), best_score=float(scores[best]),
                alternative_shift=int(shifts[alternative]), alternative_score=float(scores[alternative]))


def local_evidence(profile, beats, shift, *, start, end):
    """Audio-only source/target comparison at each displayed or queried position.

    Columns: coverage, semantic similarity, onset-pattern similarity, combined
    similarity, relative onset strength, source/target phase support, progress.
    A zero coverage flag means no verified reference at this position; it never
    masks a generation token or forces an event. The source chart stays causal.
    """
    beats = np.asarray(beats, dtype=np.float64).reshape(-1)
    out = np.zeros((len(beats), EVIDENCE_FEATURES), dtype=np.float32)
    if shift is None or not len(beats):return out
    target = np.floor(beats).astype(np.int64); source = target - int(shift)
    valid = ((beats >= float(start)) & (beats < float(end)) &
             (target >= 0) & (target < profile.beats) & (source >= 0) & (source < profile.beats))
    rows = np.flatnonzero(valid)
    if not len(rows):return out
    t, s = target[rows], source[rows]
    semantic = (profile.semantic[t] * profile.semantic[s]).sum(1)
    onset = (profile.onset[t] * profile.onset[s]).sum(1)
    raw_t, raw_s = profile.onset_raw[t], profile.onset_raw[s]
    strength_t, strength_s = raw_t.mean(1), raw_s.mean(1)
    strength = (strength_t-strength_s)/(strength_t+strength_s+1e-8)
    bins = np.minimum(REF_ONSET_BINS-1, ((beats[rows]-t)*REF_ONSET_BINS).astype(np.int64))
    support_t = raw_t[np.arange(len(rows)), bins]/np.maximum(raw_t.max(1),1e-8)
    support_s = raw_s[np.arange(len(rows)), bins]/np.maximum(raw_s.max(1),1e-8)
    out[rows] = np.stack((np.ones(len(rows)),semantic,onset,
        REF_SEMANTIC_WEIGHT*semantic+(1-REF_SEMANTIC_WEIGHT)*onset,
        strength,support_s,support_t,(beats[rows]-float(start))/max(1.,float(end)-float(start))),1)
    return out


def migration_state(model, parent_state):
    """Preserve learned v1 residuals; add only missing v2 parameters on finetune."""
    target = model.state_dict()
    state = {k:v for k,v in parent_state.items() if not k.startswith(STATE_PREFIXES) or k in target}
    missing = {k for k in target if k.startswith(STATE_PREFIXES) and k not in state}
    return state, missing


def trust_features(score):
    """present, score, below-gate flag, then REF_TRUST_BINS bins above the gate."""
    values = np.zeros(TRUST_FEATURES, dtype=np.float32)
    if score is None:
        return values
    score = float(np.clip(score, -1., 1.))
    values[0], values[1] = 1., score
    if score < REF_MIN_SCORE:
        values[2] = 1.
    else:
        position = (score - REF_MIN_SCORE) / max(1e-6, 1. - REF_MIN_SCORE)
        values[3 + min(REF_TRUST_BINS - 1, int(position * REF_TRUST_BINS))] = 1.
    return values


def reference_points(points, cursor, shift, *, display_ahead, max_ahead_events, beats=None, end=None):
    """Committed events shown on the target timeline (source beat + shift).

    ``points`` is sorted by beat. With ``beats``/``end`` the indexed training
    prefix is used without copying the full song.
    """
    cursor = Fraction(cursor)
    anchor = int(math.floor(cursor))
    lo = Fraction(anchor - REF_LEAD_BEATS - shift)
    hi = Fraction(anchor + int(display_ahead) - shift)
    if beats is None:
        beats = [p.beat for p in points]
        end = len(points)
    first = bisect_left(beats, lo, hi=end)
    last = bisect_left(beats, hi, lo=first, hi=end)
    from maichart.event_timing import EventPoint
    shown = [EventPoint(p.beat + shift, p.note_text) for p in points[first:last]]
    lead = [p for p in shown if p.beat <= cursor][-REF_MAX_LEAD_EVENTS:]
    ahead = [p for p in shown if p.beat > cursor][:max(0, int(max_ahead_events))]
    return lead + ahead


class ProfileCache:
    """Small LRU of song profiles; songs are grouped by the audio-local sampler."""

    def __init__(self, limit=REF_PROFILE_CACHE):
        self.limit = int(limit)
        self.items = OrderedDict()

    def get(self, key, build):
        value = self.items.get(key)
        if value is not None:
            self.items.move_to_end(key)
            return value
        value = build()
        self.items[key] = value
        while len(self.items) > self.limit:
            self.items.popitem(last=False)
        return value
