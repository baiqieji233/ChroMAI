"""Window-level density guidance and audible boundary detection.

The model already sees local audio. These helpers additionally express a
window's intended density through the two radar controls. The legacy mode
compares RMS to its song median; multifeature mode estimates musical tension
from rhythm, timbre, harmonic change and local contrasts as well as loudness.
Neither estimate is a reliable measurement of human emotion or play difficulty.

Direct function callers and existing training contracts retain ``loudness`` as
the default. Generation can select multifeature following, or apply an edited
curve through event_cli. Audible boundaries continue using independent
loudness/onset evidence. ``tail_damping`` remains an optional musical prior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maichart.generation.event_audio import FEATURE_RATE_HZ

# The two radar controls that describe *how much* is happening.  Style controls
# (slide/hold/touch bias, directionality, ...) are a preference, not an intensity,
# so following the music must not rewrite them.
INTENSITY_KEYS = ("note_density", "peak")

# A window shorter than this many frames has no usable intensity estimate.
MIN_WINDOW_FRAMES = 3


def intensity_series(audio) -> tuple[np.ndarray, np.ndarray]:
    """``(times, relative_rms)`` from the extractor's own song-relative channel."""

    from maichart.generation.event_audio import (
        BASE_ACOUSTIC_KEYS,
        N_CHROMA,
        N_MELS,
        STRUCTURE_KEYS,
    )

    features = np.asarray(audio.features, dtype=np.float64)
    times = np.asarray(audio.times, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] == 0:
        return times, np.zeros(0, dtype=np.float64)
    column = N_MELS + N_CHROMA + len(BASE_ACOUSTIC_KEYS) + STRUCTURE_KEYS.index("relative_rms")
    if column >= features.shape[1]:
        return times, np.zeros(0, dtype=np.float64)
    return times, features[:, column]


def _smooth(values: list[float], width: int) -> list[float]:
    """Centred moving average, so a single loud bar cannot spike the factor."""

    if width <= 1 or len(values) <= 2:
        return list(values)
    half = width // 2
    smoothed = []
    for index in range(len(values)):
        low = max(0, index - half)
        high = min(len(values), index + half + 1)
        smoothed.append(float(np.mean(values[low:high])))
    return smoothed


def window_factors(
    audio,
    windows,
    *,
    tempo_changes,
    first: float,
    span: float = 0.5,
    smoothing: int = 3,
    song_duration: float | None = None,
    mode: str = "loudness",
) -> list[float]:
    """One intensity factor per window, centred on 1.0 and clipped to ``span``.

    ``1.0`` means baseline density intention, based on song-median loudness in
    legacy mode or the multifeature musical tension proxy in the new mode.
    A lower factor requests fewer notes and a higher factor more. ``span`` bounds the change
    (0.5 => 0.5x..1.5x) so the user's requested density stays recognisable.
    """

    from maichart.event_timing import seconds_at_beat

    if not windows:
        return []
    if mode not in {"loudness", "multifeature"}:
        raise ValueError("intensity mode must be loudness or multifeature")
    if mode == "multifeature":
        # Existing audio features retain rhythm/timbre/section information.
        # Reuse them for automatic following without changing the encoder's
        # feature dimensions or decoding the waveform a second time.
        from maichart.generation.music_intensity import (
            curve_window_values, intensity_profile_from_event_audio,
        )
        profile = intensity_profile_from_event_audio(audio)
        limit = max(0.0, min(0.95, float(span)))
        factors = []
        for start, end in windows:
            low = float(seconds_at_beat(start, tempo_changes, first))
            high = float(seconds_at_beat(end, tempo_changes, first))
            if song_duration:
                high = min(high, float(song_duration))
            value = 2.0 * curve_window_values(profile, max(0.0, low), max(0.0, high)) if high > low else 1.0
            factors.append(float(np.clip(value, 1.0 - limit, 1.0 + limit)))
        return factors
    times, values = intensity_series(audio)
    if values.size == 0:
        return [1.0] * len(windows)
    reference = float(np.median(values))
    if not np.isfinite(reference) or reference <= 1e-6:
        return [1.0] * len(windows)

    raw: list[float] = []
    for start, end in windows:
        low = float(seconds_at_beat(start, tempo_changes, first))
        high = float(seconds_at_beat(end, tempo_changes, first))
        if song_duration:
            high = min(high, float(song_duration))
        if high <= low:
            raw.append(1.0)
            continue
        mask = (times >= low) & (times < high)
        if int(mask.sum()) < MIN_WINDOW_FRAMES:
            raw.append(1.0)
            continue
        raw.append(float(values[mask].mean()) / reference)

    limit = max(0.0, min(0.95, float(span)))
    return [float(min(1.0 + limit, max(1.0 - limit, value)))
            for value in _smooth(raw, int(smoothing))]


def tail_factors(count: int, *, windows: int = 4, strength: float = 0.6) -> list[float]:
    """Linear taper over the last ``windows`` windows (1.0 elsewhere).

    The final window gets ``1 - strength``; earlier tail windows taper linearly.
    Off by default: it is a musical prior about outros, not a measurement.
    """

    factors = [1.0] * max(0, int(count))
    if not factors or windows <= 0 or strength <= 0:
        return factors
    length = min(int(windows), len(factors))
    amount = max(0.0, min(0.95, float(strength)))
    # offset 0 is the *start* of the tail, so the very last window gets the full
    # strength and the taper rises back to 1.0 as it walks backwards.
    for offset in range(length):
        step = offset + 1
        factors[len(factors) - length + offset] = 1.0 - amount * (step / length)
    return factors


def combine(*series: list[float]) -> list[float]:
    """Element-wise product of equal-length factor series."""

    if not series:
        return []
    result = list(series[0])
    for values in series[1:]:
        result = [a * b for a, b in zip(result, values)]
    return result


def describe(factors: list[float]) -> str:
    """One-line audit string for the generation log."""

    if not factors:
        return "none"
    return (f"n={len(factors)} min={min(factors):.2f} max={max(factors):.2f} "
            f"mean={float(np.mean(factors)):.2f}")


# ---------------------------------------------------------------------------
# Audible bounds: where the music actually starts and stops
# ---------------------------------------------------------------------------
# The generation clock is the *file*: windows are cut from beat zero to the end
# of the detected beat grid, so the silent lead-in and the trailing silence are
# asked for notes like any other window.  Measured on a real run: the music was
# audible from 2.44 s to 142.94 s (the first two seconds had relative_rms 0.000)
# while the chart wrote 8 notes before 2.44 s and 9 notes after 142.94 s, up to
# 146.12 s.  These helpers find the audible span so generation can stay inside it.

DEFAULT_SILENCE_THRESHOLD = 0.15   # share of the song's median loudness
DEFAULT_SMOOTHING_SEC = 0.25       # a single loud frame must not define a boundary


@dataclass(frozen=True)
class LoudnessProfile:
    """Smoothed, song-relative loudness: 1.0 is the song's own median."""

    times: np.ndarray
    values: np.ndarray
    median: float


def loudness_profile(audio, *, smoothing_sec: float = DEFAULT_SMOOTHING_SEC) -> LoudnessProfile | None:
    """One pass over the audio for both the bounds and the per-window shares.

    The reference is the median loudness, but a track that is silent for more than
    half of its length has a median of zero -- falling back to the 90th percentile
    keeps the audible span measurable instead of silently disabling the trim.
    """

    times, values = intensity_series(audio)
    if values.size == 0:
        return None
    median = float(np.median(values))
    if not np.isfinite(median) or median <= 1e-6:
        median = float(np.percentile(values, 90))
    if not np.isfinite(median) or median <= 1e-6:
        return None
    width = max(1, int(round(max(smoothing_sec, 0.0) * FEATURE_RATE_HZ)))
    if width > 1 and values.size > width:
        kernel = np.ones(width, dtype=np.float64) / width
        smoothed = np.convolve(values, kernel, mode="same")
    else:
        smoothed = values.astype(np.float64, copy=True)
    return LoudnessProfile(times=np.asarray(times, dtype=np.float64), values=smoothed, median=median)


def audible_bounds(profile: LoudnessProfile | None, *,
                   threshold: float = DEFAULT_SILENCE_THRESHOLD) -> tuple[float, float] | None:
    """``(first, last)`` audible second, or ``None`` when nothing is audible."""

    if profile is None or profile.values.size == 0:
        return None
    mask = profile.values > float(threshold)
    if not mask.any():
        return None
    indices = np.flatnonzero(mask)
    return float(profile.times[indices[0]]), float(profile.times[indices[-1]])


def window_shares(profile: LoudnessProfile | None, windows, *, tempo_changes, first: float,
                  threshold: float = DEFAULT_SILENCE_THRESHOLD,
                  song_duration: float | None = None) -> list[float]:
    """Share of each window's frames that carry audible music (0.0 .. 1.0)."""

    from maichart.event_timing import seconds_at_beat

    if profile is None:
        return [1.0] * len(windows)
    shares: list[float] = []
    for start, end in windows:
        low = float(seconds_at_beat(start, tempo_changes, first))
        high = float(seconds_at_beat(end, tempo_changes, first))
        if song_duration:
            high = min(high, float(song_duration))
        if high <= low:
            shares.append(0.0)
            continue
        mask = (profile.times >= low) & (profile.times < high)
        if not mask.any():
            shares.append(0.0)
            continue
        shares.append(float(np.mean(profile.values[mask] > float(threshold))))
    return shares


def trimmed_span(bounds: tuple[float, float] | None, *, lead: float,
                 song_duration: float | None = None) -> tuple[float, float] | None:
    """Grow the audible span by ``lead`` seconds, clamped to the file."""

    if bounds is None:
        return None
    low = max(0.0, float(bounds[0]) - max(0.0, float(lead)))
    high = float(bounds[1]) + max(0.0, float(lead))
    if song_duration:
        high = min(high, float(song_duration))
    return (low, high) if high > low else None


# ---------------------------------------------------------------------------
# Music activity: loudness *or* onset evidence, which is what the bounds need
# ---------------------------------------------------------------------------
# Loudness alone is not enough.  On the real track that motivated this, the music
# starts with a quiet attack at 1.98-2.00s whose loudness stays below 15% of the
# song's median until 2.40s: a loudness-only bound called the first 0.4s silence
# and cut the two notes the model had correctly placed on that pickup, leaving a
# 3.25s hole at the start of the chart.  The onset envelope shows the attack, and
# on a digitally silent lead-in it is exactly zero, so it cannot invent one either.

DEFAULT_ONSET_QUANTILE = 0.70


def onset_series(audio) -> tuple[np.ndarray, np.ndarray]:
    """``(times, onset_strength)`` from the extractor's onset channel."""

    from maichart.generation.event_audio import (
        BASE_ACOUSTIC_KEYS,
        N_CHROMA,
        N_MELS,
    )

    features = np.asarray(audio.features, dtype=np.float64)
    times = np.asarray(audio.times, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] == 0:
        return times, np.zeros(0, dtype=np.float64)
    column = N_MELS + N_CHROMA + BASE_ACOUSTIC_KEYS.index("onset_strength")
    if column >= features.shape[1]:
        return times, np.zeros(0, dtype=np.float64)
    return times, features[:, column]


def _smooth_series(values: np.ndarray, width: int) -> np.ndarray:
    if width > 1 and values.size > width:
        kernel = np.ones(width, dtype=np.float64) / width
        return np.convolve(values, kernel, mode="same")
    return values.astype(np.float64, copy=True)


@dataclass(frozen=True)
class MusicActivity:
    """Where the track has music, from loudness and onset evidence."""

    times: np.ndarray
    loudness: np.ndarray
    onset: np.ndarray
    loudness_threshold: float
    onset_threshold: float
    median: float

    def active(self) -> np.ndarray:
        return (self.loudness > self.loudness_threshold) | (self.onset > self.onset_threshold)

    def bounds(self) -> tuple[float, float] | None:
        """First and last second with any musical evidence."""

        mask = self.active()
        if not mask.any():
            return None
        indices = np.flatnonzero(mask)
        return float(self.times[indices[0]]), float(self.times[indices[-1]])

    def component_bounds(self) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        """The two independent spans, for the log (auditability)."""

        def span(mask: np.ndarray) -> tuple[float, float] | None:
            if not mask.any():
                return None
            indices = np.flatnonzero(mask)
            return float(self.times[indices[0]]), float(self.times[indices[-1]])

        return span(self.loudness > self.loudness_threshold), span(self.onset > self.onset_threshold)

    def window_shares(self, windows, *, tempo_changes, first: float,
                      song_duration: float | None = None) -> list[float]:
        """Share of each window's frames with any musical evidence."""

        from maichart.event_timing import seconds_at_beat

        mask = self.active()
        shares: list[float] = []
        for start, end in windows:
            low = float(seconds_at_beat(start, tempo_changes, first))
            high = float(seconds_at_beat(end, tempo_changes, first))
            if song_duration:
                high = min(high, float(song_duration))
            if high <= low:
                shares.append(0.0)
                continue
            inside = (self.times >= low) & (self.times < high)
            if not inside.any():
                shares.append(0.0)
                continue
            shares.append(float(np.mean(mask[inside])))
        return shares

    def describe(self) -> str:
        loudness, onset = self.component_bounds()
        def text(span):
            return f"{span[0]:.2f}-{span[1]:.2f}s" if span else "none"
        return (f"loudness={text(loudness)} onset={text(onset)} "
                f"loudness_threshold={self.loudness_threshold:.2f} "
                f"onset_threshold={self.onset_threshold:.3f}")


def music_activity(audio, *, loudness_threshold: float = DEFAULT_SILENCE_THRESHOLD,
                   onset_quantile: float = DEFAULT_ONSET_QUANTILE,
                   smoothing_sec: float = DEFAULT_SMOOTHING_SEC) -> MusicActivity | None:
    """One pass over the audio: smoothed loudness and onset envelopes."""

    times, values = intensity_series(audio)
    if values.size == 0:
        return None
    median = float(np.median(values))
    if not np.isfinite(median) or median <= 1e-6:
        median = float(np.percentile(values, 90))
    if not np.isfinite(median) or median <= 1e-6:
        return None
    width = max(1, int(round(max(smoothing_sec, 0.0) * FEATURE_RATE_HZ)))
    onset_times, onset_values = onset_series(audio)
    if onset_values.size != values.size:
        onset_values = np.zeros_like(values)
    quantile = min(0.99, max(0.0, float(onset_quantile)))
    onset_threshold = float(np.percentile(onset_values, quantile * 100.0)) if onset_values.size else 0.0
    return MusicActivity(
        times=np.asarray(times, dtype=np.float64),
        loudness=_smooth_series(values, width),
        onset=_smooth_series(onset_values, width),
        loudness_threshold=float(loudness_threshold),
        onset_threshold=onset_threshold,
        median=median,
    )
