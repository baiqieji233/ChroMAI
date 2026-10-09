"""The vectorised timing context must match the per-frame reference exactly.

`_timing_context` builds ten per-frame context columns (progress, local BPM,
beat/bar/phrase phase) and is called once per training sample -- on the full
corpus the per-frame Python loop cost 26 minutes of CPU per epoch.  It is now
vectorised, so equivalence with the scalar reference is pinned here across the
shapes real charts produce: constant tempo, multi-entry maps, maps whose
boundaries land between frames, negative `first`, and frames past the last
boundary.
"""

import numpy as np
import pytest
from pathlib import Path

from maichart.generation.event_audio import (
    _beats_at_audio_offsets,
    _tempos_at_beats,
    _time_to_beat,
    _tempo_at_beat,
    _timing_context,
    _timing_context_scalar,
)


def _scalar_context(times, **kwargs):
    return _timing_context_scalar(times, **kwargs)


def _vector_context(times, **kwargs):
    return _timing_context(times, **kwargs)


def _maps():
    from fractions import Fraction

    single = [(Fraction(0), 134.0)]
    two = [(Fraction(0), 150.0), (Fraction(32), 120.0)]
    # Sorted by beat, as the chart parser emits them; a boundary between frames
    # (64/3) is included on purpose.
    many = [(Fraction(0), 178.0), (Fraction(16), 89.0), (Fraction(64, 3), 133.0),
            (Fraction(48), 200.0), (Fraction(96), 156.0)]
    return {"single": single, "two": two, "many": many}


def test_unsorted_boundaries_fall_back_to_the_scalar_path():
    """A hand-built unordered map must still behave exactly as the reference."""

    from fractions import Fraction

    entries = [(Fraction(0), 178.0), (Fraction(48), 200.0), (Fraction(64, 3), 133.0)]
    times = np.arange(200, dtype=np.float64) / 50.0
    kwargs = dict(duration=180.0, first=0.0, start_sec=0.0, start_beat=0.0,
                  map_entries=entries, tempo_changes=None, fallback_bpm=178.0)
    np.testing.assert_array_equal(_vector_context(times, **kwargs),
                                  _scalar_context(times, **kwargs))


@pytest.mark.parametrize("name", ["single", "two", "many"])
@pytest.mark.parametrize("first", [0.0, 0.5, -1.25])
def test_vectorised_context_matches_the_scalar_reference(name, first):
    entries = _maps()[name]
    fallback = entries[0][1]
    for start_sec in (0.0, 7.3, 33.75):
        times = start_sec + np.arange(420, dtype=np.float64) / 50.0
        kwargs = dict(duration=180.0, first=first, start_sec=start_sec,
                      start_beat=0.0, map_entries=entries, tempo_changes=None,
                      fallback_bpm=fallback)
        # The anchor check compares start_beat against the map; derive the beat
        # the map itself reports so both paths agree on the anchor.
        kwargs["start_beat"] = float(_time_to_beat(start_sec - first, entries))
        fast = _vector_context(times, **kwargs)
        slow = _scalar_context(times, **kwargs)
        assert fast.shape == slow.shape
        assert fast.dtype == np.float32
        np.testing.assert_allclose(fast, slow, rtol=0, atol=1e-6)


def test_vectorised_helpers_match_their_scalar_originals():
    entries = _maps()["many"]
    offsets = np.concatenate([
        np.linspace(-30.0, 0.0, 40),
        np.linspace(0.0, 400.0, 400),
        np.array([1e-9, 1e-6, 0.5]),
    ])
    fast = _beats_at_audio_offsets(offsets, entries)
    slow = np.array([_time_to_beat(float(value), entries) for value in offsets])
    np.testing.assert_allclose(fast, slow, rtol=0, atol=1e-9)

    beats = np.concatenate([np.linspace(-4.0, 0.0, 20), np.linspace(0.0, 140.0, 300)])
    fast_bpm = _tempos_at_beats(beats, entries, 100.0)
    slow_bpm = np.array([_tempo_at_beat(float(value), entries, 100.0) for value in beats])
    np.testing.assert_allclose(fast_bpm, slow_bpm, rtol=0, atol=1e-12)


def test_frames_past_the_last_boundary_keep_the_last_tempo():
    entries = _maps()["two"]
    offsets = np.array([1e6])
    assert _beats_at_audio_offsets(offsets, entries)[0] == pytest.approx(
        _time_to_beat(float(offsets[0]), entries))


def test_anchor_mismatch_still_raises():
    entries = _maps()["single"]
    times = np.arange(10, dtype=np.float64) / 50.0
    with pytest.raises(ValueError, match="start_beat does not agree"):
        _vector_context(times, duration=60.0, first=0.0, start_sec=0.0, start_beat=99.0,
                        map_entries=entries, tempo_changes=None, fallback_bpm=134.0)


def test_non_positive_tempo_is_rejected():
    from fractions import Fraction

    entries = [(Fraction(0), -10.0)]
    times = np.arange(5, dtype=np.float64) / 50.0
    with pytest.raises(ValueError):
        _vector_context(times, duration=60.0, first=0.0, start_sec=0.0, start_beat=0.0,
                        map_entries=entries, tempo_changes=None, fallback_bpm=-10.0)


def test_resample_rows_matches_linear_interp_with_zero_outside_file():
    """The vectorised resampler interpolates inside the file and zero-fills outside.

    98 np.interp calls per sample were 98 GIL round-trips on the prefetch thread;
    the single-pass version has to keep the same values (float64 interpolation,
    Repeating the edge frame outside the file leaks a song's final spectrum into
    post-audio windows and can create false tail notes, so out-of-range samples
    are explicitly silent.
    """

    from maichart.generation.event_audio import _resample_rows

    rng = np.random.default_rng(0)
    for frames, columns in ((2, 98), (37, 5), (500, 98), (2193, 98)):
        values = rng.standard_normal((frames, columns)).astype(np.float32)
        source = np.arange(frames, dtype=np.float32) / 50.0
        for count in (1, 17, 533, 2000):
            targets = (rng.random(count).astype(np.float32) * (frames + 20) / 50.0) - 0.2
            reference = np.column_stack([
                np.interp(targets, source, values[:, column]) for column in range(columns)
            ]).astype(np.float32)
            reference[(targets < source[0]) | (targets > source[-1])] = 0.0
            got = _resample_rows(values, source, targets)
            np.testing.assert_allclose(got, reference, rtol=0, atol=1e-6)
        # Targets exactly on source samples, and far outside the range.
        edges = np.concatenate([source[:3], source[-3:], np.array([-5.0, 1e6], dtype=np.float32)])
        reference = np.column_stack([
            np.interp(edges, source, values[:, column]) for column in range(columns)
        ]).astype(np.float32)
        reference[(edges < source[0]) | (edges > source[-1])] = 0.0
        np.testing.assert_allclose(_resample_rows(values, source, edges), reference,
                                   rtol=0, atol=1e-6)


def test_resample_rows_handles_degenerate_inputs():
    from maichart.generation.event_audio import _resample_rows

    single = np.ones((1, 3), dtype=np.float32)
    out = _resample_rows(single, np.array([0.0], dtype=np.float32), np.arange(4, dtype=np.float32))
    assert out.shape == (4, 3) and np.allclose(out[0], 1.0) and np.allclose(out[1:], 0.0)
    one_source = np.arange(8, dtype=np.float32).reshape(4, 2)
    out = _resample_rows(one_source, np.array([0.5], dtype=np.float32),
                         np.array([0.0, 0.25, 0.5], dtype=np.float32))
    assert out.shape == (3, 2) and np.allclose(out[:2], 0.0) and np.allclose(out[2], one_source[0])
    empty = _resample_rows(np.zeros((10, 4), dtype=np.float32),
                           np.arange(10, dtype=np.float32), np.array([], dtype=np.float32))
    assert empty.shape == (0, 4)


def test_ctx_dataset_caches_the_tempo_map_per_song():
    """Rebuilding the normalized map per window was repeated work."""

    source = (Path(__file__).resolve().parents[1] / "src/maichart/training/event_ctx_dataset.py").read_text(encoding="utf-8")
    assert "_tempo_cache" in source
    calls = source.count("_normalize_tempo_map(")
    assert calls == 1, f"the map must be built in exactly one place, found {calls}"
    """A callable tempo map has no closed form; it must keep working."""

    def warped(beat: float) -> float:
        return 120.0 + 10.0 * np.sin(beat / 8.0)

    times = np.arange(60, dtype=np.float64) / 50.0
    context = _vector_context(times, duration=60.0, first=0.0, start_sec=0.0, start_beat=0.0,
                              map_entries=[], tempo_changes=warped, fallback_bpm=120.0)
    assert context.shape == (60, 10)
    assert np.isfinite(context).all()
