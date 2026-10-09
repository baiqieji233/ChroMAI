"""Intensity following: the fix for "flat charts" and for a burst at the outro.

Measured problem (a trained checkpoint, real song): the condition vector is
song-level and constant, so the model emitted a nearly flat chart -- per-16-beat
event counts with a coefficient of variation of 0.22, correlated -0.37 with the
music's own relative loudness, the project's own `rms_density_correlation` at
0.095 against a gold p05 of 0.354, and the *quietest* window of the song receiving
the model's maximum event count.

These tests pin the mechanism: quiet windows must ask for less, loud ones for
more, the span must bound the change, style controls must be untouched, and the
tail taper must be a monotone, off-by-default prior.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch

from maichart.event_timing import EventTempo
from maichart.generation.conditions import (
    GENERATION_SAMPLING_DEFAULTS,
    GenerationCondition,
    scale_intensity,
)
from maichart.generation.event_model import EventSequence, EventVocabulary
from maichart.generation.intensity import (
    INTENSITY_KEYS,
    combine,
    describe,
    intensity_series,
    tail_factors,
    window_factors,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeAudio:
    """Minimal EventAudio stand-in with a controllable relative_rms channel."""

    def __init__(self, values: list[float], frames_per_second: int = 50):
        from maichart.generation.event_audio import (
            BASE_ACOUSTIC_KEYS,
            N_CHROMA,
            N_MELS,
            STRUCTURE_KEYS,
        )

        width = N_MELS + N_CHROMA + len(BASE_ACOUSTIC_KEYS) + len(STRUCTURE_KEYS)
        column = N_MELS + N_CHROMA + len(BASE_ACOUSTIC_KEYS) + STRUCTURE_KEYS.index("relative_rms")
        features = np.zeros((len(values), width), dtype=np.float32)
        features[:, column] = values
        self.features = features
        self.times = np.arange(len(values), dtype=np.float32) / frames_per_second
        self.duration_sec = len(values) / frames_per_second
        self.tempo_bpm = 120.0
        self.tempo_changes = []
        self.metadata = {}


def _windows(count: int, beats: int = 16) -> list[tuple[Fraction, Fraction]]:
    return [(Fraction(index * beats), Fraction((index + 1) * beats)) for index in range(count)]


# 120 BPM => 2 beats per second => a 16-beat window is 8 s = 400 frames at 50 Hz.
_CONDITION = GenerationCondition(level=13.0, radar={key: 0.3 for key in INTENSITY_KEYS}, style={})
VOCABULARY = EventVocabulary(["AT:0", "AT:1", "CHAR:1", "CHAR:2", "CHAR:3"])
TEMPO = [EventTempo(Fraction(0), 120.0)]
WINDOW_FRAMES = 400


def _audio_from_windows(levels: list[float]) -> FakeAudio:
    values: list[float] = []
    for level in levels:
        values.extend([level] * WINDOW_FRAMES)
    return FakeAudio(values)


def test_intensity_series_reads_the_relative_rms_channel():
    audio = FakeAudio([1.0] * 10)
    times, values = intensity_series(audio)
    assert len(times) == len(values) == 10
    assert np.allclose(values, 1.0)


def test_quiet_windows_ask_for_less_and_loud_ones_for_more():
    # Window 2 is twice the song median, window 1 is a quarter of it.
    audio = _audio_from_windows([1.0, 0.25, 1.0, 2.0])
    factors = window_factors(audio, _windows(4), tempo_changes=TEMPO, first=0.0, span=0.5, smoothing=1)
    assert factors[1] < 1.0 < factors[3]
    # Median of the four windows is 1.0, so the factors are 1, .25, 1, 2 clipped.
    assert factors[1] == pytest.approx(0.5)   # clipped by span
    assert factors[3] == pytest.approx(1.5)   # clipped by span
    assert factors[0] == pytest.approx(1.0)


def test_span_bounds_the_change():
    audio = _audio_from_windows([1.0, 0.05, 1.0, 20.0])
    for span in (0.2, 0.5, 0.8):
        factors = window_factors(audio, _windows(4), tempo_changes=TEMPO, first=0.0, span=span, smoothing=1)
        assert all(1 - span - 1e-9 <= value <= 1 + span + 1e-9 for value in factors)


def test_smoothing_damps_a_single_loud_window():
    # 1.3x stays inside the 0.5 span, so clipping cannot mask the smoothing.
    audio = _audio_from_windows([1.0, 1.0, 1.3, 1.0, 1.0])
    raw = window_factors(audio, _windows(5), tempo_changes=TEMPO, first=0.0, span=0.5, smoothing=1)
    smoothed = window_factors(audio, _windows(5), tempo_changes=TEMPO, first=0.0, span=0.5, smoothing=3)
    assert raw[2] == pytest.approx(1.3)
    assert smoothed[2] < raw[2], "a lone loud bar must not spike the factor"
    assert smoothed[1] > raw[1] and smoothed[3] > raw[3]


def test_silent_or_unmeasurable_audio_is_neutral():
    silent = FakeAudio([0.0] * 100)
    assert window_factors(silent, _windows(2), tempo_changes=TEMPO, first=0.0) == [1.0, 1.0]
    empty = FakeAudio([])
    assert window_factors(empty, _windows(2), tempo_changes=TEMPO, first=0.0) == [1.0, 1.0]
    assert window_factors(silent, [], tempo_changes=TEMPO, first=0.0) == []


def test_windows_past_the_song_end_stay_neutral():
    audio = _audio_from_windows([1.0, 1.0])
    # Four windows but the audio covers only two: the tail has no frames.
    factors = window_factors(audio, _windows(4), tempo_changes=TEMPO, first=0.0, span=0.5,
                             smoothing=1, song_duration=audio.duration_sec)
    assert factors[2:] == [1.0, 1.0]


def test_tail_damping_is_monotone_and_reaches_its_strength():
    factors = tail_factors(10, windows=4, strength=0.6)
    assert factors[:6] == [1.0] * 6
    assert factors[-1] == pytest.approx(0.4)
    assert factors[-2] == pytest.approx(0.55)
    assert all(a >= b for a, b in zip(factors, factors[1:])), "must taper downwards"
    # Off by default at the API level too.
    assert tail_factors(10, windows=0, strength=0.6) == [1.0] * 10
    assert tail_factors(10, windows=4, strength=0.0) == [1.0] * 10
    assert tail_factors(0) == []
    # Never exceeds the window count.
    assert tail_factors(2, windows=9, strength=0.5) == pytest.approx([0.75, 0.5])


def test_combine_multiplies_series():
    assert combine([1.0, 0.5], [0.4, 1.0]) == pytest.approx([0.4, 0.5])
    assert combine() == []


def test_describe_reports_the_range():
    text = describe([0.5, 1.0, 1.5])
    assert "min=0.50" in text and "max=1.50" in text and "n=3" in text
    assert describe([]) == "none"


def test_scale_intensity_touches_only_the_density_controls():
    condition = GenerationCondition(level=13.3, radar={key: 0.3 for key in INTENSITY_KEYS},
                                    style={"tap_bias": 0.72, "melody_following": 0.9})
    quiet = scale_intensity(condition, 0.5)
    loud = scale_intensity(condition, 1.5)
    for key in INTENSITY_KEYS:
        assert quiet.radar[key] == pytest.approx(0.15)
        assert loud.radar[key] == pytest.approx(0.45)
    # Style is a preference, not an intensity: untouched.
    assert quiet.style == condition.style
    assert loud.radar.keys() == condition.radar.keys()
    # Level is the difficulty label, not an intensity either.
    assert quiet.level == condition.level
    # The original object is not mutated.
    assert condition.radar["note_density"] == 0.3


def test_scale_intensity_clips_to_the_trained_range():
    condition = GenerationCondition(level=13.0, radar={"note_density": 0.9, "peak": 0.05})
    scaled = scale_intensity(condition, 2.0)
    assert scaled.radar["note_density"] == 1.0
    assert scaled.radar["peak"] == pytest.approx(0.1)


def test_cli_defaults_and_flags():
    from maichart.generation.event_cli import build_parser

    required = ["--checkpoint", "c.pt", "--audio", "a.mp3", "--output", "o.txt"]
    args = build_parser().parse_args(required)
    assert args.intensity_follow is True, "following the music is the default"
    assert args.intensity_follow_span == GENERATION_SAMPLING_DEFAULTS["intensity_follow_span"]
    assert args.tail_damping is False, "the outro prior is opt-in"
    assert args.tail_damping_windows == GENERATION_SAMPLING_DEFAULTS["tail_damping_windows"]
    assert args.tail_damping_strength == GENERATION_SAMPLING_DEFAULTS["tail_damping_strength"]

    off = build_parser().parse_args(required + ["--no-intensity-follow", "--tail-damping"])
    assert off.intensity_follow is False and off.tail_damping is True
    explicit = build_parser().parse_args(required + ["--intensity-follow-span", "0.25"])
    assert explicit.intensity_follow_span == pytest.approx(0.25)


def test_decode_windows_accepts_and_applies_the_factors():
    import inspect

    from maichart.generation import event_cli

    parameters = inspect.signature(event_cli._decode_windows).parameters
    for name in ("intensity_follow", "intensity_span", "tail_damping",
                 "tail_damping_windows", "tail_damping_strength"):
        assert name in parameters, f"{name} must reach the window loop"
    source = inspect.getsource(event_cli._decode_windows)
    assert "window_factors(" in source and "tail_factors(" in source
    assert "scale_intensity(condition, intensity[index])" in source
    # The audit line must expose the factor and the density it produced.
    assert "intensity={intensity[index]:.2f}" in source
    assert "note_density={window_condition.radar['note_density']:.3f}" in source
    # The user's condition object (not a vector) is what arrives here, so the
    # named controls can be scaled per window.
    call = (ROOT / "src/maichart/generation/event_cli.py").read_text(encoding="utf-8")
    assert "audio, global_audio, condition, device,\n                windows=windows" in call


def test_server_forwards_the_new_flags():
    source = (ROOT / "src/maichart/webui/server.py").read_text(encoding="utf-8")
    for needle in ('args.append("--intensity-follow")', 'args.append("--no-intensity-follow")',
                   '"--intensity-follow-span"', 'args.append("--tail-damping")',
                   '"--tail-damping-windows"', '"--tail-damping-strength"'):
        assert needle in source, f"server must pass {needle}"
    assert 'payload.get("intensity_follow", GENERATION_SAMPLING_DEFAULTS["intensity_follow"])' in source
    assert 'payload.get("tail_damping", GENERATION_SAMPLING_DEFAULTS["tail_damping"])' in source


def test_audible_bounds_find_the_music_inside_silence():
    """The real failure: music audible 2.44-142.94s, notes written outside it."""

    from maichart.generation.intensity import audible_bounds, loudness_profile

    # 2 s of digital silence, 10 s of music, 3 s of trailing silence (50 Hz).
    values = [0.0] * 100 + [1.0] * 500 + [0.0] * 150
    profile = loudness_profile(FakeAudio(values), smoothing_sec=0.25)
    bounds = audible_bounds(profile, threshold=0.15)
    assert bounds is not None
    low, high = bounds
    assert 1.9 <= low <= 2.6, low       # the music starts at 2.0 s
    assert 11.9 <= high <= 12.6, high   # and ends at 12.0 s


def test_audible_bounds_ignore_a_single_loud_frame():
    """One blip must not define the song's start; the smoothing removes it."""

    from maichart.generation.intensity import audible_bounds, loudness_profile

    values = [0.05] * 400
    values[10] = 1.2  # a single loud frame at 0.2 s (1.2/12 frames < threshold)
    values[300] = 1.2
    profile = loudness_profile(FakeAudio(values), smoothing_sec=0.25)
    assert audible_bounds(profile, threshold=0.15) is None
    # Without smoothing the blip would define a boundary.
    raw = loudness_profile(FakeAudio(values), smoothing_sec=0.0)
    assert audible_bounds(raw, threshold=0.15) is not None


def test_audible_bounds_none_when_the_track_has_no_music():
    from maichart.generation.intensity import audible_bounds, loudness_profile

    assert audible_bounds(loudness_profile(FakeAudio([0.0] * 200))) is None
    assert audible_bounds(loudness_profile(FakeAudio([]))) is None
    assert audible_bounds(None) is None


def test_window_shares_and_trimmed_span():
    from maichart.generation.intensity import loudness_profile, trimmed_span, window_shares

    values = [0.0] * 100 + [1.0] * 500 + [0.0] * 200   # 16 s
    profile = loudness_profile(FakeAudio(values), smoothing_sec=0.0)
    # 120 BPM, 16-beat windows = 8 s each; window 1 covers 0-8 s (2 s silent),
    # window 2 covers 8-16 s (music to 12 s, then silence).
    shares = window_shares(profile, _windows(2), tempo_changes=TEMPO, first=0.0, threshold=0.15)
    assert shares[0] == pytest.approx(0.75, abs=0.02)
    assert shares[1] == pytest.approx(0.50, abs=0.02)
    # A window entirely inside the lead-in silence has nothing audible.
    silent = window_shares(profile, [(Fraction(0), Fraction(2))], tempo_changes=TEMPO, first=0.0)
    assert silent == [0.0]

    assert trimmed_span((2.0, 12.0), lead=0.0, song_duration=15.0) == (2.0, 12.0)
    assert trimmed_span((2.0, 12.0), lead=0.5, song_duration=12.2) == (1.5, 12.2)
    assert trimmed_span(None, lead=0.5) is None
    assert trimmed_span((5.0, 5.0), lead=0.0) is None


def test_trim_sequence_drops_events_outside_the_audible_span():
    """The intro burst and the post-music tail must not survive."""

    from maichart.generation.event_cli import _trim_sequence

    # 120 BPM: beat 4 = 2.0 s, beat 8 = 4.0 s, beat 24 = 12.0 s, beat 32 = 16.0 s.
    sequence = EventSequence([Fraction(4), Fraction(8), Fraction(24), Fraction(32)],
                             ["1", "2", "3", "4"])
    kept = _trim_sequence(sequence, Fraction(0), tempo_changes=TEMPO, first=0.0,
                          low_sec=3.0, high_sec=13.0)
    assert kept.beats == [Fraction(8), Fraction(24)]
    assert kept.note_texts == ["2", "3"]
    # Nothing removed -> the same object comes back (no needless copies).
    untouched = _trim_sequence(sequence, Fraction(0), tempo_changes=TEMPO, first=0.0,
                               low_sec=0.0, high_sec=99.0)
    assert untouched is sequence
    assert _trim_sequence(EventSequence([], []), Fraction(0), tempo_changes=TEMPO, first=0.0,
                          low_sec=1.0, high_sec=2.0).beats == []


def test_decode_windows_skips_silent_windows_and_reports_audible_share(monkeypatch):
    """A window with no audible music must cost no decode and produce no events."""

    from maichart.generation import event_cli

    calls = []

    def fake_decode(*args, **kwargs):
        calls.append(kwargs.get("prefix_ids"))
        return EventSequence([Fraction(1), Fraction(2)], ["1", "2"])

    monkeypatch.setattr(event_cli, "_decode", fake_decode)
    monkeypatch.setattr(event_cli, "audio_segment_tensor", lambda *args, **kwargs: torch.zeros((2, 8)))

    # Two 16-beat windows; only the second has music.
    audio = FakeAudio([0.0] * 400 + [1.0] * 400)
    results = event_cli._decode_windows(
        object(), VOCABULARY, audio, torch.zeros((4, 8)), _CONDITION, torch.device("cpu"),
        windows=_windows(2), tempo_changes=TEMPO, first=0.0, song_duration=16.0, bpm=120.0,
        max_tokens=8, temperature=0.8, sample=False, seed=1, guidance_scale=1.0,
        audio_bounds_trim=True, silence_window_share=0.05, silence_threshold=0.15,
        audio_bounds_lead=0.0,
    )
    assert len(calls) == 1, "the silent window must not be decoded"
    assert [len(window.sequence.beats) for window in results] == [2]


def test_cli_defaults_for_the_audio_bounds():
    from maichart.generation.event_cli import build_parser

    required = ["--checkpoint", "c.pt", "--audio", "a.mp3", "--output", "o.txt"]
    args = build_parser().parse_args(required)
    assert args.audio_bounds_trim is True, "staying inside the audible span is the default"
    assert args.audio_bounds_lead == 0.0, 'strict by default: no notes over silence'
    assert args.silence_threshold == GENERATION_SAMPLING_DEFAULTS["silence_threshold"]
    assert args.silence_window_share == GENERATION_SAMPLING_DEFAULTS["silence_window_share"]
    off = build_parser().parse_args(required + ["--no-audio-bounds-trim"])
    assert off.audio_bounds_trim is False
    tuned = build_parser().parse_args(required + ["--audio-bounds-lead", "1.5",
                                                 "--silence-threshold", "0.3",
                                                 "--silence-window-share", "0.25"])
    assert (tuned.audio_bounds_lead, tuned.silence_threshold, tuned.silence_window_share) == (1.5, 0.3, 0.25)


def test_file_span_filter_is_hard_even_without_audible_trim():
    from fractions import Fraction
    from maichart.event_timing import EventPoint, EventTempo
    from maichart.generation.event_cli import _filter_points_to_file_span

    tempo = [EventTempo(Fraction(0), 120.0)]
    points = [EventPoint(Fraction(-1), "1"), EventPoint(Fraction(1), "2"),
              EventPoint(Fraction(5), "3")]
    kept = _filter_points_to_file_span(points, tempo_changes=tempo, first=0.0, duration_sec=1.0)
    assert [p.note_text for p in kept] == ["2"]
def test_server_forwards_the_audio_bounds_flags():
    source = (ROOT / "src/maichart/webui/server.py").read_text(encoding="utf-8")
    for needle in ('args.append("--audio-bounds-trim")', 'args.append("--no-audio-bounds-trim")',
                   '"--audio-bounds-lead"', '"--silence-threshold"', '"--silence-window-share"'):
        assert needle in source, f"server must pass {needle}"
    assert 'payload.get("audio_bounds_trim", GENERATION_SAMPLING_DEFAULTS["audio_bounds_trim"])' in source


def test_panel_exposes_the_audio_bounds_fields():
    html = (ROOT / "src/maichart/webui/static/index.html").read_text(encoding="utf-8")
    app = (ROOT / "src/maichart/webui/static/app.js").read_text(encoding="utf-8")

    def field(element_id: str) -> str:
        found = html[html.index(f'id="{element_id}"'):]
        return found[:found.index(">")]

    assert "checked" in field("audioBoundsTrim"), "trimming is on by default"
    # Strict by default: 0.5s of tolerance put a real run's first note 0.46s
    # before the music started, which is the defect being fixed.
    assert 'value="0"' in field("audioBoundsLead")
    assert 'value="0.15"' in field("silenceThreshold")
    assert 'value="0.05"' in field("silenceWindowShare")
    for element_id in ("audioBoundsTrim", "audioBoundsLead", "silenceThreshold", "silenceWindowShare"):
        assert f'"{element_id}"' in app, f"{element_id} must be part of the panel state"
    assert '"audioBoundsTrim", "audioBoundsLead", "silenceThreshold", "silenceWindowShare"' in app
    for key in ("audio_bounds_trim:", "audio_bounds_lead:", "silence_threshold:",
                "silence_window_share:"):
        assert key in app, f"generation payload must carry {key}"
    # The help text must state the measured failure it fixes, including why
    # loudness alone is not enough (quiet attack at 1.98s, loudness at 2.40s).
    assert "146.12" in html and "1.98" in html and "2.40" in html


def test_panel_exposes_the_fields_and_sends_them():
    html = (ROOT / "src/maichart/webui/static/index.html").read_text(encoding="utf-8")
    app = (ROOT / "src/maichart/webui/static/app.js").read_text(encoding="utf-8")

    def field(element_id: str) -> str:
        found = html[html.index(f'id="{element_id}"'):]
        return found[:found.index(">")]

    assert "checked" in field("intensityFollow"), "intensity following is on by default"
    assert "checked" not in field("tailDamping"), "tail damping is off by default"
    assert 'value="0.5"' in field("intensityFollowSpan")
    assert 'value="4"' in field("tailDampingWindows")
    assert 'value="0.6"' in field("tailDampingStrength")

    for element_id in ("intensityFollow", "intensityFollowSpan", "tailDamping",
                       "tailDampingWindows", "tailDampingStrength"):
        assert f'"{element_id}"' in app, f"{element_id} must be part of the panel state"
    # Persisted + resettable, and actually sent to the server.
    assert '"intensityFollow", "intensityFollowSpan"' in app
    assert '"tailDamping", "tailDampingWindows", "tailDampingStrength"' in app
    for key in ("intensity_follow:", "intensity_follow_span:", "tail_damping:",
                "tail_damping_windows:", "tail_damping_strength:"):
        assert key in app, f"generation payload must carry {key}"
    # Explain the current algorithm rather than promoting a historical RMS
    # correlation measurement as proof of musical/emotional alignment.
    assert "多维音乐张力" in html and 'id="intensityMode"' in html
