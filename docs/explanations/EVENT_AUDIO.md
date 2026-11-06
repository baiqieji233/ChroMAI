# Event Audio Representation

`maichart.generation.event_audio` is the audio contract for the new
event-level model. It is intentionally separate from the existing
`maichart.audio` and `maichart.generation.inference` modules; those files and
their seven-value frame baseline are unchanged.

## Fixed Contract

The extractor produces a dense 50 Hz `EventAudio` object. This rate is an
audio-time contract and does not use or depend on Simai chart `division`:

```python
from fractions import Fraction

from maichart.generation.event_audio import (
    EventAudio, FEATURE_KEYS, INPUT_DIM, audio_segment_tensor,
    compress_event_audio, extract_event_audio,
)

audio = extract_event_audio("song.wav", cache_dir="cache/event_audio")
x = audio_segment_tensor(
    audio, 12.0, 20.0,
    song_duration=audio.duration_sec,
    bpm=180.0,
    first=-0.25,
    start_beat=36,
    tempo_changes=[(Fraction(0), 180.0), (Fraction(64), 150.0)],
)
assert x.shape[1] == INPUT_DIM == 100
```

`EventAudio.features` is `[T, 95]`, `float32`, and contains:

1. `logmel_00` through `logmel_79`: 80 log-mel channels.
2. `chroma_00` through `chroma_11`: 12 chroma channels.
3. `onset_strength`, `rms`, `spectral_flux`: robustly normalized onset,
   energy, and spectral flux channels.

`audio_segment_tensor` adds five timing channels, so the complete
`FEATURE_KEYS` order is:

```python
(
    "logmel_00", "logmel_01", "logmel_02", "logmel_03", "logmel_04",
    "logmel_05", "logmel_06", "logmel_07", "logmel_08", "logmel_09",
    "logmel_10", "logmel_11", "logmel_12", "logmel_13", "logmel_14",
    "logmel_15", "logmel_16", "logmel_17", "logmel_18", "logmel_19",
    "logmel_20", "logmel_21", "logmel_22", "logmel_23", "logmel_24",
    "logmel_25", "logmel_26", "logmel_27", "logmel_28", "logmel_29",
    "logmel_30", "logmel_31", "logmel_32", "logmel_33", "logmel_34",
    "logmel_35", "logmel_36", "logmel_37", "logmel_38", "logmel_39",
    "logmel_40", "logmel_41", "logmel_42", "logmel_43", "logmel_44",
    "logmel_45", "logmel_46", "logmel_47", "logmel_48", "logmel_49",
    "logmel_50", "logmel_51", "logmel_52", "logmel_53", "logmel_54",
    "logmel_55", "logmel_56", "logmel_57", "logmel_58", "logmel_59",
    "logmel_60", "logmel_61", "logmel_62", "logmel_63", "logmel_64",
    "logmel_65", "logmel_66", "logmel_67", "logmel_68", "logmel_69",
    "logmel_70", "logmel_71", "logmel_72", "logmel_73", "logmel_74",
    "logmel_75", "logmel_76", "logmel_77", "logmel_78", "logmel_79",
    "chroma_00", "chroma_01", "chroma_02", "chroma_03", "chroma_04",
    "chroma_05", "chroma_06", "chroma_07", "chroma_08", "chroma_09",
    "chroma_10", "chroma_11",
    "onset_strength", "rms", "spectral_flux",
    "song_progress", "song_duration", "local_tempo_bpm",
    "beat_phase_sin", "beat_phase_cos",
)
```

Therefore the fixed model input is **`INPUT_DIM = 100`**, not the legacy 7
dimensions. The acoustic base is kept separate in the cache because timing
context depends on chart ground truth or a user override.

The five context values are:

- `song_progress`: frame timestamp divided by full song duration, clipped to
  `[0, 1]`.
- `song_duration`: duration normalized to 300 seconds, clipped to `[0, 1]`.
- `local_tempo_bpm`: local BPM divided by 240. The raw BPM is not lost; this
  is a stable model-scale representation.
- `beat_phase_sin`, `beat_phase_cos`: the exact fractional phase of the full
  song chart beat. Together they avoid the discontinuity at beat 0/1.

## Audio Time and Chart Time

The anchors must be explicit:

- `first` is the audio time in seconds at chart beat 0 for the **whole song**.
  It is the same semantic as `&first`.
- `start_sec` and `end_sec` are audio timestamps in the whole-song audio.
- `start_beat` is the full-song chart beat at `start_sec`; it is not reset to
  zero for a segment. It is retained as an explicit API argument for segment
  sampling and callback maps.
- A tempo list is a full-song map of `(beat, bpm)`, for example
  `[(Fraction(0), 180), (Fraction(64), 150)]`. Beat boundaries are chart
  beats, not seconds.

For a normal piecewise-constant map, time is integrated from full-song beat 0
using `first`. This means `first`, `start_beat`, and `tempo_changes` must agree:
for example, with `first=0`, BPM 120, `start_sec=0.5`, the corresponding
`start_beat` is 1. Supplying a GT map is preferred during training. A callback
`tempo_changes(beat) -> bpm` is also accepted for continuously warped maps;
the callback is evaluated by small-step integration.

If an anchor is not known, use `first=0.0`, `start_beat=0.0`, and an explicit
user BPM only as a declared approximation. Do not infer chart beat phase from
the audio-only tempo candidate metadata.

## Global Memory

The event model may use a full-song memory in addition to a local segment:

```python
global_memory = compress_event_audio(
    audio,
    song_duration=audio.duration_sec,
    bpm=180,
    first=0.0,
    tempo_changes=gt_tempo_changes,
    frames=64,
)
```

This returns `[64, 100]` by default. Acoustic rows are resampled, but context
is evaluated at each memory timestamp. In particular, beat phase is represented
as a unit-circle pair and is not averaged as a scalar, so the global memory
keeps full-song progress, local tempo, and phase in the same dimensions as a
segment tensor.

## Tempo Policy

Extraction records `tempo_bpm` and `tempo_changes` as **audio-time candidates**
only. The default downstream interpretation is constant tempo. Candidate
changes require a minimum relative change of 8%, at least four consistent
intervals, local median smoothing, and 0.5x/1x/2x normalization against the
global estimate. This avoids turning a small `129 -> 132` estimator drift into
a real tempo event and reduces half/double-tempo errors.

The candidate metadata has `candidate: true` and a confidence value. It does
not claim that tempo estimation is completely solved. Generation should use:

1. GT tempo map for training local BPM context and timing supervision.
2. A user BPM override for production when the chart timing is known.
3. Constant `tempo_bpm` as the conservative fallback.

If an audio candidate is promoted by an application, it must first be mapped
from audio seconds to chart beats using the same `first` anchor and then
reviewed; this module does not silently promote it.

## Cache API

`extract_event_audio(..., cache_dir=...)` writes two files:

- `*.event_audio.json`: schema, feature configuration, source digest,
  timestamps, duration, tempo metadata, and the NPZ filename.
- `*.event_audio.npz`: compressed `float32` features and times.

The filename and metadata include `event-audio-v1`, sample rate, hop length,
FFT/bin settings, normalization version, and the SHA-256 digest of the whole
audio file. A cache with a different feature configuration or metadata version
is rejected; it is never reused as if compatible. `save_event_audio` and
`load_event_audio` are available for explicit cache operations.
`load_event_audio(path, sample_rate=..., hop_length=...)` can additionally
perform a strict caller-side configuration check; the no-argument form still
validates the cache schema and its self-consistent configuration. The explicit
aliases `save_event_audio_json`/`load_event_audio_json` describe the paired
JSON+NPZ format, while `save_event_audio_npz`/`load_event_audio_npz` provide a
self-contained binary format.

## Dependencies and Tests

The implementation uses `librosa`, `numpy`, `soundfile` for test WAVs, and
`torch` only when constructing model tensors. It does not download or require
large pretrained weights.

Run the focused tests with:

```powershell
python -m pytest tests/test_event_audio.py -q
```
