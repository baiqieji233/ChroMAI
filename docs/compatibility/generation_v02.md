# Conditional generation v0.2

This version is a runnable research baseline, not a finished Master chart
authoring system.

## Pipeline

```text
maidata.txt + audio
  -> ChartIR / frame labels / cached audio features
  -> conditioned frame Transformer
  -> note presence + button mask + broad note type
  -> deterministic Simai compiler
  -> maidata.txt and optional ChartIR JSON
```

The model predicts at a fixed beat grid. With the default `division=16`, each
frame advances by `4/16 = 1/4` beat and therefore uses 480 integer ticks when
`TICKS_PER_BEAT=1920`. The compiler writes `(BPM)` changes at beat positions;
the tick coordinate does not change when BPM changes. An optional tempo map is
accepted as JSON:

```json
[
  {"beat": "0", "bpm": 120},
  {"beat": "32", "bpm": 150}
]
```

The compiler integrates the tempo map when filling `ChartIR.time_sec`. When no
`--bpm` or `--tempo-map` is supplied, generation uses the local tempo changes
stored by audio analysis. Boundaries are converted from audio seconds to chart
beats relative to `--first`, quantized to the selected frame grid, and emitted
as `(BPM)` directives. `--tempo-map` remains the exact manual override;
supplying `--bpm` explicitly selects a constant-BPM interpretation.

## Install

```powershell
py -m pip install -e ".[dev,audio,training]"
```

The audio extra is only needed when passing `--audio`; precomputed
`audio_features.json` can be used without loading librosa during inference.

## Prepare data

The inherited preprocessing commands build ChartIR, frame labels, audio
features, alignment reports, and a song-level split:

```powershell
maichart preprocess build raw_chart_data/output `
  --cache-dir cache `
  -o manifests/training_manifest.json `
  --division 16
maichart preprocess split manifests/training_manifest.json `
  --out-dir manifests/splits --seed 42 --split-by-song
```

For training, each manifest difficulty must have usable ChartIR, frame-label,
alignment, and audio-feature cache files. Splitting by song is important: the
same song must not appear in both train and validation sets.

## Train

```powershell
maichart-train `
  --manifest manifests/splits/train.json `
  --cache-dir cache `
  --checkpoint-dir checkpoints/generator `
  --segment-frames 512 `
  --max-len 4096 `
  --batch-size 2 `
  --grad-accumulation 4 `
  --d-model 256 --nhead 4 --num-layers 4 `
  --epochs 20
```

`segment-frames` controls training memory. `max-len` controls the Transformer
position encoding and inference window; long songs are decoded in windows.
On a 12 GB laptop GPU, start with `d_model=256`, four layers, batch size 1 or
2, and AMP enabled. Increase width only after the data pipeline overfits a
small sample and the generated timing is correct.

## Generate

Using cached features:

```powershell
maichart-generate `
  --checkpoint checkpoints/generator/generator_last.pt `
  --audio-features cache/audio_features/song.audio_features.json `
  --bpm 180 `
  --condition-file configs/generation_condition.example.json `
  --output generated/maidata.txt `
  --output-ir generated/chart.json
```

Or analyze an audio file during inference:

```powershell
maichart-generate `
  --checkpoint checkpoints/generator/generator_last.pt `
  --audio song.wav --bpm 180 `
  --level 13.5 --output generated/maidata.txt
```

`--candidates` defaults to `1`. Passing `--candidates 8` manually enables
stochastic candidate emission and writes numbered files. There is no ranking,
gameplay harness, or infeasible-target repair in v0.2 by design. The default
path therefore remains easy to inspect and reproduce.

## Training conditions

The raw ZIP and preprocessing manifest contain only audio/chart supervision:
maidata, ChartIR, frame labels, audio features, and alignment reports. Radar
and Style are not stored as dataset fields. `GeneratorDataset` derives both
condition vectors from the loaded frame labels in memory for each sample, so
changing the derivation code does not require rewriting every ZIP.

## Current limitations

- Frame labels supervise broad type and position decisions; exact hold/slide
  durations and slide geometry are currently fixed by the compiler.
- Style fields such as melody following and pattern reuse are accepted as
  controls, but the inherited frame labels cannot fully supervise them yet.
- Radar values are normalized project coordinates, not an implementation of a
  proprietary official radar formula.
- The model is non-autoregressive over frames. The next major upgrade is an
  event-level decoder with exact duration/path heads and motif memory.
