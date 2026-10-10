# Rhythm Evaluation Prep

`tools/run_rhythm_evaluation.py` reports descriptive chart evidence without combining it into a score. It accepts generated event chart JSON, wrapper `.chart.json` output, or ChartIR JSON whose `raw` field contains the source maidata. Tempo changes and exact beats are read from the serialized event chart; the tool uses top-level `first` as audio_first when present, otherwise it integrates the embedded chart's `first + time_offset`.

```powershell
python tools/run_rhythm_evaluation.py webui_workspace/generated/baseline webui_workspace/generated/v3 `
  --output .scratch/rhythm_evaluation.json
```

The report separates event type counts/ratios, notes per second and one-second type counts, blank measures and longest blank run, repeated event signatures/motifs, duplicate heads, and notes before/after audio or declared chart end. Rhythm details use sorted, distinct onset beats: `fine_note_ratio` keeps the training target definition (gaps at most 33/128 beat), while the exact rational gap histogram and 1/3-, 1/4-, and 1/6-beat counts make other subdivisions visible. `short_gap_share` is a separate descriptive share for gaps at most 1/2 beat. Repeated onset type patterns ignore button identity and remain separate from `spatial_repetition`, which describes long single-Tap paths and two-key alternation.

`spatial_repetition` is descriptive evidence, not a bad-chart score or a claim about hand assignment. Eligible events are onsets containing exactly one parsed Tap at button 1–8; Break taps are included. A chord or any onset containing Hold, Slide, Touch, or another non-Tap event interrupts a path. A gap greater than one beat also disconnects it. `1→2→...→8→1` is called clockwise and the reverse counterclockwise. The two-key alternation requires at least three onsets, such as `1,2,1`; a two-note pair alone is not a repeated pattern. Each reported path contains its onset count, exact beat endpoints, seconds endpoints, and minimum/maximum observed gaps in beats and seconds. Sequences can cross bars and tempo changes, but never cross the interruption rules above.

Audio duration is trusted from `audio_duration_sec` or top-level `duration_sec`. The legacy `quality.duration_sec` fallback is accepted only when `audio_first` is exactly zero; with a nonzero first timestamp it stays unavailable because that legacy field may describe chart span rather than file duration. `--reference-dir` pairs exact relative filenames. If both charts contain `audio_sha256`, a mismatch is marked incompatible and no deltas are reported; equal hashes are verified. Missing hashes retain the legacy matched count under `matched_unverified` with `filename_only_unverified` identity status. Per-second density deltas and onset alignment describe agreement with that one reference only; they do not establish complete chart intent. Onset alignment counts distinct onset times within 50 ms and reports precision, recall, and F1.

Directory discovery skips `.history` trees and `*.original.chart.json` shadows. For a compatible matched reference, `reference_pair.spatial_repetition` retains the generated and reference descriptive structures side by side. Stored static playability diagnostics and coverage are copied into the report with `verified: false`; they are evidence from the chart JSON, not an independent or complete playability check. Checkpoint identity fields are retained as provenance and do not establish audio pairing. Missing duration/end metadata stays marked unavailable. The evaluator does not select a nearest reference or award more notes a better result.

## Existing Artifacts

The current local chart JSONs were evaluated into `.scratch/existing_chart_rhythm_metrics.json`. The two `v3_gen` artifacts each contain 41 notes, about 0.336 notes/s, 78.4% blank measures, a 17.6-second longest blank run, and no duplicate heads; the reported aggregate metrics are the same for both. `generated/maidata.chart.json` contains 773 notes, 4.81 notes/s, 4.9% blank measures, and a 3.79-second longest blank run. The constrained seafoodcup2 sample contains 7 notes and 50% blank measures. These charts are not paired to the same audio/reference, and their JSON does not establish a verified baseline checkpoint identity; the numbers describe those files only. The seafoodcup2 sample remains secondary evidence.

## Source Audit

The full source is `webui_workspace/datasets/full训练集/training_manifest.json`: 1,525 songs and 5,938 difficulties, with 1,872 `event_usable_for_training` difficulties. Every one of its 1,525 maidata paths and 1,525 audio paths exists under the dataset root. The legacy `usable_for_training` flag is false/missing for these entries, so the existing `maichart preprocess split` helper would select zero songs from this manifest.

The existing `webui_workspace/v3_throughput/event_last.pt` is a one-epoch composed-source checkpoint, not a full-training checkpoint. Its `dataset_audit.json` names `v3_full_train.json` (872 charts / 21,411 windows) and `v3_full_val.json` (51 / 1,108); `merged_from` lists `E:\后训练（官谱）\training_manifest.json` and `E:\后训练（自制）\training_manifest.json`. Keep it as a separate composed-data reference.

## Fixed Panel

The reproducible 10-song panel is `.scratch/rhythm_eval_10song_panel.json`. It was selected from 1,858 readable event-usable charts (14 were invalid or empty) at difficulty indices 4–6 and levels 12–15. The entries preserve absolute maidata/audio paths. Selection bases use measured values: minimum and maximum note density, maximum hold/touch/slide/chord/fine ratio, then density quartiles. These are numeric selection rules, not human-written style labels. Use the same `audio_path` in every model run; do not use `gold_val_manifest.json` for this panel.

For the stage1 heldout comparison, `.scratch/full_v3_stage1/heldout_panel.json` is selected only from `.scratch/full_v3_stage1/validation_manifest.json`. The panel builder reads each eligible audio file, caches its SHA-256 by path during that invocation, and drops repeated content hashes even when song IDs differ. It records each selected audio digest, the complete tempo map, `first`, `time_offset`, `audio_first`, and `end_beat`. The split report is checked against the exact validation-manifest bytes, and candidate hashes present in its training set are excluded. This protects stage1 fine-tune comparisons from train audio. The existing baselinev2 may already have seen the full source collection, so this panel does not establish a blind baselinev2 evaluation.

Rebuild and re-audit the heldout panel with:

```powershell
python tools/run_rhythm_evaluation.py `
  --panel-manifest ".scratch/full_v3_stage1/validation_manifest.json" `
  --panel-split-report ".scratch/full_v3_stage1/split_report.json" `
  --panel-output ".scratch/full_v3_stage1/heldout_panel.json" `
  --output ".scratch/full_v3_stage1/heldout_panel_evaluation.json" `
  --panel-size 10
```

| Song ID | Diff. | Selection basis | Notes/s | Hold | Touch | Slide | Chord | Fine |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 835 | 5 | Lowest density | 1.445 | 0.000 | 0.000 | 0.975 | 0.100 | 0.000 |
| 10070 | 5 | Highest density | 9.374 | 0.118 | 0.346 | 0.061 | 0.251 | 0.726 |
| 11597 | 5 | Highest hold ratio | 3.352 | 0.710 | 0.038 | 0.097 | 0.002 | 0.450 |
| 11948 | 5 | Highest touch ratio | 7.520 | 0.024 | 0.686 | 0.050 | 0.385 | 0.170 |
| 753 | 5 | Highest slide ratio | 1.708 | 0.000 | 0.000 | 1.000 | 0.275 | 0.000 |
| 378 | 5 | Highest chord ratio | 3.229 | 0.000 | 0.000 | 0.401 | 1.000 | 0.021 |
| 11333 | 6 | Highest fine ratio | 8.017 | 0.017 | 0.013 | 0.021 | 0.049 | 0.924 |
| 11731 | 5 | Density 25th percentile | 4.328 | 0.092 | 0.048 | 0.145 | 0.413 | 0.077 |
| 759 | 5 | Density 50th percentile | 5.002 | 0.051 | 0.000 | 0.206 | 0.315 | 0.042 |
| 11412 | 5 | Density 75th percentile | 5.805 | 0.039 | 0.028 | 0.146 | 0.094 | 0.000 |

## Full Split

The derived manifests are `.scratch/full_v3_stage1/train_manifest.json` and `validation_manifest.json`. They were grouped by SHA-256 of complete audio bytes, then assigned deterministically by `int(first_8_hex, 16) % 10`: bucket 0 is validation and buckets 1–9 are training. Thus different paths containing identical audio stay together. All selected difficulties are retained on their song, and their maidata/audio references are absolute. The original manifest and source files are unchanged.

| Split | Songs | Difficulties | Unique audio hashes |
|---|---:|---:|---:|
| Train | 1,343 | 1,695 | 1,342 |
| Validation | 145 | 177 | 145 |

The audit found 1,487 unique audio content hashes among the 1,488 songs with event-usable difficulties; one content hash is shared by two songs. The train/validation hash intersection is empty. Source and output SHA-256 values plus complete split hash lists are in `.scratch/full_v3_stage1/split_report.json`.

## Training Start Parameters

Do not resume from `v3_throughput`; start a fresh full-source run after reviewing the split report. For the 12 GB GPU, begin with microbatch 1 and gradient accumulation 8; AMP remains enabled by default. This keeps the requested model dimensions while limiting the initial activation footprint:

```powershell
python -m maichart.training.train_event_ctx `
  --manifest ".scratch/full_v3_stage1/train_manifest.json" `
  --validation-manifest ".scratch/full_v3_stage1/validation_manifest.json" `
  --cache-dir "webui_workspace/datasets/full训练集/cache" `
  --checkpoint-dir "checkpoints/event_ctx_full_v3_stage1" `
  --epochs 15 --batch-size 1 --grad-accumulation 8 `
  --segment-beats 16 --max-tokens 1024 `
  --d-model 320 --nhead 8 --num-layers 8 --dim-feedforward 1280 `
  --planner-profile music_structure_v3 `
  --context-beats 8 --audio-margin-beats 4 --context-tokens 192 --overlap-beats 0 `
  --device cuda --audio-cache-gb 4 --cuda-memory-budget-gb 7.2 `
  --seed 20261010
```

Review the first epoch's peak memory and per-batch time before changing batch size. Compare the generated charts with this fixed panel using the same 10 audio files; a reference delta is only reported when a chart pair is supplied with an exact matching relative filename.
