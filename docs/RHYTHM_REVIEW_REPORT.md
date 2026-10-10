# Rhythm Review Markdown

Render one or more complete `run_rhythm_evaluation.py` reports with:

```powershell
python tools/render_rhythm_review.py `
  .scratch/full_v3_stage1/quality_probe_epoch7/evaluation_spatial.json `
  --output .scratch/full_v3_stage1/quality_probe_epoch7/QUALITY_REVIEW.md
```

Each chart uses its explicit `arm` field when present, otherwise the chart's parent directory in `generated/<arm>/song.chart.json` or `generated/<run>/<arm>/song.chart.json`. Arm summaries use equal chart weights; each cell shows the number of charts with that field over the arm's chart count. A `notes_at_or_after_audio_end` value is shown only when `audio_end_check_available` is true. A disabled/unavailable playability check does not turn zero-filled payload counts into measured zero conflicts. Missing fields remain `未测量`, never zero. If the input report has `artifact_completeness`, its expected/produced/missing/unverifiable counts determine the shown status; input/evaluated chart counts alone are not treated as proof that all generation artifacts were produced. Without that object, artifact completeness is reported as unverified. Multiple report files can be passed together to create one review document.

The document reports note/onset counts, onset-gap counts and shares, type ratios, longest consecutive blank duration, boundary counts, static playability status and remaining hard conflicts, plus reference onset P/R/F1 and pairing identity. SHA-256-verified and unverified/filename-only reference metrics are shown on separate arm-summary rows. Filename-only metrics remain visible, with their `filename_only_unverified` identity explicit. It also lists the start/end seconds and keys for the longest clockwise, counterclockwise, and two-key runs so a reviewer can audition those spans. These descriptive fields are not combined into a score or automatic judgment.

The output path is rejected when it resolves to any input report path. Inputs are read as UTF-8/UTF-8-BOM JSON, so paths and titles may contain spaces or Chinese text.
