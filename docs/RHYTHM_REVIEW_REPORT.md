# Rhythm Review Markdown

Render one or more complete `run_rhythm_evaluation.py` reports with:

```powershell
python tools/render_rhythm_review.py `
  .scratch/full_v3_stage1/quality_probe_epoch7/evaluation_spatial.json `
  --output .scratch/full_v3_stage1/quality_probe_epoch7/QUALITY_REVIEW.md
```

Each chart is assigned to the directory immediately below `generated/` in its recorded path. Arm summaries use equal chart weights; each cell shows the number of charts with that field over the arm's chart count. Missing reference, boundary, quality, or playability fields remain `未测量`, never zero. Multiple report files can be passed together to create one review document.

The document reports note/onset counts, onset-gap counts and shares, type ratios, blank seconds, boundary counts, static playability status and remaining hard conflicts, plus reference onset P/R/F1 and pairing identity. It also lists the start/end seconds and keys for the longest clockwise, counterclockwise, and two-key runs so a reviewer can audition those spans. These descriptive fields are not combined into a score or automatic judgment.

The output path is rejected when it resolves to any input report path. Inputs are read as UTF-8/UTF-8-BOM JSON, so paths and titles may contain spaces or Chinese text.
