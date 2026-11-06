# Training Dataset Rhythm Report

`scripts/annotate_rhythm_patterns.py` reads a `training_manifest.json` and its maidata sources, then writes a separate JSON report. It does not extract audio, train a model, or modify the manifest or chart files.

```powershell
python scripts/annotate_rhythm_patterns.py `
  --dataset webui_workspace/datasets/full训练集 `
  --output webui_workspace/datasets/full训练集/rhythm_report.json
```

`--dataset` accepts either a manifest file or a dataset directory containing `training_manifest.json`. If `--output` is omitted, the tool writes `<manifest-stem>.rhythm_report.json` next to the manifest. An output path that resolves to the source manifest is rejected.

Windows are 16 beats by default. Use `--window-beats` to change the length. Repeat `--difficulty` to select difficulty indices, pass `--min-level` and `--max-level` for an inclusive level range, or use `--limit` to analyze only the first eligible chart difficulties. For example:

```powershell
python scripts/annotate_rhythm_patterns.py `
  --dataset webui_workspace/datasets/full训练集/training_manifest.json `
  --difficulty 5 --difficulty 6 --min-level 12 --max-level 15 `
  --limit 20 --output rhythm_sample.json
```

The report includes aggregate and per-level `fine_note_ratio` distributions, counts of filtered and skipped difficulties by reason, per-chart window rows, and up to eight nonempty example windows. Beat coordinates are exact rational strings; seconds are calculated through the chart tempo map and include the chart's audio offset. The ratio uses the shared `music_plan_targets.fine_note_ratio` definition, so 16th-note spacing and tuplets follow the same boundary as planner targets.

Only `maidata_path` and the difficulty index are needed to parse a chart, matching the event dataset loader. Relative source paths are resolved using the manifest folder, declared `source_root`, parent-folder fallback, and the loader's one-level moved-manifest fallback. Composed manifests with absolute source paths work directly.
