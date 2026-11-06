# Event Token Diagnostics

`tools/evaluate_event_tokens.py` evaluates an `event_ctx_v1` checkpoint on deterministic, evenly spaced validation windows. It uses the normal event dataset, `collate_events`, and causal model `forward` path. The chart's correct preceding tokens are supplied to the decoder, so this measures teacher-forced next-token behavior. It does not run free generation or certify chart quality.

The diagnostic uses the generation defaults for radar/style and each chart's level and difficulty metadata. It does not pass `plan_target`; the model's predicted planner output is used. Context target tokens and padding are excluded by the dataset's existing loss mask and collator mask.

For an alignment ablation, pass `--audio-ablation shuffle_acoustic`. It makes a second forward pass on the same windows after deterministically permuting only local acoustic and audio-derived rhythm frames. Pure time/beat columns, full-song memory, conditions, correct decoder history, and target tokens stay fixed; the predicted planner is recomputed from the shuffled input. This breaks local audio-time alignment; it is not an audio-autonomy score.

The JSON reports overall scored-token accuracy, exact rational gap groups for `AT` top-1 accuracy, character-token accuracy and true-token probability for Hold/Slide/Touch syntax markers, ordinary syntax characters separately, and how often `<event_end>` is predicted where a chart character is expected. `/` chord separators and `*` slide-branch separators have their own groups, since neither alone identifies a Slide note. These character metrics are not full note-type F1. Each window retains its ID, source, interval, tempo metadata, supplied conditions, predicted planner values, and metrics.

Example:

```powershell
python tools/evaluate_event_tokens.py `
  --checkpoint checkpoints/music_structure_v3_full_stage1/event_epoch_007.pt `
  --manifest .scratch/full_v3_stage1/validation_manifest.json `
  --cache-dir cache/events --limit 128 --output .scratch/full_v3_stage1/event_tokens_epoch7.json
```
