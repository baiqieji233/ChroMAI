# Local Onset Alignment Evaluation

`tools/evaluate_onset_alignment.py` scores a trained `chart_onsets_v1` head on deterministic `EventCtxDataset` windows. It runs `encode_audio` and the local onset head only. It does not run the autoregressive decoder or inject the teacher plan.

The default selection is 128 uniformly spaced windows from the manifest's dataset order. Each row records the window index, song and source paths, beat range, tempo map, known level/difficulty, and requested/effective generation-default conditions after the checkpoint contract guard. Raw target, quality weight, and core-valid tracks come from the dataset; `pool_onset_targets` applies the same per-sample pooling policy as the model.

The global frame metrics include soft-target BCE and Brier score, the binary positive fraction at target `>= 0.5`, and weighted average precision. Average precision groups tied scores before each precision/recall step; it is not trapezoidal area under a sampled PR curve. Threshold rows at 0.3, 0.5, and 0.7 include precision, recall, predicted-positive bin counts, target-positive bin counts, and weighted confusion mass. Padding, margin, and zero-quality bins contribute no score.

`constant_sample_mean_brier_oracle` uses the evaluation slice's weighted target mean as a constant prediction. It is an oracle reference fitted on the same labels and is unavailable to a deployed model. It must not be interpreted as a deployable baseline.

The optional `--shuffle-acoustic` arm reuses `evaluate_event_tokens.shuffle_acoustic_frames`: it permutes local acoustic and rhythm rows while retaining timing-context columns, global memory, and conditions. This tests local audio-time alignment sensitivity. It does not prove audio autonomy because global memory and target-derived labels remain.

Example:

```powershell
python tools/evaluate_onset_alignment.py `
  --checkpoint .scratch/full_v3_stage1/onset_integration/checkpoint/event_last.pt `
  --manifest .scratch/full_v3_stage1/validation_manifest.json `
  --cache-dir cache/events --limit 16 --batch-size 2 --shuffle-acoustic `
  --output .scratch/full_v3_stage1/onset_integration/frame_evaluation.json
```

The report records checkpoint epoch and global step. Frame agreement measures correspondence with chart-author onset choices, not acoustic ground truth, free-running generation quality, or playability. This tool does not attempt exact-event peak matching because pooled bins do not preserve an explicit frame-center map for arbitrary adaptive-pooling ratios.
