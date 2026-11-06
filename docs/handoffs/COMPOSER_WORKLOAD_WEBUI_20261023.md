# Composer Workload WebUI Handoff

Date: 2026-10-23

## Scope

This change integrates the Composer learned workload controls and diagnostics
into the main workbench. It does not change other model families or route the
audio-reference curve editor into Composer generation. The separate Composer
page remains outside the main workflow.

## File-Level Changes

1. `src/maichart/webui/static/index.html`: adds the Composer-only numeric
   `composerCurveStrength` field, range 0 through 2, default 1. Existing
   device, temperature, seed, and condition controls remain shared as before.
   The audio-analysis note now distinguishes its reference-only curve from
   the workload curve stored in the generation report.
2. `src/maichart/webui/static/app.js`: persists the new field with the other
   generation controls and forwards it as `curve_strength`. Composer training
   shows `workload_loss`, `curve_slope_loss`, and `workload_rate_mae`. The MAE
   label explicitly identifies log1p density units; it is not a percentage or
   an audio-fidelity score.
3. `src/maichart/webui/composer_api.py`: validates a finite `curve_strength`
   in [0, 2], forwards `--curve-strength`, and records the effective value in
   the task snapshot. Invalid values fail before creating a task. Default is
   1 for API callers that omit the field.
4. Composer generation summaries report whether a learned workload curve is
   present, the number of curve timestamps, guidance strength, guard checks,
   rejections, skipped candidates, and the stop reason. Full `times_sec`,
   `event_rate`, `head_rate`, and type-fraction arrays remain available in the
   existing downloadable `.composer.json`; they do not block chart export.
5. Summaries and completion notices count `playability.remaining_hard`, not
   merely `remaining_actionable_hard`. A report-only hard finding therefore
   remains visible even if zero findings are eligible for automatic repair.
   Partial detector coverage is also shown as needing review.
6. Generation contract is `composer_generation_controls_v5_workload_sampling`.
   The main UI rejects an older service before submitting controls that the
   old process might ignore. The main implementation must publish the same
   contract from `/api/status` in `server.py` and restart the served WebUI.

## Tests Prepared

- `tests/test_composer_generation_controls.py`: explicit and default strength
  forwarding; effective settings; rejection of out-of-range/nonfinite input.
- `tests/test_composer_generation_ui.js`: control range/default, payload,
  service contract, missing curve, guidance disabled at zero, guard counters,
  and hard findings that are report-only.
- `tests/test_composer_training_ui.js`: numeric workload/slope losses and
  log1p density error without percentage formatting.

No tests or GPU work were run by this UI implementation agent. Run these in
the final staged verification after Composer core changes are complete.

## Limits

The curve is learned workload, not an objective measure of musical quality.
A zero remaining-hard count is a detector result, not player certification.
Old checkpoints can lack workload capability, which is shown explicitly.
