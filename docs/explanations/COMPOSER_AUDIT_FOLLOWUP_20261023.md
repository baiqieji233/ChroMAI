# Composer audit follow-up

## Confirmed and repaired

- `PlanGrid` no longer lets a `[0, first]` or tail sliver dominate planning
  interpolation. Boundary slivers are merged when materially shorter than the
  adjacent span, and planner weights include temporal support. This prevents a
  single event from becoming an artificial extreme events-per-second target.
- Recovery proposals now carry the causal `IncrementalState` and use the same
  occupied/blocked-head filtering and bounded action retries as generation.
  Recovery no longer holds the dataset-wide feature lock during model rollout;
  per-song table/action-state construction is synchronized independently.
- Diagnosis rebuilds an evicted acoustic context instead of indexing the LRU
  directly, and masks candidates outside the known audio end.
- Inference skips the repeated CPU LensBank query and host-to-device copy when
  all candidate embeddings are already cached. The grounded model accepts that
  inference path without requiring redundant raw patches.
- Resume now restores structural and policy arguments from the checkpoint in the
  CLI, main WebUI, and standalone Composer page. Explicit user overrides remain
  rejected for exact resume. The new planning policy changes the grounded
  training semantic version, so old v7 grounded checkpoints require finetune.
- Invalid-candidate residual diagnostics are now masked, and action-factor
  relation computation reuses normalized static lane buffers.

## Deliberately not treated as confirmed defects

- The claimed 1.9x model slowdown attribution needs a profiler trace; no broad
  kernel rewrite was made from static inspection alone.
- Slide `<`/`>` geometric orientation may deserve a future domain evaluation,
  but changing factor semantics without a corpus-level validation would corrupt
  the action contract.
- Removing the unused inherited `norm` parameter would alter checkpoint state
  layout for no behavioral gain, so it was left compatible.

## Validation

Python focused Composer suite: 73 passed. Composer WebUI Node suite: 9 passed.
Modified Composer modules compile with `py_compile`; `git diff --check` passes.
No long training run or full generation benchmark was started.
