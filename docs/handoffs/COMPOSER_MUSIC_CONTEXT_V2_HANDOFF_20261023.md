# Composer music-context v2 — implementation handoff

## Scope and backup

2026-10-23. User authorized model/data/generation changes and stopping training.
Backup: `backups/composer_music_alignment_20261023_101702/source_before.zip`;
the same directory contains the pre-existing working-tree patch and active-task
snapshot. Task `657e02a68227` (Prism_composer_v5) was cancelled via WebUI. Its
checkpoints remain intact. Existing uncommitted changes belong to earlier work
and are included in the backup. No automatic fresh restart is requested here.

## Decisions before implementation

- Version new training as `music_context_v2`; old models retain their own
  architecture and generation policy. No silent resume across contracts.
- Add descriptive committed-to-candidate acoustic interval evidence, recent
  history cross-attention, an audio-only pointer objective, and coherent
  symbolic-context dropout. Do not invent recovery labels by attaching an
  arbitrary generated prefix to an authored suffix.
- Match predicted trailing 1s/4s event/head rates with causal ledger windows;
  retain authored symmetric workload supervision and predicted-only inputs.
  Excluded chart regions are unknown; suppress supervision, never label rest.
- Share the legal-atom output matrix across conditional slots. Keep STOP
  supervision and the full legal catalog; do not pack away zero targets.
- Cache ordered authored-prefix statistics and reduce repeated acoustic-bank
  copying and per-tensor CUDA synchronizations. No full per-event x candidate
  feature materialization, lossy cache, or altered sampler for speed claims.
- Supply lightweight workload/ledger/history interventions, phase-recovery and
  coverage diagnostics. They report evidence, never block chart output.

Testing starts only after the implementation stage is complete. Short tests
prove contracts and numerical paths, not musical quality or absence of loops.

## Implementation and verification

### Stage 1: core implementation complete

- `model.py`: optional `music_context_v2`, candidate-to-history attention,
  audio-only pointer CE, coherent chart-modality dropout (default .15),
  trainable 1s/4s trailing-rate heads and detached same-window discrepancy,
  shared legal-atom output plus slot embeddings. Symmetric curve head remains
  detached from decision losses, while its projection trains normally.
- `data.py`, `workload.py`: uniform-only whole-song symmetric workload labels
  on v2, matching trailing labels with channel-specific exclusion masks,
  reusable ordered head prefixes, bounded authored type/rhythm exit weighting,
  canonical within-event atom ordering. Target labels never become model input.
- `candidates.py`: `absolute_recovery_v5` includes fine/coarse phase escape
  anchors and sparse 24/32/48/64/96/128-beat rest choices, bounded by 64 seconds
  and the file end. v4 remains unchanged for old checkpoints. This still does
  not guarantee arbitrary Fraction reachability; miss categories are logged.
- `generate.py`: matching interval/trailing inputs, committed anchor distinct
  from search cursor, canonical slot ordering only for v2, incremental head
  prefixes, history coverage in seconds/beats. No musical density hard gates.
- `checkpoint.py`: old generation remains supported; v1 -> v2 initialization
  is explicit finetune with averaged slot weights and new trainable paths.
  It is approximate, not equivalent generation. Optimizer resume across this
  change is rejected. Fresh/finetune cannot overwrite an existing last weight.
- `acoustics.py`: validated native banks avoid repeated full value scans,
  argsorts and full-song indexed copies when constructing compact contexts;
  the untrusted/legacy constructor keeps its sanitizing path.
- `train.py`: consolidated finite checks preserve rejection of bad updates;
  global gradient norm replaces per-parameter scalar synchronization. Fixed
  batch overrun when locality exceeds batch size, invalid accumulation-window
  handling and FP16 skipped-update scaler state. Optional CUDA events report
  H2D/forward/backward/update timings; prefetch statistics accompany logs.
- WebUI main/standalone controls pass dropout, auxiliary weight, and optional
  profiling batch count. Dashboard shows the two new losses. Server training
  capability was versioned to reject an old still-running backend.
- `diagnose.py`: read-only fixed-state workload and ledger interventions,
  reordered acoustic evidence, coherent symbolic omission, consistent deletion
  of the most recent event with state/ledger/candidates recomputed, and exact
  one/two-step phase reachability. Altered prefixes get no invented CE label.

Tests begin after this stage. No full-scale training or listening claim is
made. Additional proposals (prototype/factorized atoms, learned rest cursors,
weak-label critics, arbitrary repeated-history augmentation, source separation,
whole-song memory) remain separate experiments, not assumed fixes.

### Stage 2: short verification and baseline observations

The initial focused suite reached 55 passes and found a genuine new
list/NumPy-index mismatch in generation's incremental head prefix. Fixed at
the prefix-query boundary; generation smoke and old double-Hold continuity
regressions then passed. A stale tiny resume-test dataset also needed the
song-group metadata required by the existing locality sampler.

Read-only v5 checkpoint interventions were run on ONE chart and THREE events
(0, 175, 349), not a corpus quality evaluation. Artifact:
`.scratch/composer_v5_interventions_20261023.json`. At events 175/349,
symmetrical workload permutation changed pointer JS by about 1.2e-6/3.7e-6;
zeroing ledger changed it by .151/.055. Reordering acoustic evidence changed
it by .011/.002 (after retaining the timing/tempo/meter columns during the
acoustic-only permutation). These are heterogeneous interventions, not a causal variance
decomposition, and do not prove the ledger is wrong. They demonstrate weak
curve sensitivity in these specific states and motivate audio-only decision
supervision and coherent dropout rather than simply adding more curve losses.

The old v5 two-step geometric phase probe was reachable in the eight tested
cases; one needed TWO events. This does not prove learned/free-rollout recovery.
There were still pruned authored targets on that chart, and candidate misses
remain reported rather than silently injected as targets.

### Verification results

- Focused Python suite: **58 passed**, covering Composer train, generation,
  old double-Hold continuity, exact mid-epoch resume, workload, guards, mapped
  banks, memory budgets and new music-context paths.
- JavaScript: **11 passed**, including main-workbench controls and dashboard.
- `git diff --check`: passed. No full repository regression or long training.
- Read-only `diagnose` CLI completed both on the actual old v5 checkpoint
  (one chart, three states) and a tiny new v2 checkpoint. The uniform-coverage
  report recorded one unsampled 1s bin on the 123-bin old-chart probe set;
  uniform sampling is not a guarantee of every second being seen each epoch.
- Actual-data CUDA smoke: one real chart, 350/350 authored candidate recall,
  three BF16 updates with d_model=384/layers=6, batch=8, threaded prefetch,
  finite losses/gradients/weights, training and inference checkpoints saved
  under `.scratch/composer_music_v2_smoke`. This chart had 139 vocabulary
  entries; it does not benchmark a full-corpus epoch.
- The same three-update checkpoint completed a six-event CUDA generation
  smoke through guard, export and `.composer.json` curve reporting. This is
  deliberately too little training for a musical-quality comparison.
- Representative 18,454-vocabulary synthetic CUDA test on RTX 5070 Ti Laptop,
  batch32, 384d/6 layers, 3 updates each (last two timed): old 85,832,255 params
  / 56,838,320 action params / 2.204 GiB peak / 73.1ms per update; new
  36,715,626 / 7,104,790 / 1.353 GiB / 68.3ms. Single-event FP32 model-only
  inference in that short probe was 7.35ms old vs 8.59ms new. History attention
  has a cost; there is NO claim of universal inference acceleration. Actual
  generation skips the audio-only auxiliary branch. Artifact and reproducible
  local script: `.scratch/composer_music_v2_cuda_probe.json` and `.py`.
- WebUI 7860 serves the new capability `music_context_shared_slots_v2` and all
  three new controls; no active training task is left running.

### Operating/migration notes

Fresh training automatically uses `music_context_v2`. For an existing workload
checkpoint choose **finetune** with a NEW output directory, e.g.
`checkpoints/Prism_composer_v6`; averaging independent slot heads means this
needs actual adaptation and may initially change chord/STOP distributions.
Do not claim it preserves an old model's output or that three updates teach
the new paths. Old v5 remains independently usable for generation.

Recommended starting controls: symbolic dropout .15, audio-pointer weight .15,
curve strength 1.0. Profiling defaults off; set `profile_steps=8` for a small
timing sample. Keep existing batch/accumulation when comparing semantics.
For new architecture quality evaluation use matched data/budget, fixed audio
and seeds, weak/strong sections, rests, repeated choruses, transitions, tails,
and runtime MiaCode. Diagnostics are informational, not output acceptance gates.

Example diagnostic command (read-only, eight states, at most two charts):

```powershell
python -m maichart.composer.diagnose --checkpoint checkpoints/Prism_composer_v5/composer_last.pt --manifest E:\全量官谱\training_manifest.json --limit-charts 2 --samples 8 --device cuda --output .scratch/composer_interventions.json
```

The new architecture is intended to reduce history shortcuts; preventing all
malicious repetitions is NOT established by contract tests. Remaining research
includes full rolling free-generation recovery, alternative legitimate author
choices, long motifs and unseen atom combinations. No reranker hard veto or
invented anti-note labels were added.
