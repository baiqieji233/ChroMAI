# Composer Workload Repair - 2026-10-23

## Evidence and scope

Latest inspected output: `webui_workspace/generated/6ae2d034fb21`, v3 epoch 6,
176500 optimizer steps, 793 events. Original playability report had 10 hard
findings; one MultiTouch finding remained in report-only mode. This is not a
stale-checkpoint-only problem. No listening-based quality guarantee is claimed.

The v3 task `a3824e280d30` was stopped through the port-7860 task API before
editing shared training code. Existing v1/v2/v3 checkpoints are preserved.
Source backup is `backups/composer_pre_workload_*.zip`.

## Confirmed defects being repaired

1. Future head-rate labels summed cumulative offsets instead of adjacent gaps,
   underestimating density and marking continuous phrases as long rests.
2. v3 phase filtering rejected ordinary 1/6 and 1/12 phrases; its fallback
   could exceed the configured candidate budget or preserve a locked phase.
3. Distance-to-node attenuation suppressed relative RMS, energy slope and
   chroma change as well as attack, without measuring source audio.
4. Slide waiting intervals were treated as unconditional hand/key ownership;
   double-Hold action failure could end the whole song. The prior weakened
   event-count test concealed this regression and will be restored.
5. Excluded-region labels and nonzero quality weights were ignored by Composer.

## Implementation record

1. `acoustics.py`, `audio_bank.py`, `candidates.py`: source-rate prefix-energy
   measurements at 20ms background centers plus native evidence peaks repair attack/RMS/detail/slope without
   extra FFTs per sample; sparse Mel/Chroma remain interpolated. A versioned
   mapped bank prevents stale native/legacy cache reuse. Context pooling is
   real-time weighted and has .5s/2s/8s/whole-audio windows. No amplitude gate.
2. `candidates.py`: relative offsets remain legal; absolute grid recovery
   anchors add ways out of /48 phase. Candidate selection caches only fractional
   phase instead of whole-song absolute positions; budget remains bounded.
   High-frequency musical-phase features distinguish timing candidates.
3. `workload.py`, `data.py`: five authored workload channels (event/head log
   rate, Hold/Slide/Touch fractions), uniform probes include rests; seconds are
   absolute audio time. Six causal ledger channels cover 1s/4s recent density.
   Curves near excluded regions are masked instead of learning exclusion as rest.
4. `data.py`: future rates now use adjacent gaps; original history is retained
   while excluded targets are removed. Positive chart/region weights reach
   losses. Stable-rhythm exit examples receive a bounded 2x transition weight.
5. `model.py`, `train.py`: audio+condition-only workload predictor is supervised
   on all candidate and uniform probes; its predicted values (never authored
   values) feed nonlinear candidate fusion. Authored slope changes are targets,
   not unconditional smoothness penalties. Slot decoding conditions on previous
   actions, with teacher labels at train time and actual committed slots at
   generation time. History timing dropout preserves two recent anchors.
6. `generate.py`, `guard.py`: remove unconditional Slide/waiting-hand bans.
   Restore double-Hold candidate checks; failed transactions retry another
   configuration/time and never end the chart. Source action ends must stay in
   the audio file. New mapped runtime Muri findings reject uncommitted proposals;
   soft references remain reports. Post-repair points no longer silently fall
   back to uncorrected output when empty. Learned curves are exported for review.
7. `checkpoint.py`, training semantics: explicit legacy/new architecture profile
   and feature dimensions; legacy checkpoints remain generation-compatible,
   but cannot resume into the new architecture silently. A new fresh run or a
   future explicit parameter migration is required.
8. `generate.py`: keep search cursor separate from the last committed event in
   delta-time features; bound retry attempts without ending the chart. Native
   head masks are precomputed and vectorized. Training/inference no longer
   perform an unused extra full-vocabulary action projection.
9. Generation report exports predicted and realized workload curves, mean
   head-rate deviation and correlation. These describe adherence to the learned
   plan, not musical quality, and do not reject output. See the separate WebUI
   handoff for curve controls, training metrics and unresolved Muri visibility.

Tests remain deferred until integration and documentation are complete. No
old baseline weights are overwritten, and no fresh training is launched in
this implementation stage.

After the implementation tests passed (`38` Python tests across Composer,
workload, guard and controls; `9` Node UI tests), the new contract is launched
as a fresh `Prism_composer_v4` run. WebUI task `4629a789b2c4` reached epoch 1
batch 1 after preparing 1,787 charts / 949,046 samples. Candidate recall was
`0.996154`, with 1,461 phase-recovery misses recorded. Runtime reported
`chart_composer_training_v4_audio_workload_slots`, native dynamics policy,
141-dim candidate input, workload loss `0.2742`, and slope loss `0.0496` on the
first batch. v3 remains the last legacy-workload checkpoint for fixed-output
comparison and is not used as a parent.

## Quality boundary

Learned workload is supervised from authored charts on the absolute audio
seconds clock, including rests. Waveform loudness is evidence, never a hard
density target. Reference curves are labels only; both train and generate
consume predicted curves. Old checkpoints must retain their own feature policy.

## Numerical stability repair

The first v4 BF16 run used `2e-4` and produced non-finite losses and 184 NaN
weight tensors around optimizer step 2000, while CPU samples and a full
forward/backward remained finite. New workload-slot fresh runs clamp effective
learning rate to `5e-5`, reject non-finite losses/gradients before optimizer
updates, and log skipped updates. The invalid checkpoint is discarded; v4 is
restarted fresh.

## Follow-up repair: tail integration and generation controls (2026-10-23)

The stopped v4 run exposed a concrete data-path defect. `AcousticContext`
constructed its sampling indices with `unique(searchsorted(...))` and clipped
afterwards. When the requested duration extended beyond the final acoustic
node, two different indices collapsed to the final row after clipping. The
resulting duplicate timestamp created a zero-width tail interval and emitted a
NumPy divide warning; this was the direct cause of the observed non-finite
batches. The fix clips before deduplicating and also normalizes/merges any
duplicate timestamps supplied by a legacy bank. No authored audio values are
discarded or amplitude-normalized by this fix.

Training now records song/event indices and output tensor names for non-finite
batches, rejects non-finite parameters immediately after an optimizer step, and
never writes a checkpoint after such a step. The host-cache guard continues to
evict its LRU under memory pressure.

Composer generation compatibility was completed in the WebUI. The main and
standalone Composer controls expose and persist independent candidate-position
and action temperatures plus workload-curve strength. The backend and status
contract is `composer_generation_controls_v5_workload_sampling`; the summary
reports effective values and only calls decoding greedy when both independent
temperatures are zero. Old v4 services are rejected instead of silently
ignoring the new controls.

The invalid `Prism_composer_v4` run was stopped before another checkpoint was
written. A fresh run should be started after focused verification; its first
100 batches should contain no `composer_nonfinite_loss` and no NumPy
invalid-divide warning.

The repaired fresh task `049d03897dde` was then launched with the same v4
configuration and `5e-5` effective learning rate. Through epoch 1 batch 100 it
reported no non-finite loss, gradient, parameter, or NumPy divide warning.
Data throughput improved from roughly 0.83 to 0.97 batch/s over the observed
window, while host free physical memory stabilized near 9.3 GiB. The task is
left running; this is a numerical/runtime smoke check, not evidence of final
musical quality.

## Data throughput repair (2026-10-23)

The first resumed run was slow because the GPU was waiting for Python data
preparation. At batch 8,900, cumulative timing was about 5,349 seconds in
`data_sec` versus 1,303 seconds in model compute. The resident feature LRU was
limited to about 3 GiB while random sampling visited 1,787 charts, so the same
mapped bank files were repeatedly reopened after eviction. CUDA allocation was
only about 1 GiB, confirming this was an input pipeline bottleneck rather than
GPU saturation.

The data path now keeps a bounded secondary LRU of read-only mmap bank views,
uses four parallel sample preparation threads behind the ordered prefetcher,
and dynamically shrinks both caches when Windows physical/commit headroom
falls. This changes no sample contents, labels, candidate policy, or optimizer
contract. The WebUI exposes `样本准备线程` (`sample_workers`, 1--8; default 4)
alongside prefetch depth.

After restarting from optimizer step 9,000, the repaired task reached batch
9,325 at about 1.62 batch/s with roughly 6.1 GiB free physical memory and a
bounded 768 mapped-bank entry cache. This is materially faster than the prior
1.32 batch/s run and avoids the earlier collapse to near-zero cache budget.
The same task later reached batch 14,025 at about 1.87 batch/s, with roughly
6.9 GiB free physical memory and no non-finite or CUDA failure log. The first
few hundred resumed batches are slower while the mmap view cache warms; the
steady-state rate is the relevant comparison.

## Fresh v4 restart and context-cache repair (2026-10-23)

At the user's request, `Prism_composer_v4` was cleared and restarted fresh as
task `94b5cc7f0454` with the same model/training contract. A further source of
CPU wait was found: only 64 `AcousticContext` objects were retained for 1,787
charts, so random sampling repeatedly rebuilt the workload integration prefix.
The cache is now memory-aware and retains up to 256 contexts at normal
headroom, shrinking under pressure. The fresh task reports
`context_entries=256`, bounded mmap entries, no non-finite updates, and about
1.56 batch/s by batch 375 while warming. Batch 32 still leaves GPU utilization
moderate because the model's per-batch compute is small; increasing batch size
at a safe checkpoint boundary is the next available GPU-utilization knob.
