# Continuous stream follow-up audit — 2026-10-18

Status: fixes implemented and focused CPU/browser verification passed; production fresh launch pending below.

## Scope and evidence

Starting commit: `958ad10`, branch `codex/chart-stream-v1`.
Preserve old checkpoint generation and finetuning, exact rational positions, and authored rests.
Do not introduce loudness thresholds or force-fill gaps. User subsequently authorized restarting
fresh training through port 7860 after fixes and verification, retaining previous settings in a new directory.

Confirmed by code tracing and targeted CPU regressions:

1. History summary pooling depends on padded batch width; padding-only summaries are visible.
2. The shared preparer removes excluded-region notes, while stream supervision ignores exclusions
   and all chart/regional weights. Removed notes become apparent authored silence.
3. Planner labels use substring tests, missing Slide shapes. Chord labels now use parsed note counts,
   avoiding string slash assumptions; no claim that a duration-slash example occurred in production.
4. Stream retains an unbounded per-chart query dictionary, duplicating songs across difficulties.
5. Checkpoints do not check codec ordering or effective dataset identity on resume; writers have
   no output lock and fresh/finetune may overwrite existing output. FP16 scaler state is omitted.
6. Context rotation inserts synthetic BOS tokens into later history, unlike training histories.
7. Curve preview falls back to 120 BPM while generation estimates missing BPM from the audio.
8. Candidate rejection attributes every retained diagnostic to a candidate if any new diagnostic exists.
9. The reference guard accepts an isolated multi-Tap cell through its old fast path without running
   runtime checks. Regression: `1/3/6` is rejected as MultiTouch now.
10. WebUI never parses stream progress: numbers exist in logs but not the metric panel/progress state.
11. The stream API ignores selected multi-dataset composition and the quality-enabled checkbox.
12. Dropout and seed are trainer options but are not transported from the WebUI.

## Completed changes and reasoning

- Batch-independent masked summaries with unchanged parameter shapes.
- Preserve excluded events as causal context, suppress their supervision and crossing transitions,
  mask affected curve interpolation support, apply independent quality-weighted objective reductions.
- Version the changed training semantics; allow old generation/finetune, reject unsafe strict resume.
- Parse planner head types; bound/reuse query cache and avoid unused descriptor computation.
- Validate inputs before preparation, lock outputs, verify data/codec, preserve scaler state,
  report effective resume optimizer settings, and reduce metric host synchronization.
- Match history and BPM contracts, correct diagnostic attribution.
- Do not bypass reference checks on isolated chords. Correct per-diagnostic counterfactual attribution.
- Honor multiple selected manifests and the quality checkbox; expose effective manifest in task results.
- Add actual Dropout/seed controls, training metrics, cache counters, throughput, target-build progress,
  checkpoint save notices, and readable stream task names. Correct memory-slot help text.

Training semantics are now `chart_stream_training_v4_quality_memory`. Parameter shapes and the
generation codec remain unchanged. Old checkpoints still load for generation and finetuning; strict
resume requires the current supervision contract, optimizer state and dataset fingerprint.

Quality policy: original excluded-region events remain teacher-forced causal context, not positive
targets. An event or EOS crossing an excluded interval is masked. Curve interpolation is masked if
either contributing source bin intersects an exclusion; plan labels using that density are also
masked. Positive weights multiply gradients, including batch size one. Each objective has its own
nonzero supervised-position denominator. This avoids both fake rests and a zero-denominator blow-up.

Cache policy: 75% of the configured audio budget belongs to prepared audio, 25% to a bounded whole-song
query LRU. Difficulties with identical source content and clocks share queries. There is also an existing
process-wide 64 MiB scratch query cache, plus worker/preparation overhead; this is not a total RAM cap.
Unannotated preparation caches remain reusable. Only preparations that must retain previously deleted
excluded events gain a new cache identity. No blanket cache invalidation was introduced.

Efficiency: remove unused form descriptor construction; cache parsed configuration type labels;
accumulate detached metrics on the training device and transfer them together at log/save intervals
instead of synchronizing CUDA for every metric on every batch. CUDA batches use pinned host memory.
These eliminate identified overhead; no full-run speedup percentage is claimed.

Checkpoint/output safety: unique atomic temporary names, OS output-directory lock, existing-weight
protection for fresh/finetune, codec-order and fixed input-dimension checks, FP16 scaler persistence,
effective optimizer lr/weight-decay reporting after resume. Partial-epoch resume still explicitly
replays the unfinished epoch; it is not bit-for-bit sampler/RNG continuation.

## Verification and handoff

Final main Python batch: **118 passed** (stream models, masks, loss gradients, boundaries, generation
loop, checkpoint contracts, HTTP API, shared preparation cache and curve training). A separate curve/API
batch passed **82** tests, overlapping the main batch; do not add these as unique coverage. Final JS
batch: **16 passed**. App JavaScript syntax check passed. One initial fixture used an invalid one-second
preparation region; it was corrected to two seconds and the trainer now rejects this invalid range early.

CPU fresh → resume: optimizer updates 2 → 4; warm preparation accepted. Old fixture checkpoint →
finetune: one complete epoch. Changed chart bytes with unchanged manifest: strict resume rejected.
Three-event bounded CPU generation: exact parse roundtrip, zero runtime hard conflicts; reported
`incomplete` because the test intentionally set an event budget. This was a plumbing test, not music QA.

Actual Edge on **7860**: clicked the training button, completed an isolated CPU fixture, saved a step
checkpoint, verified visible Dropout/seed controls and rendered loss/accuracy panel; no page errors.
Task `bec80d6666d4`. Artifacts: `.scratch/stream_contract_20261018/browser_result.json`,
`training_controls.png`, `training_metrics.png`, `cpu_result.json`, and fixture `train.log`.
Both idle WebUI processes were reloaded with the fixes. Existing checkpoints remain preserved.

## Remaining architecture limits (not silently presented as repaired)

- The current summary is derived from recent history, not trainable retrieval over the entire chart.
  Generation retains a causal archive for diagnostics, but distant motif retrieval is not fed as trained
  model memory. Help text now states this accurately. Adding it requires a trained new memory contract.
- The predicted curve is an auxiliary output/preview; editing the shared curve editor is not yet a
  stream generation constraint. Planner intent and Radar/Style are soft conditioning, not quality proofs.
- The two chord-intent dimensions remain redundant for weight compatibility. This batch corrects their
  labels but does not silently repurpose a trained output dimension.
- No new event/audio alignment head, arbitrary incorrect-prefix supervision, silence-loudness gate,
  fixed rhythm grid, or forced rest filling was introduced. Musical quality and exposure bias still
  require listening/free-generation comparisons as training progresses.
- Region exclusions are conditional defects: annotation-free data do not trigger them. This audit does
  not establish that every prior checkpoint was degraded by every listed bug.

## Authorized production restart

After verification, launch fresh via 7860, inheriting latest effective settings: width 384, 6 layers,
8 heads, FFN 768, context 1024, history 256, memory 32, batch 4, accumulation 8, 80 epochs, lr 2e-4,
dropout .1, weight decay .01, plan .15, AT/EOS auxiliary .25, audio budget 8 GiB, preparation workers 4,
save every 500 optimizer updates. Use a new `Prism_stream_v3_fresh_20261018` output directory.
Actual task ID and startup evidence will be appended after launch.

Musical quality remains an empirical question: corrected masks/losses and valid output do not prove
good rhythm selection, configuration expression, or convergence.
