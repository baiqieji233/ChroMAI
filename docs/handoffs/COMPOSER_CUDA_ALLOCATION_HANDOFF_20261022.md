# Composer CUDA allocation investigation, 2026-10-22

## Report and evidence

Task `e79e578258df` failed on the first batch in a BF16 cuBLAS GEMM near
`pointer`. Requested configuration: batch 32, width 384, six layers in each
encoder, FFN 1536, history 128, candidates 96, eight action slots, vocabulary
18,454, fused AdamW, two prefetched batches. Model: 85,499,706 parameters.

The original pointer traceback alone does not identify the first failing
asynchronous CUDA operation. A reproduction with `CUDA_LAUNCH_BLOCKING=1`
instead failed earlier in a history FFN dropout allocation: 20 MiB allocation
failed while CUDA reported 9.52 GiB free and only 1.12 GiB allocated by Torch.
This is not evidence that batch 32 exceeds the dedicated VRAM capacity.

Live environment: RTX 5070 Ti Laptop, 12,227 MiB, driver 581.15, Torch
2.7.0+cu128, CUDA build 12.8. Torch's compiled architectures include sm_120.
An isolated same-size BF16 forward/backward succeeded at 1.42 GiB peak.

Real-data reproduction then completed two optimizer updates before any edits.
Host observations on that run: available physical RAM fell from 2.275 to
1.100 GiB; available Windows commit fell from 3.084 to 0.990 GiB. Consequently,
host allocation/commit pressure is a substantial suspect, not a conclusively
isolated cause of the original cuBLAS failure. The failure is intermittent;
do not present a successful short run as proof of a fully repaired CUDA driver.
No evidence currently justifies replacing BF16 GEMMs or reducing model capacity.

## Changes

- Backed up the touched original train/data modules in
  `backups/composer_cuda_before_20261022_153936.zip`.
- Added Windows `GlobalMemoryStatusEx` observations: available physical RAM,
  available commit, total RAM and commit limit. Unsupported platforms retain
  the previous cache policy.
- The Composer feature LRU now checks headroom at most once a second. Its
  effective capacity never exceeds the requested capacity; it shrinks when
  necessary to leave approximately 2 GiB physical and 4 GiB commit headroom.
  These are conservative cache-policy margins, not guarantees or estimates of
  the exact driver allocation requirement. Existing resident cache bytes are
  counted when calculating growth, so capacity can recover as memory recovers.
- Report effective cache budget separately from configured budget. Under severe
  host pressure it can become zero; more disk I/O is the expected tradeoff.
  This policy cannot reclaim other applications' memory, Python labels/history
  tables, or guarantee adequate commit if those alone exhaust the system.
- Low initial host headroom emits `composer_warning`. Runtime/progress contain
  `host_memory`; a CUDA error anywhere in the batch loop emits
  `composer_cuda_failure` with Torch/build, allocation counters, and host state.
  The original traceback is preserved through exception chaining. There is no
  automatic retry on a failed CUDA context or silent precision/batch downgrade.
- Fixed a confirmed bounded-run bug: the DataLoader batch sampler now ends at
  `max_steps`, so background prefetch cannot load later, unused batches after
  that limit. Normal unlimited training keeps its existing sample order.

Parameters, model state keys, checkpoint schema, features, supervision targets,
loss weights and optimizer algorithm are unchanged. Cache retention and loading
time may change; no intended change to what the model learns was introduced.
No user checkpoint was overwritten and no full user training was launched.

## Concentrated verification

After the edits, 19 tests across `test_composer_runtime_memory.py` and
`test_chart_composer_v1.py` passed. Coverage includes simulated host pressure
eviction, capacity limits, bounded iteration, diagnostic error chaining, live
Windows snapshot shape, CPU train/checkpoint/generation, and exact CPU resume.

A real-data CUDA run used the original architecture and training parameters,
normal asynchronous launch (`CUDA_LAUNCH_BLOCKING=0`), original shuffled dataset
of 947,683 samples, and eight actual batch-32 BF16 fused-AdamW updates. It also
saved a complete checkpoint in the ignored scratch directory. No CUDA error;
peak Torch allocation 2.068 GiB, reserved 2.252 GiB, end allocation 0.991 GiB.
Host free commit was about 1.85 GiB and physical free about 1.31 GiB at the end;
the feature LRU correctly stopped retaining arrays. Eight batches took about
9.78 seconds, including 7.76 seconds data waiting and startup; this is neither
a full-epoch throughput measurement nor proof that long training is stable.

Logs: `.scratch_composer_cuda_repro.log` (allocation failure),
`.scratch_composer_probe.log` (pre-edit success and host measurements),
`.scratch_composer_fixed.log` (post-edit eight-batch validation). These logs
and captured real batch are ignored local artifacts, not repository inputs.

## Next operational step

The original model/batch configuration can be retried after freeing host memory.
Keep only necessary applications/services open; do not infer a need for global
FP32 or a smaller model from this traceback. Adequate system-managed pagefile
capacity and physical RAM headroom matter. No pagefile/system settings or
unrelated application processes were changed automatically. If it recurs,
preserve the new `composer_cuda_failure` record and inspect whether free commit
is exhausted; if headroom is healthy, investigate driver/library instability
separately instead of treating this hypothesis as proven.
