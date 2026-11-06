# Composer event views and density repair

Date: 2026-10-23. Branch: `codex/composer-music-context-v2-20261023`.

## Authorization and snapshot

User authorized changes to the training contract and requested root-cause
repair of recent extremely dense same-key generation, beyond output filters.
Task `f3e488d5e119` (Prism_composer_v6) was cancelled via WebUI. The retained
last checkpoint was at optimizer update 8500; the last observed live log was
epoch 4 / batch 5375 / update 8762. No checkpoint is deleted or overwritten.

Backup: `backups/composer_density_stack_20261023_142449/source_before.zip`,
with `active_tasks.json` and `working_tree.patch` in the same directory.
The initial patch-backup command failed due to PowerShell argument syntax;
it was immediately rerun with an explicit quoted path before any code edits.

## Contract decisions

- One explicit immutable event-view policy will define in-range valid events,
  the complete causal context (including quality-excluded authored events),
  trusted label events and the candidate-reachable supervised subset.
- Excluded/unknown spans must not become rest or early-ending labels.
- Workload labels use reusable exact triangular-kernel prefix queries;
  symmetric and trailing masks are independent and match their true support.
- Future and EOS labels carry explicit validity masks; original unknown tails
  do not become fabricated EOS positives.
- Parsed head/type semantics are shared by training and new generation;
  too-small action-slot settings must never silently truncate events.
- Fingerprints include actual event/timing/audio identity and label policies;
  old weights remain loadable for generation and explicit finetune, not
  same-contract resume.

## Stage status

1. Investigation: root causes confirmed in candidate recovery, missing per-pad
   temporal input, disconnected event views, sparse interpolated workload
   labels, and native guard bypasses. Native MiaCode is not a human jack-speed
   validator; a separate configurable generation policy is necessary.
2. Implementation: unified event views are connected to the dataset; exact
   workload queries and independent label masks are connected. Per-pad causal
   timing is connected to fresh training, generation and explicit migration.
   Candidate v6 removes recovery-only sub-1/16-beat proposals. Guard transactions
   include first events and pure taps. The main UI safety control and backend
   payload have been integrated and verified on the newly started port 7862.
3. Concentrated validation completed after implementation: Python Composer
   density/event/workload/checkpoint/generation/runtime suite passed 121 tests;
   WebUI JavaScript suite passed 11 tests. `py_compile` also passed. No model
   training or long free-generation run was started.

## Resume work and compatibility

Additional snapshot before continuation:
`backups/composer_event_density_completion_20261023/source_at_resume.zip`.
The source snapshot includes the existing dirty worktree; no earlier agent or
user changes were reverted. No Composer train/generate process was running
when continuation began.

Current training semantics:
`chart_composer_training_v6_unified_events_head_timing`.
New music-context models use `head_timing_dim=99`: seen flag, log seconds age
and log beat age for each of 33 pads. Snapshots are taken strictly before the
target. This input is hidden together with symbolic history under modality
dropout; it does not enter the audio-only auxiliary branch. The added linear
projection has only `100*d_model` parameters and starts at zero for migration.
Old weights still generate; old optimizer states do not resume under the new
contract. Use explicit finetune in a new directory, or fresh for clean quality
comparison. The generation report distinguishes trained and effective candidate
policy when old v5 proposals are upgraded to the v6 safety correction.

The 50ms default same-pad head spacing is an explicit product safety policy,
not a MiaCode rule. CLI `--min-same-head-ms` and the main WebUI control accept
0..200ms; 0 disables only that policy. Native detection stays enabled. No
same-interval rule is used to relabel authored training events. Output reports
record rejection reasons and pre/postprocess density diagnostics.

Diagnostics and automatic musical metrics remain informational; output
filters alone cannot establish that the model has learned musical selection.

## Validation evidence and UI runtime

The latest two archived outputs were measured directly: `e758d8b792f6` had a
26.3 ms minimum event interval and a 39.5 ms minimum same-head interval;
`9e44ce44a6dd` had a 6.6 ms minimum event interval and the same 39.5 ms
same-head minimum. The latter is precisely the type of candidate-recovery
pathology this revision removes for new v6 training and blocks by the default
50 ms generation policy.

The existing WebUI processes on ports 7860 and 7861 were stale and still
reported the previous generation contract. The updated source was started and
verified on port 7862: the new contract is
`composer_generation_controls_v6_head_spacing`, the main UI contains the
`composerMinSameHeadMs` control, and the JavaScript payload forwards it. The
checkpoint inspection endpoint confirmed the retained v6 checkpoint remains
generatable but is not resumable under the new contract:
`chart_composer_training_v5_music_context_shared_slots`, candidate policy
`absolute_recovery_v5`, epoch 4, optimizer update 8500. It must be explicitly
finetuned into a new directory or replaced with a fresh v6 training run.

The command to stop and replace the old WebUI processes was rejected by the
automatic execution policy before running. They were left intact. The new
service is available at `http://127.0.0.1:7862` (PID 39348 at startup); no
training job was restarted. Users must use 7862 for the verified new backend.

Additional boundary fixes from the final integration review:
- Filter negative beats before calling the strict nonnegative TempoClock.
- Mask pre-audio candidates in both training and generation for negative first.
- Mask carry-in sustained-action spans as unknown when their heads occurred
  before audio start, preventing false unoccupied/rest supervision.
- Recompute global conditions when out-of-range source heads are removed.
- Positive quality confidence changes loss weights, never physical workload
  target density. Zero confidence remains unknown.
- Preserve shared action-head weights exactly when migrating an existing
  music-context model to the added timing channel.

Code fixes and focused integration tests establish the contract and safety
behavior. They do not prove musical convergence or that old weights have
learned the new timing input. Full-song A/B listening and a new training run
remain the next quality evaluation, rather than an output acceptance gate.
