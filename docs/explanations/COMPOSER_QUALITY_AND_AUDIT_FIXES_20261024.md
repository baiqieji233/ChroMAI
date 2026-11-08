# Composer Quality & Audit Fixes (2026-10-24)

> Completed-training comparison: epoch 3 finished normally at step 7416. Last/inference weights match and are finite. Six same-code, fixed-audio/condition runs are complete. Touch remains above the requested condition, note count falls, and residual hard-conflict rates move in opposite directions across the two seeds. No overall musical/playability improvement is established. See [completed training and comparison](COMPOSER_COMPLETED_TRAINING_COMPARISON_20261024_ZH.md).

> Latest follow-up: five reproduced generation/export implementation issues are fixed; 230 related Python checks and 16 frontend checks pass. Original checkpoints remain unchanged. Epoch 3 has resumed into `Prism_composer_v8_resume_20261024` from step 5000 (task `ee3b216288c4`, port 7862). See [implementation fixes and resume evidence](COMPOSER_IMPLEMENTATION_FIXES_AND_RESUME_20261024_ZH.md). Strategy changes remain deferred.

> Follow-up: v9 real-data short finetuning and a four-run, fixed-audio/two-seed generation comparison are complete. Full-data training is running in `Prism_composer_v8` (task `7aac408a2f9e`, WebUI port 7862). The v7 checkpoint remains unchanged. Additional verified launch/runtime fixes, measured results, report-only conflicts and outstanding quality checks are documented in the v9 validation follow-up（本地开发资料，不随源码发布）. Full training and musical/playability acceptance are not complete.

## 1. Confirmed Defects & Deep Root Causes Repaired

1. **`slot 0` Action Gradient Starvation & Training Contract Separation (`src/maichart/composer/train.py`, `src/maichart/composer/__init__.py`, `src/maichart/composer/checkpoint.py`)**:
   - **Root Cause**:
     - `Prism_composer_v7/composer_last.pt` (saved at `2026-10-24 08:23:56`, Epoch 9, `22,000` optimizer steps) was trained under the old `_loss` before `08:47:08`, where `action_slots` expanded from `8` to `33` (`max_required_action_slots = 33` due to 33-atom touch fireworks in the official corpus) and `_loss` averaged across all 33 slots with `effective_lr = 5e-5` (despite `--lr 2e-4`).
     - Post-`08:23:56` edits added active-slot loss masking (`active_slots[:, 1:] = action_targets[:, :-1].ne(0)`) and `audio_action_loss`, but left `GROUNDED_TRAINING_SEMANTICS` at `'chart_composer_training_v8_boundary_plans_recovery'`, allowing an incompatible `--resume` and overwriting the saved optimizer LR (`5e-5`) with `2e-4`.
   - **Fix**:
     - `_loss` masks post-`STOP` padding slots (`active_slots`), weights active non-STOP atoms at `1.0` and the single terminating `STOP` decision at `0.35`, and includes `audio_action_loss`.
     - Upgraded `GROUNDED_TRAINING_SEMANTICS` to `'chart_composer_training_v9_active_slots_phrase_tail'`. Historical `v7`/`v8` grounded checkpoints remain valid for generation (`can_generate = True`) and explicit finetune (`can_init = True`, `--init-checkpoint`), while `can_resume = False`.
     - On exact `--resume`, if `--lr` is not explicitly passed on `argv`, `effective_lr` preserves the saved optimizer's `param_groups[0]['lr']`.

2. **Macro Category Temperature Distortion & Broken Quota Formula (`src/maichart/composer/generate.py`)**:
   - **Root Cause**:
     - In the 18,453-atom vocabulary, `Tap` has only `32` atoms (`0.17%`), whereas `Hold` has `1,839` (`9.97%`), `Slide` has `16,096` (`87.23%`), and `Touch` has `486` (`2.63%`).
     - Because `F.cross_entropy` trains logits at $T = 1.0$, individual Slide/Hold atoms have logits 3–6 lower than individual Tap atoms so that their sum over thousands of atoms matches the target category probability (`13.10%` Slide, `8.18%` Hold in the corpus).
     - Earlier flat `_truncate_probabilities` (`top_p=0.90, min_p=0.04, max_k=64`) and a subsequent unit-mismatched `quota_diff` (`workload[hold/slide/touch fraction] - ledger[event/head rate]`) wiped out Hold/Slide/Touch in `deeadc419d82`, `8a164085f073`, `0e8b8dc4b3b9`, and `47a8d3ffa8ff`. Even with `macro_temp = max(temperature, 0.85)`, dividing logits by `0.85` before summing category mass suppressed `Slide` by `2.6x–4.5x` and `Hold` by `2.1x`.
   - **Fix**:
     - Removed `quota_diff` and set `_category_truncate_actions` to compute macro category probabilities (`STOP, Tap, Hold, Slide, Touch`) at the calibrated training temperature $T_{\text{macro}} = 1.0$ (`masked_logits.softmax(-1)`), while applying `action_temperature` and tail truncation (`top_p=0.85, min_p=0.08, max_k=(16, 12, 16, 8)`) strictly *within* each category.

3. **18,454-ID History Token Copy Shortcut / Mode-Locking (`src/maichart/composer/action_factors.py`, `src/maichart/composer/grounded_model.py`)**:
   - **Root Cause**:
     - In `AudioGroundedComposer`, `self.event_embedding` (`18,454` unconstrained rows, mean norm `19.36`) is `9.4x` larger in norm than `self.action_factors.embed` (mean norm `2.07`), and `Prism_composer_v7` was trained with `positions < 2` (`t-1, t-2`) exempted from `history_dropout`.
     - Ablation on `Prism_composer_v7` (`beat=280`) proved that when `t-1, t-2` in `history` were `E4, E5` (`Touch`), `self.event_embedding(history)` drove next-step `Touch` probability from `0.66%` to `98.81%` (causing a 38-line Touch mode-lock once `Touch` was sampled), and `1x, 8x` (`EX`) drove `EX` probability to `84.10%`.
   - **Fix**:
     - Added non-persistent `_lane_anchor_ids` buffer in `CompositionalActions` mapping each atom to its physical button lane (`1..8`).
     - In `AudioGroundedComposer.forward`, history embeddings blend `.75 * self.event_embedding(lane_ids) + .25 * self.event_embedding(history) + self.action_factors.embed(history)` (while keeping same-beat `selected_ids` in `decode_candidate` / `action_slot` exact), and `history_dropout` applies across all positions including `t-1` and `t-2`.

4. **Mixed Binary-Ternary Phase Drift & `{48}/{64}/{96}/{192}` Micro-Subdivisions (`src/maichart/composer/generate.py`, `src/maichart/composer/guard.py`, `src/maichart/composer/events.py`)**:
   - **Root Cause**:
     - In `event_compile.py`, `1 beat = {4}`, so beat denominators `12, 16, 24, 48` compile to Simai `{48}` (`25ms` at 200 BPM), `{64}` (`18.75ms`), `{96}` (`12.5ms`), and `{192}` (`6.25ms`), whereas `{4}, {8}, {12}, {16}, {24}, {32}` (`denominator in {1, 2, 3, 4, 6, 8}`) cover `99.54%` of all official maimai events (`956,579` events across `1,795` charts).
     - `candidates.py` only applied `_phase_transition_allowed` when `policy == 'phase_v3'`, leaving `MUSIC_CANDIDATE_POLICY_V6` (`'absolute_recovery_v6_phase_safe'`) unprotected against transitions between `/8` or `/16` binary phases and `/12` ternary offsets (which produced `{64}/{48}/{96}/{192}` bursts at `Line 69–72` in every recent generation).
   - **Fix**:
     - Added `phase_playable` in `generate.py` for `MUSIC_CANDIDATE_POLICY_V6` requiring `beat.denominator in {1, 2, 3, 4, 6, 8}` and `(previous is None or (beat - previous).denominator in {1, 2, 3, 4, 6, 8})` (with fallback to `base_playable` if all candidates are masked).
     - Added optional `intercept_tap_on_slide` switch (`--intercept-tap-on-slide`, WebUI `#composerInterceptTapOnSlide`, default `False`) and pre-blocked pads within `min_same_head_sec` before `_sample_actions`.

## 2. Controlled A/B Verification on `Prism_composer_v7/composer_last.pt`

- **`47a8d3ffa8ff` (`seed=709835971`, same audio & `condition.json`)**:
  - **Before**: `903` events, `1,338` notes (`tap: 1329 [99.3%], slide: 6, hold: 1, wifi: 2, touch: 0`), `unique_texts: 46`, `{48}/{64}/{96}/{192}` occurrences: `14`, `min_event_interval_ms: 18.75 ms` (`event_intervals_below_25ms: 2`).
  - **After**: `772` events, `972` notes (`tap: 861 [88.6%], hold: 49 [5.0%], slide: 41 [4.2%], touch: 21 [2.2%]`), `unique_texts: 97` (`+111%`), `max_consecutive_same_text: 5`, `{48}/{64}/{96}/{192}` occurrences: `0`, `min_event_interval_ms: 37.5 ms` (`event_intervals_below_25ms: 0`).
- **`0e8b8dc4b3b9` (`seed=1696576229`, same audio & `condition.json`)**:
  - **Before**: `894` events, `1,272` notes (`tap: 1264 [99.4%], hold: 7, slide: 1, touch: 0`), `unique_texts: 55`, `{48}/{64}/{96}/{192}` occurrences: `6`, `min_event_interval_ms: 18.75 ms`.
  - **After**: `792` events, `1,136` notes (`tap: 1043, slide: 64, hold: 27, touch: 2`), `unique_texts: 109` (`+98%`), `max_consecutive_same_text: 5`, `{48}/{64}/{96}/{192}` occurrences: `0`, `min_event_interval_ms: 37.5 ms`.
