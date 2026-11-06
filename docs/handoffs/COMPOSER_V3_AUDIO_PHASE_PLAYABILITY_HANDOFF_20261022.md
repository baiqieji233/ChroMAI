# Composer v3 Audio, Phase, and Playability Handoff

Date: 2026-10-22

## Scope

`chart_composer_v1` keeps the same model tensor shape and action catalog, but
v3 is a fresh training directory because the candidate/audio input behavior
changed. `Prism_composer_v2` remains the previous baseline and is not mutated.

## Changes

- Candidate generation now uses one shared phase transition rule in training
  and inference. Fine `{48}`/`{64}` relative offsets remain available, while a
  mixed binary/ternary phase is prevented from persisting into the `/48`
  denominator lock that compiled as `{192}`.
- Candidate-time interpolation has an explicit `attack_confidence_v2` policy.
  Mel, RMS, and chroma remain interpolated; narrow attack-derived columns are
  attenuated when a query is distant from sparse evidence nodes. The feature
  width remains 120, but old checkpoints must not be resumed as if their input
  distribution were unchanged.
- Generation checks active Hold/Touch Hold and in-flight Slide/Wifi state at
  each candidate. It limits available hands and blocks known occupied heads;
  the full MiaCode Python audit runs once after generation and conservatively
  deletes only mapped hard conflicts when a repair reduces the hard count.
  Soft or unmapped findings remain in the `.composer.json` report.
- Training runtime reports `phase_filter_misses` and the audio feature policy
  in its startup record and dataset fingerprint.

## Training contract

The v3 run is fresh with the same operational parameters as the stopped v2
run: 384 width, 6 layers, FFN 1536, 8 heads, batch 32, BF16, learning rate
`2e-4`, candidate count 96, history 128, action slots 8, prefetch 2, four
preparation workers, fused/auto optimizer, and 80 epochs. It writes only to
`checkpoints/Prism_composer_v3`.

The existing v2 checkpoint is retained for A/B comparison. Do not resume it
into v3; the feature-policy and phase-filter changes require a fresh dataset
fingerprint. Musical quality still requires fixed-audio generation, conflict
reports, density/blank-span inspection, and human playability review.

Fresh launch record: WebUI task `a3824e280d30`, output directory
`checkpoints/Prism_composer_v3`, started on 2026-10-22 with the parameters
listed above. The task reached epoch 1 data batches and has no parent
checkpoint.

## Verification boundary

Before launching v3, run CPU/static checks for phase transitions, attack
confidence attenuation, candidate tensor shape, post-process import, and
checkpoint/model construction. Do not treat these checks as proof of musical
quality.
