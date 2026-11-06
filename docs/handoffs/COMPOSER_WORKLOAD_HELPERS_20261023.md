# Composer Workload Helpers

Date: 2026-10-23

## Scope

New `composer/workload.py` supplies authored workload labels and committed
prefix statistics. It does not alter model, training, generation or WebUI code.

## Changes

- `event_statistics` parses note syntax with the existing note parser. It
  counts distinct starting positions rather than adding type counters or Slide
  branches. Compact chords and Touch pads remain distinct heads. The three
  flags indicate whether an event contains Hold/Touch Hold, Slide, and
  Touch/Touch Hold respectively. They are overlapping event categories, not
  mutually exclusive probabilities or required player hands. Unknown syntax
  raises an error rather than silently inventing supervision.
- `build_workload_targets` obtains absolute seconds from the exact tempo map
  and `first`. A radius-one-second triangular kernel computes event/head rates
  and type fractions. Its area is clipped to the actual audio interval so the
  head density does not silently decrease in partial intro/outro windows.
  Rates are encoded as `log1p(rate)` without an upper density cap. Type labels
  are weighted authored-event fractions, independent of RMS or peak evidence.
  Count/time prefix integrals make repeated queries logarithmic in event count.
- `causal_ledger` returns event/head rates over the last 1 and 4 seconds, then
  log time since the most recent prefix event and visible prefix span divided
  by four (clipped to one). It includes only supplied committed history with
  time no later than each query. A future candidate does not add itself.
- `tests/test_composer_workload.py` covers shared-head Slide branches, compact
  chords, Touch Hold flags, audio clock offsets, changing BPM, partial-window
  normalization, empty history, and candidate-time causality. Tests were added
  but deliberately not run during this coding stage.

## Integration Contract

`WORKLOAD_DIM=5` has `(event_rate, head_rate, hold_fraction, slide_fraction,
touch_fraction)`; the first two are log rates. `LEDGER_DIM=6` has
`(event_rate_1s, head_rate_1s, event_rate_4s, head_rate_4s,
log_seconds_since_last, history_span_4s)`; its first four rates are linear.
Use the exported channel-name tuples to avoid relying on inferred ordering.

Workload values are labels only. Feeding these original-chart labels into the
decision input would leak the reference chart. A model used in generation must
receive its own audio-conditioned prediction instead. The ledger may be model
input because its contents come exclusively from committed prefix events.

## Evidence And Limits

The existing `form/curve.py` provides versioned authored head-density targets;
`stream/grounded_model.py` provides time-integral phrase queries. This helper
retains their seconds-based approach but adds distinct event/head rates and
separate Hold/Slide/Touch event fractions. It does not equate loudness with
difficulty, enforce a density budget, or establish musical/playable quality.
