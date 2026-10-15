# song_form_v2 decode audit reproductions (historical none profile)

2026-10-15 closure: the reproductions below intentionally describe the
pre-audit `rhythm_geometry_profile=none` contract. The new
`exact_fraction_v1` contract consumes curve, history, audio and decoder beat
geometry. `tests/test_form_audit_integration.py` verifies that these paths
receive gradients and time changes affect logits. Legacy checkpoints retain
their historical behavior; adding code does not teach old weights this new
input. A fresh run or explicit geometry finetune is required.

This note records two confirmed input-contract omissions found during the
read-only audit of checkpoint `Prism_Re_v2/form_last.pt` and artifact
`webui_workspace/generated/554166c317aa`. It does not change the decoder and
does not impose a musical grid or denominator whitelist.

## Curve timestamps are metadata only

`input_example()` emits `load_curve_times_sec`, but `FormTransformer.encode()`
only validates and projects `load_curve`. The timestamp tensor is not consumed
by the encoder, positional encoding, or decoder. Consequently, the same curve
values at `[0, 1]` seconds and `[17.25, 91.5]` seconds produce bit-identical
logits. The regression is in
`tests/full_audit_decode_repros.py::test_load_curve_timestamps_are_not_consumed_by_form_transformer`.

This matters when a local curve is selected from a nonzero span: the values
retain sequence order, but the model cannot distinguish their actual clock
placement. A future fix should either encode the relative timestamps or make a
versioned contract explicit that the curve is always uniformly sampled. The
current report's `load_curve_times_sec` field must not be interpreted as model
evidence.

## Relative history tokens omit distance to the boundary

`input_example()` encodes history relative to its first retained event. With
ordinary tap events and no active sustain state, histories at beats `[0, 1]`
and `[5, 6]` become identical token and action-state tensors for the same
target span. Their metadata reports different real-time coverage, but that
distance is not passed to `FormTransformer`; logits are therefore identical.
The regression is in
`tests/full_audit_decode_repros.py::test_history_distance_is_absent_when_relative_tokens_and_action_state_match`.

This is a context-conditioning omission, not evidence that either timing is
wrong. A versioned input contract can add a scalar or bounded boundary-distance
feature while preserving old checkpoints on their existing path. Do not repair
it by snapping events to common divisions.

## Related artifact evidence

The inspected artifact reports 30 spans, 18 `empty_eos_spans`, and 58 events.
The empty spans all close with `tokens: 2` and positive
`initial_eos_minus_at` margins; this points to early EOS confidence rather than
position-search exhaustion. The chart has 29/94 off-grid note heads (30.85%)
and 14 events above denominator 32. Those values are diagnostic observations,
not grounds for imposing a hard denominator cap.
