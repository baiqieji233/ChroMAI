# song_form_v2 token metrics

`layout_diagnostics()` reports two different kinds of evidence:

- `*_position_header_accuracy` is the legacy token accuracy over `<at>`, the
  `P:` digits, and `<note>`. It remains for continuity with earlier logs, but
  a high value can hide one wrong Fraction digit among many correct header
  tokens.
- `*_position_fraction_exact_accuracy` is an event-level exact match for the
  complete target Fraction digit string. It is measured under teacher forcing:
  the target prefix is fed to the decoder and each event's target `P:` token
  sequence is compared with the argmax prediction. It is not an accuracy of
  musical beat placement during free generation.

Event-boundary decisions are reported separately. `*_event_continue_*` covers
`<at>` after BOS or `<end>`, `*_event_terminate_*` covers `<eos>` in the same
boundary positions, and `*_event_decision_*` combines them. Each has an
accuracy and token cross-entropy loss. EOS in a note body is not treated as a
valid termination decision.

The metrics are diagnostics, not hard decoding guarantees. In particular,
song_form_v2 free generation jointly chooses positions and note text, so it
does not receive a pre-supplied onset skeleton. Compare these values with
free-generation artifacts and playability checks before making a musical
quality claim.
