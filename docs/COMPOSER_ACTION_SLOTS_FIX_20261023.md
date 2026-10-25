# Composer action slot capacity repair

## Cause

The dataset contains an event with nine slash-separated atoms. EventViews correctly
rejected the default eight-slot capacity instead of silently discarding supervision.
This is a representation capacity error, not a CUDA or convergence error. Atom
count is not the number of hands required: Touch and sustained actions also count.

## Changes

- Dataset catalog scan now determines required capacity, retaining the configured
  value as a minimum. Effective capacity is reported and used before EventViews,
  target tensors, model construction, and checkpoint serialization.
- Capacity is bounded to 1..64 to expose malformed inputs without unbounded growth.
- Existing fingerprint already includes effective action_slots. Requested minimum
  is deliberately excluded: identical effective datasets keep identical identity.
- Fresh training supports expansion. Explicit finetune preserves old slot weights
  and initializes additional slot embeddings from the old mean. Legacy per-slot
  classifier migration reshapes using the source capacity before averaging.
- Resume never changes capacity: mismatch gives an explicit finetune instruction.
  Diagnostics also refuse checkpoint/data capacity mismatches.

## Limits

This repair preserves all source atoms; it does not certify their playability or
modify event filtering, timing, audio features, or label weighting. Extra slots add
training compute and require learning; an old checkpoint cannot acquire them by
generation alone. No background training is automatically launched.

## Validation

Focused tests cover nine-atom dataset targets, all supported migration sources,
old-weight preservation, and shrink rejection, plus existing Composer training,
event-view, recovery and grounded-control regression tests. Results are reported
after the single final validation pass.

Validation outcome: 71 tests passed in the focused suite; the remaining resume
test initially failed because its TinyDataset stub lacked the new dataset
capacity attributes. After updating the stub, that test passed separately
(72 total passing cases). Modified Python modules compile and git diff --check
passed. No full-dataset training or GPU performance benchmark was run.
