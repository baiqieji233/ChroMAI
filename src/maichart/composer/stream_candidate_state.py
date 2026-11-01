"""Selected-onset state from the committed prefix, using native slide lifetimes.

No target head is visible. Dataset preparation shares its parsed index; inference
uses the incremental index and queries only the selected onset, not the grid.
"""
from fractions import Fraction
import torch
from maichart.generation.action_state_spec import STATE_DIM

CANDIDATE_STATE_POLICIES = ('selected_native_v1', 'cursor_v1')


def candidate_state_at(index, stream, absolute):
    step = stream[absolute]
    if index is None or absolute == 0 or step.start < 0:
        return torch.zeros(1, STATE_DIM), torch.ones(1, dtype=torch.bool)
    previous = stream[absolute-1]
    cutoff = (min(step.start+Fraction(1, 10**12), step.end)
              if previous.kind == 2 else step.start)
    states, mask, _ = index.at(step.end, include_current=True,
        visible_before_beat=cutoff, visibility_version='causal_v2', include_metadata=False)
    return states, mask


def summarize_candidate_state(states, mask):
    """Retain mean activity, union/peak geometry, and cardinality (101 values)."""
    valid = ~mask.bool()
    values = states.float().masked_fill(~valid[..., None], 0.)
    count = valid.sum(-1)
    mean = values.sum(-2)/count.clamp_min(1)[..., None]
    peak = values.amax(-2)
    return torch.cat((mean, peak, count.float().log1p()[..., None]), -1)
