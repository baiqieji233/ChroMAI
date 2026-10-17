"""A continuation's clock is determined only by its committed prefix."""
from fractions import Fraction

import torch
from maichart.generation.action_state_spec import STATE_DIM


def prefix_action_state(index, points, committed_events):
    if not 0 <= committed_events <= len(points):
        raise ValueError('Invalid committed event count')
    if not committed_events:
        return torch.zeros(1, STATE_DIM), torch.ones(1, dtype=torch.bool), {'state_count': 0}
    return index.at(points[committed_events-1].beat, include_current=True)
