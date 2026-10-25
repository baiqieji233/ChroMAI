"""Causal sustained state evaluated at candidate times from one parsed index."""
from __future__ import annotations

import torch

from maichart.generation.action_state_spec import STATE_DIM

PREFIX_STATE_PROFILE = "candidate_prefix_v2"


def prefix_action_states(index, query_beat, *, visible_before_beat):
    """Return untruncated action rows. Only committed source beats are visible.

    The exclusive visibility cutoff is independent of the evaluation clock.
    Tempo changes are known song timing; future chart heads are never consulted.
    Old cached indexes without source beats must be rebuilt for this API.
    """
    return index.at(query_beat, include_current=True,
                    visible_before_beat=visible_before_beat,
                    visibility_version="causal_v2", max_states=None)


def prefix_state_summary(index, query_beat, *, visible_before_beat):
    """Mean of every active row, preserving the complete 50-feature layout.

    Empty prefixes yield all zeros. Count is available in the raw API metadata.
    """
    states, mask, _ = prefix_action_states(
        index, query_beat, visible_before_beat=visible_before_beat)
    active = states[~mask]
    result = torch.zeros(STATE_DIM, dtype=torch.float32)
    if len(active):
        result = active.mean(0)
    return result


def candidate_prefix_states(index, candidate_beats, *, visible_before_beat):
    """Return [C,50] states sharing one immutable committed prefix."""
    values = [prefix_state_summary(index, beat, visible_before_beat=visible_before_beat)
              for beat in candidate_beats]
    return torch.stack(values) if values else torch.zeros((0, STATE_DIM))


def committed_prefix_states(index, candidate_beats, *, committed_events):
    """Evaluate candidates after committing an exact count of source events.

    The index points must use chronological event order, as Composer does.
    This works with both full-chart training indexes and incrementally rebuilt
    generation indexes, including empty prefixes and the final committed event.
    """
    points = index.points
    if isinstance(committed_events, bool) or not isinstance(committed_events, int):
        raise TypeError("committed_events must be an integer")
    if not 0 <= committed_events <= len(points):
        raise ValueError("Invalid committed event count")
    if committed_events < len(points):
        cutoff = points[committed_events].beat
    else:
        cutoff = points[-1].beat + 1 if points else 0
    return candidate_prefix_states(index, candidate_beats, visible_before_beat=cutoff)
