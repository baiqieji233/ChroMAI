"""Versioned causal context shared by training, recovery and generation."""
from fractions import Fraction
import numpy as np
import torch

from .prefix_state import committed_prefix_states


def composer_action_state(index, committed_count, candidate_beats, policy):
    previous = index.points[committed_count-1].beat if committed_count else Fraction(0)
    if policy == 'committed_legacy_v1':
        rows, mask, _ = index.at(previous, include_current=True)
        active = rows[~mask]
        return active.mean(0) if active.numel() else torch.zeros(50)
    if policy == 'committed_causal_v2':
        return committed_prefix_states(index, [previous], committed_events=committed_count)[0]
    if policy == 'candidate_prefix_v2':
        return committed_prefix_states(index, candidate_beats, committed_events=committed_count)
    raise ValueError('Unknown action_state_policy')


def motif_tensors(index, query_beat, committed_count, history_events, max_events):
    """Retrieve configuration memory after choosing a time; never alter timing."""
    archive = index.retrieve(query_beat, committed_count=committed_count,
        history_start_index=max(0, committed_count-history_events), max_events=max_events)
    # Trim only empty atom columns, retaining all simultaneous Touch heads.
    occupied = np.flatnonzero(np.any(archive.atom_ids != 0, axis=0))
    width = int(occupied[-1])+1 if len(occupied) else 1
    return {'motif_ids': torch.from_numpy(archive.atom_ids[:, :width].copy()),
            'motif_relative_beats': torch.from_numpy(archive.relative_beats),
            'motif_mask': torch.from_numpy(archive.mask)}
