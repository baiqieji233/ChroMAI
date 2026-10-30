"""Recorded training sample order; symbolic inputs remain per-event causal prefixes."""
from __future__ import annotations

SAMPLE_SEQUENCE_POLICIES = {'independent_events_v1', 'phrase_groups_v1'}
PHRASE_GROUP_MAX_BEATS = 16
PHRASE_GROUP_MAX_TARGETS = 8


def resolve_sample_sequence_policy(requested='auto', parent_args=None, *, resume=False,
                                   configuration_quality_policy='off'):
    if requested not in {'auto', *SAMPLE_SEQUENCE_POLICIES}:
        raise ValueError('Unknown sample_sequence_policy')
    inherited = (parent_args or {}).get('sample_sequence_policy', 'independent_events_v1')
    if requested == 'auto':
        resolved = (inherited if parent_args is not None or resume else
                    'phrase_groups_v1' if configuration_quality_policy == 'song_paired_v3' else
                    'independent_events_v1')
    else:
        resolved = requested
    if resolved not in SAMPLE_SEQUENCE_POLICIES:
        raise ValueError('Unknown saved sample_sequence_policy')
    if resume and resolved != inherited:
        raise ValueError('Resume inherits sample_sequence_policy; use finetune in a new directory')
    return resolved


def phrase_sample_groups(samples, indices):
    """Partition a song in musical beats, independently of its BPM map.

    These are sampling groups only. Every target retains its original history
    prefix and loss; no bidirectional history state is shared across targets.
    """
    ordered = sorted(indices, key=lambda i: (samples[i].target, samples[i].event_index, i))
    groups, current = [], []
    for index in ordered:
        if current and (len(current) >= PHRASE_GROUP_MAX_TARGETS or
                        samples[index].target - samples[current[0]].target > PHRASE_GROUP_MAX_BEATS):
            groups.append(current)
            current = []
        current.append(index)
    if current:
        groups.append(current)
    return groups
